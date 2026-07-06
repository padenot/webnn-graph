use crate::weights::{tensor_byte_length, TensorEntry, WeightsManifest};
use anyhow::{bail, Context, Result};
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"WGWT";
const VERSION: u32 = 1;

/// Existing weights that a new extraction can reference instead of duplicating.
#[derive(Debug, Clone)]
pub struct ReusableWeightsSource {
    pub weights_file: String,
    pub manifest_file: String,
}

struct ReusableWeightsIndex {
    weights_path: PathBuf,
    manifest_path: PathBuf,
    manifest: WeightsManifest,
    file_cache: HashMap<PathBuf, Vec<u8>>,
}

impl ReusableWeightsIndex {
    fn load(source: &ReusableWeightsSource) -> Result<Self> {
        let manifest_text = fs::read_to_string(&source.manifest_file)
            .with_context(|| format!("Failed to read manifest file {}", source.manifest_file))?;
        let manifest: WeightsManifest = serde_json::from_str(&manifest_text)
            .with_context(|| format!("Failed to parse manifest JSON {}", source.manifest_file))?;
        Ok(Self {
            weights_path: PathBuf::from(&source.weights_file),
            manifest_path: PathBuf::from(&source.manifest_file),
            manifest,
            file_cache: HashMap::new(),
        })
    }

    fn resolve_entry_file(&self, entry: &TensorEntry) -> PathBuf {
        match &entry.file {
            Some(path) => resolve_path_relative_to(&self.manifest_path, path),
            None => self.weights_path.clone(),
        }
    }

    fn entry_bytes(&mut self, entry: &TensorEntry) -> Result<&[u8]> {
        let path = self.resolve_entry_file(entry);
        if !self.file_cache.contains_key(&path) {
            let bytes = fs::read(&path).with_context(|| {
                format!("Failed to read reusable weights file {}", path.display())
            })?;
            self.file_cache.insert(path.clone(), bytes);
        }
        let bytes = self.file_cache.get(&path).expect("cached file exists");
        let start = usize::try_from(entry.byte_offset)
            .with_context(|| format!("byteOffset does not fit in usize for {}", path.display()))?;
        let len = usize::try_from(entry.byte_length)
            .with_context(|| format!("byteLength does not fit in usize for {}", path.display()))?;
        let end = start
            .checked_add(len)
            .context("reusable tensor byte range overflow")?;
        if end > bytes.len() {
            bail!(
                "Reusable tensor range [{}, {}) exceeds {} bytes in {}",
                start,
                end,
                bytes.len(),
                path.display()
            );
        }
        Ok(&bytes[start..end])
    }
}

fn resolve_path_relative_to(base_file: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_file
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(path)
    }
}

fn manifest_file_reference(manifest_file: &str, target: &Path) -> String {
    let manifest_dir = Path::new(manifest_file)
        .parent()
        .unwrap_or_else(|| Path::new("."));
    if target.parent().unwrap_or_else(|| Path::new(".")) == manifest_dir {
        return target
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_else(|| target.to_str().unwrap_or_default())
            .to_string();
    }
    target.to_string_lossy().to_string()
}

fn read_weights_blob<'a>(
    path: &Path,
    file_cache: &'a mut HashMap<PathBuf, Vec<u8>>,
) -> Result<&'a [u8]> {
    if !file_cache.contains_key(path) {
        let bytes = fs::read(path)
            .with_context(|| format!("Failed to read weights file {}", path.display()))?;
        if bytes.len() < 8 || &bytes[0..4] != MAGIC {
            bail!("Invalid magic bytes in weights file {}", path.display());
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().expect("slice has length 4"));
        if version != VERSION {
            bail!(
                "Unsupported weights file version {} in {}",
                version,
                path.display()
            );
        }
        file_cache.insert(path.to_path_buf(), bytes);
    }
    Ok(file_cache.get(path).expect("cached file exists"))
}

fn try_reuse_tensor(
    sources: &mut [ReusableWeightsIndex],
    output_manifest_file: &str,
    name: &str,
    data_type: &crate::ast::DataType,
    shape: &[u32],
    bytes: &[u8],
) -> Result<Option<TensorEntry>> {
    for source in sources {
        let Some(entry) = source.manifest.tensors.get(name).cloned() else {
            continue;
        };
        if entry.data_type != *data_type
            || entry.shape != shape
            || entry.byte_length != bytes.len() as u64
        {
            continue;
        }
        if source.entry_bytes(&entry)? != bytes {
            continue;
        }
        let source_file = source.resolve_entry_file(&entry);
        return Ok(Some(TensorEntry {
            data_type: entry.data_type,
            shape: entry.shape,
            file: Some(manifest_file_reference(output_manifest_file, &source_file)),
            byte_offset: entry.byte_offset,
            byte_length: entry.byte_length,
            layout: entry.layout,
        }));
    }
    Ok(None)
}

/// Pack tensor files into a binary .weights file according to a manifest
pub fn pack_weights(manifest_path: &str, input_dir: &str, output_path: &str) -> Result<()> {
    let manifest_text =
        fs::read_to_string(manifest_path).context("Failed to read manifest file")?;
    let manifest: WeightsManifest =
        serde_json::from_str(&manifest_text).context("Failed to parse manifest JSON")?;

    // Validate manifest format
    if manifest.format != "wg-weights-manifest" {
        bail!("Invalid manifest format: {}", manifest.format);
    }
    if manifest.version != 1 {
        bail!("Unsupported manifest version: {}", manifest.version);
    }

    let mut output = File::create(output_path).context("Failed to create output file")?;

    // Write header
    output.write_all(MAGIC)?;
    output.write_all(&VERSION.to_le_bytes())?;

    // Collect tensors sorted by byte offset
    let mut sorted_tensors: Vec<(&String, &TensorEntry)> = manifest.tensors.iter().collect();
    sorted_tensors.sort_by_key(|(_, entry)| entry.byte_offset);

    let mut current_offset = 8u64; // After magic + version

    for (name, entry) in sorted_tensors {
        // Validate byte offset matches expectation
        if entry.byte_offset != current_offset {
            bail!(
                "Tensor '{}' has unexpected byte_offset: expected {}, got {}",
                name,
                current_offset,
                entry.byte_offset
            );
        }

        // Calculate expected byte length
        let expected_len = tensor_byte_length(&entry.data_type, &entry.shape);
        if entry.byte_length != expected_len {
            bail!(
                "Tensor '{}' has incorrect byte_length: expected {}, got {}",
                name,
                expected_len,
                entry.byte_length
            );
        }

        // Read tensor data from input directory
        let tensor_path = Path::new(input_dir).join(format!("{}.bin", name));
        let mut tensor_data = Vec::new();
        File::open(&tensor_path)
            .with_context(|| format!("Failed to open tensor file: {:?}", tensor_path))?
            .read_to_end(&mut tensor_data)?;

        // Validate file size
        if tensor_data.len() as u64 != entry.byte_length {
            bail!(
                "Tensor file '{}' has wrong size: expected {} bytes, got {}",
                name,
                entry.byte_length,
                tensor_data.len()
            );
        }

        // Write tensor data
        output.write_all(&tensor_data)?;
        current_offset += entry.byte_length;
    }

    crate::debug_println!(
        "Packed {} tensors ({} bytes) into {}",
        manifest.tensors.len(),
        current_offset,
        output_path
    );

    Ok(())
}

/// Unpack a binary .weights file into individual tensor files
pub fn unpack_weights(weights_path: &str, manifest_path: &str, output_dir: &str) -> Result<()> {
    let manifest_text =
        fs::read_to_string(manifest_path).context("Failed to read manifest file")?;
    let manifest: WeightsManifest =
        serde_json::from_str(&manifest_text).context("Failed to parse manifest JSON")?;

    let mut weights_file = File::open(weights_path).context("Failed to open weights file")?;

    // Read and validate header
    let mut magic = [0u8; 4];
    weights_file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        bail!("Invalid magic bytes: expected {:?}, got {:?}", MAGIC, magic);
    }

    let mut version_bytes = [0u8; 4];
    weights_file.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version != VERSION {
        bail!("Unsupported weights file version: {}", version);
    }

    // Read entire weights data
    let mut weights_data = Vec::new();
    weights_file.read_to_end(&mut weights_data)?;

    // Create output directory
    fs::create_dir_all(output_dir).context("Failed to create output directory")?;

    // Extract each tensor
    for (name, entry) in &manifest.tensors {
        let start = (entry.byte_offset - 8) as usize; // Subtract header size
        let end = start + entry.byte_length as usize;

        if end > weights_data.len() {
            bail!(
                "Tensor '{}' extends beyond weights file: offset={}, length={}, file_size={}",
                name,
                entry.byte_offset,
                entry.byte_length,
                weights_data.len() + 8
            );
        }

        let tensor_data = &weights_data[start..end];
        let output_path = Path::new(output_dir).join(format!("{}.bin", name));

        fs::write(&output_path, tensor_data)
            .with_context(|| format!("Failed to write tensor file: {:?}", output_path))?;

        crate::debug_println!(
            "Extracted tensor '{}': {} bytes ({:?}, shape={:?})",
            name,
            entry.byte_length,
            entry.data_type,
            entry.shape
        );
    }

    crate::debug_println!(
        "Unpacked {} tensors from {} into {}",
        manifest.tensors.len(),
        weights_path,
        output_dir
    );

    Ok(())
}

/// Create a manifest from a directory of tensor files
pub fn create_manifest(input_dir: &str, output_path: &str, endianness: &str) -> Result<()> {
    let mut tensors = BTreeMap::new();
    let mut current_offset = 8u64; // After magic + version

    // Scan directory for .bin files with .meta.json sidecars
    let entries = fs::read_dir(input_dir).context("Failed to read input directory")?;

    let mut tensor_files: Vec<_> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "bin"))
        .collect();

    tensor_files.sort_by_key(|e| e.file_name());

    for entry in tensor_files {
        let path = entry.path();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .context("Invalid filename")?;

        // Read metadata
        let meta_path = path.with_extension("meta.json");
        let meta_text = fs::read_to_string(&meta_path)
            .with_context(|| format!("Failed to read metadata file: {:?}", meta_path))?;
        let meta: TensorEntry =
            serde_json::from_str(&meta_text).context("Failed to parse metadata JSON")?;

        // Get file size
        let file_size = fs::metadata(&path)?.len();
        let expected_size = tensor_byte_length(&meta.data_type, &meta.shape);

        if file_size != expected_size {
            bail!(
                "Tensor '{}' file size mismatch: expected {}, got {}",
                name,
                expected_size,
                file_size
            );
        }

        // Create tensor entry
        let tensor_entry = TensorEntry {
            data_type: meta.data_type,
            shape: meta.shape,
            file: None,
            byte_offset: current_offset,
            byte_length: file_size,
            layout: meta.layout.or(Some("row-major".to_string())),
        };

        tensors.insert(name.to_string(), tensor_entry);
        current_offset += file_size;
    }

    let manifest = WeightsManifest {
        format: "wg-weights-manifest".to_string(),
        version: 1,
        endianness: endianness.to_string(),
        tensors,
    };

    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    fs::write(output_path, manifest_json).context("Failed to write manifest file")?;

    crate::debug_println!(
        "Created manifest with {} tensors: {}",
        manifest.tensors.len(),
        output_path
    );

    Ok(())
}

/// Extract inline weights from a graph and save them as external files
pub fn extract_weights(
    graph: &crate::ast::GraphJson,
    output_dir: &str,
    weights_file: &str,
    manifest_file: &str,
) -> Result<crate::ast::GraphJson> {
    use crate::ast::{ConstDecl, ConstInit};

    // Create output directory for tensors
    let tensor_dir = Path::new(output_dir).join("tensors");
    fs::create_dir_all(&tensor_dir).context("Failed to create tensor directory")?;

    let mut manifest_tensors = BTreeMap::new();
    let mut new_consts = BTreeMap::new();
    let mut current_offset = 8u64; // After magic + version

    // Process each const
    for (name, const_decl) in &graph.consts {
        match &const_decl.init {
            ConstInit::InlineBytes { bytes } => {
                // Validate byte count matches expected size
                let expected_size = tensor_byte_length(&const_decl.data_type, &const_decl.shape);
                if bytes.len() as u64 != expected_size {
                    bail!(
                        "Const '{}' has wrong byte count: expected {}, got {}",
                        name,
                        expected_size,
                        bytes.len()
                    );
                }

                // Write tensor to file
                let tensor_path = tensor_dir.join(format!("{}.bin", name));
                fs::write(&tensor_path, bytes)
                    .with_context(|| format!("Failed to write tensor file: {:?}", tensor_path))?;

                // Add to manifest
                manifest_tensors.insert(
                    name.clone(),
                    TensorEntry {
                        data_type: const_decl.data_type.clone(),
                        shape: const_decl.shape.clone(),
                        file: None,
                        byte_offset: current_offset,
                        byte_length: bytes.len() as u64,
                        layout: Some("row-major".to_string()),
                    },
                );

                current_offset += bytes.len() as u64;

                // Create new const with weights reference
                new_consts.insert(
                    name.clone(),
                    ConstDecl {
                        data_type: const_decl.data_type.clone(),
                        shape: const_decl.shape.clone(),
                        init: ConstInit::Weights {
                            r#ref: name.clone(),
                        },
                    },
                );

                crate::debug_println!(
                    "Extracted tensor '{}': {} bytes ({:?}, shape={:?})",
                    name,
                    bytes.len(),
                    const_decl.data_type,
                    const_decl.shape
                );
            }
            _ => {
                // Keep non-inline consts as-is
                new_consts.insert(name.clone(), const_decl.clone());
            }
        }
    }

    // Create manifest
    let manifest = WeightsManifest {
        format: "wg-weights-manifest".to_string(),
        version: 1,
        endianness: "little".to_string(),
        tensors: manifest_tensors,
    };

    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    fs::write(manifest_file, manifest_json).context("Failed to write manifest file")?;

    crate::debug_println!(
        "Created manifest with {} tensors: {}",
        manifest.tensors.len(),
        manifest_file
    );

    // Pack weights
    pack_weights(manifest_file, tensor_dir.to_str().unwrap(), weights_file)?;

    // Return new graph with weights references
    let mut new_graph = graph.clone();
    new_graph.consts = new_consts;

    Ok(new_graph)
}

/// Extract inline weights while reusing exact tensor matches from existing weights blobs.
///
/// Reused tensors are represented in the output manifest with a per-entry `file` reference.
/// Non-reused tensors are written to `weights_file`, so each static graph only carries its
/// shape-specific delta when a shared source contains the immutable model weights.
pub fn extract_weights_with_reuse(
    graph: &crate::ast::GraphJson,
    weights_file: &str,
    manifest_file: &str,
    reuse_sources: &[ReusableWeightsSource],
) -> Result<crate::ast::GraphJson> {
    use crate::ast::{ConstDecl, ConstInit};

    let mut reusable_sources = reuse_sources
        .iter()
        .map(ReusableWeightsIndex::load)
        .collect::<Result<Vec<_>>>()?;

    let mut weights = File::create(weights_file).context("Failed to create weights file")?;
    weights.write_all(MAGIC)?;
    weights.write_all(&VERSION.to_le_bytes())?;

    let mut manifest_tensors = BTreeMap::new();
    let mut new_consts = BTreeMap::new();
    let mut current_offset = 8u64; // After magic + version
    let mut reused_count = 0usize;
    let mut reused_bytes = 0u64;
    let mut local_count = 0usize;
    let mut local_bytes = 0u64;

    for (name, const_decl) in &graph.consts {
        match &const_decl.init {
            ConstInit::InlineBytes { bytes } => {
                let expected_size = tensor_byte_length(&const_decl.data_type, &const_decl.shape);
                if bytes.len() as u64 != expected_size {
                    bail!(
                        "Const '{}' has wrong byte count: expected {}, got {}",
                        name,
                        expected_size,
                        bytes.len()
                    );
                }

                let manifest_entry = if let Some(entry) = try_reuse_tensor(
                    &mut reusable_sources,
                    manifest_file,
                    name,
                    &const_decl.data_type,
                    &const_decl.shape,
                    bytes,
                )? {
                    reused_count += 1;
                    reused_bytes += entry.byte_length;
                    entry
                } else {
                    weights.write_all(bytes)?;
                    let entry = TensorEntry {
                        data_type: const_decl.data_type.clone(),
                        shape: const_decl.shape.clone(),
                        file: None,
                        byte_offset: current_offset,
                        byte_length: bytes.len() as u64,
                        layout: Some("row-major".to_string()),
                    };
                    current_offset += bytes.len() as u64;
                    local_count += 1;
                    local_bytes += bytes.len() as u64;
                    entry
                };

                manifest_tensors.insert(name.clone(), manifest_entry);
                new_consts.insert(
                    name.clone(),
                    ConstDecl {
                        data_type: const_decl.data_type.clone(),
                        shape: const_decl.shape.clone(),
                        init: ConstInit::Weights {
                            r#ref: name.clone(),
                        },
                    },
                );
            }
            _ => {
                new_consts.insert(name.clone(), const_decl.clone());
            }
        }
    }

    let manifest = WeightsManifest {
        format: "wg-weights-manifest".to_string(),
        version: 1,
        endianness: "little".to_string(),
        tensors: manifest_tensors,
    };

    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    fs::write(manifest_file, manifest_json).context("Failed to write manifest file")?;

    crate::debug_println!(
        "Extracted weights with reuse: reused {} tensors ({} bytes), wrote {} tensors ({} bytes)",
        reused_count,
        reused_bytes,
        local_count,
        local_bytes
    );

    let mut new_graph = graph.clone();
    new_graph.consts = new_consts;

    Ok(new_graph)
}

/// Inline external weights into a graph
pub fn inline_weights(
    graph: &crate::ast::GraphJson,
    weights_file: &str,
    manifest_file: &str,
) -> Result<crate::ast::GraphJson> {
    use crate::ast::{ConstDecl, ConstInit};

    // Read manifest
    let manifest_text =
        fs::read_to_string(manifest_file).context("Failed to read manifest file")?;
    let manifest: WeightsManifest =
        serde_json::from_str(&manifest_text).context("Failed to parse manifest JSON")?;

    // Process consts
    let mut new_consts = BTreeMap::new();
    let mut file_cache = HashMap::new();
    let default_weights_path = PathBuf::from(weights_file);

    for (name, const_decl) in &graph.consts {
        match &const_decl.init {
            ConstInit::Weights { r#ref } => {
                // Look up in manifest
                let entry = manifest.tensors.get(r#ref).with_context(|| {
                    format!("Weight reference '{}' not found in manifest", r#ref)
                })?;

                let path = entry
                    .file
                    .as_deref()
                    .map(|path| resolve_path_relative_to(Path::new(manifest_file), path))
                    .unwrap_or_else(|| default_weights_path.clone());
                let weights_data = read_weights_blob(&path, &mut file_cache)?;
                let start = usize::try_from(entry.byte_offset).with_context(|| {
                    format!(
                        "byteOffset does not fit in usize for weight reference '{}'",
                        r#ref
                    )
                })?;
                let len = usize::try_from(entry.byte_length).with_context(|| {
                    format!(
                        "byteLength does not fit in usize for weight reference '{}'",
                        r#ref
                    )
                })?;
                let end = start.checked_add(len).with_context(|| {
                    format!("byte range overflow for weight reference '{}'", r#ref)
                })?;

                if end > weights_data.len() {
                    bail!(
                        "Weight reference '{}' extends beyond weights file {}",
                        r#ref,
                        path.display()
                    );
                }

                let bytes = weights_data[start..end].to_vec();

                // Create new const with inline bytes
                new_consts.insert(
                    name.clone(),
                    ConstDecl {
                        data_type: const_decl.data_type.clone(),
                        shape: const_decl.shape.clone(),
                        init: ConstInit::InlineBytes { bytes },
                    },
                );

                crate::debug_println!(
                    "Inlined tensor '{}': {} bytes ({:?}, shape={:?})",
                    name,
                    entry.byte_length,
                    entry.data_type,
                    entry.shape
                );
            }
            _ => {
                // Keep non-weight consts as-is
                new_consts.insert(name.clone(), const_decl.clone());
            }
        }
    }

    // Return new graph with inlined bytes
    let mut new_graph = graph.clone();
    new_graph.consts = new_consts;

    Ok(new_graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{new_graph_json, ConstDecl, ConstInit, DataType};
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_pack_unpack_roundtrip() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let input_dir = temp_dir.path().join("input");
        let output_dir = temp_dir.path().join("output");
        fs::create_dir(&input_dir)?;

        // Create test tensor files
        let tensor_data = [1.0f32, 2.0, 3.0, 4.0];
        let tensor_bytes: Vec<u8> = tensor_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        fs::write(input_dir.join("test.bin"), &tensor_bytes)?;

        // Create manifest
        let manifest = WeightsManifest {
            format: "wg-weights-manifest".to_string(),
            version: 1,
            endianness: "little".to_string(),
            tensors: {
                let mut map = BTreeMap::new();
                map.insert(
                    "test".to_string(),
                    TensorEntry {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        file: None,
                        byte_offset: 8,
                        byte_length: 16,
                        layout: Some("row-major".to_string()),
                    },
                );
                map
            },
        };

        let manifest_path = temp_dir.path().join("manifest.json");
        fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;

        // Pack weights
        let weights_path = temp_dir.path().join("weights.bin");
        pack_weights(
            manifest_path.to_str().unwrap(),
            input_dir.to_str().unwrap(),
            weights_path.to_str().unwrap(),
        )?;

        // Verify packed file has correct header
        let packed_data = fs::read(&weights_path)?;
        assert_eq!(&packed_data[0..4], MAGIC);
        assert_eq!(
            u32::from_le_bytes([
                packed_data[4],
                packed_data[5],
                packed_data[6],
                packed_data[7]
            ]),
            VERSION
        );
        assert_eq!(&packed_data[8..], &tensor_bytes[..]);

        // Unpack weights
        unpack_weights(
            weights_path.to_str().unwrap(),
            manifest_path.to_str().unwrap(),
            output_dir.to_str().unwrap(),
        )?;

        // Verify unpacked data matches original
        let unpacked_data = fs::read(output_dir.join("test.bin"))?;
        assert_eq!(unpacked_data, tensor_bytes);

        Ok(())
    }

    #[test]
    fn test_pack_validates_byte_offsets() {
        let temp_dir = TempDir::new().unwrap();
        let input_dir = temp_dir.path().join("input");
        fs::create_dir(&input_dir).unwrap();

        fs::write(input_dir.join("test.bin"), [0u8; 16]).unwrap();

        let manifest = WeightsManifest {
            format: "wg-weights-manifest".to_string(),
            version: 1,
            endianness: "little".to_string(),
            tensors: {
                let mut map = BTreeMap::new();
                map.insert(
                    "test".to_string(),
                    TensorEntry {
                        data_type: DataType::Float32,
                        shape: vec![4],
                        file: None,
                        byte_offset: 100, // Wrong offset!
                        byte_length: 16,
                        layout: Some("row-major".to_string()),
                    },
                );
                map
            },
        };

        let manifest_path = temp_dir.path().join("manifest.json");
        fs::write(&manifest_path, serde_json::to_string(&manifest).unwrap()).unwrap();

        let weights_path = temp_dir.path().join("weights.bin");
        let result = pack_weights(
            manifest_path.to_str().unwrap(),
            input_dir.to_str().unwrap(),
            weights_path.to_str().unwrap(),
        );

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("unexpected byte_offset"));
    }

    #[test]
    fn test_unpack_validates_magic() {
        let temp_dir = TempDir::new().unwrap();
        let weights_path = temp_dir.path().join("bad.bin");

        // Write invalid magic
        let mut file = File::create(&weights_path).unwrap();
        file.write_all(b"XXXX").unwrap();
        file.write_all(&VERSION.to_le_bytes()).unwrap();

        let manifest = WeightsManifest {
            format: "wg-weights-manifest".to_string(),
            version: 1,
            endianness: "little".to_string(),
            tensors: BTreeMap::new(),
        };

        let manifest_path = temp_dir.path().join("manifest.json");
        fs::write(&manifest_path, serde_json::to_string(&manifest).unwrap()).unwrap();

        let output_dir = temp_dir.path().join("output");
        let result = unpack_weights(
            weights_path.to_str().unwrap(),
            manifest_path.to_str().unwrap(),
            output_dir.to_str().unwrap(),
        );

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid magic bytes"));
    }

    #[test]
    fn extract_weights_with_reuse_reuses_exact_match_and_inlines_mixed_manifest() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let source_weights_path = temp_dir.path().join("source.weights");
        let source_manifest_path = temp_dir.path().join("source.manifest.json");
        let output_weights_path = temp_dir.path().join("out.weights");
        let output_manifest_path = temp_dir.path().join("out.manifest.json");

        let reused_bytes = vec![1u8, 2, 3, 4];
        let local_bytes = vec![5u8, 6];
        let mut source_weights = Vec::new();
        source_weights.extend_from_slice(MAGIC);
        source_weights.extend_from_slice(&VERSION.to_le_bytes());
        source_weights.extend_from_slice(&reused_bytes);
        fs::write(&source_weights_path, source_weights)?;

        let mut source_manifest = WeightsManifest {
            format: "wg-weights-manifest".to_string(),
            version: 1,
            endianness: "little".to_string(),
            tensors: BTreeMap::new(),
        };
        source_manifest.tensors.insert(
            "reused".to_string(),
            TensorEntry {
                data_type: DataType::Uint8,
                shape: vec![4],
                file: None,
                byte_offset: 8,
                byte_length: 4,
                layout: Some("row-major".to_string()),
            },
        );
        fs::write(
            &source_manifest_path,
            serde_json::to_vec_pretty(&source_manifest)?,
        )?;

        let mut graph = new_graph_json();
        graph.consts.insert(
            "reused".to_string(),
            ConstDecl {
                data_type: DataType::Uint8,
                shape: vec![4],
                init: ConstInit::InlineBytes {
                    bytes: reused_bytes.clone(),
                },
            },
        );
        graph.consts.insert(
            "local".to_string(),
            ConstDecl {
                data_type: DataType::Uint8,
                shape: vec![2],
                init: ConstInit::InlineBytes {
                    bytes: local_bytes.clone(),
                },
            },
        );

        let extracted = extract_weights_with_reuse(
            &graph,
            output_weights_path.to_str().unwrap(),
            output_manifest_path.to_str().unwrap(),
            &[ReusableWeightsSource {
                weights_file: source_weights_path.to_string_lossy().to_string(),
                manifest_file: source_manifest_path.to_string_lossy().to_string(),
            }],
        )?;

        let manifest: WeightsManifest = serde_json::from_slice(&fs::read(&output_manifest_path)?)?;
        let reused_entry = &manifest.tensors["reused"];
        assert_eq!(reused_entry.file.as_deref(), Some("source.weights"));
        assert_eq!(reused_entry.byte_offset, 8);
        assert_eq!(reused_entry.byte_length, 4);
        assert_eq!(manifest.tensors["local"].file, None);
        assert_eq!(manifest.tensors["local"].byte_offset, 8);
        assert_eq!(fs::read(&output_weights_path)?, {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(MAGIC);
            bytes.extend_from_slice(&VERSION.to_le_bytes());
            bytes.extend_from_slice(&local_bytes);
            bytes
        });

        let inlined = inline_weights(
            &extracted,
            output_weights_path.to_str().unwrap(),
            output_manifest_path.to_str().unwrap(),
        )?;
        match &inlined.consts["reused"].init {
            ConstInit::InlineBytes { bytes } => assert_eq!(bytes, &reused_bytes),
            other => panic!("expected reused const to be inlined, got {other:?}"),
        }
        match &inlined.consts["local"].init {
            ConstInit::InlineBytes { bytes } => assert_eq!(bytes, &local_bytes),
            other => panic!("expected local const to be inlined, got {other:?}"),
        }

        Ok(())
    }

    #[test]
    fn extract_weights_with_reuse_rejects_mismatched_bytes() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let source_weights_path = temp_dir.path().join("source.weights");
        let source_manifest_path = temp_dir.path().join("source.manifest.json");
        let output_weights_path = temp_dir.path().join("out.weights");
        let output_manifest_path = temp_dir.path().join("out.manifest.json");

        let graph_bytes = vec![1u8, 2, 3, 4];
        let mut source_weights = Vec::new();
        source_weights.extend_from_slice(MAGIC);
        source_weights.extend_from_slice(&VERSION.to_le_bytes());
        source_weights.extend_from_slice(&[9u8, 9, 9, 9]);
        fs::write(&source_weights_path, source_weights)?;

        let mut source_manifest = WeightsManifest {
            format: "wg-weights-manifest".to_string(),
            version: 1,
            endianness: "little".to_string(),
            tensors: BTreeMap::new(),
        };
        source_manifest.tensors.insert(
            "W".to_string(),
            TensorEntry {
                data_type: DataType::Uint8,
                shape: vec![4],
                file: None,
                byte_offset: 8,
                byte_length: 4,
                layout: Some("row-major".to_string()),
            },
        );
        fs::write(
            &source_manifest_path,
            serde_json::to_vec_pretty(&source_manifest)?,
        )?;

        let mut graph = new_graph_json();
        graph.consts.insert(
            "W".to_string(),
            ConstDecl {
                data_type: DataType::Uint8,
                shape: vec![4],
                init: ConstInit::InlineBytes {
                    bytes: graph_bytes.clone(),
                },
            },
        );

        extract_weights_with_reuse(
            &graph,
            output_weights_path.to_str().unwrap(),
            output_manifest_path.to_str().unwrap(),
            &[ReusableWeightsSource {
                weights_file: source_weights_path.to_string_lossy().to_string(),
                manifest_file: source_manifest_path.to_string_lossy().to_string(),
            }],
        )?;

        let manifest: WeightsManifest = serde_json::from_slice(&fs::read(&output_manifest_path)?)?;
        assert_eq!(manifest.tensors["W"].file, None);
        assert_eq!(fs::read(&output_weights_path)?, {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(MAGIC);
            bytes.extend_from_slice(&VERSION.to_le_bytes());
            bytes.extend_from_slice(&graph_bytes);
            bytes
        });

        Ok(())
    }

    #[test]
    fn inline_weights_resolves_per_entry_file_relative_to_manifest() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let weights_dir = temp_dir.path().join("weights");
        fs::create_dir(&weights_dir)?;
        let source_weights_path = weights_dir.join("source.weights");
        let manifest_path = temp_dir.path().join("out.manifest.json");
        let fallback_weights_path = temp_dir.path().join("out.weights");

        let payload = vec![3u8, 1, 4, 1];
        let mut source_weights = Vec::new();
        source_weights.extend_from_slice(MAGIC);
        source_weights.extend_from_slice(&VERSION.to_le_bytes());
        source_weights.extend_from_slice(&payload);
        fs::write(&source_weights_path, source_weights)?;
        fs::write(&fallback_weights_path, [0u8; 8])?;

        let manifest = WeightsManifest {
            format: "wg-weights-manifest".to_string(),
            version: 1,
            endianness: "little".to_string(),
            tensors: BTreeMap::from([(
                "W".to_string(),
                TensorEntry {
                    data_type: DataType::Uint8,
                    shape: vec![4],
                    file: Some("weights/source.weights".to_string()),
                    byte_offset: 8,
                    byte_length: 4,
                    layout: Some("row-major".to_string()),
                },
            )]),
        };
        fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;

        let mut graph = new_graph_json();
        graph.consts.insert(
            "W".to_string(),
            ConstDecl {
                data_type: DataType::Uint8,
                shape: vec![4],
                init: ConstInit::Weights {
                    r#ref: "W".to_string(),
                },
            },
        );

        let inlined = inline_weights(
            &graph,
            fallback_weights_path.to_str().unwrap(),
            manifest_path.to_str().unwrap(),
        )?;
        match &inlined.consts["W"].init {
            ConstInit::InlineBytes { bytes } => assert_eq!(bytes, &payload),
            other => panic!("expected W to be inlined from per-entry file, got {other:?}"),
        }

        Ok(())
    }

    #[test]
    fn inline_weights_reports_missing_per_entry_file() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let manifest_path = temp_dir.path().join("out.manifest.json");
        let fallback_weights_path = temp_dir.path().join("out.weights");
        fs::write(&fallback_weights_path, [0u8; 8])?;

        let manifest = WeightsManifest {
            format: "wg-weights-manifest".to_string(),
            version: 1,
            endianness: "little".to_string(),
            tensors: BTreeMap::from([(
                "W".to_string(),
                TensorEntry {
                    data_type: DataType::Uint8,
                    shape: vec![4],
                    file: Some("missing.weights".to_string()),
                    byte_offset: 8,
                    byte_length: 4,
                    layout: Some("row-major".to_string()),
                },
            )]),
        };
        fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;

        let mut graph = new_graph_json();
        graph.consts.insert(
            "W".to_string(),
            ConstDecl {
                data_type: DataType::Uint8,
                shape: vec![4],
                init: ConstInit::Weights {
                    r#ref: "W".to_string(),
                },
            },
        );

        let err = inline_weights(
            &graph,
            fallback_weights_path.to_str().unwrap(),
            manifest_path.to_str().unwrap(),
        )
        .expect_err("missing per-entry file should be reported");
        assert!(err.to_string().contains("Failed to read weights file"));

        Ok(())
    }
}
