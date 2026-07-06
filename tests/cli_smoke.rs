use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;

use tempfile::TempDir;
use webnn_graph::ast::{ConstDecl, ConstInit, DataType, GraphJson};
use webnn_graph::weights::{TensorEntry, WeightsManifest};

fn webnn_graph_bin() -> String {
    std::env::var("CARGO_BIN_EXE_webnn-graph").expect("webnn-graph binary path")
}

fn write_weights(path: &Path, payloads: &[&[u8]]) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"WGWT");
    bytes.extend_from_slice(&1u32.to_le_bytes());
    for payload in payloads {
        bytes.extend_from_slice(payload);
    }
    fs::write(path, bytes).unwrap();
}

fn write_manifest(path: &Path, tensors: BTreeMap<String, TensorEntry>) {
    let manifest = WeightsManifest {
        format: "wg-weights-manifest".to_string(),
        version: 1,
        endianness: "little".to_string(),
        tensors,
    };
    fs::write(path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
}

#[test]
fn extract_weights_cli_reuses_matching_source_tensor() {
    let temp_dir = TempDir::new().unwrap();
    let input_graph_path = temp_dir.path().join("input.json");
    let source_weights_path = temp_dir.path().join("source.weights");
    let source_manifest_path = temp_dir.path().join("source.manifest.json");
    let output_weights_path = temp_dir.path().join("out.weights");
    let output_manifest_path = temp_dir.path().join("out.manifest.json");
    let output_graph_path = temp_dir.path().join("out.json");
    let output_dir = temp_dir.path().join("extracted");

    let reused = vec![1u8, 2, 3, 4];
    let local = vec![5u8, 6, 7, 8];
    write_weights(&source_weights_path, &[&reused]);
    write_manifest(
        &source_manifest_path,
        BTreeMap::from([(
            "reused".to_string(),
            TensorEntry {
                data_type: DataType::Uint8,
                shape: vec![4],
                file: None,
                byte_offset: 8,
                byte_length: reused.len() as u64,
                layout: Some("row-major".to_string()),
            },
        )]),
    );

    let mut graph = webnn_graph::ast::new_graph_json();
    graph.consts.insert(
        "local".to_string(),
        ConstDecl {
            data_type: DataType::Uint8,
            shape: vec![4],
            init: ConstInit::InlineBytes {
                bytes: local.clone(),
            },
        },
    );
    graph.consts.insert(
        "reused".to_string(),
        ConstDecl {
            data_type: DataType::Uint8,
            shape: vec![4],
            init: ConstInit::InlineBytes { bytes: reused },
        },
    );
    fs::write(
        &input_graph_path,
        serde_json::to_vec_pretty(&graph).unwrap(),
    )
    .unwrap();

    let status = Command::new(webnn_graph_bin())
        .args([
            "extract-weights",
            "--input",
            input_graph_path.to_str().unwrap(),
            "--output-dir",
            output_dir.to_str().unwrap(),
            "--weights",
            output_weights_path.to_str().unwrap(),
            "--manifest",
            output_manifest_path.to_str().unwrap(),
            "--output-graph",
            output_graph_path.to_str().unwrap(),
            "--reuse-weights",
            source_weights_path.to_str().unwrap(),
            "--reuse-manifest",
            source_manifest_path.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let manifest: WeightsManifest =
        serde_json::from_slice(&fs::read(output_manifest_path).unwrap()).unwrap();
    assert!(manifest.tensors["reused"].file.is_some());
    assert_eq!(manifest.tensors["reused"].byte_offset, 8);
    assert_eq!(manifest.tensors["local"].file, None);
    assert_eq!(fs::read(output_weights_path).unwrap(), {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"WGWT");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&local);
        bytes
    });

    let output_graph: GraphJson =
        serde_json::from_slice(&fs::read(output_graph_path).unwrap()).unwrap();
    assert!(matches!(
        output_graph.consts["reused"].init,
        ConstInit::Weights { .. }
    ));
}
