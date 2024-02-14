use std::fs;
use std::path::PathBuf;

fn main() {

    let folder_path = PathBuf::from("src").join("onnx_parser").join("protoc_generated");

    fs::create_dir_all(&folder_path).expect(&format!("Failed to create directory {}", folder_path.to_str().unwrap()));

    println!("Created folder: {}", folder_path.to_str().unwrap());

    let onnx_ml_proto3_path = PathBuf::from("protos").join("onnx-ml.proto3");
    let onnx_data_proto3_path = PathBuf::from("protos").join("onnx-data.proto3");
    let onnx_operators_ml_proto3_path = PathBuf::from("protos").join("onnx-operators-ml.proto3");

    //Create .rs files in folder src/onnx_parser/protoc_generated
    protobuf_codegen::Codegen::new()
        // Use `protoc` onnx_parser
        .protoc()
        // Use `protoc-bin-vendored` bundled protoc command
        .protoc_path(&protoc_bin_vendored::protoc_bin_path().unwrap())
        // All inputs and imports from the inputs must reside in `includes` directories.
        .includes(&["protos"])
        // Inputs must reside in some of include paths.
        .input(onnx_ml_proto3_path)
        .input(onnx_data_proto3_path)
        .input(onnx_operators_ml_proto3_path)
        // Specify output directory relative to Cargo output directory.
        .out_dir(folder_path)
        .run_from_script();
}