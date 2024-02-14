use std::fs;
use std::fs::File;
extern crate protobuf;
use protobuf::{Message};
use crate::onnx_parser::protoc_generated::onnx_ml_proto3::{TensorProto};
use std::io::Write;
use std::path::{PathBuf};
use crate::models::models::Model;
use crate::utils::shared_constants::SUPPORTED_IMAGE_FORMATS;

pub fn serialize_custom_dataset (chosen_model: &Model, folder_name: &String){
    println!("\n🤞 Serializing your custom dataset into .pb files...");
    let custom_dataset_path = PathBuf::from("models").join(chosen_model.as_str()).join(folder_name);
    let custom_dataset_serialized_path = PathBuf::from("models").join(chosen_model.as_str())
        .join(folder_name.clone() + "_serialized");
    if custom_dataset_serialized_path.exists() && custom_dataset_serialized_path.is_dir(){
        fs::remove_dir_all(&custom_dataset_serialized_path).unwrap();
    }
    fs::create_dir(&custom_dataset_serialized_path).unwrap();
    fs::create_dir(&custom_dataset_serialized_path.join("images")).unwrap();
    fs::create_dir(custom_dataset_serialized_path.join("labels")).unwrap();
    for label_dir_res in fs::read_dir(custom_dataset_path).unwrap() {
        let label_dir = label_dir_res.unwrap();
        // Check if the entry name is in numerical format and it's a folder, if not skip it
        if label_dir.file_name().to_str().unwrap().parse::<i32>().is_err() || !label_dir.metadata().unwrap().is_dir(){
            continue;
        }
        let label = label_dir.file_name().into_string().unwrap().parse::<f32>().unwrap();
        let files_dir = fs::read_dir(label_dir.path()).unwrap();
        for file_res in files_dir {
            let file = file_res.unwrap();
            if let Some(ext) = file.path().extension().and_then(|e| e.to_str()) {
                if !SUPPORTED_IMAGE_FORMATS.contains(&ext.to_uppercase().as_str()) {
                    continue;
                }
            }
            let image_path = file.path();
            let filename = image_path.file_stem().and_then(|stem| stem.to_str()).map(|s| s.to_string()).unwrap();
            let extension = image_path.extension().and_then(|ext| ext.to_str()).unwrap();
            let img_name = filename.clone() + "." + extension;
            let image_pb_path = custom_dataset_serialized_path.join("images").join(filename.clone() + ".pb");

            let serialization_function = chosen_model.get_serialization_function();
            serialization_function(&image_path, &image_pb_path);
            let num_classes = chosen_model.get_num_classes();
            let label_pb_path = custom_dataset_serialized_path.join("labels").join(filename + ".pb");
            serialize_label(label, label_pb_path, img_name, num_classes);
        }
    }
    println!("✅ Custom dataset successfully serialized!");
}


fn serialize_label(label: f32, pb_path: PathBuf, img_name: String, num_classes: usize){

    let mut label_bytes: Vec<u8> = Vec::with_capacity(num_classes * 4); // Step 2: Allocate Vec<u8>

    // Initialize a Vec<f32> with a size of num_classes, all elements set to 0.0
    let mut array: Vec<f32> = vec![0.0; num_classes];

    array[label as usize] = 1.0; // Set the value at `label` index

    for &num in &array {
        label_bytes.extend_from_slice(&num.to_le_bytes()); // Convert each f32 to 4 bytes and extend the Vec<u8>
    }


    let label_proto = TensorProto {
        dims: vec![1i64, num_classes as i64],
        data_type: 1,
        segment: Default::default(),
        float_data: vec![],
        int32_data: vec![],
        string_data: vec![],
        int64_data: vec![],
        name: img_name.to_string(),
        raw_data: label_bytes,
        external_data: vec![],
        data_location: Default::default(),
        double_data: vec![],
        uint64_data: vec![],
        special_fields: Default::default(),
        doc_string: "".to_string()
    };

    let mut buf = Vec::new();
    label_proto.write_to_vec(&mut buf).unwrap();

    // Write the byte vector to a .pb file
    let mut file = File::create(pb_path).unwrap();
    file.write_all(&buf).unwrap();
}