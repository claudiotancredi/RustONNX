use std::collections::{HashMap};
use std::fs;
use std::path::PathBuf;
use byteorder::{ByteOrder, LittleEndian};
use ndarray::{Array, ArrayBase, ArrayD, Ix, IxDyn, OwnedRepr, s};
use protobuf::Message;
use crate::models::models::Model;
use crate::onnx_parser::protoc_generated::onnx_ml_proto3::{ModelProto, TensorProto};
use crate::operators::op_operator::Operator;
use crate::operators::operators::create_operator;
use crate::utils::errors::OnnxError;

pub fn load_model(file_path: &PathBuf) -> (HashMap<String,Box<dyn Operator>>, String, String) {
    println!("🤞 Loading the model from .onnx file...");
    let model_bytes = fs::read(file_path).expect("Failed to read .onnx file.");
    let mut model = ModelProto::new();

    model
        .merge_from_bytes(&model_bytes)
        .map_err(|_| OnnxError::ONNXParserError("Failed to create the model from the bytes of the .onnx file.".to_string()))
        .unwrap();

    let input_name = model.graph.input[0].name.clone();
    let final_layer_name = model.graph.output[0].name.clone();
    let mut initializers: HashMap<String, ArrayD<f32>> = read_initializers(&model.graph.initializer).unwrap();
    let model_read = model_proto_to_hashmap(&model, &mut initializers);

    println!("✅ ONNX model successfully loaded!");

    (model_read, input_name, final_layer_name)

}

fn read_initializers(model_initializers: &[TensorProto] ) -> Result<HashMap<String, Array<f32, IxDyn>>, OnnxError> {
    let mut initializer_set: HashMap<String, Array<f32, IxDyn>> = HashMap::new();

    for initializer in model_initializers {
        // Prepare to hold the data
        let data:Vec<f32>;
        // Check the data type and extract the data accordingly
        match initializer.data_type {
            1 => { // Floating-point data
                if initializer.float_data.is_empty() {
                    data = initializer.raw_data.chunks(4)
                        .map(|chunk| {
                            let arr: [u8; 4] = chunk
                                .try_into()
                                .map_err(|_| OnnxError::ONNXParserError("Failed to convert chunk of raw data into a \
                                group of 4 bytes.".to_string())).unwrap();
                            LittleEndian::read_f32(&arr)
                        })
                        .collect();
                } else {
                    data = initializer.float_data.clone();
                }
            },
            7 => { // 64-bit integer data
                data = initializer.int64_data.iter().map(|&val| val as f32).collect();
            },
            _ => return Err(OnnxError::ONNXParserError("Unsupported data type for model initializers.".to_string()))
        }

        let dims = initializer.dims.iter().map(|&d| d as Ix).collect::<Vec<_>>();

        let dynamic_dims = IxDyn(&dims);

        let ndarray_data = Array::from_shape_vec(dynamic_dims, data)
            .map_err(|_| OnnxError::ONNXParserError("Failed to create ndarray from parsed data.".to_string())).unwrap();
        initializer_set.insert(initializer.name.clone(), ndarray_data);
    }

    Ok(initializer_set)
}

fn model_proto_to_hashmap(model: &ModelProto, initializer_set: &mut HashMap<String, ArrayD<f32>>) ->HashMap<String, Box<dyn Operator>>{
    let mut model_hm: HashMap<String, Box<dyn Operator>> = HashMap::new();

    if let Some(graph) = model.graph.as_ref() {

        // Iterate through the nodes in the graph
        for node in &graph.node {
            model_hm.insert(node.name.clone(),  create_operator(node, initializer_set).unwrap());
        }
    }
    model_hm
}

fn load_data(file_path: &PathBuf) -> Result<ArrayD<f32>, OnnxError> {
    // Read the file contents into a buffer
    let buffer = fs::read(file_path).expect("Failed to read a .pb input file for an image.");
    let mut data = TensorProto::new();

    // Decode the buffer using the generated Rust structs
    data
        .merge_from_bytes(&buffer)
        .map_err(|_| OnnxError::ONNXParserError("Failed to load the image data from the bytes of the .pb file.".to_string()))
        .unwrap();

    let dims = data.dims.iter().map(|&d| d as Ix).collect::<Vec<_>>();
    let num_elements = dims.iter().product::<usize>();
    let bytes_per_element = 4; // For f32
    let expected_length = num_elements * bytes_per_element;

    if data.raw_data.len() != expected_length {
        return Err(OnnxError::ONNXParserError("The actual length of the raw data does not match the expected length computed \
                        considering the shape of the data. If working on a custom dataset, this may be due to a \
                        wrong serialization for images.".to_string()));
    }

    raw_data_to_array(&data, dims)
}

fn load_ground_truth(file_path: &PathBuf) -> Result<ArrayD<f32>, OnnxError> {
    // Read the file contents into a buffer
    let buffer = fs::read(file_path).expect("Failed to read a .pb file with the label of a loaded image.");
    let mut data = TensorProto::new();

    // Decode the buffer using the generated Rust structs
    data
        .merge_from_bytes(&buffer)
        .map_err(|_| OnnxError::ONNXParserError("Failed to load the label data from the bytes of the .pb file.".to_string()))
        .unwrap();

    let dims = data.dims.iter().map(|&d| d as usize).collect::<Vec<_>>();
    let num_elements = dims.iter().product::<usize>();
    let bytes_per_element = 4; // For f32
    let expected_length = num_elements * bytes_per_element;

    if data.raw_data.len() != expected_length {
        return Err(OnnxError::ONNXParserError("The actual length of the raw data does not match the expected length computed \
                        considering the shape of the data. If working on a custom dataset, this may be due to a \
                        wrong serialization for labels.".to_string()));
    }

    raw_data_to_array(&data, dims)
}

fn raw_data_to_array(data: &TensorProto, dims: Vec<usize>)->Result<ArrayD<f32>, OnnxError>{

    let data_array = data.raw_data
        .chunks_exact(4)
        .map(|chunk| {
            chunk.try_into()
                .map_err(|_| OnnxError::ONNXParserError("Failed to convert chunk of raw data into a \
                                group of 4 bytes.".to_string()))
                .map(|bytes| f32::from_le_bytes(bytes))
        })
        .collect();

    match data_array {
        Ok(data) => {
            ArrayD::from_shape_vec(dims, data)
                .map_err(|_| OnnxError::ONNXParserError("Failed to create ndarray with given shape.".to_string()))
        },
        Err(_) => Err(OnnxError::ONNXParserError("Failed to manage chunks of data.".to_string())),
    }
}

pub fn load_images_and_labels(chosen_model: &Model, folder_name: &String, test_dataset: &bool) -> (Vec<ArrayBase<OwnedRepr<f32>, IxDyn>>, ArrayD<f32>, Vec<PathBuf>) {
    let label_stack;
    let mut file_paths = vec![];

    let images_vec = match test_dataset{
        true=>{
            println!("\n🤞 Loading the test dataset (images & labels) provided by ONNX...");
            let test_input_path = PathBuf::from("models").join(chosen_model.as_str()).join("test_data_set_0").join("input_0.pb");
            let test_label_path = PathBuf::from("models").join(chosen_model.as_str()).join("test_data_set_0").join("output_0.pb");
            let input_image = load_data(&test_input_path).unwrap();
            file_paths.push(test_input_path);
            label_stack = load_ground_truth(&test_label_path).unwrap();
            let mut images = vec![];
            for i in 0..input_image.shape()[0]{
                images.push(input_image.slice(s![i..i+1, .., .., ..]).to_owned().into_dyn());
            }
            println!("✅  Test dataset successfully loaded!");
            images
        },
        false=>{
            let custom_dataset_serialized_path = PathBuf::from("models").join(chosen_model.as_str())
                .join(folder_name.clone() + "_serialized");

            println!("\n🤞 Loading the custom dataset (images & labels) from the serialized .pb files...");
            let mut arrays = Vec::new();
            let serialized_folder_path = custom_dataset_serialized_path.join("images");
            for file_res in fs::read_dir(serialized_folder_path).unwrap(){
                let file_path = file_res.unwrap().path();
                let filename = &file_path.file_stem().and_then(|stem| stem.to_str())
                    .map(|s| s.to_string()).unwrap();
                let extension = &file_path.extension().and_then(|ext| ext.to_str()).unwrap();
                if !(*extension=="pb"){
                    continue;
                }
                file_paths.push(file_path.clone());
                let label_file_name = filename.clone() + "." + extension;
                let label_path = custom_dataset_serialized_path.join("labels").join(label_file_name);
                let input_image = load_data(&file_path).unwrap();
                let label = load_ground_truth(&label_path).unwrap();
                arrays.push((input_image, label));
            }

            //unzip images and associated labels
            let (images, labels): (Vec<ArrayD<f32>>, Vec<ArrayD<f32>>) = arrays.into_iter().unzip();

            let batch_size = images.len();
            let shape_label = labels[0].shape();
            //build the 2D shape for the label array
            let new_s_label = vec![batch_size, shape_label[1]];

            let flat_labels:Vec<f32> = labels.into_iter()
                .flat_map(|array| array.into_raw_vec())
                .collect();

            label_stack= ArrayD::from_shape_vec(IxDyn(&new_s_label), flat_labels)
                .map_err(|_| OnnxError::ONNXParserError("Failed to create ndarray with given shape.".to_string())).unwrap();

            println!("✅ Custom dataset successfully loaded!");
            images
        }
    };
    (images_vec, label_stack, file_paths)
}