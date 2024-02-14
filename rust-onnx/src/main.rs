mod onnx_parser;
mod models;
mod operators;
mod utils;
mod datasets;

extern crate protobuf;
use std::env;
use crate::utils::auxiliary_functions::{print_results};
use crate::utils::run::run;
use crate::onnx_parser::parser::{load_images_and_labels, load_model};
use crate::utils::menu::menu;
use crate::utils::serialization_utils::serialize_custom_dataset;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    let (chosen_model, verbose, test_dataset, folder_name, model_path) = menu();

    let (model_read, input_name, final_layer_name) = load_model(&model_path);

    if !test_dataset{
        serialize_custom_dataset(&chosen_model, &folder_name);
    }

    let (images_vec, label_stack, file_paths) = load_images_and_labels(&chosen_model, &folder_name, &test_dataset);

    let final_output = run(&images_vec, &model_read, &input_name, &verbose, &final_layer_name, chosen_model.as_str());

    print_results(chosen_model, &file_paths, &final_output, &label_stack);

}