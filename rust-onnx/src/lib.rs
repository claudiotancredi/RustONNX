#![allow(dead_code)]

mod datasets;
mod models;
mod onnx_parser;
mod operators;
mod utils;


extern crate protobuf;
use std::collections::HashMap;
use std::path::PathBuf;
use crate::operators::op_operator::Operator;
use ndarray::{Array, ArrayD, IxDyn };
use crate::onnx_parser::parser::{load_images_and_labels, load_model};
use pyo3::prelude::*;
use crate::models::models::Model;
use numpy::{ PyArrayDyn, IntoPyArray};
use crate::utils::auxiliary_functions::print_results;
use crate::utils::run::run;
use crate::utils::serialization_utils::serialize_custom_dataset;

#[pyclass]
pub struct PyModel{
    model: HashMap<String, Box<dyn Operator>>,
}

impl PyModel {
    pub fn clone_model(&self) -> HashMap<String, Box<dyn Operator>> {
        self.model.iter().map(|(name, op)
        | (name.clone(), op.clone_box())).collect()
    }


}

//pub fn load_model(file_path: &PathBuf) -> (HashMap<String,Box<dyn Operator>>, String, String)
#[pyfunction]
pub fn py_load_model(_py: Python, file_path_str: String) -> PyResult<(PyModel, String, String)> {
    let file_path = PathBuf::from(file_path_str);
    let (model, input_name, final_layer_name) = load_model(&file_path);

    let py_model = PyModel{model};

    // Return the tuple
    Ok((py_model, input_name, final_layer_name))
}

//pub fn load_images_and_labels(chosen_model: &Model, folder_name: &String, test_dataset: &bool)
    //-> (Vec<ArrayBase<OwnedRepr<f32>, IxDyn>>, ArrayD<f32>, Vec<PathBuf>)
#[pyfunction]
pub fn py_load_images_and_labels(py: Python, model_id: usize, folder_name: String, test_dataset: bool) ->
PyResult<(PyObject, PyObject, Vec<String>)>
{
    let chosen_model = Model::from_index(model_id).unwrap();


    let (images_vec, label_stack, file_paths) =
        load_images_and_labels(&chosen_model, &folder_name, &test_dataset);

    // Convert Vec<ArrayBase> to Vec<PyObject> where each PyObject is a NumPy array
    let py_vec_arraybase: Vec<PyObject> = images_vec.into_iter()
        .map(|array| array.into_pyarray(py).to_object(py))
        .collect();

    // Convert ArrayD to NumPy array
    let py_arrayd = label_stack.into_pyarray(py).to_object(py);

    // Convert Vec<PathBuf> to Vec<String>
    let py_vec_pathbuf: Vec<String> = file_paths.into_iter()
        .map(|pathbuf| pathbuf.to_string_lossy().into_owned())
        .collect();

    Ok((py_vec_arraybase.into_py(py), py_arrayd, py_vec_pathbuf))
}

/*
pub fn run(images_vec: &Vec<ArrayBase<OwnedRepr<f32>, IxDyn>>,
           model_read: &HashMap<String, Box<dyn Operator>>,
           input_name: &String,
           verbose: &bool,
           final_layer_name: &String,
           chosen_model: &str) -> ArrayBase<OwnedRepr<f32>, IxDyn>
*/

#[pyfunction]
pub fn py_run_model(py:Python, inputs: Vec<PyObject>, model: &PyModel,
                    input_name: String, verbose: bool, final_layer_name: String, id_model: usize)
    -> PyResult<PyObject>{

    let chosen_model = Model::from_index(id_model).unwrap();

    let rust_inputs = inputs.into_iter().map(|py_obj| {
        pyobject_to_arrayd(py, py_obj).unwrap()
    }).collect();


    let rust_model = model.clone_model();


    let output = run(&rust_inputs, &rust_model, &input_name, &verbose,
                     &final_layer_name, chosen_model.as_str());

    // Convert the ArrayD<f32> into a NumPy array
    let numpy_array = output.into_pyarray(py).to_object(py);

    Ok(numpy_array)

}

//pub fn serialize_custom_dataset(chosen_model: &Model, folder_name: &String)
#[pyfunction]
pub fn py_serialize_custom_dataset(_py: Python, model_id: usize,  dir_path: String) -> PyResult<()>{
    let chosen_model = Model::from_index(model_id).unwrap();

    serialize_custom_dataset(&chosen_model, &dir_path);

    Ok(())
}


fn pyobject_to_arrayd(py: Python, obj: PyObject) -> PyResult<ArrayD<f32>> {
    // First, extract the PyAny from PyObject
    let py_any: &PyAny = obj.as_ref(py);

    // Ensure it's an array
    if let Ok(array) = py_any.extract::<&PyArrayDyn<f32>>() {
        // Convert it to ArrayD
        let shape = array.shape().iter().cloned().collect::<Vec<usize>>();
        let arrayd = Array::from_shape_vec(IxDyn(&shape), array.to_vec()?).unwrap();
        Ok(arrayd)
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Expected a numpy array"))
    }
}

//pub fn print_results (model: Model, files: &Vec<PathBuf>, final_output: &ArrayBase<OwnedRepr<f32>, IxDyn>, label_stack: &ArrayD<f32>)
#[pyfunction]
pub fn py_print_results (py: Python, model_id: usize, files: Vec<String>, model_output: PyObject,
labels: PyObject) -> PyResult<()>{

    let rust_outputs = pyobject_to_arrayd(py, model_output).unwrap();
    let rust_labels = pyobject_to_arrayd(py, labels).unwrap();
    let file_paths = files.iter().map(|f| PathBuf::from(f)).collect::<Vec<PathBuf>>();

    let chosen_model = Model::from_index(model_id).unwrap();
    print_results(chosen_model, &file_paths, &rust_outputs, &rust_labels);
    Ok(())
}


#[pymodule]
fn rust_onnx_lib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_load_model, m)?)?;
    m.add_function(wrap_pyfunction!(py_load_images_and_labels, m)?)?;
    m.add_function(wrap_pyfunction!(py_run_model, m)?)?;
    m.add_function(wrap_pyfunction!(py_serialize_custom_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(py_print_results, m)?)?;
    Ok(())
}