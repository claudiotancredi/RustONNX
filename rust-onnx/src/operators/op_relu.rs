use crate::utils::errors::OnnxError;
use super::op_operator::{Initializer, Operator};
use ndarray::ArrayD;
use std::collections::HashMap;
use crate::onnx_parser::protoc_generated::onnx_ml_proto3::NodeProto;

#[derive(Clone)]
pub struct ReLU {
    op_type: String,
    node_name: String,
    input_name: String,
    output_name: String,
}

impl ReLU {
    pub fn new(node: &NodeProto, _initializers: &mut HashMap<String, ArrayD<f32>>) -> Self {
        let op_type = node.op_type.to_owned();
        let node_name = node.name.to_owned();
        let input_name = node.input[0].to_owned();
        let output_name = node.output[0].to_owned();

        Self {
            op_type,
            node_name,
            input_name,
            output_name,
        }
    }
}

impl Operator for ReLU {
    fn execute(&self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>, OnnxError> {
        let input_name = &self.input_name;
        let input = inputs.get(input_name.as_str()).ok_or_else(||
            OnnxError::TensorNotFound("Input tensor not found.".to_string()))?;

        let output_data = input.mapv(|x| if x > 0.0 { x } else { 0.0 });

        Ok(vec![output_data])
    }

    fn get_inputs(&self) -> Vec<String> {
        vec![self.input_name.clone()]
    }

    fn get_output_names(&self) -> Vec<String> {
        vec![self.output_name.clone()]
    }
    fn get_node_name(&self) -> String {
        self.node_name.clone()
    }

    fn get_op_type(&self) -> String {
        self.op_type.clone()
    }

    fn get_initializers_arr(&self) -> Vec<Initializer> {
        vec![]
    }

    fn clone_box(&self) -> Box<dyn Operator> {
        Box::new(self.clone())
    }

}