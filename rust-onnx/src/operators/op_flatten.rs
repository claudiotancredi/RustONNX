use crate::utils::errors::OnnxError;
use super::op_operator::{Initializer, Operator};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use crate::onnx_parser::protoc_generated::onnx_ml_proto3::NodeProto;

#[derive(Clone)]
pub struct Flatten {
    op_type: String,
    node_name: String,
    input_name: String,
    output_name: String,
    axis: i64,
}

impl Flatten {
    pub fn new(node: &NodeProto, _initializers: &mut HashMap<String, ArrayD<f32>>) -> Self {
        let op_type = node.op_type.to_owned();
        let node_name = node.name.to_owned();
        let input_name = node.input[0].to_owned();
        let output_name= node.output[0].to_owned();

        let opt_axis = node.attribute.iter().find(|attr| attr.name == "axis");
        let axis = match opt_axis{
            Some(x) => x.i,
            None => 1
        };
        Self { op_type, node_name, input_name, output_name, axis }
    }
}

impl Operator for Flatten {
    fn execute(&self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>, OnnxError> {
        let input_tensor = inputs.get(&self.input_name)
            .ok_or_else(||
                OnnxError::TensorNotFound("Input tensor not found.".to_string())).unwrap();

        let rank = input_tensor.ndim();

        // Normalize axis value
        let axis = if self.axis < 0 {
            rank as i64 + self.axis
        } else {
            self.axis
        } as usize;

        if axis > rank {
            return Err(OnnxError::AxisOutOfBounds("Axis is out of bounds for the tensor shape.".to_string()));
        }

        // Calculate the new shape
        let first_dim: usize = if axis == 0 { 1 } else { input_tensor.shape()[..axis].iter().product() };
        let second_dim: usize = input_tensor.shape()[axis..].iter().product();
        let new_shape = IxDyn(&[first_dim, second_dim]);

        // Create the output tensor with the same data but new shape
        let output_tensor = input_tensor.clone().into_shape(new_shape)
            .map_err(|_| OnnxError::ShapeError("Error reshaping tensor.".to_string()))?;

        Ok(vec![output_tensor])
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