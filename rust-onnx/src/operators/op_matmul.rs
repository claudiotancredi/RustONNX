use crate::utils::errors::OnnxError;
use super::op_operator::{Initializer, Operator};
use ndarray::{ArrayD};
use std::collections::HashMap;
use crate::onnx_parser::protoc_generated::onnx_ml_proto3::NodeProto;

#[derive(Clone)]
pub struct MatMul {
    op_type: String,
    node_name: String,
    inputs_name: Vec<String>,
    output_name: String,
}

impl MatMul {
    pub fn new(node: &NodeProto, _initializers: &mut HashMap<String, ArrayD<f32>>) -> Self {
        let op_type = node.op_type.to_owned();
        let inputs_name:Vec<String> = vec![node.input[0].to_owned(), node.input[1].to_owned()];
        let output_name = node.output[0].to_owned();
        let node_name = node.name.to_owned();
        Self {
            op_type,
            node_name,
            inputs_name,
            output_name,
        }
    }
}

impl Operator for MatMul {
    fn execute(&self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>, OnnxError> {
        let input_name1 = &self.inputs_name[0];
        let input_name2 = &self.inputs_name[1];

        let input1 = inputs.get(input_name1.as_str())
            .ok_or_else(||
                OnnxError::TensorNotFound("First input tensor not found.".to_string()))?.clone();
        let input2 = inputs.get(input_name2.as_str())
            .ok_or_else(||
                OnnxError::TensorNotFound("Second input tensor not found.".to_string()))?.clone();

        let input1_2d = input1.into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| OnnxError::ShapeError("input1 is not 2-dimensional.".to_string()))?;

        //Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
        let input2_2d = input2.into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| OnnxError::ShapeMismatch("input2 is not 2-dimensional.".to_string()))?;

        let (_, k_a) = input1_2d.dim();
        let (k_b, _) = input2_2d.dim();
        if k_a != k_b {
            return Err(OnnxError::ShapeMismatch
                ("The inner dimensions of A' and B' do not match.".to_string()));
        }

        let y = input1_2d.dot(&input2_2d);
        Ok(vec![y.into_dyn()])
    }

    fn get_inputs(&self) -> Vec<String> {
        self.inputs_name.clone()
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