use crate::utils::errors::OnnxError;
use super::op_operator::{Initializer, Operator};
use ndarray::ArrayD;
use std::collections::HashMap;
use crate::onnx_parser::protoc_generated::onnx_ml_proto3::NodeProto;

#[derive(Clone)]
pub struct Gemm {
    op_type: String,
    node_name: String,
    input_name: String,
    output_name: String,
    alpha: f32,
    beta: f32,
    trans_a: i64,
    trans_b: i64,
    initializers: Vec<Initializer>,
}

impl Gemm {
    pub fn new(node: &NodeProto, initializers: &mut HashMap<String, ArrayD<f32>>) -> Self {
        let op_type = node.op_type.to_owned();
        let node_name = node.name.to_owned();
        let output_name = node.output[0].to_owned();

        let input_name = node.input[0].to_owned();

        let mut initializers_vec = Vec::new();
        initializers_vec.push(Initializer::new(node.input[1].clone(), initializers.remove(&node.input[1])
            .ok_or(OnnxError::TensorNotFound("B initializer not found.".to_string()))
            .unwrap().to_owned()));

        let c = initializers.remove(&node.input[2]);
        if let Some(value) = c {
            initializers_vec.push(Initializer::new(node.input[2].clone(), value.to_owned()));
        }

        let mut alpha= 1.0;
        let mut beta= 1.0;
        let mut trans_a = 0;
        let mut trans_b = 0;

        for attribute in &node.attribute{
            match attribute.name.as_str(){
                "alpha" => alpha = attribute.f.to_owned(),
                "beta" => beta = attribute.f.to_owned(),
                "transA" => trans_a = attribute.i.to_owned(),
                "transB" => trans_b = attribute.i.to_owned(),
                _ => {}
            }
        }

        Self {
            op_type,
            node_name,
            input_name,
            output_name,
            alpha,
            beta,
            trans_a,
            trans_b,
            initializers: initializers_vec
        }
    }

    fn transpose(tensor: &ArrayD<f32>) -> ArrayD<f32> {
        tensor.t().to_owned()
    }
}

impl Operator for Gemm {
    //Y = alpha * A’ * B’ + beta * C
    fn execute(&self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>, OnnxError> {
        let a = inputs.get(&self.input_name)
            .ok_or_else(||
                OnnxError::TensorNotFound("Input tensor not found".to_string())).unwrap();
        let b = self.initializers[0].get_value();
        let mut c : Option<&ArrayD<f32>> = None;
        if self.initializers.len()>1{
            c = Some(self.initializers[1].get_value());
        }

        // Transpose A and B if needed
        let a_prime = if self.trans_a != 0 { Gemm::transpose(a) } else { a.clone() };
        let b_prime = if self.trans_b != 0 { Gemm::transpose(b) } else { b.clone() };

        //The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
        let a_prime_2d = a_prime.into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| OnnxError::DimensionalityError("a_prime is not 2-dimensional.".to_string()))?;
        //Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
        let b_prime_2d = b_prime.into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| OnnxError::DimensionalityError("b_prime is not 2-dimensional.".to_string()))?;

        // Check shapes for matrix multiplication
        let (_, k_a) = a_prime_2d.dim();
        let (k_b, _) = b_prime_2d.dim();
        if k_a != k_b {
            return Err(OnnxError::ShapeMismatch
                ("The inner dimensions of A' and B' do not match".to_string()));
        }

        // Perform matrix multiplication A' * B'
        let mut y = a_prime_2d.dot(&b_prime_2d);

        // Scale by alpha
        y *= self.alpha;

        // Add beta * C if C is provided and beta != 0
        if let Some(c_tensor) = c  {
            if self.beta != 0.0{
                y += &(c_tensor * self.beta);
            }

        }

        Ok(vec![y.into_dyn()]) // Convert to ArrayD<f32> if needed
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
        self.initializers.clone()
    }

    fn clone_box(&self) -> Box<dyn Operator> {
        Box::new(self.clone())
    }
}