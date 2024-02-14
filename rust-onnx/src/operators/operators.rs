use std::collections::HashMap;
use ndarray::ArrayD;
use crate::onnx_parser::protoc_generated::onnx_ml_proto3::NodeProto;
use crate::operators::*;
use crate::operators::op_operator::Operator;
use crate::utils::errors::OnnxError;

pub fn create_operator(node: &NodeProto, initializer_set: &mut HashMap<String, ArrayD<f32>>) -> Result<Box<dyn Operator>, OnnxError> {
    match node.op_type.as_str() {
        "Add" => {
            Ok(Box::new(op_add::Add::new(node,initializer_set)))
        },
        "BatchNormalization" => {
            Ok(Box::new(op_batchnorm::BatchNorm::new(node,initializer_set)))
        },
        "Conv" => {
            Ok(Box::new(op_conv_optimized::Conv::new(node, initializer_set)))
        },
        "Flatten" =>{
            Ok(Box::new(op_flatten::Flatten::new(node, initializer_set)))
        },
        "Gemm" => {
            Ok(Box::new(op_gemm::Gemm::new(node, initializer_set)))
        },
        "GlobalAveragePool" => {
            Ok(Box::new(op_globalaveragepooling::GlobalAveragePool::new(node, initializer_set)))
        },
        "MatMul" => {
            Ok(Box::new(op_matmul::MatMul::new(node, initializer_set)))
        },
        "MaxPool" => {
            Ok(Box::new(op_maxpool::MaxPool::new(node, initializer_set)))
        },
        "Relu" => {
            Ok(Box::new(op_relu::ReLU::new(node, initializer_set)))
        },
        "Reshape" => {
            Ok(Box::new(op_reshape::Reshape::new(node,initializer_set)))
        },
        _ => {
            Err(OnnxError::UnsupportedOperator("Found an operator that doesn't have an \
            implementation yet.".to_string()))
        }
    }
}