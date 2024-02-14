use crate::utils::errors::OnnxError;
use super::op_operator::{Initializer, Operator};
use ndarray::ArrayD;
use std::collections::HashMap;
use crate::onnx_parser::protoc_generated::onnx_ml_proto3::NodeProto;

#[derive(Clone)]
pub struct BatchNorm {
    op_type: String,
    node_name: String,
    input_name: String,
    output_name: String,
    epsilon: f32,
    initializers: Vec<Initializer>,
}
impl BatchNorm {
    pub fn new(node: &NodeProto, initializers: &mut HashMap<String, ArrayD<f32>>) -> Self {
        let op_type = node.op_type.to_owned();
        let node_name = node.name.to_owned();
        let output_name = node.output[0].to_owned();
        let input_name = node.input[0].to_owned();

        let opt_epsilon = node.attribute.iter().find(|attr| attr.name == "epsilon");
        let epsilon = match opt_epsilon{
            Some(x) => x.f,
            None => 1e-5
        };

        let mut initializers_vec = Vec::new();

        for inp_name in &node.input{
            match inp_name {
                _ if inp_name.contains("gamma") =>{
                    initializers_vec.push(Initializer::new(inp_name.to_owned(), initializers.remove(inp_name)
                        .ok_or(OnnxError::TensorNotFound("Scale initializer not found.".to_string()))
                        .unwrap()));
                } ,
                _ if inp_name.contains("beta") => {
                    initializers_vec.push(Initializer::new(inp_name.to_owned(), initializers.remove(inp_name)
                        .ok_or(OnnxError::TensorNotFound("B initializer not found.".to_string()))
                        .unwrap()));
                },
                _ if inp_name.contains("mean") =>{
                    initializers_vec.push(Initializer::new(inp_name.to_owned(), initializers.remove(inp_name)
                        .ok_or(OnnxError::TensorNotFound("Mean initializer not found.".to_string()))
                        .unwrap()));
                },
                _ if inp_name.contains("var") =>{
                    initializers_vec.push(Initializer::new(inp_name.to_owned(), initializers.remove(inp_name)
                        .ok_or(OnnxError::TensorNotFound("Var initializer not found.".to_string()))
                        .unwrap()));
                },
                _ => {  }
            }
        }

        Self {
            op_type,
            node_name,
            input_name,
            output_name,
            epsilon,
            initializers: initializers_vec
        }
    }
}

impl Operator for BatchNorm {
    fn execute(&self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>, OnnxError> {
        let x = inputs.get(&self.input_name)
            .ok_or_else(||
                OnnxError::TensorNotFound("Input tensor not found".to_string())).unwrap();
        let scale = self.initializers.iter()
            .filter(|x|x.get_name().contains("gamma"))
            .collect::<Vec<_>>()[0].get_value();
        let b = self.initializers.iter()
            .filter(|x|x.get_name().contains("beta"))
            .collect::<Vec<_>>()[0].get_value();
        let mean = self.initializers.iter()
            .filter(|x|x.get_name().contains("mean"))
            .collect::<Vec<_>>()[0].get_value();
        let var = self.initializers.iter()
            .filter(|x|x.get_name().contains("var"))
            .collect::<Vec<_>>()[0].get_value();

        // Assuming the second dimension is the channel
        let channel_axis = 1;
        let mut y = x.clone(); // Clone the shape and data

        for (((mut y_slice, mean_val), var_val), (scale_val, b_val)) in
            y.axis_iter_mut(ndarray::Axis(channel_axis))
                .zip(mean.iter())
                .zip(var.iter())
                .zip(scale.iter().zip(b.iter())) {
                for y_elem in y_slice.iter_mut() {
                    *y_elem = ((*y_elem - mean_val) / ((var_val + self.epsilon).sqrt())) * scale_val + b_val;
                }
        }

        Ok(vec![y])
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