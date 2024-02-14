use crate::utils::errors::OnnxError;
use super::op_operator::{Initializer, Operator};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use crate::onnx_parser::protoc_generated::onnx_ml_proto3::NodeProto;

#[derive(Clone)]
#[allow(dead_code)]
pub struct MaxPool {
    op_type: String,
    node_name: String,
    input_name: String,
    output_name: String,
    kernel_shape: Option<Vec<i64>>,
    strides: Option<Vec<i64>>,
    pads: Option<Vec<i64>>,
    ceil_mode: Option<i64>,
    storage_order: Option<i64>,
    dilations: Option<Vec<i64>>,
}

impl MaxPool {
    pub fn new(node: &NodeProto, _initializers: &mut HashMap<String, ArrayD<f32>>) -> Self {
        let op_type = node.op_type.to_owned();
        let node_name = node.name.to_owned();
        let input_name = node.input[0].to_owned();
        let output_name = node.output[0].to_owned();

        let mut ceil_mode : Option<i64> = None;
        let mut dilations = None;
        let mut pads= None;
        let mut strides = None;
        let mut kernel_shape = None;
        let mut storage_order = None;

        for attribute in &node.attribute {
            match attribute.name.as_str() {
                "ceil_mode" => ceil_mode = Some(attribute.i),
                "strides" => strides = Some(attribute.ints.clone()),
                "kernel_shape" => kernel_shape = Some (attribute.ints.clone()),
                "dilations" => dilations = {
                    Some(attribute.ints.clone()) },
                "pads" => pads = Some ( attribute.ints.clone()),
                "storage_order" => storage_order = Some(attribute.i),
                // Handle other attributes as needed
                _ => {}
            }
        }

        MaxPool {
            op_type,
            node_name,
            input_name,
            output_name,
            kernel_shape,
            strides,
            pads,
            ceil_mode,
            storage_order,
            dilations,
        }
    }
}

impl Operator for MaxPool {
    fn execute(&self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>, OnnxError> {
        let input_name = &self.input_name;
        let input = inputs.get(input_name)
            .ok_or_else(||
                OnnxError::TensorNotFound("Input tensor not found.".to_string())).unwrap();

        // Validate input tensor dimensions (assuming 4D tensor: [N, C, H, W])
        if input.ndim() != 4 {
            return Err(OnnxError::ShapeMismatch(format!("The input must have at least 4 dimensions but its shape is {:?}.", input.shape())));
        }

        let kernel_shape = self.kernel_shape.clone().unwrap();
        let dilations = self.dilations.clone().unwrap_or_else(|| vec![1; kernel_shape.len()]);
        let pads = self.pads.clone().unwrap_or_else(|| vec![0; kernel_shape.len()].repeat(2));
        let strides = self.strides.clone().unwrap_or_else(|| vec![1; kernel_shape.len()]);

        let n_dims = kernel_shape.len();
        let new_pads: Vec<(i64, i64)> = (0..n_dims)
            .map(|i| (pads[i ], pads[i + n_dims ]))
            .collect();

        let input_spatial_shape = input.shape()[2..].to_vec();
        let mut output_spatial_shape = vec![0; input_spatial_shape.len()];

        if let Some(_) = self.ceil_mode{
            for i in 0..input_spatial_shape.len(){
                output_spatial_shape[i] =  (((input_spatial_shape[i] as f32
                    + (new_pads[i].0 + new_pads[i].1) as f32
                    - (dilations[i] as f32 * (kernel_shape[i] as f32 - 1.0) + 1.0))
                    / strides[i] as f32)
                    + 1.0)
                    .ceil() as usize;
            }

        } else{
            for i in 0..input_spatial_shape.len(){
                output_spatial_shape[i] =  (((input_spatial_shape[i] as f32
                    + (new_pads[i].0 + new_pads[i].1) as f32
                    - (dilations[i] as f32 * (kernel_shape[i] as f32 - 1.0) + 1.0))
                    / strides[i] as f32)
                    + 1.0)
                    .floor() as usize;
            }
        }

        //Perform 2D maxpool
        let mut y_dims = input.shape()[..2].to_vec();
        y_dims.extend(&output_spatial_shape);
        let y = ArrayD::zeros(IxDyn(&y_dims));

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let height = input.shape()[2];
        let width = input.shape()[3];

        // Calculate output dimensions
        let pooled_height = output_spatial_shape[0];
        let pooled_width = output_spatial_shape[1];

        let total_channels = batch_size*channels;

        let stride_h = strides[0];
        let stride_w = strides[1];

        let x_step = height*width;
        let y_step = pooled_height*pooled_width;

        let dilation_h = dilations[0];
        let dilation_w = dilations[1];

        let x_data = input.iter().cloned().collect::<Vec<f32>>();
        let mut y_data = y.iter().cloned().collect::<Vec<f32>>();

        let kernel_height = kernel_shape[0];
        let kernel_width = kernel_shape[1];

        for c in 0..total_channels{
            let x_d = c*x_step;
            let y_d = c*y_step;
            for ph in 0..pooled_height{
                let hstart = ph as i64 * stride_h -new_pads[0].0;
                let hend = hstart+kernel_height *dilation_h;
                for pw in 0..pooled_width{
                    let wstart = pw as i64*stride_w -new_pads[1].0;
                    let wend = wstart+kernel_width *dilation_w;
                    let pool_index = ph * pooled_width+pw;
                    let mut yh=None;
                    for h in (hstart..hend).step_by(dilation_h as usize){
                        if h<0 || h>=height as i64{
                            continue;
                        }
                        for w in (wstart..wend).step_by(dilation_w as usize){
                            if w<0 || w>=width as i64{
                                continue;
                            }
                            let input_index = h*width as i64+w;
                            if (x_d as i64 + input_index )<0 || (x_d + input_index as usize) >=x_data.len(){
                                continue;
                            }
                            if yh.is_none() || x_data[x_d+input_index as usize]>yh.unwrap_or(f32::MAX){
                                yh = Some(x_data[x_d+input_index as usize]);
                            }
                        }
                    }
                    if yh.is_none(){
                        continue;
                    }
                    y_data[y_d+pool_index]=yh.unwrap();
                }
            }
        }

        let result = ArrayD::from_shape_vec(y_dims, y_data)
            .map_err(|_| OnnxError::ShapeError("Error while creating maxpool result tensor starting from its shape.".to_string()))?;
        Ok(vec![result])
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