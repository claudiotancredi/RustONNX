use super::op_operator::{Initializer, Operator};
use ndarray::{ArrayD, s, IxDyn, Dimension, ArrayBase, OwnedRepr, Axis, Array2, Ix2};
use std::collections::HashMap;
use crate::utils::errors::OnnxError;
use crate::onnx_parser::protoc_generated::onnx_ml_proto3::NodeProto;

#[derive(PartialEq, Clone)]
pub enum AutoPad {
    SameLower,
    SameUpper,
    NotSet,
    Valid,
}

#[derive(Clone)]
pub struct Conv {
    op_type: String,
    node_name: String,
    input_name: String,
    output_name: String,
    kernel_shape: Option<Vec<usize>>,
    strides: Option<Vec<usize>>,
    auto_pad: Option<AutoPad>,
    pads: Option<Vec<usize>>,
    group: usize,
    dilations: Option<Vec<usize>>,
    initializers: Vec<Initializer>,
}

impl Conv {
    pub fn new(node: &NodeProto, initializers: &mut HashMap<String, ArrayD<f32>>) -> Self {
        let op_type = node.op_type.to_owned();
        let node_name = node.name.to_owned();
        let output_name = node.output[0].to_owned();

        let input_name = node.input[0].to_owned();
        let kernel_name = node.input[1].to_owned();

        let mut initializers_vec = Vec::new();
        initializers_vec.push(Initializer::new(kernel_name.clone(), initializers.remove(kernel_name.as_str())
            .ok_or(OnnxError::TensorNotFound("W initializer not found.".to_string())).unwrap()));


        if node.input.len() == 3{
            let bias_name = node.input[2].to_owned();
            initializers_vec.push(Initializer::new(bias_name.to_owned(), initializers.remove(bias_name.as_str())
                .ok_or(OnnxError::TensorNotFound("B initializer not found.".to_string())).unwrap()));
        }

        let mut kernel_shape = None;
        let mut strides = None;
        let mut group = 1usize; // default value
        let mut dilations = None;
        let mut pads = None;
        let mut auto_pad = None;

        for attribute in &node.attribute {
            match attribute.name.as_str() {
                "kernel_shape" => kernel_shape = Some(attribute.ints.iter().map(|x| *x as usize).collect()),
                "strides" => strides = Some(attribute.ints.iter().map(|x| *x as usize).collect()),
                "group" => group = attribute.i.to_owned() as usize,
                "dilations" => dilations = Some(attribute.ints.iter().map(|x| *x  as usize).collect()),
                "pads" => pads = Some(attribute.ints.iter().map(|x| *x  as usize).collect()),
                "auto_pad" => {
                    auto_pad = Some(match String::from_utf8(attribute.s.clone()) {
                        Ok(value) => match value.as_str() {
                            "SAME_UPPER" => AutoPad::SameUpper,
                            "SAME_LOWER" => AutoPad::SameLower,
                            "VALID" => AutoPad::Valid,
                            _ => AutoPad::NotSet,
                        },
                        Err(_) => AutoPad::NotSet,
                    });
                }
                // Handle other attributes as needed
                _ => {}
            }
        }



        Conv {
            op_type,
            node_name,
            input_name,
            output_name,
            kernel_shape,
            strides,
            auto_pad,
            pads,
            group,
            dilations,
            initializers: initializers_vec
        }
    }

    fn execute_conv_optimized(x: ArrayD<f32>, w: ArrayD<f32>, b: Option<&ArrayD<f32>>, auto_pad: &Option<AutoPad>, dilations: &Vec<usize>, kernel_shape: &Vec<usize>, pads: &Vec<usize>, strides: &Vec<usize>) -> Result<Vec<ArrayD<f32>>, String> {

        let mut w_copy = w.clone();
        let mut kernel_shape_copy = kernel_shape.clone();
        let mut pads_copy = pads.clone();

        if dilations[0] != 1 || dilations.iter().min() != dilations.iter().max() {
            let nd = dilations.len();
            let mut new_kernel_shape = Vec::new();
            let mut new_shape = w_copy.shape().to_vec();
            new_shape.truncate(new_shape.len() - nd);

            for (i, &d) in dilations.iter().enumerate() {
                let di = w_copy.ndim() - nd + i;
                new_shape.push(w_copy.shape()[di] + (w_copy.shape()[di] - 1) * (d - 1));
                new_kernel_shape.push(kernel_shape_copy[i] + (kernel_shape_copy[i] - 1) * (d - 1));
            }

            let mut new_w = ArrayD::zeros(IxDyn(&new_shape));

            for idx in w_copy.indexed_iter() {
                let mut new_idx = Vec::new();

                // Manually push each dimension into the Vec
                for &dim_size in idx.0.slice() {
                    new_idx.push(dim_size);
                }
                // Extend the new_idx with zeros, the number of zeros is determined by 'nd'
                new_idx.resize(new_idx.len() + nd, 0);

                for (i, &d) in dilations.iter().enumerate() {
                    if d > 1 {
                        new_idx[w_copy.ndim() - nd + i] *= d;
                    }
                }

                new_w[new_idx.as_slice()] = *idx.1;
            }

            w_copy = ArrayD::from_shape_vec(IxDyn(&new_w.shape()), new_w.iter().cloned().collect()).unwrap();
            kernel_shape_copy = new_kernel_shape;

        }

        if auto_pad.is_some(){
            pads_copy = match auto_pad.as_ref().unwrap() {
                AutoPad::SameLower | AutoPad::SameUpper | AutoPad::Valid => {
                    let mut head = Vec::new();
                    let mut tail = Vec::new();

                    for i in 0..(x.ndim() - 2) {
                        let d = x.shape()[i];
                        let target_size = (d + strides[i] - 1) / strides[i];
                        let pad_needed = (target_size - 1) * strides[i] + kernel_shape_copy[i] - d;

                        let pad_head = match auto_pad.as_ref().unwrap() {
                            AutoPad::SameLower => (pad_needed + 1) / 2,
                            _ => pad_needed / 2,
                        };

                        let pad_tail = pad_needed - pad_head;
                        head.push(pad_head);
                        tail.push(pad_tail);
                    }

                    [head, tail].concat()
                },
                _ => pads.clone(),
            };
        }

        let im2col_res = Conv::im2col_fast(x.clone(), &kernel_shape_copy, &pads_copy, strides);

        let kernel_height = kernel_shape_copy[0];
        let kernel_width = kernel_shape_copy[1];
        let output_height = (x.shape()[2] + pads_copy[0] + pads_copy[2] -kernel_height)/strides[0]+1;
        let output_width = (x.shape()[3] + pads_copy[1] + pads_copy[3] -kernel_width)/strides[1]+1;

        //STEP 2
        let out_channels = w_copy.shape()[0];
        let flattened_size = w_copy.len() / out_channels;
        let w_copy_2d = w_copy.clone().into_shape((out_channels, flattened_size)).unwrap();
        let w_copy_2d_transposed = w_copy_2d.t();

        //STEP 3
        let dot_result = im2col_res.dot(&w_copy_2d_transposed);

        //STEP 4
        let mut res = dot_result
            .into_shape((output_height, output_width, out_channels))
            .unwrap()
            .permuted_axes([2, 0, 1]).insert_axis(Axis(0));

        if b.is_some(){
            let bias_shape = [1, b.unwrap().len(), 1, 1];
            let broadcasted_bias = b.unwrap().view().into_shape(IxDyn(&bias_shape))
                .or(Err("Bias cannot be broadcasted into the desired shape."))?;

            // Add the bias to the result tensor
            res += &broadcasted_bias;
        }

        Ok(vec![res.into_dyn()])
    }

    fn im2col_fast(x: ArrayBase<OwnedRepr<f32>, IxDyn>, kernel_shape: &Vec<usize>, pads: &Vec<usize>, strides: &Vec<usize>) -> ArrayBase<OwnedRepr<f32>, Ix2> {//-> Vec<usize>{
        let n_dims = kernel_shape.len();
        let mut nc: Vec<(usize, usize)> = vec![(0, 0); 2];

        for i in 0..n_dims{
            nc.push((pads[i], pads[i+n_dims]));
        }

        let x_padded = Conv::pad(&x, nc);

        let m = x.shape()[0];
        let n_c = x.shape()[1];
        let height = x.shape()[2];
        let width = x.shape()[3];

        let kernel_height = kernel_shape[0];
        let kernel_width = kernel_shape[1];

        let output_height = (height + pads[0] + pads[2] -kernel_height)/strides[0]+1;
        let output_width = (width + pads[1] + pads[3] -kernel_width)/strides[1]+1;

        let rows = m * output_height * output_width;
        let cols = n_c * kernel_height * kernel_width;
        let output_size = rows * cols;

        let mut output = Vec::with_capacity(output_size);

        for b in 0..m {
            for h in 0..output_height {
                let row_start = h * strides[0];
                let row_end = row_start + kernel_height;
                for w in 0..output_width {
                    let col_start = w * strides[1];
                    let col_end = col_start + kernel_width;
                    for c in 0..n_c {

                        // Extract the patch and flatten it
                        let patch = x_padded.slice(s![b, c, row_start..row_end, col_start..col_end]);
                        output.extend(patch.iter().cloned());
                    }
                }
            }
        }

        // Convert the output vector to a 2D Array
        // Swapping the rows and columns dimensions
        Array2::from_shape_vec((rows, cols), output).unwrap()

    }

    fn pad(x:&ArrayBase<OwnedRepr<f32>, IxDyn>, padding: Vec<(usize, usize)>) -> ArrayD<f32>{
        let shape = x.shape();
        let new_shape = [
            shape[0] + padding[0].0 + padding[0].1,
            shape[1] + padding[1].0 + padding[1].1,
            shape[2] + padding[2].0 + padding[2].1,
            shape[3] + padding[3].0 + padding[3].1,
        ];
        let mut padded_tensor = ArrayD::<f32>::zeros(IxDyn(&new_shape));

        padded_tensor.slice_mut(s![
            padding[0].0..shape[0] + padding[0].0,
            padding[1].0..shape[1] + padding[1].0,
            padding[2].0..shape[2] + padding[2].0,
            padding[3].0..shape[3] + padding[3].0
        ]).assign(x);

        padded_tensor

    }

}

impl Operator for Conv{
    fn execute(&self, inputs: &HashMap<String, ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>, OnnxError> {

        // 1. Retrieve input tensors X and W, and optionally B.
        // 2. Apply padding according to `auto_pad` or `pads`.
        // 3. Handle dilations and groups.
        // 4. Perform the convolution operation.
        // 5. Return the output tensor Y.

        let input_name = &self.input_name;
        let x = inputs.get(input_name).ok_or(OnnxError::TensorNotFound("Input tensor not found.".to_string()))?;
        let w = self.initializers[0].get_value();
        let auto_pad = &self.auto_pad;
        let mut b_init=None ;
        if self.initializers.len()>1{
            b_init = Some(self.initializers[1].get_value());
        }

        if x.ndim()<4{
            return Err(OnnxError::ShapeMismatch(format!("The input must have at least 4 dimensions but its shape is {:?}.", x.shape())));
        }

        let dilations = self.dilations.clone().unwrap_or_else(|| vec![1; x.ndim() - 2]);
        let kernel_shape = self.kernel_shape.clone().unwrap_or_else(|| w.shape()[2..].to_vec());
        let pads = self.pads.clone().unwrap_or_else(|| vec![0; x.ndim() - 2].repeat(2));
        let strides = self.strides.clone().unwrap_or_else(|| vec![1; x.ndim() - 2]);
        // Initial shape checks
        if x.shape()[1] != w.shape()[1] * self.group || w.shape()[0] % self.group != 0 {
            return Err(OnnxError::ShapeMismatch(format!(
                "Shape inconsistencies, X.shape={:?}, W.shape={:?}, group={}, \
                W should be {:?}.",
                x.shape(),
                w.shape(),
                self.group,
                (w.shape()[0], x.shape()[1] / self.group, w.shape()[1..].iter().product::<usize>() / x.shape()[1] * self.group)
            )));
        }

        if self.group>1{
            let mut res = vec![];
            let mut td = 0;
            let mg = w.shape()[0]/self.group;
            let dw = w.shape()[1];

            //Iterate over the batch
            for b in 0..x.shape()[0]{
                for g in 0..self.group{
                    let gx_view = x.slice(s![b..b+1, g*dw..(g+1)*dw, .., ..]);
                    let gw_view = w.slice(s![g*mg..(g+1)*mg, .., .., ..]);
                    // Check if the sliced shapes are correct
                    if gx_view.shape()[1] != dw || gw_view.shape()[0] != mg {
                        return Err(OnnxError::ShapeMismatch(
                            format!("Incorrect shape after slicing for group {}. gx_view.shape={:?}, gw_view.shape={:?}.", g, gx_view.shape(), gw_view.shape())));
                    }
                    let gx: ArrayD<f32> = ArrayD::from_shape_vec(IxDyn(&gx_view.shape()), gx_view.iter().cloned().collect())
                        .map_err(|_|OnnxError::ShapeError("Failed to create gx tensor.".to_string()))?;
                    let gw: ArrayD<f32> = ArrayD::from_shape_vec(IxDyn(&gw_view.shape()), gw_view.iter().cloned().collect())
                        .map_err(|_|OnnxError::ShapeError("Failed to create gw tensor.".to_string()))?;

                    let cv = Conv::execute_conv_optimized(gx, gw, None, auto_pad, &dilations, &kernel_shape, &pads, &strides).unwrap();
                    if b==0 {
                        td += cv[0].shape()[1];
                    }
                    res.push((b, cv))
                }
            }
            let mut new_shape = vec![x.shape()[0]];
            new_shape.extend_from_slice(&res[0].1[0].shape()[1..]);
            new_shape[1] = td;
            let mut result : ArrayD<f32> = ArrayD::zeros(IxDyn(&new_shape));
            let mut p= 0;
            for (b, cv) in res.iter(){
                let mut slice = result.slice_mut(s![*b..*b+1, p..p+cv[0].shape()[1], .., ..]);
                slice.assign(&cv[0].view());
                p+=cv[0].shape()[1];
                if p >= result.shape()[1]{
                    p=0;
                }
            }
            if b_init.is_some(){
                let mut new_shape = vec![1; result.ndim()];
                new_shape[1] = b_init.unwrap().shape()[0];
                if let Some(b_value) = b_init{
                    let tmp = b_value.clone().into_shape(IxDyn(&new_shape)).unwrap();
                    result=result+tmp;
                }
            }
            let expected_output_channels = w.shape()[0];
            let final_shape = result.shape();
            if final_shape[1] != expected_output_channels {
                return Err(OnnxError::ShapeMismatch(format!(
                    "The number of output channels in the final result does not match the expected number. Expected {}, got {}. Final shape: {:?}",
                    expected_output_channels, final_shape[1], final_shape
                )));
            }
            return Ok(vec![result]);
        }

        Ok(Conv::execute_conv_optimized(x.clone(), w.clone(), b_init, auto_pad, &dilations, &kernel_shape, &pads, &strides).unwrap())
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

