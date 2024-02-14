use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use image::imageops;
use ndarray::{Array, Array2, Array3, Array4, ArrayD, Axis};
use crate::onnx_parser::protoc_generated::onnx_ml_proto3::TensorProto;
use image::io::Reader as ImageReader;
use protobuf::Message;

pub fn serialize_imagenet_to_pb(image_path: &PathBuf, pb_path: &PathBuf) {
    const MIN_SIZE: u32 = 256;
    const CROP_SIZE: u32 = 224;
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    const SCALE_FACTOR: f32 = 255.0;
    const CHANNELS : i64 = 3;

    // Read the image file
    let mut img = ImageReader::open(image_path).unwrap().decode().unwrap();
    let (width, height) = (img.width(), img.height());

    let (scaled_width, scaled_height) = if width > height {
        (MIN_SIZE * width / height, MIN_SIZE)
    } else {
        (MIN_SIZE, MIN_SIZE * height / width)
    };

    img = img.resize(scaled_width, scaled_height, imageops::FilterType::Triangle);

    let crop_x = (scaled_width - CROP_SIZE) / 2;
    let crop_y = (scaled_height - CROP_SIZE) / 2;

    img = img.crop_imm(crop_x, crop_y, CROP_SIZE, CROP_SIZE);

    let img_rgb = img.to_rgb8();
    let raw_data = img_rgb.into_raw();

    let mut r_color = Vec::new();
    let mut g_color = Vec::new();
    let mut b_color = Vec::new();

    for i in 0..raw_data.len() / CHANNELS as usize{
        r_color.push(raw_data[CHANNELS as usize * i]);
        g_color.push(raw_data[CHANNELS as usize * i + 1]);
        b_color.push(raw_data[CHANNELS as usize * i + 2]);
    }

    let r_array: Array2<u8> =
        Array::from_shape_vec((CROP_SIZE as usize, CROP_SIZE as usize), r_color).unwrap();
    let g_array: Array2<u8> =
        Array::from_shape_vec((CROP_SIZE as usize, CROP_SIZE as usize), g_color).unwrap();
    let b_array: Array2<u8> =
        Array::from_shape_vec((CROP_SIZE as usize, CROP_SIZE as usize), b_color).unwrap();

    let mut rgb_arr: Array3<u8> =
        ndarray::stack(Axis(2), &[r_array.view(), g_array.view(), b_array.view()]).unwrap();
    // Transpose from HWC to CHW
    rgb_arr.swap_axes(0, 2);

    let mean = Array::from_shape_vec(
        (CHANNELS as usize, 1, 1),
        vec![
            MEAN[0] * SCALE_FACTOR,
            MEAN[1] * SCALE_FACTOR,
            MEAN[2] * SCALE_FACTOR,
        ],
    )
        .unwrap();

    let std = Array::from_shape_vec(
        (CHANNELS as usize, 1, 1),
        vec![
            STD[0] * SCALE_FACTOR,
            STD[1] * SCALE_FACTOR,
            STD[2] * SCALE_FACTOR,
        ],
    )
        .unwrap();

    let mut arr_float: Array3<f32> = rgb_arr.mapv(|x| x as f32);

    arr_float -= &mean;
    arr_float /= &std;

    let arr_float_batch: Array4<f32> = arr_float.insert_axis(Axis(0));
    let arr_d: ArrayD<f32> = arr_float_batch.into_dimensionality().unwrap();

    prepare_bytes_and_write_to_file(&arr_d, CHANNELS, CROP_SIZE, CROP_SIZE,  pb_path);
}
pub fn serialize_mnist_image_to_pb(image_path: &PathBuf, pb_path: &PathBuf) {
    const MNIST_SIZE : u32 = 28;
    const CHANNELS : i64 = 1;
    // Read the image file
    let mut img = ImageReader::open(image_path).unwrap().decode().unwrap();

    img = img.resize(MNIST_SIZE, MNIST_SIZE, imageops::FilterType::Triangle);

    let img_gray = img.to_luma8();
    let raw_data = img_gray.into_raw();

    let gray_array: Array2<u8> =
        Array::from_shape_vec((MNIST_SIZE as usize, MNIST_SIZE as usize), raw_data).unwrap();


    let arr_float: Array2<f32> = gray_array.mapv(|x| x as f32);

    let arr_f_im: Array3<f32> = arr_float.insert_axis(Axis(0));
    let arr_float_batch: Array4<f32> = arr_f_im.insert_axis(Axis(0));
    let arr_d: ArrayD<f32> = arr_float_batch.into_dimensionality().unwrap();

    prepare_bytes_and_write_to_file(&arr_d, CHANNELS, MNIST_SIZE, MNIST_SIZE, pb_path);
}

fn prepare_bytes_and_write_to_file(x: &ArrayD<f32>, channels: i64, height: u32, width: u32, pb_path: &PathBuf) {
    let flat: Vec<f32> = x.iter().cloned().collect(); // Step 1: Flatten the array
    let mut img_bytes: Vec<u8> = Vec::with_capacity(flat.len() * 4); // Step 2: Allocate Vec<u8>

    for &value in &flat {
        let byte_repr: [u8; 4] = value.to_le_bytes(); // Convert each f32 to 4 bytes
        img_bytes.extend_from_slice(&byte_repr); // Append bytes to Vec<u8>
    }

    let image_proto = TensorProto {
        dims: vec![1i64, channels, height as i64, width as i64],
        data_type: 1,
        segment: Default::default(),
        float_data: vec![],
        int32_data: vec![],
        string_data: vec![],
        int64_data: vec![],
        name: "data".to_string(),
        raw_data: img_bytes,
        external_data: vec![],
        data_location: Default::default(),
        double_data: vec![],
        uint64_data: vec![],
        special_fields: Default::default(),
        doc_string: "".to_string()
    };

    let mut buf = Vec::new();
    image_proto.write_to_vec(&mut buf).unwrap();

    // Write the byte vector to a .pb file
    let mut file = File::create(pb_path).unwrap();
    file.write_all(&buf).unwrap();
}
