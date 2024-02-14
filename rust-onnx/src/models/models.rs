#![allow(dead_code)]

use std::path::{PathBuf};
use crate::datasets::label_mappings::{IMAGENET_CLASSES, MNIST_CLASSES};
use crate::datasets::images_serialization::{serialize_imagenet_to_pb, serialize_mnist_image_to_pb};

pub const MODELS_NAMES: &[&str;4] = &["MNIST (opset-version=12)",
                                     "ResNet-18 (v1, opset-version=7)",
                                     "ResNet-34 (v2, opset-version=7)",
                                     "MobileNet (v2, opset-version=7)"];
#[derive(Clone)]
pub enum Model {
    MNIST,
    ResNet18v17,
    ResNet34v27,
    MobileNetv27,
}

impl Model {
    pub fn as_str(&self) -> &'static str {
        match self {
            Model::MNIST => "mnist-12",
            Model::ResNet18v17 => "resnet18-v1-7",
            Model::ResNet34v27 => "resnet34-v2-7",
            Model::MobileNetv27 => "mobilenet-v2-7",
        }
    }

    pub fn from_index(index: usize) -> Option<Model> {
        match index {
            0 => Some(Model::MNIST),
            1 => Some(Model::ResNet18v17),
            2 => Some(Model::ResNet34v27),
            3 => Some(Model::MobileNetv27),
            _ => None,
        }
    }

    pub fn get_num_classes(&self) -> usize{
        match self{
            Model::MNIST => 10,
            Model::ResNet18v17 => 1000,
            Model::ResNet34v27 => 1000,
            Model::MobileNetv27 => 1000,
        }
    }

    pub fn get_serialization_function(&self) -> fn(&PathBuf, &PathBuf){
        match self{
            Model::MNIST => serialize_mnist_image_to_pb,
            Model::ResNet18v17 => serialize_imagenet_to_pb,
            Model::ResNet34v27 => serialize_imagenet_to_pb,
            Model::MobileNetv27 => serialize_imagenet_to_pb,
        }
    }

    pub fn get_dataset_mapping(&self, index: usize) -> &str {
        match self{
            Model::MNIST => MNIST_CLASSES[index],
            Model::ResNet18v17 => IMAGENET_CLASSES[index],
            Model::ResNet34v27 => IMAGENET_CLASSES[index],
            Model::MobileNetv27 => IMAGENET_CLASSES[index],
        }
    }
}