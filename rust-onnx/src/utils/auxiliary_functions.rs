use ndarray::{ArrayBase, ArrayD, Axis, IxDyn, OwnedRepr};
use std::fs::File;
use std::io::Write;
use prettytable::{format, row, Row, Table, Cell, Attr};
extern crate protobuf;
use std::path::PathBuf;
use colored::Colorize;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use crate::models::models::Model;
use crate::utils::errors::OnnxError;

fn argmax_per_row(matrix: &ArrayD<f32>) -> Vec<usize> {
    matrix
        .axis_iter(Axis(0)) // Iterate over rows
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0) // Default to 0 if the row is empty
        })
        .collect()
}

fn compute_accuracy(vec1: &[usize], vec2: &[usize]) -> Result<f32, OnnxError> {

    if vec1.is_empty() || vec2.is_empty() {
        return Err(OnnxError::EmptyContainer("Vectors cannot be empty.".to_string()));
    }

    if vec1.len() != vec2.len() {
        return Err(OnnxError::ShapeMismatch("Vectors must be of the same length.".to_string()));
    }

    let count = vec1.iter().zip(vec2.iter()).filter(|(&x, &y)| x == y).count();
    Ok(count  as f32/vec1.len() as f32)
}

fn compute_error_rate(vec1: &[usize], vec2: &[usize]) -> Result<f32, OnnxError> {

    if vec1.is_empty() || vec2.is_empty() {
        return Err(OnnxError::EmptyContainer("Vectors cannot be empty.".to_string()));
    }

    if vec1.len() != vec2.len() {
        return Err(OnnxError::ShapeMismatch("Vectors must be of the same length.".to_string()));
    }

    let count = vec1.iter().zip(vec2.iter()).filter(|(&x, &y)| x != y).count();
    Ok(count  as f32/vec1.len() as f32)
}

pub fn display_model_info(model_name: &str, number_of_nodes: usize, number_of_nodes_in_parallel: usize) {
    let mut table = Table::new();
    table.set_format(*format::consts::FORMAT_NO_LINESEP_WITH_TITLE);
    table.set_titles(Row::new(vec![
        Cell::new("ONNX model information")
            .with_style(Attr::Bold)
            .with_hspan(2)
    ]));
    table.add_row(row![
        "Model name",
        model_name,
    ]);
    table.add_row(row![
        "Total number of nodes",
        number_of_nodes,
    ]);
    table.add_row(row![
       "Number of nodes that will run in parallel",
        number_of_nodes_in_parallel
    ]);
    println!("\n🟢 All good, starting execution...\n");
    table.printstd();
    println!();
}

pub fn print_results (model: Model, files: &Vec<PathBuf>, final_output: &ArrayBase<OwnedRepr<f32>, IxDyn>, label_stack: &ArrayD<f32>) {
    const MAX_ELEMENTS : usize = 50;
    const FILE_NAME : &str = "results.txt";

    let predictions = argmax_per_row(&final_output);

    let ground_truth = argmax_per_row(&label_stack);

    let error_rate = compute_error_rate(&predictions, &ground_truth).unwrap();

    let accuracy = compute_accuracy(&predictions, &ground_truth).unwrap();

    let table = setup_results_table(&model, files, &predictions, &ground_truth, predictions.len());

    if files.len() <= MAX_ELEMENTS {
        println!("\nResults (green = correct prediction, red = wrong prediction):\n\n{}", table.to_string());
    } else {
        let reduced_table = setup_results_table(&model, files, &predictions, &ground_truth, MAX_ELEMENTS);
        println!("\nThe dataset size is too big for all the results to be printed in the console.\nTo improve \
        readability, the results have been stored in the file \"{}\".\n\
        However, here is a sneak peek to the first {} results (green = correct prediction, red = wrong prediction):\n\n{}", &FILE_NAME, MAX_ELEMENTS, reduced_table.to_string());
        let mut out = File::create(FILE_NAME).unwrap();
        out.write_all(strip_ansi_codes(table.to_string() + &format!("\nError rate: {}\nAccuracy: {}", error_rate, accuracy)).as_bytes()).unwrap();
    }
    println!("Error rate: {}\nAccuracy: {}\n", error_rate, accuracy);
}

fn setup_results_table(model: &Model, files: &Vec<PathBuf>, predictions: &Vec<usize>, ground_truth: &Vec<usize>, max_elements: usize)->Table{
    const LIMIT_IN_PRINTING_LABELS_CHARS : usize = 18;
    let mut table = Table::new();
    table.set_format(*format::consts::FORMAT_NO_LINESEP_WITH_TITLE);
    table.set_titles(row![
            "Input file".bold(),
            "Prediction".bold(),
            "Predicted label".bold(),
            "Ground truth".bold(),
            "Ground truth label".bold()
    ]);
    for (i, f) in files.iter().take(max_elements).enumerate() {
        let correct_prediction = predictions[i] == ground_truth[i];
        let prediction_string = predictions[i].to_string();
        let gt_string = ground_truth[i].to_string();

        let (f_style, prediction_style, gt_style) = if correct_prediction {
            (f.file_name().unwrap().to_str().unwrap().bright_green(), prediction_string.bright_green(), gt_string.bright_green())
        } else {
            (f.file_name().unwrap().to_str().unwrap().bright_red(), prediction_string.bright_red(), gt_string.bright_red())
        };

        let mut predicted_label_string = model.get_dataset_mapping(predictions[i])[0..std::cmp::min(LIMIT_IN_PRINTING_LABELS_CHARS, model.get_dataset_mapping(predictions[i]).len())].to_string();
        if std::cmp::min(LIMIT_IN_PRINTING_LABELS_CHARS, model.get_dataset_mapping(predictions[i]).len()) == LIMIT_IN_PRINTING_LABELS_CHARS {
            predicted_label_string += "...";
        }
        let predicted_label = predicted_label_string.as_str();

        let mut ground_truth_label_string = model.get_dataset_mapping(ground_truth[i])[0..std::cmp::min(LIMIT_IN_PRINTING_LABELS_CHARS, model.get_dataset_mapping(ground_truth[i]).len())].to_string();
        if std::cmp::min(LIMIT_IN_PRINTING_LABELS_CHARS, model.get_dataset_mapping(ground_truth[i]).len()) == LIMIT_IN_PRINTING_LABELS_CHARS {
            ground_truth_label_string += "...";
        }
        let ground_truth_label = ground_truth_label_string.as_str();

        let (pl_style, gtl_style) = if correct_prediction {
            (predicted_label.bright_green(), ground_truth_label.bright_green())
        } else {
            (predicted_label.bright_red(), ground_truth_label.bright_red())
        };
        table.add_row(row![
                f_style, prediction_style, pl_style, gt_style, gtl_style
            ]);
    }
    table
}

fn strip_ansi_codes(input: String) -> String {
    let re = regex::Regex::new("\x1B\\[[0-9;]*[a-zA-Z]").unwrap();
    re.replace_all(input.as_str(), "").to_string()
}

pub fn setup_progress_bar(multi_progress: &MultiProgress, length: u64, show_pos_len: bool, text: &str) -> ProgressBar {
    let progress_bar = multi_progress.add(ProgressBar::new(length));
    let template_string = if show_pos_len {
        format!("\n{{bar:60.green/white}} {{percent}}% [{{pos}}/{{len}} {}]", text)
    } else {
        format!("\n{{bar:60.green/white}} {{percent}}% [{}]", text)
    };

    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template(&template_string)
            .unwrap()
            .progress_chars("█▁"),
    );
    progress_bar
}