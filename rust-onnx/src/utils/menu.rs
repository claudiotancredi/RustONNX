use std::{fs, process};
use dialoguer::{theme::ColorfulTheme, Select, Input};
use std::path::{PathBuf};
use crate::models::models::{Model, MODELS_NAMES};
use crate::utils::shared_constants::SUPPORTED_IMAGE_FORMATS;

fn print_intro() {
    const PROGRAM_NAME_AND_DESCRIPTION : &str = "
  _____           _    ____  _   _ _   ___   __
 |  __ \\         | |  / __ \\| \\ | | \\ | \\ \\ / /
 | |__) |   _ ___| |_| |  | |  \\| |  \\| |\\ V /
 |  _  / | | / __| __| |  | | . ` | . ` | > <
 | | \\ \\ |_| \\__ \\ |_| |__| | |\\  | |\\  |/ . \\
 |_|  \\_\\__,_|___/\\__|\\____/|_| \\_|_| \\_/_/ \\_\\
 ----------------------------------------------------------------------------------
    - ⚙️ Rust-based ONNX inference engine.
    - 🥷 Under the hood .onnx and .pb parser.
    - 📦 Delivered with a set of validated operators, models and datasets.
    - ➕ Easy to add new operators to extend model compatibility.
    - 🖼️ Images & labels serialization to extend the set of available datasets.
    - 🚀 Rayon-powered image-based and intra-network parallelization.
    - 🐍 Python bindings for a complete Rusty experience.
 ----------------------------------------------------------------------------------\n";

    println!(
        "{}",
        PROGRAM_NAME_AND_DESCRIPTION
    );
}

pub fn menu() -> (Model, bool, bool, String, PathBuf) {
    if let Err(_) = clearscreen::clear(){
        eprintln!("Error clearing the screen.")
    }
    print_intro();

    let mut options = Vec::new();
    for model_name in MODELS_NAMES{
        options.push("Run ".to_string() + model_name + " in inference mode");
    }
    options.push("Exit".to_string());

    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("What do you want to do?")
        .items(&options)
        .default(0)
        .interact()
        .unwrap();

    if selection == options.len()-1{
        println!("Exiting...");
        process::exit(0);
    }

    println!();

    let chosen_model = Model::from_index(selection).unwrap();

    let test_dataset = match Select::with_theme(&ColorfulTheme::default())
        .with_prompt("You now have two options. You can run ".to_string() + MODELS_NAMES[selection] + " on:\n1) \
        the test dataset provided by ONNX, with an image and a label already serialized into .pb files;\n\
        2) your custom dataset with images and labels to be serialized into .pb files.\n\
        Please select an option.")
        .items(&["Run on the test dataset", "Run on my custom dataset", "Go back to the main menu"])
        .default(0)
        .interact()
        .unwrap()
    {
        0 => true,
        1 => false,
        2 => {
            if let Err(_) = clearscreen::clear(){
                eprintln!("Error clearing the screen.")
            }
            return menu();
        }
        _ => false,
    };

    println!();

    let mut folder_name = String::new();

    if !test_dataset{
        folder_name = Input::with_theme(&ColorfulTheme::default())
            .with_prompt("Provide the folder name for your custom dataset.\n\
                Be sure to respect the following constraints:\n\
                1) the dataset folder must be placed under the folder of the model you want to run (e.g. \"mnist-12/my-dataset/\");\n\
                2) the dataset folder must include subfolders whose names match the label, in numeric format, of the images they contain;\n\
                3) at least one subfolder that follows the naming convention mentioned above must reside within the dataset folder;\n\
                4) accepted image formats are .jpg, .jpeg or .png;\n\
                5) all subfolders following the expected naming convention (i.e. label in numeric format) must include at least a .jpg, .jpeg or .png file.\n\
                For example, you may have a \"my-dataset/\" folder under \"resnet18-v1-7/\" with a \"207/\" subfolder that includes \
                a .jpg, .jpeg or .png image of a golden retriever, since 207 is the label for a golden retriever in the ImageNet dataset.\
            \n(type 'BACK' to go back to the main menu)")
            .interact()
            .unwrap();
        folder_name = folder_name.trim().to_string();
        if folder_name.to_uppercase() == "BACK" {
            if let Err(_) = clearscreen::clear(){
                eprintln!("Error clearing the screen.")
            }
            return menu();
        }
        println!();

        //Perform checks on the provided folder and provide ways to go back to the main menu in case of a wrong input

        let model_path = PathBuf::from("models").join(chosen_model.as_str());
        let dataset_path = model_path.join(format!("{}", folder_name));

        if dataset_path.exists() && dataset_path.is_dir(){
            let (subfolders_with_numerical_name_counter, subfolders_with_no_jpg_jpeg_or_png_images_counter)=
                check_dataset_subfolders_and_files(&dataset_path);

            if subfolders_with_numerical_name_counter==0{
                return select_to_go_back_to_main_menu("The folder \"".to_string() +  dataset_path.to_str().unwrap() +
                    "\\\" exists but does not include any subfolder respecting the numerical naming convention.");
            }
            if subfolders_with_no_jpg_jpeg_or_png_images_counter !=0{
                return select_to_go_back_to_main_menu("Between the subfolders respecting the numerical naming convention, there \
                    is at least a subfolder that does not include any .jpg, .jpeg or .png file.".to_string());
            }

        }
        else{
            return select_to_go_back_to_main_menu("There is no folder with this name under \"".to_string() +
                model_path.to_str().unwrap() + "\".");
        }
    }

    let verbose = match Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Run ".to_string() + MODELS_NAMES[selection] + " in verbose mode?")
        .items(&["Yes", "No", "Go back to the main menu"])
        .default(0)
        .interact()
        .unwrap()
    {
        0 => true,
        1 => false,
        2 => {
            if let Err(_) = clearscreen::clear(){
                eprintln!("Error clearing the screen.")
            }
            return menu();
        }
        _ => false,
    };
    println!();
    (chosen_model.clone(), verbose, test_dataset, folder_name, PathBuf::from("models").join(chosen_model.as_str()).join("model.onnx"))
}

fn check_dataset_subfolders_and_files(dataset_path: &PathBuf) -> (i32, i32) {
    let mut subfolders_with_numerical_name_counter = 0;
    let mut subfolders_with_no_jpg_jpeg_or_png_images_counter = 0;
    for entry in fs::read_dir(&dataset_path).unwrap(){
        let path = entry.unwrap().path();
        if path.is_dir(){
            let dir_name = path.file_name().unwrap().to_str().unwrap();
            if !dir_name.parse::<i32>().is_err(){
                subfolders_with_numerical_name_counter+=1;
                let mut jpg_jpeg_png_flag = false;
                for sub_entry in fs::read_dir(path).unwrap(){
                    let sub_path = sub_entry.unwrap().path();
                    if let Some(ext) = sub_path.extension().and_then(|e| e.to_str()) {
                        if SUPPORTED_IMAGE_FORMATS.contains(&ext.to_uppercase().as_str()) {
                            jpg_jpeg_png_flag = true;
                        }
                    }
                }
                if jpg_jpeg_png_flag ==false{
                    subfolders_with_no_jpg_jpeg_or_png_images_counter +=1;
                }
            }
        }
    }
    return (subfolders_with_numerical_name_counter, subfolders_with_no_jpg_jpeg_or_png_images_counter)
}

fn select_to_go_back_to_main_menu(prompt: String) -> (Model, bool, bool, String, PathBuf) {
    match Select::with_theme(&ColorfulTheme::default())
        .with_prompt(prompt)
        .items(&["Go back to the main menu"])
        .default(0)
        .interact()
        .unwrap()
    {
        _ => {
            if let Err(_) = clearscreen::clear(){
                eprintln!("Error clearing the screen.")
            }
            return menu();}
    };
}