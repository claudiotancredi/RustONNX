use std::collections::{HashMap, HashSet};
use std::time::Instant;
use colored::Colorize;
use indicatif::{MultiProgress, ParallelProgressIterator};
use ndarray::{ArrayBase, ArrayD, IxDyn, OwnedRepr};
use petgraph::Direction;
use petgraph::graph::DiGraph;
use rayon::prelude::*;
use crate::operators::op_operator::Operator;
use crate::utils::auxiliary_functions::{display_model_info, setup_progress_bar};
use crate::utils::errors::OnnxError;

pub fn run(images_vec: &Vec<ArrayBase<OwnedRepr<f32>, IxDyn>>,
           model_read: &HashMap<String, Box<dyn Operator>>,
           input_name: &String,
           verbose: &bool,
           final_layer_name: &String,
           chosen_model: &str) -> ArrayBase<OwnedRepr<f32>, IxDyn> {

    println!("\n🔍 Optimizing the model for intra-network parallel execution...");

    let model_graph = create_graph(model_read);
    let parallel_layers = generate_parallel_layers(&model_graph);

    let number_of_nodes_in_parallel = parallel_layers.iter().map(|v|v.len()).
                                                            filter(|v| *v>= 2).map(|v|v-1).sum::<usize>();

    println!("✅ Model successfully optimized!");

    display_model_info(chosen_model, model_read.values().len(), number_of_nodes_in_parallel);

    let multi_progress = MultiProgress::new();

    let progress_bar_images = setup_progress_bar(&multi_progress, images_vec.len() as u64, true, "images");
    let progress_bar_nodes = setup_progress_bar(&multi_progress,
                                model_read.values().len() as u64 *images_vec.len() as u64, false, "executed nodes");

    progress_bar_images.set_position(0);
    progress_bar_nodes.set_position(0);

    let network_timer = Instant::now();

    let model_final_output = images_vec.par_iter().enumerate()
        .map(|(index, img)| {

            let mut inputs = HashMap::new();
            inputs.insert(input_name.clone(), img.clone());

            //Iterate over the layers
            for layer in parallel_layers.clone(){
                let number_of_nodes_in_layer = layer.len();
                if number_of_nodes_in_layer ==1{ //No parallel execution
                    //Execute, update progress bar and print info
                    let output = model_read.get(&layer[0]).unwrap().execute(&inputs)
                        .unwrap();
                    progress_bar_nodes.inc(1);
                    if *verbose {
                        progress_bar_nodes.suspend(||{
                            println!("{}", model_read.get(&layer[0]).unwrap().to_string(&inputs, &output, index.to_string()));
                        });
                    }
                    else{
                        progress_bar_nodes.suspend(||{
                            println!("{}", format!("🚀 Executed node: {} {} for image: {}", model_read.get(&layer[0]).unwrap().get_op_type().bold(),
                                                   model_read.get(&layer[0]).unwrap().get_node_name().bold(), index.to_string().bold()));
                        });
                    }
                    //Insert the output in the inputs for the next nodes to be run
                    for (i, out) in output.iter().enumerate() {
                        inputs.insert(model_read.get(&layer[0]).unwrap().get_output_names()[i].clone(), out.to_owned());
                    }

                }
                else{
                    //Parallel execution of the nodes in the same layer
                    let layers_outputs = layer.par_iter().enumerate().map( |(k,_)|{
                        let mut inner_inputs = inputs.clone();
                        let output = model_read.get(&layer[k]).unwrap().execute(&inputs)
                            .unwrap();
                        progress_bar_nodes.inc(1);
                        if *verbose {
                            progress_bar_nodes.suspend(||{
                                println!("{}", model_read.get(&layer[k]).unwrap().to_string(&inner_inputs, &output, index.to_string()));
                            });
                        }
                        else{
                            progress_bar_nodes.suspend(||{
                                println!("{}", format!("🚀 Executed node: {} {} for image: {}", model_read.get(&layer[k]).unwrap().get_op_type().bold(),
                                                       model_read.get(&layer[k]).unwrap().get_node_name().bold(), index.to_string().bold()));
                            });
                        }
                        for (i, out) in output.iter().enumerate() {
                            inner_inputs.insert(model_read.get(&layer[k]).unwrap().get_output_names()[i].clone(), out.to_owned());
                        }

                        inner_inputs
                    }).collect::<Vec<HashMap<String, ArrayD<f32>>>>();

                    //After collecting the results, put them in the inputs hashmap
                    for hash in layers_outputs {
                        for (key, value) in hash {
                            inputs.insert(key, value);
                        }
                    }
                }
            }

            // Return the output for this particular input
            inputs.get(final_layer_name)
                .ok_or_else(||OnnxError::RuntimeError(format!("Network final output not found for image {}.", index)))
                .unwrap().clone()
        })
        .progress_with(progress_bar_images)
        .collect::<Vec<ArrayD<f32>>>();

    let shape = model_final_output[0].shape();
    let batch_size= model_final_output.len();
    let c = shape[1];
    let new_s = vec![batch_size, c ];

    let flat_vec: Vec<f32> = model_final_output.into_iter()
        .flat_map(|array| array.into_raw_vec())
        .collect();

    let run_time = network_timer.elapsed();
    println!("\n\n✅  The network has been successfully executed in {:?}\n", run_time);

    ArrayD::from_shape_vec(IxDyn(&new_s), flat_vec).map_err(|_|OnnxError::ShapeError("Failed to create output tensor from expected shape.".to_string()))
        .unwrap()
}

fn create_graph(operators: &HashMap<String, Box<dyn Operator>>) -> DiGraph<String, ()> {
    let mut graph = DiGraph::new();
    let mut output_to_node = HashMap::new();

    for operator in operators.values() {
        let node_name = operator.get_node_name();
        let outputs = operator.get_output_names();

        // Insert the node into the graph
        graph.add_node(node_name.clone());

        // Map each output to its node
        for output in outputs {
            output_to_node.insert(output, node_name.clone());
        }
    }

    // Add edges based on matching output and input names
    for operator in operators.values() {
        let node_name = operator.get_node_name();
        for input in operator.get_inputs() {
            if let Some(outputting_node_name) = output_to_node.get(&input) {
                let source_index = graph.node_indices()
                    .find(|&n| graph[n] == *outputting_node_name)
                    .expect("Node index not found");
                let target_index = graph.node_indices()
                    .find(|&n| graph[n] == *node_name)
                    .expect("Node index not found");

                // Create an edge from the source to the target node
                graph.add_edge(source_index, target_index, ());
            }
        }
    }

    graph
}

// Function to generate parallel layers from the graph
fn generate_parallel_layers(graph: &DiGraph<String, ()>) -> Vec<Vec<String>> {
    let mut layers: Vec<Vec<String>> = Vec::new();
    let mut in_degrees = HashMap::new();
    let mut to_visit = HashSet::new();

    for node_index in graph.node_indices() {
        // Compute the in_degree for the current node
        let in_degree = graph.neighbors_directed(node_index, Direction::Incoming).count();
        //Insert the in_degree into the hashmap for the current node
        in_degrees.insert(node_index, in_degree);

        // If in_degree is zero the node has not been visited yet, so insert it into the hashset
        if in_degree == 0 {
            to_visit.insert(node_index);
        }
    }

    while !to_visit.is_empty() {//Loop until there aren't any other nodes
        let mut current_layer = Vec::new();
        let mut next_to_visit = HashSet::new();

        for node_index in to_visit.drain() {//Drain clears the set returning an iterator with the elements
            // Start accumulating the nodes for the current layer
            current_layer.push(graph[node_index].clone());

            for neighbor in graph.neighbors(node_index) {
                //Check neighbors of the current node. Get the in_degree of the neighbor...
                let degree = in_degrees.get_mut(&neighbor).unwrap();
                //... and decrement it because some nodes have been visited/removed
                *degree -= 1;

                // If it becomes zero, then update the collection of nodes to visit
                if *degree == 0 {
                    next_to_visit.insert(neighbor);
                }
            }
        }

        // Update the layers
        layers.push(current_layer);

        // Update the nodes to be visited
        to_visit = next_to_visit;
    }

    layers
}