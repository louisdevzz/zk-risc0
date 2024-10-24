// These constants represent the RISC-V ELF and the image ID generated by risc0-build.
// The ELF is used for proving and the ID is used for verification.
use methods::{
    GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID
};
use risc0_zkvm::{default_prover, ExecutorEnv};
use serde::{Serialize, Deserialize};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

#[derive(Serialize, Deserialize)]
struct TrainingData {
    customer_data: Vec<(Vec<f32>, f32)>,
    epochs: usize,
    learning_rate: f32,
    test_inputs: Vec<Vec<f32>>,
}

fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    // Prepare the training data
    let training_data = TrainingData {
        customer_data: vec![
            (vec![1.0, 2.0, 3.0], 0.3),
            (vec![2.0, 3.0, 4.0], 0.5),
            (vec![3.0, 4.0, 5.0], 0.7),
            (vec![4.0, 5.0, 6.0], 0.9),
            (vec![5.0, 6.0, 7.0], 1.1),
            (vec![1.5, 2.5, 3.5], 0.4),
            (vec![2.5, 3.5, 4.5], 0.6),
            (vec![3.5, 4.5, 5.5], 0.8),
            (vec![4.5, 5.5, 6.5], 1.0),
            (vec![0.5, 1.5, 2.5], 0.2),
        ],
        epochs: 5000,
        learning_rate: 0.001,
        test_inputs: vec![
            vec![1.2, 2.2, 3.2],
            vec![2.7, 3.7, 4.7],
            vec![3.8, 4.8, 5.8],
            vec![0.8, 1.8, 2.8],
            vec![4.2, 5.2, 6.2],
        ],
    };

    // Create the ExecutorEnv and write the training data
    let env = ExecutorEnv::builder()
        .write(&training_data.customer_data).unwrap()
        .write(&training_data.epochs).unwrap()
        .write(&training_data.learning_rate).unwrap()
        .write(&training_data.test_inputs).unwrap()
        .build()
        .unwrap();

    // Obtain the default prover and prove the guest code
    let prover = default_prover();
    let prove_info = prover
        .prove(env, GUEST_CODE_FOR_ZK_PROOF_ELF)
        .unwrap();

    // Extract the receipt
    let receipt = prove_info.receipt;

    // Retrieve the trained model and predictions
    let (weights_and_biases, predictions): (Vec<(Vec<Vec<f32>>, Vec<f32>)>, Vec<f32>) = receipt.journal.decode().unwrap();

    // Print the journal contents
    println!("Guest output:");
    println!("{}", std::str::from_utf8(receipt.journal.bytes.as_slice()).unwrap());

    // Print the results
    println!("Trained model:");
    for (i, (weights, biases)) in weights_and_biases.iter().enumerate() {
        println!("Layer {}:", i + 1);
        println!("  Weights: {:?}", weights);
        println!("  Biases: {:?}", biases);
    }
    println!("Predictions for test inputs:");
    for (input, prediction) in training_data.test_inputs.iter().zip(predictions.iter()) {
        println!("  Input: {:?}, Prediction: {:.4}", input, prediction);
    }

    // Verify the receipt
    receipt.verify(GUEST_CODE_FOR_ZK_PROOF_ID).unwrap();
    println!("Proof verified successfully!");
}
