#![no_main]
use risc0_zkvm::guest::env;

struct NeuralNetwork {
    layers: Vec<Layer>,
}

struct Layer {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
}

impl NeuralNetwork {
    fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(Layer {
                weights: vec![vec![0.1; layer_sizes[i]]; layer_sizes[i + 1]],
                biases: vec![0.1; layer_sizes[i + 1]],
            });
        }
        NeuralNetwork { layers }
    }

    fn forward(&self, inputs: &[f32]) -> Vec<f32> {
        let mut current = inputs.to_vec();
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        current
    }

    fn train(&mut self, inputs: &[f32], target: f32, learning_rate: f32) {
        let mut activations = vec![inputs.to_vec()];
        let mut weighted_inputs = Vec::new();

        // Forward pass
        for layer in &self.layers {
            let (z, a) = layer.forward_with_cache(&activations.last().unwrap());
            weighted_inputs.push(z);
            activations.push(a);
        }

        // Backward pass
        let mut deltas = Vec::with_capacity(self.layers.len());
        
        // Calculate initial delta for the output layer
        let mut delta = vec![0.0; self.layers.last().unwrap().biases.len()];
        for j in 0..delta.len() {
            let z = weighted_inputs.last().unwrap()[j];
            delta[j] = (activations.last().unwrap()[j] - target) * relu_derivative(z);
        }
        deltas.push(delta);

        // Calculate deltas for hidden layers
        for l in (0..self.layers.len() - 1).rev() {
            let next_delta = deltas.last().unwrap();
            let layer = &self.layers[l + 1];
            let mut delta = vec![0.0; self.layers[l].biases.len()];
            for i in 0..delta.len() {
                delta[i] = layer.weights.iter()
                    .zip(next_delta.iter())
                    .map(|(w, &d)| w[i] * d)
                    .sum::<f32>()
                    * relu_derivative(weighted_inputs[l][i]);
            }
            deltas.push(delta);
        }
        deltas.reverse();

        // Update weights and biases
        for (l, layer) in self.layers.iter_mut().enumerate() {
            layer.update_params(&activations[l], &deltas[l], learning_rate);
        }
    }
}

impl Layer {
    fn forward(&self, inputs: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.biases.len()];
        for (j, out) in output.iter_mut().enumerate() {
            *out = self.biases[j] + self.weights[j].iter().zip(inputs).map(|(&w, &x)| w * x).sum::<f32>();
            *out = out.max(0.0); // ReLU activation
        }
        output
    }

    fn forward_with_cache(&self, inputs: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut z = vec![0.0; self.biases.len()];
        let mut a = vec![0.0; self.biases.len()];
        for j in 0..self.biases.len() {
            z[j] = self.biases[j] + self.weights[j].iter().zip(inputs).map(|(&w, &x)| w * x).sum::<f32>();
            a[j] = z[j].max(0.0); // ReLU activation
        }
        (z, a)
    }

    fn update_params(&mut self, inputs: &[f32], delta: &[f32], learning_rate: f32) {
        for j in 0..self.biases.len() {
            self.biases[j] -= learning_rate * delta[j];
            for (i, wi) in self.weights[j].iter_mut().enumerate() {
                *wi -= learning_rate * delta[j] * inputs[i];
            }
        }
    }
}

#[inline(always)]
fn relu_derivative(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn main() {
    println!("Starting the neural network training and testing process...");

    println!("Reading input data...");
    let customer_data: Vec<(Vec<f32>, f32)> = env::read();
    let epochs: usize = env::read();
    let learning_rate: f32 = env::read();
    let test_inputs: Vec<Vec<f32>> = env::read();

    println!("Input data read successfully.");
    println!("Number of training samples: {}", customer_data.len());
    println!("Number of epochs: {}", epochs);
    println!("Learning rate: {}", learning_rate);
    println!("Number of test inputs: {}", test_inputs.len());

    let input_size = customer_data[0].0.len();
    println!("Initializing neural network with input size: {}", input_size);
    let mut nn = NeuralNetwork::new(&[input_size, 32, 16, 1]);

    println!("Initial network parameters:");
    print_network_params(&nn);

    println!("\nStarting training...");
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        for (inputs, target) in &customer_data {
            nn.train(inputs, *target, learning_rate);
            let prediction = nn.forward(inputs)[0];
            let loss = (prediction - target).powi(2);
            total_loss += loss;
        }
        let avg_loss = total_loss / customer_data.len() as f32;
        
        if epoch % 10 == 0 || epoch == epochs - 1 {
            println!("Epoch {}: Average Loss = {:.6}", epoch, avg_loss);
        }
    }

    println!("\nTraining completed.");
    println!("Final network parameters:");
    print_network_params(&nn);

    println!("\nTesting the network:");
    let mut predictions = Vec::new();
    for input in &test_inputs {
        match nn.forward(input).get(0) {
            Some(&prediction) => {
                println!("Input: {:?}, Prediction: {:.4}", input, prediction);
                predictions.push(prediction);
            },
            None => {
                println!("Error: Failed to get prediction for input: {:?}", input);
                predictions.push(f32::NAN); // Push NaN for failed predictions
            }
        }
    }

    println!("\nCommitting results...");
    let weights_and_biases: Vec<(Vec<Vec<f32>>, Vec<f32>)> = nn.layers
        .iter()
        .map(|layer| (layer.weights.clone(), layer.biases.clone()))
        .collect();
    env::commit(&(weights_and_biases, predictions));

    println!("Process completed successfully.");
}

fn print_network_params(nn: &NeuralNetwork) {
    for (i, layer) in nn.layers.iter().enumerate() {
        println!("Layer {}:", i + 1);
        println!("  Weights:");
        for (j, weights) in layer.weights.iter().enumerate() {
            println!("    Neuron {}: {:?}", j, weights);
        }
        println!("  Biases: {:?}", layer.biases);
    }
}

risc0_zkvm::guest::entry!(main);
