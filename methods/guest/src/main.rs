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
        let mut delta = vec![0.0; self.layers.last().unwrap().biases.len()];
        for j in 0..delta.len() {
            let z = weighted_inputs.last().unwrap()[j];
            delta[j] = (activations.last().unwrap()[j] - target) * relu_derivative(z);
        }

        for (l, layer) in self.layers.iter_mut().enumerate().rev() {
            layer.update_params(&activations[l], &delta, learning_rate);
            if l > 0 {
                delta = layer.backward(&delta, &weighted_inputs[l - 1]);
            }
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

    fn backward(&self, next_delta: &[f32], weighted_inputs: &[f32]) -> Vec<f32> {
        let mut delta = vec![0.0; self.weights[0].len()];
        for (i, di) in delta.iter_mut().enumerate() {
            *di = self.weights.iter().zip(next_delta).map(|(w, &d)| w[i] * d).sum::<f32>() * relu_derivative(weighted_inputs[i]);
        }
        delta
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
    let mut nn = NeuralNetwork::new(&[input_size, 16, 1]); // Reduced network size

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

    println!("\nTraining completed. Starting prediction on test inputs...");
    let predictions: Vec<f32> = test_inputs.iter()
        .map(|input| {
            let pred = nn.forward(input)[0];
            println!("Input: {:?}, Prediction: {:.4}", input, pred);
            pred
        })
        .collect();

    println!("\nCalculating summary statistics...");
    let summary: Vec<(f32, f32)> = nn.layers
        .iter()
        .map(|layer| {
            let avg_weight = layer.weights.iter()
                .flat_map(|w| w.iter())
                .sum::<f32>() / (layer.weights.len() * layer.weights[0].len()) as f32;
            let avg_bias = layer.biases.iter().sum::<f32>() / layer.biases.len() as f32;
            (avg_weight, avg_bias)
        })
        .collect();

    let avg_prediction = predictions.iter().sum::<f32>() / predictions.len() as f32;

    println!("\nCommitting results...");
    env::commit(&(summary, avg_prediction));

    println!("Guest program completed successfully.");
}

risc0_zkvm::guest::entry!(main);
