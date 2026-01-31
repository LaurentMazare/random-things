//! Basic CUDA example to test tensor operations on GPU.
//!
//! Run with: cargo run --release --example basic_cuda --features cuda

use mimi::{Result, Tensor, cuda_backend::Device};

fn main() -> Result<()> {
    println!("Initializing CUDA device...");
    let device = Device::new(0)?;
    println!("CUDA device initialized successfully!");

    // Test from_vec - create a tensor from host data
    println!("\nTesting from_vec...");
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let t: Tensor<f32, Device> = Tensor::from_vec(data.clone(), vec![5], &device)?;
    println!("Created tensor with shape: {:?}", t.dims());

    // Test data() - copy back to host
    println!("\nTesting data() - copying back to host...");
    let host_data = t.to_vec()?;
    println!("Host data: {:?}", host_data);
    assert_eq!(host_data, data);
    println!("Data matches!");

    // Test zeros - create a tensor filled with zeros
    println!("\nTesting zeros...");
    let zeros: Tensor<f32, Device> = Tensor::zeros(vec![3, 4], &device)?;
    println!("Created zeros tensor with shape: {:?}", zeros.dims());
    let zeros_data = zeros.to_vec()?;
    println!("Zeros data: {:?}", zeros_data);
    assert!(zeros_data.iter().all(|&x| x == 0.0));
    println!("All zeros!");

    // Test full - create a tensor filled with a specific value
    println!("\nTesting full...");
    let full: Tensor<f32, Device> = Tensor::full(42.0, vec![2, 3], &device)?;
    println!("Created full tensor with shape: {:?}", full.dims());
    let full_data = full.to_vec()?;
    println!("Full data: {:?}", full_data);
    assert!(full_data.iter().all(|&x| x == 42.0));
    println!("All 42s!");

    // Test with f16
    println!("\nTesting with f16...");
    let f16_data: Vec<half::f16> =
        vec![1.0, 2.0, 3.0].into_iter().map(half::f16::from_f32).collect();
    let t_f16: Tensor<half::f16, Device> = Tensor::from_vec(f16_data.clone(), vec![3], &device)?;
    let f16_back = t_f16.to_vec()?;
    println!("F16 data roundtrip successful: {:?}", f16_back);

    // Test with bf16
    println!("\nTesting with bf16...");
    let bf16_data: Vec<half::bf16> =
        vec![1.0, 2.0, 3.0].into_iter().map(half::bf16::from_f32).collect();
    let t_bf16: Tensor<half::bf16, Device> = Tensor::from_vec(bf16_data.clone(), vec![3], &device)?;
    let bf16_back = t_bf16.to_vec()?;
    println!("BF16 data roundtrip successful: {:?}", bf16_back);

    println!("\nAll tests passed!");
    Ok(())
}
