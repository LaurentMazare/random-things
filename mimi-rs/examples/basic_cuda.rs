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

    // Test binary operations
    println!("\nTesting binary operations...");
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4], &device)?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![4], &device)?;

    // Addition
    let sum = a.add(&b)?;
    let sum_data = sum.to_vec()?;
    println!("a + b = {:?}", sum_data);
    assert_eq!(sum_data, vec![6.0, 8.0, 10.0, 12.0]);

    // Subtraction
    let diff = b.sub(&a)?;
    let diff_data = diff.to_vec()?;
    println!("b - a = {:?}", diff_data);
    assert_eq!(diff_data, vec![4.0, 4.0, 4.0, 4.0]);

    // Multiplication
    let prod = a.mul(&b)?;
    let prod_data = prod.to_vec()?;
    println!("a * b = {:?}", prod_data);
    assert_eq!(prod_data, vec![5.0, 12.0, 21.0, 32.0]);

    // Division
    let quot = b.div(&a)?;
    let quot_data = quot.to_vec()?;
    println!("b / a = {:?}", quot_data);
    assert_eq!(quot_data, vec![5.0, 3.0, 7.0 / 3.0, 2.0]);

    // Scale
    let scaled = a.scale(2.0)?;
    let scaled_data = scaled.to_vec()?;
    println!("a * 2 = {:?}", scaled_data);
    assert_eq!(scaled_data, vec![2.0, 4.0, 6.0, 8.0]);

    // Maximum
    let c: Tensor<f32, Device> = Tensor::from_vec(vec![3.0, 1.0, 4.0, 2.0], vec![4], &device)?;
    let max_ab = a.maximum(&c)?;
    let max_data = max_ab.to_vec()?;
    println!("max(a, c) = {:?}", max_data);
    assert_eq!(max_data, vec![3.0, 2.0, 4.0, 4.0]);

    // Minimum
    let min_ab = a.minimum(&c)?;
    let min_data = min_ab.to_vec()?;
    println!("min(a, c) = {:?}", min_data);
    assert_eq!(min_data, vec![1.0, 1.0, 3.0, 2.0]);

    // Test unary operations
    println!("\nTesting unary operations...");
    let x: Tensor<f32, Device> = Tensor::from_vec(vec![0.0, 1.0, -1.0, 2.0], vec![4], &device)?;

    // ReLU
    let relu_x = x.relu()?;
    let relu_data = relu_x.to_vec()?;
    println!("relu(x) = {:?}", relu_data);
    assert_eq!(relu_data, vec![0.0, 1.0, 0.0, 2.0]);

    // SiLU (x * sigmoid(x))
    let silu_x = x.silu()?;
    let silu_data = silu_x.to_vec()?;
    println!("silu(x) = {:?}", silu_data);
    // silu(0) = 0, silu(1) ≈ 0.731, silu(-1) ≈ -0.269, silu(2) ≈ 1.762
    assert!((silu_data[0] - 0.0).abs() < 1e-5);
    assert!((silu_data[1] - 0.7310586).abs() < 1e-5);

    // Sqrt
    let y: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 4.0, 9.0, 16.0], vec![4], &device)?;
    let sqrt_y = y.sqrt()?;
    let sqrt_data = sqrt_y.to_vec()?;
    println!("sqrt(y) = {:?}", sqrt_data);
    assert_eq!(sqrt_data, vec![1.0, 2.0, 3.0, 4.0]);

    // Sqr
    let sqr_y = y.sqr()?;
    let sqr_data = sqr_y.to_vec()?;
    println!("sqr(y) = {:?}", sqr_data);
    assert_eq!(sqr_data, vec![1.0, 16.0, 81.0, 256.0]);

    // Test matrix multiplication (GEMM)
    println!("\nTesting matrix multiplication (GEMM)...");

    // Simple 2x3 @ 3x2 = 2x2 matmul
    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    // B = [[1, 2],
    //      [3, 4],
    //      [5, 6]]
    // C = A @ B = [[1*1+2*3+3*5, 1*2+2*4+3*6],
    //              [4*1+5*3+6*5, 4*2+5*4+6*6]]
    //           = [[22, 28],
    //              [49, 64]]
    let mat_a: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &device)?;
    let mat_b: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], &device)?;
    let mat_c = mat_a.matmul(&mat_b)?;
    let mat_c_data = mat_c.to_vec()?;
    println!("A (2x3) @ B (3x2) = {:?}", mat_c_data);
    println!("Result shape: {:?}", mat_c.dims());
    assert_eq!(mat_c.dims(), &[2, 2]);
    assert_eq!(mat_c_data, vec![22.0, 28.0, 49.0, 64.0]);

    // Test with f16
    println!("\nTesting f16 matmul...");
    let mat_a_f16: Tensor<half::f16, Device> = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter().map(half::f16::from_f32).collect(),
        vec![2, 3],
        &device,
    )?;
    let mat_b_f16: Tensor<half::f16, Device> = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter().map(half::f16::from_f32).collect(),
        vec![3, 2],
        &device,
    )?;
    let mat_c_f16 = mat_a_f16.matmul(&mat_b_f16)?;
    let mat_c_f16_data: Vec<f32> = mat_c_f16.to_vec()?.iter().map(|x| x.to_f32()).collect();
    println!("F16 matmul result: {:?}", mat_c_f16_data);
    assert!((mat_c_f16_data[0] - 22.0).abs() < 0.1);
    assert!((mat_c_f16_data[1] - 28.0).abs() < 0.1);
    assert!((mat_c_f16_data[2] - 49.0).abs() < 0.1);
    assert!((mat_c_f16_data[3] - 64.0).abs() < 0.1);

    // Test batched matmul
    println!("\nTesting batched matmul...");
    // Batch of 2: each batch is 2x3 @ 3x2
    let batch_a: Tensor<f32, Device> = Tensor::from_vec(
        vec![
            // Batch 0
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 1 (all ones)
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        vec![2, 2, 3],
        &device,
    )?;
    let batch_b: Tensor<f32, Device> = Tensor::from_vec(
        vec![
            // Batch 0
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 1 (all ones)
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        vec![2, 3, 2],
        &device,
    )?;
    let batch_c = batch_a.matmul(&batch_b)?;
    let batch_c_data = batch_c.to_vec()?;
    println!("Batched matmul result: {:?}", batch_c_data);
    println!("Result shape: {:?}", batch_c.dims());
    assert_eq!(batch_c.dims(), &[2, 2, 2]);
    // Batch 0: same as before [22, 28, 49, 64]
    // Batch 1: all ones @ all ones = [[3, 3], [3, 3]]
    assert_eq!(batch_c_data[0], 22.0);
    assert_eq!(batch_c_data[4], 3.0); // First element of batch 1

    println!("\nAll tests passed!");
    Ok(())
}
