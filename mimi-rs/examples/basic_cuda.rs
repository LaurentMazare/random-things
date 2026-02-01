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

    // Test transpose
    println!("\nTesting transpose...");
    let t_2x3: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &device)?;
    let t_3x2 = t_2x3.transpose(0, 1)?;
    println!("Original shape: {:?}, Transposed shape: {:?}", t_2x3.dims(), t_3x2.dims());
    assert_eq!(t_3x2.dims(), &[3, 2]);
    let t_3x2_data = t_3x2.to_vec()?;
    // Original: [[1, 2, 3], [4, 5, 6]]
    // Transposed: [[1, 4], [2, 5], [3, 6]]
    assert_eq!(t_3x2_data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    println!("Transpose correct!");

    // Test softmax
    println!("\nTesting softmax...");
    let logits: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], vec![2, 3], &device)?;
    let probs = logits.softmax()?;
    let probs_data = probs.to_vec()?;
    println!("Softmax output: {:?}", probs_data);
    // Each row should sum to 1.0
    let row1_sum: f32 = probs_data[0..3].iter().sum();
    let row2_sum: f32 = probs_data[3..6].iter().sum();
    println!("Row sums: {}, {}", row1_sum, row2_sum);
    assert!((row1_sum - 1.0).abs() < 1e-5);
    assert!((row2_sum - 1.0).abs() < 1e-5);
    // softmax([1,2,3]) = [e^1, e^2, e^3] / sum ≈ [0.090, 0.245, 0.665]
    assert!((probs_data[0] - 0.0900306).abs() < 1e-4);
    assert!((probs_data[1] - 0.2447285).abs() < 1e-4);
    assert!((probs_data[2] - 0.6652409).abs() < 1e-4);
    println!("Softmax correct!");

    // Test RMS norm
    println!("\nTesting RMS norm...");
    let x_rms: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &device)?;
    let alpha: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3], &device)?;
    let rms_out = x_rms.rms_norm(&alpha, 1e-5)?;
    let rms_data = rms_out.to_vec()?;
    println!("RMS norm output: {:?}", rms_data);
    // RMS of [1,2,3] = sqrt((1+4+9)/3) = sqrt(14/3) ≈ 2.16
    // Normalized: [1/2.16, 2/2.16, 3/2.16] ≈ [0.46, 0.93, 1.39]
    let rms_row1 = (1.0f32 + 4.0 + 9.0) / 3.0;
    let scale1 = 1.0 / (rms_row1 + 1e-5).sqrt();
    assert!((rms_data[0] - 1.0 * scale1).abs() < 1e-4);
    assert!((rms_data[1] - 2.0 * scale1).abs() < 1e-4);
    assert!((rms_data[2] - 3.0 * scale1).abs() < 1e-4);
    println!("RMS norm correct!");

    // Test layer norm
    println!("\nTesting layer norm...");
    let x_ln: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &device)?;
    let weight: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3], &device)?;
    let bias: Tensor<f32, Device> = Tensor::from_vec(vec![0.0, 0.0, 0.0], vec![3], &device)?;
    let ln_out = x_ln.layer_norm(&weight, &bias, 1e-5)?;
    let ln_data = ln_out.to_vec()?;
    println!("Layer norm output: {:?}", ln_data);
    // For [1,2,3]: mean=2, var=2/3, std≈0.816
    // Normalized: [-1.22, 0, 1.22] approximately
    let mean1 = 2.0f32;
    let var1 = ((1.0 - mean1).powi(2) + (2.0 - mean1).powi(2) + (3.0 - mean1).powi(2)) / 3.0;
    let inv_std1 = 1.0 / (var1 + 1e-5).sqrt();
    assert!((ln_data[0] - (1.0 - mean1) * inv_std1).abs() < 1e-4);
    assert!((ln_data[1] - (2.0 - mean1) * inv_std1).abs() < 1e-4);
    assert!((ln_data[2] - (3.0 - mean1) * inv_std1).abs() < 1e-4);
    println!("Layer norm correct!");

    println!("\nAll tests passed!");
    Ok(())
}
