#![cfg(feature = "cuda")]

use mimi::{Result, Tensor, cuda_backend::Device};

fn get_device() -> Device {
    Device::new(0).expect("Failed to initialize CUDA device")
}

// =============================================================================
// Basic tensor operations
// =============================================================================

#[test]
fn test_from_vec_and_to_vec() -> Result<()> {
    let device = get_device();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let t: Tensor<f32, Device> = Tensor::from_vec(data.clone(), vec![5], &device)?;
    assert_eq!(t.dims(), &[5]);
    assert_eq!(t.to_vec()?, data);
    Ok(())
}

#[test]
fn test_zeros() -> Result<()> {
    let device = get_device();
    let zeros: Tensor<f32, Device> = Tensor::zeros(vec![3, 4], &device)?;
    assert_eq!(zeros.dims(), &[3, 4]);
    assert!(zeros.to_vec()?.iter().all(|&x| x == 0.0));
    Ok(())
}

#[test]
fn test_full() -> Result<()> {
    let device = get_device();
    let full: Tensor<f32, Device> = Tensor::full(42.0, vec![2, 3], &device)?;
    assert_eq!(full.dims(), &[2, 3]);
    assert!(full.to_vec()?.iter().all(|&x| x == 42.0));
    Ok(())
}

#[test]
fn test_f16_roundtrip() -> Result<()> {
    let device = get_device();
    let data: Vec<half::f16> = vec![1.0, 2.0, 3.0].into_iter().map(half::f16::from_f32).collect();
    let t: Tensor<half::f16, Device> = Tensor::from_vec(data.clone(), vec![3], &device)?;
    assert_eq!(t.to_vec()?, data);
    Ok(())
}

#[test]
fn test_bf16_roundtrip() -> Result<()> {
    let device = get_device();
    let data: Vec<half::bf16> = vec![1.0, 2.0, 3.0].into_iter().map(half::bf16::from_f32).collect();
    let t: Tensor<half::bf16, Device> = Tensor::from_vec(data.clone(), vec![3], &device)?;
    assert_eq!(t.to_vec()?, data);
    Ok(())
}

// =============================================================================
// Binary operations
// =============================================================================

#[test]
fn test_add() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4], &device)?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![4], &device)?;
    let c = a.add(&b)?;
    assert_eq!(c.to_vec()?, vec![6.0, 8.0, 10.0, 12.0]);
    Ok(())
}

#[test]
fn test_sub() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![4], &device)?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4], &device)?;
    let c = a.sub(&b)?;
    assert_eq!(c.to_vec()?, vec![4.0, 4.0, 4.0, 4.0]);
    Ok(())
}

#[test]
fn test_mul() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4], &device)?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![4], &device)?;
    let c = a.mul(&b)?;
    assert_eq!(c.to_vec()?, vec![5.0, 12.0, 21.0, 32.0]);
    Ok(())
}

#[test]
fn test_div() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![10.0, 12.0, 21.0, 32.0], vec![4], &device)?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![2.0, 3.0, 7.0, 8.0], vec![4], &device)?;
    let c = a.div(&b)?;
    assert_eq!(c.to_vec()?, vec![5.0, 4.0, 3.0, 4.0]);
    Ok(())
}

#[test]
fn test_scale() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4], &device)?;
    let b = a.scale(2.0)?;
    assert_eq!(b.to_vec()?, vec![2.0, 4.0, 6.0, 8.0]);
    Ok(())
}

#[test]
fn test_maximum() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 5.0, 3.0, 4.0], vec![4], &device)?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![2.0, 3.0, 4.0, 2.0], vec![4], &device)?;
    let c = a.maximum(&b)?;
    assert_eq!(c.to_vec()?, vec![2.0, 5.0, 4.0, 4.0]);
    Ok(())
}

#[test]
fn test_minimum() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 5.0, 3.0, 4.0], vec![4], &device)?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![2.0, 3.0, 4.0, 2.0], vec![4], &device)?;
    let c = a.minimum(&b)?;
    assert_eq!(c.to_vec()?, vec![1.0, 3.0, 3.0, 2.0]);
    Ok(())
}

// =============================================================================
// Unary operations
// =============================================================================

#[test]
fn test_relu() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> =
        Tensor::from_vec(vec![0.0, 1.0, -1.0, 2.0, -2.0], vec![5], &device)?;
    let y = x.relu()?;
    assert_eq!(y.to_vec()?, vec![0.0, 1.0, 0.0, 2.0, 0.0]);
    Ok(())
}

#[test]
fn test_silu() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3], &device)?;
    let y = x.silu()?;
    let y_data = y.to_vec()?;
    assert!((y_data[0] - 0.0).abs() < 1e-5);
    assert!((y_data[1] - 0.7310586).abs() < 1e-5);
    assert!((y_data[2] - (-0.26894143)).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_sqrt() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 4.0, 9.0, 16.0], vec![4], &device)?;
    let y = x.sqrt()?;
    assert_eq!(y.to_vec()?, vec![1.0, 2.0, 3.0, 4.0]);
    Ok(())
}

#[test]
fn test_sqr() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4], &device)?;
    let y = x.sqr()?;
    assert_eq!(y.to_vec()?, vec![1.0, 4.0, 9.0, 16.0]);
    Ok(())
}

#[test]
fn test_tanh() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3], &device)?;
    let y = x.tanh()?;
    let y_data = y.to_vec()?;
    assert!((y_data[0] - 0.0).abs() < 1e-5);
    assert!((y_data[1] - 0.7615942).abs() < 1e-5);
    assert!((y_data[2] - (-0.7615942)).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_sigmoid() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3], &device)?;
    let y = x.sigmoid()?;
    let y_data = y.to_vec()?;
    assert!((y_data[0] - 0.5).abs() < 1e-5);
    assert!((y_data[1] - 0.7310586).abs() < 1e-5);
    assert!((y_data[2] - 0.26894143).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_abs() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> = Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], vec![4], &device)?;
    let y = x.abs()?;
    assert_eq!(y.to_vec()?, vec![1.0, 2.0, 3.0, 4.0]);
    Ok(())
}

// =============================================================================
// Matrix multiplication
// =============================================================================

#[test]
fn test_matmul_2d() -> Result<()> {
    let device = get_device();
    // A = [[1, 2, 3], [4, 5, 6]]  (2x3)
    // B = [[1, 2], [3, 4], [5, 6]]  (3x2)
    // C = A @ B = [[22, 28], [49, 64]]  (2x2)
    let a: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &device)?;
    let b: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], &device)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.dims(), &[2, 2]);
    assert_eq!(c.to_vec()?, vec![22.0, 28.0, 49.0, 64.0]);
    Ok(())
}

#[test]
fn test_matmul_f16() -> Result<()> {
    let device = get_device();
    let a: Tensor<half::f16, Device> = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter().map(half::f16::from_f32).collect(),
        vec![2, 3],
        &device,
    )?;
    let b: Tensor<half::f16, Device> = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter().map(half::f16::from_f32).collect(),
        vec![3, 2],
        &device,
    )?;
    let c = a.matmul(&b)?;
    let c_data: Vec<f32> = c.to_vec()?.iter().map(|x| x.to_f32()).collect();
    assert!((c_data[0] - 22.0).abs() < 0.1);
    assert!((c_data[1] - 28.0).abs() < 0.1);
    assert!((c_data[2] - 49.0).abs() < 0.1);
    assert!((c_data[3] - 64.0).abs() < 0.1);
    Ok(())
}

#[test]
fn test_matmul_batched() -> Result<()> {
    let device = get_device();
    // Batch of 2: each batch is 2x3 @ 3x2
    let a: Tensor<f32, Device> = Tensor::from_vec(
        vec![
            // Batch 0: [[1,2,3], [4,5,6]]
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 1: all ones
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        vec![2, 2, 3],
        &device,
    )?;
    let b: Tensor<f32, Device> = Tensor::from_vec(
        vec![
            // Batch 0
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 1: all ones
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        vec![2, 3, 2],
        &device,
    )?;
    let c = a.matmul(&b)?;
    assert_eq!(c.dims(), &[2, 2, 2]);
    let c_data = c.to_vec()?;
    // Batch 0: [[22, 28], [49, 64]]
    assert_eq!(c_data[0], 22.0);
    assert_eq!(c_data[1], 28.0);
    assert_eq!(c_data[2], 49.0);
    assert_eq!(c_data[3], 64.0);
    // Batch 1: [[3, 3], [3, 3]]
    assert_eq!(c_data[4], 3.0);
    assert_eq!(c_data[5], 3.0);
    assert_eq!(c_data[6], 3.0);
    assert_eq!(c_data[7], 3.0);
    Ok(())
}

// =============================================================================
// Transpose
// =============================================================================

#[test]
fn test_transpose_2d() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &device)?;
    let b = a.transpose(0, 1)?;
    assert_eq!(b.dims(), &[3, 2]);
    // Original: [[1, 2, 3], [4, 5, 6]]
    // Transposed: [[1, 4], [2, 5], [3, 6]]
    assert_eq!(b.to_vec()?, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    Ok(())
}

#[test]
fn test_transpose_3d() -> Result<()> {
    let device = get_device();
    // Shape: [2, 3, 4]
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let a: Tensor<f32, Device> = Tensor::from_vec(data, vec![2, 3, 4], &device)?;

    // Transpose dims 1 and 2: [2, 3, 4] -> [2, 4, 3]
    let b = a.transpose(1, 2)?;
    assert_eq!(b.dims(), &[2, 4, 3]);
    Ok(())
}

// =============================================================================
// Softmax
// =============================================================================

#[test]
fn test_softmax() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], vec![2, 3], &device)?;
    let y = x.softmax()?;
    let y_data = y.to_vec()?;

    // Each row should sum to 1.0
    let row1_sum: f32 = y_data[0..3].iter().sum();
    let row2_sum: f32 = y_data[3..6].iter().sum();
    assert!((row1_sum - 1.0).abs() < 1e-5);
    assert!((row2_sum - 1.0).abs() < 1e-5);

    // Check expected values: softmax([1,2,3]) ≈ [0.090, 0.245, 0.665]
    assert!((y_data[0] - 0.0900306).abs() < 1e-4);
    assert!((y_data[1] - 0.2447285).abs() < 1e-4);
    assert!((y_data[2] - 0.6652409).abs() < 1e-4);
    Ok(())
}

#[test]
fn test_softmax_single_row() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> = Tensor::from_vec(vec![0.0, 0.0, 0.0], vec![1, 3], &device)?;
    let y = x.softmax()?;
    let y_data = y.to_vec()?;
    // Uniform distribution
    assert!((y_data[0] - 1.0 / 3.0).abs() < 1e-5);
    assert!((y_data[1] - 1.0 / 3.0).abs() < 1e-5);
    assert!((y_data[2] - 1.0 / 3.0).abs() < 1e-5);
    Ok(())
}

// =============================================================================
// RMS Norm
// =============================================================================

#[test]
fn test_rms_norm() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &device)?;
    let alpha: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3], &device)?;
    let y = x.rms_norm(&alpha, 1e-5)?;
    let y_data = y.to_vec()?;

    // RMS of [1,2,3] = sqrt((1+4+9)/3) = sqrt(14/3) ≈ 2.16
    let rms_row1 = (1.0f32 + 4.0 + 9.0) / 3.0;
    let scale1 = 1.0 / (rms_row1 + 1e-5).sqrt();
    assert!((y_data[0] - 1.0 * scale1).abs() < 1e-4);
    assert!((y_data[1] - 2.0 * scale1).abs() < 1e-4);
    assert!((y_data[2] - 3.0 * scale1).abs() < 1e-4);

    // RMS of [4,5,6] = sqrt((16+25+36)/3) = sqrt(77/3) ≈ 5.07
    let rms_row2 = (16.0f32 + 25.0 + 36.0) / 3.0;
    let scale2 = 1.0 / (rms_row2 + 1e-5).sqrt();
    assert!((y_data[3] - 4.0 * scale2).abs() < 1e-4);
    assert!((y_data[4] - 5.0 * scale2).abs() < 1e-4);
    assert!((y_data[5] - 6.0 * scale2).abs() < 1e-4);
    Ok(())
}

#[test]
fn test_rms_norm_with_scale() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3], &device)?;
    let alpha: Tensor<f32, Device> = Tensor::from_vec(vec![2.0, 2.0, 2.0], vec![3], &device)?;
    let y = x.rms_norm(&alpha, 1e-5)?;
    let y_data = y.to_vec()?;

    let rms = (1.0f32 + 4.0 + 9.0) / 3.0;
    let scale = 1.0 / (rms + 1e-5).sqrt();
    // Values should be doubled due to alpha=2
    assert!((y_data[0] - 1.0 * scale * 2.0).abs() < 1e-4);
    assert!((y_data[1] - 2.0 * scale * 2.0).abs() < 1e-4);
    assert!((y_data[2] - 3.0 * scale * 2.0).abs() < 1e-4);
    Ok(())
}

// =============================================================================
// Layer Norm
// =============================================================================

#[test]
fn test_layer_norm() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &device)?;
    let weight: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3], &device)?;
    let bias: Tensor<f32, Device> = Tensor::from_vec(vec![0.0, 0.0, 0.0], vec![3], &device)?;
    let y = x.layer_norm(&weight, &bias, 1e-5)?;
    let y_data = y.to_vec()?;

    // For [1,2,3]: mean=2, var=2/3
    let mean1 = 2.0f32;
    let var1 = ((1.0 - mean1).powi(2) + (2.0 - mean1).powi(2) + (3.0 - mean1).powi(2)) / 3.0;
    let inv_std1 = 1.0 / (var1 + 1e-5).sqrt();
    assert!((y_data[0] - (1.0 - mean1) * inv_std1).abs() < 1e-4);
    assert!((y_data[1] - (2.0 - mean1) * inv_std1).abs() < 1e-4);
    assert!((y_data[2] - (3.0 - mean1) * inv_std1).abs() < 1e-4);

    // For [4,5,6]: mean=5, var=2/3
    let mean2 = 5.0f32;
    let var2 = ((4.0 - mean2).powi(2) + (5.0 - mean2).powi(2) + (6.0 - mean2).powi(2)) / 3.0;
    let inv_std2 = 1.0 / (var2 + 1e-5).sqrt();
    assert!((y_data[3] - (4.0 - mean2) * inv_std2).abs() < 1e-4);
    assert!((y_data[4] - (5.0 - mean2) * inv_std2).abs() < 1e-4);
    assert!((y_data[5] - (6.0 - mean2) * inv_std2).abs() < 1e-4);
    Ok(())
}

#[test]
fn test_layer_norm_with_affine() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3], &device)?;
    let weight: Tensor<f32, Device> = Tensor::from_vec(vec![2.0, 2.0, 2.0], vec![3], &device)?;
    let bias: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3], &device)?;
    let y = x.layer_norm(&weight, &bias, 1e-5)?;
    let y_data = y.to_vec()?;

    let mean = 2.0f32;
    let var = 2.0f32 / 3.0;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    // y = (x - mean) * inv_std * weight + bias
    assert!((y_data[0] - ((1.0 - mean) * inv_std * 2.0 + 1.0)).abs() < 1e-4);
    assert!((y_data[1] - ((2.0 - mean) * inv_std * 2.0 + 1.0)).abs() < 1e-4);
    assert!((y_data[2] - ((3.0 - mean) * inv_std * 2.0 + 1.0)).abs() < 1e-4);
    Ok(())
}

// =============================================================================
// Reshape
// =============================================================================

#[test]
fn test_reshape() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> =
        Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], vec![2, 3], &device)?;

    let b = a.reshape((3, 2))?;
    assert_eq!(b.dims(), &[3, 2]);
    assert_eq!(b.to_vec()?, vec![1., 2., 3., 4., 5., 6.]);

    let c = a.reshape((6,))?;
    assert_eq!(c.dims(), &[6]);

    let d = a.reshape((1, 2, 3))?;
    assert_eq!(d.dims(), &[1, 2, 3]);
    Ok(())
}

#[test]
fn test_reshape_with_hole() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> =
        Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], vec![2, 3], &device)?;

    let b = a.reshape((3, ()))?;
    assert_eq!(b.dims(), &[3, 2]);

    let c = a.reshape(((), 2))?;
    assert_eq!(c.dims(), &[3, 2]);
    Ok(())
}
