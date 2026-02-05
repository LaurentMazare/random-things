use mimi::{CPU, CpuTensor, Result, Tensor};

#[test]
fn test_cat_dim0() -> Result<()> {
    // Two 2x3 tensors concatenated along dim 0 -> 4x3
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), &CPU)?;
    let b: CpuTensor<f32> = Tensor::from_vec(vec![7., 8., 9., 10., 11., 12.], (2, 3), &CPU)?;

    let c = Tensor::cat(&[&a, &b], 0)?;
    assert_eq!(c.dims(), &[4, 3]);
    assert_eq!(c.to_vec()?, vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);
    Ok(())
}

#[test]
fn test_cat_dim1() -> Result<()> {
    // Two 2x3 tensors concatenated along dim 1 -> 2x6
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), &CPU)?;
    let b: CpuTensor<f32> = Tensor::from_vec(vec![7., 8., 9., 10., 11., 12.], (2, 3), &CPU)?;

    let c = Tensor::cat(&[&a, &b], 1)?;
    assert_eq!(c.dims(), &[2, 6]);
    // Row 0: [1,2,3] ++ [7,8,9] = [1,2,3,7,8,9]
    // Row 1: [4,5,6] ++ [10,11,12] = [4,5,6,10,11,12]
    assert_eq!(c.to_vec()?, vec![1., 2., 3., 7., 8., 9., 4., 5., 6., 10., 11., 12.]);
    Ok(())
}

#[test]
fn test_cat_3d_dim1() -> Result<()> {
    // Two 2x2x3 tensors concatenated along dim 1 -> 2x4x3
    let a: CpuTensor<f32> =
        Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (2, 2, 3), &CPU)?;
    let b: CpuTensor<f32> =
        Tensor::from_vec((13..=24).map(|x| x as f32).collect(), (2, 2, 3), &CPU)?;

    let c = Tensor::cat(&[&a, &b], 1)?;
    assert_eq!(c.dims(), &[2, 4, 3]);
    // Batch 0: [[1,2,3],[4,5,6]] ++ [[13,14,15],[16,17,18]]
    // Batch 1: [[7,8,9],[10,11,12]] ++ [[19,20,21],[22,23,24]]
    assert_eq!(
        c.to_vec()?,
        vec![
            1., 2., 3., 4., 5., 6., 13., 14., 15., 16., 17., 18., // batch 0
            7., 8., 9., 10., 11., 12., 19., 20., 21., 22., 23., 24. // batch 1
        ]
    );
    Ok(())
}

#[test]
fn test_reshape() -> Result<()> {
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), &CPU)?;

    // Reshape to 3x2
    let b = a.reshape((3, 2))?;
    assert_eq!(b.dims(), &[3, 2]);
    assert_eq!(b.to_vec()?, vec![1., 2., 3., 4., 5., 6.]);

    // Reshape to 6
    let c = a.reshape((6,))?;
    assert_eq!(c.dims(), &[6]);

    // Reshape to 1x6
    let d = a.reshape((1, 6))?;
    assert_eq!(d.dims(), &[1, 6]);

    // Reshape to 1x2x3
    let e = a.reshape((1, 2, 3))?;
    assert_eq!(e.dims(), &[1, 2, 3]);
    Ok(())
}

#[test]
fn test_reshape_with_hole() -> Result<()> {
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), &CPU)?;

    // Reshape with inferred dimension
    let b = a.reshape((3, ()))?;
    assert_eq!(b.dims(), &[3, 2]);

    let c = a.reshape(((), 2))?;
    assert_eq!(c.dims(), &[3, 2]);

    let d = a.reshape((1, (), 3))?;
    assert_eq!(d.dims(), &[1, 2, 3]);
    Ok(())
}

#[test]
fn test_index_select_dim0() -> Result<()> {
    // Select rows from a 4x3 tensor
    let a: CpuTensor<f32> =
        Tensor::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], (4, 3), &CPU)?;

    // Select rows 0, 2, 3
    let b = a.index_select(&[0, 2, 3], 0)?;
    assert_eq!(b.dims(), &[3, 3]);
    assert_eq!(b.to_vec()?, vec![1., 2., 3., 7., 8., 9., 10., 11., 12.]);

    // Select with repetition
    let c = a.index_select(&[1, 1, 0], 0)?;
    assert_eq!(c.dims(), &[3, 3]);
    assert_eq!(c.to_vec()?, vec![4., 5., 6., 4., 5., 6., 1., 2., 3.]);
    Ok(())
}

#[test]
fn test_index_select_dim1() -> Result<()> {
    // Select columns from a 2x4 tensor
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.], (2, 4), &CPU)?;

    // Select columns 0, 2
    let b = a.index_select(&[0, 2], 1)?;
    assert_eq!(b.dims(), &[2, 2]);
    // Row 0: [1, 3], Row 1: [5, 7]
    assert_eq!(b.to_vec()?, vec![1., 3., 5., 7.]);
    Ok(())
}

#[test]
fn test_index_select_3d() -> Result<()> {
    // 2x3x2 tensor
    let a: CpuTensor<f32> =
        Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (2, 3, 2), &CPU)?;
    // Data layout:
    // Batch 0: [[1,2], [3,4], [5,6]]
    // Batch 1: [[7,8], [9,10], [11,12]]

    // Select along dim 1 (middle dimension)
    let b = a.index_select(&[0, 2], 1)?;
    assert_eq!(b.dims(), &[2, 2, 2]);
    // Batch 0: [[1,2], [5,6]]
    // Batch 1: [[7,8], [11,12]]
    assert_eq!(b.to_vec()?, vec![1., 2., 5., 6., 7., 8., 11., 12.]);
    Ok(())
}

#[test]
fn test_max_dim0() -> Result<()> {
    // 3x4 tensor, max along dim 0 -> 4
    let a: CpuTensor<f32> =
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), &CPU)?;
    // Column-wise max:
    // col 0: max(1, 8, 9) = 9
    // col 1: max(5, 2, 0) = 5
    // col 2: max(3, 7, 1) = 7
    // col 3: max(4, 6, 2) = 6
    let b = a.max(0)?;
    assert_eq!(b.dims(), &[4]);
    assert_eq!(b.to_vec()?, vec![9., 5., 7., 6.]);
    Ok(())
}

#[test]
fn test_max_dim1() -> Result<()> {
    // 3x4 tensor, max along dim 1 -> 3
    let a: CpuTensor<f32> =
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), &CPU)?;
    // Row-wise max:
    // row 0: max(1, 5, 3, 4) = 5
    // row 1: max(8, 2, 7, 6) = 8
    // row 2: max(9, 0, 1, 2) = 9
    let b = a.max(1)?;
    assert_eq!(b.dims(), &[3]);
    assert_eq!(b.to_vec()?, vec![5., 8., 9.]);
    Ok(())
}

#[test]
fn test_min_dim0() -> Result<()> {
    // 3x4 tensor, min along dim 0 -> 4
    let a: CpuTensor<f32> =
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), &CPU)?;
    // Column-wise min:
    // col 0: min(1, 8, 9) = 1
    // col 1: min(5, 2, 0) = 0
    // col 2: min(3, 7, 1) = 1
    // col 3: min(4, 6, 2) = 2
    let b = a.min(0)?;
    assert_eq!(b.dims(), &[4]);
    assert_eq!(b.to_vec()?, vec![1., 0., 1., 2.]);
    Ok(())
}

#[test]
fn test_min_dim1() -> Result<()> {
    // 3x4 tensor, min along dim 1 -> 3
    let a: CpuTensor<f32> =
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), &CPU)?;
    // Row-wise min:
    // row 0: min(1, 5, 3, 4) = 1
    // row 1: min(8, 2, 7, 6) = 2
    // row 2: min(9, 0, 1, 2) = 0
    let b = a.min(1)?;
    assert_eq!(b.dims(), &[3]);
    assert_eq!(b.to_vec()?, vec![1., 2., 0.]);
    Ok(())
}

#[test]
fn test_argmin_dim0() -> Result<()> {
    // 3x4 tensor, argmin along dim 0 -> 4
    let a: CpuTensor<f32> =
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), &CPU)?;
    // Column-wise argmin:
    // col 0: argmin(1, 8, 9) = 0
    // col 1: argmin(5, 2, 0) = 2
    // col 2: argmin(3, 7, 1) = 2
    // col 3: argmin(4, 6, 2) = 2
    let b = a.argmin(0)?;
    assert_eq!(b.dims(), &[4]);
    assert_eq!(b.to_vec()?, vec![0i64, 2, 2, 2]);
    Ok(())
}

#[test]
fn test_argmin_dim1() -> Result<()> {
    // 3x4 tensor, argmin along dim 1 -> 3
    let a: CpuTensor<f32> =
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), &CPU)?;
    // Row-wise argmin:
    // row 0: argmin(1, 5, 3, 4) = 0
    // row 1: argmin(8, 2, 7, 6) = 1
    // row 2: argmin(9, 0, 1, 2) = 1
    let b = a.argmin(1)?;
    assert_eq!(b.dims(), &[3]);
    assert_eq!(b.to_vec()?, vec![0i64, 1, 1]);
    Ok(())
}

#[test]
fn test_max_3d() -> Result<()> {
    // 2x3x2 tensor, max along dim 1 -> 2x2
    let a: CpuTensor<f32> =
        Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (2, 3, 2), &CPU)?;
    // Batch 0: [[1,2], [3,4], [5,6]] -> max along rows: [5, 6]
    // Batch 1: [[7,8], [9,10], [11,12]] -> max along rows: [11, 12]
    let b = a.max(1)?;
    assert_eq!(b.dims(), &[2, 2]);
    assert_eq!(b.to_vec()?, vec![5., 6., 11., 12.]);
    Ok(())
}

#[test]
fn test_broadcast_add_same_shape() -> Result<()> {
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4.], (2, 2), &CPU)?;
    let b: CpuTensor<f32> = Tensor::from_vec(vec![10., 20., 30., 40.], (2, 2), &CPU)?;
    let c = a.broadcast_add(&b)?;
    assert_eq!(c.dims(), &[2, 2]);
    assert_eq!(c.to_vec()?, vec![11., 22., 33., 44.]);
    Ok(())
}

#[test]
fn test_broadcast_add_1d_to_2d() -> Result<()> {
    // [2, 3] + [3] -> [2, 3]
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), &CPU)?;
    let b: CpuTensor<f32> = Tensor::from_vec(vec![10., 20., 30.], (3,), &CPU)?;
    let c = a.broadcast_add(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    // Row 0: [1+10, 2+20, 3+30] = [11, 22, 33]
    // Row 1: [4+10, 5+20, 6+30] = [14, 25, 36]
    assert_eq!(c.to_vec()?, vec![11., 22., 33., 14., 25., 36.]);
    Ok(())
}

#[test]
fn test_broadcast_mul_column() -> Result<()> {
    // [2, 3] * [2, 1] -> [2, 3]
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), &CPU)?;
    let b: CpuTensor<f32> = Tensor::from_vec(vec![2., 3.], (2, 1), &CPU)?;
    let c = a.broadcast_mul(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    // Row 0: [1*2, 2*2, 3*2] = [2, 4, 6]
    // Row 1: [4*3, 5*3, 6*3] = [12, 15, 18]
    assert_eq!(c.to_vec()?, vec![2., 4., 6., 12., 15., 18.]);
    Ok(())
}

#[test]
fn test_broadcast_sub() -> Result<()> {
    // [2, 3] - [3] -> [2, 3]
    let a: CpuTensor<f32> = Tensor::from_vec(vec![10., 20., 30., 40., 50., 60.], (2, 3), &CPU)?;
    let b: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3.], (3,), &CPU)?;
    let c = a.broadcast_sub(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(c.to_vec()?, vec![9., 18., 27., 39., 48., 57.]);
    Ok(())
}

#[test]
fn test_broadcast_div() -> Result<()> {
    // [2, 3] / [2, 1] -> [2, 3]
    let a: CpuTensor<f32> = Tensor::from_vec(vec![2., 4., 6., 9., 12., 15.], (2, 3), &CPU)?;
    let b: CpuTensor<f32> = Tensor::from_vec(vec![2., 3.], (2, 1), &CPU)?;
    let c = a.broadcast_div(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    // Row 0: [2/2, 4/2, 6/2] = [1, 2, 3]
    // Row 1: [9/3, 12/3, 15/3] = [3, 4, 5]
    assert_eq!(c.to_vec()?, vec![1., 2., 3., 3., 4., 5.]);
    Ok(())
}

#[test]
fn test_broadcast_3d() -> Result<()> {
    // [2, 3, 4] + [4] -> [2, 3, 4]
    let a: CpuTensor<f32> =
        Tensor::from_vec((1..=24).map(|x| x as f32).collect(), (2, 3, 4), &CPU)?;
    let b: CpuTensor<f32> = Tensor::from_vec(vec![100., 200., 300., 400.], (4,), &CPU)?;
    let c = a.broadcast_add(&b)?;
    assert_eq!(c.dims(), &[2, 3, 4]);
    let c_vec = c.to_vec()?;
    // First element: 1 + 100 = 101
    assert_eq!(c_vec[0], 101.);
    // Second element: 2 + 200 = 202
    assert_eq!(c_vec[1], 202.);
    // Fifth element: 5 + 100 = 105
    assert_eq!(c_vec[4], 105.);
    Ok(())
}

#[test]
fn test_unsqueeze_dim0() -> Result<()> {
    // [3, 4] -> unsqueeze(0) -> [1, 3, 4]
    let a: CpuTensor<f32> = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (3, 4), &CPU)?;
    let b = a.unsqueeze(0)?;
    assert_eq!(b.dims(), &[1, 3, 4]);
    assert_eq!(b.to_vec()?, a.to_vec()?);
    Ok(())
}

#[test]
fn test_unsqueeze_dim1() -> Result<()> {
    // [3, 4] -> unsqueeze(1) -> [3, 1, 4]
    let a: CpuTensor<f32> = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (3, 4), &CPU)?;
    let b = a.unsqueeze(1)?;
    assert_eq!(b.dims(), &[3, 1, 4]);
    assert_eq!(b.to_vec()?, a.to_vec()?);
    Ok(())
}

#[test]
fn test_unsqueeze_dim_last() -> Result<()> {
    // [3, 4] -> unsqueeze(2) -> [3, 4, 1]
    let a: CpuTensor<f32> = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (3, 4), &CPU)?;
    let b = a.unsqueeze(2)?;
    assert_eq!(b.dims(), &[3, 4, 1]);
    assert_eq!(b.to_vec()?, a.to_vec()?);
    Ok(())
}

#[test]
fn test_unsqueeze_1d() -> Result<()> {
    // [4] -> unsqueeze(0) -> [1, 4]
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4.], (4,), &CPU)?;
    let b = a.unsqueeze(0)?;
    assert_eq!(b.dims(), &[1, 4]);

    // [4] -> unsqueeze(1) -> [4, 1]
    let c = a.unsqueeze(1)?;
    assert_eq!(c.dims(), &[4, 1]);
    Ok(())
}

#[test]
fn test_pad_with_zeros_1d() -> Result<()> {
    // [4] -> pad_with_zeros(0, 2, 3) -> [9]
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4.], (4,), &CPU)?;
    let b = a.pad_with_zeros(0, 2, 3)?;
    assert_eq!(b.dims(), &[9]);
    assert_eq!(b.to_vec()?, vec![0., 0., 1., 2., 3., 4., 0., 0., 0.]);
    Ok(())
}

#[test]
fn test_pad_with_zeros_2d_dim0() -> Result<()> {
    // [2, 3] -> pad_with_zeros(0, 1, 1) -> [4, 3]
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), &CPU)?;
    let b = a.pad_with_zeros(0, 1, 1)?;
    assert_eq!(b.dims(), &[4, 3]);
    // Row 0: zeros, Row 1: [1,2,3], Row 2: [4,5,6], Row 3: zeros
    assert_eq!(b.to_vec()?, vec![0., 0., 0., 1., 2., 3., 4., 5., 6., 0., 0., 0.]);
    Ok(())
}

#[test]
fn test_pad_with_zeros_2d_dim1() -> Result<()> {
    // [2, 3] -> pad_with_zeros(1, 1, 2) -> [2, 6]
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), &CPU)?;
    let b = a.pad_with_zeros(1, 1, 2)?;
    assert_eq!(b.dims(), &[2, 6]);
    // Row 0: [0, 1, 2, 3, 0, 0]
    // Row 1: [0, 4, 5, 6, 0, 0]
    assert_eq!(b.to_vec()?, vec![0., 1., 2., 3., 0., 0., 0., 4., 5., 6., 0., 0.]);
    Ok(())
}

#[test]
fn test_pad_with_zeros_3d() -> Result<()> {
    // [2, 2, 3] -> pad_with_zeros(1, 1, 0) -> [2, 3, 3]
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let a: CpuTensor<f32> = Tensor::from_vec(data, (2, 2, 3), &CPU)?;
    let b = a.pad_with_zeros(1, 1, 0)?;
    assert_eq!(b.dims(), &[2, 3, 3]);
    // First batch: [[0,0,0], [1,2,3], [4,5,6]]
    // Second batch: [[0,0,0], [7,8,9], [10,11,12]]
    assert_eq!(
        b.to_vec()?,
        vec![0., 0., 0., 1., 2., 3., 4., 5., 6., 0., 0., 0., 7., 8., 9., 10., 11., 12.]
    );
    Ok(())
}

#[test]
fn test_conv1d_simple() -> Result<()> {
    // Input: (batch=1, in_channels=1, length=5)
    // Kernel: (out_channels=1, in_channels=1, kernel_size=3)
    // No padding, stride=1, groups=1
    let input: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5.], (1, 1, 5), &CPU)?;
    let kernel: CpuTensor<f32> = Tensor::from_vec(vec![1., 0., -1.], (1, 1, 3), &CPU)?;

    let output = input.conv1d(&kernel, None, 1, 0, 1, 1)?;
    assert_eq!(output.dims(), &[1, 1, 3]);
    // output[i] = input[i]*1 + input[i+1]*0 + input[i+2]*(-1)
    // output[0] = 1 - 3 = -2
    // output[1] = 2 - 4 = -2
    // output[2] = 3 - 5 = -2
    assert_eq!(output.to_vec()?, vec![-2., -2., -2.]);
    Ok(())
}

#[test]
fn test_conv1d_with_padding() -> Result<()> {
    // Input: (batch=1, in_channels=1, length=4)
    // Kernel: (out_channels=1, in_channels=1, kernel_size=3)
    // Padding=1, stride=1
    let input: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4.], (1, 1, 4), &CPU)?;
    let kernel: CpuTensor<f32> = Tensor::from_vec(vec![1., 1., 1.], (1, 1, 3), &CPU)?;

    let output = input.conv1d(&kernel, None, 1, 1, 1, 1)?;
    assert_eq!(output.dims(), &[1, 1, 4]);
    // With padding=1, we have [0, 1, 2, 3, 4, 0] as effective input
    // output[0] = 0 + 1 + 2 = 3
    // output[1] = 1 + 2 + 3 = 6
    // output[2] = 2 + 3 + 4 = 9
    // output[3] = 3 + 4 + 0 = 7
    assert_eq!(output.to_vec()?, vec![3., 6., 9., 7.]);
    Ok(())
}

#[test]
fn test_conv1d_with_stride() -> Result<()> {
    // Input: (batch=1, in_channels=1, length=6)
    // Kernel: (out_channels=1, in_channels=1, kernel_size=2)
    // Stride=2
    let input: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (1, 1, 6), &CPU)?;
    let kernel: CpuTensor<f32> = Tensor::from_vec(vec![1., 1.], (1, 1, 2), &CPU)?;

    let output = input.conv1d(&kernel, None, 2, 0, 1, 1)?;
    // out_length = (6 - 2) / 2 + 1 = 3
    assert_eq!(output.dims(), &[1, 1, 3]);
    // output[0] = 1 + 2 = 3
    // output[1] = 3 + 4 = 7
    // output[2] = 5 + 6 = 11
    assert_eq!(output.to_vec()?, vec![3., 7., 11.]);
    Ok(())
}

#[test]
fn test_conv1d_with_bias() -> Result<()> {
    let input: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5.], (1, 1, 5), &CPU)?;
    let kernel: CpuTensor<f32> = Tensor::from_vec(vec![1., 1., 1.], (1, 1, 3), &CPU)?;
    let bias: CpuTensor<f32> = Tensor::from_vec(vec![10.], (1,), &CPU)?;

    let output = input.conv1d(&kernel, Some(&bias), 1, 0, 1, 1)?;
    assert_eq!(output.dims(), &[1, 1, 3]);
    // Without bias: [6, 9, 12], with bias: [16, 19, 22]
    assert_eq!(output.to_vec()?, vec![16., 19., 22.]);
    Ok(())
}

#[test]
fn test_conv1d_multi_channel() -> Result<()> {
    // Input: (batch=1, in_channels=2, length=3)
    // Kernel: (out_channels=2, in_channels=2, kernel_size=2)
    let input: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (1, 2, 3), &CPU)?;
    // kernel[0] for out_channel 0: [[1,1], [0,0]] - only uses in_channel 0
    // kernel[1] for out_channel 1: [[0,0], [1,1]] - only uses in_channel 1
    let kernel: CpuTensor<f32> =
        Tensor::from_vec(vec![1., 1., 0., 0., 0., 0., 1., 1.], (2, 2, 2), &CPU)?;

    let output = input.conv1d(&kernel, None, 1, 0, 1, 1)?;
    assert_eq!(output.dims(), &[1, 2, 2]);
    // out[0,0] = 1+2 = 3, out[0,1] = 2+3 = 5
    // out[1,0] = 4+5 = 9, out[1,1] = 5+6 = 11
    assert_eq!(output.to_vec()?, vec![3., 5., 9., 11.]);
    Ok(())
}

#[test]
fn test_conv_transpose1d_simple() -> Result<()> {
    // Input: (batch=1, in_channels=1, length=3)
    // Kernel: (in_channels=1, out_channels=1, kernel_size=3)
    // stride=1, no padding
    let input: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3.], (1, 1, 3), &CPU)?;
    let kernel: CpuTensor<f32> = Tensor::from_vec(vec![1., 1., 1.], (1, 1, 3), &CPU)?;

    let output = input.conv_transpose1d(&kernel, None, 1, 0, 0, 1)?;
    // out_length = (3-1)*1 + 3 + 0 - 0 = 5
    assert_eq!(output.dims(), &[1, 1, 5]);
    // Each input value contributes to 3 consecutive output positions
    // output[0] = 1*1 = 1
    // output[1] = 1*1 + 2*1 = 3
    // output[2] = 1*1 + 2*1 + 3*1 = 6
    // output[3] = 2*1 + 3*1 = 5
    // output[4] = 3*1 = 3
    assert_eq!(output.to_vec()?, vec![1., 3., 6., 5., 3.]);
    Ok(())
}

#[test]
fn test_conv_transpose1d_with_stride() -> Result<()> {
    // Input: (batch=1, in_channels=1, length=3)
    // Kernel: (in_channels=1, out_channels=1, kernel_size=2)
    // stride=2
    let input: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3.], (1, 1, 3), &CPU)?;
    let kernel: CpuTensor<f32> = Tensor::from_vec(vec![1., 1.], (1, 1, 2), &CPU)?;

    let output = input.conv_transpose1d(&kernel, None, 2, 0, 0, 1)?;
    // out_length = (3-1)*2 + 2 + 0 - 0 = 6
    assert_eq!(output.dims(), &[1, 1, 6]);
    // Input at position i contributes to output positions i*stride + k
    // input[0]=1 -> output[0], output[1]
    // input[1]=2 -> output[2], output[3]
    // input[2]=3 -> output[4], output[5]
    assert_eq!(output.to_vec()?, vec![1., 1., 2., 2., 3., 3.]);
    Ok(())
}

#[test]
fn test_conv_transpose1d_with_bias() -> Result<()> {
    let input: CpuTensor<f32> = Tensor::from_vec(vec![1., 2.], (1, 1, 2), &CPU)?;
    let kernel: CpuTensor<f32> = Tensor::from_vec(vec![1., 1.], (1, 1, 2), &CPU)?;
    let bias: CpuTensor<f32> = Tensor::from_vec(vec![5.], (1,), &CPU)?;

    let output = input.conv_transpose1d(&kernel, Some(&bias), 1, 0, 0, 1)?;
    // out_length = (2-1)*1 + 2 = 3
    assert_eq!(output.dims(), &[1, 1, 3]);
    // Without bias: [1, 3, 2], with bias: [6, 8, 7]
    assert_eq!(output.to_vec()?, vec![6., 8., 7.]);
    Ok(())
}

#[test]
fn test_pad_with_same_1d() -> Result<()> {
    // [1, 2, 3, 4] -> pad_with_same(0, 2, 3) -> [1, 1, 1, 2, 3, 4, 4, 4, 4]
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4.], (4,), &CPU)?;
    let b = a.pad_with_same(0, 2, 3)?;
    assert_eq!(b.dims(), &[9]);
    assert_eq!(b.to_vec()?, vec![1., 1., 1., 2., 3., 4., 4., 4., 4.]);
    Ok(())
}

#[test]
fn test_pad_with_same_2d_dim0() -> Result<()> {
    // [2, 3] -> pad_with_same(0, 1, 1) -> [4, 3]
    // Replicates first and last rows
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), &CPU)?;
    let b = a.pad_with_same(0, 1, 1)?;
    assert_eq!(b.dims(), &[4, 3]);
    // Row 0: copy of row 0 = [1,2,3]
    // Row 1: original row 0 = [1,2,3]
    // Row 2: original row 1 = [4,5,6]
    // Row 3: copy of row 1 = [4,5,6]
    assert_eq!(b.to_vec()?, vec![1., 2., 3., 1., 2., 3., 4., 5., 6., 4., 5., 6.]);
    Ok(())
}

#[test]
fn test_pad_with_same_2d_dim1() -> Result<()> {
    // [2, 3] -> pad_with_same(1, 1, 2) -> [2, 6]
    // Replicates first and last columns
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), &CPU)?;
    let b = a.pad_with_same(1, 1, 2)?;
    assert_eq!(b.dims(), &[2, 6]);
    // Row 0: [1, 1, 2, 3, 3, 3]
    // Row 1: [4, 4, 5, 6, 6, 6]
    assert_eq!(b.to_vec()?, vec![1., 1., 2., 3., 3., 3., 4., 4., 5., 6., 6., 6.]);
    Ok(())
}

#[test]
fn test_pad_with_same_3d() -> Result<()> {
    // [2, 2, 2] -> pad_with_same(1, 1, 1) -> [2, 4, 2]
    let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let a: CpuTensor<f32> = Tensor::from_vec(data, (2, 2, 2), &CPU)?;
    let b = a.pad_with_same(1, 1, 1)?;
    assert_eq!(b.dims(), &[2, 4, 2]);
    // First batch [2,2]: [[1,2], [3,4]] -> [[1,2], [1,2], [3,4], [3,4]]
    // Second batch [2,2]: [[5,6], [7,8]] -> [[5,6], [5,6], [7,8], [7,8]]
    assert_eq!(b.to_vec()?, vec![1., 2., 1., 2., 3., 4., 3., 4., 5., 6., 5., 6., 7., 8., 7., 8.]);
    Ok(())
}

#[test]
fn test_sum_keepdim_1d() -> Result<()> {
    // [5] -> sum_keepdim(0) -> [1]
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5.], (5,), &CPU)?;
    let b = a.sum_keepdim(vec![0])?;
    assert_eq!(b.dims(), &[1]);
    assert_eq!(b.to_vec()?, vec![15.]);
    Ok(())
}

#[test]
fn test_sum_keepdim_2d_dim0() -> Result<()> {
    // [3, 4] -> sum_keepdim(0) -> [1, 4]
    let a: CpuTensor<f32> =
        Tensor::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], (3, 4), &CPU)?;
    let b = a.sum_keepdim(vec![0])?;
    assert_eq!(b.dims(), &[1, 4]);
    // Column sums: 1+5+9=15, 2+6+10=18, 3+7+11=21, 4+8+12=24
    assert_eq!(b.to_vec()?, vec![15., 18., 21., 24.]);
    Ok(())
}

#[test]
fn test_sum_keepdim_2d_dim1() -> Result<()> {
    // [3, 4] -> sum_keepdim(1) -> [3, 1]
    let a: CpuTensor<f32> =
        Tensor::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], (3, 4), &CPU)?;
    let b = a.sum_keepdim(vec![1])?;
    assert_eq!(b.dims(), &[3, 1]);
    // Row sums: 1+2+3+4=10, 5+6+7+8=26, 9+10+11+12=42
    assert_eq!(b.to_vec()?, vec![10., 26., 42.]);
    Ok(())
}

#[test]
fn test_sum_keepdim_3d() -> Result<()> {
    // [2, 3, 2] -> sum_keepdim(1) -> [2, 1, 2]
    let a: CpuTensor<f32> =
        Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (2, 3, 2), &CPU)?;
    let b = a.sum_keepdim(vec![1])?;
    assert_eq!(b.dims(), &[2, 1, 2]);
    // Batch 0: [[1,2], [3,4], [5,6]] -> sum along dim 1 -> [9, 12]
    // Batch 1: [[7,8], [9,10], [11,12]] -> sum along dim 1 -> [27, 30]
    assert_eq!(b.to_vec()?, vec![9., 12., 27., 30.]);
    Ok(())
}

#[test]
fn test_sum_keepdim_multiple_dims() -> Result<()> {
    // [2, 3, 4] -> sum_keepdim([1, 2]) -> [2, 1, 1]
    let a: CpuTensor<f32> =
        Tensor::from_vec((1..=24).map(|x| x as f32).collect(), (2, 3, 4), &CPU)?;
    let b = a.sum_keepdim(vec![1, 2])?;
    assert_eq!(b.dims(), &[2, 1, 1]);
    // Batch 0: sum of 1..12 = 78
    // Batch 1: sum of 13..24 = 222
    assert_eq!(b.to_vec()?, vec![78., 222.]);
    Ok(())
}

#[test]
fn test_slice_set_dim0() -> Result<()> {
    // dst: [4, 3], src: [2, 3], set at offset 1 along dim 0
    let dst: CpuTensor<f32> = Tensor::zeros((4, 3), &CPU)?;
    let src: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), &CPU)?;

    dst.slice_set(&src, 0, 1)?;
    // Row 0: [0, 0, 0]
    // Row 1: [1, 2, 3]
    // Row 2: [4, 5, 6]
    // Row 3: [0, 0, 0]
    assert_eq!(dst.to_vec()?, vec![0., 0., 0., 1., 2., 3., 4., 5., 6., 0., 0., 0.]);
    Ok(())
}

#[test]
fn test_slice_set_dim1() -> Result<()> {
    // dst: [2, 6], src: [2, 3], set at offset 2 along dim 1
    let dst: CpuTensor<f32> = Tensor::zeros((2, 6), &CPU)?;
    let src: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), &CPU)?;

    dst.slice_set(&src, 1, 2)?;
    // Row 0: [0, 0, 1, 2, 3, 0]
    // Row 1: [0, 0, 4, 5, 6, 0]
    assert_eq!(dst.to_vec()?, vec![0., 0., 1., 2., 3., 0., 0., 0., 4., 5., 6., 0.]);
    Ok(())
}

#[test]
fn test_slice_set_3d() -> Result<()> {
    // dst: [2, 4, 3], src: [2, 2, 3], set at offset 1 along dim 1
    let dst: CpuTensor<f32> = Tensor::zeros((2, 4, 3), &CPU)?;
    let src: CpuTensor<f32> =
        Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (2, 2, 3), &CPU)?;

    dst.slice_set(&src, 1, 1)?;
    // Batch 0: [[0,0,0], [1,2,3], [4,5,6], [0,0,0]]
    // Batch 1: [[0,0,0], [7,8,9], [10,11,12], [0,0,0]]
    assert_eq!(
        dst.to_vec()?,
        vec![
            0., 0., 0., 1., 2., 3., 4., 5., 6., 0., 0., 0., // batch 0
            0., 0., 0., 7., 8., 9., 10., 11., 12., 0., 0., 0. // batch 1
        ]
    );
    Ok(())
}

#[test]
fn test_slice_set_at_start() -> Result<()> {
    // dst: [4, 2], src: [2, 2], set at offset 0
    let dst: CpuTensor<f32> = Tensor::full(9., (4, 2), &CPU)?;
    let src: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4.], (2, 2), &CPU)?;

    dst.slice_set(&src, 0, 0)?;
    assert_eq!(dst.to_vec()?, vec![1., 2., 3., 4., 9., 9., 9., 9.]);
    Ok(())
}

#[test]
fn test_slice_set_at_end() -> Result<()> {
    // dst: [4, 2], src: [2, 2], set at offset 2 (at the end)
    let dst: CpuTensor<f32> = Tensor::full(9., (4, 2), &CPU)?;
    let src: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4.], (2, 2), &CPU)?;

    dst.slice_set(&src, 0, 2)?;
    assert_eq!(dst.to_vec()?, vec![9., 9., 9., 9., 1., 2., 3., 4.]);
    Ok(())
}

#[test]
fn test_slice_set_1d() -> Result<()> {
    // dst: [8], src: [3], set at offset 2
    let dst: CpuTensor<f32> = Tensor::zeros((8,), &CPU)?;
    let src: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3.], (3,), &CPU)?;

    dst.slice_set(&src, 0, 2)?;
    assert_eq!(dst.to_vec()?, vec![0., 0., 1., 2., 3., 0., 0., 0.]);
    Ok(())
}
