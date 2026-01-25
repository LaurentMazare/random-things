use mimi::{CpuTensor, Result, Tensor};

#[test]
fn test_cat_dim0() -> Result<()> {
    // Two 2x3 tensors concatenated along dim 0 -> 4x3
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), &())?;
    let b: CpuTensor<f32> = Tensor::from_vec(vec![7., 8., 9., 10., 11., 12.], (2, 3), &())?;

    let c = Tensor::cat(&[&a, &b], 0)?;
    assert_eq!(c.dims(), &[4, 3]);
    assert_eq!(c.to_vec()?, vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);
    Ok(())
}

#[test]
fn test_cat_dim1() -> Result<()> {
    // Two 2x3 tensors concatenated along dim 1 -> 2x6
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), &())?;
    let b: CpuTensor<f32> = Tensor::from_vec(vec![7., 8., 9., 10., 11., 12.], (2, 3), &())?;

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
    let a: CpuTensor<f32> = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (2, 2, 3), &())?;
    let b: CpuTensor<f32> =
        Tensor::from_vec((13..=24).map(|x| x as f32).collect(), (2, 2, 3), &())?;

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
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), &())?;

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
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), &())?;

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
        Tensor::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], (4, 3), &())?;

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
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.], (2, 4), &())?;

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
    let a: CpuTensor<f32> = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (2, 3, 2), &())?;
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
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), &())?;
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
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), &())?;
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
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), &())?;
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
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), &())?;
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
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), &())?;
    // Column-wise argmin:
    // col 0: argmin(1, 8, 9) = 0
    // col 1: argmin(5, 2, 0) = 2
    // col 2: argmin(3, 7, 1) = 2
    // col 3: argmin(4, 6, 2) = 2
    let b = a.argmin(0)?;
    assert_eq!(b.dims(), &[4]);
    assert_eq!(b.to_vec()?, vec![0., 2., 2., 2.]);
    Ok(())
}

#[test]
fn test_argmin_dim1() -> Result<()> {
    // 3x4 tensor, argmin along dim 1 -> 3
    let a: CpuTensor<f32> =
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), &())?;
    // Row-wise argmin:
    // row 0: argmin(1, 5, 3, 4) = 0
    // row 1: argmin(8, 2, 7, 6) = 1
    // row 2: argmin(9, 0, 1, 2) = 1
    let b = a.argmin(1)?;
    assert_eq!(b.dims(), &[3]);
    assert_eq!(b.to_vec()?, vec![0., 1., 1.]);
    Ok(())
}

#[test]
fn test_max_3d() -> Result<()> {
    // 2x3x2 tensor, max along dim 1 -> 2x2
    let a: CpuTensor<f32> = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (2, 3, 2), &())?;
    // Batch 0: [[1,2], [3,4], [5,6]] -> max along rows: [5, 6]
    // Batch 1: [[7,8], [9,10], [11,12]] -> max along rows: [11, 12]
    let b = a.max(1)?;
    assert_eq!(b.dims(), &[2, 2]);
    assert_eq!(b.to_vec()?, vec![5., 6., 11., 12.]);
    Ok(())
}

#[test]
fn test_broadcast_add_same_shape() -> Result<()> {
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4.], (2, 2), &())?;
    let b: CpuTensor<f32> = Tensor::from_vec(vec![10., 20., 30., 40.], (2, 2), &())?;
    let c = a.broadcast_add(&b)?;
    assert_eq!(c.dims(), &[2, 2]);
    assert_eq!(c.to_vec()?, vec![11., 22., 33., 44.]);
    Ok(())
}

#[test]
fn test_broadcast_add_1d_to_2d() -> Result<()> {
    // [2, 3] + [3] -> [2, 3]
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), &())?;
    let b: CpuTensor<f32> = Tensor::from_vec(vec![10., 20., 30.], (3,), &())?;
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
    let a: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), &())?;
    let b: CpuTensor<f32> = Tensor::from_vec(vec![2., 3.], (2, 1), &())?;
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
    let a: CpuTensor<f32> = Tensor::from_vec(vec![10., 20., 30., 40., 50., 60.], (2, 3), &())?;
    let b: CpuTensor<f32> = Tensor::from_vec(vec![1., 2., 3.], (3,), &())?;
    let c = a.broadcast_sub(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(c.to_vec()?, vec![9., 18., 27., 39., 48., 57.]);
    Ok(())
}

#[test]
fn test_broadcast_div() -> Result<()> {
    // [2, 3] / [2, 1] -> [2, 3]
    let a: CpuTensor<f32> = Tensor::from_vec(vec![2., 4., 6., 9., 12., 15.], (2, 3), &())?;
    let b: CpuTensor<f32> = Tensor::from_vec(vec![2., 3.], (2, 1), &())?;
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
    let a: CpuTensor<f32> = Tensor::from_vec((1..=24).map(|x| x as f32).collect(), (2, 3, 4), &())?;
    let b: CpuTensor<f32> = Tensor::from_vec(vec![100., 200., 300., 400.], (4,), &())?;
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
