use mimi::Tensor;

#[test]
fn test_cat_dim0() {
    // Two 2x3 tensors concatenated along dim 0 -> 4x3
    let a: Tensor<f32> = Tensor { data: vec![1., 2., 3., 4., 5., 6.], shape: (2, 3).into() };
    let b: Tensor<f32> = Tensor { data: vec![7., 8., 9., 10., 11., 12.], shape: (2, 3).into() };

    let c = Tensor::cat(&[&a, &b], 0).unwrap();
    assert_eq!(c.dims(), &[4, 3]);
    assert_eq!(c.data, vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);
}

#[test]
fn test_cat_dim1() {
    // Two 2x3 tensors concatenated along dim 1 -> 2x6
    let a: Tensor<f32> = Tensor { data: vec![1., 2., 3., 4., 5., 6.], shape: (2, 3).into() };
    let b: Tensor<f32> = Tensor { data: vec![7., 8., 9., 10., 11., 12.], shape: (2, 3).into() };

    let c = Tensor::cat(&[&a, &b], 1).unwrap();
    assert_eq!(c.dims(), &[2, 6]);
    // Row 0: [1,2,3] ++ [7,8,9] = [1,2,3,7,8,9]
    // Row 1: [4,5,6] ++ [10,11,12] = [4,5,6,10,11,12]
    assert_eq!(c.data, vec![1., 2., 3., 7., 8., 9., 4., 5., 6., 10., 11., 12.]);
}

#[test]
fn test_cat_3d_dim1() {
    // Two 2x2x3 tensors concatenated along dim 1 -> 2x4x3
    let a: Tensor<f32> =
        Tensor { data: (1..=12).map(|x| x as f32).collect(), shape: (2, 2, 3).into() };
    let b: Tensor<f32> =
        Tensor { data: (13..=24).map(|x| x as f32).collect(), shape: (2, 2, 3).into() };

    let c = Tensor::cat(&[&a, &b], 1).unwrap();
    assert_eq!(c.dims(), &[2, 4, 3]);
    // Batch 0: [[1,2,3],[4,5,6]] ++ [[13,14,15],[16,17,18]]
    // Batch 1: [[7,8,9],[10,11,12]] ++ [[19,20,21],[22,23,24]]
    assert_eq!(
        c.data,
        vec![
            1., 2., 3., 4., 5., 6., 13., 14., 15., 16., 17., 18., // batch 0
            7., 8., 9., 10., 11., 12., 19., 20., 21., 22., 23., 24. // batch 1
        ]
    );
}

#[test]
fn test_reshape() {
    let a: Tensor<f32> = Tensor { data: vec![1., 2., 3., 4., 5., 6.], shape: (2, 3).into() };

    // Reshape to 3x2
    let b = a.reshape((3, 2)).unwrap();
    assert_eq!(b.dims(), &[3, 2]);
    assert_eq!(b.data, vec![1., 2., 3., 4., 5., 6.]);

    // Reshape to 6
    let c = a.reshape((6,)).unwrap();
    assert_eq!(c.dims(), &[6]);

    // Reshape to 1x6
    let d = a.reshape((1, 6)).unwrap();
    assert_eq!(d.dims(), &[1, 6]);

    // Reshape to 1x2x3
    let e = a.reshape((1, 2, 3)).unwrap();
    assert_eq!(e.dims(), &[1, 2, 3]);
}

#[test]
fn test_reshape_with_hole() {
    let a: Tensor<f32> = Tensor { data: vec![1., 2., 3., 4., 5., 6.], shape: (2, 3).into() };

    // Reshape with inferred dimension
    let b = a.reshape((3, ())).unwrap();
    assert_eq!(b.dims(), &[3, 2]);

    let c = a.reshape(((), 2)).unwrap();
    assert_eq!(c.dims(), &[3, 2]);

    let d = a.reshape((1, (), 3)).unwrap();
    assert_eq!(d.dims(), &[1, 2, 3]);
}

#[test]
fn test_index_select_dim0() {
    // Select rows from a 4x3 tensor
    let a: Tensor<f32> = Tensor {
        data: vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        shape: (4, 3).into(),
    };

    // Select rows 0, 2, 3
    let b = a.index_select(&[0, 2, 3], 0).unwrap();
    assert_eq!(b.dims(), &[3, 3]);
    assert_eq!(b.data, vec![1., 2., 3., 7., 8., 9., 10., 11., 12.]);

    // Select with repetition
    let c = a.index_select(&[1, 1, 0], 0).unwrap();
    assert_eq!(c.dims(), &[3, 3]);
    assert_eq!(c.data, vec![4., 5., 6., 4., 5., 6., 1., 2., 3.]);
}

#[test]
fn test_index_select_dim1() {
    // Select columns from a 2x4 tensor
    let a: Tensor<f32> =
        Tensor { data: vec![1., 2., 3., 4., 5., 6., 7., 8.], shape: (2, 4).into() };

    // Select columns 0, 2
    let b = a.index_select(&[0, 2], 1).unwrap();
    assert_eq!(b.dims(), &[2, 2]);
    // Row 0: [1, 3], Row 1: [5, 7]
    assert_eq!(b.data, vec![1., 3., 5., 7.]);
}

#[test]
fn test_index_select_3d() {
    // 2x3x2 tensor
    let a: Tensor<f32> =
        Tensor { data: (1..=12).map(|x| x as f32).collect(), shape: (2, 3, 2).into() };
    // Data layout:
    // Batch 0: [[1,2], [3,4], [5,6]]
    // Batch 1: [[7,8], [9,10], [11,12]]

    // Select along dim 1 (middle dimension)
    let b = a.index_select(&[0, 2], 1).unwrap();
    assert_eq!(b.dims(), &[2, 2, 2]);
    // Batch 0: [[1,2], [5,6]]
    // Batch 1: [[7,8], [11,12]]
    assert_eq!(b.data, vec![1., 2., 5., 6., 7., 8., 11., 12.]);
}
