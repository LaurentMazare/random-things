#![allow(clippy::identity_op)]
#![allow(clippy::erasing_op)]
// Flags for compiler-explorer: --C target-cpu=aarch64 -C opt-level=3 -C target-feature=+neon
use core::arch::aarch64::*;

mod local;

macro_rules! test_simd {
    ($N:expr) => {
    let vs = [0f32, 1.0, 2.0, 3.0];
    unsafe {
        let mut v = [vld1q_f32(vs.as_ptr()); $N];
        let iters = 1_000_000_000;
        let a = vdupq_n_f32(0.02);
        let start_time = std::time::Instant::now();
        for _ in 1..iters {
            // v = vfmaq_f32(b, v, a);
            seq_macro::seq!(I in 0..$N {
            v[I] = vfmaq_laneq_f32(v[I], a, a, (I % 4) as i32);
            });
        }
        let _ = std::hint::black_box(v);
        let iters = $N * iters;
        let dt = start_time.elapsed().as_secs_f64() / (iters as f64) * 1e9;
        println!("{} {:.2}giter/s {dt}ns {}GFLOPS", $N, 1.0 / dt, 4.0 * 2.0 / dt);
    }
}
}

fn dummy_mm_t(lhs: &[f32], rhs: &[f32], dst: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut v = 0f32;
            for _k in 0..k {
                v += lhs[i * k + _k] * rhs[j * k + _k];
            }
            dst[i * n + j] = v;
        }
    }
}

fn dummy_mm_tt(lhs: &[f32], rhs: &[f32], dst: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut v = 0f32;
            for _k in 0..k {
                v += lhs[_k * m + i] * rhs[_k * n + j];
            }
            dst[i * n + j] = v;
        }
    }
}

#[inline(always)]
fn sum_f32x4(v: float32x4_t) -> f32 {
    unsafe {
        let r = vadd_f32(vget_high_f32(v), vget_low_f32(v));
        vget_lane_f32(vpadd_f32(r, r), 0)
    }
}

fn simd_mm_t(lhs: &[f32], rhs: &[f32], dst: &mut [f32], m: usize, n: usize, k: usize) {
    unsafe {
        let end_k = k / 4 * 4;
        for i in 0..m {
            for j in 0..n {
                let mut acc = vdupq_n_f32(0f32);
                for _k in (0..k).step_by(4) {
                    let lhs = vld1q_f32(lhs.as_ptr().wrapping_add(i * k + _k));
                    let rhs = vld1q_f32(rhs.as_ptr().wrapping_add(j * k + _k));
                    acc = vfmaq_f32(acc, lhs, rhs)
                }
                let mut v = sum_f32x4(acc);
                for _k in end_k..k {
                    v += lhs[i * k + _k] * rhs[j * k + _k];
                }
                dst[i * n + j] = v;
            }
        }
    }
}

fn vec_align(l: usize) -> *mut f32 {
    use std::alloc::{alloc, Layout};
    let layout = Layout::from_size_align(l * 4, 16).unwrap();
    let ptr = unsafe { alloc(layout) as *mut f32 };
    ptr
}

fn tiled_simd_mm_t<const BM: usize, const BN: usize, const BK: usize>(
    lhs: &[f32],
    rhs: &[f32],
    dst: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    let tmp_l = vec_align(BM * BK);
    let tmp_r = vec_align(BK * BN);
    let tmp_d = vec_align(BM * BN);
    let tmp_l = unsafe { std::slice::from_raw_parts_mut(tmp_l, BM * BK) };
    let tmp_r = unsafe { std::slice::from_raw_parts_mut(tmp_r, BN * BK) };
    let tmp_d = unsafe { std::slice::from_raw_parts_mut(tmp_d, BN * BM) };
    for b_m in (0..m).step_by(BM) {
        for b_n in (0..n).step_by(BN) {
            tmp_d.fill(0f32);
            for b_k in (0..k).step_by(BK) {
                for i_m in 0..BM {
                    let ii_m = (b_m + i_m) * k + b_k;
                    tmp_l[i_m * BK..(i_m + 1) * BK].copy_from_slice(&lhs[ii_m..(ii_m + BK)])
                }
                for i_n in 0..BN {
                    let ii_n = (b_n + i_n) * k + b_k;
                    tmp_r[i_n * BK..(i_n + 1) * BK].copy_from_slice(&rhs[ii_n..(ii_n + BK)])
                }
                // block_mm
                for i_m in 0..BM {
                    for i_n in 0..BN {
                        unsafe {
                            let v = {
                                let mut acc = vdupq_n_f32(0f32);
                                for i_k in (0..BK).step_by(4) {
                                    let l = vld1q_f32(tmp_l.as_ptr().wrapping_add(i_m * BK + i_k));
                                    let r = vld1q_f32(tmp_r.as_ptr().wrapping_add(i_n * BK + i_k));
                                    acc = vfmaq_f32(acc, l, r)
                                }
                                sum_f32x4(acc)
                            };
                            *tmp_d.get_unchecked_mut(i_m * BN + i_n) += v;
                        }
                    }
                }
            }
            // copy back tmp_d
            for i_m in 0..BM {
                let ii = (b_m + i_m) * n + b_n;
                dst[ii..ii + BN].copy_from_slice(&tmp_d[(i_m * BN)..(i_m * BN + BN)])
            }
        }
    }
}

unsafe fn ld1(ptr: *const f32) -> float32x4_t {
    unsafe { vld1q_f32(ptr) }
}

unsafe fn _ld1(ptr: *const f32) -> float32x4_t {
    let result: float32x4_t;
    unsafe {
        core::arch::asm!(
            "ld1 {{v0.4s}}, [{ptr}]",
            ptr = in(reg) ptr,
            out("v0") result,
            options(pure, nomem, nostack)
        );
        result
    }
}

#[inline(always)]
pub fn micro_4x4(lhs: *const f32, rhs: *const f32, dst: *mut f32, k: usize, dst_stride: usize) {
    unsafe {
        let mut accs = (
            ld1(dst.add(0 * dst_stride)),
            ld1(dst.add(1 * dst_stride)),
            ld1(dst.add(2 * dst_stride)),
            ld1(dst.add(3 * dst_stride)),
        );
        let mut accs2 = (
            ld1(dst.add(0 * dst_stride + 4)),
            ld1(dst.add(1 * dst_stride + 4)),
            ld1(dst.add(2 * dst_stride + 4)),
            ld1(dst.add(3 * dst_stride + 4)),
        );
        let mut accs3 = (
            ld1(dst.add(0 * dst_stride + 8)),
            ld1(dst.add(1 * dst_stride + 8)),
            ld1(dst.add(2 * dst_stride + 8)),
            ld1(dst.add(3 * dst_stride + 8)),
        );
        let mut accs4 = (
            ld1(dst.add(0 * dst_stride + 12)),
            ld1(dst.add(1 * dst_stride + 12)),
            ld1(dst.add(2 * dst_stride + 12)),
            ld1(dst.add(3 * dst_stride + 12)),
        );
        for i_k in (0..k).step_by(4) {
            seq_macro::seq!(I_K in 0..4 {
            // This should not be dst_stride but rather the actual lhs/rhs strides
            {
                let i = i_k + I_K;
                let l = ld1(lhs.add(i * dst_stride));
                let r = ld1(rhs.add(i * dst_stride));
                let r2 = ld1(rhs.add(i * dst_stride + 4));
                let r3 = ld1(rhs.add(i * dst_stride + 8));
                let r4 = ld1(rhs.add(i * dst_stride + 12));
                seq_macro::seq!(I in 0..4 {
                    accs.I = vfmaq_laneq_f32(accs.I, r, l, I);
                });
                seq_macro::seq!(I in 0..4 {
                    accs2.I = vfmaq_laneq_f32(accs2.I, r2, l, I);
                });
                seq_macro::seq!(I in 0..4 {
                    accs3.I = vfmaq_laneq_f32(accs3.I, r3, l, I);
                });
                seq_macro::seq!(I in 0..4 {
                    accs4.I = vfmaq_laneq_f32(accs4.I, r4, l, I);
                });
                // std::hint::black_box((accs, accs2));
            }
            });
        }
        seq_macro::seq!(I in 0..4 {
        vst1q_f32(dst.add(I * dst_stride), accs.I);
        vst1q_f32(dst.add(I * dst_stride + 4), accs2.I);
        vst1q_f32(dst.add(I * dst_stride + 8), accs3.I);
        vst1q_f32(dst.add(I * dst_stride + 12), accs4.I);
        });
    }
}

// lhs shape (m, k), stride (1, m)
// rhs shape (k, n), stride (n, 1)
// dst shape (m, n), stride (n, 1)
#[inline(never)]
fn micro_simd_mm_t<const BM: usize, const BN: usize, const BK: usize>(
    lhs: &[f32],
    rhs: &[f32],
    dst: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    let tmp_l = vec_align(BM * BK);
    let tmp_r = vec_align(BK * BN);
    let tmp_d = vec_align(BM * BN);
    let tmp_l = unsafe { std::slice::from_raw_parts_mut(tmp_l, BM * BK) };
    let tmp_r = unsafe { std::slice::from_raw_parts_mut(tmp_r, BN * BK) };
    let tmp_d = unsafe { std::slice::from_raw_parts_mut(tmp_d, BN * BM) };
    for b_m in (0..m).step_by(BM) {
        for b_k in (0..k).step_by(BK) {
            for i_k in 0..BK {
                let ii_k = (b_k + i_k) * m + b_m;
                tmp_l[i_k * BM..(i_k + 1) * BM].copy_from_slice(&lhs[ii_k..(ii_k + BM)]);
            }
            for b_n in (0..n).step_by(BN) {
                tmp_d.fill(0f32);
                for i_k in 0..BK {
                    let ii_k = (b_k + i_k) * n + b_n;
                    tmp_r[i_k * BN..(i_k + 1) * BN].copy_from_slice(&rhs[ii_k..(ii_k + BN)])
                }
                let tmp_l = tmp_l.as_ptr();
                let tmp_r = tmp_r.as_ptr();
                // block_mm
                for i_m in (0..BM).step_by(4) {
                    let tmp_d = tmp_d.as_mut_ptr();
                    for i_n in (0..BN).step_by(16) {
                        unsafe {
                            let tmp_l = tmp_l.add(i_m);
                            let tmp_r = tmp_r.add(i_n);
                            let tmp_d = tmp_d.add(i_m * BN + i_n);
                            micro_4x4(tmp_l, tmp_r, tmp_d, BK, BN);
                        }
                    }
                }
                // copy back tmp_d
                for i_m in 0..BM {
                    let ii = (b_m + i_m) * n + b_n;
                    dst[ii..ii + BN].copy_from_slice(&tmp_d[(i_m * BN)..(i_m * BN + BN)])
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Algo {
    Naive,
    NaiveT,
    Simd,
    TiledSimd,
    MicroSimd,
    Local(local::Algo),
}

impl Algo {
    fn mm(&self, lhs: &[f32], rhs: &[f32], dst: &mut [f32], m: usize, n: usize, k: usize) {
        match self {
            Algo::Naive => dummy_mm_t(lhs, rhs, dst, m, n, k),
            Algo::NaiveT => dummy_mm_tt(lhs, rhs, dst, m, n, k),
            Algo::Simd => simd_mm_t(lhs, rhs, dst, m, n, k),
            Algo::TiledSimd => tiled_simd_mm_t::<32, 32, 32>(lhs, rhs, dst, m, n, k),
            Algo::MicroSimd => micro_simd_mm_t::<32, 32, 32>(lhs, rhs, dst, m, n, k),
            Algo::Local(l) => l.mm(lhs, rhs, dst, m, n, k),
        }
    }

    fn run(&self, lhs: &[f32], rhs: &[f32], dst: &mut [f32], m: usize, n: usize, k: usize) {
        let start_time = std::time::Instant::now();
        self.mm(lhs, rhs, dst, m, n, k);
        let dt = start_time.elapsed().as_secs_f64();
        let s: f32 = dst.iter().sum();
        let gflops = (m * n * k * 2) as f64 / dt * 1e-9;
        println!("{self:?} sum {s}, {:.2}ms, {gflops:.2}GFLOPS", dt * 1000.0);
    }
}

#[inline(never)]
fn test_simd16() {
    test_simd!(16usize);
}

fn main() {
    use rand::{Rng, SeedableRng};

    let mut rng = rand::rngs::StdRng::seed_from_u64(42424242);
    let (m, n, k) = (1536, 1536, 1536);
    // let (m, n, k) = (2, 1, 9);
    let lhs: Vec<f32> = (0..(m * k)).map(|_| rng.random::<f32>() - 0.5).collect();
    let rhs: Vec<f32> = (0..(n * k)).map(|_| rng.random::<f32>() - 0.5).collect();
    let mut dst = vec![0f32; m * n];

    if false {
        test_simd16();
        test_simd16();
        test_simd16();
        test_simd16();
    }
    if false {
        test_simd!(16usize);
        test_simd!(24usize);
        test_simd!(4usize);
        test_simd!(8usize);
        test_simd!(12usize);
        test_simd!(16usize);
        test_simd!(20usize);
        test_simd!(24usize);
        test_simd!(32usize);
    }
    if false {
        Algo::Naive.run(&lhs, &rhs, &mut dst, m, n, k);
        Algo::Simd.run(&lhs, &rhs, &mut dst, m, n, k);
        Algo::TiledSimd.run(&lhs, &rhs, &mut dst, m, n, k);
        Algo::MicroSimd.run(&lhs, &rhs, &mut dst, m, n, k);
    }
    for _ in 0..0 {
        Algo::MicroSimd.run(&lhs, &rhs, &mut dst, m, n, k);
    }
    for _ in 0..20 {
        Algo::Local(local::Algo::Local).run(&lhs, &rhs, &mut dst, m, n, k);
    }
    Algo::Local(local::Algo::Kernel).run(&lhs, &rhs, &mut dst, m, n, k);
    Algo::Local(local::Algo::Naive).run(&lhs, &rhs, &mut dst, m, n, k);
    if false {
        Algo::NaiveT.run(&lhs, &rhs, &mut dst, m, n, k);
    }
}
