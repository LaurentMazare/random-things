#[inline(always)]
fn pack_lhs<const MR: usize, const MC: usize>(
    lhs: *const f32,
    lhs_packed: *mut f32,
    kc: usize,
    m: usize,
) {
    #[inline(always)]
    fn pack_panel_lhs<const MR: usize>(
        lhs: *const f32,
        lhs_packed: *mut f32,
        mr: usize,
        kc: usize,
        m: usize,
    ) {
        unsafe {
            for p in 0..kc {
                for i_m in 0..mr {
                    *lhs_packed.add(p * MR + i_m) = *lhs.add(p * m + i_m);
                }
                for i_m in mr..MR {
                    *lhs_packed.add(p * MR + i_m) = 0.0;
                }
            }
        }
    }

    unsafe {
        for i_m in (0..MC).step_by(MR) {
            let mr = usize::min(MR, MC - i_m);
            pack_panel_lhs::<MR>(lhs.add(i_m), lhs_packed.add(i_m * kc), mr, kc, m);
        }
    }
}

#[inline(always)]
fn pack_rhs<const NR: usize, const NC: usize>(
    rhs: *const f32,
    rhs_packed: *mut f32,
    kc: usize,
    k: usize,
) {
    #[inline(always)]
    fn pack_panel_rhs<const NR: usize>(
        rhs: *const f32,
        rhs_packed: *mut f32,
        nr: usize,
        kc: usize,
        k: usize,
    ) {
        unsafe {
            for p in 0..kc {
                for i_n in 0..nr {
                    *rhs_packed.add(p * NR + i_n) = *rhs.add(i_n * k + p);
                }
                for i_n in nr..NR {
                    *rhs_packed.add(p * NR + i_n) = 0.0;
                }
            }
        }
    }

    unsafe {
        for i_n in (0..NC).step_by(NR) {
            let nr = usize::min(NR, NC - i_n);
            pack_panel_rhs::<NR>(rhs.add(i_n * k), rhs_packed.add(i_n * kc), nr, kc, k);
        }
    }
}

#[inline(always)]
fn kernel_16x6(
    lhs: *const f32,
    rhs: *const f32,
    dst: *mut f32,
    m: usize,
    _n: usize,
    k: usize,
    nc: usize,
) {
    use core::arch::aarch64::*;
    unsafe {
        let mut c_accum: [[float32x4_t; 4]; 6] = [[vdupq_n_f32(0f32); 4]; 6];
        for p in 0..k {
            let a0 = vld1q_f32(lhs.add(p * m));
            let a1 = vld1q_f32(lhs.add(p * m + 4));
            let a2 = vld1q_f32(lhs.add(p * m + 8));
            let a3 = vld1q_f32(lhs.add(p * m + 12));
            for i_n in 0..nc {
                let b = vdupq_n_f32(*rhs.add(i_n * k + p));
                c_accum[i_n][0] = vfmaq_f32(c_accum[i_n][0], a0, b);
                c_accum[i_n][1] = vfmaq_f32(c_accum[i_n][1], a1, b);
                c_accum[i_n][2] = vfmaq_f32(c_accum[i_n][2], a2, b);
                c_accum[i_n][3] = vfmaq_f32(c_accum[i_n][3], a3, b);
            }
        }
        for i_n in 0..nc {
            vst1q_f32(dst.add(i_n * m), c_accum[i_n][0]);
            vst1q_f32(dst.add(i_n * m + 4), c_accum[i_n][1]);
            vst1q_f32(dst.add(i_n * m + 8), c_accum[i_n][2]);
            vst1q_f32(dst.add(i_n * m + 12), c_accum[i_n][3]);
        }
    }
}

#[inline(never)]
fn mm_kernel(lhs: &[f32], rhs: &[f32], dst: &mut [f32], m: usize, n: usize, k: usize) {
    if lhs.len() < m * k {
        panic!("lhs is too small: {} < {m} {k}", lhs.len())
    }
    if rhs.len() < n * k {
        panic!("rhs is too small: {} < {n} {k}", rhs.len())
    }
    if dst.len() < m * n {
        panic!("dst is too small: {} < {m} {n}", dst.len())
    }
    let lhs = lhs.as_ptr();
    let rhs = rhs.as_ptr();
    let dst = dst.as_mut_ptr();
    for i_m in (0..m).step_by(16) {
        for i_n in (0..n).step_by(6) {
            let nc = usize::min(6, n - i_n);
            unsafe {
                kernel_16x6(
                    lhs.add(i_m),
                    rhs.add(i_n * k),
                    dst.add(i_n * m + i_m),
                    m,
                    n,
                    k,
                    nc,
                );
            }
        }
    }
}

#[inline(always)]
fn kernel16_acc<const M: usize, const N: usize>(
    lhs: *const f32,
    rhs: *const f32,
    dst: *mut f32,
    nr: usize,
    kr: usize,
    m: usize,
) {
    use core::arch::aarch64::*;
    unsafe {
        // TODO: Initializing to zero here is not necessary.
        let mut c_accum: [[float32x4_t; 4]; N] = [[vdupq_n_f32(0.); 4]; N];
        // laurent: I tried unrolling this loop via a const generic and a switch
        // based on nr for values 1 to 6 but it did not improve performance
        for i_n in 0..nr {
            c_accum[i_n][0] = vld1q_f32(dst.add(i_n * m));
            c_accum[i_n][1] = vld1q_f32(dst.add(i_n * m + 4));
            c_accum[i_n][2] = vld1q_f32(dst.add(i_n * m + 8));
            c_accum[i_n][3] = vld1q_f32(dst.add(i_n * m + 12));
        }
        for p in 0..kr {
            let a0 = vld1q_f32(lhs.add(p * M));
            let a1 = vld1q_f32(lhs.add(p * M + 4));
            let a2 = vld1q_f32(lhs.add(p * M + 8));
            let a3 = vld1q_f32(lhs.add(p * M + 12));
            for i_n in 0..nr {
                let b = vdupq_n_f32(*rhs.add(i_n + p * 6));
                c_accum[i_n][0] = vfmaq_f32(c_accum[i_n][0], a0, b);
                c_accum[i_n][1] = vfmaq_f32(c_accum[i_n][1], a1, b);
                c_accum[i_n][2] = vfmaq_f32(c_accum[i_n][2], a2, b);
                c_accum[i_n][3] = vfmaq_f32(c_accum[i_n][3], a3, b);
            }
        }
        for i_n in 0..nr {
            vst1q_f32(dst.add(i_n * m), c_accum[i_n][0]);
            vst1q_f32(dst.add(i_n * m + 4), c_accum[i_n][1]);
            vst1q_f32(dst.add(i_n * m + 8), c_accum[i_n][2]);
            vst1q_f32(dst.add(i_n * m + 12), c_accum[i_n][3]);
        }
    }
}

// The inline never here results in a speed-up on large matrixes which don't
// need masked access (e.g. on 2048x2048x2048 this goes from 73ms to 70ms).
// Probably because the code starts to fill the L1 instruction cache?
#[inline(never)]
fn kernel16_acc_masked<const M: usize, const N: usize>(
    lhs: *const f32,
    rhs: *const f32,
    dst: *mut f32,
    mr: usize,
    nr: usize,
    kr: usize,
    m: usize,
) {
    use core::arch::aarch64::*;
    unsafe {
        // TODO: It would be better to have proper masked load here, similar to
        // _mm256_maskload_ps on the x86 version.
        // nvalues can be higher than 4 in which case only 4 values are loaded.
        #[inline(always)]
        fn vld1q_f32_masked(src: *const f32, nvalues: usize) -> float32x4_t {
            unsafe {
                match nvalues {
                    0 => vdupq_n_f32(0f32),
                    1 => {
                        let v = vdupq_n_f32(0f32);
                        vsetq_lane_f32::<0>(*src, v)
                    }
                    2 => {
                        let v = vdupq_n_f32(0f32);
                        let v = vsetq_lane_f32::<0>(*src, v);
                        vsetq_lane_f32::<1>(*src.add(1), v)
                    }
                    3 => {
                        let v = vdupq_n_f32(0f32);
                        let v = vsetq_lane_f32::<0>(*src, v);
                        let v = vsetq_lane_f32::<1>(*src.add(1), v);
                        vsetq_lane_f32::<2>(*src.add(2), v)
                    }
                    4.. => vld1q_f32(src),
                }
            }
        }

        #[inline(always)]
        fn vst1q_f32_masked(dst: *mut f32, src: float32x4_t, nvalues: usize) {
            unsafe {
                match nvalues {
                    0 => {}
                    1 => *dst = vgetq_lane_f32(src, 0),
                    2 => vst1_f32(dst, vget_low_f32(src)),
                    3 => {
                        vst1_f32(dst, vget_low_f32(src));
                        *dst.add(2) = vgetq_lane_f32(src, 2);
                    }
                    4.. => vst1q_f32(dst, src),
                }
            }
        }

        // TODO: Initializing to zero here is not necessary.
        let mut c_accum: [[float32x4_t; 4]; N] = [[vdupq_n_f32(0.); 4]; N];
        // laurent: I tried unrolling this loop via a const generic and a switch
        // based on nr for values 1 to 6 but it did not improve performance
        for i_n in 0..nr {
            c_accum[i_n][0] = vld1q_f32_masked(dst.add(i_n * m), mr);
            c_accum[i_n][1] = vld1q_f32_masked(dst.add(i_n * m + 4), mr.saturating_sub(4));
            c_accum[i_n][2] = vld1q_f32_masked(dst.add(i_n * m + 8), mr.saturating_sub(8));
            c_accum[i_n][3] = vld1q_f32_masked(dst.add(i_n * m + 12), mr.saturating_sub(12));
        }
        for p in 0..kr {
            let a0 = vld1q_f32(lhs.add(p * M));
            let a1 = vld1q_f32(lhs.add(p * M + 4));
            let a2 = vld1q_f32(lhs.add(p * M + 8));
            let a3 = vld1q_f32(lhs.add(p * M + 12));
            for i_n in 0..nr {
                let b = vdupq_n_f32(*rhs.add(i_n + p * 6));
                c_accum[i_n][0] = vfmaq_f32(c_accum[i_n][0], a0, b);
                c_accum[i_n][1] = vfmaq_f32(c_accum[i_n][1], a1, b);
                c_accum[i_n][2] = vfmaq_f32(c_accum[i_n][2], a2, b);
                c_accum[i_n][3] = vfmaq_f32(c_accum[i_n][3], a3, b);
            }
        }
        for i_n in 0..nr {
            vst1q_f32_masked(dst.add(i_n * m), c_accum[i_n][0], mr);
            vst1q_f32_masked(dst.add(i_n * m + 4), c_accum[i_n][1], mr.saturating_sub(4));
            vst1q_f32_masked(dst.add(i_n * m + 8), c_accum[i_n][2], mr.saturating_sub(8));
            vst1q_f32_masked(
                dst.add(i_n * m + 12),
                c_accum[i_n][3],
                mr.saturating_sub(12),
            );
        }
    }
}

#[inline(never)]
fn mm_local<const MC: usize, const NC: usize, const KC: usize, const MR: usize, const NR: usize>(
    lhs: &[f32],
    rhs: &[f32],
    dst: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    dst.fill(0.0);
    if lhs.len() < m * k {
        panic!("lhs is too small: {} < {m} {k}", lhs.len())
    }
    if rhs.len() < n * k {
        panic!("rhs is too small: {} < {n} {k}", rhs.len())
    }
    if dst.len() < m * n {
        panic!("dst is too small: {} < {m} {n}", dst.len())
    }
    let lhs = lhs.as_ptr();
    let rhs = rhs.as_ptr();
    let dst = dst.as_mut_ptr();

    // TODO: these should be aligned to the simd width.
    let mut lhs_packed = vec![0f32; MC * KC];
    let mut rhs_packed = vec![0f32; NC * KC];
    let lhs_packed = lhs_packed.as_mut_ptr();
    let rhs_packed = rhs_packed.as_mut_ptr();
    unsafe {
        for i_n in (0..n).step_by(NC) {
            let nc = usize::min(NC, n - i_n);
            for p in (0..k).step_by(KC) {
                let kc = usize::min(KC, k - p);
                pack_rhs::<NR, NC>(rhs.add(i_n * k + p), rhs_packed, kc, k);
                for i_m in (0..m).step_by(MC) {
                    let mc = usize::min(MC, m - i_m);
                    pack_lhs::<MR, MC>(lhs.add(p * m + i_m), lhs_packed, kc, m);
                    for jr in (0..nc).step_by(NR) {
                        for ir in (0..mc).step_by(MR) {
                            let mr = usize::min(MR, mc - ir);
                            let nr = usize::min(NR, nc - jr);
                            if mr == MR {
                                kernel16_acc::<MR, NR>(
                                    lhs_packed.add(ir * kc),
                                    rhs_packed.add(jr * kc),
                                    dst.add((i_n + jr) * m + (i_m + ir)),
                                    nr,
                                    kc,
                                    m,
                                );
                            } else {
                                kernel16_acc_masked::<MR, NR>(
                                    lhs_packed.add(ir * kc),
                                    rhs_packed.add(jr * kc),
                                    dst.add((i_n + jr) * m + (i_m + ir)),
                                    mr,
                                    nr,
                                    kc,
                                    m,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

fn dummy_mm_t(lhs: &[f32], rhs: &[f32], dst: &mut [f32], m: usize, n: usize, k: usize) {
    for i_m in 0..m {
        for i_n in 0..n {
            let mut v = 0f32;
            for _k in 0..k {
                v += lhs[m * _k + i_m] * rhs[i_n * k + _k];
            }
            dst[i_n * m + i_m] = v;
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Algo {
    Naive,
    Local,
    Kernel,
}

impl Algo {
    pub fn mm(&self, lhs: &[f32], rhs: &[f32], dst: &mut [f32], m: usize, n: usize, k: usize) {
        match self {
            Algo::Naive => dummy_mm_t(lhs, rhs, dst, m, n, k),
            Algo::Kernel => {
                mm_kernel(lhs, rhs, dst, m, n, k);
            }
            Algo::Local => {
                const MR: usize = 16;
                const NR: usize = 6;
                const MC: usize = MR * 32;
                const NC: usize = NR * 256;
                const KC: usize = 512;

                mm_local::<MC, NC, KC, MR, NR>(lhs, rhs, dst, m, n, k);
            }
        }
    }
}
