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
    use std::arch::x86_64::*;
    unsafe {
        let mut c_accum: [[__m256; 2]; 6] = [[_mm256_setzero_ps(); 2]; 6];
        for p in 0..k {
            let a0 = _mm256_loadu_ps(lhs.add(p * m));
            let a1 = _mm256_loadu_ps(lhs.add(p * m + 8));
            for i_n in 0..nc {
                let b = _mm256_broadcast_ss(&*rhs.add(i_n * k + p));
                c_accum[i_n][0] = _mm256_fmadd_ps(a0, b, c_accum[i_n][0]);
                c_accum[i_n][1] = _mm256_fmadd_ps(a1, b, c_accum[i_n][1]);
            }
        }
        for i_n in 0..nc {
            _mm256_storeu_ps(dst.add(i_n * m), c_accum[i_n][0]);
            _mm256_storeu_ps(dst.add(i_n * m + 8), c_accum[i_n][1]);
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
fn kernel16_acc<const N: usize>(
    lhs: *const f32,
    rhs: *const f32,
    dst: *mut f32,
    mr: usize,
    nr: usize,
    kr: usize,
    m: usize,
) {
    use std::arch::x86_64::*;
    unsafe {
        // TODO: Initializing to zero here is not necessary.
        let mut c_accum: [[__m256; 2]; N] = [[_mm256_setzero_ps(); 2]; N];
        // laurent: I tried unrolling this loop via a const generic and a switch
        // based on nr for values 1 to 6 but it did not improve performance
        for i_n in 0..nr {
            c_accum[i_n][0] = _mm256_loadu_ps(dst.add(i_n * m));
            c_accum[i_n][1] = _mm256_loadu_ps(dst.add(i_n * m + 8));
        }
        for p in 0..kr {
            let a0 = _mm256_loadu_ps(lhs.add(p * mr));
            let a1 = _mm256_loadu_ps(lhs.add(p * mr + 8));
            for i_n in 0..nr {
                let b = _mm256_broadcast_ss(&*rhs.add(i_n + p * 6));
                c_accum[i_n][0] = _mm256_fmadd_ps(a0, b, c_accum[i_n][0]);
                c_accum[i_n][1] = _mm256_fmadd_ps(a1, b, c_accum[i_n][1]);
            }
        }
        for i_n in 0..nr {
            _mm256_storeu_ps(dst.add(i_n * m), c_accum[i_n][0]);
            _mm256_storeu_ps(dst.add(i_n * m + 8), c_accum[i_n][1]);
        }
    }
}

#[inline(always)]
fn kernel16_acc_masked<const N: usize>(
    lhs: *const f32,
    rhs: *const f32,
    dst: *mut f32,
    mr: usize,
    nr: usize,
    kr: usize,
    m: usize,
) {
    use std::arch::x86_64::*;

    fn mm256_mask(mr: usize) -> __m256i {
        unsafe {
            match mr {
                0 => _mm256_setzero_si256(),
                1 => _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1),
                2 => _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1),
                3 => _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1),
                4 => _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1),
                5 => _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1),
                6 => _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1),
                7 => _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1),
                8.. => _mm256_set1_epi32(-1),
            }
        }
    }

    unsafe {
        let mask0 = mm256_mask(mr);
        let mask1 = mm256_mask(mr.saturating_sub(8));
        // TODO: Initializing to zero here is not necessary.
        let mut c_accum: [[__m256; 2]; N] = [[_mm256_setzero_ps(); 2]; N];
        // laurent: I tried unrolling this loop via a const generic and a switch
        // based on nr for values 1 to 6 but it did not improve performance
        for i_n in 0..nr {
            c_accum[i_n][0] = _mm256_maskload_ps(dst.add(i_n * m), mask0);
            c_accum[i_n][1] = _mm256_maskload_ps(dst.add(i_n * m + 8), mask1);
        }
        for p in 0..kr {
            let a0 = _mm256_loadu_ps(lhs.add(p * mr));
            let a1 = _mm256_loadu_ps(lhs.add(p * mr + 8));
            for i_n in 0..nr {
                let b = _mm256_broadcast_ss(&*rhs.add(i_n + p * 6));
                c_accum[i_n][0] = _mm256_fmadd_ps(a0, b, c_accum[i_n][0]);
                c_accum[i_n][1] = _mm256_fmadd_ps(a1, b, c_accum[i_n][1]);
            }
        }
        for i_n in 0..nr {
            _mm256_maskstore_ps(dst.add(i_n * m), mask0, c_accum[i_n][0]);
            _mm256_maskstore_ps(dst.add(i_n * m + 8), mask1, c_accum[i_n][1]);
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
                                kernel16_acc::<NR>(
                                    lhs_packed.add(ir * kc),
                                    rhs_packed.add(jr * kc),
                                    dst.add((i_n + jr) * m + (i_m + ir)),
                                    mr,
                                    nr,
                                    kc,
                                    m,
                                );
                            } else {
                                kernel16_acc_masked::<NR>(
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
enum Algo {
    Naive,
    Local,
    Kernel,
}

impl Algo {
    fn mm(&self, lhs: &[f32], rhs: &[f32], dst: &mut [f32], m: usize, n: usize, k: usize) {
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

    fn run(&self, lhs: &[f32], rhs: &[f32], dst: &mut [f32], m: usize, n: usize, k: usize) {
        let start_time = std::time::Instant::now();
        self.mm(lhs, rhs, dst, m, n, k);
        let dt = start_time.elapsed().as_secs_f64();
        let s: f32 = dst.iter().sum();
        let gflops = (m * n * k * 2) as f64 / dt * 1e-9;
        println!("{self:?} sum {s}, {:.2}ms, {gflops:.2}GFLOPS", dt * 1000.0);
    }
}

fn main() {
    use rand::{Rng, SeedableRng};

    let mut rng = rand::rngs::StdRng::seed_from_u64(42424242);
    let (m, n, k) = (2048, 2048, 2048);
    // let (m, n, k) = (1, 2, 2);
    let lhs: Vec<f32> = (0..(m * k)).map(|_| rng.random::<f32>() - 0.5).collect();
    let rhs: Vec<f32> = (0..(n * k)).map(|_| rng.random::<f32>() - 0.5).collect();
    let mut dst = vec![0f32; m * n];

    for _ in 0..20 {
        Algo::Local.run(&lhs, &rhs, &mut dst, m, n, k);
    }
    Algo::Kernel.run(&lhs, &rhs, &mut dst, m, n, k);
    if true {
        Algo::Naive.run(&lhs, &rhs, &mut dst, m, n, k);
    }
}
