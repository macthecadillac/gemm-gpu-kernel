use std::time::Instant;
use rayon::prelude::*;

const TILESIZE: usize = 128;

#[derive(Debug)]
struct Matrix {
    nrow: usize,
    ncol: usize,
    data: Vec<f64>
}

impl Matrix {
    fn index(
        &self,
        i_start: usize,
        i_end: usize,
        j_start: usize,
        j_end: usize
    ) -> Strides {
        assert!(i_end <= self.nrow);
        assert!(j_end <= self.ncol);
        let m = i_end - i_start;
        let n = j_end - j_start;
        Strides { m, n, i_start, i_end, j_start, j_end }
    }
}

#[derive(Copy, Clone)]
#[allow(dead_code)]
struct Strides {
    m: usize,
    n: usize,
    i_start: usize,
    i_end: usize,
    j_start: usize,
    j_end: usize
}

impl Strides {
    fn quadrants(&self) -> [Strides; 4] {
        // break x into tiles
        let nrow = self.i_end - self.i_start;
        let ncol = self.j_end - self.j_start;
        let padded_nrow = nrow + nrow % 2;
        let padded_ncol = ncol + ncol % 2;
        let nrow_ = padded_nrow / 2;
        let ncol_ = padded_ncol / 2;
        let m1 = Strides {
            i_end: self.i_start + nrow_,
            j_end: self.j_start + ncol_,
            ..*self
        };
        let m2 = Strides {
            i_end: self.i_start + nrow_,
            j_start: self.j_start + ncol_,
            ..*self
        };
        let m3 = Strides {
            i_start: self.i_start + nrow_,
            j_end: self.j_start + ncol_,
            ..*self
        };
        let m4 = Strides {
            i_start: self.i_start + nrow_,
            j_start: self.j_start + ncol_,
            ..*self
        };
        [m1, m2, m3, m4]
    }
}

#[inline]
fn kernel<'a>(
    ai: Strides,
    bi: Strides,
    ci: Strides,
    a: &'a Matrix,
    b: &'a Matrix,
    c: &'a mut Matrix,
    alpha: f64,
) {
    //     Your current order is effectively:
    //
    // for i {
    //     for k {
    //         for j {
    //             c[i][j] += a[i][k] * b[k][j];
    //         }
    //     }
    // }
    //
    // Register blocking uses:
    //
    // for ii in (0..m).step_by(MR) {
    //     for jj in (0..n).step_by(NR) {
    //         // Load a small C tile into SIMD accumulators.
    //
    //         for k in 0..l {
    //             // Load B[k, jj..jj+NR].
    //             // Broadcast A[ii+r, k].
    //             // FMA into every accumulator.
    //         }
    //
    //         // Store the C tile once.
    //     }
    // }
    //
    // For NEON f64, one vector holds two doubles. A simple 2×4 microkernel would look conceptually like:
    //
    // // Two rows, four columns: four f64x2 accumulator registers.
    // let mut c00 = load(&c[row0][col..col + 2]);
    // let mut c01 = load(&c[row0][col + 2..col + 4]);
    // let mut c10 = load(&c[row1][col..col + 2]);
    // let mut c11 = load(&c[row1][col + 2..col + 4]);
    //
    // for k in 0..l {
    //     let b0 = load(&b[k][col..col + 2]);
    //     let b1 = load(&b[k][col + 2..col + 4]);
    //
    //     let a0 = alpha * a[row0][k];
    //     let a1 = alpha * a[row1][k];
    //
    //     c00 = fma(c00, b0, a0);
    //     c01 = fma(c01, b1, a0);
    //     c10 = fma(c10, b0, a1);
    //     c11 = fma(c11, b1, a1);
    // }
    //
    // store(&mut c[row0][col..], c00, c01);
    // store(&mut c[row1][col..], c10, c11);
    let m = ai.i_end - ai.i_start;
    let l = ai.j_end - ai.j_start;
    let n = bi.j_end - bi.j_start;
    for i in (0..m).step_by(2) {
        let a_start = (ai.i_start + i) * ai.n + ai.j_start;
        let c_start = (ci.i_start + i) * ci.n + ci.j_start;
        let [c_row1, c_row2] = c.data.get_disjoint_mut([
            c_start..c_start + n,
            c_start + ai.n..c_start + n + ai.n
        ]).unwrap();
        // let c_row1 = &mut c.data[c_start..c_start + n];
        // let c_row2 = &mut c.data[c_start + ai.n..c_start + n + ai.n];
        // register blocking 2x4
        let mut c00 = [0.; 2];
        let mut c01 = [0.; 2];
        let mut c10 = [0.; 2];
        let mut c11 = [0.; 2];
        for k in (0..l).step_by(2) {
            let a_ik0 = [alpha * a.data[a_start + k],
                         alpha * a.data[a_start + k + 1]];
            let a_ik1 = [alpha * a.data[a_start + k + ai.n],
                         alpha * a.data[a_start + k + ai.n + 1]];
            let b_start = (bi.i_start + k) * bi.n + bi.j_start;
            let b_row1 = &b.data[b_start..b_start + n];
            let b_row2 = &b.data[b_start + bi.n..b_start + n + bi.n];

            let mut b00 = [0.; 2];
            let mut b01 = [0.; 2];
            let mut b10 = [0.; 2];
            let mut b11 = [0.; 2];
            let chunked = c_row1.chunks_mut(4)
                                .zip(b_row1.chunks(4))
                                .zip(c_row2.chunks_mut(4)
                                .zip(b_row2.chunks(4)));
            for ((c_0j, b_0j), (c_1j, b_1j)) in chunked {
                // load
                b00[0] = b_0j[0];
                b00[1] = b_0j[1];
                b01[0] = b_0j[2];
                b01[1] = b_0j[3];
                b10[0] = b_1j[0];
                b10[1] = b_1j[1];
                b11[0] = b_1j[2];
                b11[1] = b_1j[3];

                c00[0] = c_0j[0];
                c00[1] = c_0j[1];
                c01[0] = c_0j[2];
                c01[1] = c_0j[3];
                c10[0] = c_1j[0];
                c10[1] = c_1j[1];
                c11[0] = c_1j[2];
                c11[1] = c_1j[3];

                // gemm microkernel
                c00[0] = a_ik0[0].mul_add(b00[0], c00[0]);
                c00[0] = a_ik0[1].mul_add(b10[0], c00[0]);
                c00[1] = a_ik0[0].mul_add(b00[1], c00[1]);
                c00[1] = a_ik0[1].mul_add(b10[1], c00[1]);

                c01[0] = a_ik0[0].mul_add(b01[0], c01[0]);
                c01[0] = a_ik0[1].mul_add(b11[0], c01[0]);
                c01[1] = a_ik0[0].mul_add(b01[1], c01[1]);
                c01[1] = a_ik0[1].mul_add(b11[1], c01[1]);

                c10[0] = a_ik1[0].mul_add(b00[0], c10[0]);
                c10[0] = a_ik1[1].mul_add(b10[0], c10[0]);
                c10[1] = a_ik1[0].mul_add(b00[1], c10[1]);
                c10[1] = a_ik1[1].mul_add(b10[1], c10[1]);

                c11[0] = a_ik1[0].mul_add(b01[0], c11[0]);
                c11[0] = a_ik1[1].mul_add(b11[0], c11[0]);
                c11[1] = a_ik1[0].mul_add(b01[1], c11[1]);
                c11[1] = a_ik1[1].mul_add(b11[1], c11[1]);

                // store
                c_0j[0] = c00[0];
                c_0j[1] = c00[1];
                c_0j[2] = c01[0];
                c_0j[3] = c01[1];
                c_1j[0] = c10[0];
                c_1j[1] = c10[1];
                c_1j[2] = c11[0];
                c_1j[3] = c11[1];
            }
        }
    };
}

fn gemm_divide_and_conquer<'a>(
    ai: Strides,
    bi: Strides,
    ci: Strides,
    a: &'a Matrix,
    b: &'a Matrix,
    c: &'a mut Matrix,
    alpha: f64,
    beta: f64
) {
    let m = ci.i_end - ci.i_start;
    let l = bi.i_end - bi.i_start;
    let n = ci.j_end - ci.j_start;
    if m < TILESIZE || n < TILESIZE || l < TILESIZE {
        kernel(ai, bi, ci, a, b, c, alpha);
    } else {
        let [a00, a01, a10, a11] = ai.quadrants();
        let [b00, b01, b10, b11] = bi.quadrants();
        let [c00, c01, c10, c11] = ci.quadrants();

        gemm_divide_and_conquer(a00, b00, c00, a, b, c, alpha, beta);
        gemm_divide_and_conquer(a01, b10, c00, a, b, c, alpha, beta);
        gemm_divide_and_conquer(a11, b10, c10, a, b, c, alpha, beta);
        gemm_divide_and_conquer(a10, b00, c10, a, b, c, alpha, beta);
        gemm_divide_and_conquer(a01, b11, c01, a, b, c, alpha, beta);
        gemm_divide_and_conquer(a00, b01, c01, a, b, c, alpha, beta);
        gemm_divide_and_conquer(a10, b01, c11, a, b, c, alpha, beta);
        gemm_divide_and_conquer(a11, b11, c11, a, b, c, alpha, beta);
    }
}

/// GEMM takes three matrix arguments--A, B, C and computes
/// X = A B + C
/// Naive divide-and-conquer algorithm. O(n^3)
/// Assume row-major
fn gemm<'a>(
    a: &'a Matrix,
    b: &'a Matrix,
    c: &'a mut Matrix,
    alpha: f64,
    beta: f64
) {
    assert_eq!(c.nrow, a.nrow);
    assert_eq!(c.ncol, b.ncol);
    assert_eq!(a.ncol, b.nrow);

    let ai = a.index(0, a.nrow, 0, a.ncol);
    let bi = b.index(0, b.nrow, 0, b.ncol);
    let ci = c.index(0, c.nrow, 0, c.ncol);
    for c_i in c.data.iter_mut() {
        *c_i *= beta;
    }
    gemm_divide_and_conquer(ai, bi, ci, a, b, c, alpha, beta);
}

fn main() {
    let n = 4096;
    let a = Matrix { nrow: n, ncol: n, data: vec![2.; n * n] };
    let b = Matrix { nrow: n, ncol: n, data: vec![3.; n * n] };
    let mut c = Matrix { nrow: n, ncol: n, data: vec![0.1; n * n] };
    let alpha = 1.;
    let beta = 1.;

    let start_time = Instant::now();
    gemm(&a, &b, &mut c, alpha, beta);
    let end_time = Instant::now();
    println!("{:?}", c.data[0]);
    assert!(c.data.iter().all(|x| (x - 24576.1).abs() < 1e-8));
    println!("{:?}", end_time - start_time);
}
