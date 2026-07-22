use std::time::Instant;
use rayon::prelude::*;

const TILESIZE: usize = 64;

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
    ) -> MatrixIndex {
        assert!(i_end <= self.nrow);
        assert!(j_end <= self.ncol);
        let m = i_end - i_start;
        let n = j_end - j_start;
        MatrixIndex { m, n, i_start, i_end, j_start, j_end }
    }
}

#[derive(Copy, Clone)]
#[allow(dead_code)]
struct MatrixIndex {
    m: usize,
    n: usize,
    i_start: usize,
    i_end: usize,
    j_start: usize,
    j_end: usize
}

impl MatrixIndex {
    fn quadrants(&self) -> [MatrixIndex; 4] {
        // break x into tiles
        let nrow = self.i_end - self.i_start;
        let ncol = self.j_end - self.j_start;
        let padded_nrow = nrow + nrow % 2;
        let padded_ncol = ncol + ncol % 2;
        let nrow_ = padded_nrow / 2;
        let ncol_ = padded_ncol / 2;
        let m1 = MatrixIndex {
            i_end: self.i_start + nrow_,
            j_end: self.j_start + ncol_,
            ..*self
        };
        let m2 = MatrixIndex {
            i_end: self.i_start + nrow_,
            j_start: self.j_start + ncol_,
            ..*self
        };
        let m3 = MatrixIndex {
            i_start: self.i_start + nrow_,
            j_end: self.j_start + ncol_,
            ..*self
        };
        let m4 = MatrixIndex {
            i_start: self.i_start + nrow_,
            j_start: self.j_start + ncol_,
            ..*self
        };
        [m1, m2, m3, m4]
    }
}

fn kernel<'a>(
    ai: MatrixIndex,
    bi: MatrixIndex,
    ci: MatrixIndex,
    a: &'a Matrix,
    b: &'a Matrix,
    c: &'a mut Matrix,
    alpha: f64,
) {
    let m = ai.i_end - ai.i_start;
    let l = ai.j_end - ai.j_start;
    let n = bi.j_end - bi.j_start;
    for i in 0..m {
        let a_start = (ai.i_start + i) * ai.n + ai.j_start;
        let c_start = (ci.i_start + i) * ci.n + ci.j_start;
        let c_row = &mut c.data[c_start..c_start + n];
        for k in 0..l {
            let a_ik = alpha * a.data[a_start + k];
            let b_start = (bi.i_start + k) * bi.n + bi.j_start;
            let b_row = &b.data[b_start..b_start + n];
            for (c_ij, &b_kj) in c_row.iter_mut().zip(b_row) {
                *c_ij += a_ik * b_kj;
            }
        }
    };
}

fn gemm_divide_and_conquer<'a>(
    ai: MatrixIndex,
    bi: MatrixIndex,
    ci: MatrixIndex,
    a: &'a Matrix,
    b: &'a Matrix,
    c: &'a mut Matrix,
    alpha: f64,
    beta: f64
) {
    let m = ci.i_end - ci.i_start;
    let l = ai.i_end - ai.i_start;
    let n = ci.j_end - ci.j_start;
    if m < TILESIZE || n < TILESIZE || l < TILESIZE {
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
    } else {
        kernel(ai, bi, ci, a, b, c, alpha);
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
    let ci = c.index(0, b.nrow, 0, b.ncol);
    for c_i in c.data.iter_mut() {
        *c_i *= beta;
    }
    gemm_divide_and_conquer(ai, bi, ci, a, b, c, alpha, beta);
}

fn main() {
    let n = 1024;
    let a = Matrix { nrow: n, ncol: n, data: vec![2.; n * n] };
    let b = Matrix { nrow: n, ncol: n, data: vec![3.; n * n] };
    let mut c = Matrix { nrow: n, ncol: n, data: vec![0.1; n * n] };
    let alpha = 1.;
    let beta = 1.;

    let start_time = Instant::now();
    gemm(&a, &b, &mut c, alpha, beta);
    let end_time = Instant::now();
    assert!(c.data.iter().all(|x| (x - 6144.1).abs() < 1e-8));
    println!("{:?}", end_time - start_time);
}
