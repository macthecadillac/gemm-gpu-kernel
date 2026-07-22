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
    let m = ai.i_end - ai.i_start;
    let l = ai.j_end - ai.j_start;
    let n = bi.j_end - bi.j_start;
    let mut a_tile = [[0.; 4]; 4];
    let mut b_tile = [[0.; 4]; 4];
    let mut c_tile = [[0.; 4]; 4];
    for i in (0..m).step_by(4) {
        let a_start = (ai.i_start + i) * ai.n + ai.j_start;
        let c_start = (ci.i_start + i) * ci.n + ci.j_start;
        let c_rows = c.data.get_disjoint_mut([
            c_start..c_start + n,
            c_start + ci.n..c_start + n + ci.n,
            c_start + 2 * ci.n..c_start + n + 2 * ci.n,
            c_start + 3 * ci.n..c_start + n + 3 * ci.n
        ]).unwrap();
        // register blocking 4x4
        for k in (0..l).step_by(4) {
            // load a_tile
            for m in 0..4 {
                for n in 0..4 {
                    a_tile[m][n] = alpha * a.data[a_start + k + n + m * ai.n]
                }
            }

            let b_start = (bi.i_start + k) * bi.n + bi.j_start;
            let b_row1 = &b.data[b_start..b_start + n];
            let b_row2 = &b.data[b_start + bi.n..b_start + n + bi.n];
            let b_row3 = &b.data[b_start + 2 * bi.n..b_start + n + 2 * bi.n];
            let b_row4 = &b.data[b_start + 3 * bi.n..b_start + n + 3 * bi.n];
            let b_rows = [b_row1, b_row2, b_row3, b_row4];

            let row_length = c_rows[0].len();
            for z in (0..row_length).step_by(4) {
                // load b and c tiles
                for m in 0..4 {
                    for n in 0..4 {
                        b_tile[m][n] = b_rows[m][z..z + 4][n];
                        c_tile[m][n] = c_rows[m][z..z + 4][n];
                    }
                }

                for m in 0..4 {
                    for n in 0..4 {
                        for l in 0..4 {
                            c_tile[m][n] = a_tile[m][l].mul_add(b_tile[l][n], c_tile[m][n]);
                        }
                    }
                }

                // store c tile
                for m in 0..4 {
                    for n in 0..4 {
                        c_rows[m][z + n] = c_tile[m][n];
                    }
                }
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
) {
    let m = ci.i_end - ci.i_start;
    let l = bi.i_end - bi.i_start;
    let n = ci.j_end - ci.j_start;
    if m <= TILESIZE || n <= TILESIZE || l <= TILESIZE {
        kernel(ai, bi, ci, a, b, c, alpha);
    } else {
        let [a00, a01, a10, a11] = ai.quadrants();
        let [b00, b01, b10, b11] = bi.quadrants();
        let [c00, c01, c10, c11] = ci.quadrants();

        gemm_divide_and_conquer(a00, b00, c00, a, b, c, alpha);
        gemm_divide_and_conquer(a01, b10, c00, a, b, c, alpha);
        gemm_divide_and_conquer(a11, b10, c10, a, b, c, alpha);
        gemm_divide_and_conquer(a10, b00, c10, a, b, c, alpha);
        gemm_divide_and_conquer(a01, b11, c01, a, b, c, alpha);
        gemm_divide_and_conquer(a00, b01, c01, a, b, c, alpha);
        gemm_divide_and_conquer(a10, b01, c11, a, b, c, alpha);
        gemm_divide_and_conquer(a11, b11, c11, a, b, c, alpha);
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
    gemm_divide_and_conquer(ai, bi, ci, a, b, c, alpha);
}

fn main() {
    let n = 4096;
    let a_data = (0..4096 * 4096).map(|x| x as f64 / (2048. * 2048.)).collect();
    let b_data = (0..4096 * 4096).rev().map(|x| x as f64 / (2048. * 2048.) - 0.5).collect();
    let a = Matrix { nrow: n, ncol: n, data: a_data };
    let b = Matrix { nrow: n, ncol: n, data: b_data };
    let mut c = Matrix { nrow: n, ncol: n, data: vec![0.1; n * n] };
    let alpha = 1.;
    let beta = 1.;

    let start_time = Instant::now();
    gemm(&a, &b, &mut c, alpha, beta);
    let end_time = Instant::now();
    println!("{:?}", end_time - start_time);
    println!("{}", c.data[0]);
    println!("{}", c.data[1413999]);
    assert!((c.data[0] - 1.7669101716252045).abs() < 1e-8);
    assert!((c.data[1413999] - 2072.150783736631).abs() < 1e-8);
}
