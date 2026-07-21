use std::time::Instant;
use rayon::prelude::*;

const TILESIZE: usize = 4096;

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
        let m = self.nrow;
        let n = self.ncol;
        MatrixIndex { m, n, i_start, i_end, j_start, j_end }
    }
}

#[derive(Copy, Clone)]
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
    if ci.i_end - ci.i_start > TILESIZE && ci.j_end - ci.j_start > TILESIZE {
        let [a00, a01, a10, a11] = ai.quadrants();
        let [b00, b01, b10, b11] = bi.quadrants();
        let [c00, c01, c10, c11] = ci.quadrants();

        gemm_divide_and_conquer(a00, b00, c00, a, b, c, alpha, beta);
        gemm_divide_and_conquer(a10, b00, c10, a, b, c, alpha, beta);
        gemm_divide_and_conquer(a00, b01, c01, a, b, c, alpha, beta);
        gemm_divide_and_conquer(a10, b01, c11, a, b, c, alpha, beta);
        gemm_divide_and_conquer(a01, b10, c00, a, b, c, alpha, beta);
        gemm_divide_and_conquer(a11, b10, c10, a, b, c, alpha, beta);
        gemm_divide_and_conquer(a01, b11, c01, a, b, c, alpha, beta);
        gemm_divide_and_conquer(a11, b11, c11, a, b, c, alpha, beta);
    }
    let mut a_tile = Vec::new();
    let mut b_tile = Vec::new();
    let mut c_tile = Vec::new();
    a_tile.reserve_exact((ai.i_end - ai.i_start) * (ai.j_end - ai.j_start));
    b_tile.reserve_exact((bi.i_end - bi.i_start) * (bi.j_end - bi.j_start));
    c_tile.reserve_exact((ci.i_end - ci.i_start) * (ci.j_end - ci.j_start));

    // pack a in row-major
    for i in ai.i_start..ai.i_end {
        for j in ai.j_start..ai.j_end {
            a_tile.push(a.data[ai.n * i + j]);
        }
    }

    // pack b in col-major
    for j in bi.j_start..bi.j_end {
        for i in bi.i_start..bi.i_end {
            b_tile.push(b.data[bi.n * i + j]);
        }
    }

    // pack c in row-major
    for i in ci.i_start..ci.i_end {
        for j in ci.j_start..ci.j_end {
            c_tile.push(c.data[ci.n * i + j]);
        }
    }

    // tile-based gemm
    let a_m = ai.i_end - ai.i_start;
    let a_n = ai.j_end - ai.j_start;
    let b_m = bi.i_end - bi.i_start;
    let b_n = bi.j_end - bi.j_start;
    let c_m = ci.i_end - ci.i_start;
    let c_n = ci.j_end - ci.j_start;
    c_tile.par_iter_mut()
        .enumerate()
        .for_each(|(k, c_ij)| {
            let i = k / c_n;
            let j = k % c_n;
            let a_row_i = &a_tile[i * a_n..(i + 1) * a_n];
            let b_col_j = &b_tile[j * b_m..(j + 1) * b_m];
            *c_ij += alpha * a_row_i.iter()
                .zip(b_col_j.iter())
                .map(|(&x, &y)| x * y)
                .sum::<f64>();
        });
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
    println!("{:?}", end_time - start_time);
}
