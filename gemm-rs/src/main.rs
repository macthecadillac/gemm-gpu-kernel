use std::array;
use std::ops::Range;
use std::time::Instant;
// use rayon::prelude::*;

const TILESIZE: usize = 256;
const MR: usize = 4;
const NR: usize = 8;

struct Buffer<'a> {
    a_panel: &'a mut [f64],
    b_panel: &'a mut [f64]
}

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
    buffer: &'a mut Buffer
) {
    let m = ai.i_end - ai.i_start;
    let l = ai.j_end - ai.j_start;
    let n = bi.j_end - bi.j_start;

    // set up buffers
    let a_panel = &mut buffer.a_panel[0..l * MR];
    let b_panel = &mut buffer.b_panel[0..l * n];
    // this one lives in the registers
    let mut c_tile = [[0.; NR]; MR];

    // b_panel is packed in a swizzle pattern
    let mut mpr = 0;  // micro-panel row
    for j in (0..n).step_by(NR) {
        for k in 0..l {
            let row_start = (bi.i_start + k) * bi.n + bi.j_start + j;
            b_panel[mpr * NR..mpr * NR + NR].clone_from_slice(&b.data[row_start..row_start + NR]);
            mpr += 1;
        }
    }

    for i in (0..m).step_by(MR) {
        let a_start = (ai.i_start + i) * ai.n + ai.j_start;
        let c_start = (ci.i_start + i) * ci.n + ci.j_start;

        // set up c rows
        // initialize array to hold range information about the rows
        let mut range_slice: [Range<usize>; MR] = array::from_fn(|_| 0..1);
        // fill the range array with actually ranges of the rows
        for (idx, range) in (0..MR)
            .map(|row| c_start + row * ci.n..c_start + n + row * ci.n)
            .enumerate() {
            range_slice[idx] = range;
        }
        let c_rows = c.data.get_disjoint_mut(range_slice).unwrap();

        // Pack A as K-major groups of MR rows and apply alpha once.
        for k in 0..l {
            for row in 0..MR {
                a_panel[k * MR + row] = alpha * a.data[a_start + k + row * ai.n];
            }
        }

        for j in (0..n).step_by(NR) {
            for row in 0..MR {
                c_tile[row].copy_from_slice(&c_rows[row][j..j + NR]);
            }

            let panel_start = (j / NR) * l * NR;
            for k in 0..l {
                let b_start = panel_start + k * NR;
                let b_row = &b_panel[b_start..b_start + NR];

                for row in 0..MR {
                    for col in 0..NR {
                        c_tile[row][col] =
                            a_panel[k * MR + row].mul_add(b_row[col], c_tile[row][col]);
                    }
                }
            }
            // store c tile
            for row in 0..MR {
                c_rows[row][j..j + NR].copy_from_slice(&c_tile[row]);
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
    buffer: &'a mut Buffer,
) {
    let m = ci.i_end - ci.i_start;
    let l = bi.i_end - bi.i_start;
    let n = ci.j_end - ci.j_start;
    if m <= TILESIZE || n <= TILESIZE || l <= TILESIZE {
        kernel(ai, bi, ci, a, b, c, alpha, buffer);
    } else {
        let [a00, a01, a10, a11] = ai.quadrants();
        let [b00, b01, b10, b11] = bi.quadrants();
        let [c00, c01, c10, c11] = ci.quadrants();

        gemm_divide_and_conquer(a00, b00, c00, a, b, c, alpha, buffer);
        gemm_divide_and_conquer(a01, b10, c00, a, b, c, alpha, buffer);
        gemm_divide_and_conquer(a11, b10, c10, a, b, c, alpha, buffer);
        gemm_divide_and_conquer(a10, b00, c10, a, b, c, alpha, buffer);
        gemm_divide_and_conquer(a01, b11, c01, a, b, c, alpha, buffer);
        gemm_divide_and_conquer(a00, b01, c01, a, b, c, alpha, buffer);
        gemm_divide_and_conquer(a10, b01, c11, a, b, c, alpha, buffer);
        gemm_divide_and_conquer(a11, b11, c11, a, b, c, alpha, buffer);
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

    let mut a_panel = [0.; TILESIZE * NR];
    let mut b_panel = [0.; TILESIZE * TILESIZE];
    let mut buffer = Buffer { a_panel: &mut a_panel, b_panel: &mut b_panel };

    gemm_divide_and_conquer(ai, bi, ci, a, b, c, alpha, &mut buffer);
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
