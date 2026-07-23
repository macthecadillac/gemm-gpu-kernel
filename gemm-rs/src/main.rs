use std::array;
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
    m: usize,
    n: usize,
    data: Vec<f64>
}

#[derive(Copy, Clone)]
#[allow(dead_code)]
struct Strides {
    i_start: usize,
    i_end: usize,
    j_start: usize,
    j_end: usize
}

#[inline]
fn microkernel(
    l: usize,
    j: usize,
    a_panel: &[f64],
    b_panel: &[f64],
    c_rows: &mut [&mut [f64]],
    c_tile: &mut [[f64; NR]]
) {
    // load c tile
    for row in 0..MR {
        c_tile[row].copy_from_slice(&c_rows[row][j..j + NR]);
    }

    // actual matrix multiplication
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

#[inline]
fn kernel<'a>(
    ai: Strides,
    bi: Strides,
    ci: Strides,
    a: &'a Matrix,
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
    // pack_b(n, l, bi, b, b_panel);

    // this one lives in the registers
    let mut c_tile = [[0.; NR]; MR];

    for i in (0..m).step_by(MR) {
        let a_start = (ai.i_start + i) * a.n + ai.j_start;
        let c_start = (ci.i_start + i) * c.n + ci.j_start;

        // set up c rows
        // initialize array to hold range information about the rows
        let mut range_slice: [_; MR] = array::from_fn(|_| 0..1);
        // fill the range array with actually ranges of the rows
        for (idx, range) in (0..MR)
            .map(|row| c_start + row * c.n..c_start + n + row * c.n)
            .enumerate() {
            range_slice[idx] = range;
        }
        let mut c_rows = c.data.get_disjoint_mut(range_slice).unwrap();

        // pack one micro-panel of A
        for k in 0..l {
            for row in 0..MR {
                a_panel[k * MR + row] = alpha * a.data[a_start + k + row * a.n];
            }
        }

        for j in (0..n).step_by(NR) {
            microkernel(l, j, a_panel, b_panel, &mut c_rows[..], &mut c_tile);
        }
    };
}

fn gemm_divide_and_conquer<'a>(
    a: &'a Matrix,
    b: &'a Matrix,
    c: &'a mut Matrix,
    alpha: f64,
    buffer: &'a mut Buffer,
) {
    let mtile = a.m / TILESIZE;
    let ntile = b.n / TILESIZE;
    let ltile = b.m / TILESIZE;
    for kk in (0..ltile).map(|k| k * TILESIZE) {
        for jj in (0..ntile).map(|j| j * TILESIZE) {
            let bi = Strides {
                i_start: kk,
                i_end: kk + TILESIZE,
                j_start: jj,
                j_end: jj + TILESIZE,
            };
            let n = bi.j_end - bi.j_start;
            let l = bi.i_end - bi.i_start;
            let b_panel = &mut buffer.b_panel[0..TILESIZE * TILESIZE];
            // pack the entire tile into b_panel with a swizzling pattern 
            // across many micro-panels
            let mut mpr = 0;  // micro-panel row
            for j in (0..n).step_by(NR) {
                for k in 0..l {
                    let row_start = (bi.i_start + k) * b.n + bi.j_start + j;
                    b_panel[mpr * NR..mpr * NR + NR].clone_from_slice(&b.data[row_start..row_start + NR]);
                    mpr += 1;
                }
            }
            for ii in (0..mtile).map(|i| i * TILESIZE) {
                let ai = Strides {
                    i_start: ii,
                    i_end: ii + TILESIZE,
                    j_start: kk,
                    j_end: kk + TILESIZE
                };
                let ci = Strides {
                    i_start: ii,
                    i_end: ii + TILESIZE,
                    j_start: jj,
                    j_end: jj + TILESIZE
                };
                kernel(ai, bi, ci, a, c, alpha, buffer);
            }
        }
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
    // assure we are getting proper dimensions for GEMM
    assert_eq!(c.m, a.m);
    assert_eq!(c.n, b.n);
    assert_eq!(a.n, b.m);

    // work with square matrices for now
    assert_eq!(a.m, a.n);
    assert_eq!(b.m, b.n);
    assert_eq!(c.m, c.n);

    for c_i in c.data.iter_mut() {
        *c_i *= beta;
    }

    let mut a_panel = [0.; TILESIZE * MR];
    let mut b_panel = [0.; TILESIZE * TILESIZE];
    let mut buffer = Buffer { a_panel: &mut a_panel, b_panel: &mut b_panel };

    gemm_divide_and_conquer(a, b, c, alpha, &mut buffer);
}

fn main() {
    let n = 4096;
    let a_data = (0..4096 * 4096).map(|x| x as f64 / (2048. * 2048.)).collect();
    let b_data = (0..4096 * 4096).rev().map(|x| x as f64 / (2048. * 2048.) - 0.5).collect();
    let a = Matrix { m: n, n, data: a_data };
    let b = Matrix { m: n, n, data: b_data };
    let mut c = Matrix { m: n, n, data: vec![0.1; n * n] };
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
