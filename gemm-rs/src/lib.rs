use rayon::prelude::*;
use std::array;

pub const TILESIZE: usize = 256;
const MR: usize = 4;
const NR: usize = 8;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub m: usize,
    pub n: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(m: usize, n: usize, data: Vec<f64>) -> Self {
        let expected_len = m
            .checked_mul(n)
            .expect("matrix dimensions overflow usize");
        assert_eq!(
            data.len(),
            expected_len,
            "matrix data length does not match its dimensions"
        );
        Self { m, n, data }
    }
}

#[derive(Copy, Clone)]
struct Strides {
    i_start: usize,
    i_end: usize,
    j_start: usize,
    j_end: usize,
}

#[inline]
fn microkernel(
    l: usize,
    j: usize,
    a_panel: &[f64],
    b_panel: &[f64],
    c_rows: &mut [&mut [f64]],
    c_tile: &mut [[f64; NR]],
) {
    // Load C tile.
    for row in 0..MR {
        c_tile[row].copy_from_slice(&c_rows[row][j..j + NR]);
    }

    // Matrix multiplication.
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

    // Store C tile.
    for row in 0..MR {
        c_rows[row][j..j + NR].copy_from_slice(&c_tile[row]);
    }
}

#[inline]
fn kernel(
    ai: Strides,
    bi: Strides,
    c_col_start: usize,
    c_stride: usize,
    a: &Matrix,
    c_macro_tile: &mut [f64],
    alpha: f64,
    a_panel: &mut [f64],
    b_panel: &[f64],
) {
    let m = ai.i_end - ai.i_start;
    let l = ai.j_end - ai.j_start;
    let n = bi.j_end - bi.j_start;
    let a_panel = &mut a_panel[0..l * MR];
    let mut c_tile = [[0.; NR]; MR];

    for i in (0..m).step_by(MR) {
        let a_start = (ai.i_start + i) * a.n + ai.j_start;

        let range_slice: [_; MR] = array::from_fn(|row| {
            let start = (i + row) * c_stride + c_col_start;
            start..start + n
        });
        let mut c_rows = c_macro_tile.get_disjoint_mut(range_slice).unwrap();

        // Pack one micro-panel of A.
        for k in 0..l {
            for row in 0..MR {
                a_panel[k * MR + row] = alpha * a.data[a_start + k + row * a.n];
            }
        }

        for j in (0..n).step_by(NR) {
            microkernel(l, j, a_panel, b_panel, &mut c_rows[..], &mut c_tile);
        }
    }
}

fn gemm_blocked(
    a: &Matrix,
    b: &Matrix,
    c: &mut Matrix,
    alpha: f64,
    b_buffer: &mut [f64],
) {
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
            let b_panel = &mut b_buffer[0..TILESIZE * TILESIZE];

            // Pack the entire B tile as a sequence of NR-wide micro-panels.
            let mut micro_panel_row = 0;
            for j in (0..n).step_by(NR) {
                for k in 0..l {
                    let row_start = (bi.i_start + k) * b.n + bi.j_start + j;
                    b_panel[micro_panel_row * NR..micro_panel_row * NR + NR]
                        .copy_from_slice(&b.data[row_start..row_start + NR]);
                    micro_panel_row += 1;
                }
            }

            // Each Rayon task owns TILESIZE complete rows of C.
            let c_stride = c.n;
            let c_macro_len = TILESIZE * c_stride;
            c.data
                .par_chunks_exact_mut(c_macro_len)
                .enumerate()
                .for_each(|(block_idx, c_macro_tile)| {
                    let ii = block_idx * TILESIZE;
                    let ai = Strides {
                        i_start: ii,
                        i_end: ii + TILESIZE,
                        j_start: kk,
                        j_end: kk + TILESIZE,
                    };
                    let mut a_buffer = [0.; TILESIZE * MR];
                    kernel(
                        ai,
                        bi,
                        jj,
                        c_stride,
                        a,
                        c_macro_tile,
                        alpha,
                        &mut a_buffer,
                        b_panel,
                    );
                });
        }
    }
}

/// Computes `C = alpha * A * B + beta * C` for row-major square matrices.
///
/// The current blocked kernel requires every matrix dimension to be divisible
/// by [`TILESIZE`].
pub fn gemm(a: &Matrix, b: &Matrix, c: &mut Matrix, alpha: f64, beta: f64) {
    validate_storage("A", a);
    validate_storage("B", b);
    validate_storage("C", c);

    assert_eq!(c.m, a.m, "C and A must have the same row count");
    assert_eq!(c.n, b.n, "C and B must have the same column count");
    assert_eq!(a.n, b.m, "A columns must equal B rows");

    assert_eq!(a.m, a.n, "A must be square");
    assert_eq!(b.m, b.n, "B must be square");
    assert_eq!(c.m, c.n, "C must be square");
    assert_eq!(
        a.m % TILESIZE,
        0,
        "matrix dimensions must be divisible by TILESIZE ({TILESIZE})"
    );

    for value in &mut c.data {
        *value *= beta;
    }

    let mut b_buffer = [0.; TILESIZE * TILESIZE];
    gemm_blocked(a, b, c, alpha, &mut b_buffer);
}

fn validate_storage(name: &str, matrix: &Matrix) {
    let expected_len = matrix
        .m
        .checked_mul(matrix.n)
        .expect("matrix dimensions overflow usize");
    assert_eq!(
        matrix.data.len(),
        expected_len,
        "{name} data length does not match its dimensions"
    );
}
