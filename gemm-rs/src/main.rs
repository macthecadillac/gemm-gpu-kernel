const TILESIZE: usize = 64;

struct Matrix {
    nrow: usize,
    ncol: usize,
    data: Vec<f64>
}

impl Matrix {
    fn view<'a>(
        &'a self,
        i_start: usize,
        i_end: usize,
        j_start: usize,
        j_end: usize
    ) -> MatrixView<'a> {
        assert!(i_end <= self.nrow);
        assert!(j_end <= self.ncol);
        MatrixView {
            nrow: i_end - i_start,
            ncol: j_end - j_start,
            i_start: i_start,
            j_start: j_start,
            data: &self.data[..]
        }
    }

    fn view_mut<'a>(
        &'a mut self,
        i_start: usize,
        i_end: usize,
        j_start: usize,
        j_end: usize
    ) -> MatrixView<'a> {
        assert!(i_end <= self.nrow);
        assert!(j_end <= self.ncol);
        MatrixView {
            nrow: i_end - i_start,
            ncol: j_end - j_start,
            i_start: i_start,
            j_start: j_start,
            data: &mut self.data[..]
        }
    }
}

struct MatrixView<'a> {
    nrow: usize,
    ncol: usize,
    i_start: usize,
    j_start: usize,
    data: &'a [f64]
}

struct MatrixViewMut<'a> {
    nrow: usize,
    ncol: usize,
    i_start: usize,
    j_start: usize,
    data: &'a mut [f64]
}

#[derive(Copy, Clone)]
struct Matrix2x2 { data: [f64; 4] }

/// all arguments are row-major 2x2 matrices
fn mma_2x2(x: &mut Matrix2x2, a: &Matrix2x2, b: &Matrix2x2, c: &Matrix2x2) {
    let mut acc = [0.; 4];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                acc[2 * i + j] += a.data[2 * i + k] * b.data[2 * k + j];
            }
        }
    }
    for (x_, (acc_, &c_)) in x.data.iter_mut().zip(acc.into_iter().zip(c.data.iter())) {
        *x_ += acc_ + c_;
    }
}

/// GEMM takes three matrix arguments--A, B, C and computes
/// X = A B + C
/// Naive divide-and-conquer algorithm. O(n^3)
/// We assume the dimensions work
fn gemm<'a>(x: &'a mut Matrix, a: &'a Matrix, b: &'a Matrix, c: &'a Matrix) {
    assert_eq!(x.nrow, a.nrow);
    assert_eq!(x.ncol, b.ncol);
    assert_eq!(a.ncol, b.nrow);
    assert_eq!(x.nrow, c.nrow);
    assert_eq!(x.ncol, c.ncol);

    // break a matrix into tiles
    let padded_x_nrow = x.nrow + x.nrow % 2;
    let padded_x_ncol = x.ncol + x.ncol % 2;
    let x_nrow_ = padded_x_nrow / 2;
    let x_ncol_ = padded_x_ncol / 2;
}

fn main() {}
