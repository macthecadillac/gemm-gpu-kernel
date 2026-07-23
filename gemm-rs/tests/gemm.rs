use gemm_rs::{gemm, Matrix, TILESIZE};

fn patterned_matrix(n: usize, seed: usize) -> Matrix {
    let data = (0..n * n)
        .map(|index| {
            let row = index / n;
            let col = index % n;
            let value = (row * 17 + col * 31 + seed * 13) % 101;
            (value as f64 - 50.0) / 37.0
        })
        .collect();
    Matrix::new(n, n, data)
}

fn diagonal_matrix(n: usize) -> Matrix {
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        data[i * n + i] = (i as f64 % 23.0 - 11.0) / 7.0;
    }
    Matrix::new(n, n, data)
}

fn reference_gemm(a: &Matrix, b: &Matrix, c: &mut Matrix, alpha: f64, beta: f64) {
    for i in 0..a.m {
        for j in 0..b.n {
            let mut value = beta * c.data[i * c.n + j];
            for k in 0..a.n {
                value = (alpha * a.data[i * a.n + k])
                    .mul_add(b.data[k * b.n + j], value);
            }
            c.data[i * c.n + j] = value;
        }
    }
}

fn assert_matrices_close(actual: &Matrix, expected: &Matrix) {
    assert_eq!((actual.m, actual.n), (expected.m, expected.n));

    for (index, (&actual, &expected)) in actual.data.iter().zip(&expected.data).enumerate() {
        let tolerance = 1.0e-10 * (1.0 + expected.abs());
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index} differs: actual={actual}, expected={expected}, \
             tolerance={tolerance}"
        );
    }
}

#[test]
fn blocked_gemm_matches_reference_with_alpha_and_beta() {
    let n = TILESIZE;
    let a = patterned_matrix(n, 1);
    let b = patterned_matrix(n, 2);
    let initial_c = patterned_matrix(n, 3);
    let mut actual = initial_c.clone();
    let mut expected = initial_c;
    let alpha = 0.75;
    let beta = -0.25;

    gemm(&a, &b, &mut actual, alpha, beta);
    reference_gemm(&a, &b, &mut expected, alpha, beta);

    assert_matrices_close(&actual, &expected);
}

#[test]
fn blocked_gemm_crosses_m_n_and_k_tile_boundaries() {
    let n = TILESIZE * 2;
    let a = diagonal_matrix(n);
    let b = patterned_matrix(n, 10);
    let initial_c = patterned_matrix(n, 11);
    let mut actual = initial_c.clone();
    let mut expected = initial_c;
    let alpha = -0.625;
    let beta = 0.375;

    gemm(&a, &b, &mut actual, alpha, beta);

    for i in 0..n {
        let diagonal = a.data[i * n + i];
        for j in 0..n {
            let index = i * n + j;
            expected.data[index] =
                (alpha * diagonal).mul_add(b.data[index], beta * expected.data[index]);
        }
    }

    assert_matrices_close(&actual, &expected);
}

#[test]
#[should_panic(expected = "matrix dimensions must be divisible by TILESIZE")]
fn rejects_dimensions_that_the_blocked_kernel_cannot_handle() {
    let n = TILESIZE / 2;
    let a = patterned_matrix(n, 1);
    let b = patterned_matrix(n, 2);
    let mut c = patterned_matrix(n, 3);

    gemm(&a, &b, &mut c, 1.0, 0.0);
}

#[test]
#[should_panic(expected = "matrix data length does not match its dimensions")]
fn matrix_constructor_rejects_invalid_storage() {
    Matrix::new(2, 3, vec![0.0; 5]);
}

#[cfg(target_os = "macos")]
mod accelerate {
    use super::{assert_matrices_close, patterned_matrix};
    use accelerate_src as _;
    use cblas::{dgemm, Layout, Transpose};
    use gemm_rs::{gemm, Matrix, TILESIZE};
    use std::time::{Duration, Instant};

    fn accelerate_gemm(
        a: &Matrix,
        b: &Matrix,
        c: &mut Matrix,
        alpha: f64,
        beta: f64,
    ) {
        let m = i32::try_from(a.m).unwrap();
        let n = i32::try_from(b.n).unwrap();
        let k = i32::try_from(a.n).unwrap();

        // SAFETY: All matrices are contiguous row-major allocations with the
        // dimensions and leading dimensions passed below. C does not alias A or B.
        unsafe {
            dgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::None,
                m,
                n,
                k,
                alpha,
                &a.data,
                k,
                &b.data,
                n,
                beta,
                &mut c.data,
                n,
            );
        }
    }

    fn gibiflops(n: usize, elapsed: Duration) -> f64 {
        2.0 * (n as f64).powi(3) / elapsed.as_secs_f64() / 2_f64.powi(30)
    }

    #[test]
    fn accelerate_dgemm_matches_blocked_gemm() {
        let n = TILESIZE;
        let a = patterned_matrix(n, 4);
        let b = patterned_matrix(n, 5);
        let initial_c = patterned_matrix(n, 6);
        let mut actual = initial_c.clone();
        let mut accelerate = initial_c;
        let alpha = -0.5;
        let beta = 0.25;

        gemm(&a, &b, &mut actual, alpha, beta);
        accelerate_gemm(&a, &b, &mut accelerate, alpha, beta);

        assert_matrices_close(&actual, &accelerate);
    }

    #[test]
    #[ignore = "performance baseline; run in release mode with --ignored --nocapture"]
    fn accelerate_dgemm_performance_baseline() {
        let n = 4096;
        let a = patterned_matrix(n, 7);
        let b = patterned_matrix(n, 8);
        let initial_c = patterned_matrix(n, 9);
        let mut actual = initial_c.clone();
        let mut accelerate = initial_c;
        let alpha = 1.0;
        let beta = 1.0;

        // Initialize Rayon's global pool before measuring the custom kernel.
        let _ = rayon::current_num_threads();

        let start = Instant::now();
        gemm(&a, &b, &mut actual, alpha, beta);
        let custom_elapsed = start.elapsed();

        let start = Instant::now();
        accelerate_gemm(&a, &b, &mut accelerate, alpha, beta);
        let accelerate_elapsed = start.elapsed();

        assert_matrices_close(&actual, &accelerate);

        println!(
            "custom:     {custom_elapsed:?} ({:.2} GiFLOP/s)",
            gibiflops(n, custom_elapsed)
        );
        println!(
            "Accelerate: {accelerate_elapsed:?} ({:.2} GiFLOP/s)",
            gibiflops(n, accelerate_elapsed)
        );
        println!(
            "custom / Accelerate elapsed-time ratio: {:.3}",
            custom_elapsed.as_secs_f64() / accelerate_elapsed.as_secs_f64()
        );
    }
}
