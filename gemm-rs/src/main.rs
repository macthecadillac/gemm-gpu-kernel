use gemm_rs::{gemm, Matrix};
use std::time::Instant;

fn main() {
    let n = 4096;
    let a_data = (0..n * n)
        .map(|x| x as f64 / (2048. * 2048.))
        .collect();
    let b_data = (0..n * n)
        .rev()
        .map(|x| x as f64 / (2048. * 2048.) - 0.5)
        .collect();
    let a = Matrix::new(n, n, a_data);
    let b = Matrix::new(n, n, b_data);
    let mut c = Matrix::new(n, n, vec![0.1; n * n]);
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
