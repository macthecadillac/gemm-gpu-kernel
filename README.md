This is a personal exercise to practice writing GPU kernels. Use at your own
peril.

The Rust implementation is split into a library and a small executable:

```text
gemm-rs/src/lib.rs   GEMM implementation
gemm-rs/src/main.rs  4096-by-4096 timing example
gemm-rs/tests/gemm.rs
```

Run the correctness tests with:

```sh
cd gemm-rs
cargo test --release
```

On macOS, the test suite uses the `cblas` bindings with Accelerate as the BLAS
provider. The longer 4096-by-4096 performance comparison is ignored by default:

```sh
cargo test --release accelerate_dgemm_performance_baseline \
  -- --ignored --nocapture
```
