import cutlass
from cutlass import cute

import numpy as np
import torch


@cute.kernel
def hello_kernel():
    thread_idx, _, _ = cute.arch.thread_idx()
    if thread_idx == 0:
        cute.printf("Hello from GPU")


@cute.jit
def hello_world():
    cutlass.cuda.initialize_cuda_context()
    hello_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1))


@cute.jit
def print_demo(a: cutlass.Int32, b: cutlass.Constexpr[int]):
    print(f"static a: {a}")
    print(f"static b: {b}")
    cute.printf(f"dynamic a: {a}")
    cute.printf(f"dynamic b: {b}")
    layout = cute.make_layout((a, b))
    print(f"static layout: {layout}")
    cute.printf(f"dynamic layout: {layout}")


@cute.jit
def dtypes():
    a = cutlass.Int32(42)
    b = a.to(cutlass.Float32)
    c = b + 0.5
    d = c.to(cutlass.Int32)
    cute.printf(f"a={a}, b={b}, c={c}, d={d}")


@cute.jit
def tensor_demo(t: cute.Tensor):
    cute.printf(f"t[0, 0] = {t[0, 0]}")
    sub = t[(None, 0)]
    frag = cute.make_rmem_tensor(sub.layout, sub.element_type)
    frag.store(sub.load())
    cute.print_tensor(frag)


# arr = torch.arange(0, 12, dtype=torch.float32).reshape(3, 4)
# tensor_demo(cute.runtime.from_dlpack(arr))


@cute.jit
def layout_stride_demo(M: cutlass.Int32, N: cutlass.Int32):
    row_major = cute.make_layout((M, N), stride=(N, cutlass.Int32(1)))
    col_major = cute.make_layout((M, N), stride=(cutlass.Int32(1), M))
    print(f"static row-major: {row_major}")
    print(f"static col-major: {col_major}")
    cute.printf(f"dynamic row-major: {row_major}")
    cute.printf(f"dynamic col-major: {col_major}")


@cute.jit
def slicing_examples(t: cute.Tensor):
    cute.printf(f"t[1, 2] = {t[1, 2]}")
    row = t[(None, 1)]
    row_frag = cute.make_rmem_tensor(row.layout, row.element_type)
    row_frag.store(row.load())
    cute.printf("Second row:")
    cute.print_tensor(row_frag)

    col = t[(2, None)]
    col_frag = cute.make_rmem_tensor(col.layout, col.element_type)
    col_frag.store(col.load())
    cute.printf("Third column:")
    cute.print_tensor(col_frag)

    cute.printf(
        "t[2] = {} (equivalent to t[{}])",
        t[2],
        cute.make_identity_tensor(t.layout.shape)[2]
    )


# arr = torch.arange(12, dtype=torch.float32).reshape(4, 3)
# print(arr)
# slicing_examples(cute.runtime.from_dlpack(arr))


@cute.jit
def ssa_add(dst: cute.Tensor, x: cute.Tensor, y: cute.Tensor):
    xv = x.load()
    yv = y.load()
    dst.store(xv + yv)
    cute.print_tensor(dst)


# X = np.ones((2, 3), dtype=np.float32)
# Y = np.full((2, 3), 2.0, dtype=np.float32)
# Z = np.zeros((2, 3), dtype=np.float32)
# ssa_add(cute.runtime.from_dlpack(Z),
#         cute.runtime.from_dlpack(X),
#         cute.runtime.from_dlpack(Y))


@cute.jit
def ssa_reduce(a: cute.Tensor):
    v = a.load()
    total = v.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=0)
    cute.printf(f"total sum = {total}")

    row_sum = v.reduce(cute.ReductionOp.ADD, 0.0,
                       reduction_profile=(None, 1))
    row_frag = cute.make_rmem_tensor(row_sum.shape, cutlass.Float32)
    row_frag.store(row_sum)
    cute.printf("Row-wise sum:")
    cute.print_tensor(row_frag)

    col_sum = v.reduce(cute.ReductionOp.ADD, 0.0,
                        reduction_profile=(1, None))
    col_frag = cute.make_rmem_tensor(col_sum.shape, cutlass.Float32)
    col_frag.store(col_sum)
    cute.printf("Col-wise sum:")
    cute.print_tensor(col_frag)


# A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
# print(A)
# ssa_reduce(cute.runtime.from_dlpack(A))

@cute.kernel
def vadd_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    thread_idx, _, _ = cute.arch.thread_idx()
    block_idx, _, _ = cute.arch.block_idx()
    block_dim, _, _ = cute.arch.block_dim()
    idx = block_idx * block_dim + thread_idx
    m, n = gA.shape[1]
    mi = idx // n
    ni = idx % n
    gC[(None, (mi, ni))] = gA[(None, (mi, ni))].load() + gB[(None, (mi, ni))].load()


@cute.jit
def vadd(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    gA = cute.zipped_divide(A, (1, 4))
    gB = cute.zipped_divide(B, (1, 4))
    gC = cute.zipped_divide(C, (1, 4))
    threads = 256
    vadd_kernel(gA, gB, gC).launch(
        grid=(cute.size(gC, mode=[1]) // threads, 1, 1),
        block=(threads, 1, 1)
    )


@cute.kernel
def vadd_kernel_zipped(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    thread_idx, _, _ = cute.arch.thread_idx()
    block_idx, _, _ = cute.arch.block_idx()
    block_dim, _, _ = cute.arch.block_dim()
    idx = block_idx * block_dim + thread_idx
    m, n = gA.shape[1]


if __name__ == "__main__":
    M = N = 1024
    a = torch.randn(M, N, device="cuda", dtype=torch.float16)
    b = torch.randn(M, N, device="cuda", dtype=torch.float16)
    c = torch.randn(M, N, device="cuda", dtype=torch.float16)
    vadd_compiled = cute.compile(
        vadd,
        cute.runtime.from_dlpack(a),
        cute.runtime.from_dlpack(b),
        cute.runtime.from_dlpack(c)
    )
    vadd_compiled(
        cute.runtime.from_dlpack(a),
        cute.runtime.from_dlpack(b),
        cute.runtime.from_dlpack(c)
    )