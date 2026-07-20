# From veitner.bearblog.dev/cute-partitions
import cutlass
from cutlass import cute
import torch


@cute.kernel
def copy_kernel(S: cute.Tensor, D: cute.Tensor):
    thread_idx, _, _ = cute.arch.thread_idx()
    block_idx, _, _ = cute.arch.block_idx()
    block_dim, _, _ = cute.arch.block_dim()

    num = cute.size(S, mode=[2])
    block_start = block_idx * block_dim + thread_idx
    x = block_start // num
    y = block_start % num
    block_coordinate = ((None, None), x, y)
    tile_S = S[block_coordinate]
    tile_D = D[block_coordinate]

    print("Block Tile:")
    print(f"tile_S = {tile_S}")
    print(f"tile_D = {tile_D}")

    fragment = cute.make_fragment_like(tile_S)

    print("Fragment:")
    print(f"fragment = {fragment}")

    fragment.store(tile_S.load())
    tile_D.store(fragment.load())


@cute.jit
def launch_copy(
    tensor_S: cute.Tensor,
    tensor_D: cute.Tensor,
    block_shape: cute.Shape,
    num_threads: cutlass.Constexpr[cutlass.Int32]
):
    print("Tensors:")
    print(f"tensor_S = {tensor_S}")
    print(f"tensor_D = {tensor_D}")

    tiled_tensor_S = cute.tiled_divide(tensor_S, block_shape)
    tiled_tensor_D = cute.tiled_divide(tensor_D, block_shape)

    print("Block Tile Tensor:")
    print(f"tiled_tensor_S = {tiled_tensor_S}")
    print(f"tiled_tensor_D = {tiled_tensor_D}")

    m = cute.size(tiled_tensor_D, mode=[1])
    n = cute.size(tiled_tensor_D, mode=[2])
    grid_dim = (m * n) // num_threads, 1, 1
    block_dim = num_threads, 1, 1

    print("Grid and Block Configuration")
    print(f"grid_dim = {grid_dim}")
    print(f"block_dim = {block_dim}")

    kernel = copy_kernel(tiled_tensor_S, tiled_tensor_D)
    kernel.launch(grid=grid_dim, block=block_dim)


@cute.kernel
def copy_kernel_zipped(
    S: cute.Tensor,
    D: cute.Tensor,
    thread_layout: cute.Layout
):
    thread_idx, _, _ = cute.arch.thread_idx()
    block_idx, _, _ = cute.arch.block_idx()

    block_coordinate = ((None, None), block_idx)
    tile_S = S[block_coordinate]
    tile_D = D[block_coordinate]

    print("Block Tile:")
    print(f"tile_S = {tile_S}")
    print(f"tile_D = {tile_D}")

    thread_tile_S = cute.local_partition(tile_S, thread_layout, thread_idx)
    thread_tile_D = cute.local_partition(tile_D, thread_layout, thread_idx)

    print("Thread Tile:")
    print(f"thread_tile_S = {thread_tile_S}")
    print(f"thread_tile_D = {thread_tile_D}")

    fragment = cute.make_fragment_like(thread_tile_S)

    print("Fragment:")
    print(f"fragment = {fragment}")

    fragment.store(thread_tile_S.load())
    thread_tile_D.store(fragment.load())


@cute.jit
def launch_copy_zipped(
    tensor_S: cute.Tensor,
    tensor_D: cute.Tensor,
    block_shape: cute.Shape,
    thread_shape: cutlass.Shape
):
    print("Tensors:")
    print(f"tensor_S = {tensor_S}")
    print(f"tensor_D = {tensor_D}")

    tiled_tensor_S = cute.zipped_divide(tensor_S, block_shape)
    tiled_tensor_D = cute.zipped_divide(tensor_D, block_shape)

    print("Block Tile Tensor:")
    print(f"tiled_tensor_S = {tiled_tensor_S}")
    print(f"tiled_tensor_D = {tiled_tensor_D}")

    thread_layout = cute.make_layout(thread_shape)

    print("Thread Layout:")
    print(f"thread_layout = {thread_layout}")

    grid_dim = cute.size(tiled_tensor_D, mode=[1]), 1, 1
    block_dim = cute.size(thread_layout), 1, 1

    print("Grid and Block Configuration:")
    print(f"grid_dim = {grid_dim}")
    print(f"block_dim = {block_dim}")

    kernel = copy_kernel_zipped(tiled_tensor_S,
                                tiled_tensor_D,
                                thread_layout)
    kernel.launch(grid=grid_dim, block=block_dim)


if __name__ == "__main__":
    cutlass.cuda.initialize_cuda_context()
    tensor_shape = 8192, 8192
    block_shape = 1, 16
    num_threads = 256

    S = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    tensor_S = cute.runtime.from_dlpack(S, assumed_align=16)
    tensor_D = cute.runtime.from_dlpack(D, assumed_align=16)

    launch_copy(tensor_S, tensor_D, block_shape, num_threads)

    torch.testing.assert_close(S, D)

    print(S)
    print(D)

    block_shape = 32, 256
    thread_shape = 8, 32