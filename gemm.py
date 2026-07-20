import cutlass
from cutlass import cute


@cute.jit
def add_dynamicexpr(b: cutlass.Float32):
    a = cutlass.Float32(2.0)
    result = a + b
    print(f"[meta-stage] result = {result}")
    cute.printf(f"[object-stage] result = {result}")


@cute.jit
def add_constexpr(b: cutlass.Constexpr):
    a = 2.0
    result = a + b
    print(f"[meta-stage] result = {result}")
    cute.printf(f"[object-stage] result = {result}")


@cute.jit
def add_hybrid(b: cutlass.Constexpr):
    a = cutlass.Float32(2.0)
    result = a + b
    print(f"[meta-stage] result = {result}")
    cute.printf(f"[object-stage] result = {result}")
