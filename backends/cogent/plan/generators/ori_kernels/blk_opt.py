"""
Tensor Contraction Block Scheduling Optimizer

Given a tensor contraction equation and dimension sizes, determines the optimal
thread block index decomposition order to maximize L2 cache reuse.

Core principle:
  - Each block dimension (non-contraction index of the result tensor) is a candidate
    for the blockIdx decomposition.
  - For each candidate dimension, we compute how much input data can be REUSED
    when that dimension is fixed and other dimensions vary.
  - Dimensions with HIGH reuse should be in the OUTER loop (slow-varying in blockIdx)
    so their data stays in L2 cache while the inner dimensions sweep.
  - Dimensions with LOW reuse should be in the INNER loop (fast-varying in blockIdx).

Usage:
  python block_order_optimizer.py

  Or as a library:
    from block_order_optimizer import optimize_block_order
    result = optimize_block_order(
        output_indices=['a', 'b', 'c'],
        input_specs=[
            (['b', 'd', 'a'], "t2"),
            (['d', 'c'],      "v2"),
        ],
        contraction_indices=['d'],
        sizes={'a': 320, 'b': 320, 'c': 32, 'd': 312},
        tile_sizes={'a': 16, 'b': 16, 'c': 32, 'd': 8},
    )
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import math


@dataclass
class InputTensor:
    name: str
    indices: List[str]

    def depends_on(self, dim: str) -> bool:
        return dim in self.indices

    def slice_size(self, sizes: Dict[str, int], fixed_dims: List[str],
                   tile_sizes: Dict[str, int]) -> int:
        """
        Compute the data size (in elements) of this tensor when fixed_dims are
        held constant (at tile granularity) and all other dims span their full range.
        """
        total = 1
        for idx in self.indices:
            if idx in fixed_dims:
                total *= tile_sizes.get(idx, sizes[idx])
            else:
                total *= sizes[idx]
        return total

    def reused_size(self, sizes: Dict[str, int], varying_dim: str,
                    tile_sizes: Dict[str, int], block_dims: List[str]) -> int:
        """
        When varying_dim changes (one tile step), how much data from this tensor
        can be reused from the previous tile?

        If this tensor does NOT depend on varying_dim → 100% reuse (full tensor slice)
        If it DOES depend on varying_dim → 0% reuse (completely different data)
        """
        if not self.depends_on(varying_dim):
            # This tensor doesn't use varying_dim at all → fully reused
            return self.slice_size(sizes, block_dims, tile_sizes)
        else:
            return 0

    def total_footprint(self, sizes: Dict[str, int], tile_sizes: Dict[str, int],
                        block_dims: List[str]) -> int:
        """Size of data loaded by one thread block from this tensor."""
        return self.slice_size(sizes, block_dims, tile_sizes)


@dataclass
class BlockOrderResult:
    """Result of the block order optimization."""
    # Ordered from outermost (slowest) to innermost (fastest) in blockIdx
    order: List[str]
    # Per-dimension analysis
    analysis: List[dict] = field(default_factory=list)
    # Generated CUDA code snippet
    cuda_code: str = ""


def optimize_block_order(
    output_indices: List[str],
    input_specs: List[Tuple[List[str], str]],
    contraction_indices: List[str],
    sizes: Dict[str, int],
    tile_sizes: Dict[str, int],
    element_size: int = 4,  # bytes per element (4=float, 8=double)
) -> BlockOrderResult:
    """
    Determine optimal block index decomposition order.

    Args:
        output_indices: Indices of the result tensor, e.g. ['a', 'b', 'c']
        input_specs: List of (indices, name) for each input tensor
                     e.g. [(['b','d','a'], "t2"), (['d','c'], "v2")]
        contraction_indices: Indices summed over, e.g. ['d']
        sizes: Dictionary of dimension sizes, e.g. {'a': 320, 'b': 320, ...}
        tile_sizes: Dictionary of tile sizes per dimension
        element_size: Bytes per element (4 for float, 8 for double)

    Returns:
        BlockOrderResult with optimal order and analysis
    """
    inputs = [InputTensor(name=name, indices=indices)
              for indices, name in input_specs]

    # Block dimensions = output indices minus contraction indices
    block_dims = [idx for idx in output_indices if idx not in contraction_indices]

    # For each block dimension, compute the "reuse score":
    # When this dimension varies (is the fast-changing inner loop),
    # how much data from each input must be RE-LOADED vs reused?
    #
    # High reuse_score → this dim should be INNER (fast) because varying it
    #                     doesn't cause much cache thrashing
    # Low reuse_score  → this dim should be OUTER (slow) because varying it
    #                     invalidates lots of cached data
    #
    # Wait — actually the logic is the opposite:
    # We want to put dims with HIGH reload cost in the OUTER loop,
    # so they change slowly and their data stays in L2.

    analysis = []

    for dim in block_dims:
        num_blocks = math.ceil(sizes[dim] / tile_sizes.get(dim, sizes[dim]))

        # For each input, compute how much data is reused when this dim varies
        total_reused = 0
        total_footprint = 0
        tensor_details = []

        for inp in inputs:
            footprint = inp.total_footprint(sizes, tile_sizes, block_dims)
            reused = inp.reused_size(sizes, dim, tile_sizes, block_dims)
            reload = footprint - reused

            tensor_details.append({
                'tensor': inp.name,
                'depends_on_dim': inp.depends_on(dim),
                'footprint_elements': footprint,
                'footprint_bytes': footprint * element_size,
                'reused_elements': reused,
                'reload_elements': reload,
                'reload_bytes': reload * element_size,
            })

            total_reused += reused
            total_footprint += footprint

        # reload_cost: total bytes that must be re-fetched when this dim steps by one tile
        total_reload = total_footprint - total_reused
        reload_cost = total_reload * element_size

        # cache_pressure: reload_cost × num_blocks gives total extra traffic
        # if this dim is in the inner loop
        cache_pressure = reload_cost * num_blocks

        analysis.append({
            'dim': dim,
            'size': sizes[dim],
            'tile_size': tile_sizes.get(dim, sizes[dim]),
            'num_blocks': num_blocks,
            'total_footprint_bytes': total_footprint * element_size,
            'reload_cost_bytes': reload_cost,
            'cache_pressure': cache_pressure,
            'tensors': tensor_details,
        })

    # ================================================================
    # Sorting logic for optimal block order
    # ================================================================
    #
    # Goal: when blockIdx increments by 1 (adjacent blocks), the global
    # memory access patterns of the largest input should be as spatially
    # contiguous as possible → maximizes L2 cache line utilization.
    #
    # Three-level sorting (all applied together):
    #
    # 1. PRIMARY — reload_cost (descending → outer):
    #    Dimensions that cause large data re-fetches when they change
    #    should change as infrequently as possible (outer loop).
    #
    # 2. SECONDARY — spatial_locality_score (ascending → inner):
    #    Among dims with equal reload cost, the dim that produces the
    #    most spatially contiguous access when it varies should be
    #    innermost. This is the dim with the SMALLEST stride in the
    #    largest input tensor that depends on it.
    #    Column-major: indices[0] has stride 1 (most contiguous).
    #    A dim with small stride → varying it walks through contiguous
    #    memory → good for L2 cache lines → should be INNER.
    #
    # 3. TERTIARY — num_blocks (ascending → inner):
    #    Dimensions with fewer blocks contribute less to the total
    #    iteration space, so their position matters less.
    #    Place them toward inner to avoid wasting outer loop slots.

    def compute_spatial_locality_score(dim: str) -> int:
        """
        For each input that depends on this dim, compute its column-major
        stride (position-based). Return the MAX stride across all inputs.
        
        LOW score = dim is close to the contiguous (first) axis
                  = varying it gives spatially contiguous access
                  = should be INNER (fast varying in blockIdx)
        
        HIGH score = dim is far from the contiguous axis
                   = varying it jumps far in memory
                   = should be OUTER (slow varying in blockIdx)
        """
        max_stride = 0
        for inp in inputs:
            if dim in inp.indices:
                pos = inp.indices.index(dim)
                stride = 1
                for i in range(pos):
                    stride *= sizes[inp.indices[i]]
                max_stride = max(max_stride, stride)
        return max_stride

    for a in analysis:
        a['spatial_locality_score'] = compute_spatial_locality_score(a['dim'])

    # Sort: outer → inner
    #   - High reload_cost → outer (descending)
    #   - High spatial_locality_score → outer (descending, i.e. low = inner)
    #   - Low num_blocks → inner (ascending)
    analysis.sort(
        key=lambda x: (
            x['reload_cost_bytes'],       # primary: high → outer
            x['spatial_locality_score'],   # secondary: high → outer (low → inner)
            -x['num_blocks'],             # tertiary: many blocks → outer
        ),
        reverse=True
    )

    order = [a['dim'] for a in analysis]

    # Post-process: move dimensions with numBlk=1 to outermost positions.
    # Their position doesn't affect performance (only 1 block), but placing
    # them at the outer end avoids them occupying meaningful inner slots
    # and makes the output more intuitive.
    trivial = [a['dim'] for a in analysis if a['num_blocks'] == 1]
    nontrivial = [a['dim'] for a in analysis if a['num_blocks'] > 1]
    order = trivial + nontrivial

    # Generate CUDA code
    cuda_code = generate_cuda_code(order, sizes, tile_sizes,
                                   contraction_indices=contraction_indices)

    return BlockOrderResult(order=order, analysis=analysis, cuda_code=cuda_code)


def generate_cuda_code(order: List[str], sizes: Dict[str, int],
                       tile_sizes: Dict[str, int],
                       contraction_indices: List[str] = None,
                       prefix: str = "") -> str:
    """
    Generate complete CUDA code for block scheduling:
      1. TILE_* defines for all dimensions
      2. Host-side numBlk calculation and grid launch config
      3. Device-side blockIdx decomposition

    order[0] = outermost (slowest), order[-1] = innermost (fastest).
    """
    P = prefix  # optional prefix for namespacing (e.g. "TCCG_1_")
    all_dims = list(dict.fromkeys(order + (contraction_indices or [])))
    sections = []

    # ---- Section 1: Tile size defines ----
    defines = []
    defines.append("// ============================================================")
    defines.append("// Tile size defines")
    defines.append("// ============================================================")
    for dim in all_dims:
        if dim in tile_sizes:
            defines.append(f"#define {P}TILE_{dim.upper()} \t{tile_sizes[dim]}")
    sections.append("\n".join(defines))

    # ---- Section 2: Host-side code (numBlk + grid config) ----
    host = []
    host.append("// ============================================================")
    host.append("// Host-side: block count calculation and grid launch")
    host.append("// ============================================================")
    for dim in order:
        tile = tile_sizes.get(dim, sizes[dim])
        host.append(f"const int numBlk_{dim} = "
                     f"(size_{dim} + {P}TILE_{dim.upper()} - 1) / {P}TILE_{dim.upper()};")

    # Total grid size
    grid_parts = [f"numBlk_{dim}" for dim in order]
    grid_expr = " * ".join(grid_parts)
    host.append(f"const int gridSize = {grid_expr};")
    host.append("")
    host.append(f"// Launch: kernel<<<gridSize, blockSize, sharedMemSize>>>"
                f"(..., {', '.join(f'numBlk_{d}' for d in order)});")
    sections.append("\n".join(host))

    # ---- Section 3: Device-side blockIdx decomposition ----
    device = []
    device.append("// ============================================================")
    device.append(f"// Device-side: blockIdx decomposition (outer → inner)")
    device.append(f"// Order: {' → '.join(order)}")
    device.append(f"//   outermost (slowest): {order[0]}")
    device.append(f"//   innermost (fastest): {order[-1]}")
    device.append("// ============================================================")

    if len(order) == 1:
        device.append(f"const int blk_idx_{order[0]} = blockIdx.x;")
    elif len(order) == 2:
        d0, d1 = order
        device.append(f"const int blk_idx_{d0} = blockIdx.x / numBlk_{d1};")
        device.append(f"const int blk_idx_{d1} = blockIdx.x % numBlk_{d1};")
    else:
        device.append("int tmp_blkIdx;")
        for i, dim in enumerate(order):
            inner_dims = order[i + 1:]
            if i < len(order) - 1:
                divisor_parts = [f"numBlk_{d}" for d in inner_dims]
                divisor = " * ".join(divisor_parts)

                if i == 0:
                    device.append(f"const int blk_idx_{dim} = "
                                  f"blockIdx.x / ({divisor});")
                    device.append(f"tmp_blkIdx = blockIdx.x % ({divisor});")
                else:
                    device.append(f"const int blk_idx_{dim} = "
                                  f"tmp_blkIdx / ({divisor});")
                    device.append(f"tmp_blkIdx = tmp_blkIdx % ({divisor});")
            else:
                device.append(f"const int blk_idx_{dim} = tmp_blkIdx;")

    sections.append("\n".join(device))

    return "\n\n".join(sections)


def print_analysis(result: BlockOrderResult):
    """Pretty-print the optimization analysis."""
    print("=" * 70)
    print("TENSOR CONTRACTION BLOCK SCHEDULING ANALYSIS")
    print("=" * 70)

    print(f"\nOptimal blockIdx order (outer → inner): {' → '.join(result.order)}")
    print(f"  Outermost (slowest varying): {result.order[0]}")
    print(f"  Innermost (fastest varying): {result.order[-1]}")

    print("\n" + "-" * 70)
    print("Per-dimension analysis (sorted by reload cost, descending):")
    print("-" * 70)

    for a in result.analysis:
        print(f"\n  Dimension '{a['dim']}': size={a['size']}, "
              f"tile={a['tile_size']}, blocks={a['num_blocks']}")
        print(f"    Footprint per block: {a['total_footprint_bytes']:,} bytes "
              f"({a['total_footprint_bytes']/1024:.1f} KB)")
        print(f"    Reload cost per step: {a['reload_cost_bytes']:,} bytes "
              f"({a['reload_cost_bytes']/1024:.1f} KB)")
        print(f"    Spatial locality score: {a['spatial_locality_score']:,} "
              f"(low = contiguous = good for inner)")
        print(f"    Cache pressure (reload × blocks): "
              f"{a['cache_pressure']:,} bytes "
              f"({a['cache_pressure']/1024/1024:.1f} MB)")

        for t in a['tensors']:
            dep = "DEPENDS" if t['depends_on_dim'] else "independent"
            reuse_pct = (t['reused_elements'] / t['footprint_elements'] * 100
                         if t['footprint_elements'] > 0 else 0)
            print(f"      {t['tensor']}: {dep}, "
                  f"footprint={t['footprint_bytes']:,}B, "
                  f"reuse={reuse_pct:.0f}%, "
                  f"reload={t['reload_bytes']:,}B")

    print("\n" + "-" * 70)
    print("Generated CUDA code:")
    print("-" * 70)
    print(result.cuda_code)
    print()


# ============================================================================
# Examples
# ============================================================================

def example_tccg2(dtype):
    """t3[a,b,c] += t2[b,d,a] * v2[d,c]"""
    print("\n" + "=" * 70)
    print("Example: t3[a,b,c] += t2[d,c,a] * v2[b,d]")
    print("=" * 70)

    result = optimize_block_order(
        output_indices=['a', 'b', 'c'],
        input_specs=[
            (['d', 'c', 'a'], "t2"),
            (['b', 'd'],      "v2"),
        ],
        contraction_indices=['d'],
        sizes={'a': 320, 'b': 24, 'c': 296, 'd': 312},
        tile_sizes={'a': 32, 'b': 32, 'c': 16, 'd': 16},
        element_size=dtype
    )
    print_analysis(result)
    return result


def example_ccsd_t2(dtype):
    """t3[a,b,c,d] += t2[a,e,b,f] * v2[d,f,c,e]  (from previous kernel)"""
    print("\n" + "=" * 70)
    print("Example: t3[a,b,c,d] += t2[a,e,b,f] * v2[d,f,c,e]")
    print("=" * 70)

    result = optimize_block_order(
        output_indices=['a', 'b', 'c', 'd'],
        input_specs=[
            (['a', 'e', 'b', 'f'], "t2"),
            (['d', 'f', 'c', 'e'], "v2"),
        ],
        contraction_indices=['e', 'f'],
        sizes={'a': 64, 'b': 64, 'c': 64, 'd': 64, 'e': 32, 'f': 32},
        tile_sizes={'a': 8, 'b': 8, 'c': 8, 'd': 8, 'e': 4, 'f': 1},
        element_size=dtype
    )
    print_analysis(result)
    return result


def example_gemm(dtype):
    """C[i,j] += A[i,k] * B[k,j]  (standard GEMM for validation)"""
    print("\n" + "=" * 70)
    print("Example: C[i,j] += A[i,k] * B[k,j]  (standard GEMM)")
    print("=" * 70)

    result = optimize_block_order(
        output_indices=['i', 'j'],
        input_specs=[
            (['i', 'k'], "A"),
            (['k', 'j'], "B"),
        ],
        contraction_indices=['k'],
        sizes={'i': 4096, 'j': 4096, 'k': 4096},
        tile_sizes={'i': 128, 'j': 128, 'k': 32},
        element_size=dtype
    )
    print_analysis(result)
    return result


def example_custom():
    """Interactive custom tensor contraction."""
    print("\n" + "=" * 70)
    print("Custom tensor contraction")
    print("=" * 70)

    print("\nEnter the contraction equation.")
    print("Format: output[indices] += input1[indices] * input2[indices]")
    print("Example: t3[a,b,c] += t2[b,d,a] * v2[d,c]")

    eq = input("\nEquation: ").strip()

    # Simple parser
    import re
    # Match: name[i,j,...] += name[i,j,...] * name[i,j,...]
    pattern = r'(\w+)\[([\w,]+)\]\s*\+=\s*(\w+)\[([\w,]+)\]\s*\*\s*(\w+)\[([\w,]+)\]'
    m = re.match(pattern, eq)
    if not m:
        print("Could not parse equation. Please use format: out[a,b] += in1[a,k] * in2[k,b]")
        return

    out_name, out_idx_str, in1_name, in1_idx_str, in2_name, in2_idx_str = m.groups()
    out_indices = [x.strip() for x in out_idx_str.split(',')]
    in1_indices = [x.strip() for x in in1_idx_str.split(',')]
    in2_indices = [x.strip() for x in in2_idx_str.split(',')]

    # Determine contraction indices
    all_input_indices = set(in1_indices + in2_indices)
    contraction = list(all_input_indices - set(out_indices))
    print(f"Contraction indices: {contraction}")

    # Get sizes
    all_indices = sorted(set(out_indices + in1_indices + in2_indices))
    sizes = {}
    tile_sizes = {}
    for idx in all_indices:
        s = int(input(f"  size_{idx} = "))
        sizes[idx] = s
        t = int(input(f"  tile_{idx} = "))
        tile_sizes[idx] = t

    elem = int(input("  Element size (4=float, 8=double): ") or "4")

    result = optimize_block_order(
        output_indices=out_indices,
        input_specs=[
            (in1_indices, in1_name),
            (in2_indices, in2_name),
        ],
        contraction_indices=contraction,
        sizes=sizes,
        tile_sizes=tile_sizes,
        element_size=elem,
    )
    print_analysis(result)
    return result


if __name__ == "__main__":
    dtype = 4
    example_tccg2(dtype)
    # example_ccsd_t2(dtype)
    # example_gemm(dtype)