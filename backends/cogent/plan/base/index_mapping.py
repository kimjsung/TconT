import math
import itertools
import sys
from typing import NamedTuple


class FragmentAssignmentResult(NamedTuple):
    fragment_a_side: str    # a가 있는 쪽 fragment ("a" 또는 "a1")
    fragment_no_a:   str    # a가 없는 쪽 fragment index
    loop_a_side:     str    # a가 있는 쪽 outer loop
    loop_no_a:       str    # a가 없는 쪽 outer loop
    block_a_side:    list   # a가 있는 쪽 thread block (없으면 [])
    block_no_a:      list   # a가 없는 쪽 thread block (없으면 [])
    loop_order:      list   # 실제 loop 순서: no_a_loop → a_loop → no_a_frag → a_frag
    tile_index:      str    # tile 크기(>1) internal index (internal 1개면 None)
    step_index:      str    # step=1 internal index        (internal 1개면 None)
    big_tensor:      str    # tile index가 첫 번째인 tensor (internal 1개면 None)
    small_tensor:    str    # 나머지 tensor                 (internal 1개면 None)


def t3_stride_val(idx: str, t3_indices: list[str], size_map: dict) -> int:
    base = idx.rstrip("12")
    if base not in t3_indices: return None
    pos = t3_indices.index(base)
    return math.prod(size_map[i] for i in t3_indices[:pos]) if pos > 0 else 1


def tensor_stride(tensor_indices: list[str], idx: str, size_map: dict) -> int:
    if idx not in tensor_indices: return None
    pos = tensor_indices.index(idx)
    return math.prod(size_map[i] for i in tensor_indices[:pos]) if pos > 0 else 1


def compute_loop_score(loop_order: list[str], t3_indices: list[str], size_map: dict) -> float:
    n = len(loop_order)
    score = 0.0
    for loop_pos, idx in enumerate(loop_order):
        stride = t3_stride_val(idx, t3_indices, size_map)
        if stride is None: continue
        weight = math.prod(size_map[loop_order[i].rstrip("12")] for i in range(loop_pos+1, n))
        score += weight / stride
    return score


def get_external_indices(tensor_indices: list[str], internal_indices: list[str]) -> list[str]:
    external = [i for i in tensor_indices if i not in internal_indices]
    if len(external) == 1:
        return [f"{external[0]}1", f"{external[0]}2"]
    return external


def _decide_no_a_placement(no_a_external, loop_a_side, t3_indices, size_map, no_a_indices_fvi, idx_flag):
    is_split = (len(no_a_external) == 2 and
                no_a_external[0].rstrip("12") == no_a_external[1].rstrip("12"))
    
    if is_split:
        return no_a_external[0], no_a_external[1], []
    if len(no_a_external) == 1:
        return no_a_external[0], None, []

    # fragment/loop 2개만 고르고 나머지는 모두 block
    best_score, best = -1, (None, None, [])
    for frag, loop_idx in itertools.permutations(no_a_external, 2):
        if idx_flag == 1 :
            if no_a_indices_fvi not in [frag, loop_idx] :
                continue

        loop_order = [i for i in [loop_idx, loop_a_side, frag, "a"] if i]
        s = compute_loop_score(loop_order, t3_indices, size_map)
        if s > best_score:
            best_score = s
            block = [i for i in no_a_external if i not in (frag, loop_idx)]
            best = (frag, loop_idx, block)

    return best


def _assign_fragment_and_loop(
    internal_indices: list[str],
    a_side_indices:   list[str],
    no_a_indices:     list[str],
    t3_indices:       list[str],
    size_map:         dict,
) -> tuple:
    a_side_external = get_external_indices(a_side_indices, internal_indices)

    # a가 유일한 external: a1=fragment, a2=loop
    if a_side_external == ["a1", "a2"]:
        fragment_a_side = "a1"
        loop_a_side     = "a2"
        block_a_side    = []
    else:
        fragment_a_side   = "a"
        a_loop_candidates = [i for i in a_side_external
                             if i != "a" and i.rstrip("12") != "a"]
        is_a_split = (len(a_side_external) == 2 and
                      a_side_external[0].rstrip("12") == a_side_external[1].rstrip("12"))

        if len(a_loop_candidates) == 0:
            loop_a_side, block_a_side = [], []
        elif len(a_loop_candidates) == 1:
            loop_a_side, block_a_side = a_loop_candidates[0], []
        elif is_a_split:
            loop_a_side  = a_loop_candidates[0]
            block_a_side = a_loop_candidates[1:]
        else:
            first_idx = a_side_indices[0]
            if first_idx in a_loop_candidates:
                loop_a_side  = first_idx
            else:
                loop_a_side  = min(a_loop_candidates, key=lambda i: t3_stride_val(i, t3_indices, size_map) or float("inf"))
            block_a_side = [i for i in a_loop_candidates if i != loop_a_side]
    
    if no_a_indices[0] not in internal_indices :
        idx_flag = 1
    else :
        idx_flag = 0
    # a가 없는 쪽 처리
    no_a_ext = get_external_indices(no_a_indices, internal_indices)
    frag_no_a, loop_no_a, block_no_a = _decide_no_a_placement(no_a_ext, loop_a_side, t3_indices, size_map, no_a_indices[0], idx_flag)

    return fragment_a_side, loop_a_side, block_a_side, frag_no_a, loop_no_a, block_no_a


def _assign_single_internal(
    internal_indices: list[str],
    t2_indices:       list[str],
    v2_indices:       list[str],
    t3_indices:       list[str],
    size_map:         dict,
) -> FragmentAssignmentResult:
    a_side, no_a = (t2_indices, v2_indices) if "a" in t2_indices else (v2_indices, t2_indices)
    frag_a, loop_a, block_a, frag_no_a, loop_no_a, block_no_a = _assign_fragment_and_loop(internal_indices, a_side, no_a, t3_indices, size_map)
    loop_order = [i for i in [loop_no_a, loop_a, frag_no_a, frag_a] if i]

    return FragmentAssignmentResult(
        frag_a, frag_no_a, loop_a, loop_no_a, block_a, block_no_a,
        loop_order, None, None, None, None)


def _assign_double_internal(
    internal_indices: list[str],
    t2_indices:       list[str],
    v2_indices:       list[str],
    t3_indices:       list[str],
    size_map:         dict,
) -> FragmentAssignmentResult:
    first_in_internal = [i for i in internal_indices if i in {t2_indices[0], v2_indices[0]}]

    def tile_score(tile_idx):
        big = t2_indices if t2_indices[0] == tile_idx else \
              v2_indices if v2_indices[0] == tile_idx else \
              (t2_indices if (tensor_stride(t2_indices, tile_idx, size_map) or float("inf")) <=
                             (tensor_stride(v2_indices, tile_idx, size_map) or float("inf")) else v2_indices)
        ext = [i for i in big if i not in internal_indices and i != "a"]
        s   = sum(t3_stride_val(i, t3_indices, size_map) for i in ext
                  if t3_stride_val(i, t3_indices, size_map))
        return 1.0 / s if s > 0 else float("inf")

    tile_idx = first_in_internal[0] if len(first_in_internal) == 1 \
               else max(internal_indices, key=tile_score)
    step_idx = next(i for i in internal_indices if i != tile_idx)


    if t2_indices[0] == tile_idx:
        big_tensor, small_tensor = "t2", "v2"
        big, small = t2_indices, v2_indices
    elif v2_indices[0] == tile_idx:
        big_tensor, small_tensor = "v2", "t2"
        big, small = v2_indices, t2_indices
    else:
        if (tensor_stride(t2_indices, tile_idx, size_map) or float("inf")) <= (tensor_stride(v2_indices, tile_idx, size_map) or float("inf")) :
            big_tensor, small_tensor = "t2", "v2"
            big, small = t2_indices, v2_indices
        else :
            big_tensor, small_tensor = "v2", "t2"
            big, small = v2_indices, t2_indices
    
    if (big[0] in internal_indices) and (small[0] in internal_indices) and (big[0] != small[0]) :
        size_big_ext = 1
        for idx in big :
            if idx not in internal_indices :
                size_big_ext *= size_map[idx]

        size_small_ext = 1
        for idx in small :
            if idx not in internal_indices :
                size_small_ext *= size_map[idx]

        if size_big_ext < size_small_ext :
            tile_idx, step_idx = step_idx, tile_idx
        
    a_side, no_a = (t2_indices, v2_indices) if "a" in t2_indices else (v2_indices, t2_indices)
    frag_a, loop_a, block_a, frag_no_a, loop_no_a, block_no_a = _assign_fragment_and_loop(internal_indices, a_side, no_a, t3_indices, size_map)
    loop_order = [i for i in [loop_no_a, loop_a, frag_no_a, frag_a] if i]

    return FragmentAssignmentResult(
        frag_a, frag_no_a, loop_a, loop_no_a, block_a, block_no_a,
        loop_order, tile_idx, step_idx, big_tensor, small_tensor)


def assign_fragment(
    t3_indices:       list[str],
    internal_indices: list[str],
    t2_indices:       list[str],
    v2_indices:       list[str],
    size_map:         dict,
) -> FragmentAssignmentResult:
    assert len(internal_indices) in (1, 2)

    if len(internal_indices) == 1 :
        result = _assign_single_internal(internal_indices, t2_indices, v2_indices, t3_indices, size_map)
    else :
        result = _assign_double_internal(internal_indices, t2_indices, v2_indices, t3_indices, size_map)

    return result


def assign_mapping(l_tensors, index_to_extent):
    r = assign_fragment(l_tensors[0], l_tensors[1], l_tensors[2], l_tensors[3], index_to_extent)

    if r.tile_index :
        internal = [r.tile_index, r.step_index]
    else :
        internal = l_tensors[1]
    
    frag_n = [r.fragment_a_side]
    for i in r.block_a_side :
        frag_n.append(i)

    frag_m = [r.fragment_no_a]
    
    reg_n = [r.loop_a_side]
    reg_m = [r.loop_no_a]
    # print(f"Internal indices: {internal}", file=sys.stderr)
    # print(f"frag_n : {frag_n}, frag_m : {frag_m}, reg_n : {reg_n}, reg_m : {reg_m}", file=sys.stderr)
    # index_mapping = [internal, frag_n, frag_m, reg_n, reg_m]
    
    if "a" in l_tensors[2] :
        len_m = len(l_tensors[3]) - len(l_tensors[1])
        if len_m > 1 :
            for i in r.block_no_a :
                reg_m.append(i)
            index_mapping = [internal, frag_n, reg_m, reg_n, frag_m]
        else :
            for i in r.block_no_a :
                frag_m.append(i)
            index_mapping = [internal, frag_n, frag_m, reg_n, reg_m]
    else :
        len_m = len(l_tensors[2]) - len(l_tensors[1])
        if len_m > 1 :
            for i in r.block_no_a :
                reg_m.append(i)
            index_mapping = [internal, frag_n, reg_m, reg_n, frag_m]
        else :
            for i in r.block_no_a :
                frag_m.append(i)
            index_mapping = [internal, frag_n, frag_m, reg_n, reg_m]

    # print(f"Assigned mapping: {index_mapping}", file=sys.stderr)
    return index_mapping
    
