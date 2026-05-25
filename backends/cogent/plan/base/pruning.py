import math, re
import numpy as np
from typing import NamedTuple
from itertools import product as iproduct

class SplitFlagsResult(NamedTuple):
    m_val:         float
    n_val:         float
    big:           float
    small:         float
    a_flag:        int
    double_k_flag: bool

def compute_split_flags(t2_indices, v2_indices, k_indices, m, n, index_mapping, out_fvi) -> SplitFlagsResult:
    if out_fvi in t2_indices:
        n_indices, m_indices = t2_indices, v2_indices
        n_val, m_val = m, n
    else:
        n_indices, m_indices = v2_indices, t2_indices
        n_val, m_val = n, m

    n_external_len = len(n_indices) - len(k_indices)
    m_external_len = len(m_indices) - len(k_indices)

    if n_indices[0] in k_indices and m_indices[0] in k_indices and n_indices[0] != m_indices[0] :
        if index_mapping[0][0] == n_indices[0] :
            big, small, a_flag = n_val, m_val, 1
        else :
            big, small, a_flag = m_val, n_val, 0
        double_k_flag = 1
    else :
        if n_external_len > m_external_len:
            if m_val >= 2.0 * n_val:
                big, small, a_flag = m_val, n_val, 0
            else:
                big, small, a_flag = n_val, m_val, 1
        else:
            if n_val >= 2.0 * m_val:
                big, small, a_flag = n_val, m_val, 1
            else:
                big, small, a_flag = m_val, n_val, 0
        double_k_flag = 0

    return SplitFlagsResult(m_val, n_val, big, small, a_flag, double_k_flag)


def collect_split_cand_simple(tile_cand, frag_cand, out_fvi, data_type):
    result = []
    for l_index, tile_size in tile_cand:
        for cand in frag_cand:
            frag = f"{l_index[0]}1"
            reg  = f"{l_index[0]}2"
            
            if data_type == "DOUBLE" :
                if out_fvi == l_index[0] :
                    if cand == 16 :
                        continue
            else :
                if cand == 32 :
                    continue
            
            if tile_size // cand < 1 :
                continue
            result.append([[frag, cand], [reg, tile_size // cand]])
    
    return result


def collect_split_cand_mapped(tile_cand, frag_cand, small_fvi_cand, mapping_key, small_tile, size_map, out_fvi, data_type):
    result = []
    swap_flag = 0

    for l_index, tile_size in tile_cand :
        if small_tile[0] in l_index :
            fvi_flag = 1
        else :
            fvi_flag = 0

        for i in l_index :
            if i in mapping_key :
                continue
            for cand in frag_cand :
                reg_size   = tile_size // cand
                mapped_idx = mapping_key[0]

                if fvi_flag :
                    if data_type == "DOUBLE" :
                        if i == small_tile[0] and tile_size < 128 :
                            if 8 not in small_fvi_cand :
                                small_fvi_cand.append(8)
                    else :
                        if i == small_tile[0] and tile_size < 256 :
                            if 16 not in small_fvi_cand :
                                small_fvi_cand.append(16)

                    if i == small_tile[0] and reg_size not in small_fvi_cand :
                        if (i not in [out_fvi, f"{out_fvi}1", f"{out_fvi}2"]) and (mapped_idx not in [out_fvi, f"{out_fvi}1", f"{out_fvi}2"]) and (cand in small_fvi_cand) :
                            result.append([[i, cand], [mapped_idx, reg_size]])
                            swap_flag = 1
                            continue
                        else :
                            continue
                    elif mapped_idx == small_tile[0] and cand not in small_fvi_cand :
                        continue

                result.append([[mapped_idx, cand], [i, reg_size]])

    return result, swap_flag


def pow2_in_range(lo, hi) :
    lo = int(lo)
    hi = int(hi)

    if hi < 1 or lo > hi :
        return []

    lo = max(lo, 1)
    k_lo = int(np.ceil(np.log2(lo)))
    k_hi = int(np.floor(np.log2(hi)))

    if k_lo > k_hi :
        return []
    
    return [1 << k for k in range(k_lo, k_hi + 1)]


def tail_partial_ratio(x, base) :
    remain = x % base

    if remain == 0 :
        return 0.0
    
    tail_ratio = 1.0 - (remain / base)

    return tail_ratio


def tb_partial_ratio(x, base) :
    a = math.floor(x / base)
    b = math.ceil(x / base)
    tb_partial_ratio = 1.0 - (float)(a / b)

    if a == 0 :
        return 1.0

    return tb_partial_ratio


def get_big_fvi_cand(big_tile, k_indices, k_cand, size_map, index_mapping, out_fvi, data_type, reference_mapping=None):
    if reference_mapping is None:
        reference_mapping = index_mapping

    if data_type == "DOUBLE" :
        if (index_mapping[2][0] == big_tile[0]) or (index_mapping[4][0] == big_tile[0]):
            if size_map[big_tile[0]] % 2 == 0:
                return [8, 16]
            else :
                return [8]
        elif out_fvi == big_tile[0] :
            if size_map[big_tile[0]] % 2 == 0:
                return [8, 16]
            else :
                return [8]
        elif big_tile[0] in k_indices:
            return k_cand
        elif (size_map[big_tile[0]] % 16 == 0) or ((tb_partial_ratio(size_map[big_tile[0]], 16) < 0.2) and (tail_partial_ratio(size_map[big_tile[0]], 16) <= 0.5) and (size_map[big_tile[0]] % 2 == 0)) :
            return [16]
        else:
            return [8]
    else :
        frag_slot, _ = _get_big_mapping_slots(big_tile, out_fvi)
        if reference_mapping[frag_slot][0] == big_tile[0]:
            return [16]
        return [8, 16]
    

def get_frag_reg(big_tile, index_mapping, out_fvi):
    if out_fvi in big_tile:
        return index_mapping[1][0], index_mapping[3][0]
    else:
        return index_mapping[2][0], index_mapping[4][0]
    

def _get_big_mapping_slots(big_tile, out_fvi):
    if out_fvi in big_tile:
        return 1, 3
    return 2, 4


def _should_swap_big_mapping(big_tile, big_mapped_with_size, index_mapping, out_fvi, data_type):
    swap_votes = 0
    keep_votes = 0
    big_fvi = big_tile[0]

    if data_type != "FLOAT":
        return False

    if big_fvi == out_fvi:
        return False

    frag_slot, reg_slot = _get_big_mapping_slots(big_tile, out_fvi)
    if len(index_mapping[frag_slot]) > 1 or len(index_mapping[reg_slot]) > 1:
        return False
    if big_fvi != index_mapping[frag_slot][0] and big_fvi != index_mapping[reg_slot][0]:
        return False

    for pair in big_mapped_with_size:
        if len(pair) != 2:
            continue

        if pair[0][0] == big_fvi:
            big_size = pair[0][1]
            other_size = pair[1][1]
        elif pair[1][0] == big_fvi:
            big_size = pair[1][1]
            other_size = pair[0][1]
        else:
            continue

        if big_size < other_size:
            swap_votes += 1
        else:
            keep_votes += 1

    return swap_votes > 0 and swap_votes >= keep_votes


def _swap_big_mapping_roles(big_tile, index_mapping, out_fvi):
    frag_slot, reg_slot = _get_big_mapping_slots(big_tile, out_fvi)
    index_mapping[frag_slot], index_mapping[reg_slot] = index_mapping[reg_slot], index_mapping[frag_slot]


def _filter_swapped_big_mapping_candidates(big_mapped_with_size, frag):
    filtered = []

    for pair in big_mapped_with_size:
        frag_pair = None
        for item in pair:
            if item[0] == frag:
                frag_pair = item
                break

        if frag_pair is None:
            continue

        if frag_pair[1] >= 16:
            filtered.append(pair)

    return filtered


def get_big_mapped_with_size(big_tile, tile_big, frag, reg, frag_cand, big_fvi_cand, k_indices, big_indices, out_fvi, data_type):
    result = []
    is_simple = (len(big_indices) - len(k_indices) == 1)

    if data_type == 'DOUBLE' :
        if is_simple:
            for tile in tile_big:
                for cand in frag_cand:
                    result.append([[frag, cand], [reg, tile // cand]])
        else :
            if big_tile[0] in k_indices:
                for tile in tile_big:
                    for cand in frag_cand:
                        reg_size = tile // cand
                        if reg_size <= 16:
                            result.append([[frag, cand], [reg, reg_size]])

            elif big_tile[0] == frag:
                for tile in tile_big:
                    for cand in big_fvi_cand:
                        reg_size = tile // cand
                        result.append([[frag, cand], [reg, reg_size]])

            else:
                for tile in tile_big:
                    for cand in big_fvi_cand:
                        tmp_cand = cand

                        while tmp_cand >= 1:
                            frag_size = tile // tmp_cand
                            if frag_size == 8 or frag_size == 16:
                                result.append([[frag, frag_size], [reg, tmp_cand]])
                                break
                            elif frag_size < 8:
                                tmp_cand = tmp_cand // 2  # cand를 줄여서 frag_size를 키움
                            else:  # frag_size > 16
                                tmp_cand = tmp_cand * 2  # cand를 늘려서 frag_size를 줄임
                                if tmp_cand > tile:
                                    break
    else :
        if is_simple:
            for tile in tile_big:
                for cand in frag_cand:
                    if frag == out_fvi :
                        if cand == 32 :
                            continue
                    result.append([[frag, cand], [reg, tile // cand]])
        else :
            if big_tile[0] in k_indices:
                for tile in tile_big:
                    for cand in frag_cand:
                        reg_size = tile // cand
                        if reg_size <= 32:
                            if frag == out_fvi :
                                if cand == 32 :
                                    continue
                            result.append([[frag, cand], [reg, reg_size]])

            elif big_tile[0] == frag:
                for tile in tile_big:
                    for cand in big_fvi_cand:
                        reg_size = tile // cand
                        if frag == out_fvi :
                            if cand == 32 :
                                continue
                        result.append([[frag, cand], [reg, reg_size]])

            else:
                for tile in tile_big:
                    for cand in big_fvi_cand:
                        tmp_cand = cand
                        while tmp_cand >= 1:
                            frag_size = tile // tmp_cand
                            if frag_size == 16 or frag_size == 32:
                                if frag == out_fvi and frag_size == 32 :
                                    break
                                result.append([[frag, frag_size], [reg, tmp_cand]])
                                break
                            elif frag_size < 16:
                                tmp_cand = tmp_cand // 2  # cand를 줄여서 frag_size를 키움
                                if tmp_cand < 16 :
                                    break
                            else:  # frag_size > 16
                                tmp_cand = tmp_cand * 2  # cand를 늘려서 frag_size를 줄임
                                if tmp_cand > tile:  # 무한루프 방지
                                    break

    return result

def get_base_name(index_name):
    return re.sub(r'\d+', '', index_name)


def fill_remaining_indices(comb, with_a_input, without_a_input, k_indices):
    tile_list = comb

    existing_bases = set(get_base_name(name) for name, _ in tile_list)

    def get_to_add(input_indices):
        return [
            idx for idx in input_indices
            if idx not in k_indices
            and get_base_name(idx) not in existing_bases
        ]

    to_add_with_a    = get_to_add(with_a_input)
    to_add_without_a = get_to_add(without_a_input)

    result = (
        tile_list[:2]
        + [[idx, 1] for idx in to_add_with_a]
        + tile_list[2:]
        + [[idx, 1] for idx in to_add_without_a]
    )

    return result


def get_alpha_key(index_name):
    return re.sub(r'\d+', '', index_name)


def find_divisors(comb, data_type, a_flag, small_ext_cnt):
    front_val = comb[0][1] * comb[1][1] # n
    back_val  = comb[2][1] * comb[3][1] # m

    if data_type == "DOUBLE" :
        size_cond = (comb[0][1] * comb[2][1]) // 32
    else :
        size_cond = (comb[0][1] * comb[2][1]) // 64

    second_idx  = get_alpha_key(comb[1][0])
    fourth_idx  = get_alpha_key(comb[3][0])
    second_is_closer = second_idx < fourth_idx

    inherently_unbalanced = max(front_val, back_val) / min(front_val, back_val) > 4

    if data_type == "DOUBLE" :
        divisor_cands = [1, 2, 4, 8]
    else :
        divisor_cands = [1, 2, 4, 8, 16]
    
    results = []

    for d_front, d_back in iproduct(divisor_cands, divisor_cands):
        if d_front * d_back > 8:
            # print(f"Skipping divisor pair ({d_front}, {d_back}) because their product exceeds 8", file=sys.stderr)
            continue
        if d_front * d_back != size_cond:
            # print(f"Skipping divisor pair ({d_front}, {d_back}) because their product {d_front * d_back} does not match size condition {size_cond}", file=sys.stderr)
            continue

        front_result = front_val // d_front
        back_result  = back_val  // d_back

        if front_result == 0 or back_result == 0:
            # print(f"Skipping divisor pair ({d_front}, {d_back}) because it results in zero dimension (front_result: {front_result}, back_result: {back_result})", file=sys.stderr)
            continue
        if not inherently_unbalanced and max(front_result, back_result) / min(front_result, back_result) > 2:
            # print(f"Skipping divisor pair ({d_front}, {d_back}) because it results in an unbalanced split (front_result: {front_result}, back_result: {back_result})", file=sys.stderr)
            continue

        if data_type == "DOUBLE" :
            if (front_result * back_result // 64) * 4 > 128:
                # print(f"Skipping divisor pair ({d_front}, {d_back}) because it results in too large of a tile (front_result: {front_result}, back_result: {back_result})", file=sys.stderr)
                continue
        else :
            if (front_result * back_result // 256) * 8 > 128:
            # print(f"Skipping divisor pair ({d_front}, {d_back}) because it results in too large of a tile (front_result: {front_result}, back_result: {back_result})", file=sys.stderr)
                continue

        if data_type == "FLOAT" :
            if a_flag == 0 :
                if front_result < back_result :
                    continue
        else :
            if a_flag == 0 and small_ext_cnt == 1 :
                if front_result < back_result :
                    continue

        results.append((d_front, d_back))

    return results


def select_tile(x, internal_len, small_ext_cnt, big_ext_cnt, data_type):
    def get_min_val(fx, data_type):
        if data_type == "DOUBLE" :
            if fx < 16 :
                return 8
            if fx < 32 :
                return 16
            if fx < 64 :
                return 16
            if fx < 128 :
                return 32
            else :
                return 64
        else :
            if fx < 32 :
                return 16
            if fx < 64 :
                return 32
            if fx < 512 :
                return 64
            else :
                if small_ext_cnt == 1 :
                    return 64
                else :
                    return 128

    def get_max_val(fx, data_type):
        if data_type == "DOUBLE" :
            if fx < 16:  return 16
            if fx < 32:  return 32
            if fx < 64:  return 32
            if fx < 128: return 64
            else:        return 64
        else :
            if fx < 32:  return 32
            if fx < 64:  return 64
            if fx < 512:  return 64
            # if fx < 256: return 128
            else:        return 128

    if x <= 8 : 
        return 8
    
    fx = f(x)
    if data_type == "DOUBLE" :
        MAX_TILE = 64
    else :
        MAX_TILE = 128
    
    lo = get_min_val(fx, data_type)
    hi = min(MAX_TILE, get_max_val(fx, data_type))

    if internal_len > 1 and small_ext_cnt == 1 and big_ext_cnt == 1 :
        if lo / 2 >= 8 :
            lo /= 2
        if hi / 2 >= 8 :
            hi /= 2

    pows = pow2_in_range(lo, hi)
    pows = [t for t in pows if t < 2 * x]
    
    if not pows :
        return lo
    
    scored = [[t, partial_decision(x, t)] for t in pows]
    scored.sort(key=lambda v: (v[1][0], -v[0]))

    selected = scored[0][0]
    if internal_len > 1 and small_ext_cnt == 1 and big_ext_cnt == 1 :
        while selected > 0 and math.ceil(x / selected) < 15:
            selected = selected // 2

        if data_type == "DOUBLE" :
            if selected <= 8:
                selected = 8
        else :
            if selected <= 16:
                selected = 16
    
    return selected


def merge_indexed_items(lst):
    result = []
    seen = set()
    
    for item in lst:
        base = re.sub(r'\d+$', '', item)
        if base not in seen:
            seen.add(base)
            result.append(base)
    
    return result


def f(x) :
    size = 1 << int(np.log2(x))
    return size


def partial_decision(x, base) :
    tail = tail_partial_ratio(x, base)
    tb = tb_partial_ratio(x, base)

    return [tail, tb, tail * tb]


def tile_k_range(k, k_cand, data_type) :    
    k_values = [kt for kt in k_cand]

    result = {}
    for kt in k_values :
        max_stage = min(math.ceil(k / kt), 4)
        if max_stage < 1 :
            max_stage = 1
        
        if data_type == "DOUBLE" :
            if kt == 4:
                if max_stage >= 3 :
                    min_stage = 3
                else :
                    min_stage = 1
            elif kt == 16 :
                max_stage = min(2, max_stage)
                min_stage = 1
            else :
                max_stage = min(3, max_stage)
                min_stage = 1
        else :
            if kt == 8:
                if max_stage >= 3 :
                    min_stage = 3
                else :
                    min_stage = 1
            elif kt == 32 :
                max_stage = min(2, max_stage)
                min_stage = 1
            else :
                max_stage = min(2, max_stage)
                min_stage = 1

        if min_stage <= max_stage :
            result[kt] = list(range(min_stage, max_stage + 1))
        else :
            result[kt] = []
    
    tile_candidates = []
    for kt, stages in result.items() :
        for st in stages :
            tile_candidates.append([kt, st])

    return tile_candidates


def tile_range_small_v2(x, internal_len, small_ext_cnt, big_ext_cnt, t3_indices, small_indices, k_indices, size_map, index_mapping, out_fvi, data_type) :
    from itertools import combinations
    ext_mapped = [index for index in small_indices if index not in k_indices]

    if out_fvi in ext_mapped :
        flag = 1
        if len(ext_mapped) > 1 :
            if small_indices[0] in t3_indices and small_indices[0] != out_fvi :
                mapped = [[out_fvi, small_indices[0]]]
            else :
                candidates = [x for x in small_indices if x != out_fvi and x != small_indices[0]]
                for item in t3_indices :
                    if item != out_fvi and item in candidates :
                        mapped = [[out_fvi, item]]
                        break
        else :
            mapped = [ext_mapped]
    else :
        flag = 0
        if len(ext_mapped) > 1 :
            if small_indices[0] in t3_indices :
                candidates = [x for x in ext_mapped if x != small_indices[0]]
                mapped = [[c, small_indices[0]] for c in candidates]
            else :
                mapped = [list(c) for c in combinations(ext_mapped, 2)]
        else :
            mapped = [ext_mapped]

    mapped_with_size = []
    for comb in mapped :
        size = 1
        for c in comb :
            size *= size_map[c]
        mapped_with_size.append([size, comb])

    tmp_tile_cand = []
    for cnt, (x, comb) in enumerate(mapped_with_size) :
        tile = select_tile(x, internal_len, small_ext_cnt, big_ext_cnt, data_type)
        tmp_tile_cand.append([comb, tile])

    if flag :
        tmp_index_mapped = [index_mapping[1][0], index_mapping[3][0]]
    else :
        tmp_index_mapped = [index_mapping[2][0], index_mapping[4][0]]
    
    tmp_index_mapped = sorted(merge_indexed_items(tmp_index_mapped))

    tile_cand = []
    for comb, size in tmp_tile_cand :
        if sorted(comb) == tmp_index_mapped :
            tile_cand.append([comb, size])
        else :
            continue

    return tile_cand


def tile_range_big(x, small, k, big_internal_flag, small_internal_flag, big_tile, a_flag, internal_len, size_map, index_mapping, small_ext_cnt, out_fvi, data_type) :
    if a_flag :
        frag_mapped = index_mapping[1][0]
        reg_mapped = index_mapping[3][0]
    else :
        frag_mapped = index_mapping[2][0]
        reg_mapped = index_mapping[4][0]
    
    if len(frag_mapped) > 1 :
        split_flag = 1
        index_size = size_map[frag_mapped[0]]
    else :
        split_flag = 0
        index_size = size_map[frag_mapped] * size_map[reg_mapped]

    pow2_size = f(index_size)

    if data_type == "DOUBLE" :
        if split_flag :
            if pow2_size >= 2048 :
                if tail_partial_ratio(index_size, 128) <= 0.5 :
                    tile_size = 128
                else :
                    tile_size = 64
            elif pow2_size >= 1024 :
                tile_size = 64
            elif pow2_size >= 256 :
                tile_size = 32
            else :
                tile_size = 16

            if internal_len > 1 and big_internal_flag and small_internal_flag and big_tile[0] == index_mapping[0][0] :
                tile_size *= 2

            l_tile = [tile_size]
        else :
            if pow2_size <= 32 :
                min_val = 16
                max_val = 32
            elif pow2_size < 2048 :
                min_val = 32
                max_val = 64
            elif pow2_size < 8192 :
                if internal_len > 1 :
                    min_val = 128
                    max_val = 128
                else :
                    min_val = 64
                    max_val = 128
            else :
                if out_fvi == big_tile[0] :
                    min_val = 64
                    max_val = 128
                else :
                    if size_map[big_tile[0]] % 2 != 0 :
                        min_val = 64
                        max_val = 128
                    else : 
                        min_val = 128
                        max_val = 256

            lo = min_val
            hi = max_val
            pows = pow2_in_range(lo, hi)
            
            l_tile_partial_ratio = []
            for tile in pows :
                l_tile_partial_ratio.append([tile, partial_decision(x, tile)])
            
            l_tile = [v[0] for v in l_tile_partial_ratio]

            if internal_len > 1 and big_internal_flag and small_internal_flag and big_tile[0] != index_mapping[0][0] :
                original = l_tile.copy()
                tmp = [tile // 2 for tile in original if tile > min_val]

                l_tile = sorted(set(original + tmp))
    else :
        if split_flag :
            if pow2_size >= 4096 :
                tile_size = 128
            elif pow2_size >= 2048 :
                tile_size = 128
            elif pow2_size >= 512 :
                tile_size = 64
            elif pow2_size >= 256 :
                tile_size = 64
            else :
                tile_size = 32

            l_tile = [tile_size]
        else :
            if pow2_size <= 64 :
                min_val = 64
                max_val = 128
            elif pow2_size <= 1024 :
                min_val = 128
                max_val = 128
            elif pow2_size < 8192 :
                if out_fvi == big_tile[0] :
                    min_val = 128
                    max_val = 128
                else :
                    if ((max(small, k) / min(small, k)) < 1.2) :
                        min_val = 128
                        max_val = 128
                    else :
                        min_val = 256
                        max_val = 256
            elif pow2_size < 16384 :
                min_val = 256
                max_val = 256
            else :
                if out_fvi == big_tile[0] :
                    min_val = 128
                    if ((max(small, k) / min(small, k)) < 1.2) :
                        max_val = 128
                    else :
                        max_val = 256
                else :
                    if (size_map[big_tile[0]] % 2 != 0) and (size_map[big_tile[0]] % 4 != 0):
                        min_val = 128
                        if ((max(small, k) / min(small, k)) < 1.2) :
                            max_val = 128
                        else :
                            max_val = 256
                    else :
                        if (small_ext_cnt == 1) and ((max(small, k) / min(small, k)) < 1.2):
                            min_val = 128
                            max_val = 128
                        else :
                            min_val = 256
                            max_val = 256

            lo = min_val
            hi = max_val
            pows = pow2_in_range(lo, hi)

            l_tile_partial_ratio = []
            for tile in pows :
                l_tile_partial_ratio.append([tile, partial_decision(x, tile)])

            l_tile = [v[0] for v in l_tile_partial_ratio]

            if internal_len > 1 and big_internal_flag and small_internal_flag and big_tile[0] != index_mapping[0][0] :
                original = l_tile.copy()
                tmp = [tile // 2 for tile in original if tile > min_val]

                l_tile = sorted(set(original + tmp))

    return l_tile


def make_full_comb(external_comb, tile_k, index_mapping, a_flag, data_type, out_fvi) :
    config_struct = []
    for tk_l, comb_l in iproduct(tile_k, external_comb) :
        tk = tk_l[0]
        stage = tk_l[1]
        tile_comb = comb_l[0]
        shape = comb_l[1]
        internal = index_mapping[0]
        adjusted_tile_comb = tile_comb
        
        if data_type == "DOUBLE" :
            if tk == 16 :
                tile = 1
                if a_flag :
                    n_mapped = [index_mapping[1][0], index_mapping[3][0]]
                    for tmp in tile_comb :
                        if tmp[0] in n_mapped :
                            tile *= tmp[1]
                else :
                    m_mapped = [index_mapping[2][0], index_mapping[4][0]]
                    for tmp in tile_comb :
                        if tmp[0] in m_mapped :
                            tile *= tmp[1]
                if tile == 256 :
                    continue
        else :
            if tk == 32 :
                big_reg_mapped = index_mapping[3][0] if a_flag else index_mapping[4][0]
                adjusted_tile_comb = []
                for idx, tile_size in tile_comb:
                    if idx == big_reg_mapped:
                        adjusted_tile_comb.append([idx, max(1, tile_size // 2)])
                    else:
                        adjusted_tile_comb.append([idx, tile_size])

                tile = 1
                if a_flag :
                    n_mapped = [index_mapping[1][0], index_mapping[3][0]]
                    for tmp in adjusted_tile_comb :
                        if tmp[0] in n_mapped :
                            tile *= tmp[1]
                else :
                    m_mapped = [index_mapping[2][0], index_mapping[4][0]]
                    for tmp in adjusted_tile_comb :
                        if tmp[0] in m_mapped :
                            tile *= tmp[1]
                if tile == 1024 :
                    continue

        full_tile_comb = []
        if len(internal) > 1 :
            full_tile_comb.append([internal[0], tk])
            full_tile_comb.append([internal[1], 1])
            for comb in adjusted_tile_comb :
                full_tile_comb.append(comb)
        else :
            full_tile_comb.append([internal[0], tk])
            for comb in adjusted_tile_comb :
                full_tile_comb.append(comb)

        config_struct.append([shape, [stage], full_tile_comb])

    return config_struct


def index_based_config_selection(l_tensors, index_to_extent, index_mapping, data_type) :
    t3_indices = l_tensors[0]
    k_indices = l_tensors[1]
    t2_indices = l_tensors[2]
    v2_indices = l_tensors[3]
    out_fvi = t3_indices[0]

    m = 1
    for idx in t2_indices :
        if idx not in k_indices :
            m *= index_to_extent[idx]

    n = 1
    for idx in v2_indices :
        if idx not in k_indices :
            n *= index_to_extent[idx]
    
    k = 1
    for idx in k_indices :
        k *= index_to_extent[idx]

    initial_index_mapping = [list(v) for v in index_mapping]

    result = compute_split_flags(t2_indices, v2_indices, k_indices, m, n, index_mapping, out_fvi)
    m_val, n_val, big, small, a_flag, double_k_flag = result

    if out_fvi in t2_indices :
        if a_flag == 1 :
            small_tile_fvi = v2_indices[0]
            big_tile_fvi = t2_indices[0]
            small_ext_cnt = len(v2_indices) - len(k_indices)
            big_ext_cnt = len(t2_indices) - len(k_indices)
            small_tile = v2_indices
            big_tile = t2_indices
        else :
            small_tile_fvi = t2_indices[0]
            big_tile_fvi = v2_indices[0]
            small_ext_cnt = len(t2_indices) - len(k_indices)
            big_ext_cnt = len(v2_indices) - len(k_indices)
            big_tile = v2_indices
            small_tile = t2_indices
        m_frag_rank = len(t2_indices) - 1
        m_reg_rank = len(t2_indices) + len(v2_indices) - len(k_indices) - 1
    else :
        if a_flag == 1 :
            small_tile_fvi = t2_indices[0]
            big_tile_fvi = v2_indices[0]
            small_ext_cnt = len(t2_indices) - len(k_indices)
            big_ext_cnt = len(v2_indices) - len(k_indices)
            big_tile = v2_indices
            small_tile = t2_indices
        else :
            small_tile_fvi = v2_indices[0]
            big_tile_fvi = t2_indices[0]
            small_ext_cnt = len(v2_indices) - len(k_indices)
            big_ext_cnt = len(t2_indices) - len(k_indices)
            small_tile = v2_indices
            big_tile = t2_indices
        m_frag_rank = len(v2_indices) - 1
        m_reg_rank = len(v2_indices) + len(t2_indices) - len(k_indices) - 1

    if small_tile_fvi in k_indices :
        small_internal_flag = True
    else :
        small_internal_flag = False

    if big_tile_fvi in k_indices :
        big_internal_flag = True
    else :
        big_internal_flag = False

    if out_fvi in t2_indices : # m = v2, n = t2
        if m_val < n_val :
            big_indices = t2_indices
            small_indices = v2_indices
        else :
            big_indices = v2_indices
            small_indices = t2_indices
    else :
        if m_val < n_val :
            big_indices = v2_indices
            small_indices = t2_indices
        else :
            big_indices = t2_indices
            small_indices = v2_indices

    if data_type == "DOUBLE" :  
        if double_k_flag :
            k_cand = [8]
        else :
            if big_tile[0] == small_tile[0] and big_tile[0] in k_indices :
                if index_to_extent[big_tile[0]] % 16 == 0 :
                    k_cand = [8, 16]
                else :
                    k_cand = [8]
            elif big_tile[0] in k_indices :
                if big / small >= 4 : 
                    if (tb_partial_ratio(index_to_extent[big_tile[0]], 16) <= 0.15) and (index_to_extent[big_tile[0]] % 2 == 0):
                        k_cand = [16]
                    else :
                        k_cand = [8]
                else :
                    k_cand = [8]
            elif small_tile[0] in k_indices :
                if (tb_partial_ratio(index_to_extent[small_tile[0]], 16) < 0.5) and (index_to_extent[big_tile[0]] % 2 == 0):
                    k_cand = [8, 16]
                else :
                    k_cand = [8]
            else :
                if (tb_partial_ratio(index_to_extent[index_mapping[0][0]], 16) <= 0.15) and (index_to_extent[index_mapping[0][0]] % 2 == 0):
                    k_cand = [4, 8, 16]
                else :
                    k_cand = [4, 8]
    else :
        if double_k_flag :
            k_cand = [16]
        else :
            if big_tile[0] == small_tile[0] and big_tile[0] in k_indices :
                if index_to_extent[big_tile[0]] % 32 == 0 :
                    k_cand = [16, 32]
                else :
                    k_cand = [16]
            elif big_tile[0] in k_indices :
                if big / small >= 4 : 
                    if (tb_partial_ratio(index_to_extent[big_tile[0]], 32) <= 0.35) and ((index_to_extent[big_tile[0]] % 2 == 0) or (index_to_extent[big_tile[0]] % 4 == 0)):
                        k_cand = [32]
                    else :
                        k_cand = [16]
                else :
                    k_cand = [16]
            elif small_tile[0] in k_indices :
                if (tb_partial_ratio(index_to_extent[small_tile[0]], 32) < 0.5) and ((index_to_extent[big_tile[0]] % 2 == 0) or (index_to_extent[big_tile[0]] % 4 == 0)):
                    k_cand = [16, 32]
                else :
                    k_cand = [16]
            else :
                if (tb_partial_ratio(index_to_extent[index_mapping[0][0]], 32) <= 0.15) and ((index_to_extent[index_mapping[0][0]] % 2 == 0) or (index_to_extent[index_mapping[0][0]] % 4 == 0)):
                    k_cand = [16, 32]
                else :
                    k_cand = [8, 16]

    import sys
    tile_k = tile_k_range(k, k_cand, data_type)
    
    tile_cand = tile_range_small_v2(small, len(k_indices), small_ext_cnt, big_ext_cnt, t3_indices, small_tile, k_indices, index_to_extent, index_mapping, out_fvi, data_type)

    if data_type == "DOUBLE" :
        if out_fvi == small_tile[0] :
            small_fvi_cand = [8]
        else :
            if (small_tile[0] not in k_indices) and ((index_to_extent[small_tile[0]] % 16 == 0) or ((tb_partial_ratio(index_to_extent[small_tile[0]], 16) <= 0.5) and (tail_partial_ratio(index_to_extent[small_tile[0]], 16) <= 0.5) and (index_to_extent[small_tile[0]] % 2 == 0))) :
                small_fvi_cand = [16]
            else : 
                if (small_tile[0] not in k_indices) :
                    small_fvi_cand = [4, 8]
                else :
                    if (index_to_extent[small_tile[0]] % 2 == 0) :
                        small_fvi_cand = [8, 16]
                    else :
                        small_fvi_cand = [4, 8]
    else :
        if out_fvi == small_tile[0] :
            small_fvi_cand = [16]
        else :
            if (small_tile[0] not in k_indices) and ((index_to_extent[small_tile[0]] % 32 == 0) or ((tb_partial_ratio(index_to_extent[small_tile[0]], 32) <= 0.5) and (tail_partial_ratio(index_to_extent[small_tile[0]], 32) <= 0.5) and ((index_to_extent[small_tile[0]] % 2 == 0) or (index_to_extent[small_tile[0]] % 4 == 0)))) :
                small_fvi_cand = [32]
            else : 
                if (small_tile[0] not in k_indices) :
                    small_fvi_cand = [8, 16]
                else :
                    if ((index_to_extent[small_tile[0]] % 2 == 0) or (index_to_extent[small_tile[0]] % 4 == 0)) :
                        small_fvi_cand = [16, 32]
                    else :
                        small_fvi_cand = [8, 16]

    #
    if data_type == "DOUBLE" :
        frag_cand = [8, 16]
    else :
        if small_ext_cnt == 1 and big_ext_cnt == 1 :
            frag_cand = [16]
        else :
            frag_cand = [16, 32]

    # mapping_key = Frag_mapped_index
    if out_fvi in small_tile:
        mapping_key = index_mapping[1]
    else:
        mapping_key = index_mapping[2]

    #
    is_simple = (len(small_tile) - len(k_indices) == 1)
    if is_simple:
        split_cand = collect_split_cand_simple(tile_cand, frag_cand, out_fvi, data_type)
        swap_flag = 0
    else:
        split_cand, swap_flag = collect_split_cand_mapped(tile_cand, frag_cand, small_fvi_cand, mapping_key, small_tile, index_to_extent, out_fvi, data_type)

    frag, reg = get_frag_reg(big_tile, index_mapping, out_fvi)

    #
    big_fvi_cand = get_big_fvi_cand(big_tile, k_indices, k_cand, index_to_extent, index_mapping, out_fvi, data_type, initial_index_mapping)

    tile_big = tile_range_big(
        big,
        small,
        k,
        big_internal_flag,
        small_internal_flag,
        big_tile,
        a_flag,
        len(k_indices),
        index_to_extent,
        index_mapping,
        small_ext_cnt,
        out_fvi,
        data_type)

    big_mapped_with_size = get_big_mapped_with_size(big_tile, tile_big, frag, reg, frag_cand, big_fvi_cand, k_indices, big_indices, out_fvi, data_type)

    if _should_swap_big_mapping(big_tile, big_mapped_with_size, index_mapping, out_fvi, data_type):
        _swap_big_mapping_roles(big_tile, index_mapping, out_fvi)

        big_fvi_cand = get_big_fvi_cand(big_tile, k_indices, k_cand, index_to_extent, index_mapping, out_fvi, data_type)

        tile_big = tile_range_big(
            big,
            small,
            k,
            big_internal_flag,
            small_internal_flag,
            big_tile,
            a_flag,
            len(k_indices),
            index_to_extent,
            index_mapping,
            small_ext_cnt,
            out_fvi,
            data_type)

        frag, reg = get_frag_reg(big_tile, index_mapping, out_fvi)

        big_mapped_with_size = get_big_mapped_with_size(big_tile, tile_big, frag, reg, frag_cand, big_fvi_cand, k_indices, big_indices, out_fvi, data_type)

        filtered_big_mapped_with_size = _filter_swapped_big_mapping_candidates(big_mapped_with_size, frag)
        if filtered_big_mapped_with_size:
            big_mapped_with_size = filtered_big_mapped_with_size
        else:
            _swap_big_mapping_roles(big_tile, index_mapping, out_fvi)

            big_fvi_cand = get_big_fvi_cand(big_tile, k_indices, k_cand, index_to_extent, index_mapping, out_fvi, data_type)

            tile_big = tile_range_big(
                big,
                small,
                k,
                big_internal_flag,
                small_internal_flag,
                big_tile,
                a_flag,
                len(k_indices),
                index_to_extent,
                index_mapping,
                small_ext_cnt,
                out_fvi,
                data_type)

            frag, reg = get_frag_reg(big_tile, index_mapping, out_fvi)

            big_mapped_with_size = get_big_mapped_with_size(big_tile, tile_big, frag, reg, frag_cand, big_fvi_cand, k_indices, big_indices, out_fvi, data_type)

    from itertools import product
    if data_type == "DOUBLE" :
        if a_flag :
            if is_simple :
                big_small_comb = [big + small for big, small in product(big_mapped_with_size, split_cand) if not ((big[0][1] == 16) and (small[0][1] == 16))]
            else :
                big_small_comb = [big + small for big, small in product(big_mapped_with_size, split_cand)]
        else :
            if is_simple :
                big_small_comb = [small + big for small, big in product(split_cand, big_mapped_with_size) if not ((big[0][1] == 16) and (small[0][1] == 16))]
            else :
                big_small_comb = [small + big for small, big in product(split_cand, big_mapped_with_size)]
    else :
        if a_flag :
            if is_simple :
                big_small_comb = [big + small for big, small in product(big_mapped_with_size, split_cand) if not ((big[0][1] == 32) and (small[0][1] == 32))]
            else :
                big_small_comb = [big + small for big, small in product(big_mapped_with_size, split_cand)]
        else :
            if is_simple :
                big_small_comb = [small + big for small, big in product(split_cand, big_mapped_with_size) if not ((big[0][1] == 32) and (small[0][1] == 32))]
            else :
                big_small_comb = [small + big for small, big in product(split_cand, big_mapped_with_size)]

    valid_combinations = []
    for comb in big_small_comb:
        divisors = find_divisors(comb, data_type, a_flag, small_ext_cnt)
        # print(f"Divisors for combination {comb}: {divisors}", file=sys.stderr)
        if not divisors:
            continue
        
        front_val = comb[0][1] * comb[1][1]
        back_val  = comb[2][1] * comb[3][1]
        # print(f"Front value: {front_val}, Back value: {back_val}", file=sys.stderr)
        if double_k_flag :
            if a_flag :
                flag = (front_val < back_val)
            else :
                flag = (back_val < front_val)
            if flag :
                continue

        for d_front, d_back in divisors:
            valid_combinations.append([comb, [d_front, d_back]])
    # print(f"valid_combination : {valid_combinations}", file=sys.stderr)
    external_comb = []
    for comb, shape in valid_combinations :
        if a_flag :
            results = fill_remaining_indices(comb, big_tile, small_tile, k_indices)
        else :
            results = fill_remaining_indices(comb, small_tile, big_tile, k_indices)
        external_comb.append([results, shape, comb])
    # print(f"external_comb: {external_comb}", file=sys.stderr)
    
    config_struct = make_full_comb(external_comb, tile_k, index_mapping, a_flag, data_type, out_fvi)

    return config_struct, swap_flag, m_frag_rank, m_reg_rank


def matches_struct(cfg, struct, swap_flag, m_frag_rank, m_reg_rank):
    if cfg.warp_shape != struct[0]:
        return False

    if cfg.stage != struct[1][0]:
        return False
    
    cfg_set = set(map(tuple, cfg.list_tile_sizes))
    
    if swap_flag:
        # 원본
        original = set(map(tuple, struct[2]))
        # swap된 버전
        swapped = struct[2][:]
        swapped[m_frag_rank] = [struct[2][m_frag_rank][0], struct[2][m_reg_rank][1]]
        swapped[m_reg_rank]  = [struct[2][m_reg_rank][0], struct[2][m_frag_rank][1]]
        swapped_set = set(map(tuple, swapped))
        
        if cfg_set != original and cfg_set != swapped_set:
            return False
    else:
        if cfg_set != set(map(tuple, struct[2])):
            return False
    
    return True


def matches_any_struct(cfg, config_struct, swap_flag, m_frag_rank, m_reg_rank):
    return any(matches_struct(cfg, struct, swap_flag, m_frag_rank, m_reg_rank) for struct in config_struct)


def apply_pruning(config, config_struct, swap_flag, m_frag_rank, m_reg_rank) :
    filtered = [cfg for cfg in config if matches_any_struct(cfg, config_struct, swap_flag, m_frag_rank, m_reg_rank)]

    return filtered
