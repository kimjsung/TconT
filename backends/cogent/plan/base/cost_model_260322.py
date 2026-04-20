import re
import math
import sys
import numpy as np
import pandas as pd
import tc_helper as tc_helper

# # ----------------------------------------------------------------
# # 상수 (한 곳에서만 관리)
# # ----------------------------------------------------------------
PIPELINE_CONSTS = dict(
    mma_lat_cycles  = 16.0,   # A100 FP64 DMMA latency
    mem_lat_cycles  = 350.0,  # 128B line DRAM latency (L2 miss 기준)
    issue_lat_cycles = 32.0,
    l2_hit_cycles   = 50.0,   # L2 hit latency (partial 재사용 보정용)
    stage_cap       = 5,
    exposed_floor   = 0.6,
)

A100_DEFAULT_CAPS = {
    "sm_count": 108,
    "warp_size": 32,
    "max_threads_per_sm": 2048,
    "max_warps_per_sm": 64,
    "max_blocks_per_sm": 32,
    "max_regs_per_sm": 65536,
    "max_smem_per_sm": 167936,
    "smem_per_block_cap": 49152,
}

def infer_internal_indices(t2, v2, t3):
    # A와 B에 공통이고 C에는 없는 인덱스 = contraction index
    t2_set, v2_set, t3_set = set(t2), set(v2), set(t3)
    return [idx for idx in t2 if (idx in v2_set) and (idx not in t3_set)]

def num_inner_iters(internal_indices, split_tile_size, rep_problem_sizes):
    iters = 1
    for idx in internal_indices:
        K  = int(tc_helper.tc_helper_find_value(rep_problem_sizes, idx))
        Kt = int(tc_helper.tc_helper_find_value(split_tile_size, idx))
        iters *= int(math.ceil(K / Kt))
    return int(iters)

def _estimate_regs_per_thread(each_config) :
    input_output_address = 2 * 3
    index = len(each_config.list_tensor_C) + len(each_config.list_FRAG_K)
    pipeline = 3
    shm_address = 2 * 2
    blk = len(each_config.list_tensor_C)
    output_base = 1
    fragment = (each_config.size_FRAG_X // 8) * (each_config.size_FRAG_Y // 8) * (each_config.size_REG_X // each_config.warp_shape[0]) * (each_config.size_REG_Y // each_config.warp_shape[1]) * 4
    else_reg = 1 + 1 + 1

    regs_per_thread = input_output_address + index + pipeline + shm_address + blk + output_base + fragment + else_reg

    return regs_per_thread

def smem_order(frag_mapping, reg_mapping, internal, mapped_a, mapped_b) :
    #
    SMEM_order_a = []
    if mapped_a[0] == frag_mapping[1] :
        # REG_Y
        SMEM_order_a.append(reg_mapping[1])
        # Contraction index
        for internal_index in internal :
            if internal_index in mapped_a :
                SMEM_order_a.append(internal_index)
                break
        # FRAG_Y
        SMEM_order_a.append(frag_mapping[1])
    elif mapped_a[0] == reg_mapping[1] :
        # FRAG_Y
        SMEM_order_a.append(frag_mapping[1])
        # Contraction index
        for internal_index in internal :
            if internal_index in mapped_a :
                SMEM_order_a.append(internal_index)
                break
        # REG_Y
        SMEM_order_a.append(reg_mapping[1])
    else :
        # REG_Y
        SMEM_order_a.append(reg_mapping[1])
        # FRAG_Y
        SMEM_order_a.append(frag_mapping[1])
        # Contraction index
        for internal_index in internal :
            if internal_index in mapped_a :
                SMEM_order_a.append(internal_index)
                break
    
    #
    SMEM_order_b = []
    if mapped_b[0] == frag_mapping[0] :
        # REG_X
        SMEM_order_b.append(reg_mapping[0])
        # Contraction index
        for internal_index in internal :
            if internal_index in mapped_b :
                SMEM_order_b.append(internal_index)
                break
        # FRAG_X
        SMEM_order_b.append(frag_mapping[0])
    elif mapped_b[0] == reg_mapping[0] :
        # FRAG_X
        SMEM_order_b.append(frag_mapping[0])
        # Contraction index
        for internal_index in internal :
            if internal_index in mapped_b :
                SMEM_order_b.append(internal_index)
                break
        # REG_X
        SMEM_order_b.append(reg_mapping[0])
    else :
        # REG_X
        SMEM_order_b.append(reg_mapping[0])
        # FRAG_X
        SMEM_order_b.append(frag_mapping[0])
        # Contraction index
        for internal_index in internal :
            if internal_index in mapped_b :
                SMEM_order_b.append(internal_index)
                break

    # print(f"FROM cost_model ||| SMEM order a : {SMEM_order_a}, SMEM order b : {SMEM_order_b}", file=sys.stderr)
    return SMEM_order_a[2], SMEM_order_b[2]

def smem_padding_size(size_frag_x, size_frag_y, size_internal, size_reg_x, size_reg_y, config) :
    mapped_index = config.list_FRAG_X + config.list_FRAG_Y + [config.list_FRAG_K[0]] + config.list_REG_X + config.list_REG_Y
    
    mapped_b = []
    for idx in config.list_tensor_A :
        if idx in mapped_index :
            mapped_b.append(idx)
    
    mapped_a = []
    for idx in config.list_tensor_B :
        if idx in mapped_index :
            mapped_a.append(idx)

    frag_mapping = [config.list_FRAG_X[0]] + [config.list_FRAG_Y[0]]
    reg_mapping = config.list_REG_X + config.list_REG_Y
    smem_y_fvi, smem_x_fvi = smem_order(frag_mapping, reg_mapping, config.list_FRAG_K, mapped_a, mapped_b)

    #
    wavefront_unit = 16
    padd_per_wavefront_y = wavefront_unit // size_frag_y
    padd_per_wavefront_x = wavefront_unit // size_frag_x
    
    #
    inner_frag_padd_x = 0
    inner_frag_padd_y = 0
    inter_reg_frag_padd_x = 0
    inter_reg_frag_padd_y = 0
    reg_y_padd = 0
    reg_x_padd = 0

    #
    a_double2_flag = config.double2_flag[1]
    b_double2_flag = config.double2_flag[0]

    #
    if smem_y_fvi == config.list_FRAG_K[0] :    # SMEM Order = [Reg_Y, Frag_Y, Internal]
        reg_y_padd = 0
    elif smem_y_fvi == config.list_FRAG_Y[0] : # SMEM Order = [Reg_Y, Internal, Frag_Y]
        if a_double2_flag :
            reg_y_padd = 0
        else :
            inner_frag_padd_y = ((8 * size_internal) // 32) * padd_per_wavefront_y
            frag_count_per_y = size_frag_y // 8
            if size_frag_y == 16 and padd_per_wavefront_y * (size_internal / 4) % 4 == 0 :
                inter_reg_frag_padd_y = wavefront_unit / 8
            else :
                inter_reg_frag_padd_y = 0
            reg_y_padd = (int)(inner_frag_padd_y * frag_count_per_y + inter_reg_frag_padd_y)
    else :                            # SMEM Order = [Frag_Y, Internal, Reg_Y]
        if a_double2_flag :
            per_row_internal = wavefront_unit // size_reg_y
            row_cnt = (4 + per_row_internal - 1) // per_row_internal
            if row_cnt == 1 :
                reg_y_padd = 0
            else :
                reg_y_padd = 2 * row_cnt
        else :
            reg_y_padd = wavefront_unit // size_reg_y

    if smem_x_fvi == config.list_FRAG_K[0] :    # SMEM Order = [Reg_X, Frag_X, Internal]
        reg_x_padd = 0
    elif smem_x_fvi == config.list_FRAG_X[0] :    # SMEM Order = [Reg_X, Internal, Frag_X]
        if b_double2_flag :
            reg_x_padd = 0
        else :
            inner_frag_padd_x = ((8 * size_internal) // 32) * padd_per_wavefront_x
            frag_count_per_x = size_frag_x // 8
            if size_frag_x == 16 and padd_per_wavefront_x * (size_internal / 4) % 4 == 0 :
                inter_reg_frag_padd_x = wavefront_unit / 8
            else :
                inter_reg_frag_padd_x = 0
            reg_x_padd = (int)(inner_frag_padd_x * frag_count_per_x + inter_reg_frag_padd_x)
    else :                              # SMEM Order = [Frag_X, Internal, Reg_X]
        if b_double2_flag :            
            per_row_internal = wavefront_unit // size_reg_x
            row_cnt = (4 + per_row_internal - 1) // per_row_internal
            if row_cnt == 1 :
                reg_x_padd = 0
            else :
                reg_x_padd = 2 * row_cnt
        #
        else :
            reg_x_padd = wavefront_unit // size_reg_x
    # print(f"smem_y_fvi : {smem_y_fvi}, smem_x_fvi : {smem_x_fvi}", file=sys.stderr)
    # print(f"in COST_MODEL ||| reg_y_padd : {reg_y_padd}, reg_x_padd : {reg_x_padd}", file=sys.stderr)
    #
    if a_double2_flag :
        if smem_y_fvi == config.list_REG_Y[0] :
            total_y_padding = size_frag_y * reg_y_padd
        else :
            total_y_padding = 0
    else :
        if smem_y_fvi == config.list_REG_Y[0] :
            total_y_padding = size_reg_y * reg_y_padd
        elif smem_y_fvi == config.list_FRAG_Y[0] :
            total_y_padding = size_reg_y * reg_y_padd
        else :
            total_y_padding = 0

    #
    if b_double2_flag :
        if smem_x_fvi == config.list_REG_X[0] :
            total_x_padding = size_frag_x * reg_x_padd
        else :
            total_x_padding = 0
    else :
        if smem_x_fvi == config.list_REG_X[0] :
            total_x_padding = size_reg_x * reg_x_padd
        elif smem_x_fvi == config.list_FRAG_X[0]  :
            total_x_padding = size_reg_x * reg_x_padd
        else :
            total_x_padding = 0

    return total_y_padding, total_x_padding, reg_y_padd, reg_x_padd

def _estimate_smem_per_block_bytes(config, smem_overhead_bytes=128, elem_bytes=8) :
    #
    fx    = int(getattr(config, "size_FRAG_X", 1))
    fy    = int(getattr(config, "size_FRAG_Y", 1))
    fk    = int(getattr(config, "size_FRAG_K", 1))
    rx    = int(getattr(config, "size_REG_X", 1))
    ry    = int(getattr(config, "size_REG_Y", 1))
    stage = int(getattr(config, "stage", 0))
    
    total_y_padding, total_x_padding, reg_y_padd, reg_x_padd = smem_padding_size(fx, fy, fk, rx, ry, config)

    config.reg_y_padd = reg_y_padd
    config.reg_x_padd = reg_x_padd
    config.padding = [total_x_padding, total_y_padding]

    #
    x_smem = (fx * rx * fk + total_x_padding) * stage * elem_bytes
    y_smem = (fy * ry * fk + total_y_padding) * stage * elem_bytes
    
    #
    smem_per_block = (x_smem + y_smem) + smem_overhead_bytes

    #
    config.smem_per_block = smem_per_block
    
    return int(smem_per_block)

def _estimate_cta_per_sm_active(threads_per_block, regs_per_thread, smem_per_block, caps) :
    """Return (cta_per_sm, bottleneck, B_hw, B_threads, B_warps, B_regs, B_smem)."""
    if threads_per_block <= 0:
        return (0, "invalid_threads", 0, 0, 0, 0, 0)

    warp_size = caps.get("warp_size", 32)
    warps_per_block = (threads_per_block + warp_size - 1) // warp_size
    if warps_per_block <= 0:
        return (0, "invalid_warps", 0, 0, 0, 0, 0)

    # per-block SMEM cap check
    smem_cap = caps["smem_per_block_cap"]
    if smem_per_block > smem_cap:
        return (0, "smem_per_block_exceeds_cap", caps["max_blocks_per_sm"], 0, 0, 0, 0)

    B_hw = caps["max_blocks_per_sm"]
    B_threads = caps["max_threads_per_sm"] // threads_per_block
    B_warps = caps["max_warps_per_sm"] // warps_per_block

    regs_per_block = regs_per_thread * threads_per_block
    B_regs = caps["max_regs_per_sm"] // regs_per_block if regs_per_block > 0 else 0

    B_smem = caps["max_smem_per_sm"] // smem_per_block if smem_per_block > 0 else B_hw

    limits = {
        "B_smem": B_smem,
        "B_regs": B_regs,
        "B_warps": B_warps,
        "B_threads": B_threads,
        "B_hw": B_hw,
    }

    cta = min(limits.values())

    # tie-breaking priority: smem -> regs -> warps -> threads -> hw
    bottleneck = "unknown"
    for k in ["B_smem", "B_regs", "B_warps", "B_threads", "B_hw"] :
        if limits[k] == cta :
            bottleneck = k
            break

    return (cta, bottleneck, B_hw, B_threads, B_warps, B_regs, B_smem)

def compute_exposed_mem_fraction_v2(
    tx_per_loop, mma_per_loop, stage,
    smem_per_stage_bytes, cta_per_sm_at_stage1,
    max_smem_per_sm, warps_per_tb,
    issue_interval=4.0,
    consts=PIPELINE_CONSTS,
):
    S = max(1, min(stage, consts['stage_cap']))

    # ── 단위: 1 warp, 1 loop 기준 ──────────────────────
    tx_per_warp = tx_per_loop / max(warps_per_tb, 1)
    T_mem       = tx_per_warp * consts['mem_lat_cycles']
    T_comp      = mma_per_loop * consts['mma_lat_cycles']

    if T_mem <= 0:
        return 1.0

    # ── smem → occupancy ───────────────────────────────
    smem_total   = smem_per_stage_bytes * S
    cta_limited  = (max_smem_per_sm // smem_total) if smem_total > 0 else cta_per_sm_at_stage1
    cta_actual   = min(cta_per_sm_at_stage1, cta_limited)
    occ_ratio    = cta_actual / max(cta_per_sm_at_stage1, 1)

    # ── 1) stage hiding ────────────────────────────────
    # S-1번의 compute로 hide 가능한 시간
    # occ가 낮으면 stage buffer를 채울 warp이 부족 → 효과 감소
    T_stage_hide = (S - 1) * T_comp * occ_ratio
    T_after_stage = max(0.0, T_mem - T_stage_hide)

    # ── 2) warp switching hiding ───────────────────────
    # active warps per SM = cta_actual × warps_per_tb
    # 이 warp들이 순차적으로 실행되면서 서로의 latency를 hide
    # hide 가능한 시간 = (active_warps - 1) × issue_interval
    # (1개 warp 기준: 나머지 warp들이 실행되는 동안 대기)
    active_warps_per_sm  = cta_actual * warps_per_tb
    T_warp_hide          = max(0.0, active_warps_per_sm - 1) * issue_interval
    warp_hide_frac       = min(1.0, T_warp_hide / max(T_mem, 1.0))

    T_after_warp = T_after_stage * (1.0 - warp_hide_frac)

    # ── 최종 exposed fraction ──────────────────────────
    frac = T_after_warp / T_mem
    return max(consts['exposed_floor'], min(1.0, frac))


def make_tile_dict(list_tile_sizes, combined_tile_size=None):
    """
    list_tile_sizes의 원본 + combined_tile_size의 base key를
    모두 포함하는 dict를 만든다.
    combined에 'b'가 있고 list_tile_sizes에 'b1','b2'만 있으면
    b1*b2를 'b'로 등록.
    """
    d = {k: int(v) for k, v in list_tile_sizes}

    if combined_tile_size is None:
        return d

    for key, val in combined_tile_size:
        key = str(key)
        if key not in d:
            # combined에는 있지만 tile_sizes에 없으면 combined 값을 직접 사용
            d[key] = int(val)

    return d

def make_rep_dict(rep_problem_size):
    return {k: int(v) for k, v in rep_problem_size}

def build_tb_cover_vectorized(
    tensor_idx_list, split_tile_size_dict, mapped_index_set,
    fvi_tile_size, double2_flag, num_warp,
    warp_shape,            # ← 추가: [warp_x, warp_y]
    frag_x_idx,            # ← 추가: FRAG_X에 매핑된 index
    frag_y_idx,            # ← 추가: FRAG_Y에 매핑된 index
    WARP_THREAD_X=8,
    WARP_THREAD_Y=4,
):
    n_idx    = len(tensor_idx_list)
    warp_ids = np.arange(num_warp, dtype=np.int32)

    warp_x = warp_shape[0]
    warp_y = warp_shape[1]

    # ---- thread layout: warp_shape 반영 ----
    tmp = 1
    thread_layout = {}
    for i, idx in enumerate(tensor_idx_list):
        tile = int(split_tile_size_dict[idx])

        if tmp >= 32:
            thread_layout[idx] = 1
        elif idx not in mapped_index_set:
            thread_layout[idx] = 0
        else:
            if i == 0:
                # FVI 처리 (기존과 동일)
                if double2_flag:
                    thread_layout[idx] = fvi_tile_size // 2
                    tmp *= fvi_tile_size // 2
                else:
                    thread_layout[idx] = fvi_tile_size
                    tmp *= fvi_tile_size
            else:
                # ── warp_shape 반영 ──────────────────────────
                if idx == frag_x_idx:
                    # x 방향: warp_x개 warp × WARP_THREAD_X threads
                    covered = min(tile, warp_x * WARP_THREAD_X)
                elif idx == frag_y_idx:
                    # y 방향: warp_y개 warp × WARP_THREAD_Y threads
                    covered = min(tile, warp_y * WARP_THREAD_Y)
                else:
                    rest    = 32 // tmp
                    covered = min(rest, tile)

                thread_layout[idx] = covered
                tmp *= covered

    # ---- warp_cover_base 계산 ----
    warp_cover_base = {}
    for i, idx in enumerate(tensor_idx_list):
        if i == 0 and double2_flag:
            warp_cover_base[idx] = thread_layout[idx] * 2
        else:
            warp_cover_base[idx] = thread_layout[idx]

    # ---- base_cover_arr: (n_idx,) ----
    base_cover_arr = np.array(                          # ← 여기서 정의
        [warp_cover_base[idx] for idx in tensor_idx_list], dtype=np.int32
    )

    # ---- mapped 차원의 tensor 내 위치 ----
    mapped_in_tensor = [idx for idx in tensor_idx_list if idx in mapped_index_set]
    mapped_dim_pos   = [tensor_idx_list.index(idx) for idx in mapped_in_tensor]  # ← 여기서 정의

    # ---- radices 계산 ----
    radices = []
    for key in mapped_in_tensor:
        base = warp_cover_base.get(key, 0)
        tile = split_tile_size_dict[key]
        radices.append(1 if base <= 0 else max(1, math.ceil(tile / base)))

    total_radix = 1
    for r in radices:
        total_radix *= r

    repeat_factor      = max(1, math.ceil(total_radix / num_warp))
    effective_num_warp = min(num_warp, total_radix)

    # ---- warp별 cover/start 배열 초기화 ----
    warp_start_arr = np.zeros((num_warp, n_idx), dtype=np.int32)
    warp_cover_arr = np.tile(base_cover_arr, (num_warp, 1))  # (W, n_idx)

    # ---- mixed-radix 분해 ----
    x = warp_ids.copy()
    for dim_pos, key, radix in zip(mapped_dim_pos, mapped_in_tensor, radices):
        base = warp_cover_base.get(key, 0)
        tile = int(split_tile_size_dict[key])
        step = x % radix
        x    = x // radix
        if base > 0:
            warp_begin         = (base * step).astype(np.int32)
            warp_end           = np.minimum(base * (step + 1), tile).astype(np.int32)
            warp_begin_clipped = np.minimum(warp_begin, tile)
            actual_cover       = np.maximum(warp_end - warp_begin_clipped, 0)
            warp_start_arr[:, dim_pos] = warp_begin_clipped
            warp_cover_arr[:, dim_pos] = actual_cover

    # total_radix < num_warp이면 초과 warp은 cover=0
    if effective_num_warp < num_warp:
        warp_cover_arr[effective_num_warp:, :] = 0

    return warp_cover_arr, warp_start_arr, repeat_factor


def warp_tx_vectorized_v2(
    tensor_idx_list,
    warp_cover_arr,
    warp_start_arr,
    tb_global_starts,
    rep_arr,
    vec_elems,
    contig_elems,
    tile_sizes_arr,      # ← 추가: np.array([tile_d[idx] for idx in tensor_idx_list])
    elem_bytes=8,
    cache_line_bytes=128,
):
    TB = tb_global_starts.shape[0]

    tb_start = tb_global_starts[:, np.newaxis, :]
    w_start  = warp_start_arr[np.newaxis, :, :]
    w_cover  = warp_cover_arr[np.newaxis, :, :]
    rep      = rep_arr[np.newaxis, np.newaxis, :]
    tile     = tile_sizes_arr[np.newaxis, np.newaxis, :]  # ← 추가

    global_start = tb_start + w_start

    # ── 수정: cover=0(unmapped)이면 tile 전체를 읽는 것으로 처리 ──
    # 기존: valid_c = clip(min(w_cover, rep - global_start), 0, None)
    # 수정: unmapped 차원은 w_cover 대신 tile로 대체
    is_unmapped  = (w_cover <= 0)                                   # (1, W, D) bool
    cover_to_use = np.where(is_unmapped, tile, w_cover)            # (1, W, D)
    valid_c      = np.clip(np.minimum(cover_to_use, rep - global_start), 0, None)

    # unmapped 차원을 모든 warp이 동일하게 읽으므로
    # warp 간 unique line 중복 제거: unmapped 차원이 있는 경우
    # 해당 차원의 tx를 warp 수로 나눠서 1번만 카운트
    # (모든 warp이 같은 라인을 읽으므로 실제 unique line은 동일)
    any_unmapped = is_unmapped.any(axis=2)  # (1, W) → True인 warp은 unmapped 차원 보유

    valid_elems = valid_c.prod(axis=2)  # (TB, W)

    # ── unmapped 차원 중복 제거 ──
    # unmapped 차원의 tile 크기 곱 = 모든 warp이 공통으로 읽는 부분
    unmapped_tile_prod = np.where(is_unmapped, tile, 1).prod(axis=2)  # (1, W)
    mapped_elems = np.where(
        unmapped_tile_prod.squeeze(0) > 0,
        valid_elems / unmapped_tile_prod.squeeze(0)[np.newaxis, :],
        valid_elems
    )  # (TB, W): mapped 차원만의 element 수

    # cache line 계산은 valid_elems 전체 기준,
    # 단 unmapped가 있는 warp은 1개 warp 기준으로만 카운트
    elems_per_line     = cache_line_bytes // elem_bytes
    lines_per_block    = max(1, math.ceil(contig_elems * elem_bytes / cache_line_bytes))
    num_contig_blocks  = np.where(valid_elems > 0,
                                  np.ceil(valid_elems / max(contig_elems, 1)), 0)
    cache_lines        = num_contig_blocks * lines_per_block  # (TB, W)

    # unmapped 차원이 있는 경우: 모든 warp이 동일 line → warp 0만 카운트
    # warp 0 외의 warp에서 unmapped가 있으면 0으로 마스킹
    num_warp = warp_cover_arr.shape[0]
    if num_warp > 1:
        unmapped_mask = any_unmapped.squeeze(0)  # (W,) bool
        # warp 0은 항상 포함, warp 1+에서 unmapped이면 제거
        duplicate_mask = np.zeros(num_warp, dtype=bool)
        duplicate_mask[1:] = unmapped_mask[1:]
        cache_lines[:, duplicate_mask] = 0

    load_issues = np.where(valid_elems > 0,
                           np.ceil(valid_elems / vec_elems), 0)
    load_issues[:, duplicate_mask] = 0  # issue도 동일하게 중복 제거

    tx_per_tb    = cache_lines.sum(axis=1).astype(np.float64)
    issue_per_tb = load_issues.sum(axis=1).astype(np.float64)
    return tx_per_tb, issue_per_tb


def compute_contig_elems(tensor_idx_list, each_config, tile_d, rep_d):
    """
    GM4의 size_continuous_elements 계산 로직을 함수로 분리.
    innermost부터 tile == rep_size인 동안 누적.
    """
    other_tensor = (each_config.list_tensor_B
                    if tensor_idx_list is each_config.list_tensor_A
                    else each_config.list_tensor_A)
    other_set = set(other_tensor)

    fvi_is_external = (tensor_idx_list[0] not in other_set)

    contig = 1
    is_cont = True
    for idx in tensor_idx_list:
        is_external = (idx not in other_set)
        # FVI 타입과 같은 종류(외부/내부)인 차원만 연속성 체크
        if is_external != fvi_is_external:
            break
        if is_cont:
            contig *= tile_d[idx]
            if tile_d[idx] != rep_d.get(idx, tile_d[idx]):
                is_cont = False
        else:
            break
    return max(1, contig)

def compute_contig_elems_warp_aware(tensor_idx_list, warp_cover_arr, tile_d):
    """
    warp 0 기준으로 innermost부터 연속 접근 element 수 계산.
    - unmapped(cover=0): tile 전체가 연속
    - mapped이고 cover==tile: 전체 커버 → 연속
    - mapped이고 cover<tile: 여기서 끊김
    """
    contig = 1
    for i, idx in enumerate(tensor_idx_list):
        cover_i = int(warp_cover_arr[0, i])   # warp 0 기준
        tile_i  = tile_d[idx]
        if cover_i <= 0:
            contig *= tile_i      # unmapped: 전체 연속
        elif cover_i >= tile_i:
            contig *= tile_i      # 전체 커버
        else:
            contig *= cover_i     # 부분 커버 → 끊김
            break
    return max(1, contig)


def compute_store_cost_warp_aware(each_config, tile_d, num_warp, num_TBs,
                                   elem_bytes=8, line_bytes=128):
    """
    store_matrix_sync 1회:
      8×8 fragment
      frag_x(a) 방향: 8 elems × 8 bytes = 64B 연속
      frag_y 방향: row 간 stride → 각 row가 독립적인 cache line 접근
      → 1 row = 64B = 128B cache line 1개 (절반만 사용)
      → 8 rows = 8 cache lines per store_matrix_sync
    
    load cost와 동일하게 128B cache line 수로 표현
    """
    warp_shape_x = each_config.warp_shape[0]
    warp_shape_y = each_config.warp_shape[1]

    size_REG_X = 1
    for idx in each_config.list_REG_X:
        size_REG_X *= tile_d[idx]
    size_REG_Y = 1
    for idx in each_config.list_REG_Y:
        size_REG_Y *= tile_d[idx]

    size_FRAG_X = each_config.size_FRAG_X
    size_FRAG_Y = each_config.size_FRAG_Y

    # ---- 1 warp당 fragment 개수 ----
    frags_reg_x  = size_REG_X  // warp_shape_x
    frags_reg_y  = size_REG_Y  // warp_shape_y
    frags_frag_x = size_FRAG_X // 8
    frags_frag_y = size_FRAG_Y // 8
    frags_per_warp = frags_reg_x * frags_reg_y * frags_frag_x * frags_frag_y

    # ---- store_matrix_sync 1회당 cache line 수 ----
    # frag_y 방향 stride로 인해 각 row가 독립적인 cache line 접근
    # 1 row = 8 elems × 8 bytes = 64B → 128B line 1개 (절반 사용)
    rows_per_frag      = 8
    row_bytes          = 8 * elem_bytes                          # = 64B
    lines_per_row      = math.ceil(row_bytes / line_bytes)       # = 1
    lines_per_call     = rows_per_frag * lines_per_row           # = 8

    # ---- 1 TB 총 store cache line 수 ----
    store_lines_per_tb = frags_per_warp * lines_per_call * num_warp
    store_lines_total  = store_lines_per_tb * num_TBs

    return float(store_lines_total), float(store_lines_per_tb)


def compute_cost_with_pipeline(
    total_load_tx, cost_store,
    tx_per_loop, mma_per_loop, stage,
    smem_per_stage_bytes, cta_per_sm_at_stage1,
    max_smem_per_sm, warps_per_tb, idx,
    issue_interval=4.0,
    consts=PIPELINE_CONSTS, 
):
    S = max(1, min(stage, consts['stage_cap']))

    # ── smem → occupancy ──────────────────────────────
    smem_total  = smem_per_stage_bytes * S
    cta_limited = (max_smem_per_sm // smem_total) if smem_total > 0 else cta_per_sm_at_stage1
    cta_actual  = min(cta_per_sm_at_stage1, cta_limited)
    occ_ratio   = cta_actual / max(cta_per_sm_at_stage1, 1)

    # ── 1 loop당 시간 ─────────────────────────────────
    tx_per_warp = tx_per_loop / max(warps_per_tb, 1)
    T_mem_loop  = tx_per_warp  * consts['mem_lat_cycles']
    T_comp_loop = mma_per_loop * consts['mma_lat_cycles']

    if T_mem_loop <= 0:
        return cost_store, 0.0

    # ── arithmetic intensity 반영 ─────────────────────
    # rho = T_mem / T_comp: 1보다 크면 memory bound, 작으면 compute bound
    # rho가 작을수록 pipeline hiding 효과가 큼
    rho = T_mem_loop / T_comp_loop

    # ── stage hiding 비율 ─────────────────────────────
    # S-1번의 compute로 hide 가능:
    # rho <= 1/(S-1)이면 완전 hide 가능
    # rho가 클수록 hide 비율 감소
    if S > 1:
        max_hide_by_stage = min(1.0, (S - 1) / max(rho, 1e-6)) * occ_ratio
    else:
        max_hide_by_stage = 0.0
    stage_hide_frac = min(max_hide_by_stage, 1.0)

    # ── warp hiding 비율 ──────────────────────────────
    active_warps   = cta_actual * warps_per_tb
    T_warp_hide    = max(0.0, active_warps - 1) * issue_interval
    warp_hide_frac = min(0.5, T_warp_hide / max(T_mem_loop, 1.0))

    # ── 전체 hiding ───────────────────────────────────
    total_hide_frac = min(stage_hide_frac + warp_hide_frac, 1.0)

    # ── bandwidth floor ───────────────────────────────
    # rho가 작을수록(compute bound에 가까울수록) floor를 낮게 설정
    # → compute bound kernel은 memory가 거의 완전히 hide됨
    # rho >= 1: memory bound → floor = 0.15
    # rho << 1: compute bound → floor ≈ 0
    BANDWIDTH_FLOOR = max(0.05, min(0.15, rho * 0.15))
    total_hide_frac = min(total_hide_frac, 1.0 - BANDWIDTH_FLOOR)

    mem_efficiency    = 1.0 - total_hide_frac
    effective_load_tx = total_load_tx * mem_efficiency
    total_cost        = effective_load_tx + cost_store

    return total_cost, mem_efficiency

def compute_cost_with_pipeline_v2(
    total_load_tx, cost_store,
    tx_per_loop, mma_per_loop, stage,
    smem_per_stage_bytes, cta_per_sm_at_stage1,
    max_smem_per_sm, warps_per_tb, idx,
    issue_interval=4.0,
    consts=PIPELINE_CONSTS,
):
    S = max(1, min(stage, consts['stage_cap']))

    # ── smem → occupancy ──────────────────────────────
    smem_total  = smem_per_stage_bytes * S
    cta_limited = (max_smem_per_sm // smem_total) if smem_total > 0 else cta_per_sm_at_stage1
    cta_actual  = min(cta_per_sm_at_stage1, cta_limited)
    occ_ratio   = cta_actual / max(cta_per_sm_at_stage1, 1)

    if tx_per_loop <= 0:
        return cost_store, 0.0


    RIDGE_POINT  = 4.85   # A100 FP64 flops/byte
    FLOPS_PER_MMA = 512   # FP64 FMA count per MMA inst (8×8×4×2)
    BYTES_PER_LINE = 128

    ai  = (mma_per_loop * FLOPS_PER_MMA) / (tx_per_loop * BYTES_PER_LINE)
    rho = RIDGE_POINT / max(ai, 1e-6)

    # ── stage hiding ──────────────────────────────────
    if S > 1:
        stage_hide_frac = min(1.0, (S - 1) / rho) * occ_ratio
    else:
        stage_hide_frac = 0.0

    # ── warp hiding ───────────────────────────────────
    # T_mem_loop: warp당 기준
    T_mem_loop  = (tx_per_loop / max(warps_per_tb, 1)) * consts['mem_lat_cycles']
    active_warps   = cta_actual * warps_per_tb
    T_warp_hide    = max(0.0, active_warps - 1) * issue_interval
    warp_hide_frac = min(0.5, T_warp_hide / max(T_mem_loop, 1.0))

    # ── 전체 hiding ───────────────────────────────────
    total_hide_frac = min(stage_hide_frac + warp_hide_frac, 1.0)

    # ── bandwidth floor ───────────────────────────────
    BANDWIDTH_FLOOR = max(0.05, min(0.15, rho * 0.15))
    total_hide_frac = min(total_hide_frac, 1.0 - BANDWIDTH_FLOOR)

    mem_efficiency    = 1.0 - total_hide_frac
    effective_load_tx = total_load_tx * mem_efficiency
    total_cost        = effective_load_tx + cost_store

    return total_cost, mem_efficiency

def tc_gen_cost_model_v2(each_config, l_comb, idx):
    # ---- 0) 사전 dict 변환 (함수당 1회) ----
    tile_d = make_tile_dict(each_config.list_tile_sizes, each_config.combined_tile_size)
    rep_d  = make_rep_dict(each_config.list_representative_problem_size)
    mapped_set = set(
        [each_config.list_FRAG_X[0], each_config.list_FRAG_Y[0], each_config.list_FRAG_K[0]]
        + each_config.list_REG_X + each_config.list_REG_Y
    )

    t2, v2, t3 = each_config.list_tensor_A, each_config.list_tensor_B, each_config.list_tensor_C
    num_warp = (each_config.size_FRAG_X * each_config.size_FRAG_Y) // 32
    num_TBs  = int(each_config.num_TBs)

    if t3[0] in t2 :
        t2_double2 = each_config.double2_flag[0]
        v2_double2 = each_config.double2_flag[1]
    else :
        t2_double2 = each_config.double2_flag[1]
        v2_double2 = each_config.double2_flag[0]

    # ---- 1) warp cover/start 배열 계산 (vectorized) ----
    frag_x_idx = each_config.list_FRAG_X[0]
    frag_y_idx = each_config.list_FRAG_Y[0]
    warp_shape = each_config.warp_shape          # [warp_x, warp_y]

    t2_cover, t2_start, t2_repeat = build_tb_cover_vectorized(
        t2, tile_d, mapped_set, tile_d[t2[0]], t2_double2, num_warp,
        warp_shape=warp_shape,
        frag_x_idx=frag_x_idx,
        frag_y_idx=frag_y_idx,
    )
    v2_cover, v2_start, v2_repeat = build_tb_cover_vectorized(
        v2, tile_d, mapped_set, tile_d[v2[0]], v2_double2, num_warp,
        warp_shape=warp_shape,
        frag_x_idx=frag_x_idx,
        frag_y_idx=frag_y_idx,
    )

    # ---- 2) TB 시작 좌표 행렬 (기존 blk_idx 로직, 이미 numpy) ----
    idx_tile_list = list(reversed(each_config.combined_tile_size))
    idx_names     = [x[0] for x in idx_tile_list]
    tile_arr      = np.array([tile_d[x[0]] for x in idx_tile_list], dtype=np.int64)
    rep_arr_full  = np.array([rep_d[x[0]]  for x in idx_tile_list], dtype=np.int64)
    num_tb_each   = np.ceil(rep_arr_full / tile_arr).astype(np.int64)
    n_idx         = len(idx_tile_list)

    bidx = np.arange(num_TBs, dtype=np.int64)
    blk_cols = []
    for i in range(n_idx):
        stride = int(num_tb_each[i+1:].prod()) if i + 1 < n_idx else 1
        blk_cols.append(bidx // stride)
        bidx = bidx % stride
    blk_idx = np.stack(blk_cols, axis=1)           # (TB, D)
    tb_global_starts = (blk_idx * tile_arr).astype(np.int64)  # (TB, D)

    # ---- 3) tensor별 rep_arr 슬라이스 (차원 순서 맞추기) ----
    def rep_for_tensor(tensor_idx_list):
        return np.array([rep_d[idx] for idx in tensor_idx_list], dtype=np.int64)

    def tb_starts_for_tensor(tensor_idx_list):
        """(TB, D_tensor) — 해당 텐서의 인덱스 순서로 tb_global_starts 슬라이스"""
        col_map = {name: i for i, name in enumerate(idx_names)}
        cols = [col_map[idx] for idx in tensor_idx_list if idx in col_map]
        if not cols:
            return np.zeros((num_TBs, len(tensor_idx_list)), dtype=np.int64)
        result = np.zeros((num_TBs, len(tensor_idx_list)), dtype=np.int64)
        for out_i, idx in enumerate(tensor_idx_list):
            if idx in col_map:
                result[:, out_i] = tb_global_starts[:, col_map[idx]]
        return result

    # ---- 4) warp-aware tx 계산 (vectorized) ----
    internal_indices = infer_internal_indices(t2, v2, t3)
    inner_iters = num_inner_iters(internal_indices, each_config.list_tile_sizes, each_config.list_representative_problem_size)

    t2_vec = 2 if t2_double2 else 1
    v2_vec = 2 if v2_double2 else 1

    t2_contig = compute_contig_elems_warp_aware(t2, t2_cover, tile_d)
    v2_contig = compute_contig_elems_warp_aware(v2, v2_cover, tile_d)

    t2_tile_arr = np.array([tile_d[idx] for idx in t2], dtype=np.int64)
    v2_tile_arr = np.array([tile_d[idx] for idx in v2], dtype=np.int64)

    t2_tx_per_tb, t2_issue_per_tb = warp_tx_vectorized_v2(
        t2, t2_cover, t2_start,
        tb_starts_for_tensor(t2), rep_for_tensor(t2),
        vec_elems=t2_vec,
        contig_elems=t2_contig,
        tile_sizes_arr=t2_tile_arr,   # ← 추가
    )
    v2_tx_per_tb, v2_issue_per_tb = warp_tx_vectorized_v2(
        v2, v2_cover, v2_start,
        tb_starts_for_tensor(v2), rep_for_tensor(v2),
        vec_elems=v2_vec,
        contig_elems=v2_contig,
        tile_sizes_arr=v2_tile_arr,   # ← 추가
    )

    t2_tx_total   = float((t2_tx_per_tb * inner_iters * t2_repeat).sum())
    v2_tx_total   = float((v2_tx_per_tb * inner_iters * v2_repeat).sum())
    total_load_tx = t2_tx_total + v2_tx_total

    # issue도 동일하게
    t2_issue_total   = float((t2_issue_per_tb * inner_iters * t2_repeat).sum())
    v2_issue_total   = float((v2_issue_per_tb * inner_iters * v2_repeat).sum())
    total_load_issue = t2_issue_total + v2_issue_total

    # tx_per_loop도 repeat 반영
    tx_per_loop_bw    = float((t2_tx_per_tb * t2_repeat + v2_tx_per_tb * v2_repeat).mean())
    tx_per_loop_issue = float((t2_issue_per_tb * t2_repeat + v2_issue_per_tb * v2_repeat).mean())

    ISSUE_TO_LINE_RATIO = PIPELINE_CONSTS['issue_lat_cycles'] / PIPELINE_CONSTS['mem_lat_cycles']
    tx_per_loop = max(tx_per_loop_bw, tx_per_loop_issue * ISSUE_TO_LINE_RATIO)

    smem_per_block = _estimate_smem_per_block_bytes(each_config, 128, 8)
    regs_per_thread = _estimate_regs_per_thread(each_config)
    threads_per_block = (each_config.size_FRAG_X * each_config.size_FRAG_Y)

    (cta, bottleneck, B_hw, B_threads, B_warps, B_regs, B_smem) = _estimate_cta_per_sm_active(
        threads_per_block=threads_per_block,
        regs_per_thread=regs_per_thread,
        smem_per_block=smem_per_block,
        caps=A100_DEFAULT_CAPS,
    )

    smem_total_stage1 = int(smem_per_block)
    stage_val         = int(getattr(each_config, 'stage', 1))

    # stage=1일 때 smem을 stage 수로 나누면 1 stage당 smem
    smem_per_stage = smem_total_stage1 // max(stage_val, 1)

    store_lines_total, store_lines_per_tb = compute_store_cost_warp_aware(
        each_config, tile_d, num_warp, num_TBs
    )

    cost_store = store_lines_total

    S = max(1, min(stage_val, PIPELINE_CONSTS['stage_cap']))

    # ── 1) smem 증가로 인한 occupancy 감소 ──────────────────
    smem_total = smem_per_stage * S
    if smem_total > 0:
        cta_limited_by_smem = A100_DEFAULT_CAPS['max_smem_per_sm'] // smem_total
    else:
        cta_limited_by_smem = cta

    cost_total_v2, mem_efficiency = compute_cost_with_pipeline(
        total_load_tx    = total_load_tx,
        cost_store       = cost_store,
        tx_per_loop      = tx_per_loop,
        mma_per_loop     = each_config.flops_per_loop / num_warp,
        stage            = stage_val,
        smem_per_stage_bytes = smem_per_stage,
        cta_per_sm_at_stage1 = cta,
        max_smem_per_sm  = A100_DEFAULT_CAPS['max_smem_per_sm'],
        warps_per_tb     = num_warp,
        idx = idx
    )

    # ---- 저장 ----
    each_config.cost_total_v2      = cost_total_v2
    each_config.exposed_frac_v2    = mem_efficiency

    # ---- 저장 ----
    each_config.t2_tx_per_tb        = t2_tx_per_tb
    each_config.v2_tx_per_tb        = v2_tx_per_tb
    each_config.cost_load_total_v2  = total_load_tx
    each_config.cost_store_total_v2 = cost_store
    each_config.tx_per_loop_v2      = tx_per_loop
    each_config.cta                 = min(cta, cta_limited_by_smem)
    each_config.t2_issue_per_tb     = t2_issue_per_tb
    each_config.v2_issue_per_tb     = v2_issue_per_tb
    each_config.cost_load_issue_v2  = total_load_issue
    each_config.tx_per_loop_bw_v2   = tx_per_loop_bw
    each_config.tx_per_loop_issue_v2= tx_per_loop_issue