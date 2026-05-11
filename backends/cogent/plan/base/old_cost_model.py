import re
import math
import numpy as np
import tc_helper as tc_helper
import backends.cogent.plan.base.cost_model as cm_v2

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

#
def tc_gen_cost_models_TBs(each_config, idx, opt_print) :
    #
    if opt_print == 1:
        print ("===[", idx, "]==================== [Cost Model][TBs] ==========================")
        print (f"Tile Sizes           : {each_config.list_tile_sizes}")
        print (f"Representative Sizes : {each_config.list_representative_problem_size}")
        print (f"Split Info           : {each_config.list_splits}")
        print ("============================================================================")

    #
    list_possible_comb_splits = []

    #
    for each_split in each_config.list_splits:
        #
        idx_base   = each_split[0]
        idx_first  = each_split[1]
        idx_second = each_split[2]
        
        idx_base_repre_size = tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, idx_base)
        list_possible_cases = []
        
        #
        list_possible_cases.append([[idx_base, idx_base_repre_size], [idx_first, 1], [idx_second, 1]])
        
        #
        list_possible_comb_splits.append(list_possible_cases)
    
    #
    #   External Indices: related to # of TBs, and Full-Tiles for External Indices
    #
    list_possible_representative_problem_sizes = []
    
    #
    #   [Assumption] len(list_possible_comb_splits) == 1 or 2.
    #
    if len(list_possible_comb_splits) == 1 :
        #
        for each_comb in list_possible_comb_splits[0] :
            #
            tmp_list    = []
            combined_tile_size = []
            tmp_num_TBs = 1
            
            #
            tmp_list.append(each_comb[0])

            #
            one_tile_size = tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_comb[1][0])
            second_tile_size = tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_comb[2][0])
            tmp_num_TBs *= math.ceil(each_comb[0][1] / (one_tile_size * second_tile_size))
            
            #
            combined_tile_size.append([each_comb[0][0], one_tile_size * second_tile_size])
            
            #
            for each_ext_idx in each_config.list_tensor_C :
                if tc_helper.tc_helper_find_value(each_comb, each_ext_idx) == -1 :
                    tmp_list.append([each_ext_idx, tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_ext_idx)])
                    combined_tile_size.append([each_ext_idx, tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_ext_idx)])
                    tmp_num_TBs *= math.ceil(tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_ext_idx) / tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_ext_idx))

            #
            list_possible_representative_problem_sizes.append([tmp_list, tmp_num_TBs])
        
        #
        each_config.add_split_representative_problem_size(list_possible_representative_problem_sizes[0][0])
        each_config.num_TBs = list_possible_representative_problem_sizes[0][1]

        #
        sorted_combined_tile_size = sorted(combined_tile_size, key=lambda x: x[0])
        each_config.combined_tile_size = sorted_combined_tile_size

        return list_possible_representative_problem_sizes[0]
    #
    elif len(list_possible_comb_splits) == 2:
        #
        tmp_list    = []
        combined_tile_size = []
        tmp_num_TBs = 1
        
        #
        for each_comb in list_possible_comb_splits[0]:
            #
            tmp_list.append(each_comb[0])
            
            #
            one_tile_size = tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_comb[1][0])
            second_tile_size = tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_comb[2][0])
            tmp_num_TBs *= math.ceil(each_comb[0][1] / (one_tile_size * second_tile_size))

            #
            combined_tile_size.append([each_comb[0][0], one_tile_size * second_tile_size])

        #
        for each_comb in list_possible_comb_splits[1]:
            #
            tmp_list.append(each_comb[0])

            #
            one_tile_size = tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_comb[1][0])
            second_tile_size = tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_comb[2][0])
            tmp_num_TBs *= math.ceil(each_comb[0][1] / (one_tile_size * second_tile_size))

            #
            combined_tile_size.append([each_comb[0][0], one_tile_size * second_tile_size])

        #
        tmp_split_index = []
        for each_comb in list_possible_comb_splits :
            for each_input in each_comb :
                for each_idx in each_input :
                    tmp_split_index.append(each_idx)

        #
        for each_ext_idx in each_config.list_tensor_C :
            if tc_helper.tc_helper_find_value(tmp_split_index, each_ext_idx) == -1 :
                tmp_list.append([each_ext_idx, tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_ext_idx)])
                combined_tile_size.append([each_ext_idx, tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_ext_idx)])
                tmp_num_TBs *= math.ceil(tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_ext_idx) / tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_ext_idx))

        #
        list_possible_representative_problem_sizes.append([tmp_list, tmp_num_TBs])

        #
        each_config.add_split_representative_problem_size(list_possible_representative_problem_sizes[0][0])
        each_config.num_TBs = list_possible_representative_problem_sizes[0][1]
        
        #
        sorted_combined_tile_size = sorted(combined_tile_size, key=lambda x: x[0])
        each_config.combined_tile_size = sorted_combined_tile_size

        return list_possible_representative_problem_sizes[0]
    #
    else:
        #
        combined_tile_size = []

        #
        list_possible_representative_problem_sizes = each_config.list_representative_problem_size

        #
        tmp_num_TBs = 1
        for each_ext_idx in each_config.list_tensor_C:
            tmp_num_TBs *= math.ceil(tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_ext_idx) / tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_ext_idx))
            combined_tile_size.append([each_ext_idx, tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_ext_idx)])

        #
        each_config.add_split_representative_problem_size(each_config.list_representative_problem_size)
        each_config.num_TBs = tmp_num_TBs
        
        #
        sorted_combined_tile_size = sorted(combined_tile_size, key=lambda x: x[0])
        each_config.combined_tile_size = sorted_combined_tile_size
        
        return [each_config.list_representative_problem_size, tmp_num_TBs]

#
def tc_gen_cost_models_GM(each_config, l_comb, idx, opt_print) :
    #
    if opt_print == 1:
        print(f"===[{ idx }]=============== [Cost Model][GMEM Load Inputs] =====================")
        print(f"Index Mappings : FRAG_X <- {each_config.list_FRAG_X}, FRAG_Y <- {each_config.list_FRAG_Y}")
        print(f"               : FRAG_K <- {each_config.list_FRAG_K}")
        print(f"               : REG_X  <- {each_config.list_REG_X}, REG_Y <- {each_config.list_REG_Y}")
        print(f"Tile Sizes     : {each_config.list_tile_sizes}")
        print(f"Pipeline Stage : {each_config.stage}")
        print(f"list_comb      : {l_comb}")
        print("============================================================================")
    
    #
    size_TB = each_config.size_FRAG_X * each_config.size_FRAG_Y
    
    #
    #   For Internal Indicies,
    #
    size_FRAG_K = 1
    size_N_K    = 1
    for each_int_idx in each_config.list_FRAG_K :
        size_FRAG_K *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_int_idx)
        size_N_K *= tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_int_idx)

    #
    #   # of "main" loop (calculated by N_K / T_K)
    #
    steps_main_loops = math.ceil(size_N_K / size_FRAG_K)

    #
    #   Check Types of Input such as [E_K, ...] or [E_A, ...]
    #
    opt_load_A_ext = -1     # -1: FVI = internal
    opt_load_B_ext = -1     #  1: FVI = external
    if tc_helper.tc_helper_find_index(each_config.list_tensor_B, each_config.list_tensor_A[0]) == -1 :
        opt_load_A_ext = 1
    
    if tc_helper.tc_helper_find_index(each_config.list_tensor_A, each_config.list_tensor_B[0]) == -1 :
        opt_load_B_ext = 1

    #
    #   Initial Values
    #
    size_continuous_elements_A = 1
    size_continuous_elements_B = 1
    size_continuous_elements_C = 1

    #
    if opt_print == 1 :
        print("-1 : FVI = internal, 1 : FVI = external")
        print(f"opt_load_A_ext : {opt_load_A_ext}, opt_load_B_ext : {opt_load_B_ext}")

    #
    #   Input: A (Continuous)
    #
    is_continuous = 1
    for each_idx in each_config.list_tensor_A :
        #
        if opt_load_A_ext == 1 : # FVI is external index
            # Internal
            if tc_helper.tc_helper_find_index(each_config.list_tensor_B, each_idx) != -1 :
                break

            # External
            else :
                # Need to Check if This Index is Continuous Or NOT.
                if is_continuous == 1 :
                    #
                    size_continuous_elements_A *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
                    
                    #
                    if tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx) != tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_idx) :
                        is_continuous = -1
                else :
                    break
        else : # FVI is internal index
            # External
            if tc_helper.tc_helper_find_index(each_config.list_tensor_B, each_idx) == -1 :
                break

            # Internal
            else :
                #
                if is_continuous == 1 :
                    #
                    size_continuous_elements_A *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
                    
                    #
                    if tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx) != tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_idx) :
                        is_continuous = -1
                else:
                    break
    
    #
    if opt_print == 1 :
        print (f"[A] is_continuous : {is_continuous}, size_continuous_elements_A : {size_continuous_elements_A}")

    #
    #   Input: A (FRAG and REG)
    #
    size_A_E_FRAG = 1
    size_A_K_FRAG = 1
    size_A_E_REG  = 1
    size_A_FVI    = 1

    #
    for cnt, each_idx in enumerate(each_config.list_tensor_A) :
        #
        if cnt == 0 :
            size_A_FVI = tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)

        #
        if tc_helper.tc_helper_find_index(each_config.list_tensor_B, each_idx) == -1 : # External Index
            # FRAG
            if tc_helper.tc_helper_find_index(each_config.list_REG_X, each_idx) == -1 and tc_helper.tc_helper_find_index(each_config.list_REG_Y, each_idx) == -1 :
                size_A_E_FRAG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
            # REG
            else :
                size_A_E_REG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
        else : # Internal Index
            size_A_K_FRAG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
    
    #
    if opt_print == 1 :
        print(f"|FVI_A|  = {size_A_FVI}")
        print(f"|SMEM_A| = {size_A_E_REG * size_A_E_FRAG * size_A_K_FRAG}, ({size_A_E_REG} * {size_A_E_FRAG} * {size_A_K_FRAG})")
        print(f"|FRAG_X| = {each_config.size_FRAG_X}, |FRAG_Y| = {each_config.size_FRAG_Y}")
        print(f"|TB|     = {size_TB}")

    #
    vol_A_per_TB = min(size_TB, size_A_E_REG * size_A_E_FRAG * size_A_K_FRAG)

    #
    times_inner_A_FVI = 1
    times_inner_A_TB = math.ceil(vol_A_per_TB / size_A_FVI)
    steps_inner_A_loops = math.ceil(size_A_E_REG * size_A_E_FRAG * size_A_K_FRAG / size_TB)
    
    # times_inner_A_FVI = 1

    # #
    # if each_config.double2_flag[0] == 1:
    #     vol_A_per_TB = min((2 * size_TB), size_A_E_REG * size_A_E_FRAG * size_A_K_FRAG)

    #     #
    #     times_inner_A_TB = math.ceil(vol_A_per_TB / (2 * size_A_FVI))
    #     steps_inner_A_loops = math.ceil(size_A_E_REG * size_A_E_FRAG * size_A_K_FRAG / (2 * size_TB))
    # else :
    #     vol_A_per_TB = min(size_TB, size_A_E_REG * size_A_E_FRAG * size_A_K_FRAG)

    #     #
    #     times_inner_A_TB = math.ceil(vol_A_per_TB / size_A_FVI)
    #     steps_inner_A_loops = math.ceil(size_A_E_REG * size_A_E_FRAG * size_A_K_FRAG / size_TB)

    #
    estimated_DRAM_transaction_A_per_FVI         = times_inner_A_FVI
    estimated_DRAM_transaction_A_per_TB          = estimated_DRAM_transaction_A_per_FVI * times_inner_A_TB
    estimated_DRAM_transaction_A_per_inner_loops = estimated_DRAM_transaction_A_per_TB * steps_inner_A_loops
    estimated_DRAM_transaction_A_per_main_loops  = estimated_DRAM_transaction_A_per_inner_loops * steps_main_loops

    #
    if opt_print == 1 :
        print(f"[A] estimated_DRAM_transaction_per_FVI         : {estimated_DRAM_transaction_A_per_FVI}")
        print(f"[A] estimated_DRAM_transaction_per_TB          : {estimated_DRAM_transaction_A_per_TB}")
        print(f"[A] estimated_DRAM_transaction_per_inner_loops : {estimated_DRAM_transaction_A_per_inner_loops}")
        print(f"[A] estimated_DRAM_transaction_per_main_loops  : {estimated_DRAM_transaction_A_per_main_loops}")

    #
    #   To Calculate The Cost of Loading Input Tensor per a Thread Block
    #
    cost_TB_load_A = estimated_DRAM_transaction_A_per_main_loops

    #
    #   Input: B
    #
    is_continuous = 1
    for each_idx in each_config.list_tensor_B :
        #
        if opt_load_B_ext == 1 :
            # Internal
            if tc_helper.tc_helper_find_index(each_config.list_tensor_A, each_idx) != -1 :
                break

            # External
            else :
                #
                if is_continuous == 1 :
                    #
                    size_continuous_elements_B *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
                    
                    #
                    if tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx) != tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_idx) :
                        is_continuous = -1
                else :
                    break
        else :
            # External
            if tc_helper.tc_helper_find_index(each_config.list_tensor_A, each_idx) == -1 :
                break

            # Internal
            else :
                #
                if is_continuous == 1 :
                    #
                    size_continuous_elements_B *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
                    
                    #
                    if tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx) != tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_idx) :
                        is_continuous = -1
                else :
                    break
    
    #
    if opt_print == 1 :
        print (f"[B] is_continuous : {is_continuous}, size_continuous_elements_B : {size_continuous_elements_B}")

    #
    #   Input: B (FRAG and REG)
    #
    size_B_E_FRAG = 1
    size_B_K_FRAG = 1
    size_B_E_REG  = 1
    size_B_FVI    = 1

    #
    for cnt, each_idx in enumerate(each_config.list_tensor_B) :
        #
        if cnt == 0 :
            size_B_FVI = tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
        
        #
        if tc_helper.tc_helper_find_index(each_config.list_tensor_A, each_idx) == -1 :
            # FRAG
            if tc_helper.tc_helper_find_index(each_config.list_REG_X, each_idx) == -1 and tc_helper.tc_helper_find_index(each_config.list_REG_Y, each_idx) == -1 :
                size_B_E_FRAG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
            # REG
            else :
                size_B_E_REG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
        else :
            size_B_K_FRAG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
    
    #
    if opt_print == 1 :
        print(f"|FVI_B|  = {size_B_FVI}")
        print(f"|SMEM_B| = {size_B_E_REG * size_B_E_FRAG * size_B_K_FRAG}, ({size_B_E_REG} * {size_B_E_FRAG} * {size_B_K_FRAG})")
        print(f"|FRAG_X| = {each_config.size_FRAG_X}, |FRAG_Y| = {each_config.size_FRAG_Y}")
        print(f"|TB|     = {size_TB}")

    #
    vol_B_per_TB = min(size_TB, size_B_E_REG * size_B_E_FRAG * size_B_K_FRAG)

    #
    times_inner_B_FVI = 1
    times_inner_B_TB = math.ceil(vol_B_per_TB / size_B_FVI)
    steps_inner_B_loops = math.ceil(size_B_E_REG * size_B_E_FRAG * size_B_K_FRAG / size_TB)

    # times_inner_B_FVI = 1

    # if each_config.double2_flag[1] == 1 :
    #     vol_B_per_TB = min((2 * size_TB), size_B_E_REG * size_B_E_FRAG * size_B_K_FRAG)

    #     times_inner_B_TB = math.ceil(vol_B_per_TB / (2 * size_B_FVI))
    #     steps_inner_B_loops = math.ceil(size_B_E_REG * size_B_E_FRAG * size_B_K_FRAG / (2 * size_TB))
    # else :
    #     vol_B_per_TB = min(size_TB, size_B_E_REG * size_B_E_FRAG * size_B_K_FRAG)

    #     times_inner_B_TB = math.ceil(vol_B_per_TB / size_B_FVI)
    #     steps_inner_B_loops = math.ceil(size_B_E_REG * size_B_E_FRAG * size_B_K_FRAG / size_TB)

    #
    estimated_DRAM_transaction_B_per_FVI         = times_inner_B_FVI
    estimated_DRAM_transaction_B_per_TB          = estimated_DRAM_transaction_B_per_FVI * times_inner_B_TB
    estimated_DRAM_transaction_B_per_inner_loops = estimated_DRAM_transaction_B_per_TB * steps_inner_B_loops
    estimated_DRAM_transaction_B_per_main_loops  = estimated_DRAM_transaction_B_per_inner_loops * steps_main_loops

    #
    if opt_print == 1 :
        print(f"[B] estimated_DRAM_transaction_per_FVI         : {estimated_DRAM_transaction_B_per_FVI}")
        print(f"[B] estimated_DRAM_transaction_per_TB          : {estimated_DRAM_transaction_B_per_TB}")
        print(f"[B] estimated_DRAM_transaction_per_inner_loops : {estimated_DRAM_transaction_B_per_inner_loops}")
        print(f"[B] estimated_DRAM_transaction_per_main_loops  : {estimated_DRAM_transaction_B_per_main_loops}")

    #
    #   To Calculate The Cost of Loading Input Tensor per a Thread Block
    #
    cost_TB_load_B = estimated_DRAM_transaction_B_per_main_loops

    #
    size_output_TB = 1
    for each_idx in each_config.list_tensor_C :
        size_output_TB *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)

    #
    size_output_fragment = 8 * 8
    cnt_output_Fragment = math.ceil(size_output_TB / size_output_fragment)

    #
    per_row_transaction_inner_output_fragment = 8
    cost_TB_store_C = cnt_output_Fragment * per_row_transaction_inner_output_fragment
    
    #
    #   The # of Thread Blocks
    #
    num_TBs = l_comb[1]

    if opt_print == 1 :
        print (">>> # of TBs: ", num_TBs)

    #
    each_config.cost_load_TB       = (cost_TB_load_A + cost_TB_load_B) 
    each_config.cost_load_input   = (cost_TB_load_A + cost_TB_load_B) * num_TBs
    each_config.cost_store_output = cost_TB_store_C * num_TBs
    each_config.cost_total        = each_config.cost_load_input + each_config.cost_store_output

    #
    each_config.steps_main_loops  = steps_main_loops

    #
    if opt_print == 1 :
        print(f"Cost Input (Load)        : {each_config.cost_load_input}")
        print(f"Cost Output (Store)      : {each_config.cost_store_output}")
        print(f"Total Cost               : {each_config.cost_total}") 
        print(f"# of steps for main-loop : {each_config.steps_main_loops}")
        print ("============================================================================")

#
def estimate_lines128_strided(vol_elems, contig_elems, elem_bytes=8, align_factor=1.0):
    """
    vol_elems:    이번 inner-loop에서 TB가 로드해야 하는 element 수 (double 기준)
    contig_elems: "연속으로 붙어있는" element 수 (stride가 끊기기 전까지)
    elem_bytes:   FP64=8
    align_factor: 128B boundary crossing/불완전 정렬에 대한 보정 (기본 1.0)
    """
    if vol_elems <= 0:
        return 0

    contig_elems = max(1, int(contig_elems))
    # stride 때문에 contig block 단위로 쪼개진다고 가정
    num_blocks = int(math.ceil(vol_elems / contig_elems))

    # 한 block이 차지하는 바이트
    block_bytes = contig_elems * elem_bytes

    # 128B line을 몇 개 터치하는지 (연속 16 doubles면 128B => 1 line)
    lines_per_block = int(math.ceil(block_bytes / 128.0))
    lines = num_blocks * lines_per_block

    # 정렬/경계 crossing 보정
    lines = int(math.ceil(lines * float(align_factor)))

    return lines

#
def tc_gen_cost_models_GM2(each_config, l_comb, idx, opt_print) :
    #
    if opt_print == 1:
        print(f"===[{ idx }]=============== [Cost Model][GMEM Load Inputs] =====================")
        print(f"Index Mappings : FRAG_X <- {each_config.list_FRAG_X}, FRAG_Y <- {each_config.list_FRAG_Y}")
        print(f"               : FRAG_K <- {each_config.list_FRAG_K}")
        print(f"               : REG_X  <- {each_config.list_REG_X}, REG_Y <- {each_config.list_REG_Y}")
        print(f"Tile Sizes     : {each_config.list_tile_sizes}")
        print(f"Pipeline stage : {each_config.stage}")
        print(f"list_comb      : {l_comb}")
        # [UPDATED] double2 flag 출력 추가
        if hasattr(each_config, 'double2_flag'):
             print(f"Double2 Flags  : A={each_config.double2_flag[0]}, B={each_config.double2_flag[1]}")
        print("============================================================================")
    
    #
    size_TB = each_config.size_FRAG_X * each_config.size_FRAG_Y
    
    #
    #   For Internal Indicies,
    #
    size_FRAG_K = 1
    size_N_K    = 1
    for each_int_idx in each_config.list_FRAG_K :
        size_FRAG_K *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_int_idx)
        size_N_K *= tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_int_idx)

    #
    #   # of "main" loop (calculated by N_K / T_K)
    #
    steps_main_loops = math.ceil(size_N_K / size_FRAG_K)
    #stage_factor = max(1, steps_main_loops - (each_config.stage - 1)) / steps_main_loops

    #
    #   Check Types of Input such as [E_K, ...] or [E_A, ...]
    #
    opt_load_A_ext = -1     # -1: FVI = internal
    opt_load_B_ext = -1     #  1: FVI = external
    if tc_helper.tc_helper_find_index(each_config.list_tensor_B, each_config.list_tensor_A[0]) == -1 :
        opt_load_A_ext = 1
    
    if tc_helper.tc_helper_find_index(each_config.list_tensor_A, each_config.list_tensor_B[0]) == -1 :
        opt_load_B_ext = 1

    #
    #   Initial Values
    #
    size_continuous_elements_A = 1
    size_continuous_elements_B = 1
    size_continuous_elements_C = 1

    #
    if opt_print == 1 :
        print("-1 : FVI = internal, 1 : FVI = external")
        print(f"opt_load_A_ext : {opt_load_A_ext}, opt_load_B_ext : {opt_load_B_ext}")

    # ... (is_continuous calculation for A - unchanged) ...
    #   Input: A (Continuous)
    #
    is_continuous = 1
    for each_idx in each_config.list_tensor_A :
        if opt_load_A_ext == 1 : 
            if tc_helper.tc_helper_find_index(each_config.list_tensor_B, each_idx) != -1 : break
            else :
                if is_continuous == 1 :
                    size_continuous_elements_A *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
                    if tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx) != tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_idx) :
                        is_continuous = -1
                else : break
        else : 
            if tc_helper.tc_helper_find_index(each_config.list_tensor_B, each_idx) == -1 : break
            else :
                if is_continuous == 1 :
                    size_continuous_elements_A *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
                    if tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx) != tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_idx) :
                        is_continuous = -1
                else: break
    
    if opt_print == 1 :
        print (f"[A] is_continuous : {is_continuous}, size_continuous_elements_A : {size_continuous_elements_A}")

    #
    #   Input: A (FRAG and REG)
    #
    size_A_E_FRAG = 1
    size_A_K_FRAG = 1
    size_A_E_REG  = 1
    size_A_FVI    = 1

    for cnt, each_idx in enumerate(each_config.list_tensor_A) :
        if cnt == 0 :
            size_A_FVI = tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
        if tc_helper.tc_helper_find_index(each_config.list_tensor_B, each_idx) == -1 : # External Index
            if tc_helper.tc_helper_find_index(each_config.list_REG_X, each_idx) == -1 and tc_helper.tc_helper_find_index(each_config.list_REG_Y, each_idx) == -1 :
                size_A_E_FRAG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
            else :
                size_A_E_REG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
        else : # Internal Index
            size_A_K_FRAG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
    
    if opt_print == 1 :
        print(f"|FVI_A|  = {size_A_FVI}")
        print(f"|SMEM_A| = {size_A_E_REG * size_A_E_FRAG * size_A_K_FRAG}")

    #
    #   ### [UPDATED] Cost Calculation Logic for A with double2
    #
    vol_A_per_TB = min(size_TB, size_A_E_REG * size_A_E_FRAG * size_A_K_FRAG)

    # 1. double2 flag 확인 및 vector width 설정
    vec_width_A = 1
    if hasattr(each_config, 'double2_flag') and each_config.double2_flag[0] == 1:
        vec_width_A = 2

    times_inner_A_FVI = 1
    times_inner_A_TB = math.ceil(vol_A_per_TB / size_A_FVI)
    steps_inner_A_loops = math.ceil(size_A_E_REG * size_A_E_FRAG * size_A_K_FRAG / size_TB)

    # 2. Transaction 횟수를 vector width로 나눔 (1회 로드 당 2배의 데이터)
    estimated_DRAM_transaction_A_per_FVI         = times_inner_A_FVI / vec_width_A
    #estimated_DRAM_transaction_A_per_TB          = estimated_DRAM_transaction_A_per_FVI * times_inner_A_TB
    estimated_DRAM_transaction_A_per_inner_loops = estimated_DRAM_transaction_A_per_FVI * steps_inner_A_loops
    estimated_DRAM_transaction_A_per_main_loops  = estimated_DRAM_transaction_A_per_inner_loops * steps_main_loops

    if opt_print == 1 :
        print(f"[A] Vector Width (double2)                     : {vec_width_A}")
        print(f"[A] estimated_DRAM_transaction_per_FVI         : {estimated_DRAM_transaction_A_per_FVI}")
        #print(f"[A] estimated_DRAM_transaction_per_TB          : {estimated_DRAM_transaction_A_per_TB}")
        print(f"[A] estimated_DRAM_transaction_per_main_loops  : {estimated_DRAM_transaction_A_per_main_loops}")

    # 3. 최종 Cost는 정수로 반올림 (Transaction은 정수 단위)
    cost_TB_load_A = math.ceil(estimated_DRAM_transaction_A_per_main_loops)
    
    # ... (is_continuous calculation for B - unchanged) ...
    #   Input: B
    #
    is_continuous = 1
    for each_idx in each_config.list_tensor_B :
        if opt_load_B_ext == 1 :
            if tc_helper.tc_helper_find_index(each_config.list_tensor_A, each_idx) != -1 : break
            else :
                if is_continuous == 1 :
                    size_continuous_elements_B *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
                    if tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx) != tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_idx) :
                        is_continuous = -1
                else : break
        else :
            if tc_helper.tc_helper_find_index(each_config.list_tensor_A, each_idx) == -1 : break
            else :
                if is_continuous == 1 :
                    size_continuous_elements_B *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
                    if tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx) != tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_idx) :
                        is_continuous = -1
                else : break
    
    if opt_print == 1 :
        print (f"[B] is_continuous : {is_continuous}, size_continuous_elements_B : {size_continuous_elements_B}")

    #
    #   Input: B (FRAG and REG)
    #
    size_B_E_FRAG = 1
    size_B_K_FRAG = 1
    size_B_E_REG  = 1
    size_B_FVI    = 1

    for cnt, each_idx in enumerate(each_config.list_tensor_B) :
        if cnt == 0 :
            size_B_FVI = tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
        if tc_helper.tc_helper_find_index(each_config.list_tensor_A, each_idx) == -1 :
            if tc_helper.tc_helper_find_index(each_config.list_REG_X, each_idx) == -1 and tc_helper.tc_helper_find_index(each_config.list_REG_Y, each_idx) == -1 :
                size_B_E_FRAG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
            else :
                size_B_E_REG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
        else :
            size_B_K_FRAG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
    
    if opt_print == 1 :
        print(f"|FVI_B|  = {size_B_FVI}")
        print(f"|SMEM_B| = {size_B_E_REG * size_B_E_FRAG * size_B_K_FRAG}")

    #
    #   ### [UPDATED] Cost Calculation Logic for B with double2
    #
    vol_B_per_TB = min(size_TB, size_B_E_REG * size_B_E_FRAG * size_B_K_FRAG)

    # 1. double2 flag 확인 및 vector width 설정
    vec_width_B = 1
    if hasattr(each_config, 'double2_flag') and each_config.double2_flag[1] == 1:
        vec_width_B = 2

    times_inner_B_FVI = 1
    times_inner_B_TB = math.ceil(vol_B_per_TB / size_B_FVI)
    steps_inner_B_loops = math.ceil(size_B_E_REG * size_B_E_FRAG * size_B_K_FRAG / size_TB)

    # 2. Transaction 횟수를 vector width로 나눔
    estimated_DRAM_transaction_B_per_FVI         = times_inner_B_FVI / vec_width_B
    #estimated_DRAM_transaction_B_per_TB          = estimated_DRAM_transaction_B_per_FVI * times_inner_B_TB
    estimated_DRAM_transaction_B_per_inner_loops = estimated_DRAM_transaction_B_per_FVI * steps_inner_B_loops
    estimated_DRAM_transaction_B_per_main_loops  = estimated_DRAM_transaction_B_per_inner_loops * steps_main_loops

    if opt_print == 1 :
        print(f"[B] Vector Width (double2)                     : {vec_width_B}")
        print(f"[B] estimated_DRAM_transaction_per_FVI         : {estimated_DRAM_transaction_B_per_FVI}")
        #print(f"[B] estimated_DRAM_transaction_per_TB          : {estimated_DRAM_transaction_B_per_TB}")
        print(f"[B] estimated_DRAM_transaction_per_main_loops  : {estimated_DRAM_transaction_B_per_main_loops}")

    # 3. 최종 Cost는 정수로 반올림
    cost_TB_load_B = math.ceil(estimated_DRAM_transaction_B_per_main_loops)

    #
    #   Output C Calculation (unchanged)
    #
    size_output_TB = 1
    for each_idx in each_config.list_tensor_C :
        size_output_TB *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)

    size_output_fragment = 8 * 8
    cnt_output_Fragment = math.ceil(size_output_TB / size_output_fragment)

    per_row_transaction_inner_output_fragment = 8
    cost_TB_store_C = cnt_output_Fragment * per_row_transaction_inner_output_fragment
    
    #
    #   The # of Thread Blocks
    #
    num_TBs = l_comb[1]

    if opt_print == 1 :
        print (">>> # of TBs: ", num_TBs)

    #
    each_config.cost_load_TB_d2       = (cost_TB_load_A + cost_TB_load_B) 
    each_config.cost_load_input_d2    = (cost_TB_load_A + cost_TB_load_B) * num_TBs
    each_config.cost_store_output_d2  = cost_TB_store_C * num_TBs
    each_config.cost_total_d2         = each_config.cost_load_input_d2 + each_config.cost_store_output_d2

    #
    each_config.steps_main_loops_d2  = steps_main_loops

    #
    each_config.transaction_per_loop = estimated_DRAM_transaction_A_per_inner_loops + estimated_DRAM_transaction_B_per_inner_loops
    
    #
    if opt_print == 1 :
        print(f"Cost Input (Load)        : {each_config.cost_load_input_d2}")
        print(f"Cost Output (Store)      : {each_config.cost_store_output_d2}")
        print(f"Total Cost               : {each_config.cost_total_d2}")
        print(f"Transaction per loop     : {each_config.transaction_per_loop}")
        print(f"# of steps for main-loop : {each_config.steps_main_loops_d2}")
        print ("============================================================================")

#
def tc_gen_cost_models_GM3(each_config, l_comb, idx, opt_print) :
    #
    if opt_print == 1:
        print(f"===[{ idx }]=============== [Cost Model][GMEM Load Inputs] =====================")
        print(f"Index Mappings : FRAG_X <- {each_config.list_FRAG_X}, FRAG_Y <- {each_config.list_FRAG_Y}")
        print(f"               : FRAG_K <- {each_config.list_FRAG_K}")
        print(f"               : REG_X  <- {each_config.list_REG_X}, REG_Y <- {each_config.list_REG_Y}")
        print(f"Tile Sizes     : {each_config.list_tile_sizes}")
        print(f"Pipeline stage : {each_config.stage}")
        print(f"list_comb      : {l_comb}")
        if hasattr(each_config, 'double2_flag'):
             print(f"Double2 Flags  : A={each_config.double2_flag[0]}, B={each_config.double2_flag[1]}")
        print("============================================================================")
    
    #
    size_TB = each_config.size_FRAG_X * each_config.size_FRAG_Y
    
    #
    #   For Internal Indicies,
    #
    size_FRAG_K = 1
    size_N_K    = 1
    for each_int_idx in each_config.list_FRAG_K :
        size_FRAG_K *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_int_idx)
        size_N_K *= tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_int_idx)

    #
    #   # of "main" loop (calculated by N_K / T_K)
    #
    steps_main_loops = math.ceil(size_N_K / size_FRAG_K)
    #stage_factor = max(1, steps_main_loops - (each_config.stage - 1)) / steps_main_loops

    #
    #   Check Types of Input such as [E_K, ...] or [E_A, ...]
    #
    opt_load_A_ext = -1     # -1: FVI = internal
    opt_load_B_ext = -1     #  1: FVI = external
    if tc_helper.tc_helper_find_index(each_config.list_tensor_B, each_config.list_tensor_A[0]) == -1 :
        opt_load_A_ext = 1
    
    if tc_helper.tc_helper_find_index(each_config.list_tensor_A, each_config.list_tensor_B[0]) == -1 :
        opt_load_B_ext = 1

    #
    #   Initial Values
    #
    size_continuous_elements_A = 1
    size_continuous_elements_B = 1
    size_continuous_elements_C = 1

    #
    if opt_print == 1 :
        print("-1 : FVI = internal, 1 : FVI = external")
        print(f"opt_load_A_ext : {opt_load_A_ext}, opt_load_B_ext : {opt_load_B_ext}")

    # ... (is_continuous calculation for A - unchanged) ...
    #   Input: A (Continuous)
    #
    is_continuous = 1
    for each_idx in each_config.list_tensor_A :
        if opt_load_A_ext == 1 : 
            if tc_helper.tc_helper_find_index(each_config.list_tensor_B, each_idx) != -1 : break
            else :
                if is_continuous == 1 :
                    size_continuous_elements_A *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
                    if tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx) != tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_idx) :
                        is_continuous = -1
                else : break
        else : 
            if tc_helper.tc_helper_find_index(each_config.list_tensor_B, each_idx) == -1 : break
            else :
                if is_continuous == 1 :
                    size_continuous_elements_A *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
                    if tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx) != tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_idx) :
                        is_continuous = -1
                else: break
    
    if opt_print == 1 :
        print (f"[A] is_continuous : {is_continuous}, size_continuous_elements_A : {size_continuous_elements_A}")

    #
    #   Input: A (FRAG and REG)
    #
    size_A_E_FRAG = 1
    size_A_K_FRAG = 1
    size_A_E_REG  = 1
    size_A_FVI    = 1

    for cnt, each_idx in enumerate(each_config.list_tensor_A) :
        if cnt == 0 :
            size_A_FVI = tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
        if tc_helper.tc_helper_find_index(each_config.list_tensor_B, each_idx) == -1 : # External Index
            if tc_helper.tc_helper_find_index(each_config.list_REG_X, each_idx) == -1 and tc_helper.tc_helper_find_index(each_config.list_REG_Y, each_idx) == -1 :
                size_A_E_FRAG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
            else :
                size_A_E_REG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
        else : # Internal Index
            size_A_K_FRAG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
    
    if opt_print == 1 :
        print(f"|FVI_A|  = {size_A_FVI}")
        print(f"|SMEM_A| = {size_A_E_REG * size_A_E_FRAG * size_A_K_FRAG}")

    #
    #   ### [UPDATED] Cost Calculation Logic for A with double2
    #
    # vol_A_per_TB = min(size_TB, size_A_E_REG * size_A_E_FRAG * size_A_K_FRAG)

    # # 1. double2 flag 확인 및 vector width 설정
    # vec_width_A = 1
    # if hasattr(each_config, 'double2_flag') and each_config.double2_flag[0] == 1:
    #     vec_width_A = 2

    # times_inner_A_FVI = 1
    # times_inner_A_TB = math.ceil(vol_A_per_TB / size_A_FVI)
    # steps_inner_A_loops = math.ceil(size_A_E_REG * size_A_E_FRAG * size_A_K_FRAG / size_TB)

    # # 2. Transaction 횟수를 vector width로 나눔 (1회 로드 당 2배의 데이터)
    # estimated_DRAM_transaction_A_per_FVI         = times_inner_A_FVI / vec_width_A
    # #estimated_DRAM_transaction_A_per_TB          = estimated_DRAM_transaction_A_per_FVI * times_inner_A_TB
    # estimated_DRAM_transaction_A_per_inner_loops = estimated_DRAM_transaction_A_per_FVI * steps_inner_A_loops
    # estimated_DRAM_transaction_A_per_main_loops  = estimated_DRAM_transaction_A_per_inner_loops * steps_main_loops

    # if opt_print == 1 :
    #     print(f"[A] Vector Width (double2)                     : {vec_width_A}")
    #     print(f"[A] estimated_DRAM_transaction_per_FVI         : {estimated_DRAM_transaction_A_per_FVI}")
    #     #print(f"[A] estimated_DRAM_transaction_per_TB          : {estimated_DRAM_transaction_A_per_TB}")
    #     print(f"[A] estimated_DRAM_transaction_per_main_loops  : {estimated_DRAM_transaction_A_per_main_loops}")

    # # 3. 최종 Cost는 정수로 반올림 (Transaction은 정수 단위)
    # cost_TB_load_A = math.ceil(estimated_DRAM_transaction_A_per_main_loops)

    total_A_tile_elems = size_A_E_REG * size_A_E_FRAG * size_A_K_FRAG

    # inner loop 횟수 (네 코드 그대로 유지)
    steps_inner_A_loops = math.ceil(total_A_tile_elems / size_TB)

    # inner-loop에서 로드하는 element 수(대부분 size_TB, 마지막은 remainder)
    vol_A_per_inner = min(size_TB, total_A_tile_elems)

    # double2 flag
    vec_width_A = 1
    if hasattr(each_config, 'double2_flag') and each_config.double2_flag[0] == 1:
        vec_width_A = 2

    # ---- 핵심: transaction(=stride로 나뉜 contiguous block 수/128B line 수)은 double2로 나누지 않음
    # 네 의도대로라면 contig_elems가 16이면 "16 doubles = 128B" 단위로 끊김
    # size_continuous_elements_A가 16보다 클 수도 있으니 그대로 사용(연속이 더 길면 block_bytes가 커져 lines_per_block이 늘어남)
    align_factor_A = 1.0  # 필요하면 1.25 같은 보정 사용 가능
    lines128_A_per_inner = estimate_lines128_strided(
        vol_elems=vol_A_per_inner,
        contig_elems=size_continuous_elements_A,
        elem_bytes=8,
        align_factor=align_factor_A
    )

    # inst(issue) 쪽만 double2 반영: (대략) element 수 / vec_width
    # 실제 thread mapping까지 반영하면 더 좋아지지만, 최소 수정으로는 이 정도가 안정적
    mem_issue_A_per_inner = int(math.ceil(vol_A_per_inner / float(vec_width_A)))

    # main loop 반영
    lines128_A_per_main = lines128_A_per_inner * steps_inner_A_loops * steps_main_loops
    mem_issue_A_per_main = mem_issue_A_per_inner * steps_inner_A_loops * steps_main_loops

    # 기존 cost 변수에선 transaction(비용)을 line 기반으로 넣는 게 더 일관됨
    cost_TB_load_A = int(math.ceil(lines128_A_per_main))
    
    # ... (is_continuous calculation for B - unchanged) ...
    #   Input: B
    #
    is_continuous = 1
    for each_idx in each_config.list_tensor_B :
        if opt_load_B_ext == 1 :
            if tc_helper.tc_helper_find_index(each_config.list_tensor_A, each_idx) != -1 : break
            else :
                if is_continuous == 1 :
                    size_continuous_elements_B *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
                    if tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx) != tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_idx) :
                        is_continuous = -1
                else : break
        else :
            if tc_helper.tc_helper_find_index(each_config.list_tensor_A, each_idx) == -1 : break
            else :
                if is_continuous == 1 :
                    size_continuous_elements_B *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
                    if tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx) != tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_idx) :
                        is_continuous = -1
                else : break
    
    if opt_print == 1 :
        print (f"[B] is_continuous : {is_continuous}, size_continuous_elements_B : {size_continuous_elements_B}")

    #
    #   Input: B (FRAG and REG)
    #
    size_B_E_FRAG = 1
    size_B_K_FRAG = 1
    size_B_E_REG  = 1
    size_B_FVI    = 1

    for cnt, each_idx in enumerate(each_config.list_tensor_B) :
        if cnt == 0 :
            size_B_FVI = tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
        if tc_helper.tc_helper_find_index(each_config.list_tensor_A, each_idx) == -1 :
            if tc_helper.tc_helper_find_index(each_config.list_REG_X, each_idx) == -1 and tc_helper.tc_helper_find_index(each_config.list_REG_Y, each_idx) == -1 :
                size_B_E_FRAG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
            else :
                size_B_E_REG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
        else :
            size_B_K_FRAG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
    
    if opt_print == 1 :
        print(f"|FVI_B|  = {size_B_FVI}")
        print(f"|SMEM_B| = {size_B_E_REG * size_B_E_FRAG * size_B_K_FRAG}")

    #
    #   ### [UPDATED] Cost Calculation Logic for B with double2
    #
    # vol_B_per_TB = min(size_TB, size_B_E_REG * size_B_E_FRAG * size_B_K_FRAG)

    # # 1. double2 flag 확인 및 vector width 설정
    # vec_width_B = 1
    # if hasattr(each_config, 'double2_flag') and each_config.double2_flag[1] == 1:
    #     vec_width_B = 2

    # times_inner_B_FVI = 1
    # times_inner_B_TB = math.ceil(vol_B_per_TB / size_B_FVI)
    # steps_inner_B_loops = math.ceil(size_B_E_REG * size_B_E_FRAG * size_B_K_FRAG / size_TB)

    # # 2. Transaction 횟수를 vector width로 나눔
    # estimated_DRAM_transaction_B_per_FVI         = times_inner_B_FVI / vec_width_B
    # #estimated_DRAM_transaction_B_per_TB          = estimated_DRAM_transaction_B_per_FVI * times_inner_B_TB
    # estimated_DRAM_transaction_B_per_inner_loops = estimated_DRAM_transaction_B_per_FVI * steps_inner_B_loops
    # estimated_DRAM_transaction_B_per_main_loops  = estimated_DRAM_transaction_B_per_inner_loops * steps_main_loops

    # if opt_print == 1 :
    #     print(f"[B] Vector Width (double2)                     : {vec_width_B}")
    #     print(f"[B] estimated_DRAM_transaction_per_FVI         : {estimated_DRAM_transaction_B_per_FVI}")
    #     #print(f"[B] estimated_DRAM_transaction_per_TB          : {estimated_DRAM_transaction_B_per_TB}")
    #     print(f"[B] estimated_DRAM_transaction_per_main_loops  : {estimated_DRAM_transaction_B_per_main_loops}")

    # # 3. 최종 Cost는 정수로 반올림
    # cost_TB_load_B = math.ceil(estimated_DRAM_transaction_B_per_main_loops)

    total_B_tile_elems = size_B_E_REG * size_B_E_FRAG * size_B_K_FRAG
    steps_inner_B_loops = math.ceil(total_B_tile_elems / size_TB)
    vol_B_per_inner = min(size_TB, total_B_tile_elems)

    vec_width_B = 1
    if hasattr(each_config, 'double2_flag') and each_config.double2_flag[1] == 1:
        vec_width_B = 2

    align_factor_B = 1.0
    lines128_B_per_inner = estimate_lines128_strided(
        vol_elems=vol_B_per_inner,
        contig_elems=size_continuous_elements_B,
        elem_bytes=8,
        align_factor=align_factor_B
    )
    mem_issue_B_per_inner = int(math.ceil(vol_B_per_inner / float(vec_width_B)))

    lines128_B_per_main = lines128_B_per_inner * steps_inner_B_loops * steps_main_loops
    mem_issue_B_per_main = mem_issue_B_per_inner * steps_inner_B_loops * steps_main_loops

    cost_TB_load_B = int(math.ceil(lines128_B_per_main))

    if opt_print == 1:
        print(f"[B] Vector Width (double2)                 : {vec_width_B}")
        print(f"[B] contig_elems (stride block)           : {size_continuous_elements_B}")
        print(f"[B] vol_B_per_inner (elems)               : {vol_B_per_inner}")
        print(f"[B] lines128_B_per_inner                  : {lines128_B_per_inner}")
        print(f"[B] mem_issue_B_per_inner (approx)        : {mem_issue_B_per_inner}")
        print(f"[B] cost_TB_load_B (lines128 over main)   : {cost_TB_load_B}")

    #
    #   Output C Calculation (unchanged)
    #
    size_output_TB = 1
    for each_idx in each_config.list_tensor_C :
        size_output_TB *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)

    size_output_fragment = 8 * 8
    cnt_output_Fragment = math.ceil(size_output_TB / size_output_fragment)

    per_row_transaction_inner_output_fragment = 8
    cost_TB_store_C = cnt_output_Fragment * per_row_transaction_inner_output_fragment
    
    #
    #   The # of Thread Blocks
    #
    num_TBs = l_comb[1]

    if opt_print == 1 :
        print (">>> # of TBs: ", num_TBs)

    #
    each_config.cost_load_TB_d2       = (cost_TB_load_A + cost_TB_load_B) 
    each_config.cost_load_input_d2    = (cost_TB_load_A + cost_TB_load_B) * num_TBs
    each_config.cost_store_output_d2  = cost_TB_store_C * num_TBs
    each_config.cost_total_d2         = each_config.cost_load_input_d2 + each_config.cost_store_output_d2

    #
    each_config.steps_main_loops_d2  = steps_main_loops

    #
    lines128_per_loop = lines128_A_per_inner * steps_inner_A_loops + lines128_B_per_inner * steps_inner_B_loops
    bytes_per_loop = (vol_A_per_inner * steps_inner_A_loops + vol_B_per_inner * steps_inner_B_loops) * 8

    each_config.transaction_per_loop = lines128_per_loop  # 호환 유지(의미는 lines128)

    # 추가로 디버깅/학습용 feature도 저장
    each_config.bytes_per_loop = bytes_per_loop
    each_config.lines128_per_loop = lines128_per_loop
    each_config.mem_issue_per_loop = mem_issue_A_per_inner * steps_inner_A_loops + mem_issue_B_per_inner * steps_inner_B_loops

    #each_config.transaction_per_loop = estimated_DRAM_transaction_A_per_inner_loops + estimated_DRAM_transaction_B_per_inner_loops
    
    #
    if opt_print == 1:
        print(f"Cost Input (Load)        : {each_config.cost_load_input_d2}")
        print(f"Cost Output (Store)      : {each_config.cost_store_output_d2}")
        print(f"Total Cost               : {each_config.cost_total_d2}")
        print(f"Transaction per loop     : {each_config.transaction_per_loop} (interpreted as lines128)")
        print(f"Bytes per loop           : {each_config.bytes_per_loop}")
        print(f"Mem-issue per loop       : {each_config.mem_issue_per_loop} (approx, affected by double2)")
        print(f"# of steps for main-loop : {each_config.steps_main_loops_d2}")
        print("============================================================================")

def tc_gen_cost_models_GM4(each_config, l_comb, idx, opt_print) :
    #
    if opt_print == 1:
        print(f"===[{ idx }]=============== [Cost Model][GMEM Load Inputs] =====================")
        print(f"Index Mappings : FRAG_X <- {each_config.list_FRAG_X}, FRAG_Y <- {each_config.list_FRAG_Y}")
        print(f"               : FRAG_K <- {each_config.list_FRAG_K}")
        print(f"               : REG_X  <- {each_config.list_REG_X}, REG_Y <- {each_config.list_REG_Y}")
        print(f"Tile Sizes     : {each_config.list_tile_sizes}")
        print(f"Pipeline stage : {each_config.stage}")
        print(f"list_comb      : {l_comb}")
        if hasattr(each_config, 'double2_flag'):
             print(f"Double2 Flags  : A={each_config.double2_flag[0]}, B={each_config.double2_flag[1]}")
        print("============================================================================")
    
    #
    size_TB = each_config.size_FRAG_X * each_config.size_FRAG_Y
    
    #
    #   For Internal Indicies,
    #
    size_FRAG_K = 1
    size_N_K    = 1
    for each_int_idx in each_config.list_FRAG_K :
        size_FRAG_K *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_int_idx)
        size_N_K *= tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_int_idx)

    #
    #   # of "main" loop (calculated by N_K / T_K)
    #
    steps_main_loops = math.ceil(size_N_K / size_FRAG_K)
    #stage_factor = max(1, steps_main_loops - (each_config.stage - 1)) / steps_main_loops

    #
    #   Check Types of Input such as [E_K, ...] or [E_A, ...]
    #
    opt_load_A_ext = -1     # -1: FVI = internal
    opt_load_B_ext = -1     #  1: FVI = external
    if tc_helper.tc_helper_find_index(each_config.list_tensor_B, each_config.list_tensor_A[0]) == -1 :
        opt_load_A_ext = 1
    
    if tc_helper.tc_helper_find_index(each_config.list_tensor_A, each_config.list_tensor_B[0]) == -1 :
        opt_load_B_ext = 1

    #
    #   Initial Values
    #
    size_continuous_elements_A = 1
    size_continuous_elements_B = 1
    size_continuous_elements_C = 1

    #
    if opt_print == 1 :
        print("-1 : FVI = internal, 1 : FVI = external")
        print(f"opt_load_A_ext : {opt_load_A_ext}, opt_load_B_ext : {opt_load_B_ext}")

    # ... (is_continuous calculation for A - unchanged) ...
    #   Input: A (Continuous)
    #
    is_continuous = 1
    for each_idx in each_config.list_tensor_A :
        if opt_load_A_ext == 1 : 
            if tc_helper.tc_helper_find_index(each_config.list_tensor_B, each_idx) != -1 : break
            else :
                if is_continuous == 1 :
                    size_continuous_elements_A *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
                    if tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx) != tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_idx) :
                        is_continuous = -1
                else : break
        else : 
            if tc_helper.tc_helper_find_index(each_config.list_tensor_B, each_idx) == -1 : break
            else :
                if is_continuous == 1 :
                    size_continuous_elements_A *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
                    if tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx) != tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_idx) :
                        is_continuous = -1
                else: break
    
    if opt_print == 1 :
        print (f"[A] is_continuous : {is_continuous}, size_continuous_elements_A : {size_continuous_elements_A}")

    #
    #   Input: A (FRAG and REG)
    #
    size_A_E_FRAG = 1
    size_A_K_FRAG = 1
    size_A_E_REG  = 1
    size_A_FVI    = 1

    for cnt, each_idx in enumerate(each_config.list_tensor_A) :
        if cnt == 0 :
            size_A_FVI = tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
        if tc_helper.tc_helper_find_index(each_config.list_tensor_B, each_idx) == -1 : # External Index
            if tc_helper.tc_helper_find_index(each_config.list_REG_X, each_idx) == -1 and tc_helper.tc_helper_find_index(each_config.list_REG_Y, each_idx) == -1 :
                size_A_E_FRAG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
            else :
                size_A_E_REG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
        else : # Internal Index
            size_A_K_FRAG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
    
    if opt_print == 1 :
        print(f"|FVI_A|  = {size_A_FVI}")
        print(f"|SMEM_A| = {size_A_E_REG * size_A_E_FRAG * size_A_K_FRAG}")

    #
    #   ### [UPDATED] Cost Calculation Logic for A with double2
    #
    # vol_A_per_TB = min(size_TB, size_A_E_REG * size_A_E_FRAG * size_A_K_FRAG)

    # # 1. double2 flag 확인 및 vector width 설정
    # vec_width_A = 1
    # if hasattr(each_config, 'double2_flag') and each_config.double2_flag[0] == 1:
    #     vec_width_A = 2

    # times_inner_A_FVI = 1
    # times_inner_A_TB = math.ceil(vol_A_per_TB / size_A_FVI)
    # steps_inner_A_loops = math.ceil(size_A_E_REG * size_A_E_FRAG * size_A_K_FRAG / size_TB)

    # # 2. Transaction 횟수를 vector width로 나눔 (1회 로드 당 2배의 데이터)
    # estimated_DRAM_transaction_A_per_FVI         = times_inner_A_FVI / vec_width_A
    # #estimated_DRAM_transaction_A_per_TB          = estimated_DRAM_transaction_A_per_FVI * times_inner_A_TB
    # estimated_DRAM_transaction_A_per_inner_loops = estimated_DRAM_transaction_A_per_FVI * steps_inner_A_loops
    # estimated_DRAM_transaction_A_per_main_loops  = estimated_DRAM_transaction_A_per_inner_loops * steps_main_loops

    # if opt_print == 1 :
    #     print(f"[A] Vector Width (double2)                     : {vec_width_A}")
    #     print(f"[A] estimated_DRAM_transaction_per_FVI         : {estimated_DRAM_transaction_A_per_FVI}")
    #     #print(f"[A] estimated_DRAM_transaction_per_TB          : {estimated_DRAM_transaction_A_per_TB}")
    #     print(f"[A] estimated_DRAM_transaction_per_main_loops  : {estimated_DRAM_transaction_A_per_main_loops}")

    # # 3. 최종 Cost는 정수로 반올림 (Transaction은 정수 단위)
    # cost_TB_load_A = math.ceil(estimated_DRAM_transaction_A_per_main_loops)

    total_A_tile_elems = size_A_E_REG * size_A_E_FRAG * size_A_K_FRAG

    # inner loop 횟수 (네 코드 그대로 유지)
    steps_inner_A_loops = math.ceil(total_A_tile_elems / size_TB)

    # inner-loop에서 로드하는 element 수(대부분 size_TB, 마지막은 remainder)
    vol_A_per_inner = min(size_TB, total_A_tile_elems)
    
    # double2 flag
    vec_width_A = 1
    if hasattr(each_config, 'double2_flag') and each_config.double2_flag[0] == 1:
        vec_width_A = 2

    # ---- 핵심: transaction(=stride로 나뉜 contiguous block 수/128B line 수)은 double2로 나누지 않음
    # 네 의도대로라면 contig_elems가 16이면 "16 doubles = 128B" 단위로 끊김
    # size_continuous_elements_A가 16보다 클 수도 있으니 그대로 사용(연속이 더 길면 block_bytes가 커져 lines_per_block이 늘어남)
    align_factor_A = 1.0  # 필요하면 1.25 같은 보정 사용 가능
    lines128_A_per_inner = estimate_lines128_strided(
        vol_elems=vol_A_per_inner,
        contig_elems=size_continuous_elements_A,
        elem_bytes=8,
        align_factor=align_factor_A
    )

    # inst(issue) 쪽만 double2 반영: (대략) element 수 / vec_width
    # 실제 thread mapping까지 반영하면 더 좋아지지만, 최소 수정으로는 이 정도가 안정적
    mem_issue_A_per_inner = int(math.ceil(vol_A_per_inner / float(vec_width_A)))

    # main loop 반영
    lines128_A_per_main = lines128_A_per_inner * steps_inner_A_loops * steps_main_loops
    mem_issue_A_per_main = mem_issue_A_per_inner * steps_inner_A_loops * steps_main_loops

    # 기존 cost 변수에선 transaction(비용)을 line 기반으로 넣는 게 더 일관됨
    cost_TB_load_A = int(math.ceil(lines128_A_per_main))
    
    # ... (is_continuous calculation for B - unchanged) ...
    #   Input: B
    #
    is_continuous = 1
    for each_idx in each_config.list_tensor_B :
        if opt_load_B_ext == 1 :
            if tc_helper.tc_helper_find_index(each_config.list_tensor_A, each_idx) != -1 : break
            else :
                if is_continuous == 1 :
                    size_continuous_elements_B *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
                    if tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx) != tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_idx) :
                        is_continuous = -1
                else : break
        else :
            if tc_helper.tc_helper_find_index(each_config.list_tensor_A, each_idx) == -1 : break
            else :
                if is_continuous == 1 :
                    size_continuous_elements_B *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
                    if tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx) != tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, each_idx) :
                        is_continuous = -1
                else : break
    
    if opt_print == 1 :
        print (f"[B] is_continuous : {is_continuous}, size_continuous_elements_B : {size_continuous_elements_B}")

    #
    #   Input: B (FRAG and REG)
    #
    size_B_E_FRAG = 1
    size_B_K_FRAG = 1
    size_B_E_REG  = 1
    size_B_FVI    = 1

    for cnt, each_idx in enumerate(each_config.list_tensor_B) :
        if cnt == 0 :
            size_B_FVI = tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
        if tc_helper.tc_helper_find_index(each_config.list_tensor_A, each_idx) == -1 :
            if tc_helper.tc_helper_find_index(each_config.list_REG_X, each_idx) == -1 and tc_helper.tc_helper_find_index(each_config.list_REG_Y, each_idx) == -1 :
                size_B_E_FRAG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
            else :
                size_B_E_REG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
        else :
            size_B_K_FRAG *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)
    
    if opt_print == 1 :
        print(f"|FVI_B|  = {size_B_FVI}")
        print(f"|SMEM_B| = {size_B_E_REG * size_B_E_FRAG * size_B_K_FRAG}")

    #
    #   ### [UPDATED] Cost Calculation Logic for B with double2
    #
    # vol_B_per_TB = min(size_TB, size_B_E_REG * size_B_E_FRAG * size_B_K_FRAG)

    # # 1. double2 flag 확인 및 vector width 설정
    # vec_width_B = 1
    # if hasattr(each_config, 'double2_flag') and each_config.double2_flag[1] == 1:
    #     vec_width_B = 2

    # times_inner_B_FVI = 1
    # times_inner_B_TB = math.ceil(vol_B_per_TB / size_B_FVI)
    # steps_inner_B_loops = math.ceil(size_B_E_REG * size_B_E_FRAG * size_B_K_FRAG / size_TB)

    # # 2. Transaction 횟수를 vector width로 나눔
    # estimated_DRAM_transaction_B_per_FVI         = times_inner_B_FVI / vec_width_B
    # #estimated_DRAM_transaction_B_per_TB          = estimated_DRAM_transaction_B_per_FVI * times_inner_B_TB
    # estimated_DRAM_transaction_B_per_inner_loops = estimated_DRAM_transaction_B_per_FVI * steps_inner_B_loops
    # estimated_DRAM_transaction_B_per_main_loops  = estimated_DRAM_transaction_B_per_inner_loops * steps_main_loops

    # if opt_print == 1 :
    #     print(f"[B] Vector Width (double2)                     : {vec_width_B}")
    #     print(f"[B] estimated_DRAM_transaction_per_FVI         : {estimated_DRAM_transaction_B_per_FVI}")
    #     #print(f"[B] estimated_DRAM_transaction_per_TB          : {estimated_DRAM_transaction_B_per_TB}")
    #     print(f"[B] estimated_DRAM_transaction_per_main_loops  : {estimated_DRAM_transaction_B_per_main_loops}")

    # # 3. 최종 Cost는 정수로 반올림
    # cost_TB_load_B = math.ceil(estimated_DRAM_transaction_B_per_main_loops)

    total_B_tile_elems = size_B_E_REG * size_B_E_FRAG * size_B_K_FRAG
    steps_inner_B_loops = math.ceil(total_B_tile_elems / size_TB)
    vol_B_per_inner = min(size_TB, total_B_tile_elems)

    vec_width_B = 1
    if hasattr(each_config, 'double2_flag') and each_config.double2_flag[1] == 1:
        vec_width_B = 2

    align_factor_B = 1.0
    lines128_B_per_inner = estimate_lines128_strided(
        vol_elems=vol_B_per_inner,
        contig_elems=size_continuous_elements_B,
        elem_bytes=8,
        align_factor=align_factor_B
    )
    mem_issue_B_per_inner = int(math.ceil(vol_B_per_inner / float(vec_width_B)))

    lines128_B_per_main = lines128_B_per_inner * steps_inner_B_loops * steps_main_loops
    mem_issue_B_per_main = mem_issue_B_per_inner * steps_inner_B_loops * steps_main_loops

    cost_TB_load_B = int(math.ceil(lines128_B_per_main))

    if opt_print == 1:
        print(f"[B] Vector Width (double2)                 : {vec_width_B}")
        print(f"[B] contig_elems (stride block)           : {size_continuous_elements_B}")
        print(f"[B] vol_B_per_inner (elems)               : {vol_B_per_inner}")
        print(f"[B] lines128_B_per_inner                  : {lines128_B_per_inner}")
        print(f"[B] mem_issue_B_per_inner (approx)        : {mem_issue_B_per_inner}")
        print(f"[B] cost_TB_load_B (lines128 over main)   : {cost_TB_load_B}")

    #
    #   Output C Calculation (unchanged)
    #
    size_output_TB = 1
    for each_idx in each_config.list_tensor_C :
        size_output_TB *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)

    size_output_fragment = 8 * 8
    cnt_output_Fragment = math.ceil(size_output_TB / size_output_fragment)

    per_row_transaction_inner_output_fragment = 8
    cost_TB_store_C = cnt_output_Fragment * per_row_transaction_inner_output_fragment
    
    #
    #   The # of Thread Blocks
    #
    num_TBs = l_comb[1]

    if opt_print == 1 :
        print (">>> # of TBs: ", num_TBs)

    #
    each_config.cost_load_TB_d2       = (cost_TB_load_A + cost_TB_load_B) 
    each_config.cost_load_input_d2    = (cost_TB_load_A + cost_TB_load_B) * num_TBs
    each_config.cost_store_output_d2  = cost_TB_store_C * num_TBs
    each_config.cost_total_d2         = each_config.cost_load_input_d2 + each_config.cost_store_output_d2

    #
    each_config.steps_main_loops_d2  = steps_main_loops

    #
    lines128_per_loop = lines128_A_per_inner * steps_inner_A_loops + lines128_B_per_inner * steps_inner_B_loops
    bytes_per_loop = (vol_A_per_inner * steps_inner_A_loops + vol_B_per_inner * steps_inner_B_loops) * 8

    each_config.transaction_per_loop = lines128_per_loop  # 호환 유지(의미는 lines128)

    # 추가로 디버깅/학습용 feature도 저장
    each_config.bytes_per_loop = bytes_per_loop
    each_config.lines128_per_loop = lines128_per_loop
    each_config.mem_issue_per_loop = mem_issue_A_per_inner * steps_inner_A_loops + mem_issue_B_per_inner * steps_inner_B_loops

    #each_config.transaction_per_loop = estimated_DRAM_transaction_A_per_inner_loops + estimated_DRAM_transaction_B_per_inner_loops
    
    #
    if opt_print == 1:
        print(f"Cost Input (Load)        : {each_config.cost_load_input_d2}")
        print(f"Cost Output (Store)      : {each_config.cost_store_output_d2}")
        print(f"Total Cost               : {each_config.cost_total_d2}")
        print(f"Transaction per loop     : {each_config.transaction_per_loop} (interpreted as lines128)")
        print(f"Bytes per loop           : {each_config.bytes_per_loop}")
        print(f"Mem-issue per loop       : {each_config.mem_issue_per_loop} (approx, affected by double2)")
        print(f"# of steps for main-loop : {each_config.steps_main_loops_d2}")
        print("============================================================================")

#
def tc_gen_cost_models_Kernels(each_config) :
    #
    opt_full_ext = True
    opt_full_int = True

    #
    for each_idx_tile in each_config.list_tile_sizes :
        #
        idx_name = each_idx_tile[0]
        idx_tile = each_idx_tile[1]

        #
        if tc_helper.tc_helper_find_index(each_config.list_FRAG_K, idx_name) != -1 :
            if tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, idx_name) % idx_tile != 0:
                opt_full_int = False               
        else :
            if tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, idx_name) % idx_tile != 0 :
                opt_full_ext = False
    
    #
    each_config.kernel_full_ext = opt_full_ext
    each_config.kernel_full_int = opt_full_int

#
def tc_gen_cost_models_Computes(each_config) :
    #
    size_output_TB = 1
    for each_idx in each_config.list_tensor_C :
        size_output_TB *= tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx)

    #
    size_FRAG_X = each_config.size_FRAG_X
    size_FRAG_Y = each_config.size_FRAG_Y
    size_FRAG_K = each_config.size_FRAG_K
    size_REG_X  = each_config.size_REG_X
    size_REG_Y  = each_config.size_REG_Y
    
    #
    x_shape = each_config.warp_shape[0]
    y_shape = each_config.warp_shape[1]

    #
    a_frag_per_warp = ((size_FRAG_Y / 8) * size_REG_Y) / y_shape
    b_frag_per_warp = ((size_FRAG_X / 8) * size_REG_X) / x_shape

    #
    reg_per_thread = a_frag_per_warp * b_frag_per_warp * 2

    #
    each_config.kernel_arithmetic_intensity = reg_per_thread / (a_frag_per_warp + b_frag_per_warp)
    each_config.flops_per_loop = (size_output_TB / 64) * (size_FRAG_K / 4)


#
def tc_gen_cost_models_pipeline(each_config) :
    #
    rho = each_config.transaction_per_loop / each_config.flops_per_loop
    rho_eff = rho / each_config.stage
    exposed_mem_fraction = rho_eff / (1.0 + rho_eff)

    each_config.cost_load_TB_d2_overlap = math.ceil(each_config.cost_load_TB_d2 * exposed_mem_fraction)
    each_config.cost_total_d2_overlap = each_config.cost_load_TB_d2_overlap * each_config.num_TBs + each_config.cost_store_output_d2

    # each_config.cost_load_TB_overlap = math.ceil(each_config.cost_load_TB * exposed_mem_fraction)
    # each_config.cost_total_overlap = each_config.cost_load_TB_overlap * each_config.num_TBs + each_config.cost_store_output

    each_config.overlap_frac = 1.0 - exposed_mem_fraction

#
def tc_gen_cost_models_pipeline2(each_config, p_any,
                                mma_lat_cycles=16.0,      # A100 FP64 m8n8k4 DMMA latency 근사
                                mem_line_cycles=200.0,    # 128B line 당 유효 비용(튜닝 파라미터)
                                stage_cap=4,              # stage 포화(4~6 추천)
                                exposed_floor=0.01,       # load cost가 0으로 꺼지지 않게 최소 노출 비율
                                exposed_ceiling=1.0,
                                alpha=3.0):     # 최대 1.0
    """
    overlap 모델:
      T_mem  = transaction_per_loop * mem_line_cycles
      T_comp = flops_per_loop(=mma_inst) * mma_lat_cycles
      T_mem_eff = T_mem / min(stage, stage_cap)
      exposed_mem_fraction = T_mem_eff / (T_mem_eff + T_comp)

    exposed_mem_fraction은 "메모리 때문에 실제로 드러나는(load가 가려지지 않는) 비율".
    """

    # 방어
    tx = float(getattr(each_config, "transaction_per_loop", 0.0) or 0.0)
    mma = float(getattr(each_config, "flops_per_loop", 0.0) or 0.0)  # 이름은 flops_per_loop지만 mma_inst로 사용
    stg = int(getattr(each_config, "stage", 1) or 1)
    stg_eff = max(1, min(stg, stage_cap))
    stage_scale = np.sqrt(stg_eff)

    # time proxy
    T_mem = tx * float(mem_line_cycles)
    T_comp = mma * float(mma_lat_cycles)

    # stage overlap 반영
    T_mem_eff = T_mem / float(stage_scale)

    # clip/floor
    exposed = T_mem_eff / (T_mem_eff + T_comp)
    exposed = max(exposed_floor, min(exposed_ceiling, exposed))
    #exposed = exposed * (1.0 + 0.5 * p_any)

    # 적용
    each_config.overlap_frac = 1.0 - exposed  # 보기 편하게 유지

    each_config.cost_load_TB_d2_overlap = math.ceil(each_config.cost_load_TB_d2 * exposed)
    each_config.cost_total_d2_overlap = (each_config.cost_load_TB_d2_overlap * each_config.num_TBs + each_config.cost_store_output_d2) * (1.0 + alpha * p_any)


#
def tc_gen_cost_full_partial(each_config) :
    #
    idx_tile_list = list(reversed(each_config.combined_tile_size))  # [(idx, tile), ...]
    num_index     = len(idx_tile_list)
    
    #
    mapped_index = [(each_config.list_FRAG_X)[0]] + [(each_config.list_FRAG_Y)[0]] + [(each_config.list_FRAG_K)[0]] + each_config.list_REG_X + each_config.list_REG_Y
    
    #
    num_tb_each_idx = []
    idx_names       = []
    tile_sizes      = []
    rep_sizes       = []
    idx_index       = []

    for i, (idx, tile_size) in enumerate(idx_tile_list) :
        size_i = tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, idx)
        nblk_i = int(math.ceil(size_i / tile_size))

        idx_names.append(idx)
        idx_index.append([i, idx[0]])
        tile_sizes.append(int(tile_size))
        rep_sizes.append(int(size_i))
        num_tb_each_idx.append(nblk_i)

    #
    prod_num_tbs = int(np.prod(num_tb_each_idx)) if num_index > 0 else 0
    if prod_num_tbs != int(each_config.num_TBs) :
        print(f"[WARN] product(num_tb_each_idx)={prod_num_tbs} != each_config.num_TBs={each_config.num_TBs}")
        raise ValueError("num_TBs mismatch")

    #
    bidx         = np.arange(int(each_config.num_TBs), dtype=np.int64)
    blk_idx_cols = []
    for i in range(num_index) :
        stride = int(np.prod(num_tb_each_idx[i+1:])) if (i + 1) < num_index else 1
        blk_i  = bidx // stride
        blk_idx_cols.append(blk_i)
        bidx   = bidx % stride

    blk_idx = np.stack(blk_idx_cols, axis=1)  # (num_TBs, num_index)
    # blk_idx[tb, i] = tb번째 block의 i번째 차원 block index

    #
    partial_cols = []
    for i in range(num_index) :
        nblk = num_tb_each_idx[i]
        size_i = rep_sizes[i]
        tile_i = tile_sizes[i]

        is_partial = (blk_idx[:, i] == (nblk - 1)) & ((size_i % tile_i) != 0)
        partial_cols.append(is_partial.astype(np.int8))

    partial_flags = np.stack(partial_cols, axis=1)  # (num_TBs, num_index), 0=full, 1=partial

    # match index of each input index
    t2 = each_config.list_tensor_A
    v2 = each_config.list_tensor_B
    t3 = each_config.list_tensor_C
    split_tile_size = each_config.list_tile_sizes
    num_warp = (each_config.size_FRAG_X * each_config.size_FRAG_Y) // 32
    t2_mapped_index = [i for i in t2 if i in mapped_index]
    v2_mapped_index = [i for i in v2 if i in mapped_index]
    
    #
    per_loop_load_t2 = 1
    for i, idx in enumerate(t2) :
        per_loop_load_t2 *= tc_helper.tc_helper_find_value(split_tile_size, idx)
    
    #
    per_loop_load_v2 = 1
    for i, idx in enumerate(v2) :
        per_loop_load_v2 *= tc_helper.tc_helper_find_value(split_tile_size, idx)

    #
    t2_fvi_tile_size = tc_helper.tc_helper_find_value(split_tile_size, t2[0])
    v2_fvi_tile_size = tc_helper.tc_helper_find_value(split_tile_size, v2[0])

    #
    double2_flag = each_config.double2_flag
    if t3[0] in t2 :
        t2_double2_flag = double2_flag[0]
        v2_double2_flag = double2_flag[1]
    else :
        t2_double2_flag = double2_flag[0]
        v2_double2_flag = double2_flag[1]
    
    # t2
    tmp = 1
    t2_thread_layout_in_warp = {}
    for i, idx in enumerate(t2) :
        #
        tile = tc_helper.tc_helper_find_value(split_tile_size, idx)
        
        #
        if (tmp >= 32) :
            t2_thread_layout_in_warp[idx] = 1
        elif (idx not in mapped_index) :
            t2_thread_layout_in_warp[idx] = 0
        else :
            if i == 0 :
                if t2_double2_flag :
                    t2_thread_layout_in_warp[idx] = t2_fvi_tile_size // 2
                    tmp *= t2_fvi_tile_size // 2
                else :
                    t2_thread_layout_in_warp[idx] = t2_fvi_tile_size
                    tmp *= t2_fvi_tile_size
            else :
                rest = 32 // tmp
                covered = min(rest, tile)
                t2_thread_layout_in_warp[idx] = covered
                tmp *= covered

    #
    t2_warp_cover = {}
    for i, idx in enumerate(t2) :
        if i == 0 and t2_double2_flag :
            t2_warp_cover[idx] = t2_thread_layout_in_warp[idx] * 2
        else :
            t2_warp_cover[idx] = t2_thread_layout_in_warp[idx]

    #
    radices = []
    for key in t2_mapped_index:
        base = t2_warp_cover.get(key, 0)
        tile = tc_helper.tc_helper_find_value(split_tile_size, key)

        if base <= 0:
            radix = 1  # base=0이면 이 차원은 분해에 기여 못하니 1로 둠(값은 계속 0 유지)
        else:
            radix = max(1, math.ceil(tile / base))
        radices.append(radix)

    # 2) warp_id를 mixed-radix로 분해해서 step 결정
    t2_tb_cover = {}
    for warp_id in range(num_warp):
        t2_tb_cover[warp_id] = t2_warp_cover.copy()

        x = warp_id
        for key, radix in zip(t2_mapped_index, radices):
            base = t2_warp_cover.get(key, 0)
            tile = tc_helper.tc_helper_find_value(split_tile_size, key)

            step = x % radix
            x //= radix

            if base <= 0:
                # base가 0이면 그대로 0 유지 (원하면 여기서 다른 규칙 적용 가능)
                t2_tb_cover[warp_id][key] = base
            else:
                val = base * (step + 1)
                # tile을 넘으면 cap (안 넘을 수도 있지만 안전)
                t2_tb_cover[warp_id][key] = min(val, tile)

    # v2
    tmp = 1
    v2_thread_layout_in_warp = {}
    for i, idx in enumerate(v2) :
        #
        tile = tc_helper.tc_helper_find_value(split_tile_size, idx)
        
        #
        if (tmp >= 32) :
            v2_thread_layout_in_warp[idx] = 1
        elif (idx not in mapped_index) :
            v2_thread_layout_in_warp[idx] = 0
        else :
            if i == 0 :
                if v2_double2_flag :
                    v2_thread_layout_in_warp[idx] = v2_fvi_tile_size // 2
                    tmp *= v2_fvi_tile_size // 2
                else :
                    v2_thread_layout_in_warp[idx] = v2_fvi_tile_size
                    tmp *= v2_fvi_tile_size
            else :
                rest = 32 // tmp
                covered = min(rest, tile)
                v2_thread_layout_in_warp[idx] = covered
                tmp *= covered

    #
    v2_warp_cover = {}
    for i, idx in enumerate(v2) :
        if i == 0 and v2_double2_flag :
            v2_warp_cover[idx] = v2_thread_layout_in_warp[idx] * 2
        else :
            v2_warp_cover[idx] = v2_thread_layout_in_warp[idx]

    #
    radices = []
    for key in v2_mapped_index:
        base = v2_warp_cover.get(key, 0)
        tile = tc_helper.tc_helper_find_value(split_tile_size, key)

        if base <= 0:
            radix = 1  # base=0이면 이 차원은 분해에 기여 못하니 1로 둠(값은 계속 0 유지)
        else:
            radix = max(1, math.ceil(tile / base))
        radices.append(radix)

    # 2) warp_id를 mixed-radix로 분해해서 step 결정
    v2_tb_cover = {}
    for warp_id in range(num_warp):
        v2_tb_cover[warp_id] = v2_warp_cover.copy()

        x = warp_id
        for key, radix in zip(v2_mapped_index, radices):
            base = v2_warp_cover.get(key, 0)
            tile = tc_helper.tc_helper_find_value(split_tile_size, key)

            step = x % radix
            x //= radix

            if base <= 0:
                # base가 0이면 그대로 0 유지 (원하면 여기서 다른 규칙 적용 가능)
                v2_tb_cover[warp_id][key] = base
            else:
                val = base * (step + 1)
                # tile을 넘으면 cap (안 넘을 수도 있지만 안전)
                v2_tb_cover[warp_id][key] = min(val, tile)

    partial_ratio = partial_flags.sum(axis=0) / each_config.num_TBs
    total_partial_ratio = partial_ratio.sum()
    p_any = float((partial_flags.sum(axis=1) > 0).mean())

    # 원하는 형태로 반환
    return p_any     # (N, D) 0=full, 1=partial


# -----------------------------
# Split-aware utilities
# -----------------------------
_split_pat = re.compile(r"^([A-Za-z_]+)(\d+)$")

def _base_name(idx):
    s = str(idx)
    m = _split_pat.match(s)
    return m.group(1) if m else s

def _split_groups(idx_names_str):
    """
    idx_names_str: list[str] like ['a1','a2','b',...]
    return: { 'a': [('a1',1),('a2',2)], ... } (digit asc)
    """
    g = {}
    for s in idx_names_str:
        m = _split_pat.match(s)
        if not m:
            continue
        base, d = m.group(1), int(m.group(2))
        g.setdefault(base, []).append((s, d))
    out = {}
    for base, items in g.items():
        if len(items) >= 2:
            out[base] = sorted(items, key=lambda x: x[1])
    return out

def build_tb_partial_penalty_split_aware(blk_idx, idx_names, tile_sizes, rep_size_dict):
    """
    TB별 partial penalty를 '원래(base) index' 기준으로 만든다.
    - split base(예: a1,a2)는 base='a'에 대해 penalty(1~2)를 overflow 비율로 계산
    - unsplit은 기존 remainder 기반으로 penalty=1 or 2

    return:
      penalty_by_base: dict[base] -> np.ndarray(shape=(num_TBs,), float32)  (1~2)
    """
    num_TBs = blk_idx.shape[0]
    idx_names_str = [str(x) for x in idx_names]

    name2dim = {s: i for i, s in enumerate(idx_names_str)}
    name2tile = {s: int(tile_sizes[i]) for i, s in enumerate(idx_names_str)}

    penalty_by_base = {}

    # 1) split 그룹 처리
    groups = _split_groups(idx_names_str)
    used_children = set()

    for base, items in groups.items():
        # 현재 커널 규칙: 1,2로만 split (a1,a2)라 가정
        if base not in rep_size_dict:
            continue
        if len(items) != 2:
            continue

        (c1, _d1), (c2, _d2) = items[0], items[1]  # a1, a2
        used_children.update([c1, c2])

        A = int(rep_size_dict[base])

        Ta1 = name2tile[c1]
        Ta2 = name2tile[c2]
        Ta  = Ta1 * Ta2

        dim1 = name2dim[c1]
        dim2 = name2dim[c2]

        a1_blk = blk_idx[:, dim1].astype(np.int64)
        a2_blk = blk_idx[:, dim2].astype(np.int64)

        # 너 커널의 위치 환산식: a = a2 * Ta1 + a1
        a_start = a2_blk * Ta1 + a1_blk

        ov = np.maximum(0, a_start + Ta - A).astype(np.float64)
        frac = 1.0 - (ov / float(Ta))
        frac = np.clip(frac, 0.0, 1.0)

        # penalty: 1~2 (overflow 클수록 증가)
        penalty = 1.0 + (1.0 - frac)
        penalty_by_base[base] = penalty.astype(np.float32)

    # 2) unsplit (또는 split child가 아닌) 처리
    for s in idx_names_str:
        if s in used_children:
            continue
        base = _base_name(s)
        if base in penalty_by_base:
            continue
        if base not in rep_size_dict:
            continue

        dim = name2dim[s]
        tile = name2tile[s]
        size = int(rep_size_dict[base])

        nblk = int(math.ceil(size / tile))
        is_partial = (blk_idx[:, dim] == (nblk - 1)) & ((size % tile) != 0)

        penalty = np.ones(num_TBs, dtype=np.float32)
        penalty[is_partial] = 2.0
        penalty_by_base[base] = penalty

    return penalty_by_base

def tensor_load_bases(tensor_idx_list, tb_cover_by_warp, mapped_index):
    """
    텐서에서 warp load에 관여하는 base index 집합:
    - idx가 mapped_index에 있고
    - 어떤 warp에서든 cover>0이면 관여로 간주
    """
    active = set()
    for idx in tensor_idx_list:
        if idx not in mapped_index:
            continue
        b = _base_name(idx)
        for w in tb_cover_by_warp.keys():
            if tb_cover_by_warp[w].get(idx, 0) > 0:
                active.add(b)
                break
    return active

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

def tb_repeat_factor(per_loop_load, num_warp, double2_flag):
    vec = 2 if double2_flag else 1
    once = num_warp * 32 * vec
    return int(math.ceil(int(per_loop_load) / int(once)))

def warp_transactions_for_tensor(tensor_idx_list, tb_cover_by_warp, vec_elems):
    """
    warp들이 읽는 element 수(cover 곱)를 vec_elems로 나눠 ceil한 값의 합으로 tx 근사
    """
    total_tx = 0
    for _, cover in tb_cover_by_warp.items():
        elems = 1
        for idx in tensor_idx_list:
            c = cover.get(idx, 0)
            if c > 0:
                elems *= int(c)
        total_tx += int(math.ceil(elems / vec_elems))
    
    return int(total_tx)

import math

def build_tb_cover_for_tensor_base_method(
    tensor_idx_list,
    split_tile_size,
    mapped_index,
    fvi_tile_size,
    double2_flag,
    num_warp
):
    """
    t2/v2에서 하던 방식과 동일하게,
    - thread_layout_in_warp 계산
    - warp_cover 계산
    - mixed-radix로 warp_id마다 cover 확장해서 tb_cover_by_warp 생성

    return:
      tb_cover_by_warp: dict[warp_id] -> dict[idx] = cover_elems
    """

    # -------------------------
    # 1) thread layout in warp
    # -------------------------
    tmp = 1
    thread_layout_in_warp = {}

    for i, idx in enumerate(tensor_idx_list):
        tile = int(tc_helper.tc_helper_find_value(split_tile_size, idx))

        if tmp >= 32:
            thread_layout_in_warp[idx] = 1
        elif idx not in mapped_index:
            thread_layout_in_warp[idx] = 0
        else:
            if i == 0:
                if double2_flag:
                    # double2면 thread당 2elem이므로 "thread layout"은 절반으로 잡고
                    # warp_cover에서 다시 *2 해주는 방식(t2/v2와 동일)
                    thread_layout_in_warp[idx] = int(fvi_tile_size // 2)
                    tmp *= int(fvi_tile_size // 2)
                else:
                    thread_layout_in_warp[idx] = int(fvi_tile_size)
                    tmp *= int(fvi_tile_size)
            else:
                rest = int(32 // tmp)
                covered = int(min(rest, tile))
                thread_layout_in_warp[idx] = covered
                tmp *= covered

    # -------------------------
    # 2) warp cover
    # -------------------------
    warp_cover = {}
    for i, idx in enumerate(tensor_idx_list):
        if i == 0 and double2_flag:
            warp_cover[idx] = int(thread_layout_in_warp[idx]) * 2
        else:
            warp_cover[idx] = int(thread_layout_in_warp[idx])

    # -------------------------
    # 3) mixed-radix 분해로 warp별 cover 확장
    # -------------------------
    mapped_in_tensor = [i for i in tensor_idx_list if i in mapped_index]

    radices = []
    for key in mapped_in_tensor:
        base = int(warp_cover.get(key, 0))
        tile = int(tc_helper.tc_helper_find_value(split_tile_size, key))
        if base <= 0:
            radix = 1
        else:
            radix = max(1, int(math.ceil(tile / base)))
        radices.append(radix)

    tb_cover_by_warp = {}
    for warp_id in range(int(num_warp)):
        tb_cover_by_warp[warp_id] = warp_cover.copy()

        x = int(warp_id)
        for key, radix in zip(mapped_in_tensor, radices):
            base = int(warp_cover.get(key, 0))
            tile = int(tc_helper.tc_helper_find_value(split_tile_size, key))

            step = x % int(radix)
            x //= int(radix)

            if base <= 0:
                tb_cover_by_warp[warp_id][key] = base
            else:
                tb_cover_by_warp[warp_id][key] = int(min(base * (step + 1), tile))

    return tb_cover_by_warp

def _max_penalty_for_bases(penalty_by_base, bases, num_TBs, base_floor=1.5):
    """
    bases: set/list of base strings
    return: np.ndarray (num_TBs,) with max penalty across those bases, default base_floor
    """
    arrs = []
    for b in bases:
        a = penalty_by_base.get(b, None)
        if a is not None:
            arrs.append(a.astype(np.float32, copy=False))
    if not arrs:
        return np.full(num_TBs, base_floor, dtype=np.float32)
    M = np.stack(arrs, axis=0)  # (B, num_TBs)
    out = M.max(axis=0)
    # 기본값(1.5) 보장
    out = np.maximum(out, np.float32(base_floor))
    return out

def _reg_valid_product(penalty_by_base, reg_bases_in_out, num_TBs):
    """
    reg_valid[tb] = Π_b clip(2 - penalty[b][tb], 0, 1)
    """
    arrs = []
    for b in reg_bases_in_out:
        a = penalty_by_base.get(b, None)
        if a is None:
            continue
        # valid = clip(2 - p, 0, 1)
        v = np.clip(2.0 - a.astype(np.float32, copy=False), 0.0, 1.0)
        arrs.append(v)
    if not arrs:
        return np.ones(num_TBs, dtype=np.float32)
    V = np.stack(arrs, axis=0)  # (R, num_TBs)
    return V.prod(axis=0)

# -----------------------------
# Integrated main function
# -----------------------------
def tc_gen_cost_full_partial2(each_config):
    # ---- 0) (idx, tile) 순서 고정 ----
    idx_tile_list = list(reversed(each_config.combined_tile_size))  # [(idx, tile), ...]
    config_len = len(idx_tile_list)

    mapped_index = (
        [(each_config.list_FRAG_X)[0]] +
        [(each_config.list_FRAG_Y)[0]] +
        [(each_config.list_FRAG_K)[0]] +
        each_config.list_REG_X +
        each_config.list_REG_Y
    )

    # ---- 1) num_tb_each_idx 계산 + idx 이름도 같이 보관 ----
    num_tb_each_idx = []
    idx_names = []
    tile_sizes = []
    rep_sizes = []
    idx_index = []

    for i, (idx, tile_size) in enumerate(idx_tile_list):
        size_i = int(tc_helper.tc_helper_find_value(each_config.list_representative_problem_size, idx))
        nblk_i = int(math.ceil(size_i / int(tile_size)))

        idx_names.append(idx)
        idx_index.append([i, idx[0]])
        tile_sizes.append(int(tile_size))
        rep_sizes.append(int(size_i))
        num_tb_each_idx.append(nblk_i)

    # ---- 2) TB 개수 검증 ----
    prod_num_tbs = int(np.prod(num_tb_each_idx)) if config_len > 0 else 0
    if prod_num_tbs != int(each_config.num_TBs):
        print(f"[WARN] product(num_tb_each_idx)={prod_num_tbs} != each_config.num_TBs={each_config.num_TBs}")
        raise ValueError("num_TBs mismatch")

    # ---- 3) linear blockIdx -> multi-dim blk_idx ----
    bidx = np.arange(int(each_config.num_TBs), dtype=np.int64)
    blk_idx_cols = []

    for i in range(config_len):
        stride = int(np.prod(num_tb_each_idx[i + 1:])) if (i + 1) < config_len else 1
        blk_i = bidx // stride
        blk_idx_cols.append(blk_i)
        bidx = bidx % stride

    blk_idx = np.stack(blk_idx_cols, axis=1)  # (num_TBs, config_len)

    # ------------------------------------------------------------
    # Split-aware partial penalty (TB별, base index 기준)
    # ------------------------------------------------------------
    # rep_size_dict: base index 크기 딕셔너리 (a1/a2의 base는 'a'로 들어있어야 함)
    rep_size_dict = {}
    for k, v in each_config.list_representative_problem_size:
        rep_size_dict[str(k)] = int(v)

    penalty_by_base = build_tb_partial_penalty_split_aware(
        blk_idx=blk_idx,
        idx_names=idx_names,
        tile_sizes=tile_sizes,
        rep_size_dict=rep_size_dict
    )

    # ------------------------------------------------------------
    # 텐서/타일/warp 구성
    # ------------------------------------------------------------
    t2 = each_config.list_tensor_A
    v2 = each_config.list_tensor_B
    t3 = each_config.list_tensor_C
    split_tile_size = each_config.list_tile_sizes

    num_warp = (each_config.size_FRAG_X * each_config.size_FRAG_Y) // 32

    t2_mapped_index = [i for i in t2 if i in mapped_index]
    v2_mapped_index = [i for i in v2 if i in mapped_index]

    #
    per_loop_load_t2 = 1
    for i, idx in enumerate(t2) :
        per_loop_load_t2 *= tc_helper.tc_helper_find_value(split_tile_size, idx)
    
    #
    per_loop_load_v2 = 1
    for i, idx in enumerate(v2) :
        per_loop_load_v2 *= tc_helper.tc_helper_find_value(split_tile_size, idx)

    # FVI tile sizes
    t2_fvi_tile_size = int(tc_helper.tc_helper_find_value(split_tile_size, t2[0]))
    v2_fvi_tile_size = int(tc_helper.tc_helper_find_value(split_tile_size, v2[0]))

    # double2 flags
    double2_flag = each_config.double2_flag
    if t3[0] in t2:
        t2_double2_flag = bool(double2_flag[0])
        v2_double2_flag = bool(double2_flag[1])
    else:
        t2_double2_flag = bool(double2_flag[1])
        v2_double2_flag = bool(double2_flag[0])

    # -------------------------
    # t2 thread layout in warp
    # -------------------------
    tmp = 1
    t2_thread_layout_in_warp = {}
    for i, idx in enumerate(t2):
        tile = int(tc_helper.tc_helper_find_value(split_tile_size, idx))

        if tmp >= 32:
            t2_thread_layout_in_warp[idx] = 1
        elif idx not in mapped_index:
            t2_thread_layout_in_warp[idx] = 0
        else:
            if i == 0:
                if t2_double2_flag:
                    t2_thread_layout_in_warp[idx] = t2_fvi_tile_size // 2
                    tmp *= (t2_fvi_tile_size // 2)
                else:
                    t2_thread_layout_in_warp[idx] = t2_fvi_tile_size
                    tmp *= t2_fvi_tile_size
            else:
                rest = 32 // tmp
                covered = min(rest, tile)
                t2_thread_layout_in_warp[idx] = covered
                tmp *= covered

    t2_warp_cover = {}
    for i, idx in enumerate(t2):
        if i == 0 and t2_double2_flag:
            t2_warp_cover[idx] = t2_thread_layout_in_warp[idx] * 2
        else:
            t2_warp_cover[idx] = t2_thread_layout_in_warp[idx]

    # t2: mixed-radix -> warp cover within TB
    t2_radices = []
    for key in t2_mapped_index:
        base = t2_warp_cover.get(key, 0)
        tile = int(tc_helper.tc_helper_find_value(split_tile_size, key))
        radix = 1 if base <= 0 else max(1, int(math.ceil(tile / base)))
        t2_radices.append(radix)

    t2_tb_cover = {}
    for warp_id in range(num_warp):
        t2_tb_cover[warp_id] = t2_warp_cover.copy()
        x = warp_id
        for key, radix in zip(t2_mapped_index, t2_radices):
            base = t2_warp_cover.get(key, 0)
            tile = int(tc_helper.tc_helper_find_value(split_tile_size, key))
            step = x % radix
            x //= radix
            if base <= 0:
                t2_tb_cover[warp_id][key] = base
            else:
                t2_tb_cover[warp_id][key] = min(base * (step + 1), tile)

    # -------------------------
    # v2 thread layout in warp
    # -------------------------
    tmp = 1
    v2_thread_layout_in_warp = {}
    for i, idx in enumerate(v2):
        tile = int(tc_helper.tc_helper_find_value(split_tile_size, idx))

        if tmp >= 32:
            v2_thread_layout_in_warp[idx] = 1
        elif idx not in mapped_index:
            v2_thread_layout_in_warp[idx] = 0
        else:
            if i == 0:
                if v2_double2_flag:
                    v2_thread_layout_in_warp[idx] = v2_fvi_tile_size // 2
                    tmp *= (v2_fvi_tile_size // 2)
                else:
                    v2_thread_layout_in_warp[idx] = v2_fvi_tile_size
                    tmp *= v2_fvi_tile_size
            else:
                rest = 32 // tmp
                covered = min(rest, tile)
                v2_thread_layout_in_warp[idx] = covered
                tmp *= covered

    v2_warp_cover = {}
    for i, idx in enumerate(v2):
        if i == 0 and v2_double2_flag:
            v2_warp_cover[idx] = v2_thread_layout_in_warp[idx] * 2
        else:
            v2_warp_cover[idx] = v2_thread_layout_in_warp[idx]

    # v2: mixed-radix -> warp cover within TB
    v2_radices = []
    for key in v2_mapped_index:
        base = v2_warp_cover.get(key, 0)
        tile = int(tc_helper.tc_helper_find_value(split_tile_size, key))
        radix = 1 if base <= 0 else max(1, int(math.ceil(tile / base)))
        v2_radices.append(radix)

    v2_tb_cover = {}
    for warp_id in range(num_warp):
        v2_tb_cover[warp_id] = v2_warp_cover.copy()
        x = warp_id
        for key, radix in zip(v2_mapped_index, v2_radices):
            base = v2_warp_cover.get(key, 0)
            tile = int(tc_helper.tc_helper_find_value(split_tile_size, key))
            step = x % radix
            x //= radix
            if base <= 0:
                v2_tb_cover[warp_id][key] = base
            else:
                v2_tb_cover[warp_id][key] = min(base * (step + 1), tile)

    # ------------------------------------------------------------
    # Transaction cost 계산 (split-aware partial 반영 + contraction loop 반영)
    # ------------------------------------------------------------
    # 1) contraction loop 반복 횟수
    internal_indices = infer_internal_indices(t2, v2, t3)
    inner_iters = num_inner_iters(internal_indices, split_tile_size, each_config.list_representative_problem_size)

    # 2) 텐서별 "load에 관여하는 base" (split child -> base로 묶임)
    t2_load_bases = tensor_load_bases(t2, t2_tb_cover, mapped_index)
    v2_load_bases = tensor_load_bases(v2, v2_tb_cover, mapped_index)

    # 3) base tx (TB마다 동일하다고 근사: warp 분배는 TB 내부 구조로 동일)
    t2_vec = 2 if t2_double2_flag else 1
    v2_vec = 2 if v2_double2_flag else 1

    t2_base_tx_per_tb = warp_transactions_for_tensor(t2, t2_tb_cover, vec_elems=t2_vec)
    v2_base_tx_per_tb = warp_transactions_for_tensor(v2, v2_tb_cover, vec_elems=v2_vec)
    repeat_t2 = tb_repeat_factor(per_loop_load_t2, num_warp, t2_double2_flag)
    repeat_v2 = tb_repeat_factor(per_loop_load_v2, num_warp, v2_double2_flag)

    # 4) TB별 penalty 적용
    # num_TBs = int(each_config.num_TBs)
    # t2_tx_cost_tb = np.zeros(num_TBs, dtype=np.float64)
    # v2_tx_cost_tb = np.zeros(num_TBs, dtype=np.float64)
    # t2_tx_cost_tb_per_loop = np.zeros(num_TBs, dtype=np.float64)
    # v2_tx_cost_tb_per_loop = np.zeros(num_TBs, dtype=np.float64)

    # for tb in range(num_TBs):
    #     # "하나라도 partial이면 2배"에 가장 가까운 동작: max penalty 사용
    #     t2_pen = 1.5
    #     for b in t2_load_bases:
    #         if b in penalty_by_base:
    #             t2_pen = max(t2_pen, float(penalty_by_base[b][tb]))

    #     v2_pen = 1.5
    #     for b in v2_load_bases:
    #         if b in penalty_by_base:
    #             v2_pen = max(v2_pen, float(penalty_by_base[b][tb]))

    #     t2_tx_cost_tb[tb] = t2_base_tx_per_tb * t2_pen * inner_iters * repeat_t2
    #     v2_tx_cost_tb[tb] = v2_base_tx_per_tb * v2_pen * inner_iters * repeat_v2

    #     t2_tx_cost_tb_per_loop[tb] = t2_base_tx_per_tb * t2_pen * repeat_t2
    #     v2_tx_cost_tb_per_loop[tb] = v2_base_tx_per_tb * v2_pen * repeat_v2

    # t2_total_tx_cost = float(t2_tx_cost_tb.sum())
    # v2_total_tx_cost = float(v2_tx_cost_tb.sum())
    # # mean_t2_per_loop_tx = float(t2_tx_cost_tb_per_loop.mean())
    # # mean_v2_per_loop_tx = float(v2_tx_cost_tb_per_loop.mean())

    # total_tx_cost = t2_total_tx_cost + v2_total_tx_cost

    # #
    # each_config.partial_total_cost_d2 = total_tx_cost + each_config.cost_store_output_d2
    # each_config.partial_load_a_cost_d2 = t2_total_tx_cost
    # each_config.partial_load_b_cost_d2 = v2_total_tx_cost
    # each_config.partial_load_a_cost_tb_d2 = t2_tx_cost_tb
    # each_config.partial_load_b_cost_tb_d2 = v2_tx_cost_tb


    # --- t2/v2 penalty (vectorized) ---
    num_TBs = int(each_config.num_TBs)
    t2_pen_vec = _max_penalty_for_bases(penalty_by_base, t2_load_bases, num_TBs, base_floor=1.5)
    v2_pen_vec = _max_penalty_for_bases(penalty_by_base, v2_load_bases, num_TBs, base_floor=1.5)

    # tx cost per TB (vectorized)
    t2_tx_cost_tb = (t2_base_tx_per_tb * t2_pen_vec * inner_iters * repeat_t2).astype(np.float64)
    v2_tx_cost_tb = (v2_base_tx_per_tb * v2_pen_vec * inner_iters * repeat_v2).astype(np.float64)

    t2_tx_cost_tb_per_loop = (t2_base_tx_per_tb * t2_pen_vec * repeat_t2).astype(np.float64)
    v2_tx_cost_tb_per_loop = (v2_base_tx_per_tb * v2_pen_vec * repeat_v2).astype(np.float64)

    t2_total_tx_cost = float(t2_tx_cost_tb.sum())
    v2_total_tx_cost = float(v2_tx_cost_tb.sum())
    total_tx_cost = t2_total_tx_cost + v2_total_tx_cost

    # 기존 each_config 저장 (그대로)
    each_config.partial_load_a_cost_d2 = t2_total_tx_cost
    each_config.partial_load_b_cost_d2 = v2_total_tx_cost
    each_config.partial_load_a_cost_tb_d2 = t2_tx_cost_tb
    each_config.partial_load_b_cost_tb_d2 = v2_tx_cost_tb
    
    # --- t2/v2 penalty (vectorized) ---
    frag_x = (each_config.list_FRAG_X)[0]
    frag_y = (each_config.list_FRAG_Y)[0]

    # combined_tile_size는 base 형태라 했으니 base_name은 그냥 str로 충분
    frag_x_b = str(frag_x)
    frag_y_b = str(frag_y)

    # output tile element 수
    size_output_TB = 1
    for each_idx in each_config.list_tensor_C:
        size_output_TB *= int(tc_helper.tc_helper_find_value(each_config.list_tile_sizes, each_idx))

    repeat_store = int(math.ceil(size_output_TB / float(num_warp * 32)))

    # REG base들 (output에 실제로 들어간 것만)
    reg_bases = set(str(x) for x in (each_config.list_REG_X + each_config.list_REG_Y))
    out_bases = set(str(x) for x in each_config.list_tensor_C)
    reg_bases_in_out = [b for b in reg_bases if b in out_bases]

    # output store에 관여하는 base들(원하면 out_bases 전체 쓰면 됨)
    # 여기서는 store penalty max를 out_bases 전체로 적용
    store_bases_for_pen = out_bases

    # TB별 store cost 배열
    store_cost_tb_d2 = np.zeros(num_TBs, dtype=np.float64)

    # store_matrix_sync base tx (TB 내부 구조가 동일하다고 보고 TB당 동일한 base tx로 근사)
    # -> "한 번 store issue에서" tx proxy. (load에서 쓰던 warp_transactions_for_tensor를 재사용)
    # output에서도 같은 tb_cover를 만들고 싶으면 아래처럼:
    out_fvi_tile_size = int(tc_helper.tc_helper_find_value(split_tile_size, t3[0]))
    t3_tb_cover = build_tb_cover_for_tensor_base_method(
        tensor_idx_list=t3,
        split_tile_size=split_tile_size,
        mapped_index=mapped_index,
        fvi_tile_size=out_fvi_tile_size,
        double2_flag=0,
        num_warp=num_warp
    )
    store_base_tx_per_tb = float(warp_transactions_for_tensor(t3, t3_tb_cover, vec_elems=1))

    # for tb in range(num_TBs):
    #     # (A) 일반 store penalty: 원래 방식(max), 기본 1.5
    #     store_pen = 1.5
    #     for b in store_bases_for_pen:
    #         if b in penalty_by_base:
    #             store_pen = max(store_pen, float(penalty_by_base[b][tb]))

    #     # (B) FRAG partial이면 store 2배
    #     frag_factor = 1.0
    #     if (frag_x_b in penalty_by_base and float(penalty_by_base[frag_x_b][tb]) > 1.0) or \
    #        (frag_y_b in penalty_by_base and float(penalty_by_base[frag_y_b][tb]) > 1.0):
    #         frag_factor = 2.0

    #     # (C) REG partial이면 해당 fragment store skip -> 저장량 감소
    #     #     (여러 reg 축이 있으면 "곱"으로 근사. 과도하면 min으로 바꿔도 됨)
    #     reg_valid = 1.0
    #     for b in reg_bases_in_out:
    #         if b in penalty_by_base:
    #             p = float(penalty_by_base[b][tb])
    #             # penalty=1이면 full, penalty=2이면 완전 partial -> valid를 0.5로 근사
    #             # split-aware penalty(1~2)면 valid = 2 - p (p=1->1, p=2->0)
    #             valid = max(0.0, min(1.0, 2.0 - p))
    #             reg_valid *= valid

    #     # store는 보통 TB당 1번(내부 K-loop와 무관)
    #     store_cost_tb_d2[tb] = store_base_tx_per_tb * store_pen * repeat_store * frag_factor * reg_valid

    # cost_store_output_d2 = float(store_cost_tb_d2.sum())

    # --- TB별 store cost (vectorized) ---

    # (A) store_pen_vec: TB별 max penalty (floor=1.5)
    store_pen_vec = np.full(num_TBs, 1.5, dtype=np.float64)
    for b in store_bases_for_pen:
        arr = penalty_by_base.get(b, None)
        if arr is None:
            continue
        store_pen_vec = np.maximum(store_pen_vec, np.asarray(arr, dtype=np.float64))

    # (B) frag_factor_vec: frag_x 또는 frag_y가 partial(>1)면 2배
    frag_partial = np.zeros(num_TBs, dtype=bool)

    px = penalty_by_base.get(frag_x_b, None)
    if px is not None:
        frag_partial |= (np.asarray(px, dtype=np.float64) > 1.0)

    py = penalty_by_base.get(frag_y_b, None)
    if py is not None:
        frag_partial |= (np.asarray(py, dtype=np.float64) > 1.0)

    frag_factor_vec = np.where(frag_partial, 2.0, 1.0).astype(np.float64)

    # (C) reg_valid_vec: ∏ clamp(2 - p, 0..1)
    reg_valid_vec = np.ones(num_TBs, dtype=np.float64)
    for b in reg_bases_in_out:
        arr = penalty_by_base.get(b, None)
        if arr is None:
            continue
        p = np.asarray(arr, dtype=np.float64)
        valid = np.clip(2.0 - p, 0.0, 1.0)
        reg_valid_vec *= valid

    # 최종 store cost TB별 벡터
    store_cost_tb_d2 = (
        float(store_base_tx_per_tb)
        * store_pen_vec
        * float(repeat_store)
        * frag_factor_vec
        * reg_valid_vec
    ).astype(np.float64)

    cost_store_output_d2 = float(store_cost_tb_d2.sum())

    #
    # tx = (mean_t2_per_loop_tx + mean_v2_per_loop_tx) / 2.0
    tx = float((t2_tx_cost_tb_per_loop + v2_tx_cost_tb_per_loop).sum())
    each_config.partial_transaction_per_loop = tx
    mma = each_config.flops_per_loop * each_config.num_TBs
    stg = each_config.stage
    stg_eff = max(1, min(stg, 5))
    stage_scale = np.sqrt(stg_eff)

    # time proxy
    mma_lat_cycles  = 16.0
    mem_line_cycles = 350.0
    T_mem = tx * float(mem_line_cycles)
    T_comp = mma * float(mma_lat_cycles)

    # stage overlap 반영
    T_mem_eff = T_mem / float(stage_scale)

    # clip/floor
    exposed_floor   = 0.01
    exposed_ceiling = 1.0
    exposed = T_mem_eff / (T_mem_eff + T_comp)
    exposed = max(exposed_floor, min(exposed_ceiling, exposed))
    
    # total cost 업데이트 (store 바뀌었으니 갱신)
    each_config.partial_overlap_frac = 1.0 - exposed
    each_config.t2_total_tx_cost = t2_total_tx_cost
    each_config.v2_total_tx_cost = v2_total_tx_cost
    each_config.partial_total_cost_d2 = total_tx_cost + cost_store_output_d2
    each_config.partial_total_cost_d2_overlap = math.ceil(total_tx_cost * exposed + cost_store_output_d2)

    
#
def cost_model_total(l_configurations) :
    #
    opt_print = 0

    #
    if opt_print == 1 :
        print ("========================= [cost_model][total_cost] =========================")
    
    #
    for idx, each_config in enumerate(l_configurations) :
        l_comb = tc_gen_cost_models_TBs(each_config, idx, opt_print)
        # p_any = tc_gen_cost_full_partial(each_config)
        # tc_gen_cost_models_GM(each_config, l_comb, idx, opt_print)
        # tc_gen_cost_models_GM4(each_config, l_comb, idx, opt_print)
        # tc_gen_cost_models_Kernels(each_config)
        tc_gen_cost_models_Computes(each_config)
        # tc_gen_cost_models_pipeline2(each_config, p_any)
        # tc_gen_cost_full_partial2(each_config)
        cm_v2.tc_gen_cost_model_v2(each_config, l_comb, idx)

    #
    if opt_print == 1 :
        print ("============================================================================")