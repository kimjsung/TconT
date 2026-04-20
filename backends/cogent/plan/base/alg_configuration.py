import sys
import copy
import math

import tc_helper                as tc_helper
import base.class_configuration     as config

# revised 260309
def compute_total_partial_ratio(l_representative_problem_size, each_config, shape, size_FRAG_X, size_FRAG_Y, size_REG_X, size_REG_Y) :
    #
    fragx_idx = each_config[1]
    fragy_idx = each_config[2]
    rx_idx = each_config[3]
    ry_idx = each_config[4]

    #
    if len(fragx_idx[0]) > 1 :
        tmp = size_FRAG_X * size_REG_X
        extend = tc_helper.tc_helper_find_value(l_representative_problem_size, fragx_idx[0][0])
        if extend <= tmp :
            tmp_partial = 1
        else :
            tmp_partial = math.floor(extend / tmp) / math.ceil(extend / tmp)

        partial_x = tmp_partial
        non_split_reg_x = True
    else :
        extend_fx = tc_helper.tc_helper_find_value(l_representative_problem_size, fragx_idx[0])
        if extend_fx <= size_FRAG_X :
            tmp_partial_fx = 1
        else :
            tmp_partial_fx = math.floor(extend_fx / size_FRAG_X) / math.ceil(extend_fx / size_FRAG_X)
        
        extend_rx = tc_helper.tc_helper_find_value(l_representative_problem_size, rx_idx[0])
        if extend_rx <= size_REG_X :
            tmp_partial_rx = 1
        else :
            tmp_partial_rx = math.floor(extend_rx / size_REG_X) / math.ceil(extend_rx / size_REG_X)

        partial_x = tmp_partial_fx * tmp_partial_rx

        if size_REG_X / shape[0] == 1 :
            non_split_reg_x = False
        else :
            non_split_reg_x = True

    #
    if len(fragy_idx[0]) > 1 :
        tmp = size_FRAG_Y * size_REG_Y
        extend = tc_helper.tc_helper_find_value(l_representative_problem_size, fragy_idx[0][0])
        if extend <= tmp :
            tmp_partial = 1
        else :
            tmp_partial = math.floor(extend / tmp) / math.ceil(extend / tmp)

        partial_y = tmp_partial
        non_split_reg_y = True
    else :
        extend_fy = tc_helper.tc_helper_find_value(l_representative_problem_size, fragy_idx[0])
        if extend_fy <= size_FRAG_Y :
            tmp_partial_fy = 1
        else :
            tmp_partial_fy = math.floor(extend_fy / size_FRAG_Y) / math.ceil(extend_fy / size_FRAG_Y)
        
        extend_ry = tc_helper.tc_helper_find_value(l_representative_problem_size, ry_idx[0])
        if extend_ry <= size_REG_Y:
            tmp_partial_ry = 1
        else :
            tmp_partial_ry = math.floor(extend_ry / size_REG_Y) / math.ceil(extend_ry / size_REG_Y)

        partial_y = tmp_partial_fy * tmp_partial_ry

        if size_REG_Y / shape[1] == 1 :
            non_split_reg_y = False
        else :
            non_split_reg_y = True

    #
    all_partial_ratio = partial_x * partial_y

    return all_partial_ratio, non_split_reg_x, non_split_reg_y

#
def build_configurations(each_tc, l_info_split_idx, l_representative_problem_size, index_mapping, swap_flag, opt_print, data_type):
    #
    l_configurations_class = []

    #
    if opt_print == 1:
        print("========================= [Configurations Pruning] =========================")
        print(f"tensor_contraction : {each_tc}")
        print(f"l_representative_problem_size : {l_representative_problem_size}")
        print("============================================================================")

    #
    if data_type == "DOUBLE" :
        l_tiles_FRAG_X      = [8, 16]
        l_tiles_FRAG_Y      = [8, 16]
        l_tiles_FRAG_K      = [4, 8, 16]
        l_tiles_REG_X       = [1, 2, 4, 8, 16]
        l_tiles_REG_Y       = [1, 2, 4, 8, 16]
        element_size        = 8
    else:
        l_tiles_FRAG_X      = [8, 16]
        l_tiles_FRAG_Y      = [8, 16]
        l_tiles_FRAG_K      = [4, 8, 16]
        l_tiles_REG_X       = [1, 2, 4, 8]
        l_tiles_REG_Y       = [1, 2, 4, 8]
        element_size        = 4
    
    #
    #   Pre-Processing from the Given Inputs such as a Tensor Contraction and a Representative Problem 
    #   each_tensor_contraction[0]: output-tensor name
    #   each_tensor_contraction[1]: output-tensor indices
    #   each_tensor_contraction[2]: operator (+= or -=)
    #   each_tensor_contraction[3]: internal indices
    #   each_tensor_contraction[4]: input-tensor name
    #   each_tensor_contraction[5]: input-tensor indices
    #   each_tensor_contraction[6]: input-tensor name
    #   each_tensor_contraction[7]: input-tensor indices
    #
    l_output_tensor      = each_tc[1]
    l_internal_indices   = each_tc[3]
    l_input_tensor_left  = each_tc[5]
    l_input_tensor_right = each_tc[7]

    #
    max_SMEM_per_block = (49152 - 128) * 0.85

    #
    if opt_print == 1 :
        print("============================ [Enumerations-ALL] ============================")
        print(f" List of |FRAG_K|             : {l_tiles_FRAG_K}")
        print(f" List of |FRAG_X| or |FRAG_Y| : {l_tiles_FRAG_X}")
        print(f" List of |FRAG_X| or |FRAG_Y| : {l_tiles_FRAG_X}")
        print(f" List of |REG_X| or |REG_Y|   : {l_tiles_REG_X}")
        print(f" MAX Shared Memory            : 49152 (bytes)")
        print(f" Given Tensor Contraction     : {each_tc}")
        print(f" > Output Tensor              : {l_output_tensor}")
        print(f" > Input Tensor (LEFT)        : {l_input_tensor_left}")
        print(f" > Input Tensor (RIGHT)       : {l_input_tensor_right}")
        print(f" > Internal Indices           : {l_internal_indices}")
        print("============================================================================")

    #
    #   Assumption: Indices in an input tensor will be mapped on one of x-axis and y-axis exclusively.
    #               This is just for "a single tensor contraction."
    #
    #   [0] One of Input Tensors whose one of indices is the FVI in the output tensor will be mapped on x-axis.
    #
    for each_idx in l_input_tensor_left :
        if each_idx == l_output_tensor[0] :
            opt_swap = 1

    for each_idx in l_input_tensor_right :
        if each_idx == l_output_tensor[0] :
            opt_swap = 2

    #
    #   options-- prints
    #
    opt_print_K   = 0
    opt_print_E_L = 0
    opt_print_E_R = 0

    #
    #   [Assumption]
    #
    if opt_swap == 1 :
        # print("[Code Generator][config_pruning] L. Tensor has THE FVI in the Output")
        if opt_print == 1 :
            print(f" > Input Tensor (LEFT)  : {l_input_tensor_left}")
            print(f" > Input Tensor (RIGHT) : {l_input_tensor_right}")
    else :
        # print("[Code Generator][config_pruning] R. Tensor has THE FVI in the Output")
        l_input_tensor_left  = each_tc[7]
        l_input_tensor_right = each_tc[5]
        if opt_print == 1 :
            print(f" > Input Tensor (LEFT)  : {l_input_tensor_left}")
            print(f" > Input Tensor (RIGHT) : {l_input_tensor_right}")
    # print("============================================================================")

    #
    #   [Internal Indices]
    #
    list_partial_config_FRAG_K = alg_config_K(l_internal_indices, l_representative_problem_size, l_tiles_FRAG_K, opt_print_K)
    
    #   [Completed][Partial-Configurations][K] --- FRAG_K
    # print(f"[Code Generator][config_pruning] # of Configurations--- K : {len(list_partial_config_FRAG_K)}")
    if opt_print == 1 :
        print("============================================================================")
        for each_partial_config in list_partial_config_FRAG_K :
            print(f"each_partial_config_K : {each_partial_config}")
        print("============================================================================")

    #
    #   [External Indices][LEFT]
    #
    list_partial_config_LEFT_FRAG_REG = alg_config_E_L(l_input_tensor_left, l_output_tensor, l_internal_indices, l_representative_problem_size, l_tiles_FRAG_X, l_tiles_REG_X, opt_print_E_L)
    # print("============================================================================")
    # print(f"[Code Generator][config_pruning] # of Configurations--- E (LEFT) : {len(list_partial_config_LEFT_FRAG_REG)}")
    # print("============================================================================")
    if opt_print == 1 :
        print("============================================================================")
        for each_partial_config in list_partial_config_LEFT_FRAG_REG :
            print(f"each_partial_config_E_L : {each_partial_config}", file=sys.stderr)
        print("============================================================================")

    #
    #   [External Indices][RIGHT]
    #
    list_partial_config_RIHGT_FRAG_REG = alg_config_E_R(l_input_tensor_right, l_output_tensor, l_internal_indices, l_representative_problem_size, l_tiles_FRAG_Y, l_tiles_REG_Y, opt_print_E_R)

    #   [Completed][Partial-Configurations][E] --- TB && REG
    # print("============================================================================")
    # print(f"[Code Generator][config_pruning] # of Configurations--- E (RIGHT) : {len(list_partial_config_RIHGT_FRAG_REG)}")
    # print("============================================================================")
    if opt_print == 1 :
        print("============================================================================")
        for each_partial_config in list_partial_config_RIHGT_FRAG_REG :
            print(f"each_partial_config_E_R : {each_partial_config}")
        print("============================================================================")
    
    #
    #   [Total Configurations] = |K| * (|E_L| * |R_L|) * (|E_R| * |R_R|)
    #
    opt_print_drop = 0
    dropped_L      = 0
    dropped_R      = 0
    mapping_info   = []
    fvi_left  = l_input_tensor_left[0]
    fvi_right = l_input_tensor_right[0]
    
    #
    for each_config_K in list_partial_config_FRAG_K :
        #
        tmp_FRAG_K            = each_config_K[1]
        tmp_FRAG_K_tile_sizes = each_config_K[2]

        #
        if (fvi_left in tmp_FRAG_K) and (fvi_right not in tmp_FRAG_K) :
            if tc_helper.tc_helper_find_value(tmp_FRAG_K_tile_sizes, fvi_left) <= 4 :
                continue
        #
        elif (fvi_left not in tmp_FRAG_K) and (fvi_right in tmp_FRAG_K) :
            if tc_helper.tc_helper_find_value(tmp_FRAG_K_tile_sizes, fvi_right) <= 4 :
                continue
        # revised 260309
        elif (fvi_left in tmp_FRAG_K) and (fvi_right in tmp_FRAG_K) :
            if (tc_helper.tc_helper_find_value(tmp_FRAG_K_tile_sizes, fvi_left) <= 4) and (tc_helper.tc_helper_find_value(tmp_FRAG_K_tile_sizes, fvi_right) == 1) :
                continue
            elif (tc_helper.tc_helper_find_value(tmp_FRAG_K_tile_sizes, fvi_left) == 1) and (tc_helper.tc_helper_find_value(tmp_FRAG_K_tile_sizes, fvi_right) <= 4) :
                continue

        #
        for each_config_L in list_partial_config_LEFT_FRAG_REG :
            #
            tmp_FRAG_X       = each_config_L[0]
            tmp_REG_X        = each_config_L[1]
            tmp_X_tile_sizes = each_config_L[2]
            
            #
            if len(l_info_split_idx) != 0 :
                if (l_info_split_idx[0] != []) and (len(tmp_FRAG_X) == 1) and (len(tmp_REG_X) == 1) and (len(tmp_X_tile_sizes) == 2) :
                    #
                    if (tmp_FRAG_X[0] in l_info_split_idx[0]) and (tmp_REG_X[0] in l_info_split_idx[0]) :
                        #
                        tile_size = tmp_X_tile_sizes[0][1] * tmp_X_tile_sizes[1][1]
                        index_size = tc_helper.tc_helper_find_value(l_representative_problem_size, l_info_split_idx[0][0])
                        
                        #
                        if 2.0 * index_size < tile_size :
                            dropped_L += 1
                            continue

            #
            if tmp_FRAG_X[0].endswith("2") :
                #
                dropped_L += 1
                
                #
                if opt_print_drop == 1 :
                    print(f"Dropped Base config_L : {each_config_L}")
                continue

            #
            for each_config_R in list_partial_config_RIHGT_FRAG_REG :
                #
                tmp_FRAG_Y       = each_config_R[0]
                tmp_REG_Y        = each_config_R[1][0]
                tmp_Y_tile_sizes = each_config_R[1][1]

                #
                if len(l_info_split_idx) != 0 :
                    if (l_info_split_idx[1] != []) and (len(tmp_FRAG_Y) == 1) and (len(tmp_REG_Y) == 1) and (len(tmp_Y_tile_sizes) == 2) :
                        #
                        if (tmp_FRAG_Y[0] in l_info_split_idx[1]) and (tmp_REG_Y[0] in l_info_split_idx[1]) :
                            #
                            tile_size = tmp_Y_tile_sizes[0][1] * tmp_Y_tile_sizes[1][1]
                            index_size = tc_helper.tc_helper_find_value(l_representative_problem_size, l_info_split_idx[1][0])

                            #
                            if 2.0 * index_size < tile_size :
                                dropped_R += 1
                                continue
                
                #
                if tmp_FRAG_Y[0].endswith("2") :
                    #
                    dropped_R += 1
                    
                    #
                    if opt_print_drop == 1 :
                        print(f"Dropped Base config_R : {each_config_R}")
                    continue
                
                #
                tmp_combined_tile_sizes = []
                for each_tile in tmp_FRAG_K_tile_sizes :
                    tmp_combined_tile_sizes.append(each_tile)
                
                for each_tile in tmp_X_tile_sizes :
                    tmp_combined_tile_sizes.append(each_tile)
                
                for each_tile in tmp_Y_tile_sizes :
                    tmp_combined_tile_sizes.append(each_tile)

                #
                mapping_info.append([tmp_FRAG_K, tmp_FRAG_X, tmp_FRAG_Y, tmp_REG_X, tmp_REG_Y, tmp_combined_tile_sizes])

    #
    flag = 0
    size_M = 1
    for each_index in l_input_tensor_left :
        if each_index not in l_internal_indices :
            if len(l_info_split_idx) > 0 :
                if each_index in l_info_split_idx[0] :
                    if flag == 1 :
                        continue
                    flag = 1
            size_M *= tc_helper.tc_helper_find_value(l_representative_problem_size, each_index)
    
    #
    flag = 0
    size_N = 1
    for each_index in l_input_tensor_right :
        if each_index not in l_internal_indices :
            if len(l_info_split_idx) > 0 :
                if each_index in l_info_split_idx[1] :
                    if flag == 1 :
                        continue
                    flag = 1
            size_N *= tc_helper.tc_helper_find_value(l_representative_problem_size, each_index)

    #
    size_K = 1
    for each_index in l_internal_indices :
        size_K *= tc_helper.tc_helper_find_value(l_representative_problem_size, each_index)

    #
    ori_idx = 0
    d1 = 0
    d2 = 0
    d3 = 0
    d4 = 0
    d5 = 0
    # print(f"swap_flag : {swap_flag}", file=sys.stderr)
    for each_config in mapping_info :
        # print(f"each_config : {each_config}", file=sys.stderr)
        # print(f"index_mapping : {index_mapping}", file=sys.stderr)
        if each_config[0] != index_mapping[0] :
            # print(f"1) each_config : {each_config}", file=sys.stderr)
            continue
        if sorted(each_config[1]) != sorted(index_mapping[1]) :
            # print(f"2) each_config : {each_config}", file=sys.stderr)
            continue 
        if each_config[3] != index_mapping[3] :
            # print(f"3) each_config : {each_config}", file=sys.stderr)
            continue
        if swap_flag == 0 :
            if each_config[2] != index_mapping[2] :
                # print(f"4-1) each_config : {each_config}", file=sys.stderr)
                continue
            
            if each_config[4] != index_mapping[4] :
                # print(f"5-1) each_config : {each_config}", file=sys.stderr)
                continue
        else :
            if each_config[2] != (index_mapping[4] + index_mapping[2][1:]) :
                # print(f"4-2) each_config : {each_config[2]}, index_mapping : {index_mapping[4]}\n", file=sys.stderr)
                continue
            
            if each_config[4] != [index_mapping[2][0]] :
                # print(f"5-2) each_config : {each_config[4]}, index_mapping : {index_mapping[2]}", file=sys.stderr)
                continue
        #
        if tc_helper.tc_helper_find_value(each_config[5], fvi_left) < 4 and (fvi_left not in each_config[0]) :
            d1 += 1
            # print(f"6) each_config : {each_config}", file=sys.stderr)
            continue

        #
        if tc_helper.tc_helper_find_value(each_config[5], fvi_right) < 4 and (fvi_right not in each_config[0]) :
            d2 += 1
            # print(f"7) each_config : {each_config}", file=sys.stderr)
            continue

        #
        size_FRAG_K = 1
        for each_idx in each_config[0] :
            size_FRAG_K *= tc_helper.tc_helper_find_value(each_config[5], each_idx)

        #
        size_FRAG_X = 1
        for each_idx in each_config[1] :
            size_FRAG_X *= tc_helper.tc_helper_find_value(each_config[5], each_idx)

        #
        size_FRAG_Y = 1
        for each_idx in each_config[2] :
            size_FRAG_Y *= tc_helper.tc_helper_find_value(each_config[5], each_idx)
        
        #
        size_REG_X = 1
        for each_idx in each_config[3] :
            size_REG_X *= tc_helper.tc_helper_find_value(each_config[5], each_idx)

        #
        size_REG_Y = 1
        for each_idx in each_config[4] :
            size_REG_Y *= tc_helper.tc_helper_find_value(each_config[5], each_idx)

        #
        m_tile_cnt = size_M / (size_FRAG_X * size_REG_X)
        n_tile_cnt = size_N / (size_FRAG_Y * size_REG_Y)
        if m_tile_cnt >= n_tile_cnt :
            mn_tile_cnt_ratio = m_tile_cnt / n_tile_cnt
        else :
            mn_tile_cnt_ratio = n_tile_cnt / m_tile_cnt

        #
        tmp_pairs = []
        warp_cnt = (size_FRAG_X * size_FRAG_Y) // 32
        for x in range(1, int(warp_cnt**0.5) + 1) :
            if warp_cnt % x == 0 :
                y = int(warp_cnt // x)
                tmp_pairs.append([x, y])
                if x != y :
                    tmp_pairs.append([y, x])
        
        # revised 260309
        if (warp_cnt == 2) and ((size_FRAG_X * size_REG_X == 128) or (size_FRAG_Y * size_REG_Y == 128)) :
            # print(f"8) each_config : {each_config}", file=sys.stderr)
            continue
        
        #
        warp_shape = []
        for pair in tmp_pairs :
            if size_REG_X % pair[0] != 0 or size_REG_Y % pair[1] != 0 :
                d3 += 1
                # print(f"9) each_config : {each_config}, pair : {tmp_pairs}", file=sys.stderr)
                continue
            else :
                warp_shape.append(pair)
        
        #
        if not warp_shape :
            d4 += 1
            # print(f"10) each_config : {each_config}", file=sys.stderr)
            continue
        
        #
        size_SMEM_L = 1
        for each_idx in l_input_tensor_left:
            if tc_helper.tc_helper_find_value(each_config[5], each_idx) != -1 :
                size_SMEM_L *= tc_helper.tc_helper_find_value(each_config[5], each_idx)

        #
        size_SMEM_R = 1
        for each_idx in l_input_tensor_right:
            if tc_helper.tc_helper_find_value(each_config[5], each_idx) != -1 :
                size_SMEM_R *= tc_helper.tc_helper_find_value(each_config[5], each_idx)

        #
        if max_SMEM_per_block <= (element_size * 1 * (size_SMEM_L + size_SMEM_R)) :
            d5 += 1
            # print(f"11) each_config : {each_config}", file=sys.stderr)
            continue
        
        #
        stages    = []
        tmp_stage = 1
        k_loops   = math.ceil(tc_helper.tc_helper_find_value(l_representative_problem_size, each_config[0][0]) / size_FRAG_K)
    
        while (element_size * tmp_stage * (size_SMEM_L + size_SMEM_R) < max_SMEM_per_block) and (tmp_stage <= 5) and (tmp_stage <= k_loops) :
            stages.append(tmp_stage)
            tmp_stage += 1

        #
        tile_ratio = (size_FRAG_X * size_REG_X) / (size_FRAG_Y * size_REG_Y)
        
        #
        # 0 : double / 1 : double2
        #
        opt_print_d2 = 0
        t = 0
        for stage in stages :
            for shape in warp_shape :
                #
                double2_left_flag, double2_right_flag = determine_double2(l_representative_problem_size, each_config[5], fvi_left, fvi_right, size_REG_X, size_REG_Y, shape, opt_print_d2)

                #
                if (fvi_left in each_config[3]) and (tc_helper.tc_helper_find_value(each_config[5], fvi_left) == 16) and (double2_left_flag == 0) :
                    # print(f"12) each_config : {each_config}", file=sys.stderr)
                    continue
                
                #
                if (fvi_right in each_config[4]) and (tc_helper.tc_helper_find_value(each_config[5], fvi_right) == 16) and (double2_right_flag == 0) :
                    # print(f"13) each_config : {each_config}", file=sys.stderr)
                    continue
                
                #
                if (fvi_left in each_config[3]) and (tc_helper.tc_helper_find_value(each_config[5], fvi_left) < 8) and (double2_left_flag == 1) :
                    # print(f"14) each_config : {each_config}", file=sys.stderr)
                    # continue
                    double2_left_flag = 0
                
                #
                if (fvi_right in each_config[4]) and (tc_helper.tc_helper_find_value(each_config[5], fvi_right) < 8) and (double2_right_flag == 1) :
                    # print(f"15) each_config : {each_config}", file=sys.stderr)
                    # continue
                    double2_right_flag = 0
                
                #
                num_frag_regs = 2 * ((size_FRAG_X // 8) * (size_FRAG_Y // 8) * (size_REG_X // shape[0]) * (size_REG_Y // shape[1]))
                if num_frag_regs > 64  : #or num_frag_regs < 4 :
                    # print(f"16) each_config : {each_config}, num_frag_regs : {num_frag_regs}", file=sys.stderr)
                    continue
                
                # revised 260309
                if (size_REG_X != 1) and (size_REG_Y != 1) : 
                    if ((size_REG_X // shape[0]) == 1) and ((size_REG_Y // shape[1]) == 1) :
                        # print(f"17) each_config : {each_config}", file=sys.stderr)
                        continue
                
                # revised 260309
                tile_m_size = size_FRAG_Y * size_REG_Y
                tile_n_size = size_FRAG_X * size_REG_X
                
                if tile_m_size > tile_n_size :
                    if shape[1] < shape[0] :
                        # print(f"18) each_config : {each_config}", file=sys.stderr)
                        continue
                elif tile_m_size < tile_n_size :
                    if shape[0] < shape[1] :
                        # print(f"19) each_config : {each_config}", file=sys.stderr)
                        continue
                
                # revised 260309
                all_partial_ratio, non_split_reg_x, non_split_reg_y = compute_total_partial_ratio(l_representative_problem_size, each_config, shape, size_FRAG_X, size_FRAG_Y, size_REG_X, size_REG_Y)
                # if all_partial_ratio < 0.5 :
                #     continue
                if not (non_split_reg_x and non_split_reg_y) :
                    # print(f"20) each_config : {each_config}", file=sys.stderr)
                    continue

                #
                frag_ratio = ((size_FRAG_X / 8) * (size_REG_X / shape[0])) / ((size_FRAG_Y / 8) * (size_REG_Y / shape[1]))

                #
                for producer_cnt in range(0, 1) :
                    #
                    tmp_config = config.Config()
                    
                    #
                    tmp_config.add_tensor_C(l_output_tensor)
                    tmp_config.add_tensor_A(l_input_tensor_left)
                    tmp_config.add_tensor_B(l_input_tensor_right)
                    
                    #
                    tmp_config.add_MNK(size_M, size_N, size_K)

                    #
                    tmp_config.add_FRAG_K(each_config[0])
                    tmp_config.add_FRAG_X(each_config[1])
                    tmp_config.add_FRAG_Y(each_config[2])
                    tmp_config.add_REG_X(each_config[3])
                    tmp_config.add_REG_Y(each_config[4])
                    
                    #
                    tmp_config.add_tile_size(each_config[5])

                    #
                    tmp_config.add_split_index(l_info_split_idx)
                    
                    #
                    tmp_config.add_representative_problem_size(l_representative_problem_size)
                    
                    #
                    tmp_config.add_WARP_SHAPE(shape)

                    #
                    tmp_config.add_STAGE(stage)
                    tmp_config.add_PRODUCER_CNT(producer_cnt)
                    tmp_config.add_double2(double2_left_flag, double2_right_flag)
                    
                    #
                    tmp_config.idx = ori_idx
                    
                    #
                    tmp_config.size_FRAG_K = size_FRAG_K
                    tmp_config.size_FRAG_X = size_FRAG_X
                    tmp_config.size_FRAG_Y = size_FRAG_Y
                    tmp_config.size_REG_X = size_REG_X
                    tmp_config.size_REG_Y = size_REG_Y

                    #
                    tmp_config.num_Frag_Regs = num_frag_regs
                    
                    #
                    tmp_config.mn_ratio = mn_tile_cnt_ratio
                    tmp_config.frag_ratio = frag_ratio
                    tmp_config.tile_ratio = tile_ratio
                    
                    #
                    l_configurations_class.append(tmp_config)

                    #
                    ori_idx += 1
                t += 1
    
    #
    # print(f"[Code Generator][Configurations] # of Configurations --- Dropped (E_L + E_R) : {dropped_L + dropped_R} ({dropped_L} + {dropped_R})")
    # print(f"[Code Generator][Configurations] # of Configurations --- Base : {len(mapping_info)}")
    # print(f"[Code Generator][Configurations] # of Configurations --- Full : {len(l_configurations_class)}", file=sys.stderr)
    
    #
    return l_configurations_class

#
def determine_double2(l_representative_problem_size, l_tile_sizes, fvi_left, fvi_right, size_REG_X, size_REG_Y, shape, opt_print) :
    #
    double2_left_flag = 1

    #
    if tc_helper.tc_helper_find_value(l_representative_problem_size, fvi_left) % 2 != 0 :
        if opt_print :
            print(f"[Code Generator][determine_double2] FVI in LEFT has an odd size : {fvi_left} -> {tc_helper.tc_helper_find_value(l_representative_problem_size, fvi_left)}")
        double2_left_flag &= 0

    #
    if int(size_REG_X / shape[0]) < 2 :
        if opt_print :
            print(f"[Code Generator][determine_double2] REG_X per warp in LEFT is less than 2 : {size_REG_X} / {shape[0]} = {int(size_REG_X / shape[0])}")
        double2_left_flag &= 0

    #
    if tc_helper.tc_helper_find_value(l_tile_sizes, fvi_left) % 2 != 0 :
        if opt_print :
            print(f"[Code Generator][determine_double2] Tile size of FVI in LEFT is odd : {fvi_left} -> {tc_helper.tc_helper_find_value(l_tile_sizes, fvi_left)}")
        double2_left_flag &= 0

    #
    double2_right_flag = 1

    #
    if tc_helper.tc_helper_find_value(l_representative_problem_size, fvi_right) % 2 != 0 :
        if opt_print :
            print(f"[Code Generator][determine_double2] FVI in RIGHT has an odd size : {fvi_right} -> {tc_helper.tc_helper_find_value(l_representative_problem_size, fvi_right)}")
        double2_right_flag &= 0

    #
    if int(size_REG_Y / shape[1]) < 2 :
        if opt_print :
            print(f"[Code Generator][determine_double2] REG_Y per warp in RIGHT is less than 2 : {size_REG_Y} / {shape[1]} = {int(size_REG_Y / shape[1])}")
        double2_right_flag &= 0

    #
    if tc_helper.tc_helper_find_value(l_tile_sizes, fvi_right) % 2 != 0 :
        if opt_print :
            print(f"[Code Generator][determine_double2] Tile size of FVI in RIGHT is odd : {fvi_right} -> {tc_helper.tc_helper_find_value(l_tile_sizes, fvi_right)}")
        double2_right_flag &= 0
    
    # to do
    # fvi가 frag, internal일 때, partial일 때 못쓰는거 -> compute에서 0 처리가 가능하면 괜찮음

    return double2_left_flag, double2_right_flag

#
#   [Configuration][Algorithm][Right][Thread-Block] --- "FRAG_X" && REG_X
#
def alg_config_E_R(l_input_tensor, l_output_tensor, l_internal_indices, l_representative_problem_size, l_tiles_FRAG_X, l_tile_REG, opt_print):
    #
    l_partial_config_E = []

    #
    if opt_print == 1 :
        print ("============================================================================")
        print(f"[Code Generator][config_pruning][alg_E_R] l_input_tensor : {l_input_tensor}")

    #
    for each_idx in l_input_tensor :
        if each_idx == l_output_tensor[0] :
            print("[Code Generator][config_pruning][alg_E_R] ERROR : Given Right Tensor has the Output's FVI")
            sys.exit()
    
    # Per Each |FRAG| Size,
    for each_size_FRAG in l_tiles_FRAG_X :
        if opt_print == 1 :
            print("============================================================================")
            print(f"[Code Generator][config_pruning][alg_E_R] |FRAG| = {each_size_FRAG}")

        #
        for start_idx in range(0, len(l_input_tensor)) :
            #
            if tc_helper.tc_helper_find_index(l_internal_indices, l_input_tensor[start_idx]) != -1 :
                continue
            
            #
            vol_FRAG             = 1
            vol_FRAG_prev        = 1
            l_FRAG               = []
            list_temp_tile_sizes = []
            opt_done             = -1

            #
            for target_idx in range(start_idx, len(l_input_tensor)) :
                #
                if tc_helper.tc_helper_find_index(l_internal_indices, l_input_tensor[target_idx]) != -1 :
                    continue

                # #
                # vol_FRAG *= tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])
                # #   |FRAG'| >= |FRAG|
                # if vol_FRAG >= each_size_FRAG :
                #     #   |FRAG'| > |FRAG|
                #     if vol_FRAG > each_size_FRAG :
                #         #
                #         blocking_tile_size = int(each_size_FRAG / vol_FRAG_prev)
                #         l_FRAG.append(l_input_tensor[target_idx])
                #         list_temp_tile_sizes.append([l_input_tensor[target_idx], blocking_tile_size])
                #     #   |FRAG'| = |FRAG|
                #     else :
                #         l_FRAG.append(l_input_tensor[target_idx])
                #         list_temp_tile_sizes.append([l_input_tensor[target_idx], tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])])
                #     #
                #     opt_done = 1
                #     break
                # #   |FRAG'| < |FRAG|
                # else :
                #     l_FRAG.append(l_input_tensor[target_idx])
                #     list_temp_tile_sizes.append([l_input_tensor[target_idx], tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])])

                # #
                # vol_FRAG_prev *= tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])

                # revised 260309
                blocking_tile_size = int(each_size_FRAG)
                l_FRAG.append(l_input_tensor[target_idx])
                list_temp_tile_sizes.append([l_input_tensor[target_idx], blocking_tile_size])
                opt_done = 1
                break
            
            #
            if opt_done == -1 :
                for target_idx in range(0, start_idx) :
                    #
                    if tc_helper.tc_helper_find_index(l_internal_indices, l_input_tensor[target_idx]) != -1 :
                        continue

                    #
                    vol_FRAG *= tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])

                    #   |FRAG'| >= |FRAG|
                    if vol_FRAG >= each_size_FRAG :
                        #   |FRAG'| > |FRAG|
                        if vol_FRAG > each_size_FRAG :
                            #
                            blocking_tile_size = int(each_size_FRAG / vol_FRAG_prev)
                            l_FRAG.append(l_input_tensor[target_idx])
                            list_temp_tile_sizes.append([l_input_tensor[target_idx], blocking_tile_size])
                        #   |FRAG'| = |FRAG|
                        else :
                            l_FRAG.append(l_input_tensor[target_idx])
                            list_temp_tile_sizes.append([l_input_tensor[target_idx], tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])])
                        
                        #
                        opt_done = 1
                        break
                    #   |FRAG'| < |FRAG|
                    else :
                        l_FRAG.append(l_input_tensor[target_idx])
                        list_temp_tile_sizes.append([l_input_tensor[target_idx], tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])])
                    
                    #
                    vol_FRAG_prev *= tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])
            #      
            if opt_done == 1 :
                # Mapping REG
                l_partial_config_E_R = alg_config_E_R_R(l_input_tensor, l_internal_indices, l_representative_problem_size, l_tile_REG, l_FRAG, list_temp_tile_sizes, opt_print)

                # Check unmapped external indices
                if len(l_partial_config_E_R) > 0 :
                    # Per Each Config_E_R
                    for each_config_E_R in l_partial_config_E_R :
                        #
                        if opt_print == 1 :
                            print(f"[3] each_config_E_R : {each_config_E_R}")

                        # Check if there exists unmapped indices or not.
                        list_FRAG_copied = copy.deepcopy(l_FRAG)
                        for each_idx in l_input_tensor :
                            #
                            if tc_helper.tc_helper_find_index(l_FRAG, each_idx) != -1 :
                                continue
                            if tc_helper.tc_helper_find_index(l_internal_indices, each_idx) != -1 :
                                continue
                            if tc_helper.tc_helper_find_index(each_config_E_R[0], each_idx) != -1 :
                                continue
                            
                            #
                            list_FRAG_copied.append(each_idx)
                            each_config_E_R[1].append([each_idx, 1])
                        #
                        l_partial_config_E.append([list_FRAG_copied, each_config_E_R])

    #
    return l_partial_config_E

#
#   [Configuration][Algorithm][Right][Register] --- FRAG_X && "REG_X"
#
def alg_config_E_R_R(l_input_tensor, l_internal_indices, l_representative_problem_size, l_tile_REG, l_base_FRAG, l_base_tile_sizezs, opt_print) :
    #
    l_partial_config_E_R = []
    
    #
    if opt_print == 1 :
        print("[Code Generator][config_pruning][alg_E_R][alg_E_R_R]=================================")
        print(f"Input Tensor        : {l_input_tensor}")
        print(f"Representative Size : {l_representative_problem_size}")
        print(f"l_FRAG              : {l_base_FRAG}, {l_base_tile_sizezs}")
        print("============================================================================")

    #
    for each_size_REG in l_tile_REG :
        #
        opt_fvi = -1

        #
        if opt_print == 1 :
            print(f"|REG| = {each_size_REG}")
        
        #
        for start_idx in range(0, len(l_input_tensor)) :
            #
            if opt_print == 1 :
                print(f"[REG] start_idx : {start_idx}, {l_input_tensor[start_idx]}")
            
            #
            if tc_helper.tc_helper_find_index(l_base_FRAG, l_input_tensor[start_idx]) != -1 :
                continue

            #
            if tc_helper.tc_helper_find_index(l_internal_indices, l_input_tensor[start_idx]) != -1 :
                continue
            
            #

            #
            vol_REG                  = 1
            vol_REG_prev             = 1
            l_REG                    = []
            l_inherited_FRAG         = copy.deepcopy(l_base_FRAG)
            l_inherited_tile_sizes   = copy.deepcopy(l_base_tile_sizezs)
            opt_done                 = -1

            #
            for target_idx in range(start_idx, len(l_input_tensor)) :
                #
                if tc_helper.tc_helper_find_index(l_internal_indices, l_input_tensor[target_idx]) != -1 :
                    continue
                #
                if tc_helper.tc_helper_find_index(l_base_FRAG, l_input_tensor[target_idx]) != -1 :
                    continue

                # #
                # vol_REG *= tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])

                # #   |REG'| >= |REG|
                # if vol_REG >= each_size_REG :
                #     #   |REG'| > |REG|
                #     if vol_REG > each_size_REG :
                #         blocking_tile_size = int(each_size_REG / vol_REG_prev)
                #         l_REG.append(l_input_tensor[target_idx])
                #         l_inherited_tile_sizes.append([l_input_tensor[target_idx], blocking_tile_size])
                #     #   |REG'| = |REG|
                #     else :
                #         l_REG.append(l_input_tensor[target_idx])
                #         l_inherited_tile_sizes.append([l_input_tensor[target_idx], tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])])
                #     #
                #     opt_done = 1
                #     break
                # #   |REG'| < |REG|
                # else :
                #     l_REG.append(l_input_tensor[target_idx])
                #     l_inherited_tile_sizes.append([l_input_tensor[target_idx], tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])])

                # revised 260309
                blocking_tile_size = int(each_size_REG)
                l_REG.append(l_input_tensor[target_idx])
                l_inherited_tile_sizes.append([l_input_tensor[target_idx], blocking_tile_size])
                opt_done = 1
                break

            #
            if opt_done == -1 :
                for target_idx in range(0, start_idx):
                    #
                    if tc_helper.tc_helper_find_index(l_internal_indices, l_input_tensor[target_idx]) != -1 :
                        continue
                    #
                    if tc_helper.tc_helper_find_index(l_base_FRAG, l_input_tensor[target_idx]) != -1 :
                        continue
                    #
                    vol_REG *= tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])

                    #   |REG'| >= |REG|
                    if vol_REG >= each_size_REG :
                        #   |REG'| > |REG|
                        if vol_REG > each_size_REG:
                            blocking_tile_size = int(each_size_REG / vol_REG_prev)
                            l_REG.append(l_input_tensor[target_idx])
                            l_inherited_tile_sizes.append([l_input_tensor[target_idx], blocking_tile_size])
                        #   |REG'| = |REG|
                        else :
                            l_REG.append(l_input_tensor[target_idx])
                            l_inherited_tile_sizes.append([l_input_tensor[target_idx], tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])])
                        #
                        opt_done = 1
                        break
                    #   |REG'| < |REG|
                    else :
                        l_REG.append(l_input_tensor[target_idx])
                        l_inherited_tile_sizes.append([l_input_tensor[target_idx], tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])])

            #
            if opt_done == 1 :
                opt_fvi = 1
                if opt_print == 1 :
                    print(f"l_REG         : {l_REG}")
                    print(f"l_tile_sizezs : {l_inherited_tile_sizes}")
                l_partial_config_E_R.append([l_REG, l_inherited_tile_sizes])
    
        #
        # if opt_fvi == -1 :
        #     #
        #     opt_double_check = -1
        #     for each_idx in l_input_tensor:
        #         if tc_helper.tc_helper_find_index(l_internal_indices, each_idx) != -1 :
        #             continue
        #         if tc_helper.tc_helper_find_index(l_base_FRAG, each_idx) != -1 :
        #             continue
        #         #
        #         opt_double_check = 1
            
        #     if opt_double_check == 1 :
        #         l_inherited_tile_sizes = copy.deepcopy(l_base_tile_sizezs)
        #         l_inherited_tile_sizes.append([l_input_tensor[0], each_size_REG])
        #         l_partial_config_E_R.append([[l_input_tensor[0]], l_inherited_tile_sizes])
    
    #
    return l_partial_config_E_R

#
#   [Configuration][Algorithm][LEFT][Thread-Block] --- "FRAG_Y" && REG_Y
#
def alg_config_E_L(l_input_tensor, l_output_tensor, l_internal_indices, l_representative_problem_size, l_tiles_FRAG_X, l_tile_REG, opt_print) :
    #
    opt_print_E_L_R = 0
    l_partial_config_E = []

    #
    if opt_print == 1 :
        print("============================================================================")
        print(f"[Code Generator][config_pruning][alg_E_L] l_input_tensor : {l_input_tensor}")

    #
    opt_has_output_fvi = -1
    for each_idx in l_input_tensor :
        if each_idx == l_output_tensor[0] :
            opt_has_output_fvi = 1

    #
    if opt_has_output_fvi == -1 :
        print("[Code Generator][config_pruning][alg_E_L] ERROR!")

    # Per Each |FRAG| Size,
    for each_size_FRAG in l_tiles_FRAG_X :
        if opt_print == 1 :
            print("============================================================================")
            print(f"[Code Generator][config_pruning][alg_E_L] |FRAG| : {each_size_FRAG}")
        
        #
        default_vol_FRAG             = 1
        default_list_FRAG            = []
        default_list_temp_tile_sizes = []
        default_opt_done             = -1
        
        #
        default_vol_FRAG *= tc_helper.tc_helper_find_value(l_representative_problem_size, l_output_tensor[0]) # l_output_tensor[0] = fvi
        default_list_FRAG.append(l_output_tensor[0])
        
        # #   |FRAG_X'| >= |FRAG_X| : Compare representative size of fvi and tile size of FRAG_X 
        # if default_vol_FRAG >= each_size_FRAG :
        #     #   |FRAG_X'| > |FRAG_X|
        #     if default_vol_FRAG > each_size_FRAG :
        #         blocking_tile_size = int(each_size_FRAG)
        #         default_list_temp_tile_sizes.append([l_output_tensor[0], blocking_tile_size])
        #     #   |FRAG_X'| = |FRAG_X|
        #     else :
        #         default_list_temp_tile_sizes.append([l_output_tensor[0], tc_helper.tc_helper_find_value(l_representative_problem_size, l_output_tensor[0])])
        #     #
        #     default_opt_done = 1
        # #   |FRAG_X'| < |FRAG_X|
        # else :
        #     default_list_temp_tile_sizes.append([l_output_tensor[0], tc_helper.tc_helper_find_value(l_representative_problem_size, l_output_tensor[0])])

        # revised 260309
        blocking_tile_size = int(each_size_FRAG)
        default_list_temp_tile_sizes.append([l_output_tensor[0], blocking_tile_size])
        default_opt_done = 1

        # [Normal Case]
        if default_opt_done == -1 :
            #
            for start_idx in range(0, len(l_input_tensor)) :
                #
                vol_FRAG             = default_vol_FRAG
                vol_FRAG_prev        = default_vol_FRAG
                l_FRAG               = copy.deepcopy(default_list_FRAG)
                list_temp_tile_sizes = copy.deepcopy(default_list_temp_tile_sizes)
                opt_done             = -1

                # [Constraint #1]
                if l_input_tensor[start_idx] == l_output_tensor[0] :
                    continue

                # [Constraint #2]
                if tc_helper.tc_helper_find_index(l_internal_indices, l_input_tensor[start_idx]) != -1 :
                    continue

                #
                for target_idx in range(start_idx, len(l_input_tensor)) :
                    # [Constraint #1] If Input[target_idx] == Output[0],
                    if l_input_tensor[target_idx] == l_output_tensor[0]:
                        continue

                    # [Constraint #2] If Intput[target_idx] == Internal Index
                    if tc_helper.tc_helper_find_index(l_internal_indices, l_input_tensor[target_idx]) != -1 :
                        continue

                    #
                    vol_FRAG *= tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])

                    #   |TB_X'| >= |TB_X|
                    if vol_FRAG >= each_size_FRAG :
                        #   |TB_X'| > |TB_X|
                        if vol_FRAG > each_size_FRAG :
                            blocking_tile_size = int(each_size_FRAG / vol_FRAG_prev)
                            l_FRAG.append(l_input_tensor[target_idx])
                            list_temp_tile_sizes.append([l_input_tensor[target_idx], blocking_tile_size])
                        #   |TB_X'| = |TB_X|
                        else:
                            l_FRAG.append(l_input_tensor[target_idx])
                            list_temp_tile_sizes.append([l_input_tensor[target_idx], tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])])
                        #
                        opt_done = 1
                        break
                    #   |TB_X'| < |TB_X|
                    else:
                        l_FRAG.append(l_input_tensor[target_idx])
                        list_temp_tile_sizes.append([l_input_tensor[target_idx], tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])])

                    #
                    vol_FRAG_prev *= tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])
                    
                #
                if opt_done == -1 :
                    for target_idx in range(0, start_idx) :
                        # [Constraint #1] If Input[target_idx] == Output[0],
                        if l_input_tensor[target_idx] == l_output_tensor[0] :
                            continue

                        # [Constraint #2] If Intput[target_idx] == Internal Index
                        if tc_helper.tc_helper_find_index(l_internal_indices, l_input_tensor[target_idx]) != -1 :
                            continue
                        
                        #
                        print(f"[2] {start_idx}, {target_idx} : {l_input_tensor[target_idx]}, opt_done : {opt_done}")
                        
                        #
                        if opt_done == -1 :
                            vol_FRAG *= tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])
                            #   |TB_X'| >= |TB_X|
                            if vol_FRAG >= each_size_FRAG :
                                print(f"[2] >> {l_input_tensor[target_idx]} is mapped")
                                #   |TB_X'| > |TB_X|
                                if vol_FRAG > each_size_FRAG :
                                    blocking_tile_size = int(each_size_FRAG / vol_FRAG_prev)
                                    opt_start_mapped = 1
                                    l_FRAG.append(l_input_tensor[target_idx])
                                    list_temp_tile_sizes.append([l_input_tensor[target_idx], blocking_tile_size])
                                #   |TB_X'| = |TB_X|
                                else :
                                    opt_start_mapped = 1
                                    l_FRAG.append(l_input_tensor[target_idx])
                                    list_temp_tile_sizes.append([l_input_tensor[target_idx], tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])])
                                
                                #
                                opt_done = 1
                                break
                            #   |TB_X'| < |TB_X|
                            else :
                                opt_start_mapped = 1
                                l_FRAG.append(l_input_tensor[target_idx])
                                list_temp_tile_sizes.append([l_input_tensor[target_idx], tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])])
                            #
                            vol_FRAG_prev *= tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])
                #
                if opt_done == 1 :
                    list_partial_config_E_L = alg_config_E_L_R(l_input_tensor, l_output_tensor, l_internal_indices, l_representative_problem_size, l_tile_REG, l_FRAG, list_temp_tile_sizes, opt_print_E_L_R)
                    for each_config_E_L in list_partial_config_E_L :
                        if opt_print == 1 :
                            print(f"[1] each_config_E_L : {each_config_E_L}")
                        l_partial_config_E.append(each_config_E_L)
                        
                #
                #   "TB" is fully mapped at the begining (special case: the output FVI)
                #
        #
        #   |TB| is fully mapped when |Special Case|
        #
        else :
            #
            if opt_print == 1 :
                print(f"[Inputs] Mapping : {default_list_FRAG}, Tile Sizes : {default_list_temp_tile_sizes}")

            #
            list_partial_config_E_L = alg_config_E_L_R(l_input_tensor, l_output_tensor, l_internal_indices, l_representative_problem_size, l_tile_REG, default_list_FRAG, default_list_temp_tile_sizes, opt_print_E_L_R)
            
            #
            for each_config_E_L in list_partial_config_E_L :
                if opt_print == 1 :
                    print(f"[2] each_config_E_L : {each_config_E_L}")
                l_partial_config_E.append(each_config_E_L)
    
    return l_partial_config_E

#
#   [Configuration][Algorithm][LEFT][Register] --- FRAG_Y && "REG_Y"
#
def alg_config_E_L_R(l_input_tensor, input_out_tensor, l_internal_indices, l_representative_problem_size, l_tile_REG, l_FRAG, l_tile_sizes, opt_print):
    #
    if opt_print == 1 :
        print("============================================================================")
        print(f"[alg_E_L_R] l_FRAG : {l_FRAG}")
        print(f"[alg_E_L_R] l_tile_sizes : {l_tile_sizes}")

    #
    l_partial_config_R = []

    # Each Tile-Size
    for each_size_REG in l_tile_REG :
        #
        if opt_print == 1 :
            print("============================================================================")
            print(f"|REG| = {each_size_REG}")

        #
        for start_idx in range(0, len(l_input_tensor)) :
            #
            vol_REG              = 1
            vol_REG_prev         = 1
            l_REG                = []
            l_inherited_FRAG     = copy.deepcopy(l_FRAG)
            list_temp_tile_sizes = copy.deepcopy(l_tile_sizes)
            opt_done             = -1

            #
            if opt_print == 1 :
                print(f"start_idx : {start_idx}, opt_done : {opt_done}")

            #
            if tc_helper.tc_helper_find_index(l_FRAG, l_input_tensor[start_idx]) != -1 : # FRAG_mapped_index == l_input_tensor[start_idx] : continue => skip already mapped index
                if opt_print == 1 :
                    print(f"{l_input_tensor[start_idx]} is already mapped on l_FRAG")
                    print("============================================================================")
                opt_done = 1
                continue
            
            #
            if tc_helper.tc_helper_find_index(l_internal_indices, l_input_tensor[start_idx]) != -1 : # internal_index == l_input_tensor[start_idx] : continue => skip already mapped index
                if opt_print == 1 :
                    print(f"{l_input_tensor[start_idx]} is internal index")
                    print("============================================================================")
                opt_done = 1
                continue

            #
            if opt_print == 1 :
                print(f"(pruned) start_idx : {start_idx}, opt_done : {opt_done}")

            #
            for target_idx in range(start_idx, len(l_input_tensor)) :
                #
                if opt_print == 1 :
                    print(f">1> target_idx : {target_idx}, opt_done : {opt_done}")
                
                # [Check] This index is already mapped on TB
                if tc_helper.tc_helper_find_index(l_FRAG, l_input_tensor[target_idx]) != -1 :
                    continue

                # [Check] This index is one of internal indices
                if tc_helper.tc_helper_find_index(l_internal_indices, l_input_tensor[target_idx]) != -1 :
                    continue

                #
                if opt_print == 1 :
                    print(f"l_input_tensor[{target_idx}] : {l_input_tensor[target_idx]}")

                #
                target_idx_representative_size = tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])
                vol_REG *= target_idx_representative_size

                # #   |REG'| >= |REG|
                # if vol_REG >= each_size_REG :
                #     #   |REG'| > |REG|
                #     if vol_REG > each_size_REG :
                #         blocking_tile_size = int(each_size_REG / vol_REG_prev)
                #         l_REG.append(l_input_tensor[target_idx])
                #         list_temp_tile_sizes.append([l_input_tensor[target_idx], blocking_tile_size])
                #     #   |REG'| = |REG|
                #     else :
                #         l_REG.append(l_input_tensor[target_idx])
                #         list_temp_tile_sizes.append([l_input_tensor[target_idx], target_idx_representative_size])
                #     #
                #     #   [Done] "REG" is fully mapped, but need to check if there are unmapped indices or not.
                #     #
                #     opt_done = 1
                #     break
                # #   |REG'| < |REG|
                # else :
                #     l_REG.append(l_input_tensor[target_idx])
                #     list_temp_tile_sizes.append([l_input_tensor[target_idx], target_idx_representative_size])

                # revised 260309
                blocking_tile_size = int(each_size_REG / vol_REG_prev)
                l_REG.append(l_input_tensor[target_idx])
                list_temp_tile_sizes.append([l_input_tensor[target_idx], blocking_tile_size])
                opt_done = 1
                break

            #
            if opt_print == 1 :
                print(f"l_REG : {l_REG}")
            
            #
            if opt_done == -1 :
                for target_idx in range(0, start_idx) :
                    #
                    if opt_print == 1 :
                        print(f">2> target_idx : {target_idx}, opt_done : {opt_done}")
            
                    #
                    target_idx_representative_size = tc_helper.tc_helper_find_value(l_representative_problem_size, l_input_tensor[target_idx])
                    vol_REG *= target_idx_representative_size

                    #   |REG'| >= |REG|
                    if vol_REG >= each_size_REG :
                        #   |REG'| > |REG|
                        if vol_REG > each_size_REG :
                            blocking_tile_size = int(each_size_REG / vol_REG_prev)
                            opt_start_mapped = 1
                            l_REG.append(l_input_tensor[target_idx])
                            list_temp_tile_sizes.append([l_input_tensor[target_idx], blocking_tile_size])
                        #   |REG'| = |REG|
                        else :
                            opt_start_mapped = 1
                            l_REG.append(l_input_tensor[target_idx])
                            list_temp_tile_sizes.append([l_input_tensor[target_idx], target_idx_representative_size])
                        #
                        #   [Done] "REG" is fully mapped, but need to check if there are unmapped indices or not.
                        #
                        opt_done = 1
                        break   # for "target_idx"
                    #   |REG'| < |REG|
                    else :
                        opt_start_mapped = 1
                        l_REG.append(l_input_tensor[target_idx])
                        list_temp_tile_sizes.append([l_input_tensor[target_idx], target_idx_representative_size])
            
            #
            if opt_done == 1 :
                for each_idx in l_input_tensor :
                    #   "each_idx" is mapped on REG
                    if tc_helper.tc_helper_find_index(l_REG, each_idx) != -1 :
                        continue
                    #   "each_idx" is mapped on FRAG
                    if tc_helper.tc_helper_find_index(l_FRAG, each_idx) != -1 :
                        continue
                    #   "each_idx" is an internal index
                    if tc_helper.tc_helper_find_index(l_internal_indices, each_idx) != -1 :
                        continue

                    #
                    l_inherited_FRAG.append(each_idx)
                    list_temp_tile_sizes.append([each_idx, 1])
                #
                #   [Configuration][Partial] FRAG and REG
                #
                l_partial_config_R.append([l_inherited_FRAG, l_REG, list_temp_tile_sizes])    

            #
            if opt_print == 1 :
                print(f"l_partial_config_R : {l_partial_config_R}")
                print ("============================================================================")

        #
        if opt_done == -1 :
            opt_fvi_input   = -1
            list_tmp_fvi    = []
            for each_idx in l_input_tensor :
                #   "each_idx" -> REG
                if tc_helper.tc_helper_find_index(l_REG, each_idx) != -1 :
                    continue
                
                #   "each_idx" -> TB
                if tc_helper.tc_helper_find_index(l_FRAG, each_idx) != -1 :
                    continue
                
                #   "each_idx" -> K
                if tc_helper.tc_helper_find_index(l_internal_indices, each_idx) != -1 :
                    continue
                
                #   "each_idx" == input's FVI
                if each_idx == l_input_tensor[0] :
                    continue
                
                #
                opt_fvi_input = 1

            #
            if opt_fvi_input == -1 :
                #  
                if tc_helper.tc_helper_find_index(l_FRAG, l_input_tensor[0]) == -1 :
                    #
                    l_REG.append(l_input_tensor[0])
                    
                    #
                    list_temp_tile_sizes.append([l_input_tensor[0], each_size_REG])
                    
                    #
                    l_partial_config_R.append([l_inherited_FRAG, l_REG, list_temp_tile_sizes])
    
    #
    return l_partial_config_R

#
#   [Configuration][Algorithm][FRAG_K]
#   opt_print: -1 (off), 0 (basic info.), 1 (basic info. + debug info.)
#
def alg_config_K(l_internal_indices, l_representative_problem_size, l_tiles_FRAG_K, opt_print) :
    #
    list_partial_config_K = []

    # Each Tile-Size
    for each_size_FRAG in l_tiles_FRAG_K :
        # 
        for start_idx in range(0, len(l_internal_indices)) :
            #
            vol_FRAG_K              = 1
            vol_FRAG_K_prev         = 1
            list_FRAG_K             = []
            list_temp_tile_sizes    = []
            opt_done                = -1

            #
            for target_idx in range(start_idx, len(l_internal_indices)) :
                #
                if opt_done == -1 :
                    # vol_FRAG_K *= tc_helper.tc_helper_find_value(l_representative_problem_size, l_internal_indices[target_idx])
                    
                    # #   |FRAG_K'| >= |FRAG_K|
                    # if vol_FRAG_K >= each_size_FRAG :
                    #     #   |FRAG_K'| > |FRAG_K|
                    #     if vol_FRAG_K > each_size_FRAG :
                    #         blocking_tile_size = int(each_size_FRAG / vol_FRAG_K_prev)
                    #         list_FRAG_K.append(l_internal_indices[target_idx])
                    #         list_temp_tile_sizes.append([l_internal_indices[target_idx], blocking_tile_size])
                    #     #   |FRAG_K'| = |FRAG_K|
                    #     else :
                    #         list_FRAG_K.append(l_internal_indices[target_idx])
                    #         list_temp_tile_sizes.append([l_internal_indices[target_idx], tc_helper.tc_helper_find_value(l_representative_problem_size, l_internal_indices[target_idx])])
                    #     #
                    #     opt_done = 1
                    # #   |FRAG_K'| < |FRAG_K|
                    # else :
                    #     list_FRAG_K.append(l_internal_indices[target_idx])
                    #     list_temp_tile_sizes.append([l_internal_indices[target_idx], tc_helper.tc_helper_find_value(l_representative_problem_size, l_internal_indices[target_idx])])
                    # #
                    # vol_FRAG_K_prev *= tc_helper.tc_helper_find_value(l_representative_problem_size, l_internal_indices[target_idx])

                    # revised 260309
                    blocking_tile_size = int(each_size_FRAG)
                    list_FRAG_K.append(l_internal_indices[target_idx])
                    list_temp_tile_sizes.append([l_internal_indices[target_idx], blocking_tile_size])
                    opt_done = 1
                #
                else :
                    list_FRAG_K.append(l_internal_indices[target_idx])
                    list_temp_tile_sizes.append([l_internal_indices[target_idx], 1])

            #
            for target_idx in range(0, start_idx) :
                #
                if opt_done == -1 :
                    vol_FRAG_K *= tc_helper.tc_helper_find_value(l_representative_problem_size, l_internal_indices[target_idx])

                    #   |FRAG_K'| >= |FRAG_K|
                    if vol_FRAG_K >= each_size_FRAG :
                        #   |FRAG_K'| > |FRAG_K|
                        if vol_FRAG_K > each_size_FRAG :
                            blocking_tile_size = each_size_FRAG / vol_FRAG_K_prev
                            list_FRAG_K.append(l_internal_indices[target_idx])
                            list_temp_tile_sizes.append([l_internal_indices[target_idx], int(blocking_tile_size)])
                        #   |FRAG_K'| = |FRAG_K|
                        else :
                            list_FRAG_K.append(l_internal_indices[target_idx])
                            list_temp_tile_sizes.append([l_internal_indices[target_idx], tc_helper.tc_helper_find_value(l_representative_problem_size, l_internal_indices[target_idx])])
                        #
                        opt_done = 1
                    #   |FRAG_K'| < |FRAG_K|
                    else :
                        list_FRAG_K.append(l_internal_indices[target_idx])
                        list_temp_tile_sizes.append([l_internal_indices[target_idx], tc_helper.tc_helper_find_value(l_representative_problem_size, l_internal_indices[target_idx])])
                    #
                    vol_FRAG_K_prev *= tc_helper.tc_helper_find_value(l_representative_problem_size, l_internal_indices[target_idx])
                #
                else :
                    list_FRAG_K.append(l_internal_indices[target_idx])
                    list_temp_tile_sizes.append([l_internal_indices[target_idx], 1])

            #
            if opt_done == 1 :
                list_partial_config_K.append([each_size_FRAG, list_FRAG_K, list_temp_tile_sizes])    

            #
            if opt_print == 1 :
                print(f"|FRAG_K| = {each_size_FRAG}, opt_done = {opt_done}")
                print(f"[Final Result] list_FRAG_K          : {list_FRAG_K}")
                print(f"[Final Result] list_temp_tile_sizes : {list_temp_tile_sizes}")

    #
    #
    #
    return list_partial_config_K

#
def alg_config_FVI(l_input_tensor_left, l_input_tensor_right, l_internal_indices, l_tile_FVI, opt_print_FVI) :
    #
    l_left_FVI           = []
    l_right_FVI          = []
    l_partial_config_FVI = []
    fvi_l                = l_input_tensor_left[0]
    fvi_r                = l_input_tensor_right[0]

    #
    for each_size_FVI in l_tile_FVI :
        #
        if opt_print_FVI == 1 :
            print(f"each_size_FVI : {each_size_FVI}, left_FVI : {fvi_l}, right_FVI : {fvi_r}")
        
        #
        l_left_FVI.append([l_input_tensor_left[0], each_size_FVI])
        l_right_FVI.append([l_input_tensor_right[0], each_size_FVI])
    
    #
    for left in l_left_FVI :
        for right in l_right_FVI :
            l_partial_config_FVI.append([left, right])
    
    #
    if (fvi_l in l_internal_indices) and (fvi_r in l_internal_indices) :
        #
        tmp_config = []
        
        #
        for i, config in enumerate(l_partial_config_FVI) :
            #
            left, right = config[0].copy(), config[1].copy()

            #
            if i % 2 == 0 :
                right[1] = 1
            else :
                left[1] = 1

            #
            tmp_config.append([left, right])
        
        #
        l_partial_config_FVI = tmp_config

    #
    if opt_print_FVI == 1 :
        print(f"l_partial_config_FVI : {l_partial_config_FVI}")
        print("============================================================================")

    #
    #
    #
    return l_partial_config_FVI