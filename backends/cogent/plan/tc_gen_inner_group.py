import os
import re, copy
from collections import OrderedDict
import sys

from base.proc_configuration import get_configurations
from base.proc_configuration import transform_config_inner_group
from tc_helper import tc_helper_find_value
#
def tc_gen_inner_group(equation_info, tensors, index_to_extent, equation, variant_num, opt_print, data_type) :
    #
    if opt_print == 1 :
        print("====================== Step 2: Creating Inner-Groups =======================")
        print("[Code Generator][tc_gen_inner_group] Working...")
        print(" Only Support the First Outer-Group")

    #
    # Make all configuartions
    #
    l_configurations_outer_group = list()
    l_split_outer_group = copy.deepcopy(equation_info)
    str_binary_input = get_configurations(l_split_outer_group, tensors, index_to_extent, l_configurations_outer_group, equation, variant_num, opt_print, data_type)

    # configuration_info_flag = 1
    # if configuration_info_flag :
    #     for idx, each_config_outer_group in enumerate(l_configurations_outer_group) :
    #         print("============================================================================")
    #         print(f"Config #. {idx}")
    #         each_config_outer_group.print_configuration(1)
    #         print("============================================================================")
    #     sys.exit()
    
    #
    info_each_inner_group = transform_config_inner_group(l_configurations_outer_group)

    #
    non_split_all_index = equation_info[0][2]

    #
    # print(f"[Code Generator][tc_gen_inner_group] # of Outer-Groups : {len(equation_info)}")
    
    #
    for each_outer_group in equation_info :
        #
        # print(f"[Code Generator][tc_gen_inner_group] # of Tensor Contractions (Candidates) within an Outer-Group : {len(each_outer_group[1])}")

        #
        #   Within an Outer-Group, there might be several Inner-Groups.
        #
        l_inner_groups            = list()
        l_each_group_mapping_frag = list()
        l_each_group_mapping_2D   = list()
        l_each_group_mapping_reg  = list()
        l_t3_slices_size          = list()
        l_t3_interface_info       = list()
        l_t3_temp_inputs          = list()

        #
        #   To Create "Interface"
        #
        idx_count           = 0
        str_common_output   = ""
        for each_tc in each_outer_group[1] :
            l_t3_temp_inputs.append([each_tc[4], each_tc[6]])
            if idx_count == 0 :
                str_common_output = each_tc[0]
            idx_count = idx_count + 1   

        #
        #   l_interface_info: [0] All Index, [1] Output, [2] Inputs, [3] Conditions, [4] Options
        #
        l_t3_interface_info.append([non_split_all_index, str_common_output, l_t3_temp_inputs])

        #
        #   (Temporary)
        #                           [0]          [1]          [2]        [3]            [4]
        #   each_manual_group: Mapping_TB, Mapping_TB_2D, Mapping_Reg, Slices, Split-Info(Repre-size)
        #
        for each_manual_group in info_each_inner_group :
            #
            l_each_group_mapping_frag   = each_manual_group[0]
            l_each_group_mapping_2D     = each_manual_group[1]
            l_each_group_mapping_reg    = each_manual_group[2]
            l_t3_slices_size            = each_manual_group[3]
            l_info_index_size           = each_manual_group[4]
            l_each_group_mapping_FRAG_K = each_manual_group[5]
            warp_shape                  = each_manual_group[6]
            stage                       = each_manual_group[7]
            producer_cnt                = each_manual_group[8]
            double2_flag                = each_manual_group[9]
            padding                     = each_manual_group[10]
            l_tensor_contractions       = list()

            #
            if opt_print == 1 :
                print(f"[Code Generator][tc_gen_inner_group] Picked Tiles : {l_t3_slices_size}")

            #
            merged = OrderedDict()
            base = re.compile(r"^([A-Za-z_]+)")
            for idx, val in l_t3_slices_size :
                tmp = base.match(idx).group(1)
                if tmp in merged :
                    merged[tmp] *= val
                else :
                    merged[tmp] = val
            l_merged_t3_slices_size = [[k, merged[k]] for k in merged]

            #
            for each_tc in each_outer_group[1] :
                l_tensor_contractions.append(each_tc)

            l_inner_groups.append([l_each_group_mapping_frag, l_each_group_mapping_2D, l_each_group_mapping_reg, l_tensor_contractions, 
                                   l_merged_t3_slices_size, l_t3_slices_size, l_info_index_size, l_each_group_mapping_FRAG_K, warp_shape, stage, producer_cnt, double2_flag, padding])

    #
    if opt_print == 1 :
        #
        print("============================================================================")
        print("===================== Step 2: [Output] Inner-Groups ========================")
        print(f" # of Inner-Groups : {len(l_inner_groups)}")
        print("============================================================================")
        
        #
        for each_inner_group in l_inner_groups :
            #
            print("============================================================================")
            print(f"Mapping All              : {each_inner_group[0]}")
            print(f"Mapping FRAG             : {each_inner_group[1]}")
            print(f"Mapping REG              : {each_inner_group[2]}")
            print(f"Mapping FRAG_K           : {each_inner_group[7]}")
            print(f"Slices                   : {each_inner_group[4]}")
            print(f"Split Slices             : {each_inner_group[5]}")
            print(f"Warp Shape               : {each_inner_group[8]}")
            print(f"Pipeline Stage           : {each_inner_group[9]}")
            print(f"Producer Count           : {each_inner_group[10]}")
            print(f"Double2 Flag             : {each_inner_group[11]}")

            print(f"# of Tensor Contractions : {len(each_inner_group[3])}")
            
            #
            for each_tc in each_inner_group[3] :
                print(f"Each Tensor Contraction : {each_tc}")
            
            #
            print("============================================================================")
        
        #
        print("============================================================================")

    #
    return l_inner_groups, l_t3_interface_info, str_binary_input

#
def tc_gen_processing_inner_group(l_inner_groups, l_split_outer_group, opt_print) :
    #
    if opt_print == 1 :
        print("=================== Step 3: Processing Inner-Groups ========================")
        print(" Creates Data Structures used to create a Kernel based on a given inner group.")

    #
    l_temp_inner_output = list()

    #
    for each_inner_group in l_inner_groups :
        #
        l_temp_input_tensors    = list()
        l_temp_input_addrs      = list()
        l_temp_external_indices = list()
        l_temp_internal_indices = list()
        l_temp_all_indices      = list()

        #
        if opt_print == 1 :
            print("============================================================================")
            print(f"[Code Generator][tc_gen_processing_inner_group] Mapping All    : {each_inner_group[0]}")
            print(f"[Code Generator][tc_gen_processing_inner_group] Mapping FRAG   : {each_inner_group[1]}")
            print(f"[Code Generator][tc_gen_processing_inner_group] Mapping REG    : {each_inner_group[2]}")
            print(f"[Code Generator][tc_gen_processing_inner_group] Mapping FRAG_K : {each_inner_group[7]}")
            print(f"[Code Generator][tc_gen_processing_inner_group] Slices         : {each_inner_group[4]}")
            print(f"[Code Generator][tc_gen_processing_inner_group] Warp Shape     : {each_inner_group[8]}")
            print(f"[Code Generator][tc_gen_processing_inner_group] Pipeline Stage : {each_inner_group[9]}")
            print(f"[Code Generator][tc_gen_processing_inner_group] Producer Count : {each_inner_group[10]}")
            print(f"[Code Generator][tc_gen_processing_inner_group] Double2 flag   : {each_inner_group[11]}")

        #
        l_temp_external_indices = l_split_outer_group[0][0]
        l_temp_internal_indices = each_inner_group[3][0][3]
        
        #
        for each_ext_idx in l_temp_external_indices :
            l_temp_all_indices.append([each_ext_idx, 16])

        #
        for each_int_idx in l_temp_internal_indices :
            l_temp_all_indices.append([each_int_idx, 16])

        #
        for each_tc in each_inner_group[3] :
            #
            str_left_mapping = ""
            for left_idx in each_tc[5] :
                if left_idx == each_inner_group[2][0] :
                    str_left_mapping = "x"
                if left_idx == each_inner_group[2][1] :
                    str_left_mapping = "y"

            #
            str_right_mapping = ""
            for right_idx in each_tc[7] :
                if right_idx == each_inner_group[2][0] :
                    str_right_mapping = "x"
                if right_idx == each_inner_group[2][1] :
                    str_right_mapping = "y"

            #
            #   Create lists called input_tensors and input_addrs (which can be combined in the future)
            #
            l_temp_input_tensors.append([[each_tc[4], l_split_outer_group[0][1][0][5]], [each_tc[6], l_split_outer_group[0][1][0][7]], each_tc[2]])
            l_temp_input_addrs.append([ [16, "STR_SD2_" + each_tc[4].capitalize() + "_H7", str_left_mapping,  each_tc[4], l_split_outer_group[0][1][0][5]],
                                        [16, "STR_SD2_" + each_tc[6].capitalize() + "_H7", str_right_mapping, each_tc[6], l_split_outer_group[0][1][0][7]], 
                                        each_tc[2]])
        
        #
        l_temp_inner_output.append([each_inner_group[0], each_inner_group[1], each_inner_group[2],
                                    l_temp_all_indices, l_temp_external_indices, l_temp_internal_indices,
                                    l_temp_input_tensors, l_temp_input_addrs, each_inner_group[4], l_split_outer_group[0][1][0][8],
                                    each_inner_group[7], each_inner_group[8], [each_inner_group[9]], each_inner_group[5], each_inner_group[10], each_inner_group[11]])
        
        #
        l_kernal_binary = [each_inner_group[0], each_inner_group[2], l_temp_input_tensors, each_inner_group[8], [each_inner_group[9]], each_inner_group[5], each_inner_group[11], each_inner_group[12]]

    #
    if opt_print == 1 :
        print("============================================================================")
    
    #
    return l_temp_inner_output, l_kernal_binary


def make_kernel_name(l_kernal_binary):
    frag_mapped = l_kernal_binary[0]
    reg_mapped  = l_kernal_binary[1]
    t2_indices  = l_kernal_binary[2][0][0][1]
    v2_indices  = l_kernal_binary[2][0][1][1]
    op          = l_kernal_binary[2][0][2]
    warp_shape  = l_kernal_binary[3]
    stage       = l_kernal_binary[4]
    tile_sizes  = l_kernal_binary[5]
    d2_flag     = l_kernal_binary[6]

    op_map  = {'+=': 'iadd', '-=': 'isub'}
    op_str  = op_map.get(op, op)

    t2_str    = 't2.' + '_'.join(t2_indices)           # t2.b.d.a
    v2_str    = 'v2.' + '_'.join(v2_indices)           # v2.d.c
    frag_str  = 'frag.' + '_'.join(frag_mapped)        # frag.a.c1
    reg_str   = 'reg.'  + '_'.join(reg_mapped)         # reg.b.c2
    tile_str = '.'.join(f"{k}_{v}" for k, v in tile_sizes)  # d8.a8.b16.c1.16.c2.2
    warp_str  = 'w.' + 'x'.join(map(str, warp_shape))  # w4x1
    stage_str = 's.' + ''.join(map(str, stage))         # s3
    d2_str    = 'd2.' + 'x'.join(map(str, d2_flag))   # d2.1x1

    parts = [
        t2_str, v2_str, op_str,   # 핵심 그룹
        tile_str,                  # 핵심 그룹
        frag_str, reg_str,         # 매핑 그룹
        warp_str, stage_str, d2_str  # 실행 설정 그룹
    ]

    kernel_bin = 'kernel__' + '__'.join(parts)
        
    return kernel_bin


def make_launch_config(l_kernal_binary, kernel_bin, l_external_index, l_internal_index, combined_tile_size, index_to_extent) :
    frag_mapped = l_kernal_binary[0]
    reg_mapped  = l_kernal_binary[1]
    t2_indices  = l_kernal_binary[2][0][0][1]
    v2_indices  = l_kernal_binary[2][0][1][1]
    warp_shape  = l_kernal_binary[3]
    stage       = l_kernal_binary[4]
    tile_sizes  = l_kernal_binary[5]
    internal_index = tile_sizes[0][0]
    padding     = l_kernal_binary[7]

    #
    split_indices = []
    if len(t2_indices) - len(l_internal_index) == 1 :
        for idx in t2_indices :
            if idx not in l_internal_index :
                split_indices.append(idx)
    if len(v2_indices) - len(l_internal_index) == 1 :
        for idx in v2_indices :
            if idx not in l_internal_index :
                split_indices.append(idx)

    #
    frag_full = True
    for frag_mapped_idx in frag_mapped :
        if frag_mapped_idx.rstrip('12') in split_indices :
            continue
        if index_to_extent[frag_mapped_idx] % tc_helper_find_value(tile_sizes, frag_mapped_idx) != 0 :
            frag_full &= False
        else :
            frag_full &= True
    
    if len(split_indices) > 0 :
        for split_idx in split_indices :
            if index_to_extent[split_idx] % tc_helper_find_value(combined_tile_size, split_idx) != 0 :
                frag_full &= False
            else :
                frag_full &= True
     
    #
    reg_full = True
    for reg_mapped_idx in reg_mapped :
        if reg_mapped_idx.rstrip('12') in split_indices :
            continue
        if index_to_extent[reg_mapped_idx] % tc_helper_find_value(tile_sizes, reg_mapped_idx) != 0 :
            reg_full &= False
        else :
            reg_full &= True
    
    #
    internal_full = True
    if index_to_extent[internal_index] % tc_helper_find_value(tile_sizes, internal_index) != 0 :
        internal_full &= False
    else :
        internal_full &= True
    
    if len(split_indices) < 2 :
        if frag_full :
            if reg_full :
                if internal_full :
                    kernel_num = 1
                else :
                    kernel_num = 2
            else :
                if internal_full :
                    kernel_num = 3
                else :
                    kernel_num = 4
        else :
            if reg_full :
                if internal_full :
                    kernel_num = 5
                else :
                    kernel_num = 6
            else :
                if internal_full :
                    kernel_num = 7
                else :
                    kernel_num = 8
    else :
        if frag_full :
            if internal_full :
                kernel_num = 1
            else :
                kernel_num = 2
        else :
            if internal_full :
                kernel_num = 3
            else :
                kernel_num = 4

    #
    smem_x = 1
    smem_y = 1
    if "a" in t2_indices :
        for idx in t2_indices :
            if idx not in l_internal_index :
                smem_x *= tc_helper_find_value(combined_tile_size, idx)
        for idx in v2_indices :
            if idx not in l_internal_index :
                smem_y *= tc_helper_find_value(combined_tile_size, idx)
        swap_flag = True
    else :
        for idx in v2_indices :
            if idx not in l_internal_index :
                smem_x *= tc_helper_find_value(combined_tile_size, idx)
        for idx in t2_indices :
            if idx not in l_internal_index :
                smem_y *= tc_helper_find_value(combined_tile_size, idx)
        swap_flag = False

    #
    block_size = warp_shape[0] * warp_shape[1] * 32

    #
    internal_tile = tile_sizes[0][1]
    size_internal = index_to_extent[internal_index]

    #
    launch_config = {
        "kernel_bin" : kernel_bin,
        "external_index" : l_external_index,
        "internal_index" : l_internal_index,
        "tile_sizes" : combined_tile_size,
        "block_size" : block_size,
        "stage" : stage,
        "smem_x" : smem_x * internal_tile + padding[0],
        "smem_y" : smem_y * internal_tile + padding[1],
        "internal" : size_internal,
        "warp_shape" : warp_shape,
        "kernel_name" : f"kernel_{kernel_num}",
        "swap_flag" : swap_flag
    }

    return launch_config 
