import os
import sys

import tc_helper            as tc_helper
import generators.interface as interface

from generators.kernels.kernel_body import tc_code_kernel
from generators.tensors_variables   import tc_code_variables
from generators.header_define       import tc_code_define

# check constraints of configurations
def tc_code_constraints(f, size_FRAG_X, size_FRAG_Y, producer_cnt) :
    #
    if(size_FRAG_X * size_FRAG_Y) < 64 :
        #
        print(f"Number of Threads in a Thread Block should be greater than or equal to 64 : {size_FRAG_X * size_FRAG_Y} < 64", file=sys.stderr)
        
        #
        f.close()
        os.remove(f.name)
        sys.exit()
    else : 
        #
        if(size_FRAG_X * size_FRAG_Y) > 1024 :
            #
            print(f"Number of Threads in a Thread Block should be less than or equal to 1024 : {size_FRAG_X * size_FRAG_Y} > 1024", file=sys.stderr)
            
            #
            f.close()
            os.remove(f.name)
            sys.exit()
    
    #
    if (size_FRAG_X * size_FRAG_Y) + (32 * producer_cnt) > 1024 :
        #
        print(f"Number of Registers per Thread exceeds the limit : {(size_FRAG_X * size_FRAG_Y) + (32 * producer_cnt)}", file=sys.stderr)
        
        #
        f.close()
        os.remove(f.name)
        sys.exit()

# generate pragma and include headers
def tc_code_include(f):
    f.write("#include <mma.h>\n")
    f.write("#include <cooperative_groups.h>\n")
    f.write("#include <cuda/barrier>\n")
    f.write("#include <cuda/pipeline>\n")
    f.write("\n")

# Input : target_number, str_target_number, str_target_number_config, list_inner_group, list_interface_info, opt_pre_computed, opt_data_type
# Output : none
def tc_code_gen(l_inner_groups, code_path, data_type, opt_pre_computed, check_cuda) :
    #
    # print("[Code Generator][tc_code_gen] Generate Kernels")
    
    # file open
    # print(f"file name : {code_path}", file=sys.stderr)
    f = open(code_path, "w")

    # write pragma and include
    tc_code_include(f)

    # name of kernel interface and kernel
    interface_name = "interface"
    kernel_name = "kernel"

    # list for generating code
    l_t3_slices_size         = list()        # tile size of t3 for each index 
    l_index_mappings         = list()        # mapping of t3 for each index

    # l_combined_t3_d_decl_var    = list()
    # l_combined_t2_d_decl_var    = list()
    # l_combined_v2_d_decl_var    = list()
    # l_combined_t3_parameters    = list()
    # l_combined_t2_parameters    = list()
    # l_combined_v2_parameters    = list()

    # tile size and index mapping
    for each_inner_group in l_inner_groups :
        l_t3_slices_size.append(each_inner_group[8])
        l_index_mappings.append([each_inner_group[1], each_inner_group[2]])

    # l_combined_var_input_left       = list()
    # l_combined_var_input_right      = list()

    # l_combined_t3_d_decl_var        = list()
    # l_combined_t2_d_decl_var        = list()
    # l_combined_v2_d_decl_var        = list()

    # l_combined_t3_parameters        = list()
    # l_combined_t2_parameters        = list()
    # l_combined_v2_parameters        = list()
    # l_combined_register_mappings    = list()
    # l_combined_inputs_int_strides   = list()

    #
    for each_inner_group in l_inner_groups :
        #
        l_var_output       = list()
        l_var_input_left    = list()
        l_var_input_right   = list()

        l_t3_d_decl_var     = list()
        l_t2_d_decl_var     = list()
        l_v2_d_decl_var     = list()
        
        l_cuda_memcpy       = list()
        l_cuda_malloc       = list()

        l_input_strides     = list()    # stride of internal index for each input tensors
                                        
                                        # input_tensors      # external_index    # internal_index
        # tc_code_variables(each_inner_group[6], each_inner_group[4], each_inner_group[5], each_inner_group[9],
        #                                 l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var,
        #                                 l_var_output, l_var_input_left, l_var_input_right, l_cuda_malloc, l_cuda_memcpy,
        #                                 l_input_strides, data_type)
        tc_code_variables(each_inner_group[6], each_inner_group[4], each_inner_group[5], each_inner_group[9],
                                        l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var,
                                        l_var_output, l_var_input_left, l_var_input_right, l_cuda_malloc, l_cuda_memcpy,
                                        l_input_strides, data_type)

                                                                                # input_tensors      # internal_index    # tiles_size 
        size_SMEM_left, size_SMEM_right = interface.tc_interface_SMEM_size(each_inner_group[6], each_inner_group[5], each_inner_group[8])
        size_FRAG_X, size_FRAG_Y = interface.tc_interface_FRAG_size(each_inner_group[1], each_inner_group[13])
        size_UNIT = interface.tc_interface_UNIT_size(each_inner_group[8], each_inner_group[5])
        split_index, split_input = interface.tc_interface_split(each_inner_group[1], each_inner_group[2], each_inner_group[4], each_inner_group[6])

        #
        tc_code_constraints(f, size_FRAG_X, size_FRAG_Y, each_inner_group[14])

        #
        fvi_flag, input_a, input_b, input_tensor_a, input_tensor_b = tc_helper.tc_code_kernel_helper_fvi(each_inner_group[4], each_inner_group[7])
        
        #
        ld_index_a, ld_index_b, ld_blk_index_a, ld_blk_index_b = tc_helper.tc_code_kernel_helper_ld_index(each_inner_group[8], each_inner_group[4], each_inner_group[2], input_tensor_a, input_tensor_b)

        #
        split_ld_index_a, split_ld_index_b = tc_helper.tc_code_kernel_helper_split_ld_index(ld_index_a, ld_index_b, split_index)
        
        #
        SMEM_order_a, SMEM_order_b, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b = tc_helper.tc_code_kernel_helper_tile_order(each_inner_group[1], each_inner_group[2], each_inner_group[5], 
                                                                                                                                            ld_index_a, ld_index_b, split_ld_index_a, split_ld_index_b, split_index)

        #        
        internal_order = tc_helper.tc_code_kernel_helper_iteration_order(each_inner_group[5], each_inner_group[13])

        #
        stride_helper = interface.tc_interface_splited_stride(each_inner_group[4], split_index)

        #
        # tc_code_define(f, each_inner_group[5], each_inner_group[12], each_inner_group[13], each_inner_group[14], input_a, input_b, SMEM_order_a, SMEM_order_b, split_index, check_cuda)
        tc_code_define(f, each_inner_group[5], each_inner_group[12], each_inner_group[13], each_inner_group[14], input_a, input_b, SMEM_order_a, SMEM_order_b, split_index, check_cuda)

        #
        # reg_y_padd, reg_x_padd = tc_code_kernel(f, kernel_name, l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var,
        #                                         each_inner_group[7], each_inner_group[4], each_inner_group[5], l_input_strides, each_inner_group[13],
        #                                         fvi_flag, input_a, input_b, input_tensor_a, input_tensor_b, internal_order,
        #                                         ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, ld_blk_index_a, ld_blk_index_b, SMEM_order_a, SMEM_order_b,
        #                                         split_index, split_input, stride_helper, each_inner_group[11], each_inner_group[14], each_inner_group[15],
        #                                         opt_pre_computed, data_type)
        reg_y_padd, reg_x_padd = tc_code_kernel(f, kernel_name, l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var, each_inner_group[9],
                                                        each_inner_group[7], each_inner_group[4], each_inner_group[5], l_input_strides, each_inner_group[13],
                                                        fvi_flag, input_a, input_b, input_tensor_a, input_tensor_b, internal_order,
                                                        ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, ld_blk_index_a, ld_blk_index_b, SMEM_order_a, SMEM_order_b,
                                                        split_index, split_input, stride_helper, each_inner_group[11], each_inner_group[14], each_inner_group[15],
                                                        opt_pre_computed, data_type)
        
        # print(f"split_ld_index_a : {split_ld_index_a}, split_ld_index_b : {split_ld_index_b}", file=sys.stderr)
        # print(f"SMEM_order_a : {SMEM_order_a}, SMEM_order_b : {SMEM_order_b}", file=sys.stderr)
        # print(f"reg_y_padd : {reg_y_padd}, reg_x_padd : {reg_x_padd}", file=sys.stderr)
        # #
        # l_combined_var_input_left.append(l_var_input_left)
        # l_combined_var_input_right.append(l_var_input_right)
        # l_combined_t3_d_decl_var.append(l_t3_d_decl_var)
        # l_combined_t2_d_decl_var.append(l_t2_d_decl_var)
        # l_combined_v2_d_decl_var.append(l_v2_d_decl_var)
        # l_combined_t3_parameters.append(l_t3_parameters)
        # l_combined_t2_parameters.append(l_t2_parameters)
        # l_combined_v2_parameters.append(l_v2_parameters)
        # l_combined_register_mappings.append(each_inner_group[2])
        # l_combined_inputs_int_strides.append(l_input_strides)

    #
    # total_y_padding, total_x_padding = interface.tc_interface_SMEM_padding(each_inner_group[13], each_inner_group[15], fvi_flag, ld_tile_order_a, ld_tile_order_b, SMEM_order_a, SMEM_order_b, reg_y_padd, reg_x_padd)

    # #
    # interface.tc_interface_runner(f, interface_name, kernel_name, l_interface_info,
                                    #  l_inner_groups[0][4], l_inner_groups[0][5],
                                    #  l_var_output, l_var_input_left, l_var_input_right, l_cuda_malloc, l_cuda_memcpy,
                                    #  size_SMEM_left, size_SMEM_right, size_UNIT,
                                    #  fvi_flag, input_a, input_b, internal_order, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, SMEM_order_a, SMEM_order_b,
                                    #  total_y_padding, total_x_padding,
                                    #  split_input, l_inner_groups[0][14], check_cuda)
    
    #
    print("[Code Generator] Kernel Generation done", file=sys.stderr)