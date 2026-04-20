import tc_helper     as tc_helper

# 
def tc_interface_SMEM_size(l_input_tensors, l_internal_index, l_tile_sizes) :
    #
    for each_input in l_input_tensors :
        #
        size_SMEM_left  = 1
        size_SMEM_right = 1

        for each_index in each_input[0][1] :
            if tc_helper.tc_helper_find_index(l_internal_index, each_index) == -1 :
                size_SMEM_left = size_SMEM_left * tc_helper.tc_helper_find_value(l_tile_sizes, each_index)  # external index of t2
        
        for each_index in each_input[1][1] :
            if tc_helper.tc_helper_find_index(l_internal_index, each_index) == -1 :
                size_SMEM_right = size_SMEM_right * tc_helper.tc_helper_find_value(l_tile_sizes, each_index)  # external index of v2

    return size_SMEM_left, size_SMEM_right

#
def tc_interface_FRAG_size(l_FRAG_mapping, l_tile_sizes) :
    #
    size_FRAG_X = 1
    for each_index in l_FRAG_mapping[0] :
        size_FRAG_X = size_FRAG_X * tc_helper.tc_helper_find_value(l_tile_sizes, each_index)

    #
    size_FRAG_Y = 1
    for each_index in l_FRAG_mapping[1] :
        size_FRAG_Y = size_FRAG_Y * tc_helper.tc_helper_find_value(l_tile_sizes, each_index)
    
    return size_FRAG_X, size_FRAG_Y

#
def tc_interface_UNIT_size(l_t3_slices, l_internal_index) :
    #
    internal_unit_size  = 1
    tmp_size            = 1

    #
    if len(l_internal_index) > 1:
        #
        for int_idx in l_internal_index:
            tmp_size = tmp_size * tc_helper.tc_helper_find_value(l_t3_slices, int_idx)
        #
        if tmp_size >= 16:
            internal_unit_size = 16
        else:
            internal_unit_size = tmp_size
    else:
        internal_unit_size = tc_helper.tc_helper_find_value(l_t3_slices, l_internal_index[0])

    return internal_unit_size

#
def tc_interface_split(l_tb_mapping, l_reg_mapping, l_external_index, l_input_tensors) :
    #
    all_tb_mapping = l_tb_mapping[0] + l_tb_mapping[1]
    all_reg_mapping = l_reg_mapping

    #
    base_tb = [''.join(filter(str.isalpha, ax)) for ax in all_tb_mapping]
    base_reg = [''.join(filter(str.isalpha, ax)) for ax in all_reg_mapping]

    #
    split_index = []
    for index in l_external_index :
        if base_tb.count(index) == 1 and base_reg.count(index) == 1:
            split_index.append(index)
    
    #
    t2_split_flag = 0
    v2_split_flag = 0
    split_input = []

    #
    if len(split_index) > 0 :
        for index in split_index :
            if index in l_input_tensors[0][0][1] :  # split t2
                t2_split_flag = 1
            if index in l_input_tensors[0][1][1] :  # split v2
                v2_split_flag = 1
    
    #
    split_input.append(t2_split_flag)
    split_input.append(v2_split_flag)

    return split_index, split_input

#
def tc_interface_splited_stride(l_external_index, split_index) :
    #
    splited_external_index = []
    
    #
    for i in l_external_index :
        #
        if i in split_index :
            splited_external_index.append(i + "1")
            splited_external_index.append(i + "2")
        else :
            splited_external_index.append(i)

    return splited_external_index

#
def tc_interface_runner_header(f, interface_name, l_interface_info) :
    #
    f.write("\n")
    f.write("// Kernel Runner for Tensor Contraction\n")
    f.write("void " + interface_name + "(")
    
    #
    idx_count = 0
    for each_index in l_interface_info[0][0]:
        if idx_count == 0:        
            f.write("int size_" + each_index)
        else:
            f.write(", int size_" + each_index)
        idx_count = idx_count + 1
    
    #
    f.write(", double *host_" + l_interface_info[0][1])

    #
    for each_pair_inputs in l_interface_info[0][2]:
        for each_input in each_pair_inputs:
            f.write(", double *host_" + each_input)

    f.write(")\n{\n")

#
def tc_interface_runner_variables(f, l_external_index, l_var_output, l_var_input_left, l_var_input_right, ld_tile_order_a, ld_tile_order_b, producer_cnt) :
    #
    f.write("\t")
    for each_output in l_var_output :
        f.write(each_output[0] + each_output[1] + ";\n")

    #
    f.write("\t")
    for t2_var in l_var_input_left :
        f.write(t2_var[0] + t2_var[1] + ";\n")
    
    #
    f.write("\t")
    for v2_var in l_var_input_right :
        f.write(v2_var[0] + v2_var[1] + ";\n")

    f.write("\n")

    #
    str_num_thread_blocks = ""
    idx_count = 0
    for each_idx in l_external_index:
        if idx_count == 0:
            str_num_thread_blocks = f"CEIL(size_{each_idx}, TILE_{each_idx.capitalize()})"
        else:
            str_num_thread_blocks = str_num_thread_blocks + f" * CEIL(size_{each_idx}, TILE_{each_idx.capitalize()})"
        idx_count = idx_count + 1
    f.write(f"\tconst int num_thread_blocks = {str_num_thread_blocks};\n")

    #
    if producer_cnt > 0 :
        f.write(f"\tconst int num_threads = TILE_{ld_tile_order_a[1].capitalize()} * TILE_{ld_tile_order_b[1].capitalize()} + PRODUCER_CNT * 32;\n\n")
    else :
        f.write(f"\tconst int num_threads = TILE_{ld_tile_order_a[1].capitalize()} * TILE_{ld_tile_order_b[1].capitalize()};\n\n")

#
def tc_interface_CUDA_malloc_memcpy(f, l_cuda_malloc, l_cuda_memcpy, check_cuda) :
    #
    for each_var in l_cuda_malloc:
        if "range"  in each_var[0]:
            continue
        if "addr"   in each_var[0]:
            continue
        if "offset" in each_var[0]:
            continue
        if "base"   in each_var[0]:
            continue
        tc_helper.tc_interface_helper_CUDA_malloc(f, each_var[0], each_var[1], each_var[2], check_cuda)

    f.write("\n")

    #
    for each_var in l_cuda_memcpy:
        if "range"  in each_var[1]:
            continue
        if "addr"   in each_var[1]:
            continue
        if "offset" in each_var[1]:
            continue
        if "base"   in each_var[1]:
            continue
        tc_helper.tc_interface_helper_CUDA_memcpy(f, each_var[1], each_var[2], each_var[0], each_var[3], 1, check_cuda)

    f.write("\n")

#
def tc_interface_Kernel_launch_variables(f, kernel_name, l_external_index, l_internal_index, input_a, input_b, kernel_num, tab, partial_information=None) :
    #
    if partial_information is not None :
        f.write("\t" * tab + f"{partial_information}")
        f.write("\t" * tab + f"dt_flag = {kernel_num};\n")

    #
    f.write("\t" * tab + f"{kernel_name}_{kernel_num}<<<gridDim, blockDim, shm_size>>>(dev_t3, dev_{input_a}, dev_{input_b},\n")
    
    #
    f.write("\t\t\t\t\t\t\t\t")
    for each_index in l_external_index :
        f.write(f" size_{each_index},")
    f.write("\n")

    #
    f.write("\t\t\t\t\t\t\t\t")
    for each_index in l_internal_index :
        f.write(f" size_{each_index},")
    f.write("\n")

    #
    f.write("\t\t\t\t\t\t\t\t")
    for each_index in l_external_index :
        f.write(f" CEIL(size_{each_index}, TILE_{each_index.capitalize()}),")
    f.write("\n")

    #
    f.write("\t\t\t\t\t\t\t\t")
    f.write(f" size_internal, shm_{input_a}, shm_{input_b});\n")

#
def tc_interface_Kernel_partial_information(kernel_num) :
    partial_information = []

    if kernel_num == 4 :
        partial_information.append("printf(\"External : Full / Internal : Full\\n\");\n")
        partial_information.append("printf(\"External : Full / Internal : Partial\\n\");\n")
        partial_information.append("printf(\"External : Partial / Internal : Full\\n\");\n")
        partial_information.append("printf(\"External : Partial / Internal : Partial\\n\");\n")
    else :
        partial_information.append("printf(\"Frag_mapped : Full, Reg_mapped : Full / Internal : FUll\\n\");\n")
        partial_information.append("printf(\"Frag_mapped : Full, Reg_mapped : Full / Internal : Partial\\n\");\n")
        partial_information.append("printf(\"Frag_mapped : Full, Reg_mapped : Partial / Internal : FUll\\n\");\n")
        partial_information.append("printf(\"Frag_mapped : Full, Reg_mapped : Partial / Internal : Partial\\n\");\n")
        partial_information.append("printf(\"Frag_mapped : Partial, Reg_mapped : Full / Internal : Full\\n\");\n")
        partial_information.append("printf(\"Frag_mapped : Partial, Reg_mapped : Full / Internal : Partial\\n\");\n")
        partial_information.append("printf(\"Frag_mapped : Partial, Reg_mapped : Partial / Internal : FUll\\n\");\n")
        partial_information.append("printf(\"Frag_mapped : Partial, Reg_mapped : Partial / Internal : Partial\\n\");\n")

    return partial_information

#
def tc_interface_SMEM_padding(l_splited_indices_size, double2_flag, fvi_flag, ld_tile_order_a, ld_tile_order_b, SMEM_order_a, SMEM_order_b, reg_y_padd, reg_x_padd) :
    #
    # if fvi_flag == 1 :
    #     a_double2_flag = double2_flag[1]
    #     b_double2_flag = double2_flag[0]
    # elif fvi_flag == 2 :
    #     a_double2_flag = double2_flag[0]
    #     b_double2_flag = double2_flag[1]
    a_double2_flag = double2_flag[1]
    b_double2_flag = double2_flag[0]

    #
    if a_double2_flag :
        if SMEM_order_a[2] == ld_tile_order_a[0] :
            total_y_padding = tc_helper.tc_helper_find_value(l_splited_indices_size, SMEM_order_a[0]) * reg_y_padd
        else :
            total_y_padding = 0
    else :
        if SMEM_order_a[2] == ld_tile_order_a[0] :
            total_y_padding = tc_helper.tc_helper_find_value(l_splited_indices_size, SMEM_order_a[2]) * reg_y_padd
        elif SMEM_order_a[2] == ld_tile_order_a[1] :
            total_y_padding = tc_helper.tc_helper_find_value(l_splited_indices_size, SMEM_order_a[0]) * reg_y_padd
        else :
            total_y_padding = 0

    #
    if b_double2_flag :
        if SMEM_order_b[2] == ld_tile_order_b[0] :
            total_x_padding = tc_helper.tc_helper_find_value(l_splited_indices_size, SMEM_order_b[0]) * reg_x_padd
        else :
            total_x_padding = 0
    else :
        if SMEM_order_b[2] == ld_tile_order_b[0] :
            total_x_padding = tc_helper.tc_helper_find_value(l_splited_indices_size, SMEM_order_b[2]) * reg_x_padd
        elif SMEM_order_b[2] == ld_tile_order_b[1] :
            total_x_padding = tc_helper.tc_helper_find_value(l_splited_indices_size, SMEM_order_b[0]) * reg_x_padd
        else :
            total_x_padding = 0

    return total_y_padding, total_x_padding

#
def tc_interface_RelatedKernels(f, kernel_name, l_external_index, l_internal_index,
                                size_SM_left, size_SM_right, size_UNIT,
                                fvi_flag, input_a, input_b, internal_order, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b,
                                total_y_padding, total_x_padding,
                                split_input, check_cuda) :
    #
    t2_split_flag = split_input[0]
    v2_split_flag = split_input[1]

    if fvi_flag == 1 :
        a_split_flag = v2_split_flag
        b_split_flag = t2_split_flag
        t2_padd = total_x_padding
        v2_padd = total_y_padding
    else :
        a_split_flag = t2_split_flag
        b_split_flag = v2_split_flag
        t2_padd = total_y_padding
        v2_padd = total_x_padding
    
    #
    str_operations = ""
    idx_count = 0
    for each_idx in l_external_index:
        if idx_count == 0:
            str_operations = f"size_{each_idx}"
        else:
            str_operations = str_operations + f" * size_{each_idx}"
        idx_count += 1

    #
    for each_idx in l_internal_index:
        str_operations = f"(long long int)({str_operations}) * size_{each_idx}"    
    f.write(f"\tlong long int operations = {str_operations};\n\n")

    #
    f.write("\tdim3 gridDim(num_thread_blocks);\n")
    f.write("\tdim3 blockDim(num_threads);\n\n")

    #
    f.write(f"\tconst uint shm_t2 = {size_SM_left} * {size_UNIT} + {t2_padd};\n")
    f.write(f"\tconst uint shm_v2 = {size_SM_right} * {size_UNIT} + {v2_padd};\n")
    f.write(f"\tconst uint shm_size = sizeof(double) * (shm_t2 + shm_v2) * PIPELINE_STAGES;\n\n")

    #
    str_internal = f"size_{internal_order[0]}"
    f.write(f"\tint size_internal = {str_internal};\n\n")
    
    #
    f.write("\tint dt_flag = 0;\n\n")

    #
    tab = 1
    f.write("\t// Kernel Launch\n")
    if a_split_flag and b_split_flag :
        #
        str_collapsed_a = f"size_{collapsed_a[0]} % TILE_{collapsed_a[0].capitalize()} == 0"
        str_collapsed_b = f"size_{collapsed_b[0]} % TILE_{collapsed_b[0].capitalize()} == 0"
        str_internal = f"size_{ld_tile_order_a[2]} % TILE_{ld_tile_order_a[2].capitalize()} == 0"
        partial_information = tc_interface_Kernel_partial_information(4)
        
        #
        for i in range(1, 5) :
            if i == 1 :
                f.write(f"\tif({str_collapsed_a} && {str_collapsed_b})\n")
                f.write("\t" * tab + "{\n"); tab += 1
            if i == 1 or i == 3 :       
                f.write("\t" * tab + f"if({str_internal})\n")
                f.write("\t" * tab + "{\n"); tab += 1
            tc_interface_Kernel_launch_variables(f, kernel_name, l_external_index, l_internal_index, input_a, input_b, i, tab, partial_information[i - 1]); tab -= 1
            
            if i == 1 or i == 3 :
                f.write("\t" * tab + "}\n")
                f.write("\t" * tab + "else\n")
                f.write("\t" * tab + "{\n"); tab += 1
            else :
                f.write("\t" * tab + "}\n"); tab -= 1
                f.write("\t" * tab + "}\n")
            
            if i == 2 :
                f.write("\t" * tab + "else\n")
                f.write("\t" * tab + "{\n"); tab += 1
    elif a_split_flag :
        #
        str_collapsed_a = f"size_{collapsed_a[0]} % TILE_{collapsed_a[0].capitalize()} == 0"
        str_frag_b = f"size_{ld_tile_order_b[1]} % TILE_{ld_tile_order_b[1].capitalize()} == 0"
        str_reg_b = f"size_{ld_tile_order_b[0]} % TILE_{ld_tile_order_b[0].capitalize()} == 0"
        str_internal = f"size_{ld_tile_order_b[2]} % TILE_{ld_tile_order_b[2].capitalize()} == 0"
        partial_information = tc_interface_Kernel_partial_information(8)

        #
        for i in range(1, 9) :
            if i == 1 :
                f.write(f"\tif({str_frag_b} && {str_collapsed_a})\n")
                f.write("\t" * tab + "{\n"); tab += 1
            if i == 1 or i == 5 :
                f.write(f"\t\tif({str_reg_b})\n")
                f.write("\t" * tab + "{\n"); tab += 1
            if i % 2 != 0 :
                f.write(f"\t\t\tif({str_internal})\n")
                f.write("\t" * tab + "{\n"); tab += 1
            tc_interface_Kernel_launch_variables(f, kernel_name, l_external_index, l_internal_index, input_a, input_b, i, tab, partial_information[i - 1]); tab -= 1
            if i % 2 != 0 :
                f.write("\t" * tab + "}\n")
                f.write("\t" * tab + "else\n")
                f.write("\t" * tab + "{\n"); tab += 1
            if i == 2 or i == 6 :
                f.write("\t" * tab + "}\n"); tab -= 1
                f.write("\t" * tab + "}\n")
                f.write("\t" * tab + "else\n")
                f.write("\t" * tab + "{\n"); tab += 1
            if i == 4 :
                f.write("\t" * tab + "}\n"); tab -= 1
                f.write("\t" * tab + "}\n"); tab -= 1
                f.write("\t" * tab + "}\n")
                f.write("\t" * tab + "else\n")
                f.write("\t" * tab + "{\n"); tab += 1
            if i == 8 :
                f.write("\t" * tab + "}\n"); tab -= 1
                f.write("\t" * tab + "}\n"); tab -= 1
                f.write("\t" * tab + "}\n"); tab -= 1
    elif b_split_flag :
        #
        str_collapsed_b = f"size_{collapsed_b[0]} % TILE_{collapsed_b[0].capitalize()} == 0"
        str_frag_a = f"size_{ld_tile_order_a[1]} % TILE_{ld_tile_order_a[1].capitalize()} == 0"
        str_reg_a = f"size_{ld_tile_order_a[0]} % TILE_{ld_tile_order_a[0].capitalize()} == 0"
        str_internal = f"size_{ld_tile_order_a[2]} % TILE_{ld_tile_order_a[2].capitalize()} == 0"
        partial_information = tc_interface_Kernel_partial_information(8)

        #
        for i in range(1, 9) :
            if i == 1 :
                f.write(f"\tif({str_frag_a} && {str_collapsed_b})\n")
                f.write("\t" * tab + "{\n"); tab += 1
            if i == 1 or i == 5 :
                f.write(f"\t\tif({str_reg_a})\n")
                f.write("\t" * tab + "{\n"); tab += 1
            if i % 2 != 0 :
                f.write(f"\t\t\tif({str_internal})\n")
                f.write("\t" * tab + "{\n"); tab += 1
            tc_interface_Kernel_launch_variables(f, kernel_name, l_external_index, l_internal_index, input_a, input_b, i, tab, partial_information[i - 1]); tab -= 1
            if i % 2 != 0 :
                f.write("\t" * tab + "}\n")
                f.write("\t" * tab + "else\n")
                f.write("\t" * tab + "{\n"); tab += 1
            if i == 2 or i == 6 :
                f.write("\t" * tab + "}\n"); tab -= 1
                f.write("\t" * tab + "}\n")
                f.write("\t" * tab + "else\n")
                f.write("\t" * tab + "{\n"); tab += 1
            if i == 4 :
                f.write("\t" * tab + "}\n"); tab -= 1
                f.write("\t" * tab + "}\n"); tab -= 1
                f.write("\t" * tab + "}\n")
                f.write("\t" * tab + "else\n")
                f.write("\t" * tab + "{\n"); tab += 1
            if i == 8 :
                f.write("\t" * tab + "}\n"); tab -= 1
                f.write("\t" * tab + "}\n"); tab -= 1
                f.write("\t" * tab + "}\n"); tab -= 1
    else :
        #
        str_frag_a = f"size_{ld_tile_order_a[1]} % TILE_{ld_tile_order_a[1].capitalize()} == 0"
        str_reg_a = f"size_{ld_tile_order_a[0]} % TILE_{ld_tile_order_a[0].capitalize()} == 0"
        str_frag_b = f"size_{ld_tile_order_b[1]} % TILE_{ld_tile_order_b[1].capitalize()} == 0"
        str_reg_b = f"size_{ld_tile_order_b[0]} % TILE_{ld_tile_order_b[0].capitalize()} == 0"
        str_internal = f"size_{ld_tile_order_b[2]} % TILE_{ld_tile_order_b[2].capitalize()} == 0"
        partial_information = tc_interface_Kernel_partial_information(8)

        #
        for i in range(1, 9) :
            if i == 1 :
                f.write(f"\tif({str_frag_b} && {str_frag_a})\n")
                f.write("\t" * tab + "{\n"); tab += 1
            if i == 1 or i == 5 :
                f.write(f"\t\tif({str_reg_b} && {str_reg_a})\n")
                f.write("\t" * tab + "{\n"); tab += 1
            if i % 2 != 0 :
                f.write(f"\t\t\tif({str_internal})\n")
                f.write("\t" * tab + "{\n"); tab += 1
            tc_interface_Kernel_launch_variables(f, kernel_name, l_external_index, l_internal_index, input_a, input_b, i, tab, partial_information[i - 1]); tab -= 1
            if i % 2 != 0 :
                f.write("\t" * tab + "}\n")
                f.write("\t" * tab + "else\n")
                f.write("\t" * tab + "{\n"); tab += 1
            if i == 2 or i == 6 :
                f.write("\t" * tab + "}\n"); tab -= 1
                f.write("\t" * tab + "}\n")
                f.write("\t" * tab + "else\n")
                f.write("\t" * tab + "{\n"); tab += 1
            if i == 4 :
                f.write("\t" * tab + "}\n"); tab -= 1
                f.write("\t" * tab + "}\n"); tab -= 1
                f.write("\t" * tab + "}\n")
                f.write("\t" * tab + "else\n")
                f.write("\t" * tab + "{\n"); tab += 1
            if i == 8 :
                f.write("\t" *  tab + "}\n"); tab -= 1
                f.write("\t" *  tab + "}\n"); tab -= 1
                f.write("\t" *  tab + "}\n"); tab -= 1

    f.write("\n")

    #
    if check_cuda == 1 :
        f.write("\tCHECK_CUDA(cudaDeviceSynchronize());\n\n")
    else :
        f.write("\tcudaDeviceSynchronize();\n\n")

#
def tc_interface_Memcpy_toHost(f, l_external_index, check_cuda) :
    #
    idx_count = 0
    str_size_output = ""
    for each_idx in l_external_index:
        if idx_count == 0:
            str_size_output = "size_" + each_idx
        else:
            str_size_output = str_size_output + " * size_" + each_idx
        idx_count = idx_count + 1
    
    #
    f.write("\t// Copying Result to Host\n")
    if check_cuda == 1 :
        f.write(f"\tCHECK_CUDA(cudaMemcpy(host_t3, dev_t3, sizeof(double) * ({str_size_output}), cudaMemcpyDeviceToHost));\n\n")
    else :
        f.write(f"\tcudaMemcpy(host_t3, dev_t3, sizeof(double) * ({str_size_output}), cudaMemcpyDeviceToHost);\n\n")

#
def tc_interface_CUDA_free(f, l_cuda_malloc, check_cuda) :
    #
    f.write("\t// Freeing Device Memory\n")
    for each_var in l_cuda_malloc:
        if "range"  in each_var[0]:
            continue
        if "addr"   in each_var[0]:
            continue
        if "offset" in each_var[0]:
            continue
        if "base"   in each_var[0]:
            continue
        tc_helper.tc_interface_helper_CUDA_free(f, each_var[0], check_cuda)

#
def tc_interface_time(f, kernel_name, l_external_index, l_internal_index, fvi_flag, input_a, input_b, split_input, check_cuda) :
    #
    t2_split_flag = split_input[0]
    v2_split_flag = split_input[1]

    if fvi_flag == 1 :
        a_split_flag = v2_split_flag
        b_split_flag = t2_split_flag
    else :
        a_split_flag = t2_split_flag
        b_split_flag = v2_split_flag

    idx_count = 0
    str_size_output = ""
    for each_idx in l_external_index:
        if idx_count == 0:
            str_size_output = "size_" + each_idx
        else:
            str_size_output = str_size_output + " * size_" + each_idx
        idx_count = idx_count + 1

    #
    f.write(f"\tCHECK_CUDA(cudaMemset(dev_t3, 0, sizeof(double) * ({str_size_output})));\n")
    f.write("\tCHECK_CUDA(cudaDeviceSynchronize());\n\n")

    #
    f.write("\tint repeat_time = 200;\n")
    f.write("\tint warm_up = 100;\n")
    f.write("\tfloat elapsed_time;\n")
    f.write("\tcudaEvent_t start, stop;\n")

    #
    if check_cuda == 1 :
        f.write("\tCHECK_CUDA(cudaEventCreate(&start));\n")
        f.write("\tCHECK_CUDA(cudaEventCreate(&stop));\n\n")
    else :
        f.write("\tcudaEventCreate(&start);\n")
        f.write("\tcudaEventCreate(&stop);\n")
    
    #
    f.write("\tfor(int i = 0; i < warm_up; i++) {\n")
    f.write("\t\tswitch(dt_flag) {\n")
    tab = 3
    #
    if a_split_flag and b_split_flag :
        for i in range(1, 5) :
            f.write("\t" * tab + f"case {i} :\n"); tab += 1
            tc_interface_Kernel_launch_variables(f, kernel_name, l_external_index, l_internal_index, input_a, input_b, i, tab)
            f.write("\t" * tab + "break;\n"); tab -= 1

        f.write("\t" * tab + "default :\n"); tab += 1
        f.write("\t" * tab + "printf(\"Error: dt_flag is not valid\\n\");\n")
        f.write("\t" * tab + "break;\n")
    else :
        for i in range(1, 9) :
            f.write("\t" * tab + f"case {i} :\n"); tab += 1
            tc_interface_Kernel_launch_variables(f, kernel_name, l_external_index, l_internal_index, input_a, input_b, i, tab)
            f.write("\t" * tab + "break;\n"); tab -= 1
        f.write("\t" * tab + "default :\n"); tab += 1
        f.write("\t" * tab + "printf(\"Error: dt_flag is not valid\\n\");\n")
        f.write("\t" * tab + "break;\n")

    f.write("\t\t}\n")
    f.write("\t}\n")

    #
    if check_cuda == 1 :
        f.write("\tCHECK_CUDA(cudaDeviceSynchronize());\n\n")
    else :
        f.write("\tcudaDeviceSynchronize();\n\n")

    #
    if check_cuda == 1 :
        f.write("\tCHECK_CUDA(cudaEventRecord(start));\n")
    else :
        f.write("\tcudaEventRecord(start);\n")

    #
    f.write(f"\tCHECK_CUDA(cudaMemset(dev_t3, 0, sizeof(double) * ({str_size_output})));\n")
    f.write("\tCHECK_CUDA(cudaDeviceSynchronize());\n\n")
    
    #
    f.write("\tfor(int i = 0; i < repeat_time; i++) {\n")
    f.write("\t\tswitch(dt_flag) {\n")
    
    #
    tab = 3
    if a_split_flag and b_split_flag :
        #
        for i in range(1, 5) :
            f.write("\t" * tab + f"case {i} :\n"); tab += 1
            tc_interface_Kernel_launch_variables(f, kernel_name, l_external_index, l_internal_index, input_a, input_b, i, tab)
            f.write("\t" * tab + "break;\n"); tab -= 1

        #
        f.write("\t" * tab + "default :\n"); tab += 1
        f.write("\t" * tab + "printf(\"Error: dt_flag is not valid\\n\");\n")
        f.write("\t" * tab + "break;\n")
    else :
        #
        for i in range(1, 9) :
            f.write("\t" * tab + f"case {i} :\n"); tab += 1
            tc_interface_Kernel_launch_variables(f, kernel_name, l_external_index, l_internal_index, input_a, input_b, i, tab)
            f.write("\t" * tab + "break;\n"); tab -= 1
        
        #
        f.write("\t" * tab + "default :\n"); tab += 1
        f.write("\t" * tab + "printf(\"Error: dt_flag is not valid\\n\");\n")
        f.write("\t" * tab + "break;\n")

    #
    f.write("\t\t}\n")
    f.write("\t}\n")

    #
    if check_cuda == 1 :
        f.write("\tCHECK_CUDA(cudaDeviceSynchronize());\n")
        f.write("\tCHECK_CUDA(cudaEventRecord(stop));\n")
        f.write("\tCHECK_CUDA(cudaEventSynchronize(start));\n")
        f.write("\tCHECK_CUDA(cudaEventSynchronize(stop));\n")
        f.write("\tCHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));\n\n")
    else :
        f.write("\tcudaEventRecord(stop);\n")
        f.write("\tcudaEventSynchronize(start);\n")
        f.write("\tcudaEventSynchronize(stop);\n")
        f.write("\tcudaEventElapsedTime(&elapsed_time, start, stop);\n\n")

    #
    f.write("\t// Kernel Execution Time\n")
    f.write("\tdouble gflops = (2 * ((double)operations / 1e9) * repeat_time) / (elapsed_time / 1000);\n")
    f.write("\tprintf(\"Time : %f msec, GFLOPS : %f\\n\", elapsed_time / repeat_time, gflops);\n\n")

#
def tc_interface_runner(f, interface_name, kernel_name, l_interface_info,
                        l_external_index, l_internal_index,
                        l_var_output, l_var_input_left, l_var_input_right, l_cuda_malloc, l_cuda_memcpy,
                        size_SM_left, size_SM_right, size_UNIT,
                        fvi_flag, input_a, input_b, internal_order, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, SMEM_order_a, SMEM_order_b,
                        total_y_padding, total_x_padding,
                        split_input, producer_cnt, check_cuda) :
    #
    tc_interface_runner_header(f, interface_name, l_interface_info)

    #
    tc_interface_runner_variables(f, l_external_index, l_var_output, l_var_input_left, l_var_input_right, ld_tile_order_a, ld_tile_order_b, producer_cnt)

    #
    tc_interface_CUDA_malloc_memcpy(f, l_cuda_malloc, l_cuda_memcpy, check_cuda)

    #
    tc_interface_RelatedKernels(f, kernel_name, l_external_index, l_internal_index,
                                size_SM_left, size_SM_right, size_UNIT,
                                fvi_flag, input_a, input_b, internal_order, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b,
                                total_y_padding, total_x_padding,
                                split_input, check_cuda)

    #
    tc_interface_Memcpy_toHost(f, l_external_index, check_cuda)

    #
    tc_interface_time(f, kernel_name, l_external_index, l_internal_index, fvi_flag, input_a, input_b, split_input, check_cuda)

    #
    tc_interface_CUDA_free(f, l_cuda_malloc, check_cuda)

    #
    f.write("}\n")