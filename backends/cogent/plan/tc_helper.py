#
def ceil(a, b) :
    return (a + b - 1) // b

#
def tc_helper_kernel_variant_info(flag: bool) -> str :
    return 'Full' if flag else 'Partial'

#
def tc_helper_find_index(list, index):
    for temp in list:
        if temp == index:
            return temp
    return -1

#
def tc_helper_find_value(list, index):
    for temp in list:
        if temp[0] == index:
            return temp[1]
    return -1

#
def tc_interface_helper_CUDA_malloc(f, var, type, size, check_cuda) :
    if check_cuda == 1 :
        f.write(f"\tCHECK_CUDA(cudaMalloc((void**)&{var}, sizeof({type}) * {size}));\n")
    else :
        f.write(f"\tcudaMalloc((void**)&{var}, sizeof({type}) * {size});\n")

#
def tc_interface_helper_CUDA_memcpy(f, dest, src, type, size, option, check_cuda):
    if option == 1:
        memcpy_opt = "cudaMemcpyHostToDevice"
    else:
        memcpy_opt = "cudaMemcpyDeviceToHost"

    if check_cuda == 1 :
        f.write(f"\tCHECK_CUDA(cudaMemcpy({dest}, {src}, sizeof({type}) * {size}, {memcpy_opt}));\n")
    else :
        f.write(f"\tcudaMemcpy({dest}, {src}, sizeof({type}) * {size}, {memcpy_opt});\n")

#
def tc_interface_helper_CUDA_free(f, var, check_cuda):
    if check_cuda == 1 :
        f.write(f"\tCHECK_CUDA(cudaFree({var}));\n")
    else :
        f.write(f"\tcudaFree({var});\n")

#
def tc_interface_helper_strides(l_input_tensors, l_tiles_size, str_eq_num, fvi_flag, ld_index_a, ld_index_b) :
    #
    for each_input in l_input_tensors :
        #
        if fvi_flag == 1 :
            left = each_input[1]
            right = each_input[0]
        else :
            left = each_input[0]
            right = each_input[1]
        
        #
        tmp_a = []
        cnt = 0
        iter_a = ld_index_a[-1]
        for each_index in left[4] :
            if each_index != iter_a :
                tmp_a.append(each_index)
            else :
                break
            # if cnt < 2 :
            #     if tc_helper_find_value(l_tile_sizes, each_index) != 1 :
            #         tmp_a.append(each_index)
            #         cnt += 1
            #     else :
            #         tmp_a.append(each_index)
        str_ld_stride_a = ""
        str_ld_stride_size_a = ""
        cnt = 0
        for idx in tmp_a :
            if cnt == 0 :
                str_ld_stride_a = "TCCG_" + str_eq_num + "_TILE_" + idx.capitalize()
                str_ld_stride_size_a = "size_" + idx
            else :
                str_ld_stride_a = str_ld_stride_a + " * TCCG_" + str_eq_num + "_TILE_" + idx.capitalize()
                str_ld_stride_size_a = str_ld_stride_size_a + " * size_" + idx
            # cnt += 1
            if cnt == 0 :
                ld_stride_a = tc_helper_find_value(l_tiles_size, idx)
            else :
                ld_stride_a = ld_stride_a * tc_helper_find_value(l_tiles_size, idx)
            cnt += 1

        #
        tmp_b = []
        cnt = 0
        iter_b = ld_index_b[-1]
        for each_index in right[4] :
            if each_index != iter_b :
                tmp_b.append(each_index)
            else : 
                break
            # if cnt < 2 :
            #     print("each_index", each_index)
            #     if tc_helper_find_value(l_tile_sizes, each_index) != 1 :
            #         tmp_b.append(each_index)
            #         cnt += 1
            #     else :
            #         tmp_b.append(each_index)
        str_ld_stride_b = ""
        str_ld_stride_size_b = ""
        cnt = 0
        for idx in tmp_b :
            if cnt == 0 :
                str_ld_stride_b = "TCCG_" + str_eq_num + "_TILE_" + idx.capitalize()
                str_ld_stride_size_b = "size_" + idx
            else :
                str_ld_stride_b = str_ld_stride_b + " * TCCG_" + str_eq_num + "_TILE_" + idx.capitalize()
                str_ld_stride_size_b = str_ld_stride_size_b + " * size_" + idx
            # cnt += 1
            if cnt == 0 :
                ld_stride_b = tc_helper_find_value(l_tiles_size, idx)
            else :
                ld_stride_b = ld_stride_a * tc_helper_find_value(l_tiles_size, idx)
            cnt += 1

    return str_ld_stride_a, ld_stride_a, str_ld_stride_b, ld_stride_b

# define left and right tensor based on FVI 
def tc_code_kernel_helper_fvi(l_external_index, l_inputs_addr) :
    #    
    fvi_flag = 0
    for each_index in l_inputs_addr[0][0][4] :
        if each_index == l_external_index[0] :
            fvi_flag = 1        # fvi is in t2
            break
    
    #
    if fvi_flag != 1 :
        for each_index in l_inputs_addr[0][1][4] :
            if each_index == l_external_index[0] :
                fvi_flag = 2    # fiv is in v2
                break
    
    #
    if fvi_flag == 1 :
        input_a = l_inputs_addr[0][1][3]
        input_b = l_inputs_addr[0][0][3]
        input_tensor_a = list(l_inputs_addr[0][1][4])
        input_tensor_b = list(l_inputs_addr[0][0][4])
    elif fvi_flag == 2 :
        input_a = l_inputs_addr[0][0][3]
        input_b = l_inputs_addr[0][1][3]
        input_tensor_a = list(l_inputs_addr[0][0][4])
        input_tensor_b = list(l_inputs_addr[0][1][4])

    return fvi_flag, input_a, input_b, input_tensor_a, input_tensor_b

# find index which has non-unit size tile
def tc_code_kernel_helper_ld_index(l_t3_slices, l_external_index, l_reg_mapping, input_tensor_a, input_tensor_b) :
    # Non-unit size indices
    ld_index_a = []
    ld_index_b = []

    # input indices except internal indices
    ld_blk_index_a = []
    ld_blk_index_b = []
    
    #
    for i in range(0, len(input_tensor_a)) :
        #
        if tc_helper_find_value(l_t3_slices, input_tensor_a[i]) != 1 :
            ld_index_a.append(input_tensor_a[i])
        else :
            if input_tensor_a[i] in l_reg_mapping :
                ld_index_a.append(input_tensor_a[i])

        #
        if input_tensor_a[i] in l_external_index :
            ld_blk_index_a.append(input_tensor_a[i])
    
    #
    for i in range(0, len(input_tensor_b)) :
        #
        if tc_helper_find_value(l_t3_slices, input_tensor_b[i]) != 1 :
            ld_index_b.append(input_tensor_b[i])
        else :
            if input_tensor_b[i] in l_reg_mapping :
                ld_index_b.append(input_tensor_b[i])
        
        #
        if input_tensor_b[i] in l_external_index :
            ld_blk_index_b.append(input_tensor_b[i])
    
    #
    r_ld_blk_index_a = list(reversed(ld_blk_index_a))
    r_ld_blk_index_b = list(reversed(ld_blk_index_b))

    return ld_index_a, ld_index_b, r_ld_blk_index_a, r_ld_blk_index_b

# split index
def tc_code_kernel_helper_split_ld_index(ld_index_a, ld_index_b, split_index) :
    #
    split_ld_index_a = []
    for index in ld_index_a :
        #
        if index in split_index :
            split_ld_index_a.append(f"{index}1")
            split_ld_index_a.append(f"{index}2")
        else :
            split_ld_index_a.append(index)

    #
    split_ld_index_b = []
    for index in ld_index_b :
        #
        if index in split_index :
            split_ld_index_b.append(f"{index}1")
            split_ld_index_b.append(f"{index}2")
        else :
            split_ld_index_b.append(index)

    return split_ld_index_a, split_ld_index_b

# decide shared memory structure and index order
def tc_code_kernel_helper_SMEM_order(l_tb_mapping, l_reg_mapping, l_internal_index, split_ld_index_a, split_ld_index_b) :
    #
    SMEM_order_a = []
    if split_ld_index_a[0] == l_tb_mapping[1][0] :
        # REG_Y
        SMEM_order_a.append(l_reg_mapping[1])
        # Contraction index
        for internal_index in l_internal_index :
            if internal_index in split_ld_index_a :
                SMEM_order_a.append(internal_index)
                break
        # FRAG_Y
        SMEM_order_a.append(l_tb_mapping[1][0])
    elif split_ld_index_a[0] == l_reg_mapping[1][0] :
        # FRAG_Y
        SMEM_order_a.append(l_tb_mapping[1][0])
        # Contraction index
        for internal_index in l_internal_index :
            if internal_index in split_ld_index_a :
                SMEM_order_a.append(internal_index)
                break
        # REG_Y
        SMEM_order_a.append(l_reg_mapping[1])
    else :
        # REG_Y
        SMEM_order_a.append(l_reg_mapping[1])
        # FRAG_Y
        SMEM_order_a.append(l_tb_mapping[1][0])
        # Contraction index
        for internal_index in l_internal_index :
            if internal_index in split_ld_index_a :
                SMEM_order_a.append(internal_index)
                break
    
    #
    SMEM_order_b = []
    if split_ld_index_b[0] == l_tb_mapping[0][0] :
        # REG_X
        SMEM_order_b.append(l_reg_mapping[0])
        # Contraction index
        for internal_index in l_internal_index :
            if internal_index in split_ld_index_b :
                SMEM_order_b.append(internal_index)
                break
        # FRAG_X
        SMEM_order_b.append(l_tb_mapping[0][0])
    elif split_ld_index_b[0] == l_reg_mapping[0][0] :
        # FRAG_X
        SMEM_order_b.append(l_tb_mapping[0][0])
        # Contraction index
        for internal_index in l_internal_index :
            if internal_index in split_ld_index_b :
                SMEM_order_b.append(internal_index)
                break
        # REG_X
        SMEM_order_b.append(l_reg_mapping[0])
    else :
        # REG_X
        SMEM_order_b.append(l_reg_mapping[0])
        # FRAG_X
        SMEM_order_b.append(l_tb_mapping[0][0])
        # Contraction index
        for internal_index in l_internal_index :
            if internal_index in split_ld_index_b :
                SMEM_order_b.append(internal_index)
                break

    return SMEM_order_a, SMEM_order_b


# rearrange indices in a predetermined order 
def tc_code_kernel_helper_tile_order(l_tb_mapping, l_reg_mapping, l_internal_index, ld_index_a, ld_index_b, split_ld_index_a, split_ld_index_b, split_index) :
    #
    SMEM_order_a, SMEM_order_b = tc_code_kernel_helper_SMEM_order(l_tb_mapping, l_reg_mapping, l_internal_index, split_ld_index_a, split_ld_index_b)

    #
    ld_tile_order_a = []
    ld_tile_order_b = []
    collapsed_a = []
    collapsed_b = []
    seen_split_index_a = set()
    seen_split_index_b = set()

    # input which contains FVI, always mapped to X
    # append reg_mapped index
    ld_tile_order_a.append(l_reg_mapping[1])
    ld_tile_order_b.append(l_reg_mapping[0])
    
    # append frag_mapped index
    ld_tile_order_a.append(l_tb_mapping[1][0])
    ld_tile_order_b.append(l_tb_mapping[0][0])

    # append contraction index
    for internal_index in l_internal_index :
        if internal_index in ld_index_a :
            ld_tile_order_a.append(internal_index)
        if internal_index in ld_index_b :
            ld_tile_order_b.append(internal_index)

    #
    for index in ld_tile_order_a :
        #
        base = ''.join(filter(str.isalpha, index))
        
        #
        if base in split_index :
            if base not in seen_split_index_a :
                seen_split_index_a.add(base)
                collapsed_a.append(base)
        else :
            collapsed_a.append(index)
    
    #
    for index in ld_tile_order_b :
        #
        base = ''.join(filter(str.isalpha, index))
        
        #
        if base in split_index :
            if base not in seen_split_index_b :
                seen_split_index_b.add(base)
                collapsed_b.append(base)
        else :
            collapsed_b.append(index)

    return SMEM_order_a, SMEM_order_b, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b

#
def tc_code_helper_iteration_min_stride(strides) :
    others = [s for s in strides if s != 1]
    return min(others) if others else 0

#
def tc_code_kernel_helper_iteration_order(l_internal_index, l_splited_indices_size) :
    #
    tmp_internal = []
    tmp_ones = []

    #
    for each_idx in l_internal_index :
        #
        tile_size = tc_helper_find_value(l_splited_indices_size, each_idx)
        
        #
        if tile_size != 1 :
            tmp_internal.append(each_idx)
        else :
            tmp_ones.append(each_idx)

    #
    internal_order = tmp_internal + tmp_ones

    return internal_order

#
def tc_kerenl_helper_make_strides(ld_tile_order_a, ld_tile_order_b, stride_helper, split_index) :
    #
    inter_m_stride = []
    for index in stride_helper :
        if index != ld_tile_order_a[0] :
            inter_m_stride.append(index)
        elif index == ld_tile_order_a[0] :
            break

    #
    inter_n_stride = []
    for index in stride_helper :
        if index != ld_tile_order_b[0] :
            inter_n_stride.append(index)
        elif index == ld_tile_order_b[0] :
            break
    
    #
    intra_stride = []
    for index in stride_helper :
        if index != ld_tile_order_a[1] :
            intra_stride.append(index)
        elif index == ld_tile_order_a[1] :
            break

    #
    collapse_m = []
    for index in split_index :
        if f"{index}1" in inter_m_stride and f"{index}2" in inter_m_stride :
            collapse_m.append(index)
    
    #
    collapse_n = []
    for index in split_index :
        if f"{index}1" in inter_n_stride and f"{index}2" in inter_n_stride :
            collapse_n.append(index)
    
    #
    collapse_intra = []
    for index in split_index :
        if f"{index}1" in intra_stride and f"{index}2" in intra_stride :
            collapse_intra.append(index)

    #
    merged_splited_index_m = []
    seen_m = set()
    for index in inter_m_stride :
        base = ''.join(filter(str.isalpha, index))
        if base in collapse_m :
            if base not in seen_m :
                merged_splited_index_m.append(base)
                seen_m.add(base)
        else :
            merged_splited_index_m.append(index)
    
    #
    merged_splited_index_n = []
    seen_n = set()
    for index in inter_n_stride :
        base = ''.join(filter(str.isalpha, index))
        if base in collapse_n :
            if base not in seen_n :
                merged_splited_index_n.append(base)
                seen_n.add(base)
        else :
            merged_splited_index_n.append(index)

    #
    merged_splited_index_intra = []
    seen_intra = set()
    for index in intra_stride :
        base = ''.join(filter(str.isalpha, index))
        if base in collapse_intra :
            if base not in seen_intra :
                merged_splited_index_intra.append(base)
                seen_intra.add(base)
        else :
            merged_splited_index_intra.append(index)

    #
    formatted_m_stride = []
    for index in merged_splited_index_m :
        if index.isalpha() :
            formatted_m_stride.append(f"size_{index}")
        else :
            formatted_m_stride.append(f"TILE_{index.capitalize()}")
    
    #
    formatted_n_stride = []
    for index in merged_splited_index_n :
        if index.isalpha() :
            formatted_n_stride.append(f"size_{index}")
        else :
            formatted_n_stride.append(f"TILE_{index.capitalize()}")

    #
    formatted_intra_stride = []
    for index in merged_splited_index_intra :
        if index.isalpha() :
            formatted_intra_stride.append(f"size_{index}")
        else :
            formatted_intra_stride.append(f"TILE_{index.capitalize()}")

    #
    str_inter_m_stride = ""
    for i in range(0, len(formatted_m_stride)) :
        if i == 0 :
            str_inter_m_stride = formatted_m_stride[i]
        else :
            str_inter_m_stride = formatted_m_stride[i] + " * " + str_inter_m_stride

    #
    str_inter_n_stride = ""
    for i in range(0, len(formatted_n_stride)) :
        if i == 0 :
            str_inter_n_stride = formatted_n_stride[i]
        else :
            str_inter_n_stride = formatted_n_stride[i] + " * " + str_inter_n_stride

    #
    str_intra_stride = ""
    for i in range(0, len(formatted_intra_stride)) :
        if i == 0 :
            str_intra_stride = formatted_intra_stride[i]
        else :
            str_intra_stride = formatted_intra_stride[i] + " * " + str_intra_stride

    return str_inter_m_stride, str_inter_n_stride, str_intra_stride