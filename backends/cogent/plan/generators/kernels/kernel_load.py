import math
import tc_helper

# generate head of __device__ load kernels
def tc_code_kernel_dev_ld_head(f, kernel_name, l_t2_d_decl_var, l_v2_d_decl_var, l_external_index, l_internal_index,
                                fvi_flag, input_a, input_b, internal_order, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b,
                                split_input, producer_cnt, data_type, opt) :
    #
    t2_split_flag = split_input[0]
    v2_split_flag = split_input[1]

    #
    if fvi_flag == 1 :
        a_decl_var = l_v2_d_decl_var
        b_decl_var = l_t2_d_decl_var
        a_split_flag = v2_split_flag
        b_split_flag = t2_split_flag
    else :
        a_decl_var = l_t2_d_decl_var
        b_decl_var = l_v2_d_decl_var
        a_split_flag = t2_split_flag
        b_split_flag = v2_split_flag

    #
    f.write(f"__device__ void {kernel_name}(")

    #
    for a_var in a_decl_var :
        if "addr" in a_var :
            continue
        if "offset" in a_var :
            continue
        f.write(f"{a_var}, ")
    
    #
    for b_var in b_decl_var :
        if "addr" in b_var :
            continue
        if "offset" in b_var :
            continue
        f.write(f"{b_var}, ")

    #
    if data_type == "DOUBLE" :
        f.write(f"double *sm_{input_a}, double *sm_{input_b},\n")
    else :
        f.write(f"float *sm_{input_a}, float *sm_{input_b},\n")

    #
    for each_index in l_external_index :
        f.write(f"int size_{each_index}, ")
    for each_index in l_internal_index :
        f.write(f"int size_{each_index}, ")
    f.write("\n")
    
    #
    for each_index in l_external_index :
        f.write(f"int blk_idx_{each_index}, ")
    f.write("\n")

    #
    if producer_cnt > 0 :
        f.write("cuda::pipeline<cuda::thread_scope_block> &pipeline, auto &producer_warp, cooperative_groups::thread_block_tile<1U, void> &thread,\n")
    else :
        f.write("cuda::pipeline<cuda::thread_scope_block> &pipeline, cooperative_groups::thread_block_tile<1U, void> &thread,\n")

    #
    f.write(f"int shm_{input_a}_offset, ")
    f.write(f"int shm_{input_b}_offset")

    #
    for internal_index in internal_order :
        f.write(f", int iter_{internal_index}")

    #
    # if data_type == "DOUBLE" :
    #     if opt == 1 :
    #         f.write(")\n")
    #     elif opt == 2 :
    #         f.write(", int internal_upperbound)\n")
    #     elif opt == 3 :
    #         f.write(f",\nint rng_{collapsed_a[0]}, int rng_{collapsed_b[0]})\n")
    #     elif opt == 4 :
    #         f.write(",\nint internal_upperbound,\n")
    #         f.write(f"int rng_{collapsed_a[0]}, int rng_{collapsed_b[0]})\n")
    #     elif opt == 5 or opt == 6:
    #         #
    #         if a_split_flag :
    #             tmp_index_a = collapsed_a[0]
    #         else :
    #             tmp_index_a = ld_tile_order_a[1]
    #         if b_split_flag :
    #             tmp_index_b = collapsed_b[0]
    #         else :
    #             tmp_index_b = ld_tile_order_b[1]
            
    #         #
    #         if opt == 5 :
    #             f.write(f",\nint rng_{tmp_index_a}, int rng_{tmp_index_b})\n")
    #         else :
    #             f.write(f",\nint rng_{tmp_index_a}, int rng_{tmp_index_b},\n")
    #             f.write("int internal_upperbound)\n")
    #     elif opt == 7 or opt == 8:
    #         #
    #         if a_split_flag :
    #             tmp_index_a = collapsed_a[0]
    #             tmp_index_b_frag = ld_tile_order_b[1]
    #             tmp_index_b_reg = ld_tile_order_b[0]
    #         elif b_split_flag :
    #             tmp_index_a_frag = ld_tile_order_a[1]
    #             tmp_index_a_reg = ld_tile_order_a[0]
    #             tmp_index_b = collapsed_b[0]
    #         else :
    #             tmp_index_a_frag = ld_tile_order_a[1]
    #             tmp_index_a_reg = ld_tile_order_a[0]
    #             tmp_index_b_frag = ld_tile_order_b[1]
    #             tmp_index_b_reg = ld_tile_order_b[0]

    #         #
    #         if opt == 7 :
    #             if a_split_flag :
    #                 f.write(f",\nint rng_{tmp_index_a}, int rng_{tmp_index_b_frag},\n")
    #                 f.write(f"int rng_{tmp_index_b_reg})\n")
    #             elif b_split_flag :
    #                 f.write(f",\nint rng_{tmp_index_a_frag}, int rng_{tmp_index_b},\n")
    #                 f.write(f"int rng_{tmp_index_a_reg})\n")
    #             else :
    #                 f.write(f",\nint rng_{tmp_index_a_frag}, int rng_{tmp_index_b_frag},\n")
    #                 f.write(f"int rng_{tmp_index_a_reg}, int rng_{tmp_index_b_reg})\n")
    #         else :
    #             if a_split_flag :
    #                 f.write(f",\nint rng_{tmp_index_a}, int rng_{tmp_index_b_frag},\n")
    #                 f.write(f"int internal_upperbound,\n")
    #                 f.write(f"int rng_{tmp_index_b_reg})\n")
    #             elif b_split_flag :
    #                 f.write(f",\nint rng_{tmp_index_a_frag}, int rng_{tmp_index_b},\n")
    #                 f.write(f"int internal_upperbound,\n")
    #                 f.write(f"int rng_{tmp_index_a_reg})\n")
    #             else :
    #                 f.write(f",\nint rng_{tmp_index_a_frag}, int rng_{tmp_index_b_frag},\n")
    #                 f.write(f"int internal_upperbound,\n")
    #                 f.write(f"int rng_{tmp_index_a_reg}, int rng_{tmp_index_b_reg})\n")
    # else :
    if opt == 1 :
        f.write(")\n")
    else :
        f.write(f",\nint size_tensor_{input_a}, int size_tensor_{input_b})\n")

#
def tc_code_kernel_dev_ld_body_memcpy(f, name, opt, tab, data_type, vector_opt) :
    if data_type == "DOUBLE" :
        if vector_opt == 1 :
            if opt == 1 :
                f.write("\t" * tab + f"cuda::memcpy_async(thread, reinterpret_cast<double2*>(&sm_{name}[sm_dst_{name}]), reinterpret_cast<const double2*>(&dev_{name}[sm_src_{name}]), cuda::aligned_size_t<16>{{sizeof(double2)}}, pipeline);\n")
            elif opt == 2 :
                f.write("\t" * tab + f"reinterpret_cast<double2*>(&sm_{name}[sm_dst_{name}])[0] = make_double2(0.0, 0.0);\n")
        else :
            if opt == 1 :
                f.write("\t" * tab + f"cuda::memcpy_async(thread, &sm_{name}[sm_dst_{name}], &dev_{name}[sm_src_{name}], cuda::aligned_size_t<8>{{8}}, pipeline);\n")
            elif opt == 2 :
                f.write("\t" * tab + f"sm_{name}[sm_dst_{name}] = 0.0;\n")
    else :
        if vector_opt == 2 :
            f.write("\t" * tab + f"cuda::memcpy_async(thread, reinterpret_cast<float4*>(&sm_{name}[sm_dst_{name}]), reinterpret_cast<const float4*>(&dev_{name}[sm_src_{name}]), cuda::aligned_size_t<16>{{sizeof(float4)}}, pipeline);\n")
        elif vector_opt == 1 :
            f.write("\t" * tab + f"cuda::memcpy_async(thread, reinterpret_cast<float2*>(&sm_{name}[sm_dst_{name}]), reinterpret_cast<const float2*>(&dev_{name}[sm_src_{name}]), cuda::aligned_size_t<8>{{sizeof(float2)}}, pipeline);\n")
        else :
            f.write("\t" * tab + f"cuda::memcpy_async(thread, reinterpret_cast<float*>(&sm_{name}[sm_dst_{name}]), reinterpret_cast<const float*>(&dev_{name}[sm_src_{name}]), cuda::aligned_size_t<4>{{sizeof(float)}}, pipeline);\n")

#
def tc_code_kernel_dev_ld_body_src_index(iter_stride_a, iter_stride_b, input_a, input_b, input_tensor_a, input_tensor_b,
                                         ld_tile_order_a, ld_tile_order_b, ld_blk_index_a, ld_blk_index_b) :
    #
    r_input_tensor_a = list(reversed(input_tensor_a))
    index_array = []
    size_dict = {}

    #
    for index in r_input_tensor_a :
        size_dict.setdefault(f"size_{index}", [])

    #
    for i in ld_blk_index_a :
        #
        tmp = f"blk_idx_{i} * TILE_{i.capitalize()}"
        
        #
        if i in r_input_tensor_a :
            start = r_input_tensor_a.index(i) + 1
        else :
            start = 0

        #
        if start < len(r_input_tensor_a) :
            for j in r_input_tensor_a[start:] :
                size_dict[f"size_{j}"].append(tmp)
        else :
            index_array.append(tmp)

    #
    for i in ld_tile_order_a :
        #
        axis = ''.join(filter(str.isalpha, i))
        str_num = ''.join(filter(str.isdigit, i)) or '1'
        num = int(str_num)

        #
        if num > 1 :
            tmp = f"{input_a}_{i} * TILE_{axis.capitalize()}{num - 1}"
        else :
            tmp = f"{input_a}_{i}"
        
        #
        idx = r_input_tensor_a.index(axis)
        if idx + 1 < len(r_input_tensor_a) :
            for index in r_input_tensor_a[idx + 1:] :
                size_dict[f"size_{index}"].append(tmp)
        else :
            index_array.append(tmp)

    #
    for index, stride, size in iter_stride_a :
        if size > 1 :
            for idx in stride :
                size_dict[f"{idx}"].append(f"iter_{index}")
        else :
            index_array.append(f"iter_{index}")

    #
    size_dict = {k: v for k, v in size_dict.items() if v}
    keys = list(size_dict.keys())
    for i, (key, value) in enumerate(size_dict.items()) :
        #
        tmp = " + ".join(value)
        
        #
        for next_key in keys[i + 1:] :
            for v in value :
                if v in size_dict[next_key] :
                    size_dict[next_key].remove(v)
            size_dict[next_key].append(f"({tmp}) * {key}")
    tmp = " + ".join(size_dict[keys[-1]])
    str_size_dict = f"({tmp}) * {keys[-1]}"

    #
    str_index_array = " + ".join(index_array)
    
    #
    str_src_sm_a = f"{str_index_array} + {str_size_dict}"

    #
    r_input_tensor_b = list(reversed(input_tensor_b))
    index_array = []
    size_dict = {}
    for index in r_input_tensor_b :
        size_dict.setdefault(f"size_{index}", [])

    #
    for i in ld_blk_index_b :
        #
        tmp = f"blk_idx_{i} * TILE_{i.capitalize()}"
        
        #
        if i in r_input_tensor_b :
            start = r_input_tensor_b.index(i) + 1
        else :
            start = 0
        
        #
        if start < len(r_input_tensor_b) :
            for j in r_input_tensor_b[start:] :
                size_dict[f"size_{j}"].append(tmp)
        else :
            index_array.append(tmp)
    
    #
    for i in ld_tile_order_b :
        #
        axis = ''.join(filter(str.isalpha, i))
        str_num = ''.join(filter(str.isdigit, i)) or '1'
        num = int(str_num)

        #
        if num > 1 :
            tmp = f"{input_b}_{i} * TILE_{axis.capitalize()}{num - 1}"
        else :
            tmp = f"{input_b}_{i}"
        
        idx = r_input_tensor_b.index(axis)
        if idx + 1 < len(r_input_tensor_b) :
            for index in r_input_tensor_b[idx + 1:] :
                size_dict[f"size_{index}"].append(tmp)
        else :
            index_array.append(tmp)
    
    #
    for index, stride, size in iter_stride_b :
        if size > 1 :
            for idx in stride :
                size_dict[f"{idx}"].append(f"iter_{index}")
        else :
            index_array.append(f"iter_{index}")
    
    #
    size_dict = {k: v for k, v in size_dict.items() if v}
    keys = list(size_dict.keys())
    for i, (key, value) in enumerate(size_dict.items()) :
        #
        tmp = " + ".join(value)
        
        #
        for next_key in keys[i + 1:] :
            for v in value :
                if v in size_dict[next_key] :
                    size_dict[next_key].remove(v)
            size_dict[next_key].append(f"({tmp}) * {key}")
    tmp = " + ".join(size_dict[keys[-1]])
    str_size_dict = f"({tmp}) * {keys[-1]}"
    
    #
    str_index_array = " + ".join(index_array)
    
    #
    str_src_sm_b = f"{str_index_array} + {str_size_dict}"

    return str_src_sm_a, str_src_sm_b

# base on the SMEM order, decide padding size
def tc_code_kernel_dev_ld_body_dst_index(l_splited_indices_size, input_a, input_b, ld_tile_order_a, ld_tile_order_b, SMEM_order_a, SMEM_order_b, a_double2_flag, b_double2_flag, data_type) :
    #
    frag_y = ld_tile_order_a[1]
    frag_x = ld_tile_order_b[1]
    internal = ld_tile_order_a[2]
    reg_y = ld_tile_order_a[0]
    reg_x = ld_tile_order_b[0]
    size_frag_y = tc_helper.tc_helper_find_value(l_splited_indices_size, frag_y)
    size_frag_x = tc_helper.tc_helper_find_value(l_splited_indices_size, frag_x)
    size_internal = tc_helper.tc_helper_find_value(l_splited_indices_size, internal)
    size_reg_y = tc_helper.tc_helper_find_value(l_splited_indices_size, reg_y)
    size_reg_x = tc_helper.tc_helper_find_value(l_splited_indices_size, reg_x)

    #
    if data_type == "DOUBLE" :
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
    dst_sm_a = []
    dst_sm_b = []
    
    #
    dst_sm_a.append(f"shm_{input_a}_offset")

    #
    if SMEM_order_a[2] == internal :    # SMEM Order = [Reg_Y, Frag_Y, Internal]
        #
        if a_double2_flag :
            #
            per_row_frag = wavefront_unit // size_internal
            row_cnt = (4 + per_row_frag - 1) // per_row_frag

            #
            dst_sm_a.append(f"({input_a}_{SMEM_order_a[0]} * (TILE_{SMEM_order_a[1].capitalize()} * TILE_{SMEM_order_a[2].capitalize()}))")
            
            #
            xor_ = (int)(math.log2(per_row_frag))
            div = row_cnt - 1

            swizzle = f"((({input_a}_{SMEM_order_a[1]} >> {xor_}) & {div}) << 2)"
            
            #
            dst_sm_a.append(f"({input_a}_{SMEM_order_a[2]} ^ {swizzle})")

            #
            dst_sm_a.append(f"({input_a}_{SMEM_order_a[1]} << {(int)(math.log2(size_internal))})")
        #
        else :
            #
            dst_sm_a.append(f"({input_a}_{SMEM_order_a[0]} * (TILE_{SMEM_order_a[1].capitalize()} * TILE_{SMEM_order_a[2].capitalize()}))")

            #
            if size_frag_y == 8 :
                if size_internal == 8 :
                    dst_sm_a.append(f"(({input_a}_{SMEM_order_a[1]} ^ (({input_a}_{SMEM_order_a[2]} >> 2) << 1)) << 2)")
                else :
                    dst_sm_a.append(f"(({input_a}_{SMEM_order_a[1]} ^ ({input_a}_{SMEM_order_a[2]} >> 2)) << 2)")
            else :
                if size_internal == 8 :
                    dst_sm_a.append(f"((({input_a}_{SMEM_order_a[1]} ^ (({input_a}_{SMEM_order_a[2]} >> 2) << 1)) & 7) << 2)")
                else :
                    dst_sm_a.append(f"((({input_a}_{SMEM_order_a[1]} ^ ({input_a}_{SMEM_order_a[2]} >> 2)) & 7) << 2)")
            
            #
            dst_sm_a.append(f"({input_a}_{SMEM_order_a[2]} & 3)")
            dst_sm_a.append(f"(({input_a}_{SMEM_order_a[2]} >> 2) << 5)")
            
            #
            if size_frag_y == 16 :
                interval_split_8 = 32 * (size_internal // 4)
                dst_sm_a.append(f"(({input_a}_{SMEM_order_a[1]} >> 3) * {interval_split_8})")
    #
    elif SMEM_order_a[2] == frag_y : # SMEM Order = [Reg_Y, Internal, Frag_Y]
        #
        if a_double2_flag :
            #
            per_row_internal = wavefront_unit // size_frag_y
            row_cnt = (4 + per_row_internal - 1) // per_row_internal

            #
            dst_sm_a.append(f"({input_a}_{SMEM_order_a[0]} * (TILE_{SMEM_order_a[1].capitalize()} * TILE_{SMEM_order_a[2].capitalize()}))")
            
            #
            xor_ = (int)(math.log2(per_row_internal))
            div = row_cnt - 1

            swizzle = f"((({input_a}_{SMEM_order_a[1]} >> {xor_}) & {div}) << 2)"
            
            #
            dst_sm_a.append(f"({input_a}_{SMEM_order_a[2]} ^ {swizzle})")

            #
            dst_sm_a.append(f"({input_a}_{SMEM_order_a[1]} << {(int)(math.log2(size_frag_y))})")
        #
        else :
            #
            inner_frag_padd_y = ((8 * size_internal) // 32) * padd_per_wavefront_y
            frag_count_per_y = size_frag_y // 8
            
            #
            if size_frag_y == 16 and padd_per_wavefront_y * (size_internal / 4) % 4 == 0 :
                inter_reg_frag_padd_y = wavefront_unit / 8
            else :
                inter_reg_frag_padd_y = 0
            
            #
            if size_frag_y == 16 :
                interval_split_8 = 32 * (size_internal // 4) + inter_reg_frag_padd_y + (size_internal // 4 - 2)
                
            #
            reg_y_padd = (int)(inner_frag_padd_y * frag_count_per_y + inter_reg_frag_padd_y)

            #
            dst_sm_a.append(f"({input_a}_{SMEM_order_a[0]} * ((TILE_{SMEM_order_a[2].capitalize()} * TILE_{SMEM_order_a[1].capitalize()}) + {reg_y_padd}))")

            #
            if size_frag_y == 8 :
                dst_sm_a.append(f"({input_a}_{SMEM_order_a[2]} << 2)")
            else :
                dst_sm_a.append(f"(({input_a}_{SMEM_order_a[2]} & 7) << 2)")
            
            #
            if size_frag_y == 16 :
                dst_sm_a.append(f"(({input_a}_{SMEM_order_a[2]} >> 3) * {interval_split_8})")
            
            #
            dst_sm_a.append(f"({input_a}_{SMEM_order_a[1]} & 3)")
            dst_sm_a.append(f"(({input_a}_{SMEM_order_a[1]} >> 2) << 5)")
            dst_sm_a.append(f"((({input_a}_{SMEM_order_a[2]} >> 2) + ({input_a}_{SMEM_order_a[1]} >> 2)) * {padd_per_wavefront_y})")
    #
    else :                            # SMEM Order = [Frag_Y, Internal, Reg_Y]
        #
        if a_double2_flag :
            # only for tile size = 8, 16
            per_row_internal = wavefront_unit // size_reg_y
            row_cnt = (4 + per_row_internal - 1) // per_row_internal
            
            #
            if row_cnt == 1 :
                reg_y_padd = 0
            else :
                reg_y_padd = 2 * row_cnt

            #
            dst_sm_a.append(f"({input_a}_{SMEM_order_a[0]} * ((TILE_{SMEM_order_a[1].capitalize()} * TILE_{SMEM_order_a[2].capitalize()}) + {reg_y_padd}))")
            
            #
            xor_ = (int)(math.log2(per_row_internal))
            div = row_cnt - 1

            swizzle = f"((({input_a}_{SMEM_order_a[1]} >> {xor_}) & {div}) << 1)"
            
            #
            dst_sm_a.append(f"({input_a}_{SMEM_order_a[2]} ^ {swizzle})")

            #
            dst_sm_a.append(f"({input_a}_{SMEM_order_a[1]} << {(int)(math.log2(size_reg_y))})")
        #
        else :
            #
            reg_y_padd = wavefront_unit // tc_helper.tc_helper_find_value(l_splited_indices_size, SMEM_order_a[2])

            #
            dst_sm_a.append(f"({input_a}_{SMEM_order_a[2]} * ((TILE_{SMEM_order_a[0].capitalize()} * TILE_{SMEM_order_a[1].capitalize()}) + {reg_y_padd}))")

            #
            if size_frag_y == 8 :
                dst_sm_a.append(f"({input_a}_{SMEM_order_a[0]} << 2)")
            else :
                dst_sm_a.append(f"(({input_a}_{SMEM_order_a[0]} & 7) << 2)")
            
            #
            dst_sm_a.append(f"({input_a}_{SMEM_order_a[1]} & 3)")
            dst_sm_a.append(f"(({input_a}_{SMEM_order_a[1]} >> 2) << 5)")

            #
            if size_frag_y == 16 :
                interval_split_8 = 32 * (size_internal // 4)
                dst_sm_a.append(f"(({input_a}_{SMEM_order_a[0]} >> 3) * {interval_split_8})")

    #
    dst_sm_b.append(f"shm_{input_b}_offset")

    #
    if SMEM_order_b[2] == internal :    # SMEM Order = [Reg_X, Frag_X, Internal]
        #
        if b_double2_flag :
            #
            per_row_frag = wavefront_unit // size_internal
            row_cnt = (4 + per_row_frag - 1) // per_row_frag

            #
            dst_sm_b.append(f"({input_b}_{SMEM_order_b[0]} * (TILE_{SMEM_order_b[1].capitalize()} * TILE_{SMEM_order_b[2].capitalize()}))")
            
            #
            xor_ = (int)(math.log2(per_row_frag))
            div = row_cnt - 1

            swizzle = f"((({input_b}_{SMEM_order_b[1]} >> {xor_}) & {div}) << 2)"
            
            #
            dst_sm_b.append(f"({input_b}_{SMEM_order_b[2]} ^ {swizzle})")

            #
            dst_sm_b.append(f"({input_b}_{SMEM_order_b[1]} << {(int)(math.log2(size_internal))})")
        #
        else :
            #
            dst_sm_b.append(f"({input_b}_{SMEM_order_b[0]} * (TILE_{SMEM_order_b[1].capitalize()} * TILE_{SMEM_order_b[2].capitalize()}))")

            #
            if size_frag_x == 8 :
                if size_internal == 8 :
                    dst_sm_b.append(f"(({input_b}_{SMEM_order_b[1]} ^ (({input_b}_{SMEM_order_b[2]} >> 2) << 1)) << 2)")
                else :
                    dst_sm_b.append(f"(({input_b}_{SMEM_order_b[1]} ^ ({input_b}_{SMEM_order_b[2]} >> 2)) << 2)")
            else :
                if size_internal == 8 :
                    dst_sm_b.append(f"((({input_b}_{SMEM_order_b[1]} ^ (({input_b}_{SMEM_order_b[2]} >> 2) << 1)) & 7) << 2)")
                else :
                    dst_sm_b.append(f"((({input_b}_{SMEM_order_b[1]} ^ ({input_b}_{SMEM_order_b[2]} >> 2)) & 7) << 2)")

            #
            dst_sm_b.append(f"({input_b}_{SMEM_order_b[2]} & 3)")
            dst_sm_b.append(f"(({input_b}_{SMEM_order_b[2]} >> 2) << 5)")
            
            #
            if size_frag_x == 16 :
                interval_split_8 = 32 * (size_internal // 4)
                dst_sm_b.append(f"(({input_b}_{SMEM_order_b[1]} >> 3) * {interval_split_8})")
    #
    elif SMEM_order_b[2] == frag_x :    # SMEM Order = [Reg_X, Internal, Frag_X]
        #
        if b_double2_flag :
            #
            per_row_internal = wavefront_unit // size_frag_x
            row_cnt = (4 + per_row_internal - 1) // per_row_internal

            #
            dst_sm_b.append(f"({input_b}_{SMEM_order_b[0]} * (TILE_{SMEM_order_b[1].capitalize()} * TILE_{SMEM_order_b[2].capitalize()}))")
            
            #
            xor_ = (int)(math.log2(per_row_internal))
            div = row_cnt - 1

            swizzle = f"((({input_b}_{SMEM_order_b[1]} >> {xor_}) & {div}) << 2)"
            
            #
            dst_sm_b.append(f"({input_b}_{SMEM_order_b[2]} ^ {swizzle})")

            #
            dst_sm_b.append(f"({input_b}_{SMEM_order_b[1]} << {(int)(math.log2(size_frag_x))})")
        #
        else :
            inner_frag_padd_x = ((8 * size_internal) // 32) * padd_per_wavefront_x
            frag_count_per_x = size_frag_x // 8

            #
            if size_frag_x == 16 and padd_per_wavefront_x * (size_internal / 4) % 4 == 0 :
                inter_reg_frag_padd_x = wavefront_unit / 8
            else :
                inter_reg_frag_padd_x = 0
            
            #
            if size_frag_x == 16 :
                interval_split_8 = 32 * (size_internal // 4) + inter_reg_frag_padd_x + (size_internal // 4 - 2)

            #
            reg_x_padd = (int)(inner_frag_padd_x * frag_count_per_x + inter_reg_frag_padd_x)

            #
            dst_sm_b.append(f"({input_b}_{SMEM_order_b[0]} * ((TILE_{SMEM_order_b[1].capitalize()} * TILE_{SMEM_order_b[2].capitalize()}) + {reg_x_padd}))")

            #
            if size_frag_x == 8 :
                dst_sm_b.append(f"({input_b}_{SMEM_order_b[2]} << 2)")
            else :
                dst_sm_b.append(f"(({input_b}_{SMEM_order_b[2]} & 7) << 2)")
            
            #
            if size_frag_x == 16 :
                dst_sm_b.append(f"(({input_b}_{SMEM_order_b[2]} >> 3) * {interval_split_8})")
            
            #
            dst_sm_b.append(f"({input_b}_{SMEM_order_b[1]} & 3)")
            dst_sm_b.append(f"(({input_b}_{SMEM_order_b[1]} >> 2) << 5)")
            dst_sm_b.append(f"((({input_b}_{SMEM_order_b[2]} >> 2) + ({input_b}_{SMEM_order_b[1]} >> 2)) * {padd_per_wavefront_x})")
    #
    else :                              # SMEM Order = [Frag_X, Internal, Reg_X]
        #
        if b_double2_flag :            
            # only for tile size = 8, 16
            per_row_internal = wavefront_unit // size_reg_x
            row_cnt = (4 + per_row_internal - 1) // per_row_internal
            
            #
            if row_cnt == 1 :
                reg_x_padd = 0
            else :
                reg_x_padd = 2 * row_cnt

            #
            dst_sm_b.append(f"({input_b}_{SMEM_order_b[0]} * ((TILE_{SMEM_order_b[1].capitalize()} * TILE_{SMEM_order_b[2].capitalize()}) + {reg_x_padd}))")
            
            #
            xor_ = (int)(math.log2(per_row_internal))
            div = row_cnt - 1

            swizzle = f"((({input_b}_{SMEM_order_b[1]} >> {xor_}) & {div}) << 1)"
            
            #
            dst_sm_b.append(f"({input_b}_{SMEM_order_b[2]} ^ {swizzle})")

            #
            dst_sm_b.append(f"({input_b}_{SMEM_order_b[1]} << {(int)(math.log2(size_reg_x))})")
        #
        else :
            reg_x_padd = wavefront_unit // tc_helper.tc_helper_find_value(l_splited_indices_size, SMEM_order_b[2])

            #
            dst_sm_b.append(f"({input_b}_{SMEM_order_b[2]} * ((TILE_{SMEM_order_b[0].capitalize()} * TILE_{SMEM_order_b[1].capitalize()}) + {reg_x_padd}))")

            #
            if size_frag_x == 8 :
                dst_sm_b.append(f"({input_b}_{SMEM_order_b[0]} << 2)")
            else :
                dst_sm_b.append(f"(({input_b}_{SMEM_order_b[0]} & 7) << 2)")
            
            #
            dst_sm_b.append(f"({input_b}_{SMEM_order_b[1]} & 3)")
            dst_sm_b.append(f"(({input_b}_{SMEM_order_b[1]} >> 2) << 5)")

            #
            if size_frag_x == 16 :
                interval_split_8 = 32 * (size_internal // 4)
                dst_sm_b.append(f"(({input_b}_{SMEM_order_b[0]} >> 3) * {interval_split_8})")

    #
    str_dst_sm_a = " + ".join(dst_sm_a)
    str_dst_sm_b = " + ".join(dst_sm_b)
    # print(f"reg_y_padd : {reg_y_padd}, reg_x_padd : {reg_x_padd}", file=sys.stderr)
    #
    reg_padd_y = [reg_y_padd, inner_frag_padd_y, inter_reg_frag_padd_y]
    reg_padd_x = [reg_x_padd, inner_frag_padd_x, inter_reg_frag_padd_x]

    return str_dst_sm_a, str_dst_sm_b, reg_padd_y, reg_padd_x

#
def tc_code_kernel_dev_ld_body_dst_index_fp32(l_splited_indices_size, input_a, input_b, ld_tile_order_a, ld_tile_order_b, SMEM_order_a, SMEM_order_b, 
                                              a_double2_flag, b_double2_flag, data_type) :
    #
    frag_y = ld_tile_order_a[1]
    frag_x = ld_tile_order_b[1]
    internal = ld_tile_order_a[2]
    reg_y = ld_tile_order_a[0]
    reg_x = ld_tile_order_b[0]
    size_frag_y = tc_helper.tc_helper_find_value(l_splited_indices_size, frag_y)
    size_frag_x = tc_helper.tc_helper_find_value(l_splited_indices_size, frag_x)
    size_internal = tc_helper.tc_helper_find_value(l_splited_indices_size, internal)
    size_reg_y = tc_helper.tc_helper_find_value(l_splited_indices_size, reg_y)
    size_reg_x = tc_helper.tc_helper_find_value(l_splited_indices_size, reg_x)

    #
    if data_type == "DOUBLE" :
        wavefront_unit = 16
    else :
        wavefront_unit = 32

    #
    inner_frag_padd_x = 0
    inner_frag_padd_y = 0
    inter_reg_frag_padd_x = 0
    inter_reg_frag_padd_y = 0
    reg_y_padd = 0
    reg_x_padd = 0

    #
    dst_sm_a = []
    dst_sm_b = []
    
    #
    dst_sm_a.append(f"shm_{input_a}_offset")

    #
    if SMEM_order_a[2] == internal :    # SMEM Order = [Reg_Y, Frag_Y, Internal] 
        #
        per_row_frag = wavefront_unit // size_internal
        row_cnt = math.ceil(8 / per_row_frag) # 4 elements per frag -> 8 block is needed

        #
        dst_sm_a.append(f"({input_a}_{SMEM_order_a[0]} * (TILE_{SMEM_order_a[1].capitalize()} * TILE_{SMEM_order_a[2].capitalize()}))")
        
        #
        log_per_row_frag = (int)(math.log2(per_row_frag))
        div = row_cnt - 1

        swizzle = f"((({input_a}_{SMEM_order_a[1]} >> {log_per_row_frag}) & {div}) << 2)"
        
        #
        dst_sm_a.append(f"({input_a}_{SMEM_order_a[2]} ^ {swizzle})")

        #
        dst_sm_a.append(f"({input_a}_{SMEM_order_a[1]} << {(int)(math.log2(size_internal))})")

    #
    elif SMEM_order_a[2] == frag_y : # SMEM Order = [Reg_Y, Internal, Frag_Y]
        #
        per_row_internal = wavefront_unit // size_frag_y
        row_cnt = math.ceil(4 / per_row_internal) # 8 elements per frag -> 4 block is needed

        #
        dst_sm_a.append(f"({input_a}_{SMEM_order_a[0]} * (TILE_{SMEM_order_a[1].capitalize()} * TILE_{SMEM_order_a[2].capitalize()}))")
        
        #
        log_per_row_internal = (int)(math.log2(per_row_internal))
        div = row_cnt - 1

        swizzle = f"((({input_a}_{SMEM_order_a[1]} >> {log_per_row_internal}) & {div}) << 3)"
        
        #
        dst_sm_a.append(f"({input_a}_{SMEM_order_a[2]} ^ {swizzle})")

        #
        dst_sm_a.append(f"({input_a}_{SMEM_order_a[1]} << {(int)(math.log2(size_frag_y))})")
    #
    else :                            # SMEM Order = [Frag_Y, Internal, Reg_Y]
        # only for tile size = 8, 16
        per_row_internal = wavefront_unit // size_reg_y
        row_cnt = math.ceil(4 / per_row_internal)
        vector_size = (int)(2 * a_double2_flag) # double2_flag for float, float = 0, float2 = 1, float4 = 2
        reg_y_padd = row_cnt * vector_size

        #
        dst_sm_a.append(f"({input_a}_{SMEM_order_a[0]} * ((TILE_{SMEM_order_a[1].capitalize()} * TILE_{SMEM_order_a[2].capitalize()}) + {reg_y_padd}))")
        
        #
        log_per_row_internal = (int)(math.log2(per_row_internal))
        
        #
        if vector_size > 0 :
            swizzle = f"((({input_a}_{SMEM_order_a[1]} & 3) >> {log_per_row_internal}) << {(int)(math.log2(vector_size))})"
        else :
            swizzle = f"(({input_a}_{SMEM_order_a[1]} & 3) >> {log_per_row_internal})"
        
        #
        dst_sm_a.append(f"({input_a}_{SMEM_order_a[2]} ^ {swizzle})")

        #
        dst_sm_a.append(f"({input_a}_{SMEM_order_a[1]} << {(int)(math.log2(size_reg_y))})")

    #
    dst_sm_b.append(f"shm_{input_b}_offset")

    #
    if SMEM_order_b[2] == internal :    # SMEM Order = [Reg_X, Frag_X, Internal]
        #
        per_row_frag = wavefront_unit // size_internal
        row_cnt = math.ceil(8 / per_row_frag)

        #
        dst_sm_b.append(f"({input_b}_{SMEM_order_b[0]} * (TILE_{SMEM_order_b[1].capitalize()} * TILE_{SMEM_order_b[2].capitalize()}))")
        
        #
        log_per_row_frag = (int)(math.log2(per_row_frag))
        div = row_cnt - 1

        swizzle = f"((({input_b}_{SMEM_order_b[1]} >> {log_per_row_frag}) & {div}) << 2)"
        
        #
        dst_sm_b.append(f"({input_b}_{SMEM_order_b[2]} ^ {swizzle})")

        #
        dst_sm_b.append(f"({input_b}_{SMEM_order_b[1]} << {(int)(math.log2(size_internal))})")

    #
    elif SMEM_order_b[2] == frag_x :    # SMEM Order = [Reg_X, Internal, Frag_X]
        #
        per_row_internal = wavefront_unit // size_frag_x
        row_cnt = math.ceil(4 / per_row_internal)

        #
        dst_sm_b.append(f"({input_b}_{SMEM_order_b[0]} * (TILE_{SMEM_order_b[1].capitalize()} * TILE_{SMEM_order_b[2].capitalize()}))")
        
        #
        log_per_row_internal = (int)(math.log2(per_row_internal))
        div = row_cnt - 1

        swizzle = f"((({input_b}_{SMEM_order_b[1]} >> {log_per_row_internal}) & {div}) << 3)"
        
        #
        dst_sm_b.append(f"({input_b}_{SMEM_order_b[2]} ^ {swizzle})")

        #
        dst_sm_b.append(f"({input_b}_{SMEM_order_b[1]} << {(int)(math.log2(size_frag_x))})")
    #
    else :                              # SMEM Order = [Frag_X, Internal, Reg_X]
        #
        per_row_internal = wavefront_unit // size_reg_x
        row_cnt = math.ceil(4 / per_row_internal)
        vector_size = (int)(2 * b_double2_flag) # double2_flag for float, float = 0, float2 = 1, float4 = 2
        reg_x_padd = row_cnt * vector_size

        #
        dst_sm_b.append(f"({input_b}_{SMEM_order_b[0]} * ((TILE_{SMEM_order_b[1].capitalize()} * TILE_{SMEM_order_b[2].capitalize()}) + {reg_x_padd}))")
        
        #
        log_per_row_internal = (int)(math.log2(per_row_internal))

        if vector_size > 0 :
            swizzle = f"((({input_b}_{SMEM_order_b[1]} & 3) >> {log_per_row_internal}) << {(int)(math.log2(vector_size))})"
        else :
            swizzle = f"(({input_b}_{SMEM_order_b[1]} & 3) >> {log_per_row_internal})"

        #
        dst_sm_b.append(f"({input_b}_{SMEM_order_b[2]} ^ {swizzle})")

        #
        dst_sm_b.append(f"({input_b}_{SMEM_order_b[1]} << {(int)(math.log2(size_reg_x))})")

    #
    str_dst_sm_a = " + ".join(dst_sm_a)
    str_dst_sm_b = " + ".join(dst_sm_b)
    # print(f"reg_y_padd : {reg_y_padd}, reg_x_padd : {reg_x_padd}", file=sys.stderr)
    #
    reg_padd_y = [reg_y_padd, inner_frag_padd_y, inter_reg_frag_padd_y]
    reg_padd_x = [reg_x_padd, inner_frag_padd_x, inter_reg_frag_padd_x]

    return str_dst_sm_a, str_dst_sm_b, reg_padd_y, reg_padd_x

#
def tc_code_kernel_dev_ld_body_load_a(f, input_a, ld_tile_order_a, collapsed_a, a_split_flag, opt, data_type, vector_opt) :
    #
    internal_partial = f"({input_a}_{ld_tile_order_a[2]} < TILE_UNIT - internal_upperbound)"
    collapsed_partial = f"({input_a}_{ld_tile_order_a[0]} * TILE_{ld_tile_order_a[1].capitalize()} + {input_a}_{ld_tile_order_a[1]} < rng_{collapsed_a[0]})"
    ori_frag_partial = f"({input_a}_{ld_tile_order_a[1]} < rng_{ld_tile_order_a[1]})"
    ori_reg_partial = f"({input_a}_{ld_tile_order_a[0]} < rng_{ld_tile_order_a[0]})"

    #
    partial_condition = []
    if a_split_flag :
        #
        if opt == 2 :
            partial_condition.append(internal_partial)
        elif opt == 3 or opt == 5 or opt == 7 :
            partial_condition.append(collapsed_partial)
        elif opt == 4 or opt == 6 or opt == 8 :
            partial_condition.append(collapsed_partial)
            partial_condition.append(internal_partial)
    else :
        #
        if opt == 2 :
            partial_condition.append(internal_partial)
        elif opt == 3 :
            partial_condition.append(ori_reg_partial)
        elif opt == 4 :
            partial_condition.append(internal_partial)
            partial_condition.append(ori_reg_partial)
        elif opt == 5 :
            partial_condition.append(ori_frag_partial)
        elif opt == 6 :
            partial_condition.append(ori_frag_partial)
            partial_condition.append(internal_partial)
        elif opt == 7 :
            partial_condition.append(ori_frag_partial)
            partial_condition.append(ori_reg_partial)
        elif opt == 8 :
            partial_condition.append(ori_frag_partial)
            partial_condition.append(internal_partial)
            partial_condition.append(ori_reg_partial)

    #
    if vector_opt == 1 :
        if opt == 1 :
            tc_code_kernel_dev_ld_body_memcpy(f, input_a, 1, 2, data_type, 1)
        else :
            str_partial_condition = " && ".join(partial_condition)
            f.write(f"\t\tif({str_partial_condition})\n\t\t" + "{\n")
            tc_code_kernel_dev_ld_body_memcpy(f, input_a, 1, 3, data_type, 1)
            f.write("\t\t}\n")
            f.write("\t\telse\n\t\t{\n")
            tc_code_kernel_dev_ld_body_memcpy(f, input_a, 2, 3, data_type, 1)
            f.write("\t\t}\n")
    else :
        if opt == 1 :
            tc_code_kernel_dev_ld_body_memcpy(f, input_a, 1, 2, data_type, 0)
        else :
            str_partial_condition = " && ".join(partial_condition)
            f.write(f"\t\tif({str_partial_condition})\n\t\t" + "{\n")
            tc_code_kernel_dev_ld_body_memcpy(f, input_a, 1, 3, data_type, 0)
            f.write("\t\t}\n")
            f.write("\t\telse\n\t\t{\n")
            tc_code_kernel_dev_ld_body_memcpy(f, input_a, 2, 3, data_type, 0)
            f.write("\t\t}\n")

#
def tc_code_kernel_dev_ld_body_load_a_fp32(f, input_a, opt, data_type, vector_opt) :
    #
    if opt == 1 :
        tc_code_kernel_dev_ld_body_memcpy(f, input_a, 1, 2, data_type, vector_opt)
    #
    else :
        if vector_opt == 0 :
            f.write(f"\t\tif(sm_src_{input_a} < size_tensor_{input_a})\n\t\t" + "{\n")
        else :
            f.write(f"\t\tif(sm_src_{input_a} + {(vector_opt * 2) - 1} < size_tensor_{input_a})\n\t\t" + "{\n")    
        tc_code_kernel_dev_ld_body_memcpy(f, input_a, 1, 3, data_type, vector_opt)
        f.write("\t\t}\n")

#
def tc_code_kernel_dev_ld_body_load_b(f, input_b, ld_tile_order_b, collapsed_b, b_split_flag, opt, data_type, vector_opt) :
    #
    internal_partial = f"({input_b}_{ld_tile_order_b[2]} < TILE_UNIT - internal_upperbound)"
    collapsed_partial = f"({input_b}_{ld_tile_order_b[0]} * TILE_{ld_tile_order_b[1].capitalize()} + {input_b}_{ld_tile_order_b[1]} < rng_{collapsed_b[0]})"        
    ori_frag_partial = f"({input_b}_{ld_tile_order_b[1]} < rng_{ld_tile_order_b[1]})"
    ori_reg_partial = f"({input_b}_{ld_tile_order_b[0]} < rng_{ld_tile_order_b[0]})"

    #
    partial_condition = []
    if b_split_flag :
        if opt == 2 :
            partial_condition.append(internal_partial)
        elif opt == 3 or opt == 5 or opt == 7 :
            partial_condition.append(collapsed_partial)
        elif opt == 4 or opt == 6 or opt == 8 :
            partial_condition.append(collapsed_partial)
            partial_condition.append(internal_partial)
    else :
        if opt == 2 :
            partial_condition.append(internal_partial)
        elif opt == 3 :
            partial_condition.append(ori_reg_partial)
        elif opt == 4 :
            partial_condition.append(internal_partial)
            partial_condition.append(ori_reg_partial)
        elif opt == 5 :
            partial_condition.append(ori_frag_partial)
        elif opt == 6 :
            partial_condition.append(ori_frag_partial)
            partial_condition.append(internal_partial)
        elif opt == 7 :
            partial_condition.append(ori_frag_partial)
            partial_condition.append(ori_reg_partial)
        elif opt == 8 :
            partial_condition.append(ori_frag_partial)
            partial_condition.append(internal_partial)
            partial_condition.append(ori_reg_partial)

    #
    if vector_opt == 1 :
        if opt == 1 :
            tc_code_kernel_dev_ld_body_memcpy(f, input_b, 1, 2, data_type, 1)
        else :
            str_partial_condition = " && ".join(partial_condition)
            f.write(f"\t\tif({str_partial_condition})\n\t\t" + "{\n")
            tc_code_kernel_dev_ld_body_memcpy(f, input_b, 1, 3, data_type, 1)
            f.write("\t\t}\n")
            f.write("\t\telse\n\t\t{\n")
            tc_code_kernel_dev_ld_body_memcpy(f, input_b, 2, 3, data_type, 1)
            f.write("\t\t}\n")
    else :
        if opt == 1 :
            tc_code_kernel_dev_ld_body_memcpy(f, input_b, 1, 2, data_type, 0)
        else :
            str_partial_condition = " && ".join(partial_condition)
            f.write(f"\t\tif({str_partial_condition})\n\t\t" + "{\n")
            tc_code_kernel_dev_ld_body_memcpy(f, input_b, 1, 3, data_type, 0)
            f.write("\t\t}\n")
            f.write("\t\telse\n\t\t{\n")
            tc_code_kernel_dev_ld_body_memcpy(f, input_b, 2, 3, data_type, 0)
            f.write("\t\t}\n")

#           
def tc_code_kernel_dev_ld_body_load_b_fp32(f, input_b, opt, data_type, vector_opt) :
    #
    if opt == 1 :
        tc_code_kernel_dev_ld_body_memcpy(f, input_b, 1, 2, data_type, vector_opt)
    #
    else :
        if vector_opt == 0 :
            f.write(f"\t\tif(sm_src_{input_b} < size_tensor_{input_b})\n\t\t" + "{\n")
        else :
            f.write(f"\t\tif(sm_src_{input_b} + {(vector_opt * 2) - 1} < size_tensor_{input_b})\n\t\t" + "{\n")    
        tc_code_kernel_dev_ld_body_memcpy(f, input_b, 1, 3, data_type, vector_opt)
        f.write("\t\t}\n")      


# generate body of __device__ load kernel
def tc_code_kernel_dev_ld_body(f, l_input_strides, l_splited_indices_size,
                                fvi_flag, input_a, input_b, input_tensor_a, input_tensor_b,
                                ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, ld_blk_index_a, ld_blk_index_b, SMEM_order_a, SMEM_order_b,
                                split_input, producer_cnt, double2_flag, data_type, opt) :
    #
    f.write("{\n")

    #
    t2_split_flag = split_input[0]
    v2_split_flag = split_input[1]

    #
    if fvi_flag == 1 :
        iter_stride_a = l_input_strides[1]
        iter_stride_b = l_input_strides[0]
        a_split_flag = v2_split_flag
        b_split_flag = t2_split_flag
    elif fvi_flag == 2 :
        iter_stride_a = l_input_strides[0]
        iter_stride_b = l_input_strides[1]
        a_split_flag = t2_split_flag
        b_split_flag = v2_split_flag

    #
    a_double2_flag = double2_flag[1]
    b_double2_flag = double2_flag[0]

    #
    str_src_sm_a, str_src_sm_b = tc_code_kernel_dev_ld_body_src_index(iter_stride_a, iter_stride_b, 
                                                                    input_a, input_b, input_tensor_a, input_tensor_b,
                                                                    ld_tile_order_a, ld_tile_order_b, ld_blk_index_a, ld_blk_index_b)
    
    #
    if data_type == "DOUBLE" :
        str_dst_sm_a, str_dst_sm_b, reg_padd_y, reg_padd_x = tc_code_kernel_dev_ld_body_dst_index(l_splited_indices_size,
                                                                                input_a, input_b, ld_tile_order_a, ld_tile_order_b, SMEM_order_a, SMEM_order_b,
                                                                                a_double2_flag, b_double2_flag, data_type)
    else :
        str_dst_sm_a, str_dst_sm_b, reg_padd_y, reg_padd_x = tc_code_kernel_dev_ld_body_dst_index_fp32(l_splited_indices_size,
                                                                                input_a, input_b, ld_tile_order_a, ld_tile_order_b, SMEM_order_a, SMEM_order_b,
                                                                                a_double2_flag, b_double2_flag, data_type)
        
    #
    if producer_cnt > 0 :
        f.write("\tconst int lane = producer_warp.thread_rank();\n")
        f.write("\tconst int producer_rank = ((threadIdx.x >> 5) << 5) + lane;\n")
        f.write("\tconst int producer_size = (PRODUCER_CNT << 5);\n\n")
    
    #
    if a_double2_flag == 2 :
        a_str_idx = "(threadIdx.x << 2)"
        a_stride = "(blockDim.x << 2)"
    #
    elif a_double2_flag == 1 :
        a_str_idx = "(threadIdx.x << 1)"
        a_stride = "(blockDim.x << 1)"
    #
    else :
        a_str_idx = "threadIdx.x"
        a_stride = "blockDim.x"

    #    
    f.write(f"\tfor(int idx = {a_str_idx}; idx < {input_a}_elements; idx += {a_stride})\n")
    f.write("\t{\n")

    #
    f.write(f"\t\tint {input_a}_{SMEM_order_a[0]} = idx / plane_{input_a};\n")
    f.write(f"\t\tint rem_{input_a} = idx - {input_a}_{SMEM_order_a[0]} * plane_{input_a};\n")
    f.write(f"\t\tint {input_a}_{SMEM_order_a[1]} = rem_{input_a} / TILE_{SMEM_order_a[2].capitalize()};\n")
    f.write(f"\t\tint {input_a}_{SMEM_order_a[2]} = rem_{input_a} - {input_a}_{SMEM_order_a[1]} * TILE_{SMEM_order_a[2].capitalize()};\n\n")

    #
    f.write(f"\t\tint sm_src_{input_a} = {str_src_sm_a};\n\n")

    #
    f.write(f"\t\tint sm_dst_{input_a} = {str_dst_sm_a};\n\n")

    #
    # if data_type == "DOUBLE" :
    #     tc_code_kernel_dev_ld_body_load_a(f, input_a, ld_tile_order_a, collapsed_a, a_split_flag, opt, data_type, a_double2_flag)
    # else :
    #     tc_code_kernel_dev_ld_body_load_a_fp32(f, input_a, opt, data_type, a_double2_flag)

    tc_code_kernel_dev_ld_body_load_a_fp32(f, input_a, opt, data_type, a_double2_flag)

    #
    f.write("\t}\n\n")

    #
    if b_double2_flag == 2 :
        b_str_idx = "(threadIdx.x << 2)"
        b_stride = "(blockDim.x << 2)"
    #
    elif b_double2_flag == 1 :
        b_str_idx = "(threadIdx.x << 1)"
        b_stride = "(blockDim.x << 1)"
    #
    else :
        b_str_idx = "threadIdx.x"
        b_stride = "blockDim.x"

    #
    f.write(f"\tfor(int idx = {b_str_idx}; idx < {input_b}_elements; idx += {b_stride})\n")
    f.write("\t{\n")

    #
    f.write(f"\t\tint {input_b}_{SMEM_order_b[0]} = idx / plane_{input_b};\n")
    f.write(f"\t\tint rem_{input_b} = idx - {input_b}_{SMEM_order_b[0]} * plane_{input_b};\n")
    f.write(f"\t\tint {input_b}_{SMEM_order_b[1]} = rem_{input_b} / TILE_{SMEM_order_b[2].capitalize()};\n")
    f.write(f"\t\tint {input_b}_{SMEM_order_b[2]} = rem_{input_b} - {input_b}_{SMEM_order_b[1]} * TILE_{SMEM_order_b[2].capitalize()};\n\n")

    #
    f.write(f"\t\tint sm_src_{input_b} = {str_src_sm_b};\n\n")

    #
    f.write(f"\t\tint sm_dst_{input_b} = {str_dst_sm_b};\n\n")

    #
    # if data_type == "DOUBLE" :
    #     tc_code_kernel_dev_ld_body_load_b(f, input_b, ld_tile_order_b, collapsed_b, b_split_flag, opt, data_type, b_double2_flag)
    # else :
    #     tc_code_kernel_dev_ld_body_load_b_fp32(f, input_b, opt, data_type, b_double2_flag)
   
    tc_code_kernel_dev_ld_body_load_b_fp32(f, input_b, opt, data_type, b_double2_flag)

    #
    f.write("\t}\n")
    
    #
    f.write("}\n\n")

    return reg_padd_y, reg_padd_x

# generate __device__ load kernel
def tc_code_kernel_dev_ld(f, kernel_name, l_t2_d_decl_var, l_v2_d_decl_var,
                        l_external_index, l_internal_index, l_input_strides, l_splited_indices_size,
                        fvi_flag, input_a, input_b, input_tensor_a, input_tensor_b, internal_order,
                        ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, ld_blk_index_a, ld_blk_index_b, SMEM_order_a, SMEM_order_b,
                        split_input, producer_cnt, double2_flag, kernel_variants, data_type) :    
    #
    for i in range(1, kernel_variants + 1) :
        #
        kernel_name_i = kernel_name + "_" + str(i)
        
        #
        tc_code_kernel_dev_ld_head(f, kernel_name_i, l_t2_d_decl_var, l_v2_d_decl_var, l_external_index, l_internal_index,
                                    fvi_flag, input_a, input_b, internal_order, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b,
                                    split_input, producer_cnt, data_type, i)
        
        #
        reg_padd_y, reg_padd_x = tc_code_kernel_dev_ld_body(f, l_input_strides, l_splited_indices_size,
                                                        fvi_flag, input_a, input_b, input_tensor_a, input_tensor_b,
                                                        ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, ld_blk_index_a, ld_blk_index_b, SMEM_order_a, SMEM_order_b,
                                                        split_input, producer_cnt, double2_flag, data_type, i)

    return reg_padd_y, reg_padd_x