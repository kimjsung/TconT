import math
import sys
import tc_helper as tc_helper

# generate head of __global__ kernel
def tc_code_kernel_head(f, kernel_name, l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var,
                        l_external_index, l_internal_index, fvi_flag, input_a, input_b, opt_pre_computed) :
    #
    f.write("extern \"C\" __global__ void ")
    f.write(kernel_name)
    f.write("(")

    #
    if fvi_flag == 1 :
        a_decl_var = l_v2_d_decl_var
        b_decl_var = l_t2_d_decl_var
    else :
        a_decl_var = l_t2_d_decl_var
        b_decl_var = l_v2_d_decl_var

    #
    for t3_var in l_t3_d_decl_var :
        if opt_pre_computed == -1 :
            if "addr" in t3_var :
                continue
            if "offset" in t3_var :
                continue
            f.write(t3_var)
            f.write(", ")
        else :
            f.write(t3_var)
            f.write(", ")

    #
    for t2_var in a_decl_var :
        if opt_pre_computed == -1 :
            if "addr" in t2_var :
                continue
            if "offset" in t2_var :
                continue
            f.write(t2_var)
            f.write(", ")
        else :
            f.write(t2_var)
            f.write(", ")

    #
    for v2_var in b_decl_var :
        if opt_pre_computed == -1 :
            if "addr" in v2_var :
                continue
            if "offset" in v2_var :
                continue
            f.write(v2_var)
            f.write(", ")
        else :
            f.write(v2_var)
            f.write(", ")
    f.write("\n")

    #
    for each_index in l_external_index :
        f.write("int size_" + each_index + ", ")
    
    #
    for each_index in l_internal_index :
        f.write("int size_" + each_index + ", ")
    f.write("\n")

    #
    for each_index in l_external_index :
        f.write("int numBlk_" + each_index + ", ")
    f.write("\n")
    
    #
    f.write("int size_internal, ")

    #
    f.write(f"int {input_a}_tile_size, ")
    f.write(f"int {input_b}_tile_size)\n")

# initialize various variables
def tc_code_kernel_initial(f, l_external_index,
                            fvi_flag, input_a, input_b, internal_order,
                            ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b,
                            split_index, split_input, stride_helper, warp_shape, producer_cnt,
                            data_type, opt) :
    #
    t2_split_flag = split_input[0]
    v2_split_flag = split_input[1]

    #
    if fvi_flag == 1 :
        a_split_flag = v2_split_flag
        b_split_flag = t2_split_flag
    else :
        a_split_flag = t2_split_flag
        b_split_flag = v2_split_flag
    
    #
    f.write("{\n")

    # cooperative groups and pipeline
    f.write("\t// For Pipeline\n")
    f.write("\tauto thread = cooperative_groups::this_thread();\n")
    f.write("\tauto block = cooperative_groups::this_thread_block();\n")

    #
    if producer_cnt > 0 :
        f.write("\tauto producer_warp = cooperative_groups::tiled_partition<32>(block);\n")
        f.write("\tconst int warp_id = threadIdx.x >> 5;\n")
        f.write("\tconst bool is_producer = (warp_id < PRODUCER_CNT);\n")
        f.write("\tconst cuda::pipeline_role thread_role = (is_producer) ? cuda::pipeline_role::producer : cuda::pipeline_role::consumer;\n")
    
    f.write("\n")

    #
    f.write("\t#pragma nv_diag_suppress static_var_with_dynamic_init\n")
    f.write("\t__shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, PIPELINE_STAGES> shared_state;\n")
    
    #
    if producer_cnt > 0 :
        f.write("\tauto pipeline = cuda::make_pipeline(block, &shared_state, thread_role);\n")
    else :
        f.write("\tauto pipeline = cuda::make_pipeline(block, &shared_state);\n\n")
    
    f.write("\n")

    # shared memory
    f.write("\t// For Shared Memory\n")
    if data_type == "DOUBLE" :
        f.write("\textern __shared__ __align__(128) double shared_memory[];\n")
        f.write(f"\tdouble *sm_{input_a} = shared_memory;\n")
        f.write(f"\tdouble *sm_{input_b} = shared_memory + (PIPELINE_STAGES * {input_a}_tile_size);\n\n")
    else :
        pass

    # index for thread blocks
    f.write("\t// Index for each Thread Blocks\n")
    f.write("\tint tmp_blkIdx;\n")
    
    #
    rev_l_external_index = reversed(l_external_index)
    len_l_external_index = len(l_external_index)
    index_count = len_l_external_index
    for each_index in rev_l_external_index :
        tmp_str = ""
        for each_num_index in range(0, index_count - 1) :
            if each_num_index ==  0:
                tmp_str = f"numBlk_{l_external_index[each_num_index]}"
            else :
                tmp_str = f"numBlk_{l_external_index[each_num_index]} * {tmp_str}"
        
        if index_count == len_l_external_index :
            f.write(f"\tconst int blk_idx_{each_index} = blockIdx.x / ({tmp_str});\n")
            f.write(f"\ttmp_blkIdx = blockIdx.x % ({tmp_str});\n")
        else :
            if index_count == 1 :
                f.write(f"\tconst int blk_idx_{each_index} = tmp_blkIdx;\n")
            else :
                f.write(f"\tconst int blk_idx_{each_index} = tmp_blkIdx / ({tmp_str});\n")
                f.write(f"\ttmp_blkIdx = tmp_blkIdx % ({tmp_str});\n")

        f.write("\n")
        index_count = index_count - 1

    # index for warp tile
    f.write("\t// Warp related variables\n")
    f.write(f"\tconst int wmiter = TILE_{ld_tile_order_a[0].capitalize()} / {warp_shape[1]};\n")
    f.write(f"\tconst int wniter = TILE_{ld_tile_order_b[0].capitalize()} / {warp_shape[0]};\n")

    #
    if producer_cnt > 0 :
        f.write("\tconst int consumer_warp_id = warp_id - PRODUCER_CNT;\n")
        f.write(f"\tconst int wrow = (consumer_warp_id >= 0) ? ((consumer_warp_id / {warp_shape[0]}) * wmiter) : 0;\n")
        f.write(f"\tconst int wcol = (consumer_warp_id >= 0) ? ((consumer_warp_id % {warp_shape[0]}) * wniter) : 0;\n\n")
    else :
        f.write(f"\tconst int wrow = ((threadIdx.x >> 5) / {warp_shape[0]}) * wmiter;\n")
        f.write(f"\tconst int wcol = ((threadIdx.x >> 5) % {warp_shape[0]}) * wniter;\n\n")

    f.write("\t// Fragment Allocation\n")
    f.write(f"\tconst int {input_a}_frag_cnt = TILE_{ld_tile_order_a[1].capitalize()} >> 3;\n")
    f.write(f"\tconst int {input_b}_frag_cnt = TILE_{ld_tile_order_b[1].capitalize()} >> 3;\n\n")

    # fragment stride
    str_inter_m_stride, str_inter_n_stride, str_intra_stride = tc_helper.tc_kerenl_helper_make_strides(ld_tile_order_a, ld_tile_order_b, stride_helper, split_index)

    #
    f.write("\t// Size of fragment stride\n")
    f.write(f"\tconst int inter_m_stride = {str_inter_m_stride};\n")
    f.write(f"\tconst int inter_n_stride = {str_inter_n_stride};\n")
    f.write(f"\tconst int intra_stride = {str_intra_stride};\n")
    f.write("\n")

    # base index of t3 for each thread blocks
    str_t3_base = ""
    for i, tile in enumerate(reversed(l_external_index)) :
        tmp = f"blk_idx_{tile} * TILE_{tile.capitalize()}"
        
        if i != 0 :
            line = f"({tmp} + {str_t3_base})"
        else :
            line = f"({tmp})"
    
        if i + 1 < len(l_external_index) :
            str_t3_base = f"{line} * size_{list(reversed(l_external_index))[i + 1]}"
        else :
            str_t3_base = line

    #
    f.write("\t// Base index of each warps for result tensor\n")
    f.write(f"\tconst int t3_base_thread = {str_t3_base}\n")
    f.write(f"\t\t\t\t\t\t\t\t+ wrow * inter_m_stride + wcol * inter_n_stride;\n\n")

    # declare WMMA fragment
    f.write("\t// WMMA fragment\n")
    f.write(f"\tnvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> t3_frag[wmiter * wniter * {input_a}_frag_cnt * {input_b}_frag_cnt];\n\n")

    # fill fragment
    tab = 1
    f.write("\t// Fill fragment\n")
    if producer_cnt > 0 :
        f.write("\t" * tab + "if(thread_role == cuda::pipeline_role::consumer)\n")
        f.write("\t" * tab + "{\n"); tab += 1

    #   
    f.write("\t" * tab + "#pragma unroll\n")
    f.write("\t" * tab + "for(int i = 0; i < wmiter * wniter * v2_frag_cnt * t2_frag_cnt; i++)\n")
    f.write("\t" * tab + "{\n"); tab += 1
    f.write("\t" * tab + "nvcuda::wmma::fill_fragment(t3_frag[i], 0.0);\n"); tab -= 1
    
    #
    for i in range(tab) :
        f.write("\t" * tab + "}\n"); tab -= 1
    
    f.write("\n")

    # share memory pipeline offset
    f.write("\t// Shared memory offset for pipelining\n")
    f.write("\tint shm_offset = 0;\n")
    
    #
    f.write(f"\n\t// Kernel {opt}\n")
    internal = tc_helper.tc_helper_kernel_variant_info(opt % 2 == 1)
    if split_input[0] == 1 and split_input[1] == 1 :
        external = tc_helper.tc_helper_kernel_variant_info(opt <= 2)
        f.write(f"\t// External : {external} / Internal : {internal}\n")
    else :
        frag = tc_helper.tc_helper_kernel_variant_info(opt <= 4)
        reg = tc_helper.tc_helper_kernel_variant_info(((opt - 1) // 2) % 2 == 0)
        f.write(f"\t// Frag_mapped : {frag}, Reg_mapped : {reg} / Internal : {internal}\n")

    #
    if opt % 2 == 0 :
        f.write("\tint internal_upperbound = 0;\n")
        f.write("\tint internal_offset;\n")
        if len(internal_order) > 1 :
            f.write("\tint iter_" + internal_order[0] + "_offset = 0;\n")
    
    #
    reg_partial_index = []
    if a_split_flag :
        reg_partial_index.append(collapsed_a[0])
    else :
        reg_partial_index.append(ld_tile_order_a[0])
    if b_split_flag :
        reg_partial_index.append(collapsed_b[0])
    else :
        reg_partial_index.append(ld_tile_order_b[0])

    #
    frag_partial_index = []
    if a_split_flag :
        frag_partial_index.append(collapsed_a[0])
    else :
        frag_partial_index.append(ld_tile_order_a[1])
    if b_split_flag :
        frag_partial_index.append(collapsed_b[0])
    else :
        frag_partial_index.append(ld_tile_order_b[1])
    
    #
    if opt == 1 or opt == 2:
        f.write("\n")
    elif (opt == 3) or (opt == 4) or (opt == 7) or (opt == 8) :
        f.write("\tint " + ", ".join(f"rng_{index}" for index in reg_partial_index) + ";\n")

        for index in reg_partial_index :
            capital_index = index.capitalize()
            tile = f"TILE_{capital_index}"
            f.write(f"""\tif((size_{index} - (blk_idx_{index} * {tile})) >= {tile})
    {{
        rng_{index} = {tile};
    }}
    else
    {{
        rng_{index} = size_{index} % {tile};
    }}""")
            f.write("\n\n")
    
    if opt > 4 :
        if opt == 7 or opt == 8 :
            f.write("\tint " + ", ".join(f"rng_{index}" for index in frag_partial_index if index not in reg_partial_index) + ";\n")
        else :
            f.write("\tint " + ", ".join(f"rng_{index}" for index in frag_partial_index) + ";\n")
        
        if opt == 7 or opt == 8 :
            for index in frag_partial_index :
                if index not in reg_partial_index :
                    capital_index = index.capitalize()
                    tile = f"TILE_{capital_index}"
                    f.write(f"""\tif((size_{index} - (blk_idx_{index} * {tile})) >= {tile})
    {{
        rng_{index} = {tile};
    }}
    else
    {{
        rng_{index} = size_{index} % {tile};
    }}""")
                    f.write("\n\n")
        else :
            for index in frag_partial_index :
                capital_index = index.capitalize()
                tile = f"TILE_{capital_index}"
                f.write(f"""\tif((size_{index} - (blk_idx_{index} * {tile})) >= {tile})
    {{
        rng_{index} = {tile};
    }}
    else
    {{
        rng_{index} = size_{index} % {tile};
    }}""")
                f.write("\n\n")

# generate head of __device__ load kernels
def tc_code_kernel_dev_ld_head(f, kernel_name, l_t2_d_decl_var, l_v2_d_decl_var, l_external_index, l_internal_index,
                                fvi_flag, input_a, input_b, internal_order, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b,
                                split_input, producer_cnt, opt) :
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
    f.write(f"double *sm_{input_a}, double *sm_{input_b},\n")

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
    if opt == 1 :
        f.write(")\n")
    elif opt == 2 :
        f.write(", int internal_upperbound)\n")
    elif opt == 3 :
        f.write(f",\nint rng_{collapsed_a[0]}, int rng_{collapsed_b[0]})\n")
    elif opt == 4 :
        f.write(",\nint internal_upperbound,\n")
        f.write(f"int rng_{collapsed_a[0]}, int rng_{collapsed_b[0]})\n")
    elif opt == 5 or opt == 6:
        #
        if a_split_flag :
            tmp_index_a = collapsed_a[0]
        else :
            tmp_index_a = ld_tile_order_a[1]
        if b_split_flag :
            tmp_index_b = collapsed_b[0]
        else :
            tmp_index_b = ld_tile_order_b[1]
        
        #
        if opt == 5 :
            f.write(f",\nint rng_{tmp_index_a}, int rng_{tmp_index_b})\n")
        else :
            f.write(f",\nint rng_{tmp_index_a}, int rng_{tmp_index_b},\n")
            f.write("int internal_upperbound)\n")
    elif opt == 7 or opt == 8:
        #
        if a_split_flag :
            tmp_index_a = collapsed_a[0]
            tmp_index_b_frag = ld_tile_order_b[1]
            tmp_index_b_reg = ld_tile_order_b[0]
        elif b_split_flag :
            tmp_index_a_frag = ld_tile_order_a[1]
            tmp_index_a_reg = ld_tile_order_a[0]
            tmp_index_b = collapsed_b[0]
        else :
            tmp_index_a_frag = ld_tile_order_a[1]
            tmp_index_a_reg = ld_tile_order_a[0]
            tmp_index_b_frag = ld_tile_order_b[1]
            tmp_index_b_reg = ld_tile_order_b[0]

        #
        if opt == 7 :
            if a_split_flag :
                f.write(f",\nint rng_{tmp_index_a}, int rng_{tmp_index_b_frag},\n")
                f.write(f"int rng_{tmp_index_b_reg})\n")
            elif b_split_flag :
                f.write(f",\nint rng_{tmp_index_a_frag}, int rng_{tmp_index_b},\n")
                f.write(f"int rng_{tmp_index_a_reg})\n")
            else :
                f.write(f",\nint rng_{tmp_index_a_frag}, int rng_{tmp_index_b_frag},\n")
                f.write(f"int rng_{tmp_index_a_reg}, int rng_{tmp_index_b_reg})\n")
        else :
            if a_split_flag :
                f.write(f",\nint rng_{tmp_index_a}, int rng_{tmp_index_b_frag},\n")
                f.write(f"int internal_upperbound,\n")
                f.write(f"int rng_{tmp_index_b_reg})\n")
            elif b_split_flag :
                f.write(f",\nint rng_{tmp_index_a_frag}, int rng_{tmp_index_b},\n")
                f.write(f"int internal_upperbound,\n")
                f.write(f"int rng_{tmp_index_a_reg})\n")
            else :
                f.write(f",\nint rng_{tmp_index_a_frag}, int rng_{tmp_index_b_frag},\n")
                f.write(f"int internal_upperbound,\n")
                f.write(f"int rng_{tmp_index_a_reg}, int rng_{tmp_index_b_reg})\n")

#
def tc_code_kernel_dev_ld_body_memcpy(f, name, opt, tab, vector_opt) :
    if  vector_opt == 1 :
        if opt == 1 :
            f.write("\t" * tab + f"cuda::memcpy_async(thread, reinterpret_cast<double2*>(&sm_{name}[sm_dst_{name}]), reinterpret_cast<const double2*>(&dev_{name}[sm_src_{name}]), cuda::aligned_size_t<16>{{sizeof(double2)}}, pipeline);\n")
        elif opt == 2 :
            f.write("\t" * tab + f"reinterpret_cast<double2*>(&sm_{name}[sm_dst_{name}])[0] = make_double2(0.0, 0.0);\n")
    else :
        if opt == 1 :
            f.write("\t" * tab + f"cuda::memcpy_async(thread, &sm_{name}[sm_dst_{name}], &dev_{name}[sm_src_{name}], cuda::aligned_size_t<8>{{8}}, pipeline);\n")
        elif opt == 2 :
            f.write("\t" * tab + f"sm_{name}[sm_dst_{name}] = 0.0;\n")

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
    # print(f"kernel_body : SMEM order a : {SMEM_order_a}, SMEM order b : {SMEM_order_b}", file=sys.stderr)
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
def tc_code_kernel_dev_ld_body_load_a(f, input_a, ld_tile_order_a, collapsed_a, a_split_flag, opt, vector_opt) :
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
            tc_code_kernel_dev_ld_body_memcpy(f, input_a, 1, 2, 1)
        else :
            str_partial_condition = " && ".join(partial_condition)
            f.write(f"\t\tif({str_partial_condition})\n\t\t" + "{\n")
            tc_code_kernel_dev_ld_body_memcpy(f, input_a, 1, 3, 1)
            f.write("\t\t}\n")
            f.write("\t\telse\n\t\t{\n")
            tc_code_kernel_dev_ld_body_memcpy(f, input_a, 2, 3, 1)
            f.write("\t\t}\n")
    else :
        if opt == 1 :
            tc_code_kernel_dev_ld_body_memcpy(f, input_a, 1, 2, 0)
        else :
            str_partial_condition = " && ".join(partial_condition)
            f.write(f"\t\tif({str_partial_condition})\n\t\t" + "{\n")
            tc_code_kernel_dev_ld_body_memcpy(f, input_a, 1, 3, 0)
            f.write("\t\t}\n")
            f.write("\t\telse\n\t\t{\n")
            tc_code_kernel_dev_ld_body_memcpy(f, input_a, 2, 3, 0)
            f.write("\t\t}\n")

#
def tc_code_kernel_dev_ld_body_load_b(f, input_b, ld_tile_order_b, collapsed_b, b_split_flag, opt, vector_opt) :
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
            tc_code_kernel_dev_ld_body_memcpy(f, input_b, 1, 2, 1)
        else :
            str_partial_condition = " && ".join(partial_condition)
            f.write(f"\t\tif({str_partial_condition})\n\t\t" + "{\n")
            tc_code_kernel_dev_ld_body_memcpy(f, input_b, 1, 3, 1)
            f.write("\t\t}\n")
            f.write("\t\telse\n\t\t{\n")
            tc_code_kernel_dev_ld_body_memcpy(f, input_b, 2, 3, 1)
            f.write("\t\t}\n")
    else :
        if opt == 1 :
            tc_code_kernel_dev_ld_body_memcpy(f, input_b, 1, 2, 0)
        else :
            str_partial_condition = " && ".join(partial_condition)
            f.write(f"\t\tif({str_partial_condition})\n\t\t" + "{\n")
            tc_code_kernel_dev_ld_body_memcpy(f, input_b, 1, 3, 0)
            f.write("\t\t}\n")
            f.write("\t\telse\n\t\t{\n")
            tc_code_kernel_dev_ld_body_memcpy(f, input_b, 2, 3, 0)
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
    str_dst_sm_a, str_dst_sm_b, reg_padd_y, reg_padd_x = tc_code_kernel_dev_ld_body_dst_index(l_splited_indices_size,
                                                                                            input_a, input_b, ld_tile_order_a, ld_tile_order_b, SMEM_order_a, SMEM_order_b,
                                                                                            a_double2_flag, b_double2_flag, data_type)
    
    #
    if producer_cnt > 0 :
        f.write("\tconst int lane = producer_warp.thread_rank();\n")
        f.write("\tconst int producer_rank = ((threadIdx.x >> 5) << 5) + lane;\n")
        f.write("\tconst int producer_size = (PRODUCER_CNT << 5);\n\n")
    
    #
    if a_double2_flag :
        if producer_cnt > 0 :
            a_str_idx = "(producer_rank << 1)"
            a_stride = "(producer_size << 1)"
        else :
            a_str_idx = "(threadIdx.x << 1)"
            a_stride = "(blockDim.x << 1)"
    else :
        if producer_cnt > 0 :
            a_str_idx = "producer_rank"
            a_stride = "producer_size"
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
    if a_double2_flag :
        tc_code_kernel_dev_ld_body_load_a(f, input_a, ld_tile_order_a, collapsed_a, a_split_flag, opt, 1)
    else :
        tc_code_kernel_dev_ld_body_load_a(f, input_a, ld_tile_order_a, collapsed_a, a_split_flag, opt, 0)

    #
    f.write("\t}\n\n")

    #
    #
    if b_double2_flag :
        if producer_cnt > 0 :
            b_str_idx = "(producer_rank << 1)"
            b_stride = "(producer_size << 1)"
        else :
            b_str_idx = "(threadIdx.x << 1)"
            b_stride = "(blockDim.x << 1)"
    else :
        if producer_cnt > 0 :
            b_str_idx = "producer_rank"
            b_stride = "producer_size"
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
    if b_double2_flag :
        tc_code_kernel_dev_ld_body_load_b(f, input_b, ld_tile_order_b, collapsed_b, b_split_flag, opt, 1)
    else :
        tc_code_kernel_dev_ld_body_load_b(f, input_b, ld_tile_order_b, collapsed_b, b_split_flag, opt, 0)
    
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
                                    split_input, producer_cnt, i)
        
        #
        reg_padd_y, reg_padd_x = tc_code_kernel_dev_ld_body(f, l_input_strides, l_splited_indices_size,
                                                        fvi_flag, input_a, input_b, input_tensor_a, input_tensor_b,
                                                        ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, ld_blk_index_a, ld_blk_index_b, SMEM_order_a, SMEM_order_b,
                                                        split_input, producer_cnt, double2_flag, data_type, i)

    return reg_padd_y, reg_padd_x

#
def tc_code_kernel_dev_compute_head(f, kernel_name, input_a, input_b, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, split_input, fvi_flag, opt) :
    #
    f.write(f"__device__ void {kernel_name}(")

    #
    f.write(f"double *__restrict sm_{input_a}, double *__restrict sm_{input_b}, const int wmiter, const int wniter, const int wrow, const int wcol,\n")
    f.write(f"const int {input_a}_frag_cnt, const int {input_b}_frag_cnt, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> *t3_frag,\n")
    f.write(f"const int shm_{input_a}_offset, const int shm_{input_b}_offset")

    #
    if fvi_flag == 1 :
        a_split_flag = split_input[1]
        b_split_flag = split_input[0]
    elif fvi_flag == 2 :
        a_split_flag = split_input[0]
        b_split_flag = split_input[1]

    #
    if opt == 1 :
        f.write(")\n")
    elif opt == 2 :
        f.write(",\n")
        f.write("int internal_upperbound)\n")
    elif opt == 3 :
        f.write(",\n")
        f.write(f"int rng_{collapsed_a[0]}, int rng_{collapsed_b[0]})\n")
    elif opt == 4 :
        f.write(",\n")
        f.write("int internal_upperbound,\n")
        f.write(f"int rng_{collapsed_a[0]}, int rng_{collapsed_b[0]})\n")
    elif opt == 5 or opt == 6:
        #
        f.write(",\n")
        
        #
        if a_split_flag :
            tmp_index_a = collapsed_a[0]
            tmp_index_b = ld_tile_order_b[1]
        elif b_split_flag :
            tmp_index_a = ld_tile_order_a[1]
            tmp_index_b = collapsed_b[0]
        else :
            tmp_index_a = ld_tile_order_a[1]
            tmp_index_b = ld_tile_order_b[1]
        
        #
        if opt == 5 :
            f.write(f"int rng_{tmp_index_a}, int rng_{tmp_index_b})\n")
        else :
            f.write(f"int rng_{tmp_index_a}, int rng_{tmp_index_b},\n")
            f.write("int internal_upperbound)\n")
    elif opt == 7 or opt == 8:
        #
        f.write(",\n")
        
        #
        if a_split_flag :
            tmp_index_a = collapsed_a[0]
            tmp_index_b_reg = ld_tile_order_b[0]
            tmp_index_b_frag = ld_tile_order_b[1]
        elif b_split_flag :
            tmp_index_a_reg = ld_tile_order_a[0]
            tmp_index_a_frag = ld_tile_order_a[1]
            tmp_index_b = collapsed_b[0]
        else :
            tmp_index_a_reg = ld_tile_order_a[0]
            tmp_index_a_frag = ld_tile_order_a[1]
            tmp_index_b_reg = ld_tile_order_b[0]
            tmp_index_b_frag = ld_tile_order_b[1]
        
        #
        if opt == 7 :
            #
            if a_split_flag :
                f.write(f"int rng_{tmp_index_a}, int rng_{tmp_index_b_frag},\n")
                f.write(f"int rng_{tmp_index_b_reg})\n")
            elif b_split_flag :
                f.write(f"int rng_{tmp_index_a_frag}, int rng_{tmp_index_b},\n")
                f.write(f"int rng_{tmp_index_a_reg})\n")
            else :
                f.write(f"int rng_{tmp_index_a_frag}, int rng_{tmp_index_b_frag},\n")
                f.write(f"int rng_{tmp_index_a_reg}, int rng_{tmp_index_b_reg})\n")
        elif opt == 8 :
            #
            if a_split_flag :
                f.write(f"int rng_{tmp_index_a}, int rng_{tmp_index_b_frag},\n")
                f.write("int internal_upperbound,\n")
                f.write(f"int rng_{tmp_index_b_reg})\n")
            elif b_split_flag :
                f.write(f"int rng_{tmp_index_a_frag}, int rng_{tmp_index_b},\n")
                f.write("int internal_upperbound,\n")
                f.write(f"int rng_{tmp_index_a_reg})\n")
            else :
                f.write(f"int rng_{tmp_index_a_frag}, int rng_{tmp_index_b_frag},\n")
                f.write("int internal_upperbound,\n")
                f.write(f"int rng_{tmp_index_a_reg}, int rng_{tmp_index_b_reg})\n")

#
def tc_code_kernel_dev_compute_body(f, l_splited_indices_size, input_a, input_b, ld_tile_order_a, ld_tile_order_b, SMEM_order_a, SMEM_order_b,
                                    warp_shape, double2_flag, reg_padd_y, reg_padd_x, fvi_flag, opt) :
    #
    # if fvi_flag == 1 :
    #     a_double2_flag = double2_flag[0]
    #     b_double2_flag = double2_flag[1]
    # elif fvi_flag == 2 :
    #     a_double2_flag = double2_flag[1]
    #     b_double2_flag = double2_flag[0]
    a_double2_flag = double2_flag[1]
    b_double2_flag = double2_flag[0]
    
    #
    left_frag_size = tc_helper.tc_helper_find_value(l_splited_indices_size, ld_tile_order_a[1])
    right_frag_size = tc_helper.tc_helper_find_value(l_splited_indices_size, ld_tile_order_b[1])
    size_internal = tc_helper.tc_helper_find_value(l_splited_indices_size, ld_tile_order_a[2])
    left_reg_size = tc_helper.tc_helper_find_value(l_splited_indices_size, ld_tile_order_a[0])
    right_reg_size = tc_helper.tc_helper_find_value(l_splited_indices_size, ld_tile_order_b[0])

    #
    right_reg_per_warp = tc_helper.tc_helper_find_value(l_splited_indices_size, ld_tile_order_b[0]) // warp_shape[0]

    #
    f.write("{\n")

    #
    if a_double2_flag :
        if reg_padd_y[0] == 0 :
            f.write(f"\tconst int ld_stride_a = TILE_{SMEM_order_a[1].capitalize()} * TILE_{SMEM_order_a[2].capitalize()}" + ";\n")
        else :
            f.write(f"\tconst int ld_stride_a = TILE_{SMEM_order_a[1].capitalize()} * TILE_{SMEM_order_a[2].capitalize()} + {reg_padd_y[0]}" + ";\n")
    else :
        if reg_padd_y[0] == 0 :
            f.write(f"\tconst int ld_stride_a = TILE_{ld_tile_order_a[1].capitalize()} * TILE_{ld_tile_order_a[2].capitalize()}" + ";\n")
        else :
            f.write(f"\tconst int ld_stride_a = TILE_{ld_tile_order_a[1].capitalize()} * TILE_{ld_tile_order_a[2].capitalize()} + {reg_padd_y[0]}" + ";\n")
    
    #
    if b_double2_flag :
        if reg_padd_x[0] == 0 :
            f.write(f"\tconst int ld_stride_b = TILE_{SMEM_order_b[1].capitalize()} * TILE_{SMEM_order_b[2].capitalize()}" + ";\n\n")
        else :
            f.write(f"\tconst int ld_stride_b = TILE_{SMEM_order_b[1].capitalize()} * TILE_{SMEM_order_b[2].capitalize()} + {reg_padd_x[0]}" + ";\n\n")
    else :
        if reg_padd_x[0] == 0 :
            f.write(f"\tconst int ld_stride_b = TILE_{ld_tile_order_b[1].capitalize()} * TILE_{ld_tile_order_b[2].capitalize()}" + ";\n\n")
        else :
            f.write(f"\tconst int ld_stride_b = TILE_{ld_tile_order_b[1].capitalize()} * TILE_{ld_tile_order_b[2].capitalize()} + {reg_padd_x[0]}" + ";\n\n")
    
    #
    if a_double2_flag and (SMEM_order_a[2] == ld_tile_order_a[0]) :
        f.write(f"\tnvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, double, nvcuda::wmma::row_major> {input_a}_frag[2];\n")
    else :
        f.write(f"\tnvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, double, nvcuda::wmma::row_major> {input_a}_frag;\n")

    #
    if b_double2_flag and (SMEM_order_b[2] == ld_tile_order_b[0]) :
        f.write(f"\tnvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, double, nvcuda::wmma::row_major> {input_b}_frag[2];\n\n")
    else :
        f.write(f"\tnvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, double, nvcuda::wmma::row_major> {input_b}_frag;\n\n")
    #
    f.write(f"\tconst int lane = threadIdx.x & 31;\n")

    #
    if a_double2_flag :
        #
        if SMEM_order_a[2] == ld_tile_order_a[0] :      # SMEM Order = [Frag_Y, Internal, Reg_Y]
            #
            per_row_internal = 16 // left_reg_size
            shift = (int)(math.log2(per_row_internal))
            mod = tc_helper.ceil(4, per_row_internal) - 1

            #
            f.write(f"\tconst int {input_a}_lane_offset = ((lane >> 2) * ld_stride_a);\n")
            f.write(f"\tconst int {input_a}_fragment_offset = ((lane & 3) << {(int)(math.log2(left_reg_size))}) + (((lane >> {shift}) & {mod}) << 1);\n")
        #
        elif SMEM_order_a[2] == ld_tile_order_a[1] :    # SMEM Order = [Reg_Y, Internal, Frag_Y]
            per_row_internal = 16 // left_frag_size
            row_cnt = tc_helper.ceil(4, per_row_internal)
            shift1 = (int)(math.log2(row_cnt))
            shift2 = (int)(math.log2(per_row_internal))

            f.write(f"\tconst int {input_a}_lane_offset = ((lane >> 2) & 3);\n")
            f.write(f"\tconst int {input_a}_fragment_offset = ((((lane & 3) << {shift1}) ^ ((lane >> {shift2}) & {row_cnt - 1})) ^ (lane >> 4));\n")
        #
        elif SMEM_order_a[2] == ld_tile_order_a[2] :    # SMEM Order = [Reg_Y, Frag_Y, Internal]
            per_row_frag = 16 // size_internal
            row_cnt = tc_helper.ceil(4, per_row_frag)
            per_row_thread = 16 // row_cnt
            shift1 = (int)(math.log2(row_cnt))
            shift2 = (int)(math.log2(per_row_thread))
            
            f.write(f"\tconst int {input_a}_lane_offset = (lane & 3);\n")
            f.write(f"\tconst int {input_a}_fragment_offset = (((lane >> 2) << {shift1}) ^ ((lane >> {shift2}) & {row_cnt - 1}));\n")
    #
    else :
        #
        if SMEM_order_a[2] == ld_tile_order_a[0] :
            f.write(f"\tconst int {input_a}_fragment_offset = lane;\n")
        elif SMEM_order_a[2] == ld_tile_order_a[1] :
            if left_frag_size == 8 :
                f.write(f"\tconst int {input_a}_fragment_offset = lane + ((lane >> 4) << 1);\n")
            else :
                f.write(f"\tconst int {input_a}_fragment_offset = lane + (lane >> 4);\n")
        elif SMEM_order_a[2] == ld_tile_order_a[2] :
            f.write(f"\tconst int {input_a}_fragment_offset = lane;\n")

    #
    if b_double2_flag :
        #
        if SMEM_order_b[2] == ld_tile_order_b[0] :
            #
            per_row_internal = 16 // right_reg_size
            shift = (int)(math.log2(per_row_internal))
            mod = tc_helper.ceil(4, per_row_internal) - 1

            #
            f.write(f"\tconst int {input_b}_lane_offset = ((lane >> 2) * ld_stride_b);\n")
            f.write(f"\tconst int {input_b}_fragment_offset = ((lane & 3) << {(int)(math.log2(right_reg_size))}) + (((lane >> {shift}) & {mod}) << 1);\n")
        #
        elif SMEM_order_b[2] == ld_tile_order_b[1] :
            #
            per_row_internal = 16 // right_frag_size
            row_cnt = tc_helper.ceil(4, per_row_internal)
            shift1 = (int)(math.log2(row_cnt))
            shift2 = (int)(math.log2(per_row_internal))

            #
            f.write(f"\tconst int {input_b}_lane_offset = ((lane >> 2) & 3);\n")
            f.write(f"\tconst int {input_b}_fragment_offset = ((((lane & 3) << {shift1}) ^ ((lane >> {shift2}) & {row_cnt - 1})) ^ (lane >> 4));\n")
        #
        elif SMEM_order_b[2] == ld_tile_order_b[2] :
            #
            per_row_frag = 16 // size_internal
            row_cnt = tc_helper.ceil(4, per_row_frag)
            per_row_thread = 16 // row_cnt
            shift1 = (int)(math.log2(row_cnt))
            shift2 = (int)(math.log2(per_row_thread))
            
            #
            f.write(f"\tconst int {input_b}_lane_offset = (lane & 3);\n")
            f.write(f"\tconst int {input_b}_fragment_offset = (((lane >> 2) << {shift1}) ^ ((lane >> {shift2}) & {row_cnt - 1}));\n")
    #
    else :
        #
        if SMEM_order_b[2] == ld_tile_order_b[0] :
            f.write(f"\tconst int {input_b}_fragment_offset = lane;\n")
        elif SMEM_order_b[2] == ld_tile_order_b[1] :
            if right_frag_size == 8 :
                f.write(f"\tconst int {input_b}_fragment_offset = lane + ((lane >> 4) << 1);\n")
            else :
                f.write(f"\tconst int {input_b}_fragment_offset = lane + (lane >> 4);\n")
        elif SMEM_order_b[2] == ld_tile_order_b[2] :
            f.write(f"\tconst int {input_b}_fragment_offset = lane;\n")
        
    f.write("\n")

    #
    if opt % 2 == 0 :
        f.write("\tfor(int ll = 0; ll < TILE_UNIT - internal_upperbound; ll += 4)\n")
    else :
        f.write("\tfor(int ll = 0; ll < TILE_UNIT; ll += 4)\n")

    #
    tab = 1
    f.write("\t" * tab + "{\n"); tab += 1

    #
    if a_double2_flag :
        #
        if SMEM_order_a[2] == ld_tile_order_a[0] :
            #
            f.write("\t" * tab + "#pragma unroll\n")
            f.write("\t" * tab + f"for(int iter_{input_a} = 0; iter_{input_a} < wmiter; iter_{input_a} += 2)\n")
            f.write("\t" * tab + "{\n"); tab += 1
            
            #
            if left_frag_size == 16 :
                #
                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_a} = 0; cnt_{input_a} < {input_a}_frag_cnt; cnt_{input_a}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1

                #
                ll_offset = (int)(math.log2(left_reg_size))

                #
                f.write("\t" * tab + f"int {input_a}_offset = shm_{input_a}_offset + ((wrow ^ {input_a}_fragment_offset) ^ iter_{input_a}) + (cnt_{input_a} * (ld_stride_a << 3)) + {input_a}_lane_offset + (ll << {ll_offset});\n")
            #
            else :
                #
                ll_offset = (int)(math.log2(left_reg_size))
                
                #
                f.write("\t" * tab + f"int {input_a}_offset = shm_{input_a}_offset + ((wrow ^{input_a}_fragment_offset) ^ iter_{input_a}) + {input_a}_lane_offset + (ll << {ll_offset});\n")

            #
            f.write("\t" * tab + f"double2 tmp = reinterpret_cast<double2*>(&sm_{input_a}[{input_a}_offset])[0];\n")
            f.write("\t" * tab + f"{input_a}_frag[0].x[0] = tmp.x;\n")
            f.write("\t" * tab + f"{input_a}_frag[1].x[0] = tmp.y;\n")
        #
        elif SMEM_order_a[2] == ld_tile_order_a[1] :
            #
            f.write("\t" * tab + "#pragma unroll\n")
            f.write("\t" * tab + f"for(int iter_{input_a} = 0; iter_{input_a} < wmiter; iter_{input_a}++)\n")
            f.write("\t" * tab + "{\n"); tab += 1

            #
            if left_frag_size == 16 :
                #
                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_a} = 0; cnt_{input_a} < {input_a}_frag_cnt; cnt_{input_a}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1

                #
                ll_offset = (int)(math.log2(left_frag_size))

                #
                f.write("\t" * tab + f"int {input_a}_offset = shm_{input_a}_offset + ((wrow + iter_{input_a}) * ld_stride_a) + (({input_a}_fragment_offset ^ (cnt_{input_a} << 1)) << 2) + {input_a}_lane_offset + (ll << {ll_offset});\n")
            #
            else :
                #
                ll_offset = (int)(math.log2(left_frag_size))

                #
                f.write("\t" * tab + f"int {input_a}_offset = shm_{input_a}_offset + ((wrow + iter_{input_a}) * ld_stride_a) + ({input_a}_fragment_offset << 2) + {input_a}_lane_offset + (ll << {ll_offset});\n")

            #
            f.write("\t" * tab + f"{input_a}_frag.x[0] = sm_{input_a}[{input_a}_offset];\n\n")
        #
        else :
            #
            f.write("\t" * tab + "#pragma unroll\n")
            f.write("\t" * tab + f"for(int iter_{input_a} = 0; iter_{input_a} < wmiter; iter_{input_a}++)\n")
            f.write("\t" * tab + "{\n"); tab += 1

            #
            if left_frag_size == 16 :
                #
                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_a} = 0; cnt_{input_a} < {input_a}_frag_cnt; cnt_{input_a}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1

                #
                cnt_offset = (int)(math.log2(8 * size_internal))

                #
                f.write("\t" * tab + f"int {input_a}_offset = shm_{input_a}_offset + ((wrow + iter_{input_a}) * ld_stride_a) + (cnt_{input_a} << {cnt_offset}) + (({input_a}_fragment_offset ^ (ll >> 2)) << 2) + {input_a}_lane_offset;\n")
            #
            else :
                #
                f.write("\t" * tab + f"int {input_a}_offset = shm_{input_a}_offset + ((wrow + iter_{input_a}) * ld_stride_a) + (({input_a}_fragment_offset ^ (ll >> 2)) << 2) + {input_a}_lane_offset;\n")

            #
            f.write("\t" * tab + f"{input_a}_frag.x[0] = sm_{input_a}[{input_a}_offset];\n\n")
    #
    else :
        #
        f.write("\t" * tab + "#pragma unroll\n")
        f.write("\t" * tab + f"for(int iter_{input_a} = 0; iter_{input_a} < wmiter; iter_{input_a}++)\n")
        f.write("\t" * tab + "{\n"); tab += 1

        #
        if SMEM_order_a[2] == ld_tile_order_a[0] :
            #
            if left_frag_size == 16 :
                #
                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_a} = 0; cnt_{input_a} < {input_a}_frag_cnt; cnt_{input_a}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1

                #
                cnt_offset = (int)(math.log2(8 * size_internal))

                #
                f.write("\t" * tab + f"int {input_a}_offset = shm_{input_a}_offset + (wrow + iter_{input_a}) * ld_stride_a + (cnt_{input_a} << {cnt_offset}) + (ll << 3);\n")
            #
            else :
                #
                f.write("\t" * tab + f"int {input_a}_offset = shm_{input_a}_offset + (wrow + iter_{input_a}) * ld_stride_a + (ll << 3);\n")
                
            #
            f.write("\t" * tab + f"{input_a}_frag.x[0] = sm_{input_a}[{input_a}_offset + {input_a}_fragment_offset];\n\n")
        #
        elif SMEM_order_a[2] == ld_tile_order_a[1] :
            #
            if left_frag_size == 16 :
                #
                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_a} = 0; cnt_{input_a} < {input_a}_frag_cnt; cnt_{input_a}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1

                #
                if reg_padd_y[2] == 0 :
                    if reg_padd_y[1] == 0 :
                        f.write("\t" * tab + f"int {input_a}_offset = shm_{input_a}_offset + (wrow + iter_{input_a}) * ld_stride_a + cnt_{input_a} * (TILE_{ld_tile_order_a[2].capitalize()} << 3) + (ll << 3) + (ll >> 2);\n")
                    else :
                        f.write("\t" * tab + f"int {input_a}_offset = shm_{input_a}_offset + (wrow + iter_{input_a}) * ld_stride_a + cnt_{input_a} * ((TILE_{ld_tile_order_a[2].capitalize()} << 3) + {reg_padd_y[1]}) + (ll << 3) + (ll >> 2);\n")
                else :
                    if reg_padd_y[1] == 0 :
                        f.write("\t" * tab + f"int {input_a}_offset = shm_{input_a}_offset + (wrow + iter_{input_a}) * ld_stride_a + cnt_{input_a} * ((TILE_{ld_tile_order_a[2].capitalize()} << 3) + {reg_padd_y[2]}) + (ll << 3) + (ll >> 2);\n")
                    else :
                        tmp_cnt_per_padd_y = reg_padd_y[1] + reg_padd_y[2]
                        f.write("\t" * tab + f"int {input_a}_offset = shm_{input_a}_offset + (wrow + iter_{input_a}) * ld_stride_a + cnt_{input_a} * ((TILE_{ld_tile_order_a[2].capitalize()} << 3) + {tmp_cnt_per_padd_y}) + (ll << 3) + (ll >> 2);\n")
            #
            else :
                #
                f.write("\t" * tab + f"int {input_a}_offset = shm_{input_a}_offset + (wrow + iter_{input_a}) * ld_stride_a + (ll << 3) + (ll >> 1);\n")
            
            #
            f.write("\t" * tab + f"{input_a}_frag.x[0] = sm_{input_a}[{input_a}_offset + {input_a}_fragment_offset];\n\n")
        #
        else :
            #
            if left_frag_size == 16 :
                #
                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_a} = 0; cnt_{input_a} < {input_a}_frag_cnt; cnt_{input_a}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1
                f.write("\t" * tab + f"int {input_a}_offset = shm_{input_a}_offset + (wrow + iter_{input_a}) * ld_stride_a + cnt_{input_a} * (TILE_{ld_tile_order_a[2].capitalize()} << 3) + (ll << 3);\n")
            #
            else :
                #
                f.write("\t" * tab + f"int {input_a}_offset = shm_{input_a}_offset + (wrow + iter_{input_a}) * ld_stride_a + (ll << 3);\n")

            #
            if size_internal == 8 :
                f.write("\t" * tab + f"{input_a}_frag.x[0] = sm_{input_a}[{input_a}_offset + ({input_a}_fragment_offset ^ (ll << 1))];\n\n")
            else :
                f.write("\t" * tab + f"{input_a}_frag.x[0] = sm_{input_a}[{input_a}_offset + ({input_a}_fragment_offset ^ ll)];\n\n")

    #
    if b_double2_flag :
        #
        if SMEM_order_b[2] == ld_tile_order_b[0] :
            #
            f.write("\t" * tab + "#pragma unroll\n")
            f.write("\t" * tab + f"for(int iter_{input_b} = 0; iter_{input_b} < wniter; iter_{input_b} += 2)\n")
            f.write("\t" * tab + "{\n"); tab += 1

            #
            if right_frag_size == 16 :
                #
                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_b} = 0; cnt_{input_b} < {input_b}_frag_cnt; cnt_{input_b}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1

                #
                ll_offset = (int)(math.log2(right_reg_size))

                #
                #for i in range(0, right_reg_per_warp, 2) :
                f.write("\t" * tab + f"int {input_b}_offset = shm_{input_b}_offset + ((wcol ^ {input_b}_fragment_offset) ^ iter_{input_b}) + (cnt_{input_b} * (ld_stride_b << 3)) + {input_b}_lane_offset + (ll << {ll_offset});\n")
            #
            else :
                #
                ll_offset = (int)(math.log2(right_reg_size))
                
                #
                #for i in range(0, right_reg_per_warp, 2) :
                f.write("\t" * tab + f"int {input_b}_offset = shm_{input_b}_offset + ((wcol ^ {input_b}_fragment_offset) ^ iter_{input_b}) + {input_b}_lane_offset + (ll << {ll_offset});\n")
        #
        elif SMEM_order_b[2] == ld_tile_order_b[1] :
            #
            f.write("\t" * tab + "#pragma unroll\n")
            f.write("\t" * tab + f"for(int iter_{input_b} = 0; iter_{input_b} < wniter; iter_{input_b}++)\n")
            f.write("\t" * tab + "{\n"); tab += 1

            #
            if right_frag_size == 16 :
                #
                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_b} = 0; cnt_{input_b} < {input_b}_frag_cnt; cnt_{input_b}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1

                #
                ll_offset = (int)(math.log2(right_frag_size))

                #
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"int {input_b}_offset = shm_{input_b}_offset + ((wcol + iter_{input_b}) * ld_stride_b) + (({input_b}_fragment_offset ^ (cnt_{input_b} << 1)) << 2) + {input_b}_lane_offset + (ll << {ll_offset});\n")
            #
            else :
                #
                ll_offset = (int)(math.log2(right_frag_size))

                #
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"int {input_b}_offset = shm_{input_b}_offset + ((wcol + iter_{input_b}) * ld_stride_b) + ({input_b}_fragment_offset << 2) + {input_b}_lane_offset + (ll << {ll_offset});\n")
        #
        else :
            #
            f.write("\t" * tab + "#pragma unroll\n")
            f.write("\t" * tab + f"for(int iter_{input_b} = 0; iter_{input_b} < wniter; iter_{input_b}++)\n")
            f.write("\t" * tab + "{\n"); tab += 1

            #
            if right_frag_size == 16 :
                #
                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_b} = 0; cnt_{input_b} < {input_b}_frag_cnt; cnt_{input_b}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1

                #
                cnt_offset = (int)(math.log2(8 * size_internal))

                #
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"int {input_b}_offset = shm_{input_b}_offset + ((wcol + iter_{input_b}) * ld_stride_b) + (cnt_{input_b} << {cnt_offset}) + (({input_b}_fragment_offset ^ (ll >> 2)) << 2) + {input_b}_lane_offset;\n")
            #
            else :
                #
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"int {input_b}_offset = shm_{input_b}_offset + ((wcol + iter_{input_b}) * ld_stride_b) + (({input_b}_fragment_offset ^ (ll >> 2)) << 2) + {input_b}_lane_offset;\n")
    #
    else :
        #
        f.write("\t" * tab + "#pragma unroll\n")
        f.write("\t" * tab + f"for(int iter_{input_b} = 0; iter_{input_b} < wniter; iter_{input_b}++)\n")
        f.write("\t" * tab + "{\n"); tab += 1

        #
        if SMEM_order_b[2] == ld_tile_order_b[0] :        
            #
            if right_frag_size == 16 :
                #
                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_b} = 0; cnt_{input_b} < {input_b}_frag_cnt; cnt_{input_b}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1

                #
                cnt_offset = (int)(math.log2(8 * size_internal))
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"int {input_b}_offset = shm_{input_b}_offset + (wcol + iter_{input_b}) * ld_stride_b + (cnt_{input_b} << {cnt_offset}) + (ll << 3);\n")
                # f.write("\t" * tab + f"int {input_b}_offset = shm_{input_b}_offset + (wcol) * ld_stride_b + (cnt_{input_b} << {cnt_offset}) + (ll << 3);\n")
                # f.write("\t" * tab + f"reinterpret_cast<double*>(&{input_b}_frag[0])[0] = sm_{input_b}[{input_b}_offset + {input_b}_fragment_offset];\n\n")
                #
                # for i in range(right_reg_per_warp) :
                #     f.write("\t" * tab + f"reinterpret_cast<double*>(&{input_b}_frag[{i}])[0] = sm_{input_b}[{input_b}_offset{i} + {input_b}_fragment_offset];\n")
                
                # #
                # f.write("\n")
            else :
                #
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"int {input_b}_offset = shm_{input_b}_offset + (wcol + iter_{input_b}) * ld_stride_b + (ll << 3);\n")
                #f.write("\t" * tab + f"reinterpret_cast<double*>(&{input_b}_frag[0])[0] = sm_{input_b}[{input_b}_offset + {input_b}_fragment_offset];\n\n")
                #
                # for i in range(right_reg_per_warp) :
                #     f.write("\t" * tab + f"reinterpret_cast<double*>(&{input_b}_frag[{i}])[0] = sm_{input_b}[{input_b}_offset{i} + {input_b}_fragment_offset];\n")
                
                # #
                # f.write("\n")
        #
        elif SMEM_order_b[2] == ld_tile_order_b[1] :
            #
            if right_frag_size == 16 :
                #
                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_b} = 0; cnt_{input_b} < {input_b}_frag_cnt; cnt_{input_b}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1

                #
                #for i in range(right_reg_per_warp) :
                #
                if reg_padd_x[2] == 0 :
                    if reg_padd_x[1] == 0 :
                        f.write("\t" * tab + f"int {input_b}_offset = shm_{input_b}_offset + (wcol + iter_{input_b}) * ld_stride_b + cnt_{input_b} * (TILE_{ld_tile_order_a[2].capitalize()} << 3) + (ll << 3) + (ll >> 2);\n")
                    else :
                        f.write("\t" * tab + f"int {input_b}_offset = shm_{input_b}_offset + (wcol + iter_{input_b}) * ld_stride_b + cnt_{input_b} * ((TILE_{ld_tile_order_a[2].capitalize()} << 3) + {reg_padd_x[1]}) + (ll << 3) + (ll >> 2);\n")
                
                else :
                    if reg_padd_x[1] == 0 :
                        f.write("\t" * tab + f"int {input_b}_offset = shm_{input_b}_offset + (wcol + iter_{input_b}) * ld_stride_b + cnt_{input_b} * ((TILE_{ld_tile_order_a[2].capitalize()} << 3) + {reg_padd_x[2]}) + (ll << 3) + (ll >> 2);\n")
                    else :
                        tmp_cnt_per_padd_x = reg_padd_x[1] + reg_padd_x[2]
                        f.write("\t" * tab + f"int {input_b}_offset = shm_{input_b}_offset + (wcol + iter_{input_b}) * ld_stride_b + cnt_{input_b} * ((TILE_{ld_tile_order_a[2].capitalize()} << 3) + {tmp_cnt_per_padd_x}) + (ll << 3) + (ll >> 2);\n")
                
                #
                #f.write("\t" * tab + f"reinterpret_cast<double*>(&{input_b}_frag[0])[0] = sm_{input_b}[{input_b}_offset + {input_b}_fragment_offset];\n\n")

                # #
                # for i in range(right_reg_per_warp) :
                #     f.write("\t" * tab + f"reinterpret_cast<double*>(&{input_b}_frag[{i}])[0] = sm_{input_b}[{input_b}_offset{i} + {input_b}_fragment_offset];\n")
                
                # f.write("\n")
            #
            else :
                #
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"int {input_b}_offset = shm_{input_b}_offset + (wcol + iter_{input_b}) * ld_stride_b + (ll << 3) + (ll >> 1);\n")
                #f.write("\t" * tab + f"reinterpret_cast<double*>(&{input_b}_frag[0])[0] = sm_{input_b}[{input_b}_offset + {input_b}_fragment_offset];\n\n")

                #
                # for i in range(right_reg_per_warp) :
                #     f.write("\t" * tab + f"reinterpret_cast<double*>(&{input_b}_frag[{i}])[0] = sm_{input_b}[{input_b}_offset{i} + {input_b}_fragment_offset];\n")
                
                # f.write("\n")
        else :
            #
            if right_frag_size == 16 :
                #
                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_b} = 0; cnt_{input_b} < {input_b}_frag_cnt; cnt_{input_b}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1
                
                #
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"int {input_b}_offset = shm_{input_b}_offset + (wcol + iter_{input_b}) * ld_stride_b + cnt_{input_b} * (TILE_{ld_tile_order_a[2].capitalize()} << 3) + (ll << 3);\n")
                
                #
                # if size_internal == 8 :
                #     #
                #     #for i in range(right_reg_per_warp) :
                #     f.write("\t" * tab + f"reinterpret_cast<double*>(&{input_b}_frag[0])[0] = sm_{input_b}[{input_b}_offset + ({input_b}_fragment_offset ^ (ll << 1))];\n\n")
                    
                #     #f.write("\n")
                # else :
                #     #
                #     #for i in range(right_reg_per_warp) :
                #     f.write("\t" * tab + f"reinterpret_cast<double*>(&{input_b}_frag[0])[0] = sm_{input_b}[{input_b}_offset + ({input_b}_fragment_offset ^ ll)];\n\n")
                    
                #     f.write("\n")
            else :
                #
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"int {input_b}_offset = shm_{input_b}_offset + (wcol + iter_{input_b}) * ld_stride_b + (ll << 3);\n")

                #
                # if size_internal == 8 :
                #     #
                #     #for i in range(right_reg_per_warp) :
                #     f.write("\t" * tab + f"reinterpret_cast<double*>(&{input_b}_frag[0])[0] = sm_{input_b}[{input_b}_offset + ({input_b}_fragment_offset ^ (ll << 1))];\n")
                    
                #     f.write("\n")
                # else :
                #     #
                #     #for i in range(right_reg_per_warp) :
                #     f.write("\t" * tab + f"reinterpret_cast<double*>(&{input_b}_frag[0])[0] = sm_{input_b}[{input_b}_offset + ({input_b}_fragment_offset ^ ll)];\n")
                    
                #     f.write("\n")

    # #
    # f.write("\t" * tab + "#pragma unroll\n")
    # f.write("\t" * tab + f"for(int iter_{input_b} = 0; iter_{input_b} < wniter; iter_{input_b}++)\n")
    # f.write("\t" * tab + "{\n"); tab += 1

    #
    if b_double2_flag and (SMEM_order_b[2] == ld_tile_order_b[0]) :
        #
        if a_double2_flag and (SMEM_order_a[2] == ld_tile_order_a[0]) :
            #
            if left_frag_size == 16 and right_frag_size == 16 :
                #for i in range(0, right_reg_per_warp, 2) :
                f.write("\t" * tab + f"const int out_idx0 = (((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = (((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx2 = ((((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx3 = ((((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            elif left_frag_size == 16 and right_frag_size == 8 :
                #for i in range(0, right_reg_per_warp, 2) :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx2 = (((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx3 = (((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a};\n")
            #
            elif left_frag_size == 8 and right_frag_size == 16 :
                #for i in range(0, right_reg_per_warp, 2) :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx2 = (((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx3 = (((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            else :
                #for i in range(0, right_reg_per_warp, 2) :
                f.write("\t" * tab + f"const int out_idx0 = (iter_{input_a} * wniter) + iter_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = (iter_{input_a} * wniter) + iter_{input_b} + 1;\n")
                f.write("\t" * tab + f"const int out_idx2 = ((iter_{input_a} + 1) * wniter) + iter_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx3 = ((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1;\n")
        else :
            #
            if left_frag_size == 16 and right_frag_size == 16 :
                #for i in range(0, right_reg_per_warp, 2) :
                f.write("\t" * tab + f"const int out_idx0 = (((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = (((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            elif left_frag_size == 16 and right_frag_size == 8 :
                #for i in range(0, right_reg_per_warp, 2) :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a};\n")
            #
            elif left_frag_size == 8 and right_frag_size == 16 :
                #for i in range(0, right_reg_per_warp, 2) :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            else :
                #for i in range(0, right_reg_per_warp, 2) :
                f.write("\t" * tab + f"const int out_idx0 = (iter_{input_a} * wniter) + iter_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = (iter_{input_a} * wniter) + iter_{input_b} + 1;\n")
    #
    else :
        #
        if a_double2_flag and (SMEM_order_a[2] == ld_tile_order_a[0]) :
            #
            if left_frag_size == 16 and right_frag_size == 16 :
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"const int out_idx0 = (((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            elif left_frag_size == 16 and right_frag_size == 8 :
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx1 = (((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
            #
            elif left_frag_size == 8 and right_frag_size == 16 :
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = (((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            else :
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a}) * wniter) + iter_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} + 1) * wniter) + iter_{input_b};\n")
        #
        else :
            #
            if left_frag_size == 16 and right_frag_size == 16 :
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"const int out_idx = (((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            elif left_frag_size == 16 and right_frag_size == 8 :
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"const int out_idx = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
            #
            elif left_frag_size == 8 and right_frag_size == 16 :
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"const int out_idx = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            else :
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"const int out_idx = (iter_{input_a} * wniter) + iter_{input_b};\n")
    
    #
    # f.write("\t" * tab + f"if(iter_{input_b} + 1 < wniter)\n")
    # f.write("\t" * tab + "{\n"); tab += 1
    
    #
    # if SMEM_order_b[2] == ld_tile_order_b[0] :        
    #     #
    #     if right_frag_size == 16 :
    #         #
    #         cnt_offset = (int)(math.log2(8 * size_internal))
    #         f.write("\t" * tab + f"int {input_b}_offset_next = shm_{input_b}_offset + (wcol + (iter_{input_b} + 1)) * ld_stride_b + (cnt_{input_b} << {cnt_offset}) + (ll << 3);\n")
    #         f.write("\t" * tab + f"reinterpret_cast<double*>(&{input_b}_frag[(iter_{input_b} + 1) & 1])[0] = sm_{input_b}[{input_b}_offset_next + {input_b}_fragment_offset];\n")
    #     else :
    #         #
    #         f.write("\t" * tab + f"int {input_b}_offset_next = shm_{input_b}_offset + (wcol + (iter_{input_b} + 1)) * ld_stride_b + (ll << 3);\n")
    #         f.write("\t" * tab + f"reinterpret_cast<double*>(&{input_b}_frag[(iter_{input_b} + 1) & 1])[0] = sm_{input_b}[{input_b}_offset_next + {input_b}_fragment_offset];\n")

    # elif SMEM_order_b[2] == ld_tile_order_b[1] :
    #     #
    #     if right_frag_size == 16 :
    #         #
    #         if reg_padd_x[2] == 0 :
    #             if reg_padd_x[1] == 0 :
    #                 f.write("\t" * tab + f"int {input_b}_offset_next = shm_{input_b}_offset + (wcol + (iter_{input_b} + 1)) * ld_stride_b + cnt_{input_b} * (TCCG_{str_eq_num}_TILE_{ld_tile_order_a[2].capitalize()} << 3) + (ll << 3) + (ll >> 2);\n")
    #             else :
    #                 f.write("\t" * tab + f"int {input_b}_offset_next = shm_{input_b}_offset + (wcol + (iter_{input_b} + 1)) * ld_stride_b + cnt_{input_b} * ((TCCG_{str_eq_num}_TILE_{ld_tile_order_a[2].capitalize()} << 3) + {reg_padd_x[1]}) + (ll << 3) + (ll >> 2);\n")
    #         else :
    #             if reg_padd_x[1] == 0 :
    #                 f.write("\t" * tab + f"int {input_b}_offset_next = shm_{input_b}_offset + (wcol + (iter_{input_b} + 1)) * ld_stride_b + cnt_{input_b} * ((TCCG_{str_eq_num}_TILE_{ld_tile_order_a[2].capitalize()} << 3) + {reg_padd_x[2]}) + (ll << 3) + (ll >> 2);\n")
    #             else :
    #                 tmp_cnt_per_padd_x = reg_padd_x[1] + reg_padd_x[2]
    #                 f.write("\t" * tab + f"int {input_b}_offset_next = shm_{input_b}_offset + (wcol + (iter_{input_b} + 1)) * ld_stride_b + cnt_{input_b} * ((TCCG_{str_eq_num}_TILE_{ld_tile_order_a[2].capitalize()} << 3) + {tmp_cnt_per_padd_x}) + (ll << 3) + (ll >> 2);\n")
    #         f.write("\t" * tab + f"reinterpret_cast<double*>(&{input_b}_frag[(iter_{input_b} + 1) & 1])[0] = sm_{input_b}[{input_b}_offset_next + {input_b}_fragment_offset];\n")
    #     else :
    #         f.write("\t" * tab + f"int {input_b}_offset_next = shm_{input_b}_offset + (wcol + (iter_{input_b} + 1)) * ld_stride_b + (ll << 3) + (ll >> 1);\n")
    #         f.write("\t" * tab + f"reinterpret_cast<double*>(&{input_b}_frag[(iter_{input_b} + 1) & 1])[0] = sm_{input_b}[{input_b}_offset_next + {input_b}_fragment_offset];\n")
    # else :
    #     #
    #     if right_frag_size == 16 :
    #         #  
    #         f.write("\t" * tab + f"int {input_b}_offset_next = shm_{input_b}_offset + (wcol + (iter_{input_b} + 1)) * ld_stride_b + cnt_{input_b} * (TCCG_{str_eq_num}_TILE_{ld_tile_order_a[2].capitalize()} << 3) + (ll << 3);\n")
            
    #         #
    #         if size_internal == 8 :
    #             f.write("\t" * tab + f"reinterpret_cast<double*>(&{input_b}_frag[(iter_{input_b} + 1) & 1])[0] = sm_{input_b}[{input_b}_offset_next + ({input_b}_fragment_offset ^ (ll << 1))];\n")
    #         else :
    #             f.write("\t" * tab + f"reinterpret_cast<double*>(&{input_b}_frag[(iter_{input_b} + 1) & 1])[0] = sm_{input_b}[{input_b}_offset_next + ({input_b}_fragment_offset ^ ll)];\n")
    #     else :
    #         #
    #         f.write("\t" * tab + f"int {input_b}_offset_next = shm_{input_b}_offset + (wcol + (iter_{input_b} + 1)) * ld_stride_b + (ll << 3);\n")

    #         #
    #         if size_internal == 8 :
    #             f.write("\t" * tab + f"reinterpret_cast<double*>(&{input_b}_frag[(iter_{input_b} + 1) & 1])[0] = sm_{input_b}[{input_b}_offset_next + ({input_b}_fragment_offset ^ (ll << 1))];\n")
    #         else :
    #             f.write("\t" * tab + f"reinterpret_cast<double*>(&{input_b}_frag[(iter_{input_b} + 1) & 1])[0] = sm_{input_b}[{input_b}_offset_next + ({input_b}_fragment_offset ^ ll)];\n")
    # tab -= 1
    # f.write("\t" * tab + "}\n\n")
    
    #
    # for i in range(right_reg_per_warp) :
    #     f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx{i}], {input_a}_frag, {input_b}_frag[{i}], t3_frag[out_idx{i}]);\n")
    
    if b_double2_flag :
        #
        if a_double2_flag and (SMEM_order_a[2] == ld_tile_order_a[0]) :
            #
            if SMEM_order_b[2] == ld_tile_order_b[0] :
            #
                #for i in range(0, right_reg_per_warp, 2) :
                f.write("\t" * tab + f"double2 tmp = reinterpret_cast<double2*>(&sm_{input_b}[{input_b}_offset])[0];\n")
                f.write("\t" * tab + f"{input_b}_frag[0].x[0] = tmp.x;\n")
                f.write("\t" * tab + f"{input_b}_frag[1].x[0] = tmp.y;\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx0], {input_a}_frag[0], {input_b}_frag[0], t3_frag[out_idx0]);\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx1], {input_a}_frag[0], {input_b}_frag[1], t3_frag[out_idx1]);\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx2], {input_a}_frag[1], {input_b}_frag[0], t3_frag[out_idx1]);\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx3], {input_a}_frag[1], {input_b}_frag[1], t3_frag[out_idx1]);\n")
            #
            elif SMEM_order_b[2] == ld_tile_order_b[1] :
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset];\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx0], {input_a}_frag[0], {input_b}_frag, t3_frag[out_idx0]);\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx1], {input_a}_frag[1], {input_b}_frag, t3_frag[out_idx1]);\n")
            #
            else :
                f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset];\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx0], {input_a}_frag[0], {input_b}_frag, t3_frag[out_idx0]);\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx1], {input_a}_frag[1], {input_b}_frag, t3_frag[out_idx1]);\n")
        #
        else :
            #
            if SMEM_order_b[2] == ld_tile_order_b[0] :
            #
                #for i in range(0, right_reg_per_warp, 2) :
                f.write("\t" * tab + f"double2 tmp = reinterpret_cast<double2*>(&sm_{input_b}[{input_b}_offset])[0];\n")
                f.write("\t" * tab + f"{input_b}_frag[0].x[0] = tmp.x;\n")
                f.write("\t" * tab + f"{input_b}_frag[1].x[0] = tmp.y;\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx0], {input_a}_frag, {input_b}_frag[0], t3_frag[out_idx0]);\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx1], {input_a}_frag, {input_b}_frag[1], t3_frag[out_idx1]);\n")
            #
            elif SMEM_order_b[2] == ld_tile_order_b[1] :
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset];\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx], {input_a}_frag, {input_b}_frag, t3_frag[out_idx]);\n")
            #
            else :
                f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset];\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx], {input_a}_frag, {input_b}_frag, t3_frag[out_idx]);\n")
    #
    else :
        #
        if a_double2_flag and (SMEM_order_a[2] == ld_tile_order_a[0]) :
            #
            if SMEM_order_b[2] == ld_tile_order_b[0] :
                #
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset + {input_b}_fragment_offset];\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx0], {input_a}_frag[0], {input_b}_frag, t3_frag[out_idx0]);\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx1], {input_a}_frag[1], {input_b}_frag, t3_frag[out_idx1]);\n")
            #
            elif SMEM_order_b[2] == ld_tile_order_b[1] :
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset + {input_b}_fragment_offset];\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx0], {input_a}_frag[0], {input_b}_frag, t3_frag[out_idx0]);\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx1], {input_a}_frag[1], {input_b}_frag, t3_frag[out_idx1]);\n")
            #
            else :
                #
                if size_internal == 8 :
                    #
                    #for i in range(right_reg_per_warp) :
                    f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset + ({input_b}_fragment_offset ^ (ll << 1))];\n")
                    f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx0], {input_a}_frag[0], {input_b}_frag, t3_frag[out_idx0]);\n")
                    f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx1], {input_a}_frag[1], {input_b}_frag, t3_frag[out_idx1]);\n")
                #
                else :
                    #
                    #for i in range(right_reg_per_warp) :
                    f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset + ({input_b}_fragment_offset ^ ll)];\n")
                    f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx0], {input_a}_frag[0], {input_b}_frag, t3_frag[out_idx0]);\n")
                    f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx1], {input_a}_frag[1], {input_b}_frag, t3_frag[out_idx1]);\n")
        #
        else :
            #
            if SMEM_order_b[2] == ld_tile_order_b[0] :
                #
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset + {input_b}_fragment_offset];\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx], {input_a}_frag, {input_b}_frag, t3_frag[out_idx]);\n")
            #
            elif SMEM_order_b[2] == ld_tile_order_b[1] :
                #for i in range(right_reg_per_warp) :
                f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset + {input_b}_fragment_offset];\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx], {input_a}_frag, {input_b}_frag, t3_frag[out_idx]);\n")
            #
            else :
                #
                if size_internal == 8 :
                    #
                    #for i in range(right_reg_per_warp) :
                    f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset + ({input_b}_fragment_offset ^ (ll << 1))];\n")
                    f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx], {input_a}_frag, {input_b}_frag, t3_frag[out_idx]);\n")
                #
                else :
                    #
                    #for i in range(right_reg_per_warp) :
                    f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset + ({input_b}_fragment_offset ^ ll)];\n")
                    f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx], {input_a}_frag, {input_b}_frag, t3_frag[out_idx]);\n")
    
    #
    for i in range(tab) :
        tab -= 1
        f.write("\t" * tab + "}\n")

    f.write("\n")

# generate __device__ compute kernel
def tc_code_kernel_dev_compute(f, kernel_name, l_splited_indices_size,
                               fvi_flag, input_a, input_b, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, SMEM_order_a, SMEM_order_b,
                               split_input, warp_shape, double2_flag, reg_padd_y, reg_padd_x, kernel_variants) :
    #
    for i in range(1, kernel_variants + 1) :
        #
        kernel_name_i = kernel_name + "_" + str(i)

        #
        tc_code_kernel_dev_compute_head(f, kernel_name_i, input_a, input_b, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, split_input, fvi_flag, i)

        #
        tc_code_kernel_dev_compute_body(f, l_splited_indices_size, input_a, input_b, ld_tile_order_a, ld_tile_order_b, SMEM_order_a, SMEM_order_b,
                                        warp_shape, double2_flag, reg_padd_y, reg_padd_x, fvi_flag, i)

#
def tc_code_kernel_body(f, l_inputs_addr, l_external_index, l_internal_index,
                        fvi_flag, input_a, input_b, internal_order,
                        ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b,
                        split_input, producer_cnt, opt) :
    #
    t2_split_flag = split_input[0]
    v2_split_flag = split_input[1]

    #
    if fvi_flag == 1 :
        a_split_flag = v2_split_flag
        b_split_flag = t2_split_flag
    elif fvi_flag == 2 :
        a_split_flag = t2_split_flag
        b_split_flag = v2_split_flag

    #
    reg_partial_index = []
    if a_split_flag :
        reg_partial_index.append(collapsed_a[0])
    else :
        reg_partial_index.append(ld_tile_order_a[0])
    if b_split_flag :
        reg_partial_index.append(collapsed_b[0])
    else :
        reg_partial_index.append(ld_tile_order_b[0])

    #
    frag_partial_index = []
    if a_split_flag :
        frag_partial_index.append(collapsed_a[0])
    else :
        frag_partial_index.append(ld_tile_order_a[1])
    if b_split_flag :
        frag_partial_index.append(collapsed_b[0])
    else :
        frag_partial_index.append(ld_tile_order_b[1])
    
    #
    f.write("\t// Tensor Contraction : " + str(l_inputs_addr[0]) + "\n")

    #
    tab = 1
    r_internal_order = reversed(internal_order)
    for cnt, idx in enumerate(r_internal_order) :
        #
        if cnt < len(internal_order) - 1 :
            f.write("\t" * tab + f"for(int iter_{idx} = 0; iter_{idx} < size_{idx}; iter_{idx} += TILE_{idx.capitalize()})\n")
            f.write("\t" * tab + "{\n"); tab += 1
        elif cnt == len(internal_order) - 1 :
            f.write("\t" * tab + f"for(int l = -((int)(PIPELINE_STAGES - 1) * TILE_UNIT); l < size_internal; l += TILE_UNIT)\n")
            f.write("\t" * tab + "{\n"); tab += 1

    #
    f.write("\t" * tab + "// Load\n")
    f.write("\t" * tab + f"int load_idx = l + (PIPELINE_STAGES - 1) * TILE_UNIT;\n\n")

    #
    if producer_cnt > 0 :
        f.write("\t" * tab + "if(thread_role == cuda::pipeline_role::producer && load_idx < size_internal)\n")
    else :    
        f.write("\t" * tab + "if(load_idx < size_internal)\n")
    f.write("\t" * tab + "{\n"); tab += 1

    # if len(l_internal_index) > 1 :
    #     if opt % 2 == 0 :
    #         f.write("\t\t\tload_idx -= iter_" + iteration_order[0] + "_offset;\n")
    #         f.write("\t\t\tint iter_" + iteration_order[0] + " = load_idx % size_" + iteration_order[0] + ";\n")
    #         f.write("\t\t\tint iter_" + iteration_order[1] + " = load_idx / size_" + iteration_order[0] + ";\n")
    #         f.write("\t\t\tinternal_offset = (iter_" + iteration_order[0] + " + TCCG_" + str_eq_num + "_TILE_" + iteration_order[0].capitalize() + ") - size_" + iteration_order[0] + ";\n")
    #         f.write("\t\t\tif(internal_offset > 0)\n")
    #         f.write("\t\t\t{\n")
    #         f.write("\t\t\t\tinternal_upperbound = internal_offset;\n")
    #         f.write("\t\t\t\titer_" + iteration_order[0] + "_offset += internal_upperbound;\n")
    #         f.write("\t\t\t}\n\n")

    #
    f.write("\t" * tab + "int next_shm_offset = (shm_offset + PIPELINE_STAGES - 1) % PIPELINE_STAGES;\n")

    #
    if opt % 2 == 0 :
        f.write("\t" * tab + f"int load_ub = max(0, (load_idx + TILE_UNIT - size_internal));\n\n")
    else :
        f.write("\n")

    #
    f.write("\t" * tab + "pipeline.producer_acquire();\n")

    #
    f.write("\t" * tab + f"load_from_GM_{opt}(dev_{input_a}, dev_{input_b}, sm_{input_a}, sm_{input_b},\n")
    
    #
    f.write("\t" * (tab + 2) + ", ".join([f"size_{idx}" for idx in l_external_index]) + ",\n")
    
    #
    f.write("\t" * (tab + 2) + ", ".join([f"size_{idx}" for idx in l_internal_index]) + ",\n")
    
    #
    f.write("\t" * (tab + 2) + ", ".join([f"blk_idx_{idx}" for idx in l_external_index]) + ",\n")

    #
    if producer_cnt > 0 :
        f.write("\t" * (tab + 2) + "pipeline, producer_warp, thread,\n")
    else :
        f.write("\t" * (tab + 2) + "pipeline, thread,\n")

    #
    f.write("\t" * (tab + 2) + f"next_shm_offset * {input_a}_tile_size, next_shm_offset * {input_b}_tile_size,")

    #
    if len(internal_order) > 1 :
        for cnt, idx in enumerate(internal_order) :
            if cnt == 0 :
                f.write(" load_idx,")
            elif cnt == len(internal_order) - 1 :
                f.write(f" iter_{idx}")
            else :
                f.write(f" iter_{idx},")
    else :
        f.write(" load_idx")

    #
    if opt == 1 :
        f.write(");\n")
    elif opt == 2 :
        f.write(",\n")
        f.write("\t" * (tab + 2) + "load_ub);\n")
    elif opt == 3 :
        f.write(",\n")
        f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[0]}, rng_{reg_partial_index[1]});\n")
    elif opt == 4 :
        f.write(",\n")
        f.write("\t" * (tab + 2) + "load_ub,\n")
        f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[0]}, rng_{reg_partial_index[1]});\n")
    elif opt == 5 :
        f.write(",\n")
        f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]});\n")
    elif opt == 6 :
        f.write(",\n")
        f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]},\n")
        f.write("\t" * (tab + 2) + "load_ub);\n")
    elif opt == 7 :
        f.write(",\n")
        if a_split_flag :
            f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]},\n")
            f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[1]});\n")
        elif b_split_flag :
            f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]},\n")
            f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[0]});\n")
        else :
            f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]},\n")
            f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[0]}, rng_{reg_partial_index[1]});\n")
    elif opt == 8 :
        f.write(",\n")
        if a_split_flag :
            f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]},\n")
            f.write("\t" * (tab + 2) + "load_ub,\n")
            f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[1]});\n")
        elif b_split_flag :
            f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]},\n")
            f.write("\t" * (tab + 2) + "load_ub,\n")
            f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[0]});\n")
        else :
            f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]},\n")
            f.write("\t" * (tab + 2) + "load_ub,\n")
            f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[0]}, rng_{reg_partial_index[1]});\n")

    #
    f.write("\t" * tab + "pipeline.producer_commit();\n"); tab -= 1

    f.write("\t" * tab + "}\n\n")

    #
    f.write("\t" * tab + "// Compute\n")

    #
    if producer_cnt > 0 :
        f.write("\t" * tab + "if(thread_role == cuda::pipeline_role::consumer && l >= 0)\n")
    else :
        f.write("\t" * tab + "if(l >= 0)\n")
    f.write("\t" * tab + "{\n"); tab += 1

    #
    if opt % 2 == 0 :
        f.write("\t" * tab + f"int compute_ub = max(0, (l + TILE_UNIT - size_internal));\n\n")
    # else :
    #     f.write("\n")

    f.write("\t" * tab + "pipeline.consumer_wait();\n")
    f.write("\t" * tab + f"compute_MMA_{opt}(sm_{input_a}, sm_{input_b}, wmiter, wniter, wrow, wcol,\n")
    f.write("\t" * (tab + 2) + f"{input_a}_frag_cnt, {input_b}_frag_cnt,\n")
    f.write("\t" * (tab + 2) + "t3_frag,\n")
    f.write("\t" * (tab + 2) + f"shm_offset * {input_a}_tile_size, shm_offset * {input_b}_tile_size")

    #
    if opt == 1 :
        f.write(");\n")
    elif opt == 2 :
        f.write(",\n")
        f.write("\t" * (tab + 2) + "compute_ub);\n")
    elif opt == 3 :
        f.write(",\n")
        f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[0]}, rng_{reg_partial_index[1]});\n")
    elif opt == 4 :
        f.write(",\n")
        f.write("\t" * (tab + 2) + "compute_ub,\n")
        f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[0]}, rng_{reg_partial_index[1]});\n")
    elif opt == 5 :
        f.write(",\n")
        f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]});\n")
    elif opt == 6 :
        f.write(",\n")
        f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]},\n")
        f.write("\t" * (tab + 2) + "compute_ub);\n")
    elif opt == 7 :
        f.write(",\n")
        if a_split_flag :
            f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]},\n")
            f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[1]});\n")
        elif b_split_flag :
            f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]},\n")
            f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[0]});\n")
        else :
            f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]},\n")
            f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[0]}, rng_{reg_partial_index[1]});\n")
    elif opt == 8 :
        f.write(",\n")
        if a_split_flag :
            f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]},\n")
            f.write("\t" * (tab + 2) + "compute_ub,\n")
            f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[1]});\n")
        elif b_split_flag :
            f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]},\n")
            f.write("\t" * (tab + 2) + "compute_ub,\n")
            f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[0]});\n")
        else :
            f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]},\n")
            f.write("\t" * (tab + 2) + "compute_ub,\n")
            f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[0]}, rng_{reg_partial_index[1]});\n")
    
    #
    f.write("\t" * tab + "pipeline.consumer_release();\n"); tab -= 1
    f.write("\t" * tab + "}\n\n")

    #
    f.write("\t" * tab + "// Update shared memory offset\n")
    f.write("\t" * tab + "shm_offset = (shm_offset + 1) % PIPELINE_STAGES;\n"); tab -= 1
    f.write("\t" * tab + "}\n")

    #
    if len(internal_order) > 1 :
        f.write("\n" + "\t" * tab + "// Update shared memory offset\n")
        f.write("\t" * tab + "shm_offset = (shm_offset + 1) % PIPELINE_STAGES;\n"); tab -= 1
    else :
        tab -= 1

    #
    for i in range(tab) :
        f.write("\t" * tab + "}\n"); tab -= 1
    f.write("\n")
    
    #
    tab = 1
    f.write("\t" * tab + "// Store result into global memory\n")

    #
    if producer_cnt > 0 :
        f.write("\t" * tab + "if(thread_role == cuda::pipeline_role::consumer)\n")
        f.write("\t" * tab + "{\n"); tab += 1

    #
    f.write("\t" * tab + "int frag_idx = 0;\n")

    #
    f.write("\t" * tab + "#pragma unroll\n")
    f.write("\t" * tab + f"for(int iter_{input_a} = 0; iter_{input_a} < wmiter; iter_{input_a}++)\n")
    f.write("\t" * tab + "{\n"); tab += 1
    f.write("\t" * tab + f"int base = t3_base_thread + iter_{input_a} * inter_m_stride;\n")

    #
    if a_split_flag :
        if opt > 2 :
            f.write("\t" * tab + f"int g_m_iter = (wrow + iter_{input_a}) * TILE_{ld_tile_order_a[1].capitalize()};\n")
    else :
        if opt == 3 or opt == 4 or opt == 7 or opt == 8 : # reg mapped partial
            f.write("\t" * tab + f"int g_m_iter = wrow + iter_{input_a};\n")

    #
    f.write("\t" * tab + "#pragma unroll\n")
    f.write("\t" * tab + f"for(int iter_{input_b} = 0; iter_{input_b} < wniter; iter_{input_b}++)\n")
    f.write("\t" * tab + "{\n"); tab += 1
    f.write("\t" * tab + f"int base_iter = base + iter_{input_b} * inter_n_stride;\n")
    
    #
    if b_split_flag :
        if opt > 2 :
            f.write("\t" * tab + f"int g_n_iter = (wcol + iter_{input_b}) * TILE_{ld_tile_order_b[1].capitalize()};\n")
    else :
        if opt == 3 or opt == 4 or opt == 7 or opt == 8 : # reg mapped partial
            f.write("\t" * tab + f"int g_n_iter = wcol + iter_{input_b};\n")

    #
    f.write("\t" * tab + "#pragma unroll\n")
    f.write("\t" * tab + f"for(int cnt_{input_a} = 0; cnt_{input_a} < {input_a}_frag_cnt; cnt_{input_a}++)\n")
    f.write("\t" * tab + "{\n"); tab += 1

    #
    if a_split_flag :
        if opt > 2 :
            f.write("\t" * tab + f"int m_base = g_m_iter + (cnt_{input_a} << 3);\n")
    else :
        f.write("\t" * tab + f"int m_base = (cnt_{input_a} << 3);\n") 
    
    #
    f.write("\t" * tab + f"int base_{input_a} = base_iter + (cnt_{input_a} << 3) * intra_stride;\n")

    #
    f.write("\t" * tab + "#pragma unroll\n")
    f.write("\t" * tab + f"for(int cnt_{input_b} = 0; cnt_{input_b} < {input_b}_frag_cnt; cnt_{input_b}++, frag_idx++)\n")
    f.write("\t" * tab + "{\n"); tab += 1

    #
    if b_split_flag :
        if opt > 2 :
            f.write("\t" * tab + f"int n_base = g_n_iter + (cnt_{input_b} << 3);\n")
    else :
        f.write("\t" * tab + f"int n_base = (cnt_{input_b} << 3);\n")

    #
    f.write("\t" * tab + f"int dst = base_{input_a} + (cnt_{input_b} << 3);\n")

    #
    if a_split_flag and b_split_flag :
        partial_a = collapsed_a[0]
        partial_b = collapsed_b[0]
    elif a_split_flag :
        if opt > 2 :
            partial_a = collapsed_a[0]
        if opt == 3 or opt == 4 or opt == 7 or opt == 8 :
            reg_partial_b = ld_tile_order_b[0]
        if opt > 4 :
            frag_partial_b = ld_tile_order_b[1]
    elif b_split_flag :
        if opt > 2 :
            partial_b = collapsed_b[0]
        if opt == 3 or opt == 4 or opt == 7 or opt == 8 :
            reg_partial_a = ld_tile_order_a[0]
        if opt > 4 :
            frag_partial_a = ld_tile_order_a[1]
    else :
        if opt == 3 or opt == 4 or opt == 7 or opt == 8 :
            reg_partial_a = ld_tile_order_a[0]
            reg_partial_b = ld_tile_order_b[0]
        if opt > 4 :
            frag_partial_a = ld_tile_order_a[1]
            frag_partial_b = ld_tile_order_b[1]

    #
    if opt == 1 or opt == 2 :
        f.write("\t" * tab + "nvcuda::wmma::store_matrix_sync(&dev_t3[dst], t3_frag[frag_idx], intra_stride, nvcuda::wmma::mem_row_major);\n"); tab -= 1
    else :
        #
        full_condition = []
        scatter_store = []
        
        #
        if a_split_flag and b_split_flag :
            full_condition.append(f"(m_base + 8 <= rng_{partial_a})")
            full_condition.append(f"(n_base + 8 <= rng_{partial_b})")
            scatter_store.append(f"scatter_store_tail(&dev_t3[dst], m_base, n_base, rng_{partial_a}, rng_{partial_b}, intra_stride, t3_frag[frag_idx]);\n")
        elif a_split_flag :
            #
            if opt == 3 or opt == 4 :
                full_condition.append(f"(m_base + 8 <= rng_{partial_a})")
                full_condition.append(f"(g_n_iter < rng_{reg_partial_b})")
                scatter_store.append(f"scatter_store_tail0(&dev_t3[dst], m_base, g_n_iter, rng_{partial_a}, rng_{reg_partial_b}, intra_stride, t3_frag[frag_idx]);\n")
            elif opt == 5 or opt == 6 :
                full_condition.append(f"(m_base + 8 <= rng_{partial_a})")
                full_condition.append(f"(n_base + 8 <= rng_{frag_partial_b})")
                scatter_store.append(f"scatter_store_tail1(&dev_t3[dst], m_base, n_base, rng_{partial_a}, rng_{frag_partial_b}, intra_stride, t3_frag[frag_idx]);\n")
            elif opt == 7 or opt == 8 :
                full_condition.append(f"(m_base + 8 <= rng_{partial_a})")
                full_condition.append(f"(g_n_iter < rng_{reg_partial_b})")
                full_condition.append(f"(n_base + 8 <= rng_{frag_partial_b})")
                scatter_store.append(f"scatter_store_tail2(&dev_t3[dst], m_base, g_n_iter, n_base, rng_{partial_a}, rng_{reg_partial_b}, rng_{frag_partial_b}, intra_stride, t3_frag[frag_idx]);\n")
        elif b_split_flag :
            #
            if opt == 3 or opt == 4 :
                full_condition.append(f"(g_m_iter < rng_{reg_partial_a})")
                full_condition.append(f"(n_base + 8 <= rng_{partial_b})")
                scatter_store.append(f"scatter_store_tail0(&dev_t3[dst], g_m_iter, n_base, rng_{reg_partial_a}, rng_{partial_b}, intra_stride, t3_frag[frag_idx]);\n")
            elif opt == 5 or opt == 6 :
                full_condition.append(f"(m_base + 8 <= rng_{frag_partial_a})")
                full_condition.append(f"(n_base + 8 <= rng_{partial_b})")
                scatter_store.append(f"scatter_store_tail1(&dev_t3[dst], m_base, n_base, rng_{frag_partial_a}, rng_{partial_b}, intra_stride, t3_frag[frag_idx]);\n")
            elif opt == 7 or opt == 8 :
                full_condition.append(f"(g_m_iter < rng_{reg_partial_a})")
                full_condition.append(f"(m_base + 8 <= rng_{frag_partial_a})")
                full_condition.append(f"(n_base + 8 <= rng_{partial_b})")
                scatter_store.append(f"scatter_store_tail2(&dev_t3[dst], g_m_iter, m_base, n_base, rng_{reg_partial_a}, rng_{frag_partial_a}, rng_{partial_b}, intra_stride, t3_frag[frag_idx]);\n")
        else :
            #
            if opt == 3 or opt == 4 :
                full_condition.append(f"(g_m_iter < rng_{reg_partial_a})")
                full_condition.append(f"(g_n_iter < rng_{reg_partial_b})")
                scatter_store.append(f"scatter_store_tail0(&dev_t3[dst], g_m_iter, g_n_iter, rng_{reg_partial_a}, rng_{reg_partial_b}, intra_stride, t3_frag[frag_idx]);\n")
            elif opt == 5 or opt == 6 :
                full_condition.append(f"(m_base + 8 <= rng_{frag_partial_a})")
                full_condition.append(f"(n_base + 8 <= rng_{frag_partial_b})")
                scatter_store.append(f"scatter_store_tail1(&dev_t3[dst], m_base, n_base, rng_{frag_partial_a}, rng_{frag_partial_b}, intra_stride, t3_frag[frag_idx]);\n")
            elif opt == 7 or opt == 8 :
                full_condition.append(f"(g_m_iter < rng_{reg_partial_a})")
                full_condition.append(f"(m_base + 8 <= rng_{frag_partial_a})")
                full_condition.append(f"(g_n_iter < rng_{reg_partial_b})")
                full_condition.append(f"(n_base + 8 <= rng_{frag_partial_b})")
                scatter_store.append(f"scatter_store_tail2(&dev_t3[dst], g_m_iter, m_base, g_n_iter, n_base, rng_{reg_partial_a}, rng_{frag_partial_a}, rng_{reg_partial_b}, rng_{frag_partial_b}, intra_stride, t3_frag[frag_idx]);\n")
        
        #
        full_condition.append("(((uintptr_t)dst & 0x1F) == 0)")
        full_condition.append("((intra_stride & 1) == 0)")
        str_full_condition = " && ".join(full_condition)
        str_scatter_store = " ".join(scatter_store)

        #
        f.write("\t" * tab + f"bool full = ({str_full_condition});\n")
        f.write("\t" * tab + f"if(full)\n")
        f.write("\t" * tab + "{\n"); tab += 1
        f.write("\t" * tab + f"nvcuda::wmma::store_matrix_sync(&dev_t3[dst], t3_frag[frag_idx], intra_stride, nvcuda::wmma::mem_row_major);\n"); tab -= 1
        f.write("\t" * tab + "}\n")
        f.write("\t" * tab + "else\n")
        f.write("\t" * tab + "{\n"); tab += 1
        f.write("\t" * tab + f"{str_scatter_store}"); tab -= 1
        f.write("\t" * tab + "}\n"); tab -= 1

    #
    for i in range(tab) :
        f.write("\t" * tab + "}\n"); tab -= 1
    
    f.write("}\n\n")

#
def tc_code_kernel_dev_scatter_store_head(f, kernel_name, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, opt, kernel_num=0) :
    #
    if opt == 0 :
        partial_a = collapsed_a[0]
        partial_b = collapsed_b[0]
    elif opt == 1 :
        partial_a = collapsed_a[0]
        reg_partial_b = ld_tile_order_b[0]
        frag_partial_b = ld_tile_order_b[1]
    elif opt == 2 :
        partial_b = collapsed_b[0]
        reg_partial_a = ld_tile_order_b[0]
        frag_partial_a = ld_tile_order_a[1]
    else :
        reg_partial_a = ld_tile_order_a[0]
        reg_partial_b = ld_tile_order_b[0]
        frag_partial_a = ld_tile_order_a[1]
        frag_partial_b = ld_tile_order_b[1]

    #
    arguments = []
    if opt == 0 :
        arguments.append(f"__device__ __forceinline__ void {kernel_name}(double *dev_t3")
    else :
        arguments.append(f"__device__ __forceinline__ void {kernel_name}{kernel_num}(double *dev_t3")

    #
    if opt == 0 :
        arguments.append("int m_base, int n_base")
        arguments.append(f"int rng_{partial_a}, int rng_{partial_b}") # 수정
    elif opt == 1 :
        #
        if kernel_num == 0 :
            arguments.append("int m_base, int g_n_iter")
            arguments.append(f"int rng_{partial_a}, int rng_{reg_partial_b}")
        elif kernel_num == 1 :
            arguments.append("int m_base, int n_base")
            arguments.append(f"int rng_{partial_a}, int rng_{frag_partial_b}")
        else :
            arguments.append("int m_base, int g_n_iter, int n_base")
            arguments.append(f"int rng_{partial_a}, int rng_{reg_partial_b}, int rng_{frag_partial_b}")
    elif opt == 2 :
        #
        if kernel_num == 0 :
            arguments.append("int g_m_iter, int n_base")
            arguments.append(f"int rng_{reg_partial_a}, int rng_{partial_b}")
        elif kernel_num == 1 :
            arguments.append("int m_base, int n_base")
            arguments.append(f"int rng_{frag_partial_a}, int rng_{partial_b}")
        else :
            arguments.append("int g_m_iter, int m_base, int n_base")
            arguments.append(f"int rng_{reg_partial_a}, int rng_{frag_partial_a}, int rng_{partial_b}")
    else :
        #
        if kernel_num == 0 :
            arguments.append("int g_m_iter, int g_n_iter")
            arguments.append(f"int rng_{reg_partial_a}, int rng_{reg_partial_b}")
        elif kernel_num == 1 :
            arguments.append("int m_base, int n_base")
            arguments.append(f"int rng_{frag_partial_a}, int rng_{frag_partial_b}")
        else :
            arguments.append("int g_m_iter, int m_base, int g_n_iter, int n_base")
            arguments.append(f"int rng_{reg_partial_a}, int rng_{frag_partial_a}, int rng_{reg_partial_b}, int rng_{frag_partial_b}")
    
    #
    arguments.append("int intra_stride,\nnvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double>&t3_frag)\n")

    #
    str_arguments = ", ".join(arguments)
    
    #
    f.write(str_arguments)
    f.write("{\n")

#
def tc_code_kernel_dev_scatter_store_body(f, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, opt, kernel_num=0) :
    #
    if opt == 0 :
        partial_a = collapsed_a[0]
        partial_b = collapsed_b[0]
    elif opt == 1 :
        partial_a = collapsed_a[0]
        reg_partial_b = ld_tile_order_b[0]
        frag_partial_b = ld_tile_order_b[1]
    elif opt == 2 :
        partial_b = collapsed_b[0]
        reg_partial_a = ld_tile_order_b[0]
        frag_partial_a = ld_tile_order_a[1]
    else :
        reg_partial_a = ld_tile_order_a[0]
        reg_partial_b = ld_tile_order_b[0]
        frag_partial_a = ld_tile_order_a[1]
        frag_partial_b = ld_tile_order_b[1]
    
    #
    f.write("\tconst int lane = threadIdx.x & 31;\n")
    f.write("\tconst int frag_m = lane >> 2;\n")
    f.write("\tconst int frag_n = (lane & 3) << 1;\n")

    #
    f.write("\tdouble2 reg = *reinterpret_cast<const double2*>(&t3_frag);\n")
    f.write("\tdouble x = reg.x;\n")
    f.write("\tdouble y = __shfl_xor_sync(0xFFFFFFFF, reg.y, 4);\n")
    f.write("\tint div4 = (lane & 7) >> 2;\n")

    #
    partial_condition = []
    if opt == 0 :
        partial_condition.append(f"(m_base + id_m) < rng_{partial_a}")
        partial_condition.append(f"(n_base + id_n) < rng_{partial_b}")
    elif opt == 1 :
        #
        if kernel_num == 0 :
            partial_condition.append(f"(m_base + id_m) < rng_{partial_a}")
            partial_condition.append(f"g_n_iter < rng_{reg_partial_b}")
        elif kernel_num == 1 :
            partial_condition.append(f"(m_base + id_m) < rng_{partial_a}")
            partial_condition.append(f"(n_base + id_n) < rng_{frag_partial_b}")
        else :
            partial_condition.append(f"(m_base + id_m) < rng_{partial_a}")
            partial_condition.append(f"g_n_iter < rng_{reg_partial_b}")
            partial_condition.append(f"(n_base + id_n) < rng_{frag_partial_b}")
    elif opt == 2 :
        #
        if kernel_num == 0 :
            partial_condition.append(f"g_m_iter < rng_{reg_partial_a}")
            partial_condition.append(f"(n_base + id_n) < rng_{partial_b}")
        elif kernel_num == 1 :
            partial_condition.append(f"(m_base + id_m) < rng_{frag_partial_a}")
            partial_condition.append(f"(n_base + id_n) < rng_{partial_b}")
        else :
            partial_condition.append(f"g_m_iter < rng_{reg_partial_a}")
            partial_condition.append(f"(m_base + id_m) < rng_{frag_partial_a}")
            partial_condition.append(f"(n_base + id_n) < rng_{partial_b}")
    else :
        #
        if kernel_num == 0 :
            partial_condition.append(f"g_m_iter < rng_{reg_partial_a}")
            partial_condition.append(f"g_n_iter < rng_{reg_partial_b}")
        elif kernel_num == 1 :
            partial_condition.append(f"(m_base + id_m) < rng_{frag_partial_a}")
            partial_condition.append(f"(n_base + id_n) < rng_{frag_partial_b}")
        else :
            partial_condition.append(f"g_m_iter < rng_{reg_partial_a}")
            partial_condition.append(f"(m_base + id_m) < rng_{frag_partial_a}")
            partial_condition.append(f"g_n_iter < rng_{reg_partial_b}")
            partial_condition.append(f"(n_base + id_n) < rng_{frag_partial_b}")

    #
    str_partial_condition = " && ".join(partial_condition)

    #
    f.write("\tint id_m = frag_m - div4;\n")
    f.write("\tint id_n = frag_n + div4;\n")
    f.write(f"\tif({str_partial_condition})\n")
    f.write("\t{\n")
    f.write("\t\tdev_t3[id_m * intra_stride + id_n] = (div4 ? y : x);\n")
    f.write("\t}\n\n")

    #
    f.write("\tid_m = frag_m + (div4 ^ 1);\n")
    f.write("\tid_n = frag_n + (div4 ^ 1);\n")
    f.write(f"\tif({str_partial_condition})\n")
    f.write("\t{\n")
    f.write("\t\tdev_t3[id_m * intra_stride + id_n] = ((div4 ^ 1) ? y : x);\n")
    f.write("\t}\n")

    f.write("}\n\n")

#
def tc_code_kernel_dev_scatter_store(f, kernel_name, fvi_flag, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, split_input) :
    #
    if fvi_flag == 1 :
        a_split_flag = split_input[1]
        b_split_flag = split_input[0]
    elif fvi_flag == 2 :
        a_split_flag = split_input[0]
        b_split_flag = split_input[1]
    
    #
    if a_split_flag and b_split_flag :
        opt = 0
    elif a_split_flag :
        opt = 1
    elif b_split_flag :
        opt = 2
    else :
        opt = 3
    
    #
    if opt == 0 :
        tc_code_kernel_dev_scatter_store_head(f, kernel_name, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, opt)
        tc_code_kernel_dev_scatter_store_body(f, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, opt)
    else :
        for i in range(3) :
            tc_code_kernel_dev_scatter_store_head(f, kernel_name, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, opt, i)
            tc_code_kernel_dev_scatter_store_body(f, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, opt, i)
    

# generate contraction kerenl
def tc_code_kernel(f, kernel_name, l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var,
                    l_inputs_addr, l_external_index, l_internal_index, l_input_strides, l_splited_indices_size,
                    fvi_flag, input_a, input_b, input_tensor_a, input_tensor_b, internal_order,
                    ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, ld_blk_index_a, ld_blk_index_b, SMEM_order_a, SMEM_order_b,
                    split_index, split_input, stride_helper, warp_shape, producer_cnt, double2_flag,
                    opt_pre_computed, data_type) :
    #
    kernel_variants = 2**max(len(collapsed_a), len(collapsed_b))

    #
    reg_padd_y, reg_padd_x = tc_code_kernel_dev_ld(f, "load_from_GM", l_t2_d_decl_var, l_v2_d_decl_var,
                                                l_external_index, l_internal_index, l_input_strides, l_splited_indices_size,
                                                fvi_flag, input_a, input_b, input_tensor_a, input_tensor_b, internal_order,
                                                ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, ld_blk_index_a, ld_blk_index_b, SMEM_order_a, SMEM_order_b,
                                                split_input, producer_cnt, double2_flag, kernel_variants, data_type)

    #
    tc_code_kernel_dev_compute(f, "compute_MMA", l_splited_indices_size,
                               fvi_flag, input_a, input_b, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, SMEM_order_a, SMEM_order_b,
                               split_input, warp_shape, double2_flag, reg_padd_y, reg_padd_x, kernel_variants)
    
    #
    tc_code_kernel_dev_scatter_store(f, "scatter_store_tail", fvi_flag, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, split_input)

    #
    for i in range(1, kernel_variants + 1) :
        #
        kernel_name_i = kernel_name + "_" + str(i)
        
        #
        tc_code_kernel_head(f, kernel_name_i, l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var,
                            l_external_index, l_internal_index, fvi_flag, input_a, input_b, opt_pre_computed)
        
        #
        tc_code_kernel_initial(f, l_external_index,
                                fvi_flag, input_a, input_b, internal_order,
                                ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b,
                                split_index, split_input, stride_helper, warp_shape, producer_cnt,
                                data_type, i)
        
        #
        tc_code_kernel_body(f, l_inputs_addr, l_external_index, l_internal_index,
                            fvi_flag, input_a, input_b, internal_order,
                            ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b,
                            split_input, producer_cnt, i)
        
    return reg_padd_y[0], reg_padd_x[0]