import tc_helper
from generators.kernels.kernel_load import tc_code_kernel_dev_ld
from generators.kernels.kernel_compute import tc_code_kernel_dev_compute
from generators.kernels.kernel_store import tc_code_kernel_dev_scatter_store
from generators.kernels.blk_opt import optimize_block_order


def _get_optimized_block_order(l_external_index, l_indices_size, l_splited_indices_size,
                               input_a, input_b, input_tensor_a, input_tensor_b,
                               internal_order, data_type):
    sizes = {idx: size for idx, size in l_indices_size}
    tile_sizes = {idx: size for idx, size in l_splited_indices_size}

    for idx in l_external_index:
        if idx not in tile_sizes and idx in sizes:
            tile_sizes[idx] = sizes[idx]

    try:
        result = optimize_block_order(
            output_indices=list(l_external_index),
            input_specs=[
                (list(input_tensor_a), input_a),
                (list(input_tensor_b), input_b),
            ],
            contraction_indices=list(internal_order),
            sizes=sizes,
            tile_sizes=tile_sizes,
            element_size=8 if data_type == "DOUBLE" else 4,
        )
        optimized_order = result.order
    except Exception:
        optimized_order = []

    if sorted(optimized_order) != sorted(l_external_index):
        return list(reversed(l_external_index))

    return optimized_order


def _write_block_idx_decomposition(f, block_order):
    if len(block_order) == 1:
        f.write(f"\tconst int blk_idx_{block_order[0]} = blockIdx.x;\n\n")
        return

    if len(block_order) == 2:
        outer_dim, inner_dim = block_order
        f.write(f"\tconst int blk_idx_{outer_dim} = blockIdx.x / numBlk_{inner_dim};\n")
        f.write(f"\tconst int blk_idx_{inner_dim} = blockIdx.x % numBlk_{inner_dim};\n\n")
        return

    f.write("\tint tmp_blkIdx;\n")
    for idx, dim in enumerate(block_order):
        inner_dims = block_order[idx + 1:]
        if inner_dims:
            divisor = " * ".join([f"numBlk_{inner_dim}" for inner_dim in inner_dims])
            if idx == 0:
                f.write(f"\tconst int blk_idx_{dim} = blockIdx.x / ({divisor});\n")
                f.write(f"\ttmp_blkIdx = blockIdx.x % ({divisor});\n")
            else:
                f.write(f"\tconst int blk_idx_{dim} = tmp_blkIdx / ({divisor});\n")
                f.write(f"\ttmp_blkIdx = tmp_blkIdx % ({divisor});\n")
        else:
            f.write(f"\tconst int blk_idx_{dim} = tmp_blkIdx;\n")

        f.write("\n")

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
def tc_code_kernel_initial(f, l_external_index, l_indices_size, l_splited_indices_size,
                            fvi_flag, input_a, input_b, input_tensor_a, input_tensor_b, internal_order,
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
        f.write("\tauto pipeline = cuda::make_pipeline(block, &shared_state);\n")
    
    f.write("\n")

    # shared memory
    f.write("\t// For Shared Memory\n")
    if data_type == "DOUBLE" :
        f.write("\textern __shared__ __align__(128) double shared_memory[];\n")
        f.write(f"\tdouble *sm_{input_a} = shared_memory;\n")
        f.write(f"\tdouble *sm_{input_b} = shared_memory + (PIPELINE_STAGES * {input_a}_tile_size);\n\n")
    else :
        f.write("\textern __shared__ __align__(128) float shared_memory[];\n")
        f.write(f"\tfloat *sm_{input_a} = shared_memory;\n")
        f.write(f"\tfloat *sm_{input_b} = shared_memory + (PIPELINE_STAGES * {input_a}_tile_size);\n\n")

    # index for thread blocks
    f.write("\t// Index for each Thread Blocks\n")
    block_order = _get_optimized_block_order(
        l_external_index, l_indices_size, l_splited_indices_size,
        input_a, input_b, input_tensor_a, input_tensor_b,
        internal_order, data_type
    )
    f.write(f"\t// block order (outer -> inner): {' -> '.join(block_order)}\n")
    _write_block_idx_decomposition(f, block_order)

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
    if data_type == "DOUBLE" :
        shift = 3
    else :
        shift = 4

    f.write(f"\tconst int {input_a}_frag_cnt = TILE_{ld_tile_order_a[1].capitalize()} >> {shift};\n")
    f.write(f"\tconst int {input_b}_frag_cnt = TILE_{ld_tile_order_b[1].capitalize()} >> {shift};\n\n")
        

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
    if data_type == "DOUBLE" :
        f.write(f"\tnvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> t3_frag[wmiter * wniter * {input_a}_frag_cnt * {input_b}_frag_cnt];\n\n")
    else :
        f.write(f"\tnvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> t3_frag[wmiter * wniter * {input_a}_frag_cnt * {input_b}_frag_cnt];\n\n")

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
        # f.write("\tint internal_upperbound = 0;\n")
        # f.write("\tint internal_offset;\n")
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

    # if data_type == "DOUBLE" :
    #     pass
    # else :
    if opt > 1 :
        size_a = []
        for idx in input_tensor_a :
            size_a.append(f"size_{idx}")
        str_size_a = "*".join(size_a)
        f.write(f"\tconst int size_tensor_{input_a} = {str_size_a};\n")

        size_b = []
        for idx in input_tensor_b :
            size_b.append(f"size_{idx}")
        str_size_b = "*".join(size_b)
        f.write(f"\tconst int size_tensor_{input_b} = {str_size_b};\n\n")

#
def tc_code_kernel_body(f, l_inputs_addr, l_external_index, l_internal_index, l_splited_indices_size,
                        fvi_flag, input_a, input_b, internal_order,
                        ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b,
                        split_input, producer_cnt, opt, data_type) :
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

    #
    f.write("\t" * tab + "int next_shm_offset = (shm_offset + PIPELINE_STAGES - 1) % PIPELINE_STAGES;\n")

    #
    # if opt % 2 == 0 :
    #     f.write("\t" * tab + f"int load_ub = max(0, (load_idx + TILE_UNIT - size_internal));\n\n")
    # else :
    #     f.write("\n")

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
    # if data_type == "DOUBLE" :
    #     #
    #     if opt == 1 :
    #         f.write(");\n")
    #     elif opt == 2 :
    #         f.write(",\n")
    #         f.write("\t" * (tab + 2) + "load_ub);\n")
    #     elif opt == 3 :
    #         f.write(",\n")
    #         f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[0]}, rng_{reg_partial_index[1]});\n")
    #     elif opt == 4 :
    #         f.write(",\n")
    #         f.write("\t" * (tab + 2) + "load_ub,\n")
    #         f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[0]}, rng_{reg_partial_index[1]});\n")
    #     elif opt == 5 :
    #         f.write(",\n")
    #         f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]});\n")
    #     elif opt == 6 :
    #         f.write(",\n")
    #         f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]},\n")
    #         f.write("\t" * (tab + 2) + "load_ub);\n")
    #     elif opt == 7 :
    #         f.write(",\n")
    #         if a_split_flag :
    #             f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]},\n")
    #             f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[1]});\n")
    #         elif b_split_flag :
    #             f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]},\n")
    #             f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[0]});\n")
    #         else :
    #             f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]},\n")
    #             f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[0]}, rng_{reg_partial_index[1]});\n")
    #     elif opt == 8 :
    #         f.write(",\n")
    #         if a_split_flag :
    #             f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]},\n")
    #             f.write("\t" * (tab + 2) + "load_ub,\n")
    #             f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[1]});\n")
    #         elif b_split_flag :
    #             f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]},\n")
    #             f.write("\t" * (tab + 2) + "load_ub,\n")
    #             f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[0]});\n")
    #         else :
    #             f.write("\t" * (tab + 2) + f"rng_{frag_partial_index[0]}, rng_{frag_partial_index[1]},\n")
    #             f.write("\t" * (tab + 2) + "load_ub,\n")
    #             f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[0]}, rng_{reg_partial_index[1]});\n")
    # else :
    #
    if opt == 1 :
        f.write(");\n")
    else :
        f.write(",\n")
        f.write("\t" * (tab + 2) + f"size_tensor_{input_a}, size_tensor_{input_b});\n")
        

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
        # if data_type == "DOUBLE" :
        #     f.write("\t" * tab + f"int compute_ub = max(0, (l + TILE_UNIT - size_internal));\n\n")
        # else :
        f.write("\t" * tab + f"int compute_ub = TILE_UNIT - max(0, (l + TILE_UNIT - size_internal));\n\n")

    f.write("\t" * tab + "pipeline.consumer_wait();\n")
    f.write("\t" * tab + f"compute_MMA_{opt}(sm_{input_a}, sm_{input_b}, wmiter, wniter, wrow, wcol,\n")
    f.write("\t" * (tab + 2) + f"{input_a}_frag_cnt, {input_b}_frag_cnt,\n")
    f.write("\t" * (tab + 2) + "t3_frag,\n")
    f.write("\t" * (tab + 2) + f"shm_offset * {input_a}_tile_size, shm_offset * {input_b}_tile_size")

    #
    if data_type == "DOUBLE" :
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
    else :
        if opt == 1 or opt == 5 :
            f.write(");\n")
        elif opt == 2 or opt == 6 :
            f.write(",\n")
            f.write("\t" * (tab + 2) + "compute_ub);\n")
        elif opt == 3 or opt == 7 :
            f.write(",\n")
            f.write("\t" * (tab + 2) + f"rng_{reg_partial_index[0]}, rng_{reg_partial_index[1]});\n")
        elif opt == 4 or opt == 8 :
            f.write(",\n")
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
    # if data_type == "DOUBLE" :
    #     f.write("\t" * tab + "int frag_idx = 0;\n")

    #     #
    #     f.write("\t" * tab + "#pragma unroll\n")
    #     f.write("\t" * tab + f"for(int iter_{input_a} = 0; iter_{input_a} < wmiter; iter_{input_a}++)\n")
    #     f.write("\t" * tab + "{\n"); tab += 1
    #     f.write("\t" * tab + f"int base = t3_base_thread + iter_{input_a} * inter_m_stride;\n")

    #     #
    #     if a_split_flag :
    #         if opt > 2 :
    #             f.write("\t" * tab + f"int g_m_iter = (wrow + iter_{input_a}) * TILE_{ld_tile_order_a[1].capitalize()};\n")
    #     else :
    #         if opt == 3 or opt == 4 or opt == 7 or opt == 8 : # reg mapped partial
    #             f.write("\t" * tab + f"int g_m_iter = wrow + iter_{input_a};\n")

    #     #
    #     f.write("\t" * tab + "#pragma unroll\n")
    #     f.write("\t" * tab + f"for(int iter_{input_b} = 0; iter_{input_b} < wniter; iter_{input_b}++)\n")
    #     f.write("\t" * tab + "{\n"); tab += 1
    #     f.write("\t" * tab + f"int base_iter = base + iter_{input_b} * inter_n_stride;\n")
        
    #     #
    #     if b_split_flag :
    #         if opt > 2 :
    #             f.write("\t" * tab + f"int g_n_iter = (wcol + iter_{input_b}) * TILE_{ld_tile_order_b[1].capitalize()};\n")
    #     else :
    #         if opt == 3 or opt == 4 or opt == 7 or opt == 8 : # reg mapped partial
    #             f.write("\t" * tab + f"int g_n_iter = wcol + iter_{input_b};\n")

    #     #
    #     f.write("\t" * tab + "#pragma unroll\n")
    #     f.write("\t" * tab + f"for(int cnt_{input_a} = 0; cnt_{input_a} < {input_a}_frag_cnt; cnt_{input_a}++)\n")
    #     f.write("\t" * tab + "{\n"); tab += 1

    #     #
    #     if a_split_flag :
    #         if opt > 2 :
    #             f.write("\t" * tab + f"int m_base = g_m_iter + (cnt_{input_a} << 3);\n")
    #     else :
    #         f.write("\t" * tab + f"int m_base = (cnt_{input_a} << 3);\n") 
        
    #     #
    #     f.write("\t" * tab + f"int base_{input_a} = base_iter + (cnt_{input_a} << 3) * intra_stride;\n")

    #     #
    #     f.write("\t" * tab + "#pragma unroll\n")
    #     f.write("\t" * tab + f"for(int cnt_{input_b} = 0; cnt_{input_b} < {input_b}_frag_cnt; cnt_{input_b}++, frag_idx++)\n")
    #     f.write("\t" * tab + "{\n"); tab += 1

    #     #
    #     if b_split_flag :
    #         if opt > 2 :
    #             f.write("\t" * tab + f"int n_base = g_n_iter + (cnt_{input_b} << 3);\n")
    #     else :
    #         f.write("\t" * tab + f"int n_base = (cnt_{input_b} << 3);\n")

    #     #
    #     f.write("\t" * tab + f"int dst = base_{input_a} + (cnt_{input_b} << 3);\n")

    #     #
    #     if a_split_flag and b_split_flag :
    #         partial_a = collapsed_a[0]
    #         partial_b = collapsed_b[0]
    #     elif a_split_flag :
    #         if opt > 2 :
    #             partial_a = collapsed_a[0]
    #         if opt == 3 or opt == 4 or opt == 7 or opt == 8 :
    #             reg_partial_b = ld_tile_order_b[0]
    #         if opt > 4 :
    #             frag_partial_b = ld_tile_order_b[1]
    #     elif b_split_flag :
    #         if opt > 2 :
    #             partial_b = collapsed_b[0]
    #         if opt == 3 or opt == 4 or opt == 7 or opt == 8 :
    #             reg_partial_a = ld_tile_order_a[0]
    #         if opt > 4 :
    #             frag_partial_a = ld_tile_order_a[1]
    #     else :
    #         if opt == 3 or opt == 4 or opt == 7 or opt == 8 :
    #             reg_partial_a = ld_tile_order_a[0]
    #             reg_partial_b = ld_tile_order_b[0]
    #         if opt > 4 :
    #             frag_partial_a = ld_tile_order_a[1]
    #             frag_partial_b = ld_tile_order_b[1]

    #     #
    #     if opt == 1 or opt == 2 :
    #         f.write("\t" * tab + "nvcuda::wmma::store_matrix_sync(&dev_t3[dst], t3_frag[frag_idx], intra_stride, nvcuda::wmma::mem_row_major);\n"); tab -= 1
    #     else :
    #         #
    #         full_condition = []
    #         scatter_store = []
            
    #         #
    #         if a_split_flag and b_split_flag :
    #             full_condition.append(f"(m_base + 8 <= rng_{partial_a})")
    #             full_condition.append(f"(n_base + 8 <= rng_{partial_b})")
    #             scatter_store.append(f"scatter_store_tail(&dev_t3[dst], m_base, n_base, rng_{partial_a}, rng_{partial_b}, intra_stride, t3_frag[frag_idx]);\n")
    #         elif a_split_flag :
    #             #
    #             if opt == 3 or opt == 4 :
    #                 full_condition.append(f"(m_base + 8 <= rng_{partial_a})")
    #                 full_condition.append(f"(g_n_iter < rng_{reg_partial_b})")
    #                 scatter_store.append(f"scatter_store_tail0(&dev_t3[dst], m_base, g_n_iter, rng_{partial_a}, rng_{reg_partial_b}, intra_stride, t3_frag[frag_idx]);\n")
    #             elif opt == 5 or opt == 6 :
    #                 full_condition.append(f"(m_base + 8 <= rng_{partial_a})")
    #                 full_condition.append(f"(n_base + 8 <= rng_{frag_partial_b})")
    #                 scatter_store.append(f"scatter_store_tail1(&dev_t3[dst], m_base, n_base, rng_{partial_a}, rng_{frag_partial_b}, intra_stride, t3_frag[frag_idx]);\n")
    #             elif opt == 7 or opt == 8 :
    #                 full_condition.append(f"(m_base + 8 <= rng_{partial_a})")
    #                 full_condition.append(f"(g_n_iter < rng_{reg_partial_b})")
    #                 full_condition.append(f"(n_base + 8 <= rng_{frag_partial_b})")
    #                 scatter_store.append(f"scatter_store_tail2(&dev_t3[dst], m_base, g_n_iter, n_base, rng_{partial_a}, rng_{reg_partial_b}, rng_{frag_partial_b}, intra_stride, t3_frag[frag_idx]);\n")
    #         elif b_split_flag :
    #             #
    #             if opt == 3 or opt == 4 :
    #                 full_condition.append(f"(g_m_iter < rng_{reg_partial_a})")
    #                 full_condition.append(f"(n_base + 8 <= rng_{partial_b})")
    #                 scatter_store.append(f"scatter_store_tail0(&dev_t3[dst], g_m_iter, n_base, rng_{reg_partial_a}, rng_{partial_b}, intra_stride, t3_frag[frag_idx]);\n")
    #             elif opt == 5 or opt == 6 :
    #                 full_condition.append(f"(m_base + 8 <= rng_{frag_partial_a})")
    #                 full_condition.append(f"(n_base + 8 <= rng_{partial_b})")
    #                 scatter_store.append(f"scatter_store_tail1(&dev_t3[dst], m_base, n_base, rng_{frag_partial_a}, rng_{partial_b}, intra_stride, t3_frag[frag_idx]);\n")
    #             elif opt == 7 or opt == 8 :
    #                 full_condition.append(f"(g_m_iter < rng_{reg_partial_a})")
    #                 full_condition.append(f"(m_base + 8 <= rng_{frag_partial_a})")
    #                 full_condition.append(f"(n_base + 8 <= rng_{partial_b})")
    #                 scatter_store.append(f"scatter_store_tail2(&dev_t3[dst], g_m_iter, m_base, n_base, rng_{reg_partial_a}, rng_{frag_partial_a}, rng_{partial_b}, intra_stride, t3_frag[frag_idx]);\n")
    #         else :
    #             #
    #             if opt == 3 or opt == 4 :
    #                 full_condition.append(f"(g_m_iter < rng_{reg_partial_a})")
    #                 full_condition.append(f"(g_n_iter < rng_{reg_partial_b})")
    #                 scatter_store.append(f"scatter_store_tail0(&dev_t3[dst], g_m_iter, g_n_iter, rng_{reg_partial_a}, rng_{reg_partial_b}, intra_stride, t3_frag[frag_idx]);\n")
    #             elif opt == 5 or opt == 6 :
    #                 full_condition.append(f"(m_base + 8 <= rng_{frag_partial_a})")
    #                 full_condition.append(f"(n_base + 8 <= rng_{frag_partial_b})")
    #                 scatter_store.append(f"scatter_store_tail1(&dev_t3[dst], m_base, n_base, rng_{frag_partial_a}, rng_{frag_partial_b}, intra_stride, t3_frag[frag_idx]);\n")
    #             elif opt == 7 or opt == 8 :
    #                 full_condition.append(f"(g_m_iter < rng_{reg_partial_a})")
    #                 full_condition.append(f"(m_base + 8 <= rng_{frag_partial_a})")
    #                 full_condition.append(f"(g_n_iter < rng_{reg_partial_b})")
    #                 full_condition.append(f"(n_base + 8 <= rng_{frag_partial_b})")
    #                 scatter_store.append(f"scatter_store_tail2(&dev_t3[dst], g_m_iter, m_base, g_n_iter, n_base, rng_{reg_partial_a}, rng_{frag_partial_a}, rng_{reg_partial_b}, rng_{frag_partial_b}, intra_stride, t3_frag[frag_idx]);\n")
            
    #         #
    #         full_condition.append("(((uintptr_t)dst & 0x1F) == 0)")
    #         full_condition.append("((intra_stride & 1) == 0)")
    #         str_full_condition = " && ".join(full_condition)
    #         str_scatter_store = " ".join(scatter_store)

    #         #
    #         f.write("\t" * tab + f"bool full = ({str_full_condition});\n")
    #         f.write("\t" * tab + f"if(full)\n")
    #         f.write("\t" * tab + "{\n"); tab += 1
    #         f.write("\t" * tab + f"nvcuda::wmma::store_matrix_sync(&dev_t3[dst], t3_frag[frag_idx], intra_stride, nvcuda::wmma::mem_row_major);\n"); tab -= 1
    #         f.write("\t" * tab + "}\n")
    #         f.write("\t" * tab + "else\n")
    #         f.write("\t" * tab + "{\n"); tab += 1
    #         f.write("\t" * tab + f"{str_scatter_store}"); tab -= 1
    #         f.write("\t" * tab + "}\n"); tab -= 1

    #     #
    #     for i in range(tab) :
    #         f.write("\t" * tab + "}\n"); tab -= 1
    # #
    # else :
    reg_partial_a = ld_tile_order_a[0]
    reg_partial_b = ld_tile_order_b[0]
    frag_partial_a = ld_tile_order_a[1]
    frag_partial_b = ld_tile_order_b[1]

    left_frag_size = tc_helper.tc_helper_find_value(l_splited_indices_size, ld_tile_order_a[1])
    right_frag_size = tc_helper.tc_helper_find_value(l_splited_indices_size, ld_tile_order_b[1])

    if data_type == "DOUBLE" :
        frag_size = 8
        cnt_offset = 3
    else :
        frag_size = 16
        cnt_offset = 4

    f.write("\t" * tab + "#pragma unroll\n")
    f.write("\t" * tab + f"for(int iter_{input_a} = 0; iter_{input_a} < wmiter; iter_{input_a}++)\n")
    f.write("\t" * tab + "{\n"); tab += 1
    
    #
    if a_split_flag :
        if opt > 2 :
            f.write("\t" * tab + f"int g_m_iter = (wrow + iter_{input_a}) * TILE_{frag_partial_a.capitalize()};\n")
            f.write("\t" * tab + f"if(g_m_iter >= rng_{reg_partial_a[0]})\n")
            f.write("\t" * (tab + 1) + "continue;\n")
            f.write("\t" * tab + f"int valid_{frag_partial_a} = min(rng_{reg_partial_a[0]} - g_m_iter, TILE_{frag_partial_a.capitalize()});\n")
    else :
        if opt == 3 or opt == 4 or opt == 7 or opt == 8 : # reg mapped partial
            f.write("\t" * tab + f"int g_m_iter = wrow + iter_{input_a};\n")
            f.write("\t" * tab + f"if(g_m_iter >= rng_{reg_partial_a})\n")
            f.write("\t" * (tab + 1) + "continue;\n")
    
    f.write("\t" * tab + f"int base = t3_base_thread + iter_{input_a} * inter_m_stride;\n")
    f.write("\t" * tab + f"int frag_idx0 = iter_{input_a} * wniter;\n\n")

    #
    f.write("\t" * tab + "#pragma unroll\n")
    f.write("\t" * tab + f"for(int iter_{input_b} = 0; iter_{input_b} < wniter; iter_{input_b}++)\n")
    f.write("\t" * tab + "{\n"); tab += 1
    
    #
    if b_split_flag :
        if opt > 2 :
            f.write("\t" * tab + f"int g_n_iter = (wcol + iter_{input_b}) * TILE_{frag_partial_b.capitalize()};\n")
            f.write("\t" * tab + f"if(g_n_iter >= rng_{reg_partial_b[0]})\n")
            f.write("\t" * (tab + 1) + "continue;\n")
            f.write("\t" * tab + f"int valid_{frag_partial_b} = min(rng_{reg_partial_b[0]} - g_n_iter, TILE_{frag_partial_b.capitalize()});\n")
    else :
        if opt == 3 or opt == 4 or opt == 7 or opt == 8 : # reg mapped partial
            f.write("\t" * tab + f"int g_n_iter = wcol + iter_{input_b};\n")
            f.write("\t" * tab + f"if(g_n_iter >= rng_{reg_partial_b})\n")
            f.write("\t" * (tab + 1) + "continue;\n")

    f.write("\t" * tab + f"int base_iter = base + iter_{input_b} * inter_n_stride;\n")
    f.write("\t" * tab + f"int frag_idx1 = frag_idx0 + iter_{input_b};\n\n")

    #
    if left_frag_size == frag_size * 2 :
        f.write("\t" * tab + "#pragma unroll\n")
        f.write("\t" * tab + f"for(int cnt_{input_a} = 0; cnt_{input_a} < {input_a}_frag_cnt; cnt_{input_a}++)\n")
        f.write("\t" * tab + "{\n"); tab += 1
        f.write("\t" * tab + f"int m_base = (cnt_{input_a} << {cnt_offset});\n")

        #
        if a_split_flag :
            if opt > 2 :
                f.write("\t" * tab + f"int frag_valid_rows = valid_{frag_partial_a} - m_base;\n")
                f.write("\t" * tab + "if(frag_valid_rows <= 0)\n")
                f.write("\t" * (tab + 1) + "continue;\n")
                f.write("\t" * tab + f"frag_valid_rows = min(frag_valid_rows, {frag_size});\n")
        #
        else :
            if opt == 5 or opt == 6 or opt == 7 or opt == 8 :
                f.write("\t" * tab + f"if(m_base >= rng_{frag_partial_a})\n")
                f.write("\t" * (tab + 1) + "continue;\n")
                f.write("\t" * tab + f"int frag_valid_rows = min(rng_{frag_partial_a} - m_base, {frag_size});\n")
    
        f.write("\t" * tab + f"int base_{input_a} = base_iter + m_base * intra_stride;\n")
        f.write("\t" * tab + f"int frag_idx2 = (frag_idx1 * {input_a}_frag_cnt) + cnt_{input_a};\n\n")
    #
    else :
        if a_split_flag :
            if opt > 2 :
                f.write("\t" * tab + f"int frag_valid_rows = valid_{frag_partial_a};\n")
                f.write("\t" * tab + f"frag_valid_rows = min(frag_valid_rows, {frag_size});\n")
        else :
            if opt == 5 or opt == 6 or opt == 7 or opt == 8 :
                f.write("\t" * tab + f"int frag_valid_rows = min(rng_{frag_partial_a}, {frag_size});\n")

        f.write("\t" * tab + f"int base_{input_a} = base_iter;\n")

    #
    if right_frag_size == frag_size * 2 :
        f.write("\t" * tab + "#pragma unroll\n")
        f.write("\t" * tab + f"for(int cnt_{input_b} = 0; cnt_{input_b} < {input_b}_frag_cnt; cnt_{input_b}++)\n")
        f.write("\t" * tab + "{\n"); tab += 1
        f.write("\t" * tab + f"int n_base = (cnt_{input_b} << {cnt_offset});\n")

        #
        if b_split_flag :
            if opt > 2 :
                f.write("\t" * tab + f"int frag_valid_cols = valid_{frag_partial_b} - n_base;\n")
                f.write("\t" * tab + "if(frag_valid_cols <= 0)\n")
                f.write("\t" * (tab + 1) + "continue;\n")
                f.write("\t" * tab + f"frag_valid_cols = min(frag_valid_cols, {frag_size});\n")
        #
        else :
            if opt == 5 or opt == 6 or opt == 7 or opt == 8 :
                f.write("\t" * tab + f"if(n_base >= rng_{frag_partial_b})\n")
                f.write("\t" * (tab + 1) + "continue;\n")
                f.write("\t" * tab + f"int frag_valid_cols = min(rng_{frag_partial_b} - n_base, {frag_size});\n")

        f.write("\t" * tab + f"int dst = base_{input_a} + n_base;\n")

        if left_frag_size == frag_size * 2 :
            f.write("\t" * tab + f"int frag_idx = (frag_idx2 * {input_b}_frag_cnt) + cnt_{input_b};\n")
        else :
            f.write("\t" * tab + f"int frag_idx = (frag_idx1 * {input_b}_frag_cnt) + cnt_{input_b};\n")
    #
    else :
        if b_split_flag :
            if opt > 2 :
                f.write("\t" * tab + f"int frag_valid_cols = valid_{frag_partial_b};\n")
                f.write("\t" * tab + f"frag_valid_cols = min(frag_valid_cols, {frag_size});\n")
        else :
            if opt == 5 or opt == 6 or opt == 7 or opt == 8 :
                f.write("\t" * tab + f"int frag_valid_cols = min(rng_{frag_partial_b}, {frag_size});\n")

        if left_frag_size == frag_size * 2 :
            f.write("\t" * tab + f"int frag_idx = frag_idx2;\n")
        else :
            f.write("\t" * tab + f"int frag_idx = frag_idx1;\n")
        f.write("\t" * tab + f"int dst = base_{input_a};\n")

    #
    if opt == 1 or opt == 2 :
        f.write("\t" * tab + "nvcuda::wmma::store_matrix_sync(&dev_t3[dst], t3_frag[frag_idx], intra_stride, nvcuda::wmma::mem_row_major);\n"); tab -= 1
    #
    else :
        full_condition = []
        scatter_store = []
        
        #
        if a_split_flag and b_split_flag :
            full_condition.append(f"(frag_valid_rows == {frag_size})")
            full_condition.append(f"(frag_valid_cols == {frag_size})")
            scatter_store.append("scatter_store_tail2(&dev_t3[dst], frag_valid_rows, frag_valid_cols, intra_stride, t3_frag[frag_idx]);\n")
        #
        elif a_split_flag :
            #
            if opt == 3 or opt == 4 :
                full_condition.append(f"(frag_valid_rows == {frag_size})")
                scatter_store.append("scatter_store_tail0(&dev_t3[dst], frag_valid_rows, intra_stride, t3_frag[frag_idx]);\n")
            #
            elif opt == 5 or opt == 6 or opt == 7 or opt == 8 :
                full_condition.append(f"(frag_valid_rows == {frag_size})")
                full_condition.append(f"(frag_valid_cols == {frag_size})")
                scatter_store.append("scatter_store_tail2(&dev_t3[dst], frag_valid_rows, frag_valid_cols, intra_stride, t3_frag[frag_idx]);\n")
        #
        elif b_split_flag :
            #
            if opt == 3 or opt == 4 :
                full_condition.append(f"(frag_valid_cols == {frag_size})")
                scatter_store.append("scatter_store_tail1(&dev_t3[dst], frag_valid_cols, intra_stride, t3_frag[frag_idx]);\n")
            #
            elif opt == 5 or opt == 6 or opt == 7 or opt == 8:
                full_condition.append(f"(frag_valid_rows == {frag_size})")
                full_condition.append(f"(frag_valid_cols == {frag_size})")
                scatter_store.append("scatter_store_tail2(&dev_t3[dst], frag_valid_rows, frag_valid_cols, intra_stride, t3_frag[frag_idx]);\n")
        #
        else :
            #
            if opt == 5 or opt == 6 or opt == 7 or opt == 8 :
                full_condition.append(f"(frag_valid_rows == {frag_size})")
                full_condition.append(f"(frag_valid_cols == {frag_size})")
                scatter_store.append("scatter_store_tail2(&dev_t3[dst], frag_valid_rows, frag_valid_cols, intra_stride, t3_frag[frag_idx]);\n")
        
        #
        if data_type == "DOUBLE" :
            full_condition.append("(((uintptr_t)dst & 0xF) == 0)")
        else :
            full_condition.append("(((uintptr_t)dst & 0x7) == 0)")
        full_condition.append("((intra_stride & 0x1) == 0)")
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

# generate contraction kerenl
def tc_code_kernel(f, kernel_name, l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var, l_indices_size,
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
                               split_input, warp_shape, double2_flag, reg_padd_y, reg_padd_x, kernel_variants, data_type)
    
    #
    tc_code_kernel_dev_scatter_store(f, "scatter_store_tail", fvi_flag, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, split_input, data_type)

    #
    for i in range(1, kernel_variants + 1) :
        #
        kernel_name_i = kernel_name + "_" + str(i)
        
        #
        tc_code_kernel_head(f, kernel_name_i, l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var,
                            l_external_index, l_internal_index, fvi_flag, input_a, input_b, opt_pre_computed)
        
        #
        tc_code_kernel_initial(f, l_external_index, l_indices_size, l_splited_indices_size,
                                fvi_flag, input_a, input_b, input_tensor_a, input_tensor_b, internal_order,
                                ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b,
                                split_index, split_input, stride_helper, warp_shape, producer_cnt,
                                data_type, i)
        
        #
        tc_code_kernel_body(f, l_inputs_addr, l_external_index, l_internal_index, l_splited_indices_size,
                            fvi_flag, input_a, input_b, internal_order,
                            ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b,
                            split_input, producer_cnt, i, data_type)
        
    return reg_padd_y[0], reg_padd_x[0]
