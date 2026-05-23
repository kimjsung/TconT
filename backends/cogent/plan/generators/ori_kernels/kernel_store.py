#
def tc_code_kernel_dev_scatter_store_head(f, kernel_name, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, opt, data_type, kernel_num=0) :
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
        reg_partial_a = ld_tile_order_a[0]
        frag_partial_a = ld_tile_order_a[1]
    else :
        reg_partial_a = ld_tile_order_a[0]
        reg_partial_b = ld_tile_order_b[0]
        frag_partial_a = ld_tile_order_a[1]
        frag_partial_b = ld_tile_order_b[1]
    
    #
    arguments = []
    if opt == 0 :
        if data_type == "DOUBLE" :
            arguments.append(f"__device__ __forceinline__ void {kernel_name}(double *__restrict__ dev_t3")
        else :
            arguments.append(f"__device__ __forceinline__ void {kernel_name}(float *__restrict__ dev_t3")  
    else :
        if data_type == "DOUBLE" :
            arguments.append(f"__device__ __forceinline__ void {kernel_name}{kernel_num}(double *__restrict__ dev_t3")
        else :
            arguments.append(f"__device__ __forceinline__ void {kernel_name}{kernel_num}(float *__restrict__ dev_t3")

    #
    if opt == 0 :
        arguments.append("int m_base, int n_base")
        arguments.append(f"int rng_{partial_a}, int rng_{partial_b}")
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
    if data_type == "DOUBLE" :
        arguments.append("int intra_stride,\nnvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double>&t3_frag)\n")
    else :
        arguments.append("int intra_stride,\nnvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float>&t3_frag)\n")

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
        reg_partial_a = ld_tile_order_a[0]
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
def tc_code_kernel_dev_scatter_store_head_fp32(f, kernel_name, ld_tile_order_a, ld_tile_order_b, opt, data_type) :
    #
    frag_partial_a = ld_tile_order_a[1]
    frag_partial_b = ld_tile_order_b[1]
    
    #
    arguments = []

    #
    arguments.append(f"__device__ __forceinline__ void {kernel_name}{opt}(float *__restrict__ dev_t3")
    
    #
    if opt == 0 :
        arguments.append(f"int rng_{frag_partial_a}")
    elif opt == 1 :
        arguments.append(f"int rng_{frag_partial_b}")
    else :
        arguments.append(f"int rng_{frag_partial_a}, int rng_{frag_partial_b}")
    
    
    #
    if data_type == "DOUBLE" :
        arguments.append("int intra_stride,\nnvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double>&t3_frag)\n")
    else :
        arguments.append("int intra_stride,\nnvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float>&t3_frag)\n")

    #
    str_arguments = ", ".join(arguments)
    
    #
    f.write(str_arguments)
    f.write("{\n")

#
def tc_code_kernel_dev_scatter_store_body_fp32(f, ld_tile_order_a, ld_tile_order_b, opt) :
    #
    frag_partial_a = ld_tile_order_a[1]
    frag_partial_b = ld_tile_order_b[1]
    
    #
    f.write("\tconst int lane = threadIdx.x & 31;\n")
    f.write("\tconst int frag_m = lane >> 2;\n")
    f.write("\tconst int frag_n = (lane & 3) << 1;\n")
    f.write("\tconst bool div = ((((uintptr_t)(&dev_t3[frag_m * intra_stride])) & 0x7) == 0);\n")

    #
    for i in range(4) :
        f.write(f"\tfloat2 reg{i} = reinterpret_cast<const float2*>(&t3_frag)[{i}];\n")
    f.write("\n")

    #
    if opt == 0 :                           # FRAG_X : full / FRAG_Y : partial
        f.write(f"\tconst bool row_lo = (frag_m < rng_{frag_partial_a});\n")
        f.write(f"\tconst bool row_hi = (frag_m + 8 < rng_{frag_partial_a});\n")
    #
    elif opt == 1 :                         # FRAG_X : partial / FRAG_Y : full
        f.write(f"\tconst bool col_lo_2 = (frag_n + 1 < rng_{frag_partial_b});\n")
        f.write(f"\tconst bool col_lo_1 = (frag_n < rng_{frag_partial_b});\n")
        f.write(f"\tconst bool col_hi_2 = (frag_n + 9 < rng_{frag_partial_b});\n")
        f.write(f"\tconst bool col_hi_1 = (frag_n + 8 < rng_{frag_partial_b});\n")
    #
    else :                                  # FRAG_X : partial / FRAG_Y : partial
        f.write(f"\tconst bool row_lo = (frag_m < rng_{frag_partial_a});\n")
        f.write(f"\tconst bool row_hi = (frag_m + 8 < rng_{frag_partial_a});\n")
        f.write(f"\tconst bool col_lo_2 = (frag_n + 1 < rng_{frag_partial_b});\n")
        f.write(f"\tconst bool col_lo_1 = (frag_n < rng_{frag_partial_b});\n")
        f.write(f"\tconst bool col_hi_2 = (frag_n + 9 < rng_{frag_partial_b});\n")
        f.write(f"\tconst bool col_hi_1 = (frag_n + 8 < rng_{frag_partial_b});\n")
    f.write("\n")
    
    #
    if opt == 0 :
        f.write("\tif(row_lo)\n")
        f.write("\t{\n")
        f.write("\t\tfloat *base = &dev_t3[frag_m * intra_stride];\n")
        f.write("\t\tif(div)\n")
        f.write("\t\t{\n")
        f.write("\t\t\treinterpret_cast<float2*>(&base[frag_n])[0] = reg0;\n")
        f.write("\t\t\treinterpret_cast<float2*>(&base[frag_n + 8])[0] = reg2;\n")
        f.write("\t\t}\n")
        f.write("\t\telse\n")
        f.write("\t\t{\n")
        f.write("\t\t\tbase[frag_n] = reg0.x;\n")
        f.write("\t\t\tbase[frag_n + 1] = reg0.y;\n")
        f.write("\t\t\tbase[frag_n + 8] = reg2.x;\n")
        f.write("\t\t\tbase[frag_n + 9] = reg2.y;\n")
        f.write("\t\t}\n")
        f.write("\t}\n\n")
        f.write("\tif(row_hi)\n")
        f.write("\t{\n")
        f.write("\t\tfloat *base = &dev_t3[(frag_m + 8) * intra_stride];\n")
        f.write("\t\tif(div)\n")
        f.write("\t\t{\n")
        f.write("\t\t\treinterpret_cast<float2*>(&base[frag_n])[0] = reg1;\n")
        f.write("\t\t\treinterpret_cast<float2*>(&base[frag_n + 8])[0] = reg3;\n")
        f.write("\t\t}\n")
        f.write("\t\telse\n")
        f.write("\t\t{\n")
        f.write("\t\t\tbase[frag_n] = reg1.x;\n")
        f.write("\t\t\tbase[frag_n + 1] = reg1.y;\n")
        f.write("\t\t\tbase[frag_n + 8] = reg3.x;\n")
        f.write("\t\t\tbase[frag_n + 9] = reg3.y;\n")
        f.write("\t\t}\n")
        f.write("\t}\n")
    #
    elif opt == 1 :
        f.write("\tfloat *base0 = &dev_t3[frag_m * intra_stride];\n")
        f.write("\tif(col_lo_2 && div)\n")
        f.write("\t{\n")
        f.write("\t\treinterpret_cast<float2*>(&base0[frag_n])[0] = reg0;\n")
        f.write("\t}\n")
        f.write("\telse if(col_lo_2)\n")
        f.write("\t{\n")
        f.write("\t\tbase0[frag_n] = reg0.x;\n")
        f.write("\t\tbase0[frag_n + 1] = reg0.y;\n")
        f.write("\t}\n")
        f.write("\telse if(col_lo_1)\n")
        f.write("\t{\n")
        f.write("\t\tbase0[frag_n] = reg0.x;\n")
        f.write("\t}\n\n")
        f.write("\tif(col_hi_2 && div)\n")
        f.write("\t{\n")
        f.write("\t\treinterpret_cast<float2*>(&base0[frag_n + 8])[0] = reg2;\n")
        f.write("\t}\n")
        f.write("\telse if(col_hi_2)\n")
        f.write("\t{\n")
        f.write("\t\tbase0[frag_n + 8] = reg2.x;\n")
        f.write("\t\tbase0[frag_n + 9] = reg2.y;\n")
        f.write("\t}\n")
        f.write("\telse if(col_hi_1)\n")
        f.write("\t{\n")
        f.write("\t\tbase0[frag_n + 8] = reg2.x;\n")
        f.write("\t}\n\n")

        f.write("\tfloat *base1 = &dev_t3[(frag_m + 8) * intra_stride];\n")
        f.write("\tif(col_lo_2 && div)\n")
        f.write("\t{\n")
        f.write("\t\treinterpret_cast<float2*>(&base1[frag_n])[0] = reg1;\n")
        f.write("\t}\n")
        f.write("\telse if(col_lo_2)\n")
        f.write("\t{\n")
        f.write("\t\tbase1[frag_n] = reg1.x;\n")
        f.write("\t\tbase1[frag_n + 1] = reg1.y;\n")
        f.write("\t}\n")
        f.write("\telse if(col_lo_1)\n")
        f.write("\t{\n")
        f.write("\t\tbase1[frag_n] = reg1.x;\n")
        f.write("\t}\n\n")
        f.write("\tif(col_hi_2 && div)\n")
        f.write("\t{\n")
        f.write("\t\treinterpret_cast<float2*>(&base1[frag_n + 8])[0] = reg3;\n")
        f.write("\t}\n")
        f.write("\telse if(col_hi_2)\n")
        f.write("\t{\n")
        f.write("\t\tbase1[frag_n + 8] = reg3.x;\n")
        f.write("\t\tbase1[frag_n + 9] = reg3.y;\n")
        f.write("\t}\n")
        f.write("\telse if(col_hi_1)\n")
        f.write("\t{\n")
        f.write("\t\tbase1[frag_n + 8] = reg3.x;\n")
        f.write("\t}\n\n")
    #
    else :
        f.write("\tif(row_lo)\n")
        f.write("\t{\n")
        f.write("\t\tfloat *base0 = &dev_t3[frag_m * intra_stride];\n")
        f.write("\t\tif(col_lo_2 && div)\n")
        f.write("\t\t{\n")
        f.write("\t\t\treinterpret_cast<float2*>(&base0[frag_n])[0] = reg0;\n")
        f.write("\t\t}\n")
        f.write("\t\telse if(col_lo_2)\n")
        f.write("\t\t{\n")
        f.write("\t\t\tbase0[frag_n] = reg0.x;\n")
        f.write("\t\t\tbase0[frag_n + 1] = reg0.y;\n")
        f.write("\t\t}\n")
        f.write("\t\telse if(col_lo_1)\n")
        f.write("\t\t{\n")
        f.write("\t\t\tbase0[frag_n] = reg0.x;\n")
        f.write("\t\t}\n\n")
        f.write("\t\tif(col_hi_2 && div)\n")
        f.write("\t\t{\n")
        f.write("\t\t\treinterpret_cast<float2*>(&base0[frag_n + 8])[0] = reg2;\n")
        f.write("\t\t}\n")
        f.write("\t\telse if(col_hi_2)\n")
        f.write("\t\t{\n")
        f.write("\t\t\tbase0[frag_n + 8] = reg2.x;\n")
        f.write("\t\t\tbase0[frag_n + 9] = reg2.y;\n")
        f.write("\t\t}\n")
        f.write("\t\telse if(col_hi_1)\n")
        f.write("\t\t{\n")
        f.write("\t\t\tbase0[frag_n + 8] = reg2.x;\n")
        f.write("\t\t}\n")
        f.write("\t}\n")

        f.write("\tif(row_hi)\n")
        f.write("\t{\n")
        f.write("\t\tfloat *base1 = &dev_t3[(frag_m + 8) * intra_stride];\n")
        f.write("\t\tif(col_lo_2 && div)\n")
        f.write("\t\t{\n")
        f.write("\t\t\treinterpret_cast<float2*>(&base1[frag_n])[0] = reg1;\n")
        f.write("\t\t}\n")
        f.write("\t\telse if(col_lo_2)\n")
        f.write("\t\t{\n")
        f.write("\t\t\tbase1[frag_n] = reg1.x;\n")
        f.write("\t\t\tbase1[frag_n + 1] = reg1.y;\n")
        f.write("\t\t}\n")
        f.write("\t\telse if(col_lo_1)\n")
        f.write("\t\t{\n")
        f.write("\t\t\tbase1[frag_n] = reg1.x;\n")
        f.write("\t\t}\n\n")
        f.write("\t\tif(col_hi_2 && div)\n")
        f.write("\t\t{\n")
        f.write("\t\t\treinterpret_cast<float2*>(&base1[frag_n + 8])[0] = reg3;\n")
        f.write("\t\t}\n")
        f.write("\t\telse if(col_hi_2)\n")
        f.write("\t\t{\n")
        f.write("\t\t\tbase1[frag_n + 8] = reg3.x;\n")
        f.write("\t\t\tbase1[frag_n + 9] = reg3.y;\n")
        f.write("\t\t}\n")
        f.write("\t\telse if(col_hi_1)\n")
        f.write("\t\t{\n")
        f.write("\t\t\tbase1[frag_n + 8] = reg3.x;\n")
        f.write("\t\t}\n")
        f.write("\t}\n")

    f.write("}\n\n")

#
def tc_code_kernel_dev_scatter_store(f, kernel_name, fvi_flag, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, split_input, data_type) :
    #
    if fvi_flag == 1 :
        a_split_flag = split_input[1]
        b_split_flag = split_input[0]
    elif fvi_flag == 2 :
        a_split_flag = split_input[0]
        b_split_flag = split_input[1]
    
    #
    if data_type == "DOUBLE" :
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
            tc_code_kernel_dev_scatter_store_head(f, kernel_name, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, opt, data_type)
            tc_code_kernel_dev_scatter_store_body(f, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, opt)
        #
        else :
            for i in range(3) :
                tc_code_kernel_dev_scatter_store_head(f, kernel_name, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, opt, data_type, i)
                tc_code_kernel_dev_scatter_store_body(f, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, opt, i)
    else :
        #
        for i in range(3) :
            tc_code_kernel_dev_scatter_store_head_fp32(f, kernel_name, ld_tile_order_a, ld_tile_order_b, i, data_type)
            tc_code_kernel_dev_scatter_store_body_fp32(f, ld_tile_order_a, ld_tile_order_b, i)