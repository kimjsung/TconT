import math
import tc_helper

#
def tc_code_kernel_dev_compute_head(f, kernel_name, input_a, input_b, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, split_input, fvi_flag, data_type, opt) :
    #
    f.write(f"__device__ void {kernel_name}(")

    #
    if data_type == "DOUBLE" :
        f.write(f"double *__restrict sm_{input_a}, double *__restrict sm_{input_b}, const int wmiter, const int wniter, const int wrow, const int wcol,\n")
        f.write(f"const int {input_a}_frag_cnt, const int {input_b}_frag_cnt, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> *t3_frag,\n")
    else :
        f.write(f"float *__restrict sm_{input_a}, float *__restrict sm_{input_b}, const int wmiter, const int wniter, const int wrow, const int wcol,\n")
        f.write(f"const int {input_a}_frag_cnt, const int {input_b}_frag_cnt, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> *t3_frag,\n")
    f.write(f"const int shm_{input_a}_offset, const int shm_{input_b}_offset")

    #
    if fvi_flag == 1 :
        a_split_flag = split_input[1]
        b_split_flag = split_input[0]
    elif fvi_flag == 2 :
        a_split_flag = split_input[0]
        b_split_flag = split_input[1]

    if data_type == "DOUBLE" :
        # #
        # if opt == 1 :
        #     f.write(")\n")
        # elif opt == 2 :
        #     f.write(",\n")
        #     f.write("int internal_upperbound)\n")
        # elif opt == 3 :
        #     f.write(",\n")
        #     f.write(f"int rng_{collapsed_a[0]}, int rng_{collapsed_b[0]})\n")
        # elif opt == 4 :
        #     f.write(",\n")
        #     f.write("int internal_upperbound,\n")
        #     f.write(f"int rng_{collapsed_a[0]}, int rng_{collapsed_b[0]})\n")
        # elif opt == 5 or opt == 6:
        #     #
        #     f.write(",\n")
            
        #     #
        #     if a_split_flag :
        #         tmp_index_a = collapsed_a[0]
        #         tmp_index_b = ld_tile_order_b[1]
        #     elif b_split_flag :
        #         tmp_index_a = ld_tile_order_a[1]
        #         tmp_index_b = collapsed_b[0]
        #     else :
        #         tmp_index_a = ld_tile_order_a[1]
        #         tmp_index_b = ld_tile_order_b[1]
            
        #     #
        #     if opt == 5 :
        #         f.write(f"int rng_{tmp_index_a}, int rng_{tmp_index_b})\n")
        #     else :
        #         f.write(f"int rng_{tmp_index_a}, int rng_{tmp_index_b},\n")
        #         f.write("int internal_upperbound)\n")
        # elif opt == 7 or opt == 8:
        #     #
        #     f.write(",\n")
            
        #     #
        #     if a_split_flag :
        #         tmp_index_a = collapsed_a[0]
        #         tmp_index_b_reg = ld_tile_order_b[0]
        #         tmp_index_b_frag = ld_tile_order_b[1]
        #     elif b_split_flag :
        #         tmp_index_a_reg = ld_tile_order_a[0]
        #         tmp_index_a_frag = ld_tile_order_a[1]
        #         tmp_index_b = collapsed_b[0]
        #     else :
        #         tmp_index_a_reg = ld_tile_order_a[0]
        #         tmp_index_a_frag = ld_tile_order_a[1]
        #         tmp_index_b_reg = ld_tile_order_b[0]
        #         tmp_index_b_frag = ld_tile_order_b[1]
            
        #     #
        #     if opt == 7 :
        #         #
        #         if a_split_flag :
        #             f.write(f"int rng_{tmp_index_a}, int rng_{tmp_index_b_frag},\n")
        #             f.write(f"int rng_{tmp_index_b_reg})\n")
        #         elif b_split_flag :
        #             f.write(f"int rng_{tmp_index_a_frag}, int rng_{tmp_index_b},\n")
        #             f.write(f"int rng_{tmp_index_a_reg})\n")
        #         else :
        #             f.write(f"int rng_{tmp_index_a_frag}, int rng_{tmp_index_b_frag},\n")
        #             f.write(f"int rng_{tmp_index_a_reg}, int rng_{tmp_index_b_reg})\n")
        #     elif opt == 8 :
        #         #
        #         if a_split_flag :
        #             f.write(f"int rng_{tmp_index_a}, int rng_{tmp_index_b_frag},\n")
        #             f.write("int internal_upperbound,\n")
        #             f.write(f"int rng_{tmp_index_b_reg})\n")
        #         elif b_split_flag :
        #             f.write(f"int rng_{tmp_index_a_frag}, int rng_{tmp_index_b},\n")
        #             f.write("int internal_upperbound,\n")
        #             f.write(f"int rng_{tmp_index_a_reg})\n")
        #         else :
        #             f.write(f"int rng_{tmp_index_a_frag}, int rng_{tmp_index_b_frag},\n")
        #             f.write("int internal_upperbound,\n")
        #             f.write(f"int rng_{tmp_index_a_reg}, int rng_{tmp_index_b_reg})\n")
        if opt % 2 != 0 :
            f.write(")\n")
        else :
            f.write(",\n")
            f.write("int internal_upperbound)\n")
    else :
        #
        if opt == 1 or opt == 5 :
            f.write(")\n")
        elif opt == 2 or opt == 6 :
            f.write(",\n")
            f.write("int internal_upperbound)\n")
        elif opt == 3 or opt == 7 :
            f.write(",\n")
            f.write(f"int rng_{collapsed_a[0]}, int rng_{collapsed_b[0]})\n")
        elif opt == 4 or opt == 8 :
            f.write(",\n")
            f.write("int internal_upperbound,\n")
            f.write(f"int rng_{collapsed_a[0]}, int rng_{collapsed_b[0]})\n")

#
def tc_code_kernel_dev_compute_body(f, l_splited_indices_size, input_a, input_b, ld_tile_order_a, ld_tile_order_b, SMEM_order_a, SMEM_order_b,
                                    warp_shape, double2_flag, reg_padd_y, reg_padd_x, fvi_flag, data_type, opt) :
    #
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
            per_row_internal = 16 // left_reg_size
            shift = (int)(math.log2(per_row_internal))
            mod = tc_helper.ceil(4, per_row_internal) - 1

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
            per_row_internal = 16 // right_reg_size
            shift = (int)(math.log2(per_row_internal))
            mod = tc_helper.ceil(4, per_row_internal) - 1

            f.write(f"\tconst int {input_b}_lane_offset = ((lane >> 2) * ld_stride_b);\n")
            f.write(f"\tconst int {input_b}_fragment_offset = ((lane & 3) << {(int)(math.log2(right_reg_size))}) + (((lane >> {shift}) & {mod}) << 1);\n")
        #
        elif SMEM_order_b[2] == ld_tile_order_b[1] :
            per_row_internal = 16 // right_frag_size
            row_cnt = tc_helper.ceil(4, per_row_internal)
            shift1 = (int)(math.log2(row_cnt))
            shift2 = (int)(math.log2(per_row_internal))

            f.write(f"\tconst int {input_b}_lane_offset = ((lane >> 2) & 3);\n")
            f.write(f"\tconst int {input_b}_fragment_offset = ((((lane & 3) << {shift1}) ^ ((lane >> {shift2}) & {row_cnt - 1})) ^ (lane >> 4));\n")
        #
        elif SMEM_order_b[2] == ld_tile_order_b[2] :
            per_row_frag = 16 // size_internal
            row_cnt = tc_helper.ceil(4, per_row_frag)
            per_row_thread = 16 // row_cnt
            shift1 = (int)(math.log2(row_cnt))
            shift2 = (int)(math.log2(per_row_thread))
            
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

    tab = 1
    if a_double2_flag :
        if SMEM_order_a[2] == ld_tile_order_a[0] :
            f.write("\t" * tab + f"const int xor_offset_{input_a} = (wrow ^ {input_a}_fragment_offset);\n")
            f.write("\t" * tab + f"const int tmp_{input_a} = shm_{input_a}_offset + {input_a}_lane_offset;\n")
        elif SMEM_order_a[2] == ld_tile_order_a[1] :
            if left_frag_size == 16 :
                f.write("\t" * tab + f"const int tmp_{input_a} = shm_{input_a}_offset + (wrow * ld_stride_a) + {input_a}_lane_offset;\n")
            else :
                f.write("\t" * tab + f"const int tmp_{input_a} = shm_{input_a}_offset + (wrow * ld_stride_a) + ({input_a}_fragment_offset << 2) + {input_a}_lane_offset;\n")
        elif SMEM_order_a[2] == ld_tile_order_a[2] :
            f.write("\t" * tab + f"const int tmp_{input_a} = shm_{input_a}_offset + (wrow * ld_stride_a) + {input_a}_lane_offset;\n")
    else :
        if SMEM_order_a[2] == ld_tile_order_a[0] :
            f.write("\t" * tab + f"const int tmp_{input_a} = shm_{input_a}_offset + (wrow * ld_stride_a) + {input_a}_fragment_offset;\n")
        elif SMEM_order_a[2] == ld_tile_order_a[1] :
            f.write("\t" * tab + f"const int tmp_{input_a} = shm_{input_a}_offset + (wrow * ld_stride_a) + {input_a}_fragment_offset;\n")
        else :
            f.write("\t" * tab + f"const int tmp_{input_a} = shm_{input_a}_offset + (wrow * ld_stride_a);\n")

    #
    if b_double2_flag :
        if SMEM_order_b[2] == ld_tile_order_b[0] :
            f.write("\t" * tab + f"const int xor_offset_{input_b} = (wcol ^ {input_b}_fragment_offset);\n")
            f.write("\t" * tab + f"const int tmp_{input_b} = shm_{input_b}_offset + {input_b}_lane_offset;\n")
        elif SMEM_order_b[2] == ld_tile_order_b[1] :
            if right_frag_size == 16 :
                f.write("\t" * tab + f"const int tmp_{input_b} = shm_{input_b}_offset + (wcol * ld_stride_b) + {input_b}_lane_offset;\n")
            else :
                f.write("\t" * tab + f"const int tmp_{input_b} = shm_{input_b}_offset + (wcol * ld_stride_b) + ({input_b}_fragment_offset << 2) + {input_b}_lane_offset;\n")
        else :
            f.write("\t" * tab + f"const int tmp_{input_b} = shm_{input_b}_offset + (wcol * ld_stride_b) + {input_b}_lane_offset;\n")
    else :
        if SMEM_order_b[2] == ld_tile_order_b[0] :        
            f.write("\t" * tab + f"const int tmp_{input_b} = shm_{input_b}_offset + (wcol * ld_stride_b) + {input_b}_fragment_offset;\n")
        elif SMEM_order_b[2] == ld_tile_order_b[1] :
            f.write("\t" * tab + f"const int tmp_{input_b} = shm_{input_b}_offset + (wcol * ld_stride_b) + {input_b}_fragment_offset;\n")
        else :
            f.write("\t" * tab + f"const int tmp_{input_b} = shm_{input_b}_offset + (wcol * ld_stride_b);\n")

    f.write("\n")

    #
    if opt % 2 == 0 :
        f.write("\tfor(int ll = 0; ll < TILE_UNIT - internal_upperbound; ll += 4)\n")
    else :
        f.write("\tfor(int ll = 0; ll < TILE_UNIT; ll += 4)\n")

    #
    tab = 1
    f.write("\t" * tab + "{\n"); tab += 1

    if a_double2_flag :
        #
        if SMEM_order_a[2] == ld_tile_order_a[0] :
            ll_offset = (int)(math.log2(left_reg_size))
            f.write("\t" * tab + f"const int ll_iter_{input_a} = ll << {ll_offset};\n")
        #
        elif SMEM_order_a[2] == ld_tile_order_a[1] :
            ll_offset = (int)(math.log2(left_frag_size))
            f.write("\t" * tab + f"const int ll_iter_{input_a} = ll << {ll_offset};\n")
        #
        else :
            f.write("\t" * tab + f"const int ll_iter_{input_a} = (({input_a}_fragment_offset ^ (ll >> 2)) << 2);\n")
    #
    else :
        #
        if SMEM_order_a[2] == ld_tile_order_a[0] :
            f.write("\t" * tab + f"const int ll_iter_{input_a} = (ll << 3);\n")
        #
        elif SMEM_order_a[2] == ld_tile_order_a[1] :
            #
            if left_frag_size == 16 :
                f.write("\t" * tab + f"const int ll_iter_{input_a} = ((ll << 3) + (ll >> 2));\n")
            #
            else :
                f.write("\t" * tab + f"const int ll_iter_{input_a} = ((ll << 3) + (ll >> 1));\n")
        #
        else :
            #
            if size_internal == 8 :
                f.write("\t" * tab + f"const int ll_iter_{input_a} = (({input_a}_fragment_offset ^ (ll << 1)) + (ll << 3));\n")
            else :
                f.write("\t" * tab + f"const int ll_iter_{input_a} = (({input_a}_fragment_offset ^ ll) + (ll << 3));\n")

    #
    if b_double2_flag :
        #
        if SMEM_order_b[2] == ld_tile_order_b[0] :
            ll_offset = (int)(math.log2(right_reg_size))
            f.write("\t" * tab + f"const int ll_iter_{input_b} = (ll << {ll_offset});\n")
        #
        elif SMEM_order_b[2] == ld_tile_order_b[1] :
            ll_offset = (int)(math.log2(right_frag_size))
            f.write("\t" * tab + f"const int ll_iter_{input_b} = (ll << {ll_offset});\n")
        #
        else :
            f.write("\t" * tab + f"const int ll_iter_{input_b} = (({input_b}_fragment_offset ^ (ll >> 2)) << 2);\n")
    #
    else :
        #
        if SMEM_order_b[2] == ld_tile_order_b[0] :
            f.write("\t" * tab + f"const int ll_iter_{input_b} = (ll << 3);\n")
        #
        elif SMEM_order_b[2] == ld_tile_order_b[1] :
            #
            if right_frag_size == 16 :
                f.write("\t" * tab + f"const int ll_iter_{input_b} = (ll << 3) + (ll >> 2);\n")
            #
            else :
                f.write("\t" * tab + f"const int ll_iter_{input_b} = (ll << 3) + (ll >> 1);\n")
        #
        else :
            f.write("\t" * tab + f"const int ll_iter_{input_b} = (ll << 3);\n")

    #
    if a_double2_flag :
        #
        if SMEM_order_a[2] == ld_tile_order_a[0] :
            f.write("\t" * tab + "#pragma unroll\n")
            f.write("\t" * tab + f"for(int iter_{input_a} = 0; iter_{input_a} < wmiter; iter_{input_a} += 2)\n")
            f.write("\t" * tab + "{\n"); tab += 1
            ll_offset = (int)(math.log2(left_reg_size))

            #
            if left_frag_size == 16 :
                f.write("\t" * tab + f"const int tmp_{input_a}1 = tmp_{input_a} + (xor_offset_{input_a} ^ iter_{input_a});\n")

                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_a} = 0; cnt_{input_a} < {input_a}_frag_cnt; cnt_{input_a}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1

                # f.write("\t" * tab + f"int {input_a}_offset = tmp_{input_a}1 + (cnt_{input_a} * (ld_stride_a << 3)) + (ll << {ll_offset});\n")
                f.write("\t" * tab + f"int {input_a}_offset = tmp_{input_a}1 + (cnt_{input_a} * (ld_stride_a << 3)) + ll_iter_{input_a};\n")
            #
            else :
                # f.write("\t" * tab + f"int {input_a}_offset = tmp_{input_a} + (xor_offset_{input_a} ^ iter_{input_a}) + (ll << {ll_offset});\n")
                f.write("\t" * tab + f"int {input_a}_offset = tmp_{input_a} + (xor_offset_{input_a} ^ iter_{input_a}) + ll_iter_{input_a};\n")

            f.write("\t" * tab + f"double2 reg_{input_a} = reinterpret_cast<double2*>(&sm_{input_a}[{input_a}_offset])[0];\n")
            f.write("\t" * tab + f"{input_a}_frag[0].x[0] = reg_{input_a}.x;\n")
            f.write("\t" * tab + f"{input_a}_frag[1].x[0] = reg_{input_a}.y;\n\n")
        #
        elif SMEM_order_a[2] == ld_tile_order_a[1] :
            f.write("\t" * tab + "#pragma unroll\n")
            f.write("\t" * tab + f"for(int iter_{input_a} = 0; iter_{input_a} < wmiter; iter_{input_a}++)\n")
            f.write("\t" * tab + "{\n"); tab += 1
            ll_offset = (int)(math.log2(left_frag_size))

            #
            if left_frag_size == 16 :
                # f.write("\t" * tab + f"const int tmp_{input_a}1 = tmp_{input_a} + (iter_{input_a} * ld_stride_a) + (ll << {ll_offset});\n")
                f.write("\t" * tab + f"const int tmp_{input_a}1 = tmp_{input_a} + (iter_{input_a} * ld_stride_a) + ll_iter_{input_a};\n")

                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_a} = 0; cnt_{input_a} < {input_a}_frag_cnt; cnt_{input_a}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1

                f.write("\t" * tab + f"const int {input_a}_offset = tmp_{input_a}1 + (({input_a}_fragment_offset ^ (cnt_{input_a} << 1)) << 2);\n")
            #
            else :
                # f.write("\t" * tab + f"int {input_a}_offset = tmp_{input_a} + (iter_{input_a} * ld_stride_a) + (ll << {ll_offset});\n")
                f.write("\t" * tab + f"const int {input_a}_offset = tmp_{input_a} + (iter_{input_a} * ld_stride_a) + ll_iter_{input_a};\n")

            f.write("\t" * tab + f"{input_a}_frag.x[0] = sm_{input_a}[{input_a}_offset];\n\n")
        #
        else :
            f.write("\t" * tab + "#pragma unroll\n")
            f.write("\t" * tab + f"for(int iter_{input_a} = 0; iter_{input_a} < wmiter; iter_{input_a}++)\n")
            f.write("\t" * tab + "{\n"); tab += 1

            #
            if left_frag_size == 16 :
                # f.write("\t" * tab + f"int tmp_{input_a}1 = tmp_{input_a} + (iter_{input_a} * ld_stride_a) + (({input_a}_fragment_offset ^ (ll >> 2)) << 2);\n")
                f.write("\t" * tab + f"const int tmp_{input_a}1 = tmp_{input_a} + (iter_{input_a} * ld_stride_a) + ll_iter_{input_a};\n")

                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_a} = 0; cnt_{input_a} < {input_a}_frag_cnt; cnt_{input_a}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1

                cnt_offset = (int)(math.log2(8 * size_internal))

                f.write("\t" * tab + f"const int {input_a}_offset = tmp_{input_a}1 + (cnt_{input_a} << {cnt_offset});\n")
            #
            else :
                # f.write("\t" * tab + f"int {input_a}_offset = tmp_{input_a} + (iter_{input_a} * ld_stride_a) + (({input_a}_fragment_offset ^ (ll >> 2)) << 2);\n")
                f.write("\t" * tab + f"const int {input_a}_offset = tmp_{input_a} + (iter_{input_a} * ld_stride_a) + ll_iter_{input_a};\n")

            f.write("\t" * tab + f"{input_a}_frag.x[0] = sm_{input_a}[{input_a}_offset];\n\n")
    #
    else :
        f.write("\t" * tab + "#pragma unroll\n")
        f.write("\t" * tab + f"for(int iter_{input_a} = 0; iter_{input_a} < wmiter; iter_{input_a}++)\n")
        f.write("\t" * tab + "{\n"); tab += 1

        #
        if SMEM_order_a[2] == ld_tile_order_a[0] :
            #
            if left_frag_size == 16 :
                # f.write("\t" * tab + f"int tmp{input_a}1 = tmp_{input_a} + (iter_{input_a} * ld_stride_a) + (ll << 3);\n")
                f.write("\t" * tab + f"const int tmp_{input_a}1 = tmp_{input_a} + (iter_{input_a} * ld_stride_a) + ll_iter_{input_a};\n")

                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_a} = 0; cnt_{input_a} < {input_a}_frag_cnt; cnt_{input_a}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1

                cnt_offset = (int)(math.log2(8 * size_internal))

                f.write("\t" * tab + f"const int {input_a}_offset = tmp_{input_a}1 + (cnt_{input_a} << {cnt_offset});\n")
            #
            else :
                # f.write("\t" * tab + f"int {input_a}_offset = tmp_{input_a} + (iter_{input_a} * ld_stride_a) + (ll << 3);\n")
                f.write("\t" * tab + f"const int {input_a}_offset = tmp_{input_a} + (iter_{input_a} * ld_stride_a) + ll_iter_{input_a};\n")

            f.write("\t" * tab + f"{input_a}_frag.x[0] = sm_{input_a}[{input_a}_offset];\n\n")
        #
        elif SMEM_order_a[2] == ld_tile_order_a[1] :
            #
            if left_frag_size == 16 :
                # f.write("\t" * tab + f"int tmp_{input_a}1 = tmp_{input_a} + (iter_{input_a} * ld_stride_a) + (ll << 3) + (ll >> 2);\n")
                f.write("\t" * tab + f"const int tmp_{input_a}1 = tmp_{input_a} + (iter_{input_a} * ld_stride_a) + ll_iter_{input_a};\n")
                    
                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_a} = 0; cnt_{input_a} < {input_a}_frag_cnt; cnt_{input_a}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1

                #
                if reg_padd_y[2] == 0 :
                    if reg_padd_y[1] == 0 :
                        f.write("\t" * tab + f"const int {input_a}_offset = tmp_{input_a}1 + (cnt_{input_a} * (TILE_{ld_tile_order_a[2].capitalize()} << 3));\n")
                    else :
                        f.write("\t" * tab + f"const int {input_a}_offset = tmp_{input_a}1 + (cnt_{input_a} * ((TILE_{ld_tile_order_a[2].capitalize()} << 3) + {reg_padd_y[1]}));\n")
                else :
                    if reg_padd_y[1] == 0 :
                        f.write("\t" * tab + f"const int {input_a}_offset = tmp_{input_a}1 + (cnt_{input_a} * ((TILE_{ld_tile_order_a[2].capitalize()} << 3) + {reg_padd_y[2]}));\n")
                    else :
                        tmp_cnt_per_padd_y = reg_padd_y[1] + reg_padd_y[2]
                        f.write("\t" * tab + f"const int {input_a}_offset = tmp_{input_a}1 + (cnt_{input_a} * ((TILE_{ld_tile_order_a[2].capitalize()} << 3) + {tmp_cnt_per_padd_y}));\n")
            #
            else :
                # f.write("\t" * tab + f"int {input_a}_offset = tmp_{input_a} + (iter_{input_a} * ld_stride_a) + (ll << 3) + (ll >> 1);\n")
                f.write("\t" * tab + f"const int {input_a}_offset = tmp_{input_a} + (iter_{input_a} * ld_stride_a) + ll_iter_{input_a};\n")
            
            f.write("\t" * tab + f"{input_a}_frag.x[0] = sm_{input_a}[{input_a}_offset];\n\n")
        #
        else :
            #
            if left_frag_size == 16 :
                # f.write("\t" * tab + f"int tmp{input_a}1 = tmp_{input_a} + (iter_{input_a} * ld_stride_a) + (ll << 3);\n")
                f.write("\t" * tab + f"const int tmp_{input_a}1 = tmp_{input_a} + (iter_{input_a} * ld_stride_a) + ll_iter_{input_a};\n")

                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_a} = 0; cnt_{input_a} < {input_a}_frag_cnt; cnt_{input_a}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1
                f.write("\t" * tab + f"const int {input_a}_offset = tmp_{input_a}1 + (cnt_{input_a} * (TILE_{ld_tile_order_a[2].capitalize()} << 3));\n")
            #
            else :
                # f.write("\t" * tab + f"int {input_a}_offset = tmp_{input_a} + (iter_{input_a} * ld_stride_a) + (ll << 3);\n")
                f.write("\t" * tab + f"const int {input_a}_offset = tmp_{input_a} + (iter_{input_a} * ld_stride_a) + ll_iter_{input_a};\n")

            #
            if size_internal == 8 :
                # f.write("\t" * tab + f"{input_a}_frag.x[0] = sm_{input_a}[{input_a}_offset + ({input_a}_fragment_offset ^ (ll << 1))];\n\n")
                f.write("\t" * tab + f"{input_a}_frag.x[0] = sm_{input_a}[{input_a}_offset];\n\n")
            else :
                # f.write("\t" * tab + f"{input_a}_frag.x[0] = sm_{input_a}[{input_a}_offset + ({input_a}_fragment_offset ^ ll)];\n\n")
                f.write("\t" * tab + f"{input_a}_frag.x[0] = sm_{input_a}[{input_a}_offset)];\n\n")

    #
    if b_double2_flag :
        #
        if SMEM_order_b[2] == ld_tile_order_b[0] :
            f.write("\t" * tab + "#pragma unroll\n")
            f.write("\t" * tab + f"for(int iter_{input_b} = 0; iter_{input_b} < wniter; iter_{input_b} += 2)\n")
            f.write("\t" * tab + "{\n"); tab += 1
            ll_offset = (int)(math.log2(right_reg_size))

            #
            if right_frag_size == 16 :
                # f.write("\t" * tab + f"int tmp_{input_b}1 = tmp_{input_b} + (xor_offset_{input_b} ^ iter_{input_b}) + (ll << {ll_offset});\n")
                f.write("\t" * tab + f"const int tmp_{input_b}1 = tmp_{input_b} + (xor_offset_{input_b} ^ iter_{input_b}) + ll_iter_{input_b};\n")
            
                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_b} = 0; cnt_{input_b} < {input_b}_frag_cnt; cnt_{input_b}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1

                f.write("\t" * tab + f"const int {input_b}_offset = tmp_{input_b}1 + (cnt_{input_b} * (ld_stride_b << 3));\n")
            #
            else :
                # f.write("\t" * tab + f"int {input_b}_offset = tmp_{input_b} + (xor_offset_{input_b} ^ iter_{input_b}) + (ll << {ll_offset});\n")
                f.write("\t" * tab + f"const int {input_b}_offset = tmp_{input_b} + (xor_offset_{input_b} ^ iter_{input_b}) + ll_iter_{input_b};\n")
        #
        elif SMEM_order_b[2] == ld_tile_order_b[1] :
            #
            f.write("\t" * tab + "#pragma unroll\n")
            f.write("\t" * tab + f"for(int iter_{input_b} = 0; iter_{input_b} < wniter; iter_{input_b}++)\n")
            f.write("\t" * tab + "{\n"); tab += 1
            ll_offset = (int)(math.log2(right_frag_size))

            #
            if right_frag_size == 16 :
                # f.write("\t" * tab + f"int tmp_{input_b}1 = tmp_{input_b} + (iter_{input_b} * ld_stride_b) + (ll << {ll_offset});\n")
                f.write("\t" * tab + f"const int tmp_{input_b}1 = tmp_{input_b} + (iter_{input_b} * ld_stride_b) + ll_iter_{input_b};\n")

                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_b} = 0; cnt_{input_b} < {input_b}_frag_cnt; cnt_{input_b}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1

                f.write("\t" * tab + f"const int {input_b}_offset = tmp_{input_b}1 + (({input_b}_fragment_offset ^ (cnt_{input_b} << 1)) << 2);\n")
            #
            else :
                # f.write("\t" * tab + f"int {input_b}_offset = tmp_{input_b} + (iter_{input_b} * ld_stride_b) + (ll << {ll_offset});\n")
                f.write("\t" * tab + f"const int {input_b}_offset = tmp_{input_b} + (iter_{input_b} * ld_stride_b) + ll_iter_{input_b};\n")
        #
        else :
            f.write("\t" * tab + "#pragma unroll\n")
            f.write("\t" * tab + f"for(int iter_{input_b} = 0; iter_{input_b} < wniter; iter_{input_b}++)\n")
            f.write("\t" * tab + "{\n"); tab += 1

            #
            if right_frag_size == 16 :
                # f.write("\t" * tab + f"int tmp_{input_b}1 = tmp_{input_b} + (iter_{input_b} * ld_stride_b) + (({input_b}_fragment_offset ^ (ll >> 2)) << 2);\n")
                f.write("\t" * tab + f"const int tmp_{input_b}1 = tmp_{input_b} + (iter_{input_b} * ld_stride_b) + ll_iter_{input_b};\n")
            
                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_b} = 0; cnt_{input_b} < {input_b}_frag_cnt; cnt_{input_b}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1

                cnt_offset = (int)(math.log2(8 * size_internal))

                f.write("\t" * tab + f"const int {input_b}_offset = tmp_{input_b}1 + (cnt_{input_b} << {cnt_offset});\n")
            #
            else :
                # f.write("\t" * tab + f"int {input_b}_offset = tmp_{input_b} + (iter_{input_b} * ld_stride_b) + (({input_b}_fragment_offset ^ (ll >> 2)) << 2);\n")
                f.write("\t" * tab + f"const int {input_b}_offset = tmp_{input_b} + (iter_{input_b} * ld_stride_b) + ll_iter_{input_b};\n")
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
                # f.write("\t" * tab + f"int tmp_{input_b}1 = tmp_{input_b} + (iter_{input_b} * ld_stride_b)+ (ll << 3);\n")
                f.write("\t" * tab + f"const int tmp_{input_b}1 = tmp_{input_b} + (iter_{input_b} * ld_stride_b)+ ll_iter_{input_b};\n")

                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_b} = 0; cnt_{input_b} < {input_b}_frag_cnt; cnt_{input_b}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1

                cnt_offset = (int)(math.log2(8 * size_internal))

                f.write("\t" * tab + f"const int {input_b}_offset = tmp_{input_b}1 + (cnt_{input_b} << {cnt_offset});\n")
            #
            else :
                # f.write("\t" * tab + f"int {input_b}_offset = tmp_{input_b} + (iter_{input_b} * ld_stride_b) + (ll << 3);\n")
                f.write("\t" * tab + f"const int {input_b}_offset = tmp_{input_b} + (iter_{input_b} * ld_stride_b) + ll_iter_{input_b};\n")
        #
        elif SMEM_order_b[2] == ld_tile_order_b[1] :
            #
            if right_frag_size == 16 :
                # f.write("\t" * tab + f"int tmp_{input_b}1 = tmp_{input_b} + (iter_{input_b} * ld_stride_b) + (ll << 3) + (ll >> 2);\n")
                f.write("\t" * tab + f"const int tmp_{input_b}1 = tmp_{input_b} + (iter_{input_b} * ld_stride_b) + ll_iter_{input_b};\n")

                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_b} = 0; cnt_{input_b} < {input_b}_frag_cnt; cnt_{input_b}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1

                #
                if reg_padd_x[2] == 0 :
                    if reg_padd_x[1] == 0 :
                        f.write("\t" * tab + f"const int {input_b}_offset = tmp_{input_b}1 + (cnt_{input_b} * (TILE_{ld_tile_order_a[2].capitalize()} << 3));\n")
                    else :
                        f.write("\t" * tab + f"const int {input_b}_offset = tmp_{input_b}1 + (cnt_{input_b} * ((TILE_{ld_tile_order_a[2].capitalize()} << 3) + {reg_padd_x[1]}));\n")
                #
                else :
                    if reg_padd_x[1] == 0 :
                        f.write("\t" * tab + f"const int {input_b}_offset = tmp_{input_b}1 + (cnt_{input_b} * ((TILE_{ld_tile_order_a[2].capitalize()} << 3) + {reg_padd_x[2]}));\n")
                    else :
                        tmp_cnt_per_padd_x = reg_padd_x[1] + reg_padd_x[2]
                        f.write("\t" * tab + f"const int {input_b}_offset = tmp_{input_b}1 + (cnt_{input_b} * ((TILE_{ld_tile_order_a[2].capitalize()} << 3) + {tmp_cnt_per_padd_x}));\n")
            #
            else :
                # f.write("\t" * tab + f"int {input_b}_offset = tmp_{input_b} + (iter_{input_b} * ld_stride_b + (ll << 3) + (ll >> 1);\n")
                f.write("\t" * tab + f"const int {input_b}_offset = tmp_{input_b} + (iter_{input_b} * ld_stride_b) + ll_iter_{input_b};\n")
        #
        else :
            #
            if right_frag_size == 16 :
                # f.write("\t" * tab + f"int tmp_{input_b}1 = tmp_{input_b} + (iter_{input_b} * ld_stride_b) + (ll << 3);\n")
                f.write("\t" * tab + f"const int tmp_{input_b}1 = tmp_{input_b} + (iter_{input_b} * ld_stride_b) + ll_iter_{input_b};\n")

                f.write("\t" * tab + "#pragma unroll\n")
                f.write("\t" * tab + f"for(int cnt_{input_b} = 0; cnt_{input_b} < {input_b}_frag_cnt; cnt_{input_b}++)\n")
                f.write("\t" * tab + "{\n"); tab += 1
                
                f.write("\t" * tab + f"const int {input_b}_offset = tmp_{input_b}1 + (cnt_{input_b} * (TILE_{ld_tile_order_a[2].capitalize()} << 3));\n")
            #
            else :
                # f.write("\t" * tab + f"int {input_b}_offset = tmp_{input_b} + (iter_{input_b} * ld_stride_b) + (ll << 3);\n")
                f.write("\t" * tab + f"const int {input_b}_offset = tmp_{input_b} + (iter_{input_b} * ld_stride_b) + ll_iter_{input_b};\n")

    #
    if b_double2_flag and (SMEM_order_b[2] == ld_tile_order_b[0]) :
        #
        if a_double2_flag and (SMEM_order_a[2] == ld_tile_order_a[0]) :
            #
            if left_frag_size == 16 and right_frag_size == 16 :
                f.write("\t" * tab + f"const int out_idx0 = (((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = (((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx2 = ((((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx3 = ((((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            elif left_frag_size == 16 and right_frag_size == 8 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx2 = (((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx3 = (((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a};\n")
            #
            elif left_frag_size == 8 and right_frag_size == 16 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx2 = (((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx3 = (((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            else :
                f.write("\t" * tab + f"const int out_idx0 = (iter_{input_a} * wniter) + iter_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = (iter_{input_a} * wniter) + iter_{input_b} + 1;\n")
                f.write("\t" * tab + f"const int out_idx2 = ((iter_{input_a} + 1) * wniter) + iter_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx3 = ((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1;\n")
        else :
            #
            if left_frag_size == 16 and right_frag_size == 16 :
                f.write("\t" * tab + f"const int out_idx0 = (((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = (((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            elif left_frag_size == 16 and right_frag_size == 8 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a};\n")
            #
            elif left_frag_size == 8 and right_frag_size == 16 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            else :
                f.write("\t" * tab + f"const int out_idx0 = (iter_{input_a} * wniter) + iter_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = (iter_{input_a} * wniter) + iter_{input_b} + 1;\n")
    #
    else :
        #
        if a_double2_flag and (SMEM_order_a[2] == ld_tile_order_a[0]) :
            #
            if left_frag_size == 16 and right_frag_size == 16 :
                f.write("\t" * tab + f"const int out_idx0 = (((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            elif left_frag_size == 16 and right_frag_size == 8 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx1 = (((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
            #
            elif left_frag_size == 8 and right_frag_size == 16 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = (((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            else :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a}) * wniter) + iter_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} + 1) * wniter) + iter_{input_b};\n")
        #
        else :
            #
            if left_frag_size == 16 and right_frag_size == 16 :
                f.write("\t" * tab + f"const int out_idx = (((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            elif left_frag_size == 16 and right_frag_size == 8 :
                f.write("\t" * tab + f"const int out_idx = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
            #
            elif left_frag_size == 8 and right_frag_size == 16 :
                f.write("\t" * tab + f"const int out_idx = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            else :
                f.write("\t" * tab + f"const int out_idx = (iter_{input_a} * wniter) + iter_{input_b};\n")
    
    #
    if b_double2_flag :
        #
        if a_double2_flag and (SMEM_order_a[2] == ld_tile_order_a[0]) :
            #
            if SMEM_order_b[2] == ld_tile_order_b[0] :
                f.write("\t" * tab + f"double2 reg_{input_b} = reinterpret_cast<double2*>(&sm_{input_b}[{input_b}_offset])[0];\n")
                f.write("\t" * tab + f"{input_b}_frag[0].x[0] = reg_{input_b}.x;\n")
                f.write("\t" * tab + f"{input_b}_frag[1].x[0] = reg_{input_b}.y;\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx0], {input_a}_frag[0], {input_b}_frag[0], t3_frag[out_idx0]);\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx1], {input_a}_frag[0], {input_b}_frag[1], t3_frag[out_idx1]);\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx2], {input_a}_frag[1], {input_b}_frag[0], t3_frag[out_idx2]);\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx3], {input_a}_frag[1], {input_b}_frag[1], t3_frag[out_idx3]);\n")
            #
            elif SMEM_order_b[2] == ld_tile_order_b[1] :
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
                f.write("\t" * tab + f"double2 reg_{input_b} = reinterpret_cast<double2*>(&sm_{input_b}[{input_b}_offset])[0];\n")
                f.write("\t" * tab + f"{input_b}_frag[0].x[0] = reg_{input_b}.x;\n")
                f.write("\t" * tab + f"{input_b}_frag[1].x[0] = reg_{input_b}.y;\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx0], {input_a}_frag, {input_b}_frag[0], t3_frag[out_idx0]);\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx1], {input_a}_frag, {input_b}_frag[1], t3_frag[out_idx1]);\n")
            #
            elif SMEM_order_b[2] == ld_tile_order_b[1] :
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
                f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset];\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx0], {input_a}_frag[0], {input_b}_frag, t3_frag[out_idx0]);\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx1], {input_a}_frag[1], {input_b}_frag, t3_frag[out_idx1]);\n")
            #
            elif SMEM_order_b[2] == ld_tile_order_b[1] :
                f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset];\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx0], {input_a}_frag[0], {input_b}_frag, t3_frag[out_idx0]);\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx1], {input_a}_frag[1], {input_b}_frag, t3_frag[out_idx1]);\n")
            #
            else :
                #
                if size_internal == 8 :
                    f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset + ({input_b}_fragment_offset ^ (ll << 1))];\n")
                    f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx0], {input_a}_frag[0], {input_b}_frag, t3_frag[out_idx0]);\n")
                    f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx1], {input_a}_frag[1], {input_b}_frag, t3_frag[out_idx1]);\n")
                #
                else :
                    f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset + ({input_b}_fragment_offset ^ ll)];\n")
                    f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx0], {input_a}_frag[0], {input_b}_frag, t3_frag[out_idx0]);\n")
                    f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx1], {input_a}_frag[1], {input_b}_frag, t3_frag[out_idx1]);\n")
        #
        else :
            #
            if SMEM_order_b[2] == ld_tile_order_b[0] :
                f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset];\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx], {input_a}_frag, {input_b}_frag, t3_frag[out_idx]);\n")
            #
            elif SMEM_order_b[2] == ld_tile_order_b[1] :
                f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset];\n")
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx], {input_a}_frag, {input_b}_frag, t3_frag[out_idx]);\n")
            #
            else :
                #
                if size_internal == 8 :
                    f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset + ({input_b}_fragment_offset ^ (ll << 1))];\n")
                    f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx], {input_a}_frag, {input_b}_frag, t3_frag[out_idx]);\n")
                #
                else :
                    f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset + ({input_b}_fragment_offset ^ ll)];\n")
                    f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx], {input_a}_frag, {input_b}_frag, t3_frag[out_idx]);\n")
    
    #
    for i in range(tab) :
        tab -= 1
        f.write("\t" * tab + "}\n")

    f.write("\n")

#
def tc_code_kernel_dev_compute_body_fp32(f, l_splited_indices_size, input_a, input_b, ld_tile_order_a, ld_tile_order_b, SMEM_order_a, SMEM_order_b,
                                    split_input, warp_shape, double2_flag, reg_padd_y, reg_padd_x, fvi_flag, data_type, opt) :
    #
    a_double2_flag = double2_flag[1]
    b_double2_flag = double2_flag[0]
    
    #
    if fvi_flag == 1 :
        a_split_flag = split_input[1]
        b_split_flag = split_input[0]
    elif fvi_flag == 2 :
        a_split_flag = split_input[0]
        b_split_flag = split_input[1]

    #
    left_frag_size = tc_helper.tc_helper_find_value(l_splited_indices_size, ld_tile_order_a[1])
    right_frag_size = tc_helper.tc_helper_find_value(l_splited_indices_size, ld_tile_order_b[1])
    size_internal = tc_helper.tc_helper_find_value(l_splited_indices_size, ld_tile_order_a[2])
    left_reg_size = tc_helper.tc_helper_find_value(l_splited_indices_size, ld_tile_order_a[0])
    right_reg_size = tc_helper.tc_helper_find_value(l_splited_indices_size, ld_tile_order_b[0])

    #
    f.write("{\n")

    #
    if reg_padd_y[0] == 0 :
        f.write(f"\tconst int ld_stride_a = TILE_{SMEM_order_a[1].capitalize()} * TILE_{SMEM_order_a[2].capitalize()}" + ";\n")
    else :
        f.write(f"\tconst int ld_stride_a = TILE_{SMEM_order_a[1].capitalize()} * TILE_{SMEM_order_a[2].capitalize()} + {reg_padd_y[0]}" + ";\n")
    
    #
    if reg_padd_x[0] == 0 :
        f.write(f"\tconst int ld_stride_b = TILE_{SMEM_order_b[1].capitalize()} * TILE_{SMEM_order_b[2].capitalize()}" + ";\n\n")
    else :
        f.write(f"\tconst int ld_stride_b = TILE_{SMEM_order_b[1].capitalize()} * TILE_{SMEM_order_b[2].capitalize()} + {reg_padd_x[0]}" + ";\n\n")
    
    #
    if SMEM_order_a[2] == ld_tile_order_a[0] :
        if a_double2_flag == 2 :
            f.write(f"\tnvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> {input_a}_frag[4];\n")
        elif a_double2_flag == 1 :
            f.write(f"\tnvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> {input_a}_frag[2];\n")
        else :
            f.write(f"\tnvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> {input_a}_frag;\n")
    else :
        f.write(f"\tnvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> {input_a}_frag;\n")

    #
    if SMEM_order_b[2] == ld_tile_order_b[0] :
        if b_double2_flag == 2 :
            f.write(f"\tnvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> {input_b}_frag[4];\n\n")
        elif b_double2_flag == 1 :
            f.write(f"\tnvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> {input_b}_frag[2];\n\n")
        else :
            f.write(f"\tnvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> {input_b}_frag;\n\n")
    else :
        f.write(f"\tnvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> {input_b}_frag;\n\n")
    
    #
    f.write(f"\tconst int lane = threadIdx.x & 31;\n")

    #
    if SMEM_order_a[2] == ld_tile_order_a[0] :      # SMEM Order = [Frag_Y, Internal, Reg_Y], FVI = REG
        per_row_internal = 32 // left_reg_size
        shift = (int)(math.log2(per_row_internal))
        if a_double2_flag == 0 :
            vector_size = 0
        else :
            vector_size = (int)(math.log2(2 * a_double2_flag))

        f.write(f"\tconst int {input_a}_lane_offset = ((lane >> 2) * ld_stride_a);\n")
        f.write(f"\tconst int {input_a}_fragment_offset = ((lane & 3) << {(int)(math.log2(left_reg_size))}) ^ (((lane & 3) >> {shift}) << {vector_size});\n")
    #
    elif SMEM_order_a[2] == ld_tile_order_a[1] :    # SMEM Order = [Reg_Y, Internal, Frag_Y], FVI = FRAG
        per_row_internal = 32 // left_frag_size
        row_cnt = tc_helper.ceil(4, per_row_internal)
        shift1 = (int)(math.log2(left_frag_size))
        shift2 = (int)(math.log2(per_row_internal))

        f.write(f"\tconst int {input_a}_lane_offset = (lane >> 2);\n")
        f.write(f"\tconst int {input_a}_fragment_offset = ((lane & 3) << {shift1}) ^ (((lane >> {shift2}) & {row_cnt - 1}) << 3);\n")
    #
    elif SMEM_order_a[2] == ld_tile_order_a[2] :    # SMEM Order = [Reg_Y, Frag_Y, Internal], FVI = Internal
        per_row_frag = 32 // size_internal
        per_row_thread = 4 * per_row_frag
        shift1 = (int)(math.log2(size_internal))
        shift2 = (int)(math.log2(per_row_thread))
        row_cnt = 32 // per_row_thread
        
        f.write(f"\tconst int {input_a}_lane_offset = (lane & 3);\n")
        f.write(f"\tconst int {input_a}_fragment_offset = ((lane >> 2) << {shift1}) ^ (((lane >> {shift2}) & {row_cnt - 1}) << 2);\n")

    #
    if SMEM_order_b[2] == ld_tile_order_b[0] :
        per_row_internal = 32 // right_reg_size
        shift = (int)(math.log2(per_row_internal))
        if b_double2_flag == 0 :
            vector_size = 0
        else :
            vector_size = (int)(math.log2(2 * b_double2_flag))

        f.write(f"\tconst int {input_b}_lane_offset = ((lane >> 2) * ld_stride_b);\n")
        f.write(f"\tconst int {input_b}_fragment_offset = ((lane & 3) << {(int)(math.log2(right_reg_size))}) ^ (((lane & 3) >> {shift}) << {vector_size});\n")
    #
    elif SMEM_order_b[2] == ld_tile_order_b[1] :
        per_row_internal = 32 // right_frag_size
        row_cnt = tc_helper.ceil(4, per_row_internal)
        shift1 = (int)(math.log2(right_frag_size))
        shift2 = (int)(math.log2(per_row_internal))

        f.write(f"\tconst int {input_b}_lane_offset = (lane >> 2);\n")
        f.write(f"\tconst int {input_b}_fragment_offset = ((lane & 3) << {shift1}) ^ (((lane >> {shift2}) & {row_cnt - 1}) << 3);\n")
    #
    elif SMEM_order_b[2] == ld_tile_order_b[2] :
        per_row_frag = 32 // size_internal
        per_row_thread = 4 * per_row_frag
        shift1 = (int)(math.log2(size_internal))
        shift2 = (int)(math.log2(per_row_thread))
        row_cnt = 32 // per_row_thread
        
        #
        f.write(f"\tconst int {input_b}_lane_offset = (lane & 3);\n")
        f.write(f"\tconst int {input_b}_fragment_offset = ((lane >> 2) << {shift1}) ^ (((lane >> {shift2}) & {row_cnt - 1}) << 2);\n")
        
    f.write("\n")

    #
    if opt % 2 == 0 :
        f.write("\tconst int k_lane = lane & 3;\n")
    
    f.write("\n")

    #
    f.write("\tfor(int ll = 0; ll < TILE_UNIT; ll += 8)\n")

    #
    tab = 1
    f.write("\t" * tab + "{\n"); tab += 1

    #
    if opt % 2 == 0 :
        f.write("\t" * tab + f"const bool k_zero_lo = ((ll + k_lane) >= internal_upperbound);\n")
        f.write("\t" * tab + "const bool k_zero_hi = ((ll + k_lane + 4) >= internal_upperbound);\n")

    #
    if SMEM_order_a[2] == ld_tile_order_a[0] :              # FVI = REG
        #
        if a_double2_flag :
            step = 2 * a_double2_flag
        #
        else :
            step = 1

        ll_offset = (int)(math.log2(left_reg_size))
        frag_offset = 4 * left_reg_size

        f.write("\t" * tab + "#pragma unroll\n")
        f.write("\t" * tab + f"for(int iter_{input_a} = 0; iter_{input_a} < wmiter; iter_{input_a} += {step})\n")
        f.write("\t" * tab + "{\n"); tab += 1
        
        if opt == 3 or opt == 4 or opt == 7 or opt == 8 :
            if a_split_flag :
                f.write("\t" * tab + f"if(((wrow + iter_{input_a}) * TILE_{ld_tile_order_a[1].capitalize()}) >= rng_{ld_tile_order_a[1][0]})\n")
            else :
                f.write("\t" * tab + f"if((wrow + iter_{input_a}) >= rng_{ld_tile_order_a[0]})\n")
            f.write("\t" * (tab + 1) + "continue;\n\n")

        #
        if left_frag_size == 32 :
            f.write("\t" * tab + "#pragma unroll\n")
            f.write("\t" * tab + f"for(int cnt_{input_a} = 0; cnt_{input_a} < {input_a}_frag_cnt; cnt_{input_a}++)\n")
            f.write("\t" * tab + "{\n"); tab += 1

            f.write("\t" * tab + f"int {input_a}_offset0 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow ^ {input_a}_fragment_offset) ^ iter_{input_a}) + (cnt_{input_a} * (ld_stride_a << 4)) + (ll << {ll_offset});\n")
            f.write("\t" * tab + f"int {input_a}_offset1 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow ^ {input_a}_fragment_offset) ^ iter_{input_a}) + (((cnt_{input_a} << 1) + 1) * (ld_stride_a << 3)) + (ll << {ll_offset});\n")
            f.write("\t" * tab + f"int {input_a}_offset2 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow ^ {input_a}_fragment_offset) ^ iter_{input_a}) + (cnt_{input_a} * (ld_stride_a << 4)) + (ll << {ll_offset}) + {frag_offset};\n")
            f.write("\t" * tab + f"int {input_a}_offset3 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow ^ {input_a}_fragment_offset) ^ iter_{input_a}) + (((cnt_{input_a} << 1) + 1) * (ld_stride_a << 3)) + (ll << {ll_offset}) + {frag_offset};\n\n")
        #
        else :
            f.write("\t" * tab + f"int {input_a}_offset0 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow ^ {input_a}_fragment_offset) ^ iter_{input_a}) + (ll << {ll_offset});\n")
            f.write("\t" * tab + f"int {input_a}_offset1 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow ^ {input_a}_fragment_offset) ^ iter_{input_a}) + (ld_stride_a << 3) + (ll << {ll_offset});\n")
            f.write("\t" * tab + f"int {input_a}_offset2 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow ^ {input_a}_fragment_offset) ^ iter_{input_a}) + (ll << {ll_offset}) + {frag_offset};\n")
            f.write("\t" * tab + f"int {input_a}_offset3 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow ^ {input_a}_fragment_offset) ^ iter_{input_a}) + (ld_stride_a << 3) + (ll << {ll_offset}) + {frag_offset};\n\n")
        
        #
        if a_double2_flag == 2 :
            f.write("\t" * tab + f"float4 tmp0 = reinterpret_cast<float4*>(&sm_{input_a}[{input_a}_offset0])[0];\n")
            f.write("\t" * tab + f"float4 tmp1 = reinterpret_cast<float4*>(&sm_{input_a}[{input_a}_offset1])[0];\n")
            f.write("\t" * tab + f"float4 tmp2 = reinterpret_cast<float4*>(&sm_{input_a}[{input_a}_offset2])[0];\n")
            f.write("\t" * tab + f"float4 tmp3 = reinterpret_cast<float4*>(&sm_{input_a}[{input_a}_offset3])[0];\n\n")

            if opt % 2 == 0 :
                f.write("\t" * tab + f"{input_a}_frag[0].x[0] = k_zero_lo ? 0.0f : tmp0.x;\n")
                f.write("\t" * tab + f"{input_a}_frag[0].x[1] = k_zero_lo ? 0.0f : tmp1.x;\n")
                f.write("\t" * tab + f"{input_a}_frag[0].x[2] = k_zero_hi ? 0.0f : tmp2.x;\n")
                f.write("\t" * tab + f"{input_a}_frag[0].x[3] = k_zero_hi ? 0.0f : tmp3.x;\n\n")

                f.write("\t" * tab + f"{input_a}_frag[1].x[0] = k_zero_lo ? 0.0f : tmp0.y;\n")
                f.write("\t" * tab + f"{input_a}_frag[1].x[1] = k_zero_lo ? 0.0f : tmp1.y;\n")
                f.write("\t" * tab + f"{input_a}_frag[1].x[2] = k_zero_hi ? 0.0f : tmp2.y;\n")
                f.write("\t" * tab + f"{input_a}_frag[1].x[3] = k_zero_hi ? 0.0f : tmp3.y;\n\n")

                f.write("\t" * tab + f"{input_a}_frag[2].x[0] = k_zero_lo ? 0.0f : tmp0.z;\n")
                f.write("\t" * tab + f"{input_a}_frag[2].x[1] = k_zero_lo ? 0.0f : tmp1.z;\n")
                f.write("\t" * tab + f"{input_a}_frag[2].x[2] = k_zero_hi ? 0.0f : tmp2.z;\n")
                f.write("\t" * tab + f"{input_a}_frag[2].x[3] = k_zero_hi ? 0.0f : tmp3.z;\n\n")

                f.write("\t" * tab + f"{input_a}_frag[3].x[0] = k_zero_lo ? 0.0f : tmp0.w;\n")
                f.write("\t" * tab + f"{input_a}_frag[3].x[1] = k_zero_lo ? 0.0f : tmp1.w;\n")
                f.write("\t" * tab + f"{input_a}_frag[3].x[2] = k_zero_hi ? 0.0f : tmp2.w;\n")
                f.write("\t" * tab + f"{input_a}_frag[3].x[3] = k_zero_hi ? 0.0f : tmp3.w;\n\n")

            else :
                f.write("\t" * tab + f"{input_a}_frag[0].x[0] = tmp0.x;\n")
                f.write("\t" * tab + f"{input_a}_frag[0].x[1] = tmp1.x;\n")
                f.write("\t" * tab + f"{input_a}_frag[0].x[2] = tmp2.x;\n")
                f.write("\t" * tab + f"{input_a}_frag[0].x[3] = tmp3.x;\n\n")

                f.write("\t" * tab + f"{input_a}_frag[1].x[0] = tmp0.y;\n")
                f.write("\t" * tab + f"{input_a}_frag[1].x[1] = tmp1.y;\n")
                f.write("\t" * tab + f"{input_a}_frag[1].x[2] = tmp2.y;\n")
                f.write("\t" * tab + f"{input_a}_frag[1].x[3] = tmp3.y;\n\n")

                f.write("\t" * tab + f"{input_a}_frag[2].x[0] = tmp0.z;\n")
                f.write("\t" * tab + f"{input_a}_frag[2].x[1] = tmp1.z;\n")
                f.write("\t" * tab + f"{input_a}_frag[2].x[2] = tmp2.z;\n")
                f.write("\t" * tab + f"{input_a}_frag[2].x[3] = tmp3.z;\n\n")

                f.write("\t" * tab + f"{input_a}_frag[3].x[0] = tmp0.w;\n")
                f.write("\t" * tab + f"{input_a}_frag[3].x[1] = tmp1.w;\n")
                f.write("\t" * tab + f"{input_a}_frag[3].x[2] = tmp2.w;\n")
                f.write("\t" * tab + f"{input_a}_frag[3].x[3] = tmp3.w;\n\n")
        #
        elif a_double2_flag == 1 :
            f.write("\t" * tab + f"float2 tmp0 = reinterpret_cast<float2*>(&sm_{input_a}[{input_a}_offset0])[0];\n")
            f.write("\t" * tab + f"float2 tmp1 = reinterpret_cast<float2*>(&sm_{input_a}[{input_a}_offset1])[0];\n")
            f.write("\t" * tab + f"float2 tmp2 = reinterpret_cast<float2*>(&sm_{input_a}[{input_a}_offset2])[0];\n")
            f.write("\t" * tab + f"float2 tmp3 = reinterpret_cast<float2*>(&sm_{input_a}[{input_a}_offset3])[0];\n\n")

            if opt % 2 == 0 :
                f.write("\t" * tab + f"{input_a}_frag[0].x[0] = k_zero_lo ? 0.0f : tmp0.x;\n")
                f.write("\t" * tab + f"{input_a}_frag[0].x[1] = k_zero_lo ? 0.0f : tmp1.x;\n")
                f.write("\t" * tab + f"{input_a}_frag[0].x[2] = k_zero_hi ? 0.0f : tmp2.x;\n")
                f.write("\t" * tab + f"{input_a}_frag[0].x[3] = k_zero_hi ? 0.0f : tmp3.x;\n\n")

                f.write("\t" * tab + f"{input_a}_frag[1].x[0] = k_zero_lo ? 0.0f : tmp0.y;\n")
                f.write("\t" * tab + f"{input_a}_frag[1].x[1] = k_zero_lo ? 0.0f : tmp1.y;\n")
                f.write("\t" * tab + f"{input_a}_frag[1].x[2] = k_zero_hi ? 0.0f : tmp2.y;\n")
                f.write("\t" * tab + f"{input_a}_frag[1].x[3] = k_zero_hi ? 0.0f : tmp3.y;\n\n")
            else :
                f.write("\t" * tab + f"{input_a}_frag[0].x[0] = tmp0.x;\n")
                f.write("\t" * tab + f"{input_a}_frag[0].x[1] = tmp1.x;\n")
                f.write("\t" * tab + f"{input_a}_frag[0].x[2] = tmp2.x;\n")
                f.write("\t" * tab + f"{input_a}_frag[0].x[3] = tmp3.x;\n\n")

                f.write("\t" * tab + f"{input_a}_frag[1].x[0] = tmp0.y;\n")
                f.write("\t" * tab + f"{input_a}_frag[1].x[1] = tmp1.y;\n")
                f.write("\t" * tab + f"{input_a}_frag[1].x[2] = tmp2.y;\n")
                f.write("\t" * tab + f"{input_a}_frag[1].x[3] = tmp3.y;\n\n")
        #
        else :
            #
            if opt % 2 == 0 :
                f.write("\t" * tab + f"{input_a}_frag.x[0] = k_zero_lo ? 0.0f : sm_{input_a}[{input_a}_offset0];\n")
                f.write("\t" * tab + f"{input_a}_frag.x[1] = k_zero_lo ? 0.0f : sm_{input_a}[{input_a}_offset1];\n")
                f.write("\t" * tab + f"{input_a}_frag.x[2] = k_zero_hi ? 0.0f : sm_{input_a}[{input_a}_offset2];\n")
                f.write("\t" * tab + f"{input_a}_frag.x[3] = k_zero_hi ? 0.0f : sm_{input_a}[{input_a}_offset3];\n\n")
            #
            else :
                f.write("\t" * tab + f"{input_a}_frag.x[0] = sm_{input_a}[{input_a}_offset0];\n")
                f.write("\t" * tab + f"{input_a}_frag.x[1] = sm_{input_a}[{input_a}_offset1];\n")
                f.write("\t" * tab + f"{input_a}_frag.x[2] = sm_{input_a}[{input_a}_offset2];\n")
                f.write("\t" * tab + f"{input_a}_frag.x[3] = sm_{input_a}[{input_a}_offset3];\n\n")
    #
    elif SMEM_order_a[2] == ld_tile_order_a[1] :            # FVI = FRAG
        #
        ll_offset = (int)(math.log2(left_frag_size))
        frag_offset = 4 * left_frag_size

        #
        f.write("\t" * tab + "#pragma unroll\n")
        f.write("\t" * tab + f"for(int iter_{input_a} = 0; iter_{input_a} < wmiter; iter_{input_a}++)\n")
        f.write("\t" * tab + "{\n"); tab += 1

        #
        if opt == 3 or opt == 4 or opt == 7 or opt == 8 :
            if a_split_flag :
                f.write("\t" * tab + f"if(((wrow + iter_{input_a}) * TILE_{ld_tile_order_a[1].capitalize()}) >= rng_{ld_tile_order_a[1][0]})\n")
            else :
                f.write("\t" * tab + f"if((wrow + iter_{input_a}) >= rng_{ld_tile_order_a[0]})\n")
            f.write("\t" * (tab + 1) + "continue;\n\n")

        #
        if left_frag_size == 32 :
            f.write("\t" * tab + "#pragma unroll\n")
            f.write("\t" * tab + f"for(int cnt_{input_a} = 0; cnt_{input_a} < {input_a}_frag_cnt; cnt_{input_a}++)\n")
            f.write("\t" * tab + "{\n"); tab += 1

            f.write("\t" * tab + f"int {input_a}_offset0 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow + iter_{input_a}) * ld_stride_a) + ({input_a}_fragment_offset ^ (cnt_{input_a} << 4)) + (ll << {ll_offset});\n")
            f.write("\t" * tab + f"int {input_a}_offset1 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow + iter_{input_a}) * ld_stride_a) + ({input_a}_fragment_offset ^ ((cnt_{input_a} << 4) + 8)) + (ll << {ll_offset});\n")
            f.write("\t" * tab + f"int {input_a}_offset2 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow + iter_{input_a}) * ld_stride_a) + ({input_a}_fragment_offset ^ (cnt_{input_a} << 4)) + (ll << {ll_offset}) + {frag_offset};\n")
            f.write("\t" * tab + f"int {input_a}_offset3 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow + iter_{input_a}) * ld_stride_a) + ({input_a}_fragment_offset ^ ((cnt_{input_a} << 4) + 8)) + (ll << {ll_offset}) + {frag_offset};\n\n")
        #
        else :
            f.write("\t" * tab + f"int {input_a}_offset0 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow + iter_{input_a}) * ld_stride_a) + ({input_a}_fragment_offset) + (ll << {ll_offset});\n")
            f.write("\t" * tab + f"int {input_a}_offset1 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow + iter_{input_a}) * ld_stride_a) + ({input_a}_fragment_offset ^ 8) + (ll << {ll_offset});\n")
            f.write("\t" * tab + f"int {input_a}_offset2 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow + iter_{input_a}) * ld_stride_a) + ({input_a}_fragment_offset) + (ll << {ll_offset}) + {frag_offset};\n")
            f.write("\t" * tab + f"int {input_a}_offset3 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow + iter_{input_a}) * ld_stride_a) + ({input_a}_fragment_offset ^ 8) + (ll << {ll_offset}) + {frag_offset};\n\n")

        #
        if opt % 2 == 0 :
            f.write("\t" * tab + f"{input_a}_frag.x[0] = k_zero_lo ? 0.0f : sm_{input_a}[{input_a}_offset0];\n")
            f.write("\t" * tab + f"{input_a}_frag.x[1] = k_zero_lo ? 0.0f : sm_{input_a}[{input_a}_offset1];\n")
            f.write("\t" * tab + f"{input_a}_frag.x[2] = k_zero_hi ? 0.0f : sm_{input_a}[{input_a}_offset2];\n")
            f.write("\t" * tab + f"{input_a}_frag.x[3] = k_zero_hi ? 0.0f : sm_{input_a}[{input_a}_offset3];\n\n")
        else :
            f.write("\t" * tab + f"{input_a}_frag.x[0] = sm_{input_a}[{input_a}_offset0];\n")
            f.write("\t" * tab + f"{input_a}_frag.x[1] = sm_{input_a}[{input_a}_offset1];\n")
            f.write("\t" * tab + f"{input_a}_frag.x[2] = sm_{input_a}[{input_a}_offset2];\n")
            f.write("\t" * tab + f"{input_a}_frag.x[3] = sm_{input_a}[{input_a}_offset3];\n\n")
    #
    else :                                                  # FVI = Internal
        #
        frag_offset = 8 * size_internal

        #
        f.write("\t" * tab + "#pragma unroll\n")
        f.write("\t" * tab + f"for(int iter_{input_a} = 0; iter_{input_a} < wmiter; iter_{input_a}++)\n")
        f.write("\t" * tab + "{\n"); tab += 1

        #
        if opt == 3 or opt == 4 or opt == 7 or opt == 8 :
            if a_split_flag :
                f.write("\t" * tab + f"if(((wrow + iter_{input_a}) * TILE_{ld_tile_order_a[1].capitalize()}) >= rng_{ld_tile_order_a[1][0]})\n")
            else :
                f.write("\t" * tab + f"if((wrow + iter_{input_a}) >= rng_{ld_tile_order_a[0]})\n")
            f.write("\t" * (tab + 1) + "continue;\n\n")

        #
        if left_frag_size == 32 :
            f.write("\t" * tab + "#pragma unroll\n")
            f.write("\t" * tab + f"for(int cnt_{input_a} = 0; cnt_{input_a} < {input_a}_frag_cnt; cnt_{input_a}++)\n")
            f.write("\t" * tab + "{\n"); tab += 1

            cnt_offset = (int)(math.log2(16 * size_internal))

            f.write("\t" * tab + f"int {input_a}_offset0 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow + iter_{input_a}) * ld_stride_a) + ({input_a}_fragment_offset ^ ll) + (cnt_{input_a} << {cnt_offset});\n")
            f.write("\t" * tab + f"int {input_a}_offset1 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow + iter_{input_a}) * ld_stride_a) + ({input_a}_fragment_offset ^ ll) + (cnt_{input_a} << {cnt_offset}) + {frag_offset};\n")
            f.write("\t" * tab + f"int {input_a}_offset2 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow + iter_{input_a}) * ld_stride_a) + ({input_a}_fragment_offset ^ (ll + 4)) + (cnt_{input_a} << {cnt_offset});\n")
            f.write("\t" * tab + f"int {input_a}_offset3 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow + iter_{input_a}) * ld_stride_a) + ({input_a}_fragment_offset ^ (ll + 4)) + (cnt_{input_a} << {cnt_offset}) + {frag_offset};\n")
        #
        else :
            f.write("\t" * tab + f"int {input_a}_offset0 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow + iter_{input_a}) * ld_stride_a) + ({input_a}_fragment_offset ^ ll);\n")
            f.write("\t" * tab + f"int {input_a}_offset1 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow + iter_{input_a}) * ld_stride_a) + ({input_a}_fragment_offset ^ ll) + {frag_offset};\n")
            f.write("\t" * tab + f"int {input_a}_offset2 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow + iter_{input_a}) * ld_stride_a) + ({input_a}_fragment_offset ^ (ll + 4));\n")
            f.write("\t" * tab + f"int {input_a}_offset3 = shm_{input_a}_offset + {input_a}_lane_offset + ((wrow + iter_{input_a}) * ld_stride_a) + ({input_a}_fragment_offset ^ (ll + 4)) + {frag_offset};\n")

        #
        if opt % 2 == 0 :
            f.write("\t" * tab + f"{input_a}_frag.x[0] = k_zero_lo ? 0.0f : sm_{input_a}[{input_a}_offset0];\n")
            f.write("\t" * tab + f"{input_a}_frag.x[1] = k_zero_lo ? 0.0f : sm_{input_a}[{input_a}_offset1];\n")
            f.write("\t" * tab + f"{input_a}_frag.x[2] = k_zero_hi ? 0.0f : sm_{input_a}[{input_a}_offset2];\n")
            f.write("\t" * tab + f"{input_a}_frag.x[3] = k_zero_hi ? 0.0f : sm_{input_a}[{input_a}_offset3];\n\n")
        else :
            f.write("\t" * tab + f"{input_a}_frag.x[0] = sm_{input_a}[{input_a}_offset0];\n")
            f.write("\t" * tab + f"{input_a}_frag.x[1] = sm_{input_a}[{input_a}_offset1];\n")
            f.write("\t" * tab + f"{input_a}_frag.x[2] = sm_{input_a}[{input_a}_offset2];\n")
            f.write("\t" * tab + f"{input_a}_frag.x[3] = sm_{input_a}[{input_a}_offset3];\n\n")

    #
    if SMEM_order_b[2] == ld_tile_order_b[0] :              # FVI = REG
        #
        if b_double2_flag :
            step = 2 * b_double2_flag
        #
        else :
            step = 1

        ll_offset = (int)(math.log2(right_reg_size))
        frag_offset = 4 * right_reg_size

        f.write("\t" * tab + "#pragma unroll\n")
        f.write("\t" * tab + f"for(int iter_{input_b} = 0; iter_{input_b} < wniter; iter_{input_b} += {step})\n")
        f.write("\t" * tab + "{\n"); tab += 1

        if opt == 3 or opt == 4 or opt == 7 or opt == 8 :
            if b_split_flag :
                f.write("\t" * tab + f"if(((wcol + iter_{input_b}) * TILE_{ld_tile_order_b[1].capitalize()}) >= rng_{ld_tile_order_b[1][0]})\n")
            else :
                f.write("\t" * tab + f"if((wcol + iter_{input_b}) >= rng_{ld_tile_order_b[0]})\n")
            f.write("\t" * (tab + 1) + "continue;\n\n")

        #
        if right_frag_size == 32 :
            f.write("\t" * tab + "#pragma unroll\n")
            f.write("\t" * tab + f"for(int cnt_{input_b} = 0; cnt_{input_b} < {input_b}_frag_cnt; cnt_{input_b}++)\n")
            f.write("\t" * tab + "{\n"); tab += 1

            f.write("\t" * tab + f"int {input_b}_offset0 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol ^ {input_b}_fragment_offset) ^ iter_{input_b}) + (cnt_{input_b} * (ld_stride_b << 4)) + (ll << {ll_offset});\n")
            f.write("\t" * tab + f"int {input_b}_offset1 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol ^ {input_b}_fragment_offset) ^ iter_{input_b}) + (cnt_{input_b} * (ld_stride_b << 4)) + (ll << {ll_offset}) + {frag_offset};\n")
            f.write("\t" * tab + f"int {input_b}_offset2 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol ^ {input_b}_fragment_offset) ^ iter_{input_b}) + (((cnt_{input_b} << 1) + 1) * (ld_stride_b << 3)) + (ll << {ll_offset});\n")
            f.write("\t" * tab + f"int {input_b}_offset3 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol ^ {input_b}_fragment_offset) ^ iter_{input_b}) + (((cnt_{input_b} << 1) + 1) * (ld_stride_b << 3)) + (ll << {ll_offset}) + {frag_offset};\n\n")
        #
        else :
            f.write("\t" * tab + f"int {input_b}_offset0 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol ^ {input_b}_fragment_offset) ^ iter_{input_b}) + (ll << {ll_offset});\n")
            f.write("\t" * tab + f"int {input_b}_offset1 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol ^ {input_b}_fragment_offset) ^ iter_{input_b}) + (ll << {ll_offset}) + {frag_offset};\n")
            f.write("\t" * tab + f"int {input_b}_offset2 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol ^ {input_b}_fragment_offset) ^ iter_{input_b}) + (ld_stride_b << 3) + (ll << {ll_offset});\n")
            f.write("\t" * tab + f"int {input_b}_offset3 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol ^ {input_b}_fragment_offset) ^ iter_{input_b}) + (ld_stride_b << 3) + (ll << {ll_offset}) + {frag_offset};\n\n")
    
        #
        if b_double2_flag == 2 :
            f.write("\t" * tab + f"float4 tmp0 = reinterpret_cast<float4*>(&sm_{input_b}[{input_b}_offset0])[0];\n")
            f.write("\t" * tab + f"float4 tmp1 = reinterpret_cast<float4*>(&sm_{input_b}[{input_b}_offset1])[0];\n")
            f.write("\t" * tab + f"float4 tmp2 = reinterpret_cast<float4*>(&sm_{input_b}[{input_b}_offset2])[0];\n")
            f.write("\t" * tab + f"float4 tmp3 = reinterpret_cast<float4*>(&sm_{input_b}[{input_b}_offset3])[0];\n\n")

            f.write("\t" * tab + f"{input_b}_frag[0].x[0] = tmp0.x;\n")
            f.write("\t" * tab + f"{input_b}_frag[0].x[1] = tmp1.x;\n")
            f.write("\t" * tab + f"{input_b}_frag[0].x[2] = tmp2.x;\n")
            f.write("\t" * tab + f"{input_b}_frag[0].x[3] = tmp3.x;\n\n")

            f.write("\t" * tab + f"{input_b}_frag[1].x[0] = tmp0.y;\n")
            f.write("\t" * tab + f"{input_b}_frag[1].x[1] = tmp1.y;\n")
            f.write("\t" * tab + f"{input_b}_frag[1].x[2] = tmp2.y;\n")
            f.write("\t" * tab + f"{input_b}_frag[1].x[3] = tmp3.y;\n\n")

            f.write("\t" * tab + f"{input_b}_frag[2].x[0] = tmp0.z;\n")
            f.write("\t" * tab + f"{input_b}_frag[2].x[1] = tmp1.z;\n")
            f.write("\t" * tab + f"{input_b}_frag[2].x[2] = tmp2.z;\n")
            f.write("\t" * tab + f"{input_b}_frag[2].x[3] = tmp3.z;\n\n")

            f.write("\t" * tab + f"{input_b}_frag[3].x[0] = tmp0.w;\n")
            f.write("\t" * tab + f"{input_b}_frag[3].x[1] = tmp1.w;\n")
            f.write("\t" * tab + f"{input_b}_frag[3].x[2] = tmp2.w;\n")
            f.write("\t" * tab + f"{input_b}_frag[3].x[3] = tmp3.w;\n\n")
        #
        elif b_double2_flag == 1 :
            f.write("\t" * tab + f"float2 tmp0 = reinterpret_cast<float2*>(&sm_{input_b}[{input_b}_offset0])[0];\n")
            f.write("\t" * tab + f"float2 tmp1 = reinterpret_cast<float2*>(&sm_{input_b}[{input_b}_offset1])[0];\n")
            f.write("\t" * tab + f"float2 tmp2 = reinterpret_cast<float2*>(&sm_{input_b}[{input_b}_offset2])[0];\n")
            f.write("\t" * tab + f"float2 tmp3 = reinterpret_cast<float2*>(&sm_{input_b}[{input_b}_offset3])[0];\n\n")

            f.write("\t" * tab + f"{input_b}_frag[0].x[0] = tmp0.x;\n")
            f.write("\t" * tab + f"{input_b}_frag[0].x[1] = tmp1.x;\n")
            f.write("\t" * tab + f"{input_b}_frag[0].x[2] = tmp2.x;\n")
            f.write("\t" * tab + f"{input_b}_frag[0].x[3] = tmp3.x;\n\n")

            f.write("\t" * tab + f"{input_b}_frag[1].x[0] = tmp0.y;\n")
            f.write("\t" * tab + f"{input_b}_frag[1].x[1] = tmp1.y;\n")
            f.write("\t" * tab + f"{input_b}_frag[1].x[2] = tmp2.y;\n")
            f.write("\t" * tab + f"{input_b}_frag[1].x[3] = tmp3.y;\n\n")
        #
        else :
            f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset0];\n")
            f.write("\t" * tab + f"{input_b}_frag.x[1] = sm_{input_b}[{input_b}_offset1];\n")
            f.write("\t" * tab + f"{input_b}_frag.x[2] = sm_{input_b}[{input_b}_offset2];\n")
            f.write("\t" * tab + f"{input_b}_frag.x[3] = sm_{input_b}[{input_b}_offset3];\n\n")
    #
    elif SMEM_order_b[2] == ld_tile_order_b[1] :
        #
        ll_offset = (int)(math.log2(right_frag_size))
        frag_offset = 4 * right_frag_size

        f.write("\t" * tab + "#pragma unroll\n")
        f.write("\t" * tab + f"for(int iter_{input_b} = 0; iter_{input_b} < wniter; iter_{input_b}++)\n")
        f.write("\t" * tab + "{\n"); tab += 1

        if opt == 3 or opt == 4 or opt == 7 or opt == 8 :
            if b_split_flag :
                f.write("\t" * tab + f"if(((wcol + iter_{input_b}) * TILE_{ld_tile_order_b[1].capitalize()}) >= rng_{ld_tile_order_b[1][0]})\n")
            else :
                f.write("\t" * tab + f"if((wcol + iter_{input_b}) >= rng_{ld_tile_order_b[0]})\n")
            f.write("\t" * (tab + 1) + "continue;\n\n")

        #
        if right_frag_size == 32 :
            #
            f.write("\t" * tab + "#pragma unroll\n")
            f.write("\t" * tab + f"for(int cnt_{input_b} = 0; cnt_{input_b} < {input_b}_frag_cnt; cnt_{input_b}++)\n")
            f.write("\t" * tab + "{\n"); tab += 1

            f.write("\t" * tab + f"int {input_b}_offset0 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol + iter_{input_b}) * ld_stride_b) + ({input_b}_fragment_offset ^ (cnt_{input_b} << 4)) + (ll << {ll_offset});\n")
            f.write("\t" * tab + f"int {input_b}_offset1 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol + iter_{input_b}) * ld_stride_b) + ({input_b}_fragment_offset ^ (cnt_{input_b} << 4)) + (ll << {ll_offset}) + {frag_offset};\n")
            f.write("\t" * tab + f"int {input_b}_offset2 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol + iter_{input_b}) * ld_stride_b) + ({input_b}_fragment_offset ^ ((cnt_{input_b} << 4) + 8)) + (ll << {ll_offset});\n")
            f.write("\t" * tab + f"int {input_b}_offset3 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol + iter_{input_b}) * ld_stride_b) + ({input_b}_fragment_offset ^ ((cnt_{input_b} << 4) + 8)) + (ll << {ll_offset}) + {frag_offset};\n\n")
        #
        else :
            f.write("\t" * tab + f"int {input_b}_offset0 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol + iter_{input_b}) * ld_stride_b) + ({input_b}_fragment_offset) + (ll << {ll_offset});\n")
            f.write("\t" * tab + f"int {input_b}_offset1 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol + iter_{input_b}) * ld_stride_b) + ({input_b}_fragment_offset) + (ll << {ll_offset}) + {frag_offset};\n")
            f.write("\t" * tab + f"int {input_b}_offset2 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol + iter_{input_b}) * ld_stride_b) + ({input_b}_fragment_offset ^ 8) + (ll << {ll_offset});\n")
            f.write("\t" * tab + f"int {input_b}_offset3 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol + iter_{input_b}) * ld_stride_b) + ({input_b}_fragment_offset ^ 8) + (ll << {ll_offset}) + {frag_offset};\n\n")

        #
        f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset0];\n")
        f.write("\t" * tab + f"{input_b}_frag.x[1] = sm_{input_b}[{input_b}_offset1];\n")
        f.write("\t" * tab + f"{input_b}_frag.x[2] = sm_{input_b}[{input_b}_offset2];\n")
        f.write("\t" * tab + f"{input_b}_frag.x[3] = sm_{input_b}[{input_b}_offset3];\n\n")
    #
    else :
        #
        frag_offset = 8 * size_internal

        f.write("\t" * tab + "#pragma unroll\n")
        f.write("\t" * tab + f"for(int iter_{input_b} = 0; iter_{input_b} < wniter; iter_{input_b}++)\n")
        f.write("\t" * tab + "{\n"); tab += 1

        if opt == 3 or opt == 4 or opt == 7 or opt == 8 :
            if b_split_flag :
                f.write("\t" * tab + f"if(((wcol + iter_{input_b}) * TILE_{ld_tile_order_b[1].capitalize()}) >= rng_{ld_tile_order_b[1][0]})\n")
            else :
                f.write("\t" * tab + f"if((wcol + iter_{input_b}) >= rng_{ld_tile_order_b[0]})\n")
            f.write("\t" * (tab + 1) + "continue;\n\n")

        #
        if right_frag_size == 32 :
            #
            f.write("\t" * tab + "#pragma unroll\n")
            f.write("\t" * tab + f"for(int cnt_{input_b} = 0; cnt_{input_b} < {input_b}_frag_cnt; cnt_{input_b}++)\n")
            f.write("\t" * tab + "{\n"); tab += 1

            cnt_offset = (int)(math.log2(16 * size_internal))

            f.write("\t" * tab + f"int {input_b}_offset0 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol + iter_{input_b}) * ld_stride_b) + ({input_b}_fragment_offset ^ ll) + (cnt_{input_b} << {cnt_offset});\n")
            f.write("\t" * tab + f"int {input_b}_offset1 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol + iter_{input_b}) * ld_stride_b) + ({input_b}_fragment_offset ^ (ll + 4)) + (cnt_{input_b} << {cnt_offset});\n")
            f.write("\t" * tab + f"int {input_b}_offset2 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol + iter_{input_b}) * ld_stride_b) + ({input_b}_fragment_offset ^ ll) + (cnt_{input_b} << {cnt_offset}) + {frag_offset};\n")
            f.write("\t" * tab + f"int {input_b}_offset3 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol + iter_{input_b}) * ld_stride_b) + ({input_b}_fragment_offset ^ (ll + 4)) + (cnt_{input_b} << {cnt_offset}) + {frag_offset};\n")
        #
        else :
            f.write("\t" * tab + f"int {input_b}_offset0 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol + iter_{input_b}) * ld_stride_b) + ({input_b}_fragment_offset ^ ll);\n")
            f.write("\t" * tab + f"int {input_b}_offset1 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol + iter_{input_b}) * ld_stride_b) + ({input_b}_fragment_offset ^ (ll + 4));\n")
            f.write("\t" * tab + f"int {input_b}_offset2 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol + iter_{input_b}) * ld_stride_b) + ({input_b}_fragment_offset ^ ll) + {frag_offset};\n")
            f.write("\t" * tab + f"int {input_b}_offset3 = shm_{input_b}_offset + {input_b}_lane_offset + ((wcol + iter_{input_b}) * ld_stride_b) + ({input_b}_fragment_offset ^ (ll + 4)) + {frag_offset};\n")

        #
        f.write("\t" * tab + f"{input_b}_frag.x[0] = sm_{input_b}[{input_b}_offset0];\n")
        f.write("\t" * tab + f"{input_b}_frag.x[1] = sm_{input_b}[{input_b}_offset1];\n")
        f.write("\t" * tab + f"{input_b}_frag.x[2] = sm_{input_b}[{input_b}_offset2];\n")
        f.write("\t" * tab + f"{input_b}_frag.x[3] = sm_{input_b}[{input_b}_offset3];\n\n")

    #
    if (b_double2_flag == 2) and (SMEM_order_b[2] == ld_tile_order_b[0]) :
        #
        if (a_double2_flag == 2) and (SMEM_order_a[2] == ld_tile_order_a[0]) :
            #
            if left_frag_size == 32 and right_frag_size == 32 :
                f.write("\t" * tab + f"const int out_idx0 = (((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = (((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx2 = (((iter_{input_a} * wniter) + iter_{input_b} + 2) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx3 = (((iter_{input_a} * wniter) + iter_{input_b} + 3) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")

                f.write("\t" * tab + f"const int out_idx4 = ((((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx5 = ((((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx6 = ((((iter_{input_a} + 1) * wniter) + iter_{input_b} + 2) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx7 = ((((iter_{input_a} + 1) * wniter) + iter_{input_b} + 3) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                
                f.write("\t" * tab + f"const int out_idx8 = ((((iter_{input_a} + 2) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx9 = ((((iter_{input_a} + 2) * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx10 = ((((iter_{input_a} + 2) * wniter) + iter_{input_b} + 2) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx11 = ((((iter_{input_a} + 2) * wniter) + iter_{input_b} + 3) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                
                f.write("\t" * tab + f"const int out_idx12 = ((((iter_{input_a} + 3) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx13 = ((((iter_{input_a} + 3) * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx14 = ((((iter_{input_a} + 3) * wniter) + iter_{input_b} + 2) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx15 = ((((iter_{input_a} + 3) * wniter) + iter_{input_b} + 3) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            elif left_frag_size == 32 and right_frag_size == 16 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx2 = ((iter_{input_a} * wniter) + iter_{input_b} + 2) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx3 = ((iter_{input_a} * wniter) + iter_{input_b} + 3) * {input_a}_frag_cnt + cnt_{input_a};\n")

                f.write("\t" * tab + f"const int out_idx4 = (((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx5 = (((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx6 = (((iter_{input_a} + 1) * wniter) + iter_{input_b} + 2) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx7 = (((iter_{input_a} + 1) * wniter) + iter_{input_b} + 3) * {input_a}_frag_cnt + cnt_{input_a};\n")

                f.write("\t" * tab + f"const int out_idx8 = (((iter_{input_a} + 2) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx9 = (((iter_{input_a} + 2) * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx10 = (((iter_{input_a} + 2) * wniter) + iter_{input_b} + 2) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx11 = (((iter_{input_a} + 2) * wniter) + iter_{input_b} + 3) * {input_a}_frag_cnt + cnt_{input_a};\n")

                f.write("\t" * tab + f"const int out_idx12 = (((iter_{input_a} + 3) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx13 = (((iter_{input_a} + 3) * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx14 = (((iter_{input_a} + 3) * wniter) + iter_{input_b} + 2) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx15 = (((iter_{input_a} + 3) * wniter) + iter_{input_b} + 3) * {input_a}_frag_cnt + cnt_{input_a};\n")
            #
            elif left_frag_size == 16 and right_frag_size == 32 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx2 = ((iter_{input_a} * wniter) + iter_{input_b} + 2) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx3 = ((iter_{input_a} * wniter) + iter_{input_b} + 3) * {input_b}_frag_cnt + cnt_{input_b};\n")

                f.write("\t" * tab + f"const int out_idx4 = (((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx5 = (((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx6 = (((iter_{input_a} + 1) * wniter) + iter_{input_b} + 2) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx7 = (((iter_{input_a} + 1) * wniter) + iter_{input_b} + 3) * {input_b}_frag_cnt + cnt_{input_b};\n")

                f.write("\t" * tab + f"const int out_idx8 = (((iter_{input_a} + 2) * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx9 = (((iter_{input_a} + 2) * wniter) + iter_{input_b} + 1) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx10 = (((iter_{input_a} + 2) * wniter) + iter_{input_b} + 2) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx11 = (((iter_{input_a} + 2) * wniter) + iter_{input_b} + 3) * {input_b}_frag_cnt + cnt_{input_b};\n")

                f.write("\t" * tab + f"const int out_idx12 = (((iter_{input_a} + 3) * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx13 = (((iter_{input_a} + 3) * wniter) + iter_{input_b} + 1) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx14 = (((iter_{input_a} + 3) * wniter) + iter_{input_b} + 2) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx15 = (((iter_{input_a} + 3) * wniter) + iter_{input_b} + 3) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            else :
                f.write("\t" * tab + f"const int out_idx0 = (iter_{input_a} * wniter) + iter_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = (iter_{input_a} * wniter) + iter_{input_b} + 1;\n")
                f.write("\t" * tab + f"const int out_idx2 = (iter_{input_a} * wniter) + iter_{input_b} + 2;\n")
                f.write("\t" * tab + f"const int out_idx3 = (iter_{input_a} * wniter) + iter_{input_b} + 3;\n")

                f.write("\t" * tab + f"const int out_idx4 = ((iter_{input_a} + 1) * wniter) + iter_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx5 = ((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1;\n")
                f.write("\t" * tab + f"const int out_idx6 = ((iter_{input_a} + 1) * wniter) + iter_{input_b} + 2;\n")
                f.write("\t" * tab + f"const int out_idx7 = ((iter_{input_a} + 1) * wniter) + iter_{input_b} + 3;\n")

                f.write("\t" * tab + f"const int out_idx8 = ((iter_{input_a} + 2) * wniter) + iter_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx9 = ((iter_{input_a} + 2) * wniter) + iter_{input_b} + 1;\n")
                f.write("\t" * tab + f"const int out_idx10 = ((iter_{input_a} + 2) * wniter) + iter_{input_b} + 2;\n")
                f.write("\t" * tab + f"const int out_idx11 = ((iter_{input_a} + 2) * wniter) + iter_{input_b} + 3;\n")

                f.write("\t" * tab + f"const int out_idx12 = ((iter_{input_a} + 3) * wniter) + iter_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx13 = ((iter_{input_a} + 3) * wniter) + iter_{input_b} + 1;\n")
                f.write("\t" * tab + f"const int out_idx14 = ((iter_{input_a} + 3) * wniter) + iter_{input_b} + 2;\n")
                f.write("\t" * tab + f"const int out_idx15 = ((iter_{input_a} + 3) * wniter) + iter_{input_b} + 3;\n")
        #
        elif (a_double2_flag == 1) and (SMEM_order_a[2] == ld_tile_order_a[0]) :
            #
            if left_frag_size == 32 and right_frag_size == 32 :
                f.write("\t" * tab + f"const int out_idx0 = (((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = (((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx2 = (((iter_{input_a} * wniter) + iter_{input_b} + 2) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx3 = (((iter_{input_a} * wniter) + iter_{input_b} + 3) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")

                f.write("\t" * tab + f"const int out_idx4 = ((((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx5 = ((((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx6 = ((((iter_{input_a} + 1) * wniter) + iter_{input_b} + 2) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx7 = ((((iter_{input_a} + 1) * wniter) + iter_{input_b} + 3) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            elif left_frag_size == 32 and right_frag_size == 16 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx2 = ((iter_{input_a} * wniter) + iter_{input_b} + 2) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx3 = ((iter_{input_a} * wniter) + iter_{input_b} + 3) * {input_a}_frag_cnt + cnt_{input_a};\n")

                f.write("\t" * tab + f"const int out_idx4 = (((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx5 = (((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx6 = (((iter_{input_a} + 1) * wniter) + iter_{input_b} + 2) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx7 = (((iter_{input_a} + 1) * wniter) + iter_{input_b} + 3) * {input_a}_frag_cnt + cnt_{input_a};\n")
            #
            elif left_frag_size == 16 and right_frag_size == 32 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx2 = ((iter_{input_a} * wniter) + iter_{input_b} + 2) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx3 = ((iter_{input_a} * wniter) + iter_{input_b} + 3) * {input_b}_frag_cnt + cnt_{input_b};\n")

                f.write("\t" * tab + f"const int out_idx4 = (((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx5 = (((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx6 = (((iter_{input_a} + 1) * wniter) + iter_{input_b} + 2) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx7 = (((iter_{input_a} + 1) * wniter) + iter_{input_b} + 3) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            else :
                f.write("\t" * tab + f"const int out_idx0 = (iter_{input_a} * wniter) + iter_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = (iter_{input_a} * wniter) + iter_{input_b} + 1;\n")
                f.write("\t" * tab + f"const int out_idx2 = (iter_{input_a} * wniter) + iter_{input_b} + 2;\n")
                f.write("\t" * tab + f"const int out_idx3 = (iter_{input_a} * wniter) + iter_{input_b} + 3;\n")

                f.write("\t" * tab + f"const int out_idx4 = ((iter_{input_a} + 1) * wniter) + iter_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx5 = ((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1;\n")
                f.write("\t" * tab + f"const int out_idx6 = ((iter_{input_a} + 1) * wniter) + iter_{input_b} + 2;\n")
                f.write("\t" * tab + f"const int out_idx7 = ((iter_{input_a} + 1) * wniter) + iter_{input_b} + 3;\n")
        #
        else :
            #
            if left_frag_size == 32 and right_frag_size == 32 :
                f.write("\t" * tab + f"const int out_idx0 = (((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = (((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx2 = (((iter_{input_a} * wniter) + iter_{input_b} + 2) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx3 = (((iter_{input_a} * wniter) + iter_{input_b} + 3) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            elif left_frag_size == 32 and right_frag_size == 16 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx2 = ((iter_{input_a} * wniter) + iter_{input_b} + 2) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx3 = ((iter_{input_a} * wniter) + iter_{input_b} + 3) * {input_a}_frag_cnt + cnt_{input_a};\n")
            #
            elif left_frag_size == 16 and right_frag_size == 32 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx2 = ((iter_{input_a} * wniter) + iter_{input_b} + 2) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx3 = ((iter_{input_a} * wniter) + iter_{input_b} + 3) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            else :
                f.write("\t" * tab + f"const int out_idx0 = (iter_{input_a} * wniter) + iter_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = (iter_{input_a} * wniter) + iter_{input_b} + 1;\n")
                f.write("\t" * tab + f"const int out_idx2 = (iter_{input_a} * wniter) + iter_{input_b} + 2;\n")
                f.write("\t" * tab + f"const int out_idx3 = (iter_{input_a} * wniter) + iter_{input_b} + 3;\n")
    #
    elif (b_double2_flag == 1) and (SMEM_order_b[2] == ld_tile_order_b[0]) :
        #
        if (a_double2_flag == 2) and (SMEM_order_a[2] == ld_tile_order_a[0]) :
            #
            if left_frag_size == 32 and right_frag_size == 32 :
                f.write("\t" * tab + f"const int out_idx0 = (((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = (((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")

                f.write("\t" * tab + f"const int out_idx2 = ((((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx3 = ((((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                
                f.write("\t" * tab + f"const int out_idx4 = ((((iter_{input_a} + 2) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx5 = ((((iter_{input_a} + 2) * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                
                f.write("\t" * tab + f"const int out_idx6 = ((((iter_{input_a} + 3) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx7 = ((((iter_{input_a} + 3) * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            elif left_frag_size == 32 and right_frag_size == 16 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a};\n")

                f.write("\t" * tab + f"const int out_idx2 = (((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx3 = (((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a};\n")

                f.write("\t" * tab + f"const int out_idx4 = (((iter_{input_a} + 2) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx5 = (((iter_{input_a} + 2) * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a};\n")

                f.write("\t" * tab + f"const int out_idx6 = (((iter_{input_a} + 3) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx7 = (((iter_{input_a} + 3) * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a};\n")
            #
            elif left_frag_size == 16 and right_frag_size == 32 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_b}_frag_cnt + cnt_{input_b};\n")

                f.write("\t" * tab + f"const int out_idx2 = (((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx3 = (((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1) * {input_b}_frag_cnt + cnt_{input_b};\n")

                f.write("\t" * tab + f"const int out_idx4 = (((iter_{input_a} + 2) * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx5 = (((iter_{input_a} + 2) * wniter) + iter_{input_b} + 1) * {input_b}_frag_cnt + cnt_{input_b};\n")

                f.write("\t" * tab + f"const int out_idx6 = (((iter_{input_a} + 3) * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx7 = (((iter_{input_a} + 3) * wniter) + iter_{input_b} + 1) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            else :
                f.write("\t" * tab + f"const int out_idx0 = (iter_{input_a} * wniter) + iter_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = (iter_{input_a} * wniter) + iter_{input_b} + 1;\n")

                f.write("\t" * tab + f"const int out_idx2 = ((iter_{input_a} + 1) * wniter) + iter_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx3 = ((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1;\n")

                f.write("\t" * tab + f"const int out_idx4 = ((iter_{input_a} + 2) * wniter) + iter_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx5 = ((iter_{input_a} + 2) * wniter) + iter_{input_b} + 1;\n")

                f.write("\t" * tab + f"const int out_idx6 = ((iter_{input_a} + 3) * wniter) + iter_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx7 = ((iter_{input_a} + 3) * wniter) + iter_{input_b} + 1;\n")
        #
        elif (a_double2_flag == 1) and (SMEM_order_a[2] == ld_tile_order_a[0]) :
            #
            if left_frag_size == 32 and right_frag_size == 32 :
                f.write("\t" * tab + f"const int out_idx0 = (((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = (((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")

                f.write("\t" * tab + f"const int out_idx2 = ((((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx3 = ((((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            elif left_frag_size == 32 and right_frag_size == 16 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a};\n")

                f.write("\t" * tab + f"const int out_idx2 = (((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx3 = (((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a};\n")
            #
            elif left_frag_size == 16 and right_frag_size == 32 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_b}_frag_cnt + cnt_{input_b};\n")

                f.write("\t" * tab + f"const int out_idx2 = (((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx3 = (((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            else :
                f.write("\t" * tab + f"const int out_idx0 = (iter_{input_a} * wniter) + iter_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = (iter_{input_a} * wniter) + iter_{input_b} + 1;\n")

                f.write("\t" * tab + f"const int out_idx2 = ((iter_{input_a} + 1) * wniter) + iter_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx3 = ((iter_{input_a} + 1) * wniter) + iter_{input_b} + 1;\n")
        #
        else :
            #
            if left_frag_size == 32 and right_frag_size == 32 :
                f.write("\t" * tab + f"const int out_idx0 = (((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = (((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            elif left_frag_size == 32 and right_frag_size == 16 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_a}_frag_cnt + cnt_{input_a};\n")
            #
            elif left_frag_size == 16 and right_frag_size == 32 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} * wniter) + iter_{input_b} + 1) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            else :
                f.write("\t" * tab + f"const int out_idx0 = (iter_{input_a} * wniter) + iter_{input_b};\n")
                f.write("\t" * tab + f"const int out_idx1 = (iter_{input_a} * wniter) + iter_{input_b} + 1;\n")
    #
    else :
        #
        if (a_double2_flag == 2) and (SMEM_order_a[2] == ld_tile_order_a[0]) :
            #
            if left_frag_size == 32 and right_frag_size == 32 :
                f.write("\t" * tab + f"const int out_idx0 = (((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")

                f.write("\t" * tab + f"const int out_idx1 = ((((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                
                f.write("\t" * tab + f"const int out_idx2 = ((((iter_{input_a} + 2) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
                
                f.write("\t" * tab + f"const int out_idx3 = ((((iter_{input_a} + 3) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            elif left_frag_size == 32 and right_frag_size == 16 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")

                f.write("\t" * tab + f"const int out_idx1 = (((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")

                f.write("\t" * tab + f"const int out_idx2 = (((iter_{input_a} + 2) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")

                f.write("\t" * tab + f"const int out_idx3 = (((iter_{input_a} + 3) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
            #
            elif left_frag_size == 16 and right_frag_size == 32 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")

                f.write("\t" * tab + f"const int out_idx1 = (((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")

                f.write("\t" * tab + f"const int out_idx2 = (((iter_{input_a} + 2) * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")

                f.write("\t" * tab + f"const int out_idx3 = (((iter_{input_a} + 3) * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            else :
                f.write("\t" * tab + f"const int out_idx0 = (iter_{input_a} * wniter) + iter_{input_b};\n")

                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} + 1) * wniter) + iter_{input_b};\n")

                f.write("\t" * tab + f"const int out_idx2 = ((iter_{input_a} + 2) * wniter) + iter_{input_b};\n")

                f.write("\t" * tab + f"const int out_idx3 = ((iter_{input_a} + 3) * wniter) + iter_{input_b};\n")
        #
        elif (a_double2_flag == 1) and (SMEM_order_a[2] == ld_tile_order_a[0]) :
            #
            if left_frag_size == 32 and right_frag_size == 32 :
                f.write("\t" * tab + f"const int out_idx0 = (((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")

                f.write("\t" * tab + f"const int out_idx1 = ((((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            elif left_frag_size == 32 and right_frag_size == 16 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")

                f.write("\t" * tab + f"const int out_idx1 = (((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
            #
            elif left_frag_size == 16 and right_frag_size == 32 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")

                f.write("\t" * tab + f"const int out_idx1 = (((iter_{input_a} + 1) * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            else :
                f.write("\t" * tab + f"const int out_idx0 = (iter_{input_a} * wniter) + iter_{input_b};\n")

                f.write("\t" * tab + f"const int out_idx1 = ((iter_{input_a} + 1) * wniter) + iter_{input_b};\n")
        #
        else :
            #
            if left_frag_size == 32 and right_frag_size == 32 :
                f.write("\t" * tab + f"const int out_idx0 = (((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            elif left_frag_size == 32 and right_frag_size == 16 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_a}_frag_cnt + cnt_{input_a};\n")
            #
            elif left_frag_size == 16 and right_frag_size == 32 :
                f.write("\t" * tab + f"const int out_idx0 = ((iter_{input_a} * wniter) + iter_{input_b}) * {input_b}_frag_cnt + cnt_{input_b};\n")
            #
            else :
                f.write("\t" * tab + f"const int out_idx0 = (iter_{input_a} * wniter) + iter_{input_b};\n")

    #
    if b_double2_flag == 2 :
        #
        if (a_double2_flag == 2) and (SMEM_order_a[2] == ld_tile_order_a[0]) :
            #
            if SMEM_order_b[2] == ld_tile_order_b[0] :
                #
                for i in range(4) :
                    for j in range(4) :
                        f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx{i * 4 + j}], {input_a}_frag[{i}], {input_b}_frag[{j}], t3_frag[out_idx{i * 4 + j}]);\n")
            #
            else :
                #
                for i in range(4) :
                    f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx{i}], {input_a}_frag[{i}], {input_b}_frag, t3_frag[out_idx{i}]);\n")
        #
        elif (a_double2_flag == 1) and (SMEM_order_a[2] == ld_tile_order_a[0]) :
            #
            if SMEM_order_b[2] == ld_tile_order_b[0] :
                #
                for i in range(2) :
                    for j in range(4) :
                        f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx{i * 4 + j}], {input_a}_frag[{i}], {input_b}_frag[{j}], t3_frag[out_idx{i * 4 + j}]);\n")
            #
            else :
                #
                for i in range(2) :
                    f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx{i}], {input_a}_frag[{i}], {input_b}_frag, t3_frag[out_idx{i}]);\n")      
        #
        else :
            #
            if SMEM_order_b[2] == ld_tile_order_b[0] :
                #
                for i in range(4) :
                    f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx{i}], {input_a}_frag, {input_b}_frag[{i}], t3_frag[out_idx{i}]);\n")
            #
            else :
                #
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx0], {input_a}_frag, {input_b}_frag, t3_frag[out_idx0]);\n")
    #
    elif b_double2_flag == 1 :
        #
        if (a_double2_flag == 2) and (SMEM_order_a[2] == ld_tile_order_a[0]) :
            #
            if SMEM_order_b[2] == ld_tile_order_b[0] :
                #
                for i in range(4) :
                    for j in range(2) :
                        f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx{i * 4 + j}], {input_a}_frag[{i}], {input_b}_frag[{j}], t3_frag[out_idx{i * 4 + j}]);\n")
            #
            else :
                #
                for i in range(4) :
                    f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx{i}], {input_a}_frag[{i}], {input_b}_frag, t3_frag[out_idx{i}]);\n")
        #
        elif (a_double2_flag == 1) and (SMEM_order_a[2] == ld_tile_order_a[0]) :
            #
            if SMEM_order_b[2] == ld_tile_order_b[0] :
                #
                for i in range(2) :
                    for j in range(2) :
                        f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx{i * 4 + j}], {input_a}_frag[{i}], {input_b}_frag[{j}], t3_frag[out_idx{i * 4 + j}]);\n")
            #
            else :
                #
                for i in range(2) :
                    f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx{i}], {input_a}_frag[{i}], {input_b}_frag, t3_frag[out_idx{i}]);\n")
        #
        else :
            #
            if SMEM_order_b[2] == ld_tile_order_b[0] :
                #
                for i in range(2) :
                    f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx{i}], {input_a}_frag, {input_b}_frag[{i}], t3_frag[out_idx{i}]);\n")
            #
            else :
                #
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx0], {input_a}_frag, {input_b}_frag, t3_frag[out_idx0]);\n")
    #
    else :
        #
        if (a_double2_flag == 2) and (SMEM_order_a[2] == ld_tile_order_a[0]) :
            #
            for i in range(4) :
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx{i}], {input_a}_frag[{i}], {input_b}_frag, t3_frag[out_idx{i}]);\n")
        #
        elif (a_double2_flag == 1) and (SMEM_order_a[2] == ld_tile_order_a[0]) :
            #
            for i in range(2) :
                f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx{i}], {input_a}_frag[{i}], {input_b}_frag, t3_frag[out_idx{i}]);\n")
        #
        else :
            #
            f.write("\t" * tab + f"nvcuda::wmma::mma_sync(t3_frag[out_idx0], {input_a}_frag, {input_b}_frag, t3_frag[out_idx0]);\n")

    #
    for i in range(tab) :
        tab -= 1
        f.write("\t" * tab + "}\n")

    f.write("\n")

# generate __device__ compute kernel
def tc_code_kernel_dev_compute(f, kernel_name, l_splited_indices_size,
                               fvi_flag, input_a, input_b, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, SMEM_order_a, SMEM_order_b,
                               split_input, warp_shape, double2_flag, reg_padd_y, reg_padd_x, kernel_variants, data_type) :
    for i in range(1, kernel_variants + 1) :
        kernel_name_i = kernel_name + "_" + str(i)

        #
        tc_code_kernel_dev_compute_head(f, kernel_name_i, input_a, input_b, ld_tile_order_a, ld_tile_order_b, collapsed_a, collapsed_b, split_input, fvi_flag, data_type, i)

        #
        if data_type == "DOUBLE" :
            tc_code_kernel_dev_compute_body(f, l_splited_indices_size, input_a, input_b, ld_tile_order_a, ld_tile_order_b, SMEM_order_a, SMEM_order_b,
                                            warp_shape, double2_flag, reg_padd_y, reg_padd_x, fvi_flag, data_type, i)
        #
        else :
            tc_code_kernel_dev_compute_body_fp32(f, l_splited_indices_size, input_a, input_b, ld_tile_order_a, ld_tile_order_b, SMEM_order_a, SMEM_order_b,
                                            split_input, warp_shape, double2_flag, reg_padd_y, reg_padd_x, fvi_flag, data_type, i)