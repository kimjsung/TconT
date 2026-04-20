# define helper
def tc_code_define_helper(f, name, value, cnt):
    tab = "\t" * cnt
    f.write("#define ")
    f.write(name)
    f.write(f" {tab}")
    f.write(str(value))
    f.write("\n")

#
def tc_gen_define_check_cuda(f) :
    f.write("#define CHECK_CUDA(call)\t\t\t\t\t\t\t\t\t\t\t\t\t\t \\\n")
    f.write("\tdo {\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t \\\n")
    f.write("\t\tcudaError_t status_ = call;\t\t\t\t\t\t\t\t\t\t\t\t \\\n")
    f.write("\t\tif(status_ != cudaSuccess) {\t\t\t\t\t\t\t\t\t\t\t \\\n")
    f.write("\t\t\tfprintf(stderr, \"CUDA error (%s:%d) : %s:%s\\n\", __FILE__, __LINE__,  \\\n")
    f.write("\t\t\t\t\tcudaGetErrorName(status_), cudaGetErrorString(status_));\t \\\n")
    f.write("\t\t\texit(EXIT_FAILURE);\t\t\t\t\t\t\t\t\t\t\t\t\t \\\n")
    f.write("\t\t}\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t \\\n")
    f.write("\t} while(0)\n")
    f.write("\n")

# write define constants for external indices
def tc_gen_define_tile_size(f, t3_each_tile) :
    # make constant string of each external index
    name = f"TILE_{t3_each_tile[0].capitalize()}"
        
    # write to file
    tc_code_define_helper(f, name, t3_each_tile[1], 2)

# write define constants for internal indices
def tc_gen_define_internal_index(f, l_internal_idx) :
    # make constant string of each internal index
    str_internal_indices = ""
    if len(l_internal_idx) > 1:
        idx_count = 0
        for int_idx in l_internal_idx:
            if idx_count == 0:
                str_internal_indices = f"(TILE_{int_idx.capitalize()}"
            else:
                str_internal_indices = str_internal_indices + f" * TILE_{int_idx.capitalize()}"
            idx_count = idx_count + 1
    else:
        idx_count = 0
        for int_idx in l_internal_idx:
            if idx_count == 0:
                str_internal_indices = f"(TILE_{int_idx.capitalize()}"
            else:
                str_internal_indices = str_internal_indices + f" * TILE_{int_idx.capitalize()}"
            idx_count = idx_count + 1
    str_internal_indices = str_internal_indices + ")\n"

    # write to file
    tc_code_define_helper(f, "TILE_UNIT", str_internal_indices, 1)

#
def tc_gen_define_split_index(f, split_index) :
    for index in split_index :
        split_str = ""
        for i in range(1,3) :
            if i == 1 :
                split_str = f"(TILE_" + index.capitalize() + str(i)
            else :
                split_str = split_str + " * TILE_" + index.capitalize() + str(i)
        split_str = split_str + ")\n"
        tc_code_define_helper(f, f"TILE_{index.capitalize()}", split_str, 2)

#
def tc_code_define_strides(f, str_eq_num, l_t3_mapping_tb_2D, name_a, name_b, str_ld_stride_a, size_ld_stride_a, str_ld_stride_b, size_ld_stride_b) :
    #
    str_num_threads = "TCCG_" + str_eq_num + "_TILE_" + l_t3_mapping_tb_2D[0][0].capitalize() + " * TCCG_" + str_eq_num + "_TILE_" + l_t3_mapping_tb_2D[1][0].capitalize()

    #
    tc_code_define_helper(f, name_a + "_ld_stride", "\t(" + str_num_threads + ") / (" + str_ld_stride_a + ")")
    tc_code_define_helper(f, name_a + "_ld_stride_size", size_ld_stride_a)
    
    f.write("\n")
    
    #
    tc_code_define_helper(f, name_b + "_ld_stride", "\t(" + str_num_threads + ") / (" + str_ld_stride_b + ")")
    tc_code_define_helper(f, name_b + "_ld_stride_size", size_ld_stride_b)
    
    f.write("\n")

#
def tc_code_define_load_strides(f, input_a, input_b, SMEM_order_a, SMEM_order_b) :
    #
    plane_a = f"(TILE_{SMEM_order_a[1].capitalize()} * TILE_{SMEM_order_a[2].capitalize()})"
    a_elements = f"(plane_{input_a} * TILE_{SMEM_order_a[0].capitalize()})\n"
    
    #
    plane_b = f"(TILE_{SMEM_order_b[1].capitalize()} * TILE_{SMEM_order_b[2].capitalize()})"
    b_elements = f"(plane_{input_b} * TILE_{SMEM_order_b[0].capitalize()})\n"
    
    #
    tc_code_define_helper(f, f"plane_{input_a}", plane_a, 3)
    tc_code_define_helper(f, f"{input_a}_elements", a_elements, 2)

    #
    tc_code_define_helper(f, f"plane_{input_b}", plane_b, 3)
    tc_code_define_helper(f, f"{input_b}_elements", b_elements, 2)

# write define constants
def tc_code_define(f, l_internal_idx, stages_count, l_splited_indices_size, producer_cnt, input_a, input_b, SMEM_order_a, SMEM_order_b, split_index, check_cuda) :
    #
    if check_cuda == 1 :
        tc_gen_define_check_cuda(f)

    #
    for each_index in l_splited_indices_size :
        tc_gen_define_tile_size(f, each_index)
    f.write("\n")

    # make and write define constants for internal index
    tc_gen_define_internal_index(f, l_internal_idx)
    
    #
    if len(split_index) > 0 :
        tc_gen_define_split_index(f, split_index)

    #
    tc_code_define_load_strides(f, input_a, input_b, SMEM_order_a, SMEM_order_b)

    # Pipeline stages
    tc_code_define_helper(f, "PIPELINE_STAGES", stages_count[0], 1)

    # Producer count
    if producer_cnt > 0 :
        tc_code_define_helper(f, "PRODUCER_CNT", producer_cnt, 2)
    f.write("\n")
    
    # write ceiling macro function
    tc_code_define_helper(f, "CEIL(a, b)",  "\t\t(((a) + (b) - 1) / (b))\n", 1)