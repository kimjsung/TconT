import tc_helper        as tc_helper

# configure the properties of the output tensor
def tc_code_variables_outputs(l_external_index, 
                            l_t3_d_decl_var, l_cuda_malloc, l_cuda_memcpy, l_var_output,
                            data_type) :
    output_size = ""
    
    idx_count = 0
    for each_idx in l_external_index :
        if idx_count == 0 :
            output_size = "size_" + each_idx
        else :
            output_size = output_size + " * size_" + each_idx
        idx_count = idx_count + 1
    
    if data_type == "DOUBLE" :
        l_var_output.append(["double *", "dev_t3"])
        l_t3_d_decl_var.append("double *dev_t3")
        l_cuda_malloc.append(["dev_t3", "double", output_size])
        l_cuda_memcpy.append(["double", "dev_t3", "host_t3", output_size])
        
    else:
        l_var_output.append(["float *", "dev_t3"])
        l_t3_d_decl_var.append("float *dev_t3")
        l_cuda_malloc.append(["dev_t3", "float", output_size])
        l_cuda_memcpy.append(["float", "dev_t3", "host_t3", output_size])

# configure the properties of the left tensor
def tc_code_variables_input_left(each_input_tensor, l_external_index, 
                                 l_t2_d_decl_var, l_cuda_malloc, l_cuda_memcpy, l_var_input_left,
                                 data_type) :
    d_input_name = "dev_" + each_input_tensor[0][0]
    h_input_name = "host_" + each_input_tensor[0][0]
    input_size = ""

    idx_count = 0
    for each_index in each_input_tensor[0][1] :
        if tc_helper.tc_helper_find_index(l_external_index, each_index) != -1 : # if not in, return -1
            if idx_count == 0 :
                input_size = "size_" + each_index
            else :
                input_size = input_size + " * size_" + each_index
            idx_count = idx_count + 1
        else :
            if idx_count == 0 :
                input_size = "size_" + each_index
            else :
                input_size = input_size + " * size_" + each_index
            idx_count = idx_count + 1

    #
    if data_type == "DOUBLE":
        l_var_input_left.append(["double *", d_input_name])
        l_t2_d_decl_var.append("double *__restrict " + d_input_name)
        l_cuda_malloc.append([d_input_name, "double", input_size])
        l_cuda_memcpy.append(["double", d_input_name, h_input_name, input_size])
    else :
        l_var_input_left.append(["float *", d_input_name])
        l_t2_d_decl_var.append("float *__restrict " + d_input_name)
        l_cuda_malloc.append([d_input_name, "float", input_size])
        l_cuda_memcpy.append(["float", d_input_name, h_input_name, input_size])


# configure the properties of the right tensor
def tc_code_variables_input_right(each_input_tensor, l_external_index, 
                                 l_v2_d_decl_var, l_cuda_malloc, l_cuda_memcpy, l_var_input_right,
                                 data_type) :
    d_input_name = "dev_" + each_input_tensor[1][0]
    h_input_name = "host_" + each_input_tensor[1][0]
    input_size = ""

    idx_count = 0
    for each_index in each_input_tensor[1][1] :
        if tc_helper.tc_helper_find_index(l_external_index, each_index) != -1 :
            if idx_count == 0 :
                input_size = "size_" + each_index
            else :
                input_size = input_size + " * size_" + each_index
            idx_count = idx_count + 1
        else :
            if idx_count == 0 :
                input_size = "size_" + each_index
            else :
                input_size = input_size + " * size_" + each_index
            idx_count = idx_count + 1

    #
    if data_type == "DOUBLE":
        l_var_input_right.append(["double *", d_input_name])
        l_v2_d_decl_var.append("double *__restrict " + d_input_name)
        l_cuda_malloc.append([d_input_name, "double", input_size])
        l_cuda_memcpy.append(["double", d_input_name, h_input_name, input_size])
    else :
        l_var_input_right.append(["float *", d_input_name])
        l_v2_d_decl_var.append("float *__restrict " + d_input_name)
        l_cuda_malloc.append([d_input_name, "float", input_size])
        l_cuda_memcpy.append(["float", d_input_name, h_input_name, input_size])

#
def tc_code_variables_internal_strides(each_input_tensor, l_internal_index, l_input_strides, l_indices_size) :
    tmp_input_left     = each_input_tensor[0]  # ['t2', ['c', 'a', 'd']]
    tmp_input_right    = each_input_tensor[1]  # ['v2', ['d', 'c', 'b']]

    #
    tmp_l = []
    for each_index in l_internal_index :
        stride = 1
        stride_left = []
        for idx in tmp_input_left[1] :
            if idx == each_index :
                break
            else:
                stride_left.append(f"size_{idx}")
            stride *=  tc_helper.tc_helper_find_value(l_indices_size, idx) 
        tmp_l.append([each_index, stride_left, stride])

    #
    tmp_r = []
    for each_index in l_internal_index :
        stride = 1
        stride_right = []
        for idx in tmp_input_right[1]:
            if idx == each_index:
                break
            else:
                stride_right.append(f"size_{idx}")
            stride *=  tc_helper.tc_helper_find_value(l_indices_size, idx)
        tmp_r.append([each_index, stride_right, stride])

    #
    l_input_strides.append(tmp_l)
    l_input_strides.append(tmp_r)


# configure the properties of tensors
def tc_code_variables(l_input_tensors, l_external_index, l_internal_index, l_indices_size,
                        l_t3_d_decl_var, l_t2_d_decl_var, l_v2_d_decl_var,
                        l_var_output, l_var_input_left, l_var_input_right, l_cuda_malloc, l_cuda_memcpy,
                        l_input_strides, data_type) :
    #
    tc_code_variables_outputs(l_external_index,
                              l_t3_d_decl_var, l_cuda_malloc, l_cuda_memcpy, l_var_output,
                              data_type)

    #
    for each_input_tensor in l_input_tensors :
        #
        tc_code_variables_input_left(each_input_tensor, l_external_index,
                                     l_t2_d_decl_var, l_cuda_malloc, l_cuda_memcpy, l_var_input_left,
                                     data_type)
        
        #
        tc_code_variables_input_right(each_input_tensor, l_external_index,
                                      l_v2_d_decl_var, l_cuda_malloc, l_cuda_memcpy, l_var_input_right,
                                      data_type)

        #
        tc_code_variables_internal_strides(each_input_tensor, l_internal_index, l_input_strides, l_indices_size)