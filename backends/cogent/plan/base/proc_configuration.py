import os
import sys
import tc_helper              as tc_helper
import base.alg_configuration as tc_alg_config
import base.cost_model        as tc_cost_model
import base.pruning           as tc_pruning
import base.index_mapping     as tc_mapping

#
def get_configurations(l_outer_group, tensors, index_to_extent, l_configurations_outer_group, equation, variant_num, opt_print, data_type) :
    #
    # if opt_print == 1 :
    #     print("=========================== [Configurations] ===============================")
    #     print(f"[Code Generator][get_configurations] # of Outer-Groups : {len(l_outer_group)}")

    #
    l_representative_problem_size = list()
    
    #
    l_indices = l_outer_group[0][2]
    
    #
    l_tccg_representative_problem_size = l_outer_group[0][1][0][9]

    #
    str_binary_input = " ".join([f"{val}" for val in l_tccg_representative_problem_size])

    #
    if len(l_indices) != len(l_tccg_representative_problem_size) :
        print(f"l_representative_problem_size from TCCG Benchmark : {l_tccg_representative_problem_size}", file=sys.stderr)
        print(f"len(l_indices) : {len(l_indices)} vs len(l_tccg_representative_problem_size) : {len(l_tccg_representative_problem_size)}", file=sys.stderr)
        for idx_count in range(0, len(l_indices)) :
            l_representative_problem_size.append([l_indices[idx_count], 16])
        print ("[ERROR] src.generators.configurations.get_configurations()", file=sys.stderr)
        sys.exit()
    else :
        for idx_count in range(0, len(l_indices)) :
            l_representative_problem_size.append([l_indices[idx_count], l_tccg_representative_problem_size[idx_count]])

    # Per Each-Outer-Group
    idx_outer_count = 1
    for each_outer_group in l_outer_group :
        # print(f" > Outer-Group #. {idx_outer_count}")
        
        #
        base_outer_group    = each_outer_group[0]
        list_tc             = each_outer_group[1]
        all_indices         = each_outer_group[2]
        list_info_split     = list()

        # For Each Tensor Contraction
        idx_tc_count = 1
        for each_tc in list_tc :
            # print(f" >> Tensor-Contraction [{idx_tc_count}] ")
            # print(f" : {each_tc}") 

            #
            #   Default:    Input(Left) ---> FRAG_X and REG_X
            #               But, if Input(Right) has the FVI in Output, Input(Right) ---> FRAG_X and REG_X
            #
            l_output_tensor      = each_tc[1]
            l_internal_indices   = each_tc[3]
            l_input_tensor_left  = each_tc[5]
            l_input_tensor_right = each_tc[7]
            l_info_split_idx     = []

            #
            num_external_left    = len(l_input_tensor_left) - len(l_internal_indices)
            num_external_right   = len(l_input_tensor_right) - len(l_internal_indices)

            #
            if num_external_left == 1 or num_external_right == 1 :
                #
                # print("\n[Code Generator][get_configurations] One of Input Tensors has only one external index, resulting in splitting freely.")
                
                # Tensor (Left)
                if num_external_left == 1 :
                    #
                    # print(f"(L) To Split First : {l_input_tensor_left}")

                    # To Find a Target Index in the Tensor
                    idx_count = 0
                    prev_idx = ""
                    for each_idx in l_input_tensor_left :
                        #
                        if tc_helper.tc_helper_find_index(l_internal_indices, each_idx) == -1 :
                            prev_idx = each_idx
                            each_tc[5].insert(idx_count,        each_idx + "1")
                            each_tc[5].insert(idx_count + 1,    each_idx + "2")
                            each_tc[5].pop(idx_count + 2)
                            break
                        #
                        idx_count += 1
                    
                    # To Modify the Output Tensor
                    idx_count = 0
                    for each_idx in l_output_tensor :
                        #
                        if each_idx == prev_idx :
                            each_tc[1].insert(idx_count,        each_idx + "1")
                            each_tc[1].insert(idx_count + 1,    each_idx + "2")
                            each_tc[1].pop(idx_count + 2)
                            list_info_split.append([each_idx, each_idx + "1", each_idx + "2"])
                            l_info_split_idx.append([each_idx, each_idx + "1", each_idx + "2"])
                            break
                        #
                        idx_count += 1

                    # To Modify the Representative Problem Size
                    idx_count = 0
                    for each_element in l_representative_problem_size :
                        #
                        if each_element[0] == prev_idx:
                            l_representative_problem_size.insert(idx_count,      [each_element[0] + "1", each_element[1]])
                            l_representative_problem_size.insert(idx_count + 1,  [each_element[0] + "2", each_element[1]])
                            break
                        #
                        idx_count += 1

                    # [Outer-Group] Assumption: Only One Tensor Contraction
                    idx_count = 0
                    for each_idx in all_indices :
                        #
                        if each_idx == prev_idx :
                            all_indices.insert(idx_count,       each_idx + "1")
                            all_indices.insert(idx_count + 1,   each_idx + "2")
                            all_indices.pop(idx_count + 2)
                            break
                        #
                        idx_count += 1

                    #
                    if num_external_right != 1 :
                        l_info_split_idx.append([])

                # Tensor (Right)
                if num_external_right == 1 :
                    #
                    # print(f"(R) To Split First : {l_input_tensor_right}")
                    
                    #
                    if num_external_left != 1 :
                        l_info_split_idx.append([])

                    # To Find a Target Index in the Tensor
                    idx_count   = 0
                    prev_idx    = ""
                    for each_idx in l_input_tensor_right :
                        #
                        if tc_helper.tc_helper_find_index(l_internal_indices, each_idx) == -1 :
                            prev_idx = each_idx
                            each_tc[7].insert(idx_count,        each_idx + "1")
                            each_tc[7].insert(idx_count + 1,    each_idx + "2")
                            each_tc[7].pop(idx_count + 2)
                            break
                        #
                        idx_count += 1

                    # To Modify the Output Tensor
                    idx_count = 0
                    for each_idx in l_output_tensor :
                        #
                        if each_idx == prev_idx :
                            each_tc[1].insert(idx_count,        each_idx + "1")
                            each_tc[1].insert(idx_count + 1,    each_idx + "2")
                            each_tc[1].pop(idx_count + 2)
                            list_info_split.append([each_idx, each_idx + "1", each_idx + "2"])
                            l_info_split_idx.append([each_idx, each_idx + "1", each_idx + "2"])
                            break
                        #
                        idx_count += 1

                    # To Modify the Representative Problem Size
                    idx_count = 0
                    for each_element in l_representative_problem_size :
                        #
                        if each_element[0] == prev_idx :
                            l_representative_problem_size.insert(idx_count,      [each_element[0] + "1", each_element[1]])
                            l_representative_problem_size.insert(idx_count + 1,  [each_element[0] + "2", each_element[1]])
                            break
                        #
                        idx_count += 1

                    # [Outer-Group] Assumption: Only One Tensor Contraction
                    idx_count = 0
                    for each_idx in all_indices :
                        #
                        if each_idx == prev_idx :
                            all_indices.insert(idx_count,       each_idx + "1")
                            all_indices.insert(idx_count + 1,   each_idx + "2")
                            all_indices.pop(idx_count + 2)
                            break
                        #
                        idx_count += 1
            # else :
            #     print("[Code Generator][get_configurations] Both Input Tensors have at lease two external indices, resulting in splitting exclusively.")
            
            #
            #   Input:  Equation for a Tensor Contraction, Representative Problem Size
            #   Output: List of Configurations 
            #
            index_mapping = tc_mapping.assign_mapping(tensors, index_to_extent)
            # print(f"index mapping : {index_mapping}", file=sys.stderr)
            config_struct, swap_flag, m_frag_rank, m_reg_rank = tc_pruning.index_based_config_selection(tensors, index_to_extent, index_mapping, data_type)
            l_config = tc_alg_config.build_configurations(each_tc, l_info_split_idx, l_representative_problem_size, index_mapping, swap_flag, opt_print, data_type)
            # print(f"index_mapping : {index_mapping}", file=sys.stderr)
            # print(f"config_struct : {config_struct}", file=sys.stderr)

            configuration_info_flag = 0

            #
            if len(l_config) < 1 :
                print("[Code Generator][get_configurations] ERROR : Problem(s) in Enumerating Configurations", file=sys.stderr)
                if configuration_info_flag :
                    os.makedirs("pruning_results", exist_ok=True)
                    with open(f"pruning_results/error_{equation}.txt", "a") as f :
                        f.write(f"eq : {equation}, variant : {variant_num}, # of l_configs before pruning : {len(l_config)}\n")
                sys.exit()
            
            #
            # print("============================================================================")
            # print(f"[Code Generator][get_configurations] # of Configurations --- Before Pruning : {len(l_config)}")
            # print("============================================================================")


            #
            #   Models: each configuration has its own cost.
            #
            pruning_flag = 1
            if pruning_flag :
                pruned_config = tc_pruning.apply_pruning(l_config, config_struct, swap_flag, m_frag_rank, m_reg_rank)

                if configuration_info_flag :
                    os.makedirs("configuration_info", exist_ok=True)
                    os.makedirs(f"configuration_info/eq_{equation}", exist_ok=True)
                    with open(f"configuration_info/eq_{equation}/var_{variant_num}.txt", "a") as f :
                        f.write(f"Equation : {equation}, Variant : {variant_num}\n")
                        f.write(f"l_config : {len(l_config)}, pruned_config : {len(pruned_config)}\n")
                        for config in config_struct :
                            print("============================================================================", file=f)
                            f.write(f"{config}\n")
                            print("============================================================================", file=f)
                        print("\n", file=f)
                        for idx, each_config_outer_group in enumerate(l_config) :
                            print("============================================================================", file=f)
                            print(f"Config #. {idx}", file=f)
                            each_config_outer_group.print_configuration(f, 1)
                            print("============================================================================", file=f)
                        print("\n", file=f)
                        for idx, each_config_outer_group in enumerate(pruned_config) :
                            print("============================================================================", file=f)
                            print(f"Pruned Config #. {idx}", file=f)
                            each_config_outer_group.print_configuration(f, 1)
                            print("============================================================================", file=f)

                if len(pruned_config) < 1 :
                    print(
                        f"[Code Generator][get_configurations] WARNING : apply_pruning() returned no configuration for "
                        f"eq={equation}, variant={variant_num}. Falling back to the best unpruned configuration.",
                        file=sys.stderr,
                    )
                    tc_cost_model.cost_model(l_config, data_type)
                    l_config.sort(key = lambda x: x.cost_total_v2)
                    l_configurations_outer_group.append(l_config[0])
                    if configuration_info_flag :
                        os.makedirs("pruning_results", exist_ok=True)
                        with open(f"pruning_results/error_{equation}.txt", "a") as f :
                            f.write(
                                f"eq : {equation}, variant : {variant_num}, # of configs before pruning : {len(l_config)}, "
                                f"# of configs after pruning : {len(pruned_config)}, fallback : best_unpruned\n"
                            )
                else :
                    tc_cost_model.cost_model(pruned_config, data_type)
                    pruned_config.sort(key = lambda x: x.cost_total_v2)
                    l_configurations_outer_group.append(pruned_config[0])
                    
                    # frag_n = pruned_config[0].list_FRAG_X[0]
                    # reg_n = pruned_config[0].list_REG_X[0]
                    # frag_n_tile = pruned_config[0].size_FRAG_X
                    # reg_n_tile = pruned_config[0].size_REG_X
                    # is_fvi_n = 1

                    # frag_m = pruned_config[0].list_FRAG_Y[0]
                    # reg_m = pruned_config[0].list_REG_Y[0]
                    # frag_m_tile = pruned_config[0].size_FRAG_Y
                    # reg_m_tile = pruned_config[0].size_REG_Y
                    # is_fvi_m = 0
                    
                    # internal = pruned_config[0].list_FRAG_K[0]
                    # internal_size = pruned_config[0].size_FRAG_K

                    # warp_shape = pruned_config[0].warp_shape

                    # smem_size = pruned_config[0].smem_per_block

                    # if configuration_info_flag :
                    #     os.makedirs("model/config_info3", exist_ok=True)
                    #     with open(f"model/config_info3/eq_{equation}.txt", "a") as f :
                    #         f.write(f"{equation},{variant_num},{frag_n},{reg_n},{frag_n_tile},{reg_n_tile},{is_fvi_n},{frag_m},{reg_m},{frag_m_tile},{reg_m_tile},{is_fvi_m},{internal},{internal_size},{warp_shape},{smem_size}\n")

                    # for i in pruned_config :
                    #     frag_n = i.list_FRAG_X[0]
                    #     reg_n = i.list_REG_X[0]
                    #     frag_n_tile = i.size_FRAG_X
                    #     reg_n_tile = i.size_REG_X
                    #     is_fvi_n = 1

                    #     frag_m = i.list_FRAG_Y[0]
                    #     reg_m = i.list_REG_Y[0]
                    #     frag_m_tile = i.size_FRAG_Y
                    #     reg_m_tile = i.size_REG_Y
                    #     is_fvi_m = 0
                        
                    #     internal = i.list_FRAG_K[0]
                    #     internal_size = i.size_FRAG_K

                    #     warp_shape = i.warp_shape

                    #     smem_size = i.smem_per_block

                    #     mem_cost = i.cost_total_v2
                    #     stage = i.stage
                    #     os.makedirs("tmp", exist_ok=True)
                    #     with open(f"tmp/eq_{equation}.txt", "a") as f :
                    #         f.write(f"{equation},{variant_num},{frag_n},{reg_n},{frag_n_tile},{reg_n_tile},{is_fvi_n},{frag_m},{reg_m},{frag_m_tile},{reg_m_tile},{is_fvi_m},{internal},{internal_size},{warp_shape},{smem_size},{mem_cost},{stage}\n")


            else :
                ###############################################################################################################
                tc_cost_model.cost_model_total(l_config)
                l_config.sort(key = lambda x: x.cost_total)
                
                mem_cost_threshold = 1.0
                mem_cost_selected_configs = [cfg for cfg in l_config if cfg.cost_norm <= mem_cost_threshold]
                
                #
                TBs_max_threshold = 1.0
                TBs_min_threshold = 0.0
                selected_configs = [cfg for cfg in mem_cost_selected_configs if ((cfg.num_TBs_norm <= TBs_max_threshold) and (cfg.num_TBs_norm >= TBs_min_threshold))]
            
                #
                for i in range(0, len(selected_configs)) :
                    l_configurations_outer_group.append(selected_configs[i])
                ###############################################################################################################

        #
        each_outer_group.append(list_info_split)
        idx_outer_count = idx_outer_count + 1

    return str_binary_input

#
def transform_config_inner_group(l_configurations_outer_group) :
    #
    info_each_inner_group = []
    
    #
    configs_to_process = l_configurations_outer_group

    #
    for each_configuration in configs_to_process :
        #
        temp_mapping_FRAG_s_3 = []
        temp_mapping_2D_s_3   = []
        temp_mapping_Reg_s_3  = []
        temp_slices_s_3       = []
        temp_mapping_FRAG_K   = []

        #    
        temp_mapping_2D_s_3.append(each_configuration.list_FRAG_X)
        temp_mapping_2D_s_3.append(each_configuration.list_FRAG_Y)

        #
        for each_axis in temp_mapping_2D_s_3:
            for each_idx in each_axis:
                temp_mapping_FRAG_s_3.append(each_idx)

        #
        temp_mapping_Reg_s_3.append(each_configuration.list_REG_X[0]) # REG_X
        temp_mapping_Reg_s_3.append(each_configuration.list_REG_Y[0]) # REG_Y

        #
        for each_pair in each_configuration.list_tile_sizes:
            temp_slices_s_3.append(each_pair)

        #
        for each_idx in each_configuration.list_FRAG_K:
            temp_mapping_FRAG_K.append(each_idx)

        #
        info_each_inner_group.append([temp_mapping_FRAG_s_3, temp_mapping_2D_s_3, temp_mapping_Reg_s_3, temp_slices_s_3, 
                                      each_configuration.list_representative_problem_size, temp_mapping_FRAG_K, each_configuration.warp_shape, each_configuration.stage,
                                      each_configuration.producer_cnt, each_configuration.double2_flag, each_configuration.padding])
    
    #
    return info_each_inner_group
