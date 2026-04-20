#
#   Configuration
#
import sys


class Config:
    def __init__(self):
        #
        self.list_representative_problem_size = []
        self.list_tile_sizes                  = []

        #
        self.list_tensor_C = []
        self.list_tensor_A = []
        self.list_tensor_B = []
        self.list_FRAG_X   = []
        self.list_FRAG_Y   = []
        self.list_FRAG_K   = []
        self.list_REG_X    = []
        self.list_REG_Y    = []
        self.list_GRID_X   = []
        self.list_splits   = []

        #
        self.list_possible_split_cases              = []
        self.list_split_representative_problem_size = []
        self.combined_tile_size = []

        #
        self.size_FRAG_X = 1
        self.size_FRAG_Y = 1
        self.size_FRAG_K = 1
        self.size_REG_X  = 1
        self.size_REG_Y  = 1

        #
        self.kernel_full_ext = True
        self.kernel_full_int = True

        #
        self.kernel_used_registers = 0

        #
        self.kernel_comp                 = 0
        self.kernel_arithmetic_intensity = 0

        #
        self.cost_load_TB      = 0
        self.cost_total        = 0
        self.cost_load_input   = 0
        self.cost_load_output  = 0
        self.cost_store_output = 0
        self.steps_main_loops  = 0

        #
        self.cost_load_TB_d2      = 0
        self.cost_total_d2        = 0
        self.cost_load_input_d2   = 0
        self.cost_load_output_d2  = 0
        self.cost_store_output_d2 = 0
        self.steps_main_loops_d2  = 0

        #
        self.transaction_per_loop = 0
        self.flops_per_loop       = 0
        self.overlap_frac         = 0
        self.bytes_per_loop       = 0
        self.lines128_per_loop    = 0
        self.mem_issue_per_loop   = 0

        #
        self.cost_total_overlap = 0
        self.cost_total_d2_overlap = 0

        #
        self.partial_total_cost_d2 = 0
        self.partial_load_a_cost_d2 = 0
        self.partial_load_b_cost_d2 = 0
        self.partial_load_a_cost_tb_d2 = 0
        self.partial_load_b_cost_tb_d2 = 0
        self.partial_total_cost_d2_overlap = 0
        self.partial_overlap_frac = 0
        self.partial_transaction_per_loop = 0
        self.t2_total_tx_cost = 0
        self.v2_total_tx_cost = 0

        #
        self.cost_norm        = 0.0

        #
        self.rank = -1

        #
        self.m = 0
        self.n = 0
        self.k = 0

        #
        self.mn_ratio = 0.0
        self.frag_ratio = 0.0
        self.tile_ratio = 0.0

        #
        self.num_TBs = 0
        self.num_Frag_Regs = 0

        #
        self.num_TBs_norm = 0.0

        #
        self.stage = 0

        #
        self.producer_cnt = 0

        #
        self.warp_shape = []
        
        #
        self.double2_flag = []

        #
        self.eff_L_per_mma = 0.0
        self.ai_mem = 0.0
        self.partial_L_per_mma = 0.0
        self.tb_to_cap_ratio = 0.0
        self.mem_metric = 0.0
        
        #
        self.v2_tx_per_tb        = 0.0
        self.t2_tx_per_tb        = 0.0
        self.v2_cost_load        = 0.0
        self.t2_cost_load        = 0.0
        self.cost_load_total_v2  = 0.0
        self.cost_store_total_v2 = 0.0
        self.cost_total_v2       = 0.0
        self.exposed_frac_v2     = 0.0
        self.tx_per_loop_v2      = 0.0
        self.cta                 = 0
        self.cost_load_issue_v2  = 0.0
        
        #
        self.padding = []
        self.reg_y_padd = 0
        self.reg_x_padd = 0
        self.smem_per_block = 0
        
        #
        self.idx = 0

    #
    def add_split_representative_problem_size(self, split_size) :
        for each_pair in split_size :
            self.list_split_representative_problem_size.append(each_pair)

    #
    def add_representative_problem_size(self, representative_problem_size) :
        for each_pair in representative_problem_size :
            self.list_representative_problem_size.append(each_pair)

    #
    def add_tile_size(self, list_input_tile_sizes) :
        for each_pair in list_input_tile_sizes :
            self.list_tile_sizes.append(each_pair)

    #
    def add_split_index(self, list_split_info) :
        for each_idx in list_split_info :
            if each_idx != [] :
                self.list_splits.append([each_idx[0], each_idx[1], each_idx[2]])

    #
    def add_tensor_C(self, tensor_C) :
        for each_info in tensor_C :
            self.list_tensor_C.append(each_info)

    #
    def add_tensor_A(self, tensor_A) :
        for each_info in tensor_A :
            self.list_tensor_A.append(each_info)
    
    #
    def del_idx_tensor_A(self, str_target_idx) :
        idx_count = 0
        for each_idx in self.list_tensor_A :
            if each_idx == str_target_idx :
                self.list_tensor_A.pop(idx_count)

            idx_count = idx_count + 1
            
    #
    def offset_tensor_A(self, str_target_idx) :
        idx_count = 0
        for each_idx in self.list_tensor_A :
            if each_idx == str_target_idx :
                return idx_count
            
            idx_count = idx_count + 1 

        return -1

    #
    def add_tensor_B(self, tensor_B) :
        for each_info in tensor_B :
            self.list_tensor_B.append(each_info)

    #
    def del_idx_tensor_B(self, str_target_idx) :
        idx_count = 0
        for each_idx in self.list_tensor_B :
            if each_idx == str_target_idx :
                self.list_tensor_B.pop(idx_count)
            
            idx_count = idx_count + 1

    #
    def offset_tensor_B(self, str_target_idx) :
        idx_count = 0
        for each_idx in self.list_tensor_B :
            if each_idx == str_target_idx :
                return idx_count
            
            idx_count = idx_count + 1
        
        return -1

    #
    def add_GRID_X(self, GRID_X) :
        for each_info in GRID_X :
            self.list_GRID_X.append(each_info)

    #
    def add_FRAG_X(self, FRAG_X) :
        for each_info in FRAG_X :
            self.list_FRAG_X.append(each_info)

    #
    def add_FRAG_Y(self, FRAG_Y) :
        for each_info in FRAG_Y :
            self.list_FRAG_Y.append(each_info)

    #
    def add_FRAG_K(self, FRAG_K) :
        for each_info in FRAG_K :
            self.list_FRAG_K.append(each_info)

    #
    def add_REG_X(self, REG_X) :
        for each_info in REG_X :
            self.list_REG_X.append(each_info)

    #
    def add_REG_Y(self, REG_Y) :
        for each_info in REG_Y :
            self.list_REG_Y.append(each_info)
    
    #
    def add_MNK(self, m, n, k) :
        self.m = m
        self.n = n
        self.k = k

    #
    def add_WARP_SHAPE(self, warp_shape) :
        self.warp_shape = warp_shape
    
    #
    def add_STAGE(self, stage) :
        self.stage = stage

    #
    def add_PRODUCER_CNT(self, producer_cnt) :
        self.producer_cnt = producer_cnt

    #
    def add_double2(self, double2_left_flag, double2_right_flag) :
        self.double2_flag = [double2_left_flag, double2_right_flag]

    #
    def print_representative_problem_size(self, f) :
        print(f"Representative Problem Size : {self.list_representative_problem_size}", file=f)
    #
    def print_split_representative_problem_size(self, f) :
        print(f"Split-Representative Problem-Size : {self.list_split_representative_problem_size}", file=f)

    #
    def print_splits(self) :
        print(f"Split Indices : {self.list_splits}")

    #
    def print_TC_Equation(self, f) :
        print(f"TC : {self.list_tensor_C} = {self.list_tensor_A} * {self.list_tensor_B}", file=f)

    #
    def print_tensor_C(self) :
        print(f"Tensor C : {self.list_tensor_C}")

    #
    def print_tensor_A(self) :
        print(f"Tensor A : {self.list_tensor_A}")
    
    #
    def print_tensor_B(self) :
        print(f"Tensor B : {self.list_tensor_B}")

    #
    def print_REG_X(self) :
        print(f"REG_X : {self.list_REG_X}")

    #
    def print_REG_Y(self) :
        print(f"REG_Y : {self.list_REG_Y}")

    #
    def print_REG(self) :
        print(f"REG_X : {self.list_REG_X}, REG_Y : {self.list_REG_Y}")

    #
    def print_FRAG_X(self) :
        print(f"FRAG_X : {self.list_FRAG_X}")

    #
    def print_FRAG_Y(self) :
        print(f"FRAG_Y : {self.list_FRAG_Y}")
    
    #
    def print_FRAG_K(self) :
        print(f"FRAG_K : {self.list_FRAG_K}")

    #
    def print_FRAG(self) :
        print(f"FRAG_X : {self.list_FRAG_X}, FRAG_Y : {self.list_FRAG_Y}, FRAG_K : {self.list_FRAG_K}")

    #
    def print_GRID_X(self) :
        print(f"BX_X : {self.list_GRID_X}")

    #
    def print_tile_sizes(self, f) :
        print(f"Tile-Sizes : {self.list_tile_sizes}", file=f)

    #
    def print_combined_tile_sizes(self, f) :
        print(f"Combined Tile-Sizes : {self.combined_tile_size}", file=f)

    #
    def print_kernel_full(self) :
        print(f"Kernel (Full) Ext : {self.kernel_full_ext}, Int : {self.kernel_full_int}")

    #
    def print_arithmetic_intensity(self) :
        print(f"Kernel--- arithmetic intensity : {self.kernel_arithmetic_intensity}")

    #
    def print_mnk(self) :
        print(f"M : {self.m}, N : {self.n}, K : {self.k}")

    #
    def print_numTBs(self) :
        print(f"The estimated number of TBs : {self.num_TBs}")
    
    #
    def print_numFragRegs(self) :
        print(f"Output Fragment Registers : {self.num_Frag_Regs}")

    #
    def print_warp_shape(self, f) :
        print(f"Warp Shape : {self.warp_shape}", file=f)

    #
    def print_stage(self, f) :
        print(f"Pipeline Stage : {self.stage}", file=f)

    #
    def print_producer_cnt(self) :
        print(f"Producer Count : {self.producer_cnt}")

    #
    def print_double2(self) :
        print(f"Double2 flag : {self.double2_flag}")

    #
    def print_mn_ratio(self) :
        print(f"MN_Ratio : {self.mn_ratio}")

    #
    def print_configuration(self, f, opt=0, str="") :        
        #
        # print(f"Ori Config #. {self.idx}")
        self.print_TC_Equation(f)
        self.print_representative_problem_size(f)
        self.print_split_representative_problem_size(f)
        # self.print_splits()
        self.print_tile_sizes(f)
        self.print_combined_tile_sizes(f)
        # self.print_FRAG()
        # self.print_REG()
        # self.print_arithmetic_intensity()
        # self.print_mnk()
        # self.print_mn_ratio()
        # self.print_numTBs()
        # self.print_numFragRegs()
        self.print_warp_shape(f)
        self.print_stage(f)
        # self.print_producer_cnt()
        # self.print_double2()

        # print(f"|FRAG| = {self.size_FRAG_X}, {self.size_FRAG_Y}, |REG| = {self.size_REG_X}, {self.size_REG_Y}, |FRAG_K| = {self.size_FRAG_K}")
        # print(f"Total-Cost : {self.cost_total}")
        # print(f"Total-Cost-d2 : {self.cost_total_d2}")
        # print(f"Transaction per loop : {self.transaction_per_loop}")
        # print(f"FLOPS per loop : {self.flops_per_loop}")
        # print(f"Overlap fraction : {self.overlap_frac}")
        # print(f"Cost-total-overlap : {self.cost_total_overlap}")
        # print(f"Cost-total-d2-overlap : {self.cost_total_d2_overlap}")
        # print(f"Bytes per loop : {self.bytes_per_loop}")
        # print(f"Lines per loop : {self.lines128_per_loop}")
        # print(f"Memory issue per loop : {self.mem_issue_per_loop}")
        # print(f"Total-Cost-d2-partial : {self.partial_total_cost_d2}")
        # print(f"Total-Cost-d2-partial-overlap : {self.partial_total_cost_d2_overlap}")
        # print(f"partial tx per loop : {self.partial_transaction_per_loop}")
        # print(f"t2_total_tx_cost : {self.t2_total_tx_cost}")
        # print(f"v2_total_tx_cost : {self.v2_total_tx_cost}")
        # print(f"partial overlap frac : {self.partial_overlap_frac}")
        # print(f"TBs-norm : {self.num_TBs_norm}")
        # print(f"v2_tx_per_tb : {self.v2_tx_per_tb}")
        # print(f"t2_tx_per_tb : {self.t2_tx_per_tb}")
        # print(f"v2_cost_load : {self.v2_cost_load}")
        # print(f"t2_cost_load : {self.t2_cost_load}")
        # print(f"cost_load_total_v2 : {self.cost_load_total_v2}")
        # print(f"cost_store_total_v2 : {self.cost_store_total_v2}")
        # print(f"cost_total_v2 : {self.cost_total_v2}")
        # print(f"exposed_frac_v2 : {self.exposed_frac_v2}")
        # print(f"tx_per_loop_v2 : {self.tx_per_loop_v2}")
        # print(f"cost_load_issue_v2 : {self.cost_load_issue_v2}")
        # print(f"cta : {self.cta}")

        # print(f"# of steps for main-loop : {self.steps_main_loops}")
        # print("============================================================================")