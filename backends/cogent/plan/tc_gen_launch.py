#!/usr/bin/python
import os
import sys
import time
import shlex
import subprocess

#
import tc_gen_main          as tc_gen
import tc_gen_configuration as tc_config

#
if len(sys.argv) != 8 :
#if len(sys.argv) != 7 :
    print ("[Generator Launcher][Launcher] Error : Wrong number of arguments.")
    sys.exit()

# Inputs
equation_number         = int(sys.argv[1])  # Equation number to be processed
t_equation_input_path   = sys.argv[2]       # Path to the equation file
configuration_number    = int(sys.argv[3])
configuration_ratio     = float(sys.argv[4])
configuration_info_flag = int(sys.argv[5])
data_type               = sys.argv[6]
variant_num             = int(sys.argv[7])

# Options
opt_print               = 0  # Print information about the equation
check_cuda              = 1  # Enable CUDA CHECK Macro in kernel codes

#
root_dir = os.getcwd()
kernels_test = os.path.join(root_dir, 'kernels_test')
#result = os.path.join(root_dir, f'results/{equation_number}_{configuration_number}_{variant_num}.txt')
result = os.path.join(root_dir, f'results/{equation_number}_{configuration_number}.txt')
binary = f'k_tccg_{equation_number}'

#fname = f"problem_{str(variant_num).zfill(4)}.in"
#equation_input_path = os.path.join(t_equation_input_path, fname)
equation_input_path = t_equation_input_path

#
print("============================================================================")
print(f"[Code Generator][Launcher] equation_number      = {equation_number}")
print(f"[Code Generator][Launcher] equation_input_path  = {equation_input_path}")
print(f"[Code Generator][Launcher] configuration_number = {configuration_number}")
print(f"[Code Generator][Launcher] data_type            = {data_type}")
print("============================================================================")

#
# pairs = []
# with open(f"./prune_test/result_pruning/{equation_number}_result.csv", "r") as f :
#     for line in f :
#         if line.startswith("#") or line.startswith("config") :
#             continue

#         cols = line.strip().split(",")
#         config = int(cols[0])
#         ori_config = int(cols[1])
#         pairs.append((config, ori_config))
# target_configs = set(c for c, _ in pairs)

eq_config_limit = {
    1: 13,  2: 2,  3: 13,   4: 4,  5: 48,
    6: 11,   7: 13,   8: 11,   9: 310,
    10: 144,  11: 144,  12: 76,   13: 263,  14: 289,
    15: 71,   16: 88,   17: 48,   18: 19,   19: 8,
    20: 557, 21: 241, 22: 196, 23: 135, 24: 226,
    25: 135, 26: 140, 27: 135, 28: 86,  29: 46,
    30: 93,  31: 76,  32: 54,  33: 54,  34: 76,
    35: 54,  36: 54,  37: 76,  38: 54,  39: 54,
    40: 54,  41: 54,  42: 76,  43: 54,  44: 54,
    45: 76,  46: 54,  47: 54,  48: 76
}

#
time_overall_start = time.time()

#
# [Step 1] Processing Inputs
#   : "tc_gen_input" should give several Outer-Groups which create their own cuda-files.
#   : An outer-group corresponds to a cuda-file.
#   : An inner-group corresponds to a kernel.
#   : An inner-group might have a tensor contraction
list_inner_groups, list_interface_info, str_binary_input = tc_config.tc_gen_configuration(equation_number, configuration_number, configuration_ratio, configuration_info_flag, equation_input_path, variant_num, opt_print, data_type)

# [Step 2] Create Kernels based on processed configurations.

for idx, config in enumerate(list_inner_groups) :
    #
    if configuration_number != -1 :
        config_num = configuration_number
    else :
        config_num = idx

    # if config_num not in target_configs :
    #     continue

    limit = eq_config_limit.get(equation_number, None)

    # limit 없으면 skip (혹은 실행)
    if limit is None:
        continue

    # limit 이상은 skip
    if config_num != limit:
        continue 

    #
    print("============================================================================")
    print(f"============================ Configuration # {config_num} =============================")

    #
    tc_gen.tc_code_gen(equation_number, [list_inner_groups[idx]], list_interface_info, -1, data_type, check_cuda)
    
    #
    config_info = [list_inner_groups[idx][0], list_inner_groups[idx][1], list_inner_groups[idx][2], list_inner_groups[idx][10], list_inner_groups[idx][11], list_inner_groups[idx][12], list_inner_groups[idx][13], list_inner_groups[idx][14], list_inner_groups[idx][15]]

    #
    os.chdir(kernels_test)
    
    #
    subprocess.run(['make', 'clean'], check=False)

    #
    try :
        print(f"Compile Equation {equation_number} - Configuration {config_num}")
        subprocess.run(['make', binary], stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e :
        print(f"[ERROR] make failed : {e.stderr.decode()}")
        os.chdir(root_dir)
        continue

    #
    os.makedirs(os.path.dirname(result), exist_ok=True)
    
    with open(result, "a") as fout :
        print(f"Config # {config_num}", file=fout)
        print(config_info, file=fout)
        fout.flush()
        try :
            cmd = [f"./{binary}"] + shlex.split(str_binary_input)
            subprocess.run(cmd, stdout=fout, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e :
            print(f"[ERROR] Execution failed : {e.stderr.decode()}")
    
    #
    os.chdir(root_dir)

#
time_overall_end = time.time()

#
print("============================================================================")
print(f"[Code Generator][Launcher] Overall Time : {time_overall_end - time_overall_start} seconds")
print("[Code Generator][Launcher] Done.")
print("============================================================================")
