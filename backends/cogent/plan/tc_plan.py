import sys, json, os
import tc_gen_inner_group as config
from generators.code_gen import tc_code_gen

if __name__ == "__main__" :
    #
    data = json.loads(sys.argv[1])

    # data type
    dtype = data["type"]

    # index for tensor
    t3 = [chr(i) for i in data["modeC"]]
    t2 = [chr(i) for i in data["modeA"]]
    v2 = [chr(i) for i in data["modeB"]]

    # index size
    extentC = data["extentC"]
    extentA = data["extentA"]
    extentB = data["extentB"]

    # operation
    op = chr(data["op"])

    # All indices
    all_indices = list(dict.fromkeys(t3 + t2 + v2))

    # Internal indices
    k_indices = [x for x in t2 if x not in t3 and x in v2]

    # extent mapping
    index_to_extent = {}
    for i, k in enumerate(t3):
        index_to_extent[k] = extentC[i]
    for i, k in enumerate(t2):
        if k not in index_to_extent:
            index_to_extent[k] = extentA[i]
    for i, k in enumerate(v2):
        if k not in index_to_extent:
            index_to_extent[k] = extentB[i]

    #
    extent_pairs = [[k, index_to_extent[k]] for k in all_indices]
    extent_values = [index_to_extent[k] for k in all_indices]

    #
    equation_info = [
        [
            t3,
            [
                [
                    't3', t3,
                    op,
                    k_indices,
                    't2', t2,
                    'v2', v2,
                    extent_pairs,
                    extent_values
                ]
            ],
            all_indices
        ]
    ]

    #
    tensors = [
        t3,
        k_indices,
        t2,
        v2
    ]

    #
    # equation = data["equation"]
    # variant_num = data["variant"]
    equation = 0
    variant_num = 0

    opt_print = 0
    l_inner_groups, str_binary_input = config.tc_gen_inner_group(equation_info, tensors, index_to_extent, equation, variant_num, opt_print, dtype)
    l_temp_inner_output, l_kernal_binary = config.tc_gen_processing_inner_group(l_inner_groups, equation_info, opt_print, dtype)

    kernel_bin = config.make_kernel_name(l_kernal_binary)
    launch_config = config.make_launch_config(l_kernal_binary, kernel_bin, l_temp_inner_output[0][4], l_temp_inner_output[0][5], l_temp_inner_output[0][8], index_to_extent)

    backend_path = "backends/cogent"
    bin_path = os.path.join(backend_path, "bin/" + kernel_bin + ".cubin")
    code_path = os.path.join(backend_path, "code/" + kernel_bin + ".cu")
    
    if not os.path.exists(bin_path) :
        print(f"Kernel does not exist at {bin_path}.\nGenerating kernel...", file=sys.stderr)
        os.makedirs(backend_path + "/code", exist_ok=True)
        tc_code_gen(l_temp_inner_output, code_path, dtype, -1, 0)
    else :
        print(f"Kernel already exists at {bin_path}.\nSkipping code generation.", file=sys.stderr)
    
    print(json.dumps(launch_config))