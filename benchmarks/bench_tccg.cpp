#include "tcont.hpp"
#include "bench_tccg.hpp"
#include "bench_tccg_helpers.hpp"

int main(int argc, char *argv[])
{
    int target_tccg_benchmark;
    
    for(int i = 1; i < argc; ++i) 
    {
        const char* arg = argv[i];

        if(std::strcmp(arg, "-b") == 0 || std::strcmp(arg, "--benchmark") == 0) 
        {
            if (i + 1 < argc) {
                target_tccg_benchmark = atoi(argv[++i]);
            }
        } 
        else if(std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) 
        {
            printf("Usage: %s [-b <benchmark_id>]\n", argv[0]);
            printf("  -b, --benchmark <id>       Specify TCCG benchmark ID (1-48)\n");
            return 0;
        }
    }

    if(target_tccg_benchmark < 0 || target_tccg_benchmark >= list_tccg_bench.size()) {
        printf("Invalid benchmark ID: %d. Valid range is 0 to %lu.\n", target_tccg_benchmark, list_tccg_bench.size() - 1);
        return -1;
    }

    printf("-----------------------------------------------------------------------\n");
    printf("[TconT] Target TCCG's Benchmark: %d\n", target_tccg_benchmark);
    printf("[TconT] column-major format.\n");
    
    print_equation(list_tccg_bench[target_tccg_benchmark]);

    int64_t size_A = compute_tensor_size(list_tccg_bench[target_tccg_benchmark].modeA, list_tccg_bench[target_tccg_benchmark].extent);
    int64_t size_B = compute_tensor_size(list_tccg_bench[target_tccg_benchmark].modeB, list_tccg_bench[target_tccg_benchmark].extent);
    int64_t size_C = compute_tensor_size(list_tccg_bench[target_tccg_benchmark].modeC, list_tccg_bench[target_tccg_benchmark].extent);

    printf("[TconT]  Size A: %12ld (words), %16ld (bytes)\n", size_A, size_A * sizeof(double));
    printf("[TconT]  Size B: %12ld (words), %16ld (bytes)\n", size_B, size_B * sizeof(double));
    printf("[TconT]  Size C: %12ld (words), %16ld (bytes)\n", size_C, size_C * sizeof(double));
    printf("[TconT] # FLOPS: %20ld\n", 2 * (int64_t)std::sqrt(size_A * size_B * size_C)); // 2 * m * n * k for GEMM
    printf("-----------------------------------------------------------------------\n");

    printf("[TconT] Selct TTGT or Direct\n");
    TconT::TCEquation desc = list_tccg_bench[target_tccg_benchmark];

    TconT::Backend b = TconT::plan(desc);

    TconT::contract(b, desc);
}