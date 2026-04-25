#pragma once
#include <cstdio>
#include <cstdlib>
#include <ctime>

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
void init_tensors(T* output, int size_output,
                  T* input_left, int size_input_left,
                  T* input_right, int size_input_right)
{
    if (!output || !input_left || !input_right) {
        std::fprintf(stderr, "Error: Null pointer passed to init_tensors.\n");
        return;
    }
    if (size_output < 0 || size_input_left < 0 || size_input_right < 0) {
        std::fprintf(stderr, "Error: Negative size passed to init_tensors.\n");
        return;
    }

    static int seeded = 0;
    static unsigned int base_seed = 0;
    if (!seeded) {
        base_seed = static_cast<unsigned int>(std::time(nullptr));
        std::srand(base_seed);
        seeded = 1;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < size_output; ++i) {
        output[i] = static_cast<T>(0);
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        unsigned int seed = base_seed;
#ifdef _OPENMP
        seed ^= static_cast<unsigned int>(omp_get_thread_num()) * 2654435761u;
#endif

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int i = 0; i < size_input_left; ++i) {
            input_left[i] = static_cast<T>(static_cast<double>(rand_r(&seed)) / RAND_MAX);
        }

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int i = 0; i < size_input_right; ++i) {
            input_right[i] = static_cast<T>(static_cast<double>(rand_r(&seed)) / RAND_MAX);
        }
    }
}
