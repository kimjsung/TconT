/**
 * @file ttgt_helpers.cpp
 * @author Jinsung Kim (kimjsung@cau.ac.kr)
 * @brief 
 * @version 0.1
 * @date 2026-02-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
    if( err != cudaSuccess )                                      \
    {   printf("CUDA Error: %s\n", cudaGetErrorName(err));          \
        printf("Error: %s\n", cudaGetErrorString(err)); exit(-1); } \
};

#define CUBLAS_CHECK(call)                                        \
do {                                                              \
    cublasStatus_t _status = (call);                              \
    if (_status != CUBLAS_STATUS_SUCCESS) {                       \
        fprintf(stderr,                                           \
                "CUBLAS error at %s:%d — %s\n",                   \
                __FILE__, __LINE__,                               \
                cublasGetStatusString(_status));                  \
        exit(EXIT_FAILURE);                                       \
    }                                                             \
} while (0)

//
void init_tensors(double* output, int size_output, 
                  double* input_left, int size_input_left, 
                  double* input_right, int size_input_right)
{
	// Basic argument validation
	if (!output || !input_left || !input_right) {
		fprintf(stderr, "Error: Null pointer passed to init_tensors.\n");
		return;
	}

	if (size_output < 0 || size_input_left < 0 || size_input_right < 0) {
		fprintf(stderr, "Error: Negative size passed to init_tensors.\n");
		return;
	}

	// Seed RNG once (used to derive per-thread seeds below)
	static int seeded = 0;
	static unsigned int base_seed = 0;
	if (!seeded) {
		base_seed = (unsigned int)time(NULL);
		srand(base_seed); // Seed the global RNG
		seeded = 1;
	}

	// Initialize output tensor to zero (parallel)
	#ifdef _OPENMP
	#pragma omp parallel for schedule(static)
	#endif
	for (int i = 0; i < size_output; i++) {
		output[i] = 0.0;
	}

	// Initialize inpout tensors with random values in [0,1) (parallel, thread-safe)
	#ifdef _OPENMP
	#pragma omp parallel
	#endif
	{
	// Thread-local seed for rand_r (avoids data race on rand())
		unsigned int seed = base_seed;
		#ifdef _OPENMP
		seed ^= (unsigned int)omp_get_thread_num() * 2654435761u; // Derive unique seed per thread
		#endif
		
		#ifdef _OPENMP
		#pragma omp for schedule(static)
		#endif
		for (int i = 0; i < size_input_left; i++) {
			// rand_r returns [0, RAND_MAX], so we divide by RAND_MAX to get [0,1)
			input_left[i] = (double)rand_r(&seed) / (double)RAND_MAX;
		}

		#ifdef _OPENMP
		#pragma omp for schedule(static)
		#endif
		for (int i = 0; i < size_input_right; i++) {
			input_right[i] = (double)rand_r(&seed) / (double)RAND_MAX;
		}
	}
}



// 
// tc* create_tc(char* string_tensor_contraction, int* int_problem_size, unsigned long long int* num_operations)
// {
//     //
//     tc* sample_sd2_1 = (tc*)malloc(sizeof(tc));
    
//     //
//     //  Input-Form: len_output, ext_1, ext_2,..., ext_n, op, len_internal, int_1, ..., int_k, len_input_left, ext_i, ..., ext_j, len_input_right, ext_m, ..., ext_n,
//     //
//     printf ("[TTGT-TTLG] Given Tensor Contraction:\n");
//     printf ("[TTGT-TTLG] %s\n", string_tensor_contraction);
//     char*   token;
//     int     len_output, len_input_left, len_input_right, len_internal;
//     int     size_output, size_input_left, size_input_right;

//     //
//     //  Output
//     //
//     token                       = strtok(string_tensor_contraction, ",");
//     len_output                  = atoi(token);
//     sample_sd2_1->len_output    = len_output;
//     sample_sd2_1->info_output   = (idx*)malloc(sizeof(idx) * len_output);

//     size_output = 1;
//     for (int i = 0; i < len_output; i++)
//     {
//         token = strtok(NULL, ",");
//         strncpy(sample_sd2_1->info_output[i].name, token, SIZE_NAME);
//         sample_sd2_1->info_output[i].size = SIZE_IDX_DEFAULT;
//         size_output *= SIZE_IDX_DEFAULT;
//     }

//     //
//     //  Operator
//     //
//     token = strtok(NULL, ",");
//     if (strncmp(token, "+=", 2) == 0)
//     {
//         sample_sd2_1->op = 1;
//     }
//     else if (strncmp(token, "-=", 2) == 0)
//     {
//         sample_sd2_1->op = 2;
//     }
//     else
//     {
//         printf ("ERROR: a given operator: %s ---> default: +=\n", token);
//         sample_sd2_1->op = 1;
//     }

//     //
//     //  Internal Indices 
//     //
//     token                               = strtok(NULL, ",");
//     len_internal                        = atoi(token);
//     sample_sd2_1->len_internal_indices  = len_internal;
//     sample_sd2_1->info_internal_indices = (idx*)malloc(sizeof(idx) * len_internal);

//     for (int i = 0; i < len_internal; i++)
//     {
//         token = strtok(NULL, ",");
//         strncpy(sample_sd2_1->info_internal_indices[i].name, token, SIZE_NAME);
//         sample_sd2_1->info_internal_indices[i].size = SIZE_IDX_DEFAULT;
//     }
    
//     //
//     //  Input-Left
//     //
//     token = strtok(NULL, ",");
//     len_input_left = atoi(token);
//     sample_sd2_1->len_input_left    = len_input_left;
//     sample_sd2_1->info_input_left   = (idx*)malloc(sizeof(idx) * len_input_left);

//     for (int i = 0; i < len_input_left; i++)
//     {
//         token = strtok(NULL, ",");
//         strncpy(sample_sd2_1->info_input_left[i].name, token, SIZE_NAME);
//     }

//     //
//     //  Input-Right
//     //
//     token = strtok(NULL, ",");
//     len_input_right = atoi(token);
//     sample_sd2_1->len_input_right   = len_input_right;
//     sample_sd2_1->info_input_right  = (idx*)malloc(sizeof(idx) * len_input_right);

//     for (int i = 0; i < len_input_right; i++)
//     {
//         token = strtok(NULL, ",");
//         strncpy(sample_sd2_1->info_input_right[i].name, token, SIZE_NAME);
//     }

//     //
//     //  Post-Process (Problem Sizes: Index's Size, Tensor's Size)
//     //
//     int     len_int_indices                             = (len_input_right + len_input_left - len_output) / 2;
//     int     len_ext_indices                             = len_output;
//     int     len_total_indices                           = len_int_indices + len_ext_indices;
//     char    char_indices[LENGTH_TMP_INDICES][SIZE_NAME] = {"a","b","c","d","e","f","g","h","i","j"};

//     //
//     size_output = 1;
//     for (int i = 0; i < len_output; i++)
//     {
//         for (int j = 0; j < LENGTH_TMP_INDICES; j++)
//         {
//             if (strcmp(sample_sd2_1->info_output[i].name, char_indices[j]) == 0)
//             {
//                 sample_sd2_1->info_output[i].size       = int_problem_size[j];
//                 sample_sd2_1->info_output[i].tile_size  = int_problem_size[j];
//                 size_output *= int_problem_size[j];
//             }
//         }
//     }

//     //
//     size_input_left = 1;
//     for (int i = 0; i < len_input_left; i++)
//     {
//         for (int j = 0; j < LENGTH_TMP_INDICES; j++)
//         {
//             if (strcmp(sample_sd2_1->info_input_left[i].name, char_indices[j]) == 0)
//             {
//                 sample_sd2_1->info_input_left[i].size       = int_problem_size[j];
//                 sample_sd2_1->info_input_left[i].tile_size  = int_problem_size[j];
//                 size_input_left *= int_problem_size[j];
//             }
//         }
//     }

//     //
//     size_input_right = 1;
//     for (int i = 0; i < len_input_right; i++)
//     {
//         for (int j = 0; j < LENGTH_TMP_INDICES; j++)
//         {
//             if (strcmp(sample_sd2_1->info_input_right[i].name, char_indices[j]) == 0)
//             {
//                 sample_sd2_1->info_input_right[i].size      = int_problem_size[j];
//                 sample_sd2_1->info_input_right[i].tile_size = int_problem_size[j];
//                 size_input_right *= int_problem_size[j];
//             }
//         }
//     }

//     //
//     for (int i = 0; i < len_internal; i++)
//     {
//         for (int j = 0; j < LENGTH_TMP_INDICES; j++)
//         {
//             if (strcmp(sample_sd2_1->info_internal_indices[i].name, char_indices[j]) == 0)
//             {
//                 sample_sd2_1->info_internal_indices[i].size = int_problem_size[j];
//             }
//         }
//     }

//     //
//     // m*num_operations = (unsigned long long int)20;
//     printf ("[TTGT-TTLG] Problem-Size: ");
//     for (int i = 0; i < len_total_indices; i++)
//     {
//         printf ("%s(%d) ", char_indices[i], int_problem_size[i]);
//         *num_operations *= (unsigned long long int)int_problem_size[i];
//     }
//     printf ("\n");

//     //
//     sample_sd2_1->size_output       = size_output;
//     sample_sd2_1->size_input_left   = size_input_left;
//     sample_sd2_1->size_input_right  = size_input_right;

//     //
//     return sample_sd2_1;   
// }



//
// void print_tensor(const char* tensor_name, idx* info_tensor, int size)
// {
//     printf (">>> %s: ", tensor_name);
//     for (int i = 0; i < size; i++)
//     {
//         printf ("%s(%d)", info_tensor[i].name, info_tensor[i].size);

//         //
//         if (i != size - 1)
//         {
//             printf (", ");
//         } 
//     }
//     printf ("\n");
// }
