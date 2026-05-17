#include "ttgt_ttlg.h"

#include <cstring>
#include <utility>

int find_size(const char* name, idx* target_list, int target_list_size)
{
    for (int i = 0; i < target_list_size; ++i) {
        if (std::strcmp(name, target_list[i].name) == 0) {
            return target_list[i].size;
        }
    }
    return 0;
}

int find_index(const char* name, idx* target_list, int target_list_size)
{
    for (int i = 0; i < target_list_size; ++i) {
        if (std::strcmp(name, target_list[i].name) == 0) {
            return i;
        }
    }
    return -1;
}

int factorial(int target)
{
    if (target <= 1) {
        return 1;
    }

    int result = 1;
    for (int i = 2; i <= target; ++i) {
        result *= i;
    }
    return result;
}

void perm(int* array, int n, int i, int* offset, int* output)
{
    if (i == n) {
        for (int j = 0; j < n; ++j) {
            output[*offset * n + j] = array[j];
        }
        (*offset)++;
        return;
    }

    for (int j = i; j < n; ++j) {
        std::swap(array[i], array[j]);
        perm(array, n, i + 1, offset, output);
        std::swap(array[i], array[j]);
    }
}
