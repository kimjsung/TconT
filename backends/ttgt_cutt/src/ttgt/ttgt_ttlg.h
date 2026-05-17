#pragma once

#include "ttgt_cutt.hpp"

int find_size(const char* name, idx* target_list, int target_list_size);
int find_index(const char* name, idx* target_list, int target_list_size);
int factorial(int target);
void perm(int* array, int n, int i, int* offset, int* output);
