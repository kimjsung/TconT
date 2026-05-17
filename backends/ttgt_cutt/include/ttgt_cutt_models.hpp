#pragma once

#include <stdio.h>

unsigned int ttgt_cuTT_model_overall_analytic(
  // const new_config* list_configs, 
  new_config* list_configs, 
  int num_configs, 
  float min_ms_trans_A, 
  float min_ms_trans_B, 
  float min_ms_trans_C, 
  unsigned int target_config);

unsigned int ttgt_cuTT_model_overall_ML(
  // const new_config* list_configs, 
  new_config* list_configs, 
  int num_configs);

// Helper function to map Compute Capability to FP64 Cores per SM
int getFP64CoresPerSM(int major, int minor);

// Helper function to map Compute Capability to FP32 Cores per SM
int getFP32CoresPerSM(int major, int minor);
