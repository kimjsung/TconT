#include "ttgt_cutt.hpp"
#include "ttgt_cutt_models.hpp"

/**
 * @brief 
 * 
 */
unsigned int ttgt_cuTT_model_overall_analytic(
  // const new_config* list_configs, 
  new_config* list_configs, 
  int num_configs, 
  float min_ms_trans_A, 
  float min_ms_trans_B, 
  float min_ms_trans_C, 
  unsigned int target_config)
{
  // printf ("[%s] Analytical Performance Model\n", __func__);
  float min_total_time = std::numeric_limits<float>::max();
  for (int i = 0; i < num_configs; i++) {    
    float total_trans_time = 0.0f;
    if (list_configs[i].trans_input_left == true)   { total_trans_time += min_ms_trans_A; } 
    if (list_configs[i].trans_input_right == true)  { total_trans_time += min_ms_trans_B; } 
    if (list_configs[i].trans_output == true)       { total_trans_time += min_ms_trans_C; } 

    if (total_trans_time < min_total_time) {
      min_total_time = total_trans_time;
    }
  }

  // printf ("[%s] min_total_time = %f ms\n", __func__, min_total_time);
  unsigned int num_selected_configs = 0;
  for (int i = 0; i < num_configs; i++) {    
    float total_trans_time = 0.0f;
    if (list_configs[i].trans_input_left == true)   { total_trans_time += min_ms_trans_A; } 
    if (list_configs[i].trans_input_right == true)  { total_trans_time += min_ms_trans_B; } 
    if (list_configs[i].trans_output == true)       { total_trans_time += min_ms_trans_C; } 

    if (total_trans_time <= min_total_time) {
      list_configs[i].useless = false;
      num_selected_configs++;
      // printf ("[%s] config %d is selected (%f <= %f)\n", __func__, i, total_trans_time, min_total_time);
    }
  }

  return num_selected_configs;
}

// Helper function to map Compute Capability to FP64 Cores per SM
int getFP64CoresPerSM(int major, int minor) {
    if (major == 9 && minor == 0) return 64;  // Hopper H100
    if (major == 8 && minor == 9) return 2;   // Ada Lovelace
    if (major == 8 && minor == 6) return 2;   // Ampere Consumer
    if (major == 8 && minor == 0) return 32;  // Ampere A100
    if (major == 7 && minor == 5) return 2;   // Turing
    if (major == 7 && minor == 0) return 32;  // Volta
    return 0; // Unsupported or older architectures
}

// Helper function to map Compute Capability to FP32 Cores per SM
int getFP32CoresPerSM(int major, int minor) {
    if (major == 9 && minor == 0) return 128;
    if (major == 8 && minor == 9) return 128;
    if (major == 8 && minor == 6) return 128;
    if (major == 8 && minor == 0) return 64;
    if (major == 7 && minor == 5) return 64;
    if (major == 7 && minor == 0) return 64;
    return 0;
}