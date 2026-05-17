#include "../ttgt_ttlg.h"

//
//  Model for TTLG
//
//  opt: (1:TYPE_TENSOR_A) A, (2:TYPE_TENSOR_B) B, (0:TYPE_TENSOR_C) C
int model_ttgt_ttlg(info_config a_config, int opt)
{
    if (opt == TYPE_TENSOR_A)
    {
        printf ("[%s] A\n", __func__);
        //for (int i = 0; )
    }

    if (opt == TYPE_TENSOR_B)
    {
        printf ("[%s] B\n", __func__);
    }

    if (opt == TYPE_TENSOR_C)
    {
        printf ("[%s] C\n", __func__);
    }

    return 0;
}
