#pragma once
#include <stdio.h>
#include <omp.h>
#include <unordered_map>
#include <cstdint>

bool validate_tccg_1(double* device, double* C, double* h_A,  double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d) {
    // abc-bda-dc
    // t3 [a,16,b,16,c,16] += sum(d,16) * t2 [b,d,a] * v2 [d,c];
	int size_C = size_idx_a * size_idx_b * size_idx_c;

    long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(3) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    {
        int tmp_r_idx = idx_a + (idx_b + (idx_c) * size_idx_b) * size_idx_a;
        for (idx_d = 0; idx_d < size_idx_d; idx_d++)
        {
            C[tmp_r_idx] += 	h_A[idx_b + (idx_d + (idx_a) * size_idx_d) * size_idx_b] * 
                                h_B[idx_d + (idx_c) * size_idx_d];
        }
        ops += size_idx_d;
    }

    printf ("======================================= Correctness Check ==========================================\n");
    double   epsilon = 0.00000001;
    int      diff    = 0;
    int      same    = 0;
    int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
    for (i = 0; i < size_C; i++)
    {
        double check = C[i] - device[i];
        if (check < 0) check *= -1;
        if (check > epsilon)
        {
            diff++;
            // if (diff < 8)
            //     printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, C[i], device[i], check);
        }
        else
        {
            same++;
        }
    }

    printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
    printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
    printf (" >>> Total Operations: %'lld\n", ops * 2);
    printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_2(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d) {
    // abc-dca-bd
    // t3 [a,16,b,16,c,16] += sum(d,16) * t2 [d,c,a] * v2 [b,d];
    int size_C = size_idx_a * size_idx_b * size_idx_c;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(3) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    {
        int tmp_r_idx = idx_a + (idx_b + (idx_c) * size_idx_b) * size_idx_a;
        for (idx_d = 0; idx_d < size_idx_d; idx_d++)
        {
            h_C_chk[tmp_r_idx] += 	h_A[idx_d + (idx_c + (idx_a) * size_idx_c) * size_idx_d] * 
                                    h_B[idx_b + (idx_d) * size_idx_b];
        }
        ops += size_idx_d;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_3(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d, int size_idx_e) {
    // abcd-dbea-ec
    // t3 [a,16,b,16,c,16,d,16] += sum(e,16) * t2 [d,b,e,a] * v2 [e,c];
    int size_C = size_idx_a * size_idx_b * size_idx_c * size_idx_d;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(4) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    for (idx_d = 0; idx_d < size_idx_d; idx_d++)
    {
        int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d) * size_idx_c) * size_idx_b) * size_idx_a;
        for (idx_e = 0; idx_e < size_idx_e; idx_e++)
        {
            h_C_chk[tmp_r_idx] += 	h_A[idx_d + (idx_b + (idx_e + (idx_a) * size_idx_e) * size_idx_b) * size_idx_d] * 
                                    h_B[idx_e + (idx_c) * size_idx_e];
        }
        ops += size_idx_e;    
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_4(double* h_C, double* h_C_chk, double* h_A,  double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d, int size_idx_e) {
    // abcd-deca-be
    // t3 [a,16,b,16,c,16,d,16] += sum(e,16) * t2 [d,e,c,a] * v2 [b,e];
    int size_C = size_idx_a * size_idx_b * size_idx_c * size_idx_d;

    long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(4) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    for (idx_d = 0; idx_d < size_idx_d; idx_d++)
    {
        int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d) * size_idx_c) * size_idx_b) * size_idx_a;
        for (idx_e = 0; idx_e < size_idx_e; idx_e++)
        {
            h_C_chk[tmp_r_idx] += 	h_A[idx_d + (idx_e + (idx_c + (idx_a) * size_idx_c) * size_idx_e) * size_idx_d] * 
                                h_B[idx_b + (idx_e) * size_idx_b];
        }
        ops += size_idx_e;        
    }

    printf ("======================================= Correctness Check ==========================================\n");
    double   epsilon = 0.00000001;
    int      diff    = 0;
    int      same    = 0;
    int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
    for (i = 0; i < size_C; i++)
    {
        double check = h_C_chk[i] - h_C[i];
        if (check < 0) check *= -1;
        if (check > epsilon)
        {
            diff++;
            // if (diff < 8)
            // 	printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
        }
        else
        {
            same++;
        }
    }

    printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
    printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
    printf (" >>> Total Operations: %'lld\n", ops * 2);
    printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_5(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d, int size_idx_e) {
    // abcd-ebad-ce
    // t3 [a,16,b,16,c,16,d,16] += sum(e,16) * t2 [e,b,a,d] * v2 [c,e];
    int size_C = size_idx_a * size_idx_b * size_idx_c * size_idx_d;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(4) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    for (idx_d = 0; idx_d < size_idx_d; idx_d++)
    {
        int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d) * size_idx_c) * size_idx_b) * size_idx_a;
        for (idx_e = 0; idx_e < size_idx_e; idx_e++)
        {
            h_C_chk[tmp_r_idx] += 	h_A[idx_e + (idx_b + (idx_a + (idx_d) * size_idx_a) * size_idx_b) * size_idx_e] * 
                                    h_B[idx_c + (idx_e) * size_idx_c];
        }
        ops += size_idx_e;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_6(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d, int size_idx_e, int size_idx_f) {
    // abcde-efbad-cf
    // t3 [a,16,b,16,c,16,d,16,e,16] += sum(f,16) * t2 [e,f,b,a,d] * v2 [c,f];
    int size_C = size_idx_a * size_idx_b * size_idx_c * size_idx_d * size_idx_e;
	
	long long int ops = 0;;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(5) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    for (idx_d = 0; idx_d < size_idx_d; idx_d++)
    for (idx_e = 0; idx_e < size_idx_e; idx_e++)
    {
        int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d + (idx_e) * size_idx_d) * size_idx_c) * size_idx_b) * size_idx_a;
        for (idx_f = 0; idx_f < size_idx_f; idx_f++)
        {   
            h_C_chk[tmp_r_idx] += 	h_A[idx_e + (idx_f + (idx_b + (idx_a + (idx_d) * size_idx_a) * size_idx_b) * size_idx_f) * size_idx_e] * 
                                    h_B[idx_c + (idx_f) * size_idx_c];
        }
        ops += size_idx_f;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_7(double* h_C, double* h_C_chk, double* h_A,  double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d, int size_idx_e, int size_idx_f) {
    // abcde-ecbfa-fd
	// t3 [a,16,b,16,c,16,d,16,e,16] += sum(f,16) * t2 [e,c,b,f,a] * v2 [f,d];
    int size_C = size_idx_a * size_idx_b * size_idx_c * size_idx_d * size_idx_e;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(5) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    for (idx_d = 0; idx_d < size_idx_d; idx_d++)
    for (idx_e = 0; idx_e < size_idx_e; idx_e++)
    {
        int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d + (idx_e) * size_idx_d) * size_idx_c) * size_idx_b) * size_idx_a;
        for (idx_f = 0; idx_f < size_idx_f; idx_f++)
        {   
            h_C_chk[tmp_r_idx] += 	h_A[idx_e + (idx_c + (idx_b + (idx_f + (idx_a) * size_idx_f) * size_idx_b) * size_idx_c) * size_idx_e] * 
                                    h_B[idx_f + (idx_d) * size_idx_f];
        }
        ops += size_idx_f;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;
	
	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_8(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d, int size_idx_e, int size_idx_f) {
    // abcde-efcad-bf
	// t3 [a,16,b,16,c,16,d,16,e,16] += sum(f,16) * t2 [e,f,c,a,d] * v2 [b,f];
    int size_C = size_idx_a * size_idx_b * size_idx_c * size_idx_d * size_idx_e;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(5) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    for (idx_d = 0; idx_d < size_idx_d; idx_d++)
    for (idx_e = 0; idx_e < size_idx_e; idx_e++)
    {
        int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d + (idx_e) * size_idx_d) * size_idx_c) * size_idx_b) * size_idx_a;
        for (idx_f = 0; idx_f < size_idx_f; idx_f++)
        {   
            h_C_chk[tmp_r_idx] += 	h_A[idx_e + (idx_f + (idx_c + (idx_a + (idx_d) * size_idx_a) * size_idx_c) * size_idx_f) * size_idx_e] * 
                                    h_B[idx_b + (idx_f) * size_idx_b];
        }
        ops += size_idx_f;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_9(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d, int size_idx_e) {
    // abcd-ea-ebcd
	// t3 [a,16,b,16,c,16,d,16] += sum(e,16) * t2 [e,a] * v2 [e,b,c,d];
    int size_C = size_idx_a * size_idx_b * size_idx_c * size_idx_d;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(4) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    for (idx_d = 0; idx_d < size_idx_d; idx_d++)
    {
        int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d) * size_idx_c) * size_idx_b) * size_idx_a;
        for (idx_e = 0; idx_e < size_idx_e; idx_e++)
        {
            h_C_chk[tmp_r_idx] += 	h_A[idx_e + (idx_a) * size_idx_e] * 
                                    h_B[idx_e + (idx_b + (idx_c + (idx_d) * size_idx_c) * size_idx_b) * size_idx_e];
        }
        ops += size_idx_e;        
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_10(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d, int size_idx_e) {
    // abcd-eb-aecd
	// t3 [a,16,b,16,c,16,d,16] += sum(e,16) * t2 [e,b] * v2 [a,e,c,d];
    int size_C = size_idx_a * size_idx_b * size_idx_c * size_idx_d;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(4) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    for (idx_d = 0; idx_d < size_idx_d; idx_d++)
    {
        int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d) * size_idx_c) * size_idx_b) * size_idx_a;
        for (idx_e = 0; idx_e < size_idx_e; idx_e++)
        {
            h_C_chk[tmp_r_idx] += 	h_A[idx_e + (idx_b) * size_idx_e] * 
                                    h_B[idx_a + (idx_e + (idx_c + (idx_d) * size_idx_c) * size_idx_e) * size_idx_a];
        }
        ops += size_idx_e;        
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;
	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_11(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d, int size_idx_e) {
    // abcd-ec-abed
	// t3 [a,16,b,16,c,16,d,16] += sum(e,16) * t2 [e,c] * v2 [a,b,e,d];
    int size_C = size_idx_a * size_idx_b * size_idx_c * size_idx_d;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(4) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    for (idx_d = 0; idx_d < size_idx_d; idx_d++)
    {
        int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d) * size_idx_c) * size_idx_b) * size_idx_a;
        for (idx_e = 0; idx_e < size_idx_e; idx_e++)
        {
            h_C_chk[tmp_r_idx] += 	h_A[idx_e + (idx_c) * size_idx_e] * 
                                    h_B[idx_a + (idx_b + (idx_e + (idx_d) * size_idx_e) * size_idx_b) * size_idx_a];
        }
        ops += size_idx_e;        
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_12(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c) {
    // ab-ac-cb
	// t3 [a,16,b,16] += sum(c,16) * t2 [a,c] * v2 [c,b];
    int size_C = size_idx_a * size_idx_b;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	double t1 = omp_get_wtime();
	#pragma omp parallel for collapse(2) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    {
        int tmp_r_idx = idx_a + (idx_b) * size_idx_a;
        for (idx_c = 0; idx_c < size_idx_c; idx_c++)    
        {
            h_C_chk[tmp_r_idx] += 	h_A[idx_a + (idx_c) * size_idx_a] * 
                                    h_B[idx_c + (idx_b) * size_idx_c];
        }
        ops += size_idx_c;        
    }
	double t2 = omp_get_wtime();
	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}
	double t3 = omp_get_wtime();
	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf("Compute : %.4f seconds\n", t2 - t1);
	printf("Validate: %.4f seconds\n", t3 - t2);
	printf("Total   : %.4f seconds\n", t3 - t1);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_13(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d) {
    // ab-acd-dbc
	// t3 [a,16,b,16] += sum(c,16,d,16) * t2 [a,c,d] * v2 [d,b,c];
    int size_C = size_idx_a * size_idx_b;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(2) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    {
        int tmp_r_idx = idx_a + (idx_b) * size_idx_a;
        for (idx_c = 0; idx_c < size_idx_c; idx_c++)
        for (idx_d = 0; idx_d < size_idx_d; idx_d++)    
        {
            h_C_chk[tmp_r_idx] += 	h_A[idx_a + (idx_c + (idx_d) * size_idx_c) * size_idx_a] * 
                                    h_B[idx_d + (idx_b + (idx_c) * size_idx_b) * size_idx_d];
        }
        ops += size_idx_c * size_idx_d;        
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_14(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d) {
	// ab-cad-dcb
	// t3 [a,16,b,16] += sum(c,16,d,16) * t2 [c,a,d] * v2 [d,c,b];
    int size_C = size_idx_a * size_idx_b;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(2) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    {
        int tmp_r_idx = idx_a + (idx_b) * size_idx_a;
        for (idx_c = 0; idx_c < size_idx_c; idx_c++)
        for (idx_d = 0; idx_d < size_idx_d; idx_d++)    
        {
            h_C_chk[tmp_r_idx] += 	h_A[idx_c + (idx_a + (idx_d) * size_idx_a) * size_idx_c] * 
                                    h_B[idx_d + (idx_c + (idx_b) * size_idx_c) * size_idx_d];
        }
        ops += size_idx_c * size_idx_d;        
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %10.8f, (Dev.) %10.8f >> (Diff.) %10.8f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_15(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d) {
    // abc-acd-db 
	// t3 [a,16,b,16,c,16] += sum(d,16) * t2 [a,c,d] * v2 [d,b];
    int size_C = size_idx_a * size_idx_b * size_idx_c;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(3) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    {
        int tmp_r_idx = idx_a + (idx_b + (idx_c) * size_idx_b) * size_idx_a;
        for (idx_d = 0; idx_d < size_idx_d; idx_d++)
        {
            h_C_chk[tmp_r_idx] += 	h_A[idx_a + (idx_c + (idx_d) * size_idx_c) * size_idx_a] * 
                                    h_B[idx_d + (idx_b) * size_idx_d];
        }
        ops += size_idx_d;        
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_16(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d) {
    // abc-ad-bdc
	// t3 [a,16,b,16,c,16] += sum(d,16) * t2 [a,d] * v2 [b,d,c];
    int size_C = size_idx_a * size_idx_b * size_idx_c;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(3) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    {
        int tmp_r_idx = idx_a + (idx_b + (idx_c) * size_idx_b) * size_idx_a;
        for (idx_d = 0; idx_d < size_idx_d; idx_d++)
        {
            h_C_chk[tmp_r_idx] += 	h_A[idx_a + (idx_d) * size_idx_a] * 
                                    h_B[idx_b + (idx_d + (idx_c) * size_idx_d) * size_idx_b];
        }
        ops += size_idx_d;        
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_17(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d) {
    // abc-adc-bd
	// t3 [a,16,b,16,c,16] += sum(d,16) * t2 [a,d,c] * v2 [b,d];
    int size_C = size_idx_a * size_idx_b * size_idx_c;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d;
    
	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(3) reduction(+:ops)
	for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    {
        int tmp_r_idx = idx_a + (idx_b + (idx_c) * size_idx_b) * size_idx_a;
        for (idx_d = 0; idx_d < size_idx_d; idx_d++)
        {
            h_C_chk[tmp_r_idx] += 	h_A[idx_a + (idx_d + (idx_c) * size_idx_d) * size_idx_a] * 
                                    h_B[idx_b + (idx_d) * size_idx_b];
        }
        ops += size_idx_d;        
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_18(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d) {
    // abc-adc-db
	// t3 [a,16,b,16,c,16] += sum(d,16) * t2 [a,d,c] * v2 [d,b];
    int size_C = size_idx_a * size_idx_b * size_idx_c;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(3) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    {
        int tmp_r_idx = idx_a + (idx_b + (idx_c) * size_idx_b) * size_idx_a;
        for (idx_d = 0; idx_d < size_idx_d; idx_d++)
        {
            h_C_chk[tmp_r_idx] += 	h_A[idx_a + (idx_d + (idx_c) * size_idx_d) * size_idx_a] * 
                                    h_B[idx_d + (idx_b) * size_idx_d];
        }
        ops += size_idx_d;        
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_19(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d, int size_idx_e) {
    // abc-adec-ebd
    // t3 [a,16,b,16,c,16] += sum(e,16,d,16) * t2 [a,d,e,c] * v2 [e,b,d];
    int size_C = size_idx_a * size_idx_b * size_idx_c;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(3) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    {
        int tmp_r_idx = idx_a + (idx_b + (idx_c) * size_idx_b) * size_idx_a;
        for (idx_d = 0; idx_d < size_idx_d; idx_d++)
        {   
            for (idx_e = 0; idx_e < size_idx_e; idx_e++)
            {
                h_C_chk[tmp_r_idx] += 	h_A[idx_a + (idx_d + (idx_e + (idx_c) * size_idx_e) * size_idx_d) * size_idx_a] * 
                                        h_B[idx_e + (idx_b + (idx_d) * size_idx_b) * size_idx_e];
            }
        }
        ops += size_idx_d * size_idx_e;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_20(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d, int size_idx_e, int size_idx_f) {
    // abcd-aebf-dfce
    // t3 [a,16,b,16,c,16,d,16] += sum(e,16,f,16) * t2 [a,e,b,f] * v2 [d,f,c,e];
    int size_C = size_idx_a * size_idx_b * size_idx_c * size_idx_d;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(4) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    for (idx_d = 0; idx_d < size_idx_d; idx_d++)
	{
		int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d) * size_idx_c) * size_idx_b) * size_idx_a;
		for (idx_e = 0; idx_e < size_idx_e; idx_e++)
        {
            for (idx_f = 0; idx_f < size_idx_f; idx_f++)
            {   
                h_C_chk[tmp_r_idx] += 	h_A[idx_a + (idx_e + (idx_b + (idx_f) * size_idx_b) * size_idx_e) * size_idx_a] * 
                                        h_B[idx_d + (idx_f + (idx_c + (idx_e) * size_idx_c) * size_idx_f) * size_idx_d];
            }
        }
		ops += size_idx_e * size_idx_f;
	}

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_21(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d, int size_idx_e, int size_idx_f) {
    // abcd-aebf-fdec
    // t3 [a,16,b,16,c,16,d,16] += sum(e,16,f,16) * t2 [a,e,b,f] * v2 [f,d,e,c];
    int size_C = size_idx_a * size_idx_b * size_idx_c * size_idx_d;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(4) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    for (idx_d = 0; idx_d < size_idx_d; idx_d++)
	{
		int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d) * size_idx_c) * size_idx_b) * size_idx_a;
		for (idx_e = 0; idx_e < size_idx_e; idx_e++)
        {
            for (idx_f = 0; idx_f < size_idx_f; idx_f++)
            {   
                h_C_chk[tmp_r_idx] += 	h_A[idx_a + (idx_e + (idx_b + (idx_f) * size_idx_b) * size_idx_e) * size_idx_a] * 
                                        h_B[idx_f + (idx_d + (idx_e + (idx_c) * size_idx_e) * size_idx_d) * size_idx_f];
            }
        }
		ops += size_idx_f * size_idx_e;
	}

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_22(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d, int size_idx_e, int size_idx_f) {
    // abcd-aecf-bfde
    // t3 [a,16,b,16,c,16,d,16] += sum(e,16,f,16) * t2 [a,e,c,f] * v2 [b,f,d,e];
    int size_C = size_idx_a * size_idx_b * size_idx_c * size_idx_d;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(4) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    for (idx_d = 0; idx_d < size_idx_d; idx_d++)
	{
		int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d) * size_idx_c) * size_idx_b) * size_idx_a;
		for (idx_e = 0; idx_e < size_idx_e; idx_e++)
        {
            for (idx_f = 0; idx_f < size_idx_f; idx_f++)
            {   
                h_C_chk[tmp_r_idx] += 	h_A[idx_a + (idx_e + (idx_c + (idx_f) * size_idx_c) * size_idx_e) * size_idx_a] * 
                                        h_B[idx_b + (idx_f + (idx_d + (idx_e) * size_idx_d) * size_idx_f) * size_idx_b];
            }
        }
		ops += size_idx_e * size_idx_f;
	}

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_23(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d, int size_idx_e, int size_idx_f) {
    // abcd-aecf-fbed
    // t3 [a,16,b,16,c,16,d,16] += sum(e,16,f,16) * t2 [a,e,c,f] * v2 [f,b,e,d];
    int size_C = size_idx_a * size_idx_b * size_idx_c * size_idx_d;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(4) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    for (idx_d = 0; idx_d < size_idx_d; idx_d++)
	{
		int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d) * size_idx_c) * size_idx_b) * size_idx_a;
		for (idx_e = 0; idx_e < size_idx_e; idx_e++)
        {
            for (idx_f = 0; idx_f < size_idx_f; idx_f++)
            {   
                h_C_chk[tmp_r_idx] += 	h_A[idx_a + (idx_e + (idx_c + (idx_f) * size_idx_c) * size_idx_e) * size_idx_a] * 
                                        h_B[idx_f + (idx_b + (idx_e + (idx_d) * size_idx_e) * size_idx_b) * size_idx_f];
            }
        }
		ops += size_idx_e * size_idx_f;
	}

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			if (diff < 8)
			printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_24(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d, int size_idx_e, int size_idx_f) {
    // abcd-aedf-bfce
    // t3 [a,16,b,16,c,16,d,16] += sum(e,16,f,16) * t2 [a,e,d,f] * v2 [b,f,c,e];
    int size_C = size_idx_a * size_idx_b * size_idx_c * size_idx_d;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(4) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    for (idx_d = 0; idx_d < size_idx_d; idx_d++)
	{
		int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d) * size_idx_c) * size_idx_b) * size_idx_a;

		for (idx_e = 0; idx_e < size_idx_e; idx_e++)
        {
            for (idx_f = 0; idx_f < size_idx_f; idx_f++)
            {   
                h_C_chk[tmp_r_idx] += 	h_A[idx_a + (idx_e + (idx_d + (idx_f) * size_idx_d) * size_idx_e) * size_idx_a] * 
                                        h_B[idx_b + (idx_f + (idx_c + (idx_e) * size_idx_c) * size_idx_f) * size_idx_b];
            }
        }
		ops += size_idx_e * size_idx_f;
	}

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_25(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d, int size_idx_e, int size_idx_f) {
    // abcd-aedf-fbec
    // t3 [a,16,b,16,c,16,d,16] += sum(e,16,f,16) * t2 [a,e,d,f] * v2 [f,b,e,c];
    int size_C = size_idx_a * size_idx_b * size_idx_c * size_idx_d;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(4) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    for (idx_d = 0; idx_d < size_idx_d; idx_d++)
	{
		int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d) * size_idx_c) * size_idx_b) * size_idx_a;

		for (idx_e = 0; idx_e < size_idx_e; idx_e++)
        {
            for (idx_f = 0; idx_f < size_idx_f; idx_f++)
            {   
                h_C_chk[tmp_r_idx] += 	h_A[idx_a + (idx_e + (idx_d + (idx_f) * size_idx_d) * size_idx_e) * size_idx_a] * 
                                        h_B[idx_f + (idx_b + (idx_e + (idx_c) * size_idx_e) * size_idx_b) * size_idx_f];
            }
        }
		ops += size_idx_e * size_idx_f;
	}

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_26(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d, int size_idx_e, int size_idx_f) {
    // abcd-aefb-fdce
    // t3 [a,16,b,16,c,16,d,16] += sum(e,16,f,16) * t2 [a,e,f,b] * v2 [f,d,c,e];
    int size_C = size_idx_a * size_idx_b * size_idx_c * size_idx_d;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(4) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    for (idx_d = 0; idx_d < size_idx_d; idx_d++)
	{
		int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d) * size_idx_c) * size_idx_b) * size_idx_a;

		for (idx_e = 0; idx_e < size_idx_e; idx_e++)
        {
            for (idx_f = 0; idx_f < size_idx_f; idx_f++)
            {   
                h_C_chk[tmp_r_idx] += 	h_A[idx_a + (idx_e + (idx_f + (idx_b) * size_idx_f) * size_idx_e) * size_idx_a] * 
                                        h_B[idx_f + (idx_d + (idx_c + (idx_e) * size_idx_c) * size_idx_d) * size_idx_f];
            }
        }
		ops += size_idx_e * size_idx_f;
	}

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_27(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d, int size_idx_e, int size_idx_f) {
    // abcd-aefc-fbed
    // t3 [a,16,b,16,c,16,d,16] += sum(e,16,f,16) * t2 [a,e,f,c] * v2 [f,b,e,d];
    int size_C = size_idx_a * size_idx_b * size_idx_c * size_idx_d;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(4) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    for (idx_d = 0; idx_d < size_idx_d; idx_d++)
	{
		int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d) * size_idx_c) * size_idx_b) * size_idx_a;

		for (idx_e = 0; idx_e < size_idx_e; idx_e++)
        {
            for (idx_f = 0; idx_f < size_idx_f; idx_f++)
            {   
                h_C_chk[tmp_r_idx] += 	h_A[idx_a + (idx_e + (idx_f + (idx_c) * size_idx_f) * size_idx_e) * size_idx_a] * 
                                        h_B[idx_f + (idx_b + (idx_e + (idx_d) * size_idx_e) * size_idx_b) * size_idx_f];
            }
        }
		ops += size_idx_e * size_idx_f;
	}

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_28(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d, int size_idx_e, int size_idx_f) {
    // abcd-eafb-fdec
    // t3 [a,16,b,16,c,16,d,16] += sum(e,16,f,16) * t2 [e,a,f,b] * v2 [f,d,e,c];
    int size_C = size_idx_a * size_idx_b * size_idx_c * size_idx_d;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(4) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    for (idx_d = 0; idx_d < size_idx_d; idx_d++)
	{
		int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d) * size_idx_c) * size_idx_b) * size_idx_a;

		for (idx_e = 0; idx_e < size_idx_e; idx_e++)
        {
            for (idx_f = 0; idx_f < size_idx_f; idx_f++)
            {   
                h_C_chk[tmp_r_idx] += 	h_A[idx_e + (idx_a + (idx_f + (idx_b) * size_idx_f) * size_idx_a) * size_idx_e] * 
                                        h_B[idx_f + (idx_d + (idx_e + (idx_c) * size_idx_e) * size_idx_d) * size_idx_f];
            }
        }
		ops += size_idx_e * size_idx_f;
	}

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_29(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d, int size_idx_e, int size_idx_f) {
    // abcd-eafc-bfde
    // t3 [a,16,b,16,c,16,d,16] += sum(e,16,f,16) * t2 [e,a,f,c] * v2 [b,f,d,e];
    int size_C = size_idx_a * size_idx_b * size_idx_c * size_idx_d;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(4) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    for (idx_d = 0; idx_d < size_idx_d; idx_d++)
	{
		int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d) * size_idx_c) * size_idx_b) * size_idx_a;

		for (idx_e = 0; idx_e < size_idx_e; idx_e++)
        {
            for (idx_f = 0; idx_f < size_idx_f; idx_f++)
            {   
                h_C_chk[tmp_r_idx] += 	h_A[idx_e + (idx_a + (idx_f + (idx_c) * size_idx_f) * size_idx_a) * size_idx_e] * 
                                        h_B[idx_b + (idx_f + (idx_d + (idx_e) * size_idx_d) * size_idx_f) * size_idx_b];
            }
        }
	}

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_30(double* h_C, double* h_C_chk, double* h_A, double* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d, int size_idx_e, int size_idx_f) {
    // abcd-eafd-fbec
    // t3 [a,16,b,16,c,16,d,16] += sum(e,16,f,16) * t2 [e,a,f,d] * v2 [f,b,e,c];
    int size_C = size_idx_a * size_idx_b * size_idx_c * size_idx_d;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(4) reduction(+:ops)
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    for (idx_d = 0; idx_d < size_idx_d; idx_d++)
	{
		int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d) * size_idx_c) * size_idx_b) * size_idx_a;

		for (idx_e = 0; idx_e < size_idx_e; idx_e++)
		{
            for (idx_f = 0; idx_f < size_idx_f; idx_f++)
            {   
                h_C_chk[tmp_r_idx] += 	h_A[idx_e + (idx_a + (idx_f + (idx_d) * size_idx_f) * size_idx_a) * size_idx_e] * 
                                        h_B[idx_f + (idx_b + (idx_e + (idx_c) * size_idx_e) * size_idx_b) * size_idx_f];
            }
        }
		ops += size_idx_e * size_idx_f;
	}

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_31(double* device, double* C, double* h_A, double* h_B, int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g) {
    int size_C = size_a * size_b * size_c * size_d * size_e * size_f;

	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(6) reduction(+:ops)
    for (idx_a = 0; idx_a < size_a; idx_a++)
    for (idx_b = 0; idx_b < size_b; idx_b++)
    for (idx_c = 0; idx_c < size_c; idx_c++)
    for (idx_d = 0; idx_d < size_d; idx_d++)
    for (idx_e = 0; idx_e < size_e; idx_e++)
    for (idx_f = 0; idx_f < size_f; idx_f++)
    {   
        for (idx_g = 0; idx_g < size_g; idx_g++)
        {
            int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;
            C[tmp_r_idx] += 	h_A[idx_d + (idx_e + (idx_g + (idx_a) * size_g) * size_e) * size_d] * 
                                    h_B[idx_g + (idx_f + (idx_b + (idx_c) * size_b) * size_f) * size_g];
        }
		ops += size_g;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = C[i] - device[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, C[i], device[i], check);
		}
		else
		{
            same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_32(double* device, double* C, double* h_A, double* h_B, int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g) {
    int size_C = size_a * size_b * size_c * size_d * size_e * size_f;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(6) reduction(+:ops)
    for (idx_a = 0; idx_a < size_a; idx_a++)
    for (idx_b = 0; idx_b < size_b; idx_b++)
    for (idx_c = 0; idx_c < size_c; idx_c++)
    for (idx_d = 0; idx_d < size_d; idx_d++)
    for (idx_e = 0; idx_e < size_e; idx_e++)
    for (idx_f = 0; idx_f < size_f; idx_f++)
    {   
        for (idx_g = 0; idx_g < size_g; idx_g++)
        {
            int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;
            C[tmp_r_idx] += 	h_A[idx_d + (idx_e + (idx_g + (idx_b) * size_g) * size_e) * size_d] * 
                                    h_B[idx_g + (idx_f + (idx_a + (idx_c) * size_a) * size_f) * size_g];
        }
        ops += size_g;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = C[i] - device[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, C[i], device[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_33(double* device, double* C, double* h_A, double* h_B, int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g) {
    int size_C = size_a * size_b * size_c * size_d * size_e * size_f;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);
	#pragma omp parallel for collapse(6) reduction(+:ops)
    for (idx_a = 0; idx_a < size_a; idx_a++)
    for (idx_b = 0; idx_b < size_b; idx_b++)
    for (idx_c = 0; idx_c < size_c; idx_c++)
    for (idx_d = 0; idx_d < size_d; idx_d++)
    for (idx_e = 0; idx_e < size_e; idx_e++)
    for (idx_f = 0; idx_f < size_f; idx_f++)
    {   
        for (idx_g = 0; idx_g < size_g; idx_g++)
        {
            int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;
            C[tmp_r_idx] += 	h_A[idx_d + (idx_e + (idx_g + (idx_c) * size_g) * size_e) * size_d] * 
                                    h_B[idx_g + (idx_f + (idx_a + (idx_b) * size_a) * size_f) * size_g];
        }
        ops += size_g;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = C[i] - device[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, C[i], device[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_34(double* device, double* C, double* h_A, double* h_B, int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g) {
    int size_C = size_a * size_b * size_c * size_d * size_e * size_f;

    long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(6) reduction(+:ops)
    for (idx_a = 0; idx_a < size_a; idx_a++)
    for (idx_b = 0; idx_b < size_b; idx_b++)
    for (idx_c = 0; idx_c < size_c; idx_c++)
    for (idx_d = 0; idx_d < size_d; idx_d++)
    for (idx_e = 0; idx_e < size_e; idx_e++)
    for (idx_f = 0; idx_f < size_f; idx_f++)
    {   
        for (idx_g = 0; idx_g < size_g; idx_g++)
        {
            int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;
            C[tmp_r_idx] += 	h_A[idx_d + (idx_f + (idx_g + (idx_a) * size_g) * size_f) * size_d] * 
                                    h_B[idx_g + (idx_e + (idx_b + (idx_c) * size_b) * size_e) * size_g];
        }
        ops += size_g;
    }

    printf ("======================================= Correctness Check ==========================================\n");
    double   epsilon = 0.00000001;
    int      diff    = 0;
    int      same    = 0;
    int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
    for (i = 0; i < size_C; i++)
    {
        double check = C[i] - device[i];
        if (check < 0) check *= -1;
        if (check > epsilon)
        {
            diff++;
            // if (diff < 8)
            // printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, C[i], device[i], check);
        }
        else
        {
            same++;
        }
    }

    printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
    printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
    printf (" >>> Total Operations: %'lld\n", ops * 2);
    printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_35(double* device, double* C, double* h_A, double* h_B, int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g) {
    int size_C = size_a * size_b * size_c * size_d * size_e * size_f;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);
	
	#pragma omp parallel for collapse(6) reduction(+:ops)
    for (idx_a = 0; idx_a < size_a; idx_a++)
    for (idx_b = 0; idx_b < size_b; idx_b++)
    for (idx_c = 0; idx_c < size_c; idx_c++)
    for (idx_d = 0; idx_d < size_d; idx_d++)
    for (idx_e = 0; idx_e < size_e; idx_e++)
    for (idx_f = 0; idx_f < size_f; idx_f++)
    {   
        for (idx_g = 0; idx_g < size_g; idx_g++)
        {
            int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;
            C[tmp_r_idx] += 	h_A[idx_d + (idx_f + (idx_g + (idx_b) * size_g) * size_f) * size_d] * 
                                    h_B[idx_g + (idx_e + (idx_a + (idx_c) * size_a) * size_e) * size_g];
        }
        ops += size_g;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = C[i] - device[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, C[i], device[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_36(double* device, double* C, double* h_A, double* h_B, int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g) {
    int size_C = size_a * size_b * size_c * size_d * size_e * size_f;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(6) reduction(+:ops)
    for (idx_a = 0; idx_a < size_a; idx_a++)
    for (idx_b = 0; idx_b < size_b; idx_b++)
    for (idx_c = 0; idx_c < size_c; idx_c++)
    for (idx_d = 0; idx_d < size_d; idx_d++)
    for (idx_e = 0; idx_e < size_e; idx_e++)
    for (idx_f = 0; idx_f < size_f; idx_f++)
    {   
        for (idx_g = 0; idx_g < size_g; idx_g++)
        {
            int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;
            C[tmp_r_idx] += 	h_A[idx_d + (idx_f + (idx_g + (idx_c) * size_g) * size_f) * size_d] * 
                                    h_B[idx_g + (idx_e + (idx_a + (idx_b) * size_a) * size_e) * size_g];
        }
        ops += size_g;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = C[i] - device[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, C[i], device[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_37(double* device, double* C, double* h_A, double* h_B, int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g) {
    int size_C = size_a * size_b * size_c * size_d * size_e * size_f;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(6) reduction(+:ops)
    for (idx_a = 0; idx_a < size_a; idx_a++)
    for (idx_b = 0; idx_b < size_b; idx_b++)
    for (idx_c = 0; idx_c < size_c; idx_c++)
    for (idx_d = 0; idx_d < size_d; idx_d++)
    for (idx_e = 0; idx_e < size_e; idx_e++)
    for (idx_f = 0; idx_f < size_f; idx_f++)
    {   
        for (idx_g = 0; idx_g < size_g; idx_g++)
        {
            int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;
            C[tmp_r_idx] += 	h_A[idx_e + (idx_f + (idx_g + (idx_a) * size_g) * size_f) * size_e] * 
                                    h_B[idx_g + (idx_d + (idx_b + (idx_c) * size_b) * size_d) * size_g];
        }
        ops += size_g;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = C[i] - device[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, C[i], device[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_38(double* device, double* C, double* h_A, double* h_B, int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g) {
    int size_C = size_a * size_b * size_c * size_d * size_e * size_f;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(6) reduction(+:ops)
    for (idx_a = 0; idx_a < size_a; idx_a++)
    for (idx_b = 0; idx_b < size_b; idx_b++)
    for (idx_c = 0; idx_c < size_c; idx_c++)
    for (idx_d = 0; idx_d < size_d; idx_d++)
    for (idx_e = 0; idx_e < size_e; idx_e++)
    for (idx_f = 0; idx_f < size_f; idx_f++)
    {   
        for (idx_g = 0; idx_g < size_g; idx_g++)
        {
            int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;
            C[tmp_r_idx] += 	h_A[idx_e + (idx_f + (idx_g + (idx_b) * size_g) * size_f) * size_e] * 
                                    h_B[idx_g + (idx_d + (idx_a + (idx_c) * size_a) * size_d) * size_g];
        }
        ops += size_g;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = C[i] - device[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, C[i], device[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_39(double* device, double* C, double* h_A, double* h_B, int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g) {
    int size_C = size_a * size_b * size_c * size_d * size_e * size_f;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);
	#pragma omp parallel for collapse(6) reduction(+:ops)
    for (idx_a = 0; idx_a < size_a; idx_a++)
    for (idx_b = 0; idx_b < size_b; idx_b++)
    for (idx_c = 0; idx_c < size_c; idx_c++)
    for (idx_d = 0; idx_d < size_d; idx_d++)
    for (idx_e = 0; idx_e < size_e; idx_e++)
    for (idx_f = 0; idx_f < size_f; idx_f++)
    {   
        for (idx_g = 0; idx_g < size_g; idx_g++)
        {
            int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;
            C[tmp_r_idx] += 	h_A[idx_e + (idx_f + (idx_g + (idx_c) * size_g) * size_f) * size_e] * 
                                    h_B[idx_g + (idx_d + (idx_a + (idx_b) * size_a) * size_d) * size_g];
        }
        ops += size_g;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = C[i] - device[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, C[i], device[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_40(double* device, double* C, double* h_A, double* h_B, int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g) {
    int size_C = size_a * size_b * size_c * size_d * size_e * size_f;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(6) reduction(+:ops)
    for (idx_a = 0; idx_a < size_a; idx_a++)
    for (idx_b = 0; idx_b < size_b; idx_b++)
    for (idx_c = 0; idx_c < size_c; idx_c++)
    for (idx_d = 0; idx_d < size_d; idx_d++)
    for (idx_e = 0; idx_e < size_e; idx_e++)
    for (idx_f = 0; idx_f < size_f; idx_f++)
    {   
        for (idx_g = 0; idx_g < size_g; idx_g++)
        {
            int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;
            C[tmp_r_idx] += 	h_A[idx_g + (idx_d + (idx_a + (idx_b) * size_a) * size_d) * size_g] * 
                                    h_B[idx_e + (idx_f + (idx_g + (idx_c) * size_g) * size_f) * size_e];
        }
        ops += size_g;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
        double check = C[i] - device[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, C[i], device[i], check);
		}
		else
		{
			same++;
		}
	}
	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_41(double* device, double* C, double* h_A, double* h_B, int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g) {
    int size_C = size_a * size_b * size_c * size_d * size_e * size_f;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(6) reduction(+:ops)
    for (idx_a = 0; idx_a < size_a; idx_a++)
    for (idx_b = 0; idx_b < size_b; idx_b++)
    for (idx_c = 0; idx_c < size_c; idx_c++)
    for (idx_d = 0; idx_d < size_d; idx_d++)
    for (idx_e = 0; idx_e < size_e; idx_e++)
    for (idx_f = 0; idx_f < size_f; idx_f++)
    {   
        for (idx_g = 0; idx_g < size_g; idx_g++)
        {
            int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;
            C[tmp_r_idx] += 	h_A[idx_g + (idx_d + (idx_a + (idx_c) * size_a) * size_d) * size_g] * 
                                    h_B[idx_e + (idx_f + (idx_g + (idx_b) * size_g) * size_f) * size_e];
        }
        ops += size_g;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = C[i] - device[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, C[i], device[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_42(double* device, double* C, double* h_A, double* h_B, int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g) {
    int size_C = size_a * size_b * size_c * size_d * size_e * size_f;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(6) reduction(+:ops)
    for (idx_a = 0; idx_a < size_a; idx_a++)
    for (idx_b = 0; idx_b < size_b; idx_b++)
    for (idx_c = 0; idx_c < size_c; idx_c++)
    for (idx_d = 0; idx_d < size_d; idx_d++)
    for (idx_e = 0; idx_e < size_e; idx_e++)
    for (idx_f = 0; idx_f < size_f; idx_f++)
    {   
        for (idx_g = 0; idx_g < size_g; idx_g++)
        {
            int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;
            C[tmp_r_idx] += 	h_A[idx_g + (idx_d + (idx_b + (idx_c) * size_b) * size_d) * size_g] * 
                                    h_B[idx_e + (idx_f + (idx_g + (idx_a) * size_g) * size_f) * size_e];
        }
        ops += size_g;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = C[i] - device[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, C[i], device[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_43(double* device, double* C, double* h_A, double* h_B, int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g) {
    int size_C = size_a * size_b * size_c * size_d * size_e * size_f;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(6) reduction(+:ops)
    for (idx_a = 0; idx_a < size_a; idx_a++)
    for (idx_b = 0; idx_b < size_b; idx_b++)
    for (idx_c = 0; idx_c < size_c; idx_c++)
    for (idx_d = 0; idx_d < size_d; idx_d++)
    for (idx_e = 0; idx_e < size_e; idx_e++)
    for (idx_f = 0; idx_f < size_f; idx_f++)
    {   
        for (idx_g = 0; idx_g < size_g; idx_g++)
        {
            int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;
            C[tmp_r_idx] += 	h_A[idx_g + (idx_e + (idx_a + (idx_b) * size_a) * size_e) * size_g] * 
                                    h_B[idx_d + (idx_f + (idx_g + (idx_c) * size_g) * size_f) * size_d];
        }
        ops += size_g;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = C[i] - device[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, C[i], device[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_44(double* device, double* C, double* h_A, double* h_B, int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g) {
    int size_C = size_a * size_b * size_c * size_d * size_e * size_f;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(6) reduction(+:ops)
    for (idx_a = 0; idx_a < size_a; idx_a++)
    for (idx_b = 0; idx_b < size_b; idx_b++)
    for (idx_c = 0; idx_c < size_c; idx_c++)
    for (idx_d = 0; idx_d < size_d; idx_d++)
    for (idx_e = 0; idx_e < size_e; idx_e++)
    for (idx_f = 0; idx_f < size_f; idx_f++)
    {   
        for (idx_g = 0; idx_g < size_g; idx_g++)
        {
            int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;
            C[tmp_r_idx] += 	h_A[idx_g + (idx_e + (idx_a + (idx_c) * size_a) * size_e) * size_g] * 
                                    h_B[idx_d + (idx_f + (idx_g + (idx_b) * size_g) * size_f) * size_d];
        }
        ops += size_g;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = C[i] - device[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, C[i], device[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_45(double* device, double* C, double* h_A, double* h_B, int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g) {
    int size_C = size_a * size_b * size_c * size_d * size_e * size_f;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(6) reduction(+:ops)
    for (idx_a = 0; idx_a < size_a; idx_a++)
    for (idx_b = 0; idx_b < size_b; idx_b++)
    for (idx_c = 0; idx_c < size_c; idx_c++)
    for (idx_d = 0; idx_d < size_d; idx_d++)
    for (idx_e = 0; idx_e < size_e; idx_e++)
    for (idx_f = 0; idx_f < size_f; idx_f++)
    {
        for (idx_g = 0; idx_g < size_g; idx_g++)
        {
            int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;
            C[tmp_r_idx] += 	h_A[idx_g + (idx_e + (idx_b + (idx_c) * size_b) * size_e) * size_g] * 
                                    h_B[idx_d + (idx_f + (idx_g + (idx_a) * size_g) * size_f) * size_d];
        }
        ops += size_g;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = C[i] - device[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, C[i], device[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_46(double* device, double* C, double* h_A, double* h_B, int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g) {
    int size_C = size_a * size_b * size_c * size_d * size_e * size_f;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(6) reduction(+:ops)
    for (idx_a = 0; idx_a < size_a; idx_a++)
    for (idx_b = 0; idx_b < size_b; idx_b++)
    for (idx_c = 0; idx_c < size_c; idx_c++)
    for (idx_d = 0; idx_d < size_d; idx_d++)
    for (idx_e = 0; idx_e < size_e; idx_e++)
    for (idx_f = 0; idx_f < size_f; idx_f++)
    {   
        for (idx_g = 0; idx_g < size_g; idx_g++)
        {
            int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;
            C[tmp_r_idx] += 	h_A[idx_g + (idx_f + (idx_a + (idx_b) * size_a) * size_f) * size_g] * 
                                    h_B[idx_d + (idx_e + (idx_g + (idx_c) * size_g) * size_e) * size_d];
            
        }
        ops += size_g;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = C[i] - device[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, C[i], device[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_47(double* device, double* C, double* h_A, double* h_B, int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g) {
    int size_C = size_a * size_b * size_c * size_d * size_e * size_f;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(6) reduction(+:ops)
    for (idx_a = 0; idx_a < size_a; idx_a++)
    for (idx_b = 0; idx_b < size_b; idx_b++)
    for (idx_c = 0; idx_c < size_c; idx_c++)
    for (idx_d = 0; idx_d < size_d; idx_d++)
    for (idx_e = 0; idx_e < size_e; idx_e++)
    for (idx_f = 0; idx_f < size_f; idx_f++)
    {   
        for (idx_g = 0; idx_g < size_g; idx_g++)
        {
            int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;
            C[tmp_r_idx] += 	h_A[idx_g + (idx_f + (idx_a + (idx_c) * size_a) * size_f) * size_g] * 
                                    h_B[idx_d + (idx_e + (idx_g + (idx_b) * size_g) * size_e) * size_d];
        }
        ops += size_g;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = C[i] - device[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, C[i], device[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool validate_tccg_48(double* device, double* C, double* h_A, double* h_B, int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g) {
    int size_C = size_a * size_b * size_c * size_d * size_e * size_f;
	
	long long int ops = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g;

	int nprocs = omp_get_num_procs();  
    omp_set_num_threads(nprocs);

	#pragma omp parallel for collapse(6) reduction(+:ops)
    for (idx_a = 0; idx_a < size_a; idx_a++)
    for (idx_b = 0; idx_b < size_b; idx_b++)
    for (idx_c = 0; idx_c < size_c; idx_c++)
    for (idx_d = 0; idx_d < size_d; idx_d++)
    for (idx_e = 0; idx_e < size_e; idx_e++)
    for (idx_f = 0; idx_f < size_f; idx_f++)
    {   
        for (idx_g = 0; idx_g < size_g; idx_g++)
        {
            int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;
            C[tmp_r_idx] += 	h_A[idx_g + (idx_f + (idx_b + (idx_c) * size_b) * size_f) * size_g] * 
                                    h_B[idx_d + (idx_e + (idx_g + (idx_a) * size_g) * size_e) * size_d];
        }
        ops += size_g;
    }

	printf ("======================================= Correctness Check ==========================================\n");
	double   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;

	#pragma omp parallel for reduction(+:diff, same)
	for (i = 0; i < size_C; i++)
	{
		double check = C[i] - device[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			// if (diff < 8)
			// printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, C[i], device[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", ops * 2);
	printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}

bool post_Correctness(double* device, double* C, double* h_A, double* h_B, std::unordered_map<int, int64_t> extent, int equation_num) {
	bool flag;
	switch(equation_num) {
		case 1 :
			flag = validate_tccg_1(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d']);
			break;
		case 2 :
			flag = validate_tccg_2(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d']);
			break;
		case 3 :
			flag = validate_tccg_3(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e']);
			break;
		case 4 :
			flag = validate_tccg_4(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e']);
			break;
		case 5 :
			flag = validate_tccg_5(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e']);
			break;
		case 6 :
			flag = validate_tccg_6(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f']);
			break;
		case 7 :
			flag = validate_tccg_7(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f']);
			break;
		case 8 :
			flag = validate_tccg_8(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f']);
			break;
		case 9 :
			flag = validate_tccg_9(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e']);
			break;
		case 10 :
			flag = validate_tccg_10(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e']);
			break;
		case 11 :
			flag = validate_tccg_11(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e']);
			break;
		case 12 :
			flag = validate_tccg_12(device, C, h_A, h_B, extent['a'], extent['b'], extent['c']);
			break;
		case 13 :
			flag = validate_tccg_13(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d']);
			break;
		case 14 :
			flag = validate_tccg_14(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d']);
			break;
		case 15 :
			flag = validate_tccg_15(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d']);
			break;
		case 16 :
			flag = validate_tccg_16(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d']);
			break;
		case 17 :
			flag = validate_tccg_17(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d']);
			break;
		case 18 :
			flag = validate_tccg_18(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d']);
			break;
		case 19 :
			flag = validate_tccg_19(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e']);
			break;
		case 20 :
			flag = validate_tccg_20(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f']);
			break;
		case 21 :
			flag = validate_tccg_21(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f']);
			break;
		case 22 :
			flag = validate_tccg_22(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f']);
			break;
		case 23 :
			flag = validate_tccg_23(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f']);
			break;
		case 24 :
			flag = validate_tccg_24(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f']);
			break;
		case 25 :
			flag = validate_tccg_25(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f']);
			break;
		case 26 :
			flag = validate_tccg_26(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f']);
			break;
		case 27 :
			flag = validate_tccg_27(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f']);
			break;
		case 28 :
			flag = validate_tccg_28(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f']);
			break;
		case 29 :
			flag = validate_tccg_29(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f']);
			break;
		case 30 :
			flag = validate_tccg_30(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f']);
			break;
		case 31 :
			flag = validate_tccg_31(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f'], extent['g']);
			break;
		case 32 :
			flag = validate_tccg_32(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f'], extent['g']);
			break;
		case 33 :
			flag = validate_tccg_33(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f'], extent['g']);
			break;
		case 34 :
			flag = validate_tccg_34(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f'], extent['g']);
			break;
		case 35 :
			flag = validate_tccg_35(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f'], extent['g']);
			break;
		case 36 :
			flag = validate_tccg_36(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f'], extent['g']);
			break;
		case 37 :
			flag = validate_tccg_37(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f'], extent['g']);
			break;
		case 38 :
			flag = validate_tccg_38(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f'], extent['g']);
			break;
		case 39 :
			flag = validate_tccg_39(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f'], extent['g']);
			break;
		case 40 :
			flag = validate_tccg_40(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f'], extent['g']);
			break;
		case 41 :
			flag = validate_tccg_41(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f'], extent['g']);
			break;
		case 42 :
			flag = validate_tccg_42(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f'], extent['g']);
			break;
		case 43 :
			flag = validate_tccg_43(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f'], extent['g']);
			break;
		case 44 :
			flag = validate_tccg_44(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f'], extent['g']);
			break;
		case 45 :
			flag = validate_tccg_45(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f'], extent['g']);
			break;
		case 46 :
			flag = validate_tccg_46(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f'], extent['g']);
			break;
		case 47 :
			flag = validate_tccg_47(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f'], extent['g']);
			break;
		case 48 :
			flag = validate_tccg_48(device, C, h_A, h_B, extent['a'], extent['b'], extent['c'], extent['d'], extent['e'], extent['f'], extent['g']);
			break;
	}

	return flag;
}