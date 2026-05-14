#include <mma.h>
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda/pipeline>

#define TILE_E 		8
#define TILE_D 		1
#define TILE_A 		8
#define TILE_C 		16
#define TILE_B1 		16
#define TILE_B2 		2

#define TILE_UNIT 	(TILE_D * TILE_E)

#define TILE_B 		(TILE_B1 * TILE_B2)

#define plane_v2 			(TILE_B1 * TILE_E)
#define v2_elements 		(plane_v2 * TILE_B2)

#define plane_t2 			(TILE_E * TILE_A)
#define t2_elements 		(plane_t2 * TILE_C)

#define PIPELINE_STAGES 	3

#define CEIL(a, b) 			(((a) + (b) - 1) / (b))

__device__ void load_from_GM_1(double *__restrict dev_v2, double *__restrict dev_t2, double *sm_v2, double *sm_t2,
int size_a, int size_b, int size_c, int size_d, int size_e, 
int blk_idx_a, int blk_idx_b, int blk_idx_c, 
cuda::pipeline<cuda::thread_scope_block> &pipeline, cooperative_groups::thread_block_tile<1U, void> &thread,
int shm_v2_offset, int shm_t2_offset, int iter_e, int iter_d)
{
	for(int idx = (threadIdx.x << 1); idx < v2_elements; idx += (blockDim.x << 1))
	{
		int v2_b2 = idx / plane_v2;
		int rem_v2 = idx - v2_b2 * plane_v2;
		int v2_b1 = rem_v2 / TILE_E;
		int v2_e = rem_v2 - v2_b1 * TILE_E;

		int sm_src_v2 = v2_e + iter_e + (blk_idx_b * TILE_B + v2_b2 * TILE_B1 + v2_b1 + (iter_d) * size_b) * size_e;

		int sm_dst_v2 = shm_v2_offset + (v2_b2 * (TILE_B1 * TILE_E)) + (v2_e ^ (((v2_b1 >> 1) & 1) << 2)) + (v2_b1 << 3);

		cuda::memcpy_async(thread, reinterpret_cast<double2*>(&sm_v2[sm_dst_v2]), reinterpret_cast<const double2*>(&dev_v2[sm_src_v2]), cuda::aligned_size_t<16>{sizeof(double2)}, pipeline);
	}

	for(int idx = (threadIdx.x << 1); idx < t2_elements; idx += (blockDim.x << 1))
	{
		int t2_c = idx / plane_t2;
		int rem_t2 = idx - t2_c * plane_t2;
		int t2_e = rem_t2 / TILE_A;
		int t2_a = rem_t2 - t2_e * TILE_A;

		int sm_src_t2 = blk_idx_a * TILE_A + t2_a + (iter_d + (t2_e + iter_e + (blk_idx_c * TILE_C + t2_c) * size_e) * size_d) * size_a;

		int sm_dst_t2 = shm_t2_offset + (t2_c * (TILE_E * TILE_A)) + (t2_a ^ (((t2_e >> 1) & 1) << 2)) + (t2_e << 3);

		cuda::memcpy_async(thread, reinterpret_cast<double2*>(&sm_t2[sm_dst_t2]), reinterpret_cast<const double2*>(&dev_t2[sm_src_t2]), cuda::aligned_size_t<16>{sizeof(double2)}, pipeline);
	}
}

__device__ void load_from_GM_2(double *__restrict dev_v2, double *__restrict dev_t2, double *sm_v2, double *sm_t2,
int size_a, int size_b, int size_c, int size_d, int size_e, 
int blk_idx_a, int blk_idx_b, int blk_idx_c, 
cuda::pipeline<cuda::thread_scope_block> &pipeline, cooperative_groups::thread_block_tile<1U, void> &thread,
int shm_v2_offset, int shm_t2_offset, int iter_e, int iter_d, int internal_upperbound)
{
	for(int idx = (threadIdx.x << 1); idx < v2_elements; idx += (blockDim.x << 1))
	{
		int v2_b2 = idx / plane_v2;
		int rem_v2 = idx - v2_b2 * plane_v2;
		int v2_b1 = rem_v2 / TILE_E;
		int v2_e = rem_v2 - v2_b1 * TILE_E;

		int sm_src_v2 = v2_e + iter_e + (blk_idx_b * TILE_B + v2_b2 * TILE_B1 + v2_b1 + (iter_d) * size_b) * size_e;

		int sm_dst_v2 = shm_v2_offset + (v2_b2 * (TILE_B1 * TILE_E)) + (v2_e ^ (((v2_b1 >> 1) & 1) << 2)) + (v2_b1 << 3);

		if((v2_e < TILE_UNIT - internal_upperbound))
		{
			cuda::memcpy_async(thread, reinterpret_cast<double2*>(&sm_v2[sm_dst_v2]), reinterpret_cast<const double2*>(&dev_v2[sm_src_v2]), cuda::aligned_size_t<16>{sizeof(double2)}, pipeline);
		}
		else
		{
			reinterpret_cast<double2*>(&sm_v2[sm_dst_v2])[0] = make_double2(0.0, 0.0);
		}
	}

	for(int idx = (threadIdx.x << 1); idx < t2_elements; idx += (blockDim.x << 1))
	{
		int t2_c = idx / plane_t2;
		int rem_t2 = idx - t2_c * plane_t2;
		int t2_e = rem_t2 / TILE_A;
		int t2_a = rem_t2 - t2_e * TILE_A;

		int sm_src_t2 = blk_idx_a * TILE_A + t2_a + (iter_d + (t2_e + iter_e + (blk_idx_c * TILE_C + t2_c) * size_e) * size_d) * size_a;

		int sm_dst_t2 = shm_t2_offset + (t2_c * (TILE_E * TILE_A)) + (t2_a ^ (((t2_e >> 1) & 1) << 2)) + (t2_e << 3);

		if((t2_e < TILE_UNIT - internal_upperbound))
		{
			cuda::memcpy_async(thread, reinterpret_cast<double2*>(&sm_t2[sm_dst_t2]), reinterpret_cast<const double2*>(&dev_t2[sm_src_t2]), cuda::aligned_size_t<16>{sizeof(double2)}, pipeline);
		}
		else
		{
			reinterpret_cast<double2*>(&sm_t2[sm_dst_t2])[0] = make_double2(0.0, 0.0);
		}
	}
}

__device__ void load_from_GM_3(double *__restrict dev_v2, double *__restrict dev_t2, double *sm_v2, double *sm_t2,
int size_a, int size_b, int size_c, int size_d, int size_e, 
int blk_idx_a, int blk_idx_b, int blk_idx_c, 
cuda::pipeline<cuda::thread_scope_block> &pipeline, cooperative_groups::thread_block_tile<1U, void> &thread,
int shm_v2_offset, int shm_t2_offset, int iter_e, int iter_d,
int rng_b, int rng_c)
{
	for(int idx = (threadIdx.x << 1); idx < v2_elements; idx += (blockDim.x << 1))
	{
		int v2_b2 = idx / plane_v2;
		int rem_v2 = idx - v2_b2 * plane_v2;
		int v2_b1 = rem_v2 / TILE_E;
		int v2_e = rem_v2 - v2_b1 * TILE_E;

		int sm_src_v2 = v2_e + iter_e + (blk_idx_b * TILE_B + v2_b2 * TILE_B1 + v2_b1 + (iter_d) * size_b) * size_e;

		int sm_dst_v2 = shm_v2_offset + (v2_b2 * (TILE_B1 * TILE_E)) + (v2_e ^ (((v2_b1 >> 1) & 1) << 2)) + (v2_b1 << 3);

		if((v2_b2 * TILE_B1 + v2_b1 < rng_b))
		{
			cuda::memcpy_async(thread, reinterpret_cast<double2*>(&sm_v2[sm_dst_v2]), reinterpret_cast<const double2*>(&dev_v2[sm_src_v2]), cuda::aligned_size_t<16>{sizeof(double2)}, pipeline);
		}
		else
		{
			reinterpret_cast<double2*>(&sm_v2[sm_dst_v2])[0] = make_double2(0.0, 0.0);
		}
	}

	for(int idx = (threadIdx.x << 1); idx < t2_elements; idx += (blockDim.x << 1))
	{
		int t2_c = idx / plane_t2;
		int rem_t2 = idx - t2_c * plane_t2;
		int t2_e = rem_t2 / TILE_A;
		int t2_a = rem_t2 - t2_e * TILE_A;

		int sm_src_t2 = blk_idx_a * TILE_A + t2_a + (iter_d + (t2_e + iter_e + (blk_idx_c * TILE_C + t2_c) * size_e) * size_d) * size_a;

		int sm_dst_t2 = shm_t2_offset + (t2_c * (TILE_E * TILE_A)) + (t2_a ^ (((t2_e >> 1) & 1) << 2)) + (t2_e << 3);

		if((t2_c < rng_c))
		{
			cuda::memcpy_async(thread, reinterpret_cast<double2*>(&sm_t2[sm_dst_t2]), reinterpret_cast<const double2*>(&dev_t2[sm_src_t2]), cuda::aligned_size_t<16>{sizeof(double2)}, pipeline);
		}
		else
		{
			reinterpret_cast<double2*>(&sm_t2[sm_dst_t2])[0] = make_double2(0.0, 0.0);
		}
	}
}

__device__ void load_from_GM_4(double *__restrict dev_v2, double *__restrict dev_t2, double *sm_v2, double *sm_t2,
int size_a, int size_b, int size_c, int size_d, int size_e, 
int blk_idx_a, int blk_idx_b, int blk_idx_c, 
cuda::pipeline<cuda::thread_scope_block> &pipeline, cooperative_groups::thread_block_tile<1U, void> &thread,
int shm_v2_offset, int shm_t2_offset, int iter_e, int iter_d,
int internal_upperbound,
int rng_b, int rng_c)
{
	for(int idx = (threadIdx.x << 1); idx < v2_elements; idx += (blockDim.x << 1))
	{
		int v2_b2 = idx / plane_v2;
		int rem_v2 = idx - v2_b2 * plane_v2;
		int v2_b1 = rem_v2 / TILE_E;
		int v2_e = rem_v2 - v2_b1 * TILE_E;

		int sm_src_v2 = v2_e + iter_e + (blk_idx_b * TILE_B + v2_b2 * TILE_B1 + v2_b1 + (iter_d) * size_b) * size_e;

		int sm_dst_v2 = shm_v2_offset + (v2_b2 * (TILE_B1 * TILE_E)) + (v2_e ^ (((v2_b1 >> 1) & 1) << 2)) + (v2_b1 << 3);

		if((v2_b2 * TILE_B1 + v2_b1 < rng_b) && (v2_e < TILE_UNIT - internal_upperbound))
		{
			cuda::memcpy_async(thread, reinterpret_cast<double2*>(&sm_v2[sm_dst_v2]), reinterpret_cast<const double2*>(&dev_v2[sm_src_v2]), cuda::aligned_size_t<16>{sizeof(double2)}, pipeline);
		}
		else
		{
			reinterpret_cast<double2*>(&sm_v2[sm_dst_v2])[0] = make_double2(0.0, 0.0);
		}
	}

	for(int idx = (threadIdx.x << 1); idx < t2_elements; idx += (blockDim.x << 1))
	{
		int t2_c = idx / plane_t2;
		int rem_t2 = idx - t2_c * plane_t2;
		int t2_e = rem_t2 / TILE_A;
		int t2_a = rem_t2 - t2_e * TILE_A;

		int sm_src_t2 = blk_idx_a * TILE_A + t2_a + (iter_d + (t2_e + iter_e + (blk_idx_c * TILE_C + t2_c) * size_e) * size_d) * size_a;

		int sm_dst_t2 = shm_t2_offset + (t2_c * (TILE_E * TILE_A)) + (t2_a ^ (((t2_e >> 1) & 1) << 2)) + (t2_e << 3);

		if((t2_e < TILE_UNIT - internal_upperbound) && (t2_c < rng_c))
		{
			cuda::memcpy_async(thread, reinterpret_cast<double2*>(&sm_t2[sm_dst_t2]), reinterpret_cast<const double2*>(&dev_t2[sm_src_t2]), cuda::aligned_size_t<16>{sizeof(double2)}, pipeline);
		}
		else
		{
			reinterpret_cast<double2*>(&sm_t2[sm_dst_t2])[0] = make_double2(0.0, 0.0);
		}
	}
}

__device__ void load_from_GM_5(double *__restrict dev_v2, double *__restrict dev_t2, double *sm_v2, double *sm_t2,
int size_a, int size_b, int size_c, int size_d, int size_e, 
int blk_idx_a, int blk_idx_b, int blk_idx_c, 
cuda::pipeline<cuda::thread_scope_block> &pipeline, cooperative_groups::thread_block_tile<1U, void> &thread,
int shm_v2_offset, int shm_t2_offset, int iter_e, int iter_d,
int rng_b, int rng_a)
{
	for(int idx = (threadIdx.x << 1); idx < v2_elements; idx += (blockDim.x << 1))
	{
		int v2_b2 = idx / plane_v2;
		int rem_v2 = idx - v2_b2 * plane_v2;
		int v2_b1 = rem_v2 / TILE_E;
		int v2_e = rem_v2 - v2_b1 * TILE_E;

		int sm_src_v2 = v2_e + iter_e + (blk_idx_b * TILE_B + v2_b2 * TILE_B1 + v2_b1 + (iter_d) * size_b) * size_e;

		int sm_dst_v2 = shm_v2_offset + (v2_b2 * (TILE_B1 * TILE_E)) + (v2_e ^ (((v2_b1 >> 1) & 1) << 2)) + (v2_b1 << 3);

		if((v2_b2 * TILE_B1 + v2_b1 < rng_b))
		{
			cuda::memcpy_async(thread, reinterpret_cast<double2*>(&sm_v2[sm_dst_v2]), reinterpret_cast<const double2*>(&dev_v2[sm_src_v2]), cuda::aligned_size_t<16>{sizeof(double2)}, pipeline);
		}
		else
		{
			reinterpret_cast<double2*>(&sm_v2[sm_dst_v2])[0] = make_double2(0.0, 0.0);
		}
	}

	for(int idx = (threadIdx.x << 1); idx < t2_elements; idx += (blockDim.x << 1))
	{
		int t2_c = idx / plane_t2;
		int rem_t2 = idx - t2_c * plane_t2;
		int t2_e = rem_t2 / TILE_A;
		int t2_a = rem_t2 - t2_e * TILE_A;

		int sm_src_t2 = blk_idx_a * TILE_A + t2_a + (iter_d + (t2_e + iter_e + (blk_idx_c * TILE_C + t2_c) * size_e) * size_d) * size_a;

		int sm_dst_t2 = shm_t2_offset + (t2_c * (TILE_E * TILE_A)) + (t2_a ^ (((t2_e >> 1) & 1) << 2)) + (t2_e << 3);

		if((t2_a < rng_a))
		{
			cuda::memcpy_async(thread, reinterpret_cast<double2*>(&sm_t2[sm_dst_t2]), reinterpret_cast<const double2*>(&dev_t2[sm_src_t2]), cuda::aligned_size_t<16>{sizeof(double2)}, pipeline);
		}
		else
		{
			reinterpret_cast<double2*>(&sm_t2[sm_dst_t2])[0] = make_double2(0.0, 0.0);
		}
	}
}

__device__ void load_from_GM_6(double *__restrict dev_v2, double *__restrict dev_t2, double *sm_v2, double *sm_t2,
int size_a, int size_b, int size_c, int size_d, int size_e, 
int blk_idx_a, int blk_idx_b, int blk_idx_c, 
cuda::pipeline<cuda::thread_scope_block> &pipeline, cooperative_groups::thread_block_tile<1U, void> &thread,
int shm_v2_offset, int shm_t2_offset, int iter_e, int iter_d,
int rng_b, int rng_a,
int internal_upperbound)
{
	for(int idx = (threadIdx.x << 1); idx < v2_elements; idx += (blockDim.x << 1))
	{
		int v2_b2 = idx / plane_v2;
		int rem_v2 = idx - v2_b2 * plane_v2;
		int v2_b1 = rem_v2 / TILE_E;
		int v2_e = rem_v2 - v2_b1 * TILE_E;

		int sm_src_v2 = v2_e + iter_e + (blk_idx_b * TILE_B + v2_b2 * TILE_B1 + v2_b1 + (iter_d) * size_b) * size_e;

		int sm_dst_v2 = shm_v2_offset + (v2_b2 * (TILE_B1 * TILE_E)) + (v2_e ^ (((v2_b1 >> 1) & 1) << 2)) + (v2_b1 << 3);

		if((v2_b2 * TILE_B1 + v2_b1 < rng_b) && (v2_e < TILE_UNIT - internal_upperbound))
		{
			cuda::memcpy_async(thread, reinterpret_cast<double2*>(&sm_v2[sm_dst_v2]), reinterpret_cast<const double2*>(&dev_v2[sm_src_v2]), cuda::aligned_size_t<16>{sizeof(double2)}, pipeline);
		}
		else
		{
			reinterpret_cast<double2*>(&sm_v2[sm_dst_v2])[0] = make_double2(0.0, 0.0);
		}
	}

	for(int idx = (threadIdx.x << 1); idx < t2_elements; idx += (blockDim.x << 1))
	{
		int t2_c = idx / plane_t2;
		int rem_t2 = idx - t2_c * plane_t2;
		int t2_e = rem_t2 / TILE_A;
		int t2_a = rem_t2 - t2_e * TILE_A;

		int sm_src_t2 = blk_idx_a * TILE_A + t2_a + (iter_d + (t2_e + iter_e + (blk_idx_c * TILE_C + t2_c) * size_e) * size_d) * size_a;

		int sm_dst_t2 = shm_t2_offset + (t2_c * (TILE_E * TILE_A)) + (t2_a ^ (((t2_e >> 1) & 1) << 2)) + (t2_e << 3);

		if((t2_a < rng_a) && (t2_e < TILE_UNIT - internal_upperbound))
		{
			cuda::memcpy_async(thread, reinterpret_cast<double2*>(&sm_t2[sm_dst_t2]), reinterpret_cast<const double2*>(&dev_t2[sm_src_t2]), cuda::aligned_size_t<16>{sizeof(double2)}, pipeline);
		}
		else
		{
			reinterpret_cast<double2*>(&sm_t2[sm_dst_t2])[0] = make_double2(0.0, 0.0);
		}
	}
}

__device__ void load_from_GM_7(double *__restrict dev_v2, double *__restrict dev_t2, double *sm_v2, double *sm_t2,
int size_a, int size_b, int size_c, int size_d, int size_e, 
int blk_idx_a, int blk_idx_b, int blk_idx_c, 
cuda::pipeline<cuda::thread_scope_block> &pipeline, cooperative_groups::thread_block_tile<1U, void> &thread,
int shm_v2_offset, int shm_t2_offset, int iter_e, int iter_d,
int rng_b, int rng_a,
int rng_c)
{
	for(int idx = (threadIdx.x << 1); idx < v2_elements; idx += (blockDim.x << 1))
	{
		int v2_b2 = idx / plane_v2;
		int rem_v2 = idx - v2_b2 * plane_v2;
		int v2_b1 = rem_v2 / TILE_E;
		int v2_e = rem_v2 - v2_b1 * TILE_E;

		int sm_src_v2 = v2_e + iter_e + (blk_idx_b * TILE_B + v2_b2 * TILE_B1 + v2_b1 + (iter_d) * size_b) * size_e;

		int sm_dst_v2 = shm_v2_offset + (v2_b2 * (TILE_B1 * TILE_E)) + (v2_e ^ (((v2_b1 >> 1) & 1) << 2)) + (v2_b1 << 3);

		if((v2_b2 * TILE_B1 + v2_b1 < rng_b))
		{
			cuda::memcpy_async(thread, reinterpret_cast<double2*>(&sm_v2[sm_dst_v2]), reinterpret_cast<const double2*>(&dev_v2[sm_src_v2]), cuda::aligned_size_t<16>{sizeof(double2)}, pipeline);
		}
		else
		{
			reinterpret_cast<double2*>(&sm_v2[sm_dst_v2])[0] = make_double2(0.0, 0.0);
		}
	}

	for(int idx = (threadIdx.x << 1); idx < t2_elements; idx += (blockDim.x << 1))
	{
		int t2_c = idx / plane_t2;
		int rem_t2 = idx - t2_c * plane_t2;
		int t2_e = rem_t2 / TILE_A;
		int t2_a = rem_t2 - t2_e * TILE_A;

		int sm_src_t2 = blk_idx_a * TILE_A + t2_a + (iter_d + (t2_e + iter_e + (blk_idx_c * TILE_C + t2_c) * size_e) * size_d) * size_a;

		int sm_dst_t2 = shm_t2_offset + (t2_c * (TILE_E * TILE_A)) + (t2_a ^ (((t2_e >> 1) & 1) << 2)) + (t2_e << 3);

		if((t2_a < rng_a) && (t2_c < rng_c))
		{
			cuda::memcpy_async(thread, reinterpret_cast<double2*>(&sm_t2[sm_dst_t2]), reinterpret_cast<const double2*>(&dev_t2[sm_src_t2]), cuda::aligned_size_t<16>{sizeof(double2)}, pipeline);
		}
		else
		{
			reinterpret_cast<double2*>(&sm_t2[sm_dst_t2])[0] = make_double2(0.0, 0.0);
		}
	}
}

__device__ void load_from_GM_8(double *__restrict dev_v2, double *__restrict dev_t2, double *sm_v2, double *sm_t2,
int size_a, int size_b, int size_c, int size_d, int size_e, 
int blk_idx_a, int blk_idx_b, int blk_idx_c, 
cuda::pipeline<cuda::thread_scope_block> &pipeline, cooperative_groups::thread_block_tile<1U, void> &thread,
int shm_v2_offset, int shm_t2_offset, int iter_e, int iter_d,
int rng_b, int rng_a,
int internal_upperbound,
int rng_c)
{
	for(int idx = (threadIdx.x << 1); idx < v2_elements; idx += (blockDim.x << 1))
	{
		int v2_b2 = idx / plane_v2;
		int rem_v2 = idx - v2_b2 * plane_v2;
		int v2_b1 = rem_v2 / TILE_E;
		int v2_e = rem_v2 - v2_b1 * TILE_E;

		int sm_src_v2 = v2_e + iter_e + (blk_idx_b * TILE_B + v2_b2 * TILE_B1 + v2_b1 + (iter_d) * size_b) * size_e;

		int sm_dst_v2 = shm_v2_offset + (v2_b2 * (TILE_B1 * TILE_E)) + (v2_e ^ (((v2_b1 >> 1) & 1) << 2)) + (v2_b1 << 3);

		if((v2_b2 * TILE_B1 + v2_b1 < rng_b) && (v2_e < TILE_UNIT - internal_upperbound))
		{
			cuda::memcpy_async(thread, reinterpret_cast<double2*>(&sm_v2[sm_dst_v2]), reinterpret_cast<const double2*>(&dev_v2[sm_src_v2]), cuda::aligned_size_t<16>{sizeof(double2)}, pipeline);
		}
		else
		{
			reinterpret_cast<double2*>(&sm_v2[sm_dst_v2])[0] = make_double2(0.0, 0.0);
		}
	}

	for(int idx = (threadIdx.x << 1); idx < t2_elements; idx += (blockDim.x << 1))
	{
		int t2_c = idx / plane_t2;
		int rem_t2 = idx - t2_c * plane_t2;
		int t2_e = rem_t2 / TILE_A;
		int t2_a = rem_t2 - t2_e * TILE_A;

		int sm_src_t2 = blk_idx_a * TILE_A + t2_a + (iter_d + (t2_e + iter_e + (blk_idx_c * TILE_C + t2_c) * size_e) * size_d) * size_a;

		int sm_dst_t2 = shm_t2_offset + (t2_c * (TILE_E * TILE_A)) + (t2_a ^ (((t2_e >> 1) & 1) << 2)) + (t2_e << 3);

		if((t2_a < rng_a) && (t2_e < TILE_UNIT - internal_upperbound) && (t2_c < rng_c))
		{
			cuda::memcpy_async(thread, reinterpret_cast<double2*>(&sm_t2[sm_dst_t2]), reinterpret_cast<const double2*>(&dev_t2[sm_src_t2]), cuda::aligned_size_t<16>{sizeof(double2)}, pipeline);
		}
		else
		{
			reinterpret_cast<double2*>(&sm_t2[sm_dst_t2])[0] = make_double2(0.0, 0.0);
		}
	}
}

__device__ void compute_MMA_1(double *__restrict sm_v2, double *__restrict sm_t2, const int wmiter, const int wniter, const int wrow, const int wcol,
const int v2_frag_cnt, const int t2_frag_cnt, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> *t3_frag,
const int shm_v2_offset, const int shm_t2_offset)
{
	const int ld_stride_a = TILE_B1 * TILE_E;
	const int ld_stride_b = TILE_E * TILE_A;

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, double, nvcuda::wmma::row_major> v2_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, double, nvcuda::wmma::row_major> t2_frag;

	const int lane = threadIdx.x & 31;
	const int v2_lane_offset = (lane & 3);
	const int v2_fragment_offset = (((lane >> 2) << 1) ^ ((lane >> 3) & 1));
	const int t2_lane_offset = ((lane >> 2) & 3);
	const int t2_fragment_offset = ((((lane & 3) << 1) ^ ((lane >> 1) & 1)) ^ (lane >> 4));

	for(int ll = 0; ll < TILE_UNIT; ll += 4)
	{
		#pragma unroll
		for(int iter_v2 = 0; iter_v2 < wmiter; iter_v2++)
		{
			#pragma unroll
			for(int cnt_v2 = 0; cnt_v2 < v2_frag_cnt; cnt_v2++)
			{
				int v2_offset = shm_v2_offset + ((wrow + iter_v2) * ld_stride_a) + (cnt_v2 << 6) + ((v2_fragment_offset ^ (ll >> 2)) << 2) + v2_lane_offset;
				v2_frag.x[0] = sm_v2[v2_offset];

				#pragma unroll
				for(int iter_t2 = 0; iter_t2 < wniter; iter_t2++)
				{
					int t2_offset = shm_t2_offset + ((wcol + iter_t2) * ld_stride_b) + (t2_fragment_offset << 2) + t2_lane_offset + (ll << 3);
					const int out_idx = ((iter_v2 * wniter) + iter_t2) * v2_frag_cnt + cnt_v2;
					t2_frag.x[0] = sm_t2[t2_offset];
					nvcuda::wmma::mma_sync(t3_frag[out_idx], v2_frag, t2_frag, t3_frag[out_idx]);
				}
			}
		}
	}
}

__device__ void compute_MMA_2(double *__restrict sm_v2, double *__restrict sm_t2, const int wmiter, const int wniter, const int wrow, const int wcol,
const int v2_frag_cnt, const int t2_frag_cnt, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> *t3_frag,
const int shm_v2_offset, const int shm_t2_offset,
int internal_upperbound)
{
	const int ld_stride_a = TILE_B1 * TILE_E;
	const int ld_stride_b = TILE_E * TILE_A;

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, double, nvcuda::wmma::row_major> v2_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, double, nvcuda::wmma::row_major> t2_frag;

	const int lane = threadIdx.x & 31;
	const int v2_lane_offset = (lane & 3);
	const int v2_fragment_offset = (((lane >> 2) << 1) ^ ((lane >> 3) & 1));
	const int t2_lane_offset = ((lane >> 2) & 3);
	const int t2_fragment_offset = ((((lane & 3) << 1) ^ ((lane >> 1) & 1)) ^ (lane >> 4));

	for(int ll = 0; ll < TILE_UNIT - internal_upperbound; ll += 4)
	{
		#pragma unroll
		for(int iter_v2 = 0; iter_v2 < wmiter; iter_v2++)
		{
			#pragma unroll
			for(int cnt_v2 = 0; cnt_v2 < v2_frag_cnt; cnt_v2++)
			{
				int v2_offset = shm_v2_offset + ((wrow + iter_v2) * ld_stride_a) + (cnt_v2 << 6) + ((v2_fragment_offset ^ (ll >> 2)) << 2) + v2_lane_offset;
				v2_frag.x[0] = sm_v2[v2_offset];

				#pragma unroll
				for(int iter_t2 = 0; iter_t2 < wniter; iter_t2++)
				{
					int t2_offset = shm_t2_offset + ((wcol + iter_t2) * ld_stride_b) + (t2_fragment_offset << 2) + t2_lane_offset + (ll << 3);
					const int out_idx = ((iter_v2 * wniter) + iter_t2) * v2_frag_cnt + cnt_v2;
					t2_frag.x[0] = sm_t2[t2_offset];
					nvcuda::wmma::mma_sync(t3_frag[out_idx], v2_frag, t2_frag, t3_frag[out_idx]);
				}
			}
		}
	}
}

__device__ void compute_MMA_3(double *__restrict sm_v2, double *__restrict sm_t2, const int wmiter, const int wniter, const int wrow, const int wcol,
const int v2_frag_cnt, const int t2_frag_cnt, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> *t3_frag,
const int shm_v2_offset, const int shm_t2_offset,
int rng_b, int rng_c)
{
	const int ld_stride_a = TILE_B1 * TILE_E;
	const int ld_stride_b = TILE_E * TILE_A;

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, double, nvcuda::wmma::row_major> v2_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, double, nvcuda::wmma::row_major> t2_frag;

	const int lane = threadIdx.x & 31;
	const int v2_lane_offset = (lane & 3);
	const int v2_fragment_offset = (((lane >> 2) << 1) ^ ((lane >> 3) & 1));
	const int t2_lane_offset = ((lane >> 2) & 3);
	const int t2_fragment_offset = ((((lane & 3) << 1) ^ ((lane >> 1) & 1)) ^ (lane >> 4));

	for(int ll = 0; ll < TILE_UNIT; ll += 4)
	{
		#pragma unroll
		for(int iter_v2 = 0; iter_v2 < wmiter; iter_v2++)
		{
			#pragma unroll
			for(int cnt_v2 = 0; cnt_v2 < v2_frag_cnt; cnt_v2++)
			{
				int v2_offset = shm_v2_offset + ((wrow + iter_v2) * ld_stride_a) + (cnt_v2 << 6) + ((v2_fragment_offset ^ (ll >> 2)) << 2) + v2_lane_offset;
				v2_frag.x[0] = sm_v2[v2_offset];

				#pragma unroll
				for(int iter_t2 = 0; iter_t2 < wniter; iter_t2++)
				{
					int t2_offset = shm_t2_offset + ((wcol + iter_t2) * ld_stride_b) + (t2_fragment_offset << 2) + t2_lane_offset + (ll << 3);
					const int out_idx = ((iter_v2 * wniter) + iter_t2) * v2_frag_cnt + cnt_v2;
					t2_frag.x[0] = sm_t2[t2_offset];
					nvcuda::wmma::mma_sync(t3_frag[out_idx], v2_frag, t2_frag, t3_frag[out_idx]);
				}
			}
		}
	}
}

__device__ void compute_MMA_4(double *__restrict sm_v2, double *__restrict sm_t2, const int wmiter, const int wniter, const int wrow, const int wcol,
const int v2_frag_cnt, const int t2_frag_cnt, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> *t3_frag,
const int shm_v2_offset, const int shm_t2_offset,
int internal_upperbound,
int rng_b, int rng_c)
{
	const int ld_stride_a = TILE_B1 * TILE_E;
	const int ld_stride_b = TILE_E * TILE_A;

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, double, nvcuda::wmma::row_major> v2_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, double, nvcuda::wmma::row_major> t2_frag;

	const int lane = threadIdx.x & 31;
	const int v2_lane_offset = (lane & 3);
	const int v2_fragment_offset = (((lane >> 2) << 1) ^ ((lane >> 3) & 1));
	const int t2_lane_offset = ((lane >> 2) & 3);
	const int t2_fragment_offset = ((((lane & 3) << 1) ^ ((lane >> 1) & 1)) ^ (lane >> 4));

	for(int ll = 0; ll < TILE_UNIT - internal_upperbound; ll += 4)
	{
		#pragma unroll
		for(int iter_v2 = 0; iter_v2 < wmiter; iter_v2++)
		{
			#pragma unroll
			for(int cnt_v2 = 0; cnt_v2 < v2_frag_cnt; cnt_v2++)
			{
				int v2_offset = shm_v2_offset + ((wrow + iter_v2) * ld_stride_a) + (cnt_v2 << 6) + ((v2_fragment_offset ^ (ll >> 2)) << 2) + v2_lane_offset;
				v2_frag.x[0] = sm_v2[v2_offset];

				#pragma unroll
				for(int iter_t2 = 0; iter_t2 < wniter; iter_t2++)
				{
					int t2_offset = shm_t2_offset + ((wcol + iter_t2) * ld_stride_b) + (t2_fragment_offset << 2) + t2_lane_offset + (ll << 3);
					const int out_idx = ((iter_v2 * wniter) + iter_t2) * v2_frag_cnt + cnt_v2;
					t2_frag.x[0] = sm_t2[t2_offset];
					nvcuda::wmma::mma_sync(t3_frag[out_idx], v2_frag, t2_frag, t3_frag[out_idx]);
				}
			}
		}
	}
}

__device__ void compute_MMA_5(double *__restrict sm_v2, double *__restrict sm_t2, const int wmiter, const int wniter, const int wrow, const int wcol,
const int v2_frag_cnt, const int t2_frag_cnt, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> *t3_frag,
const int shm_v2_offset, const int shm_t2_offset,
int rng_b, int rng_a)
{
	const int ld_stride_a = TILE_B1 * TILE_E;
	const int ld_stride_b = TILE_E * TILE_A;

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, double, nvcuda::wmma::row_major> v2_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, double, nvcuda::wmma::row_major> t2_frag;

	const int lane = threadIdx.x & 31;
	const int v2_lane_offset = (lane & 3);
	const int v2_fragment_offset = (((lane >> 2) << 1) ^ ((lane >> 3) & 1));
	const int t2_lane_offset = ((lane >> 2) & 3);
	const int t2_fragment_offset = ((((lane & 3) << 1) ^ ((lane >> 1) & 1)) ^ (lane >> 4));

	for(int ll = 0; ll < TILE_UNIT; ll += 4)
	{
		#pragma unroll
		for(int iter_v2 = 0; iter_v2 < wmiter; iter_v2++)
		{
			#pragma unroll
			for(int cnt_v2 = 0; cnt_v2 < v2_frag_cnt; cnt_v2++)
			{
				int v2_offset = shm_v2_offset + ((wrow + iter_v2) * ld_stride_a) + (cnt_v2 << 6) + ((v2_fragment_offset ^ (ll >> 2)) << 2) + v2_lane_offset;
				v2_frag.x[0] = sm_v2[v2_offset];

				#pragma unroll
				for(int iter_t2 = 0; iter_t2 < wniter; iter_t2++)
				{
					int t2_offset = shm_t2_offset + ((wcol + iter_t2) * ld_stride_b) + (t2_fragment_offset << 2) + t2_lane_offset + (ll << 3);
					const int out_idx = ((iter_v2 * wniter) + iter_t2) * v2_frag_cnt + cnt_v2;
					t2_frag.x[0] = sm_t2[t2_offset];
					nvcuda::wmma::mma_sync(t3_frag[out_idx], v2_frag, t2_frag, t3_frag[out_idx]);
				}
			}
		}
	}
}

__device__ void compute_MMA_6(double *__restrict sm_v2, double *__restrict sm_t2, const int wmiter, const int wniter, const int wrow, const int wcol,
const int v2_frag_cnt, const int t2_frag_cnt, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> *t3_frag,
const int shm_v2_offset, const int shm_t2_offset,
int rng_b, int rng_a,
int internal_upperbound)
{
	const int ld_stride_a = TILE_B1 * TILE_E;
	const int ld_stride_b = TILE_E * TILE_A;

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, double, nvcuda::wmma::row_major> v2_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, double, nvcuda::wmma::row_major> t2_frag;

	const int lane = threadIdx.x & 31;
	const int v2_lane_offset = (lane & 3);
	const int v2_fragment_offset = (((lane >> 2) << 1) ^ ((lane >> 3) & 1));
	const int t2_lane_offset = ((lane >> 2) & 3);
	const int t2_fragment_offset = ((((lane & 3) << 1) ^ ((lane >> 1) & 1)) ^ (lane >> 4));

	for(int ll = 0; ll < TILE_UNIT - internal_upperbound; ll += 4)
	{
		#pragma unroll
		for(int iter_v2 = 0; iter_v2 < wmiter; iter_v2++)
		{
			#pragma unroll
			for(int cnt_v2 = 0; cnt_v2 < v2_frag_cnt; cnt_v2++)
			{
				int v2_offset = shm_v2_offset + ((wrow + iter_v2) * ld_stride_a) + (cnt_v2 << 6) + ((v2_fragment_offset ^ (ll >> 2)) << 2) + v2_lane_offset;
				v2_frag.x[0] = sm_v2[v2_offset];

				#pragma unroll
				for(int iter_t2 = 0; iter_t2 < wniter; iter_t2++)
				{
					int t2_offset = shm_t2_offset + ((wcol + iter_t2) * ld_stride_b) + (t2_fragment_offset << 2) + t2_lane_offset + (ll << 3);
					const int out_idx = ((iter_v2 * wniter) + iter_t2) * v2_frag_cnt + cnt_v2;
					t2_frag.x[0] = sm_t2[t2_offset];
					nvcuda::wmma::mma_sync(t3_frag[out_idx], v2_frag, t2_frag, t3_frag[out_idx]);
				}
			}
		}
	}
}

__device__ void compute_MMA_7(double *__restrict sm_v2, double *__restrict sm_t2, const int wmiter, const int wniter, const int wrow, const int wcol,
const int v2_frag_cnt, const int t2_frag_cnt, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> *t3_frag,
const int shm_v2_offset, const int shm_t2_offset,
int rng_b, int rng_a,
int rng_c)
{
	const int ld_stride_a = TILE_B1 * TILE_E;
	const int ld_stride_b = TILE_E * TILE_A;

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, double, nvcuda::wmma::row_major> v2_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, double, nvcuda::wmma::row_major> t2_frag;

	const int lane = threadIdx.x & 31;
	const int v2_lane_offset = (lane & 3);
	const int v2_fragment_offset = (((lane >> 2) << 1) ^ ((lane >> 3) & 1));
	const int t2_lane_offset = ((lane >> 2) & 3);
	const int t2_fragment_offset = ((((lane & 3) << 1) ^ ((lane >> 1) & 1)) ^ (lane >> 4));

	for(int ll = 0; ll < TILE_UNIT; ll += 4)
	{
		#pragma unroll
		for(int iter_v2 = 0; iter_v2 < wmiter; iter_v2++)
		{
			#pragma unroll
			for(int cnt_v2 = 0; cnt_v2 < v2_frag_cnt; cnt_v2++)
			{
				int v2_offset = shm_v2_offset + ((wrow + iter_v2) * ld_stride_a) + (cnt_v2 << 6) + ((v2_fragment_offset ^ (ll >> 2)) << 2) + v2_lane_offset;
				v2_frag.x[0] = sm_v2[v2_offset];

				#pragma unroll
				for(int iter_t2 = 0; iter_t2 < wniter; iter_t2++)
				{
					int t2_offset = shm_t2_offset + ((wcol + iter_t2) * ld_stride_b) + (t2_fragment_offset << 2) + t2_lane_offset + (ll << 3);
					const int out_idx = ((iter_v2 * wniter) + iter_t2) * v2_frag_cnt + cnt_v2;
					t2_frag.x[0] = sm_t2[t2_offset];
					nvcuda::wmma::mma_sync(t3_frag[out_idx], v2_frag, t2_frag, t3_frag[out_idx]);
				}
			}
		}
	}
}

__device__ void compute_MMA_8(double *__restrict sm_v2, double *__restrict sm_t2, const int wmiter, const int wniter, const int wrow, const int wcol,
const int v2_frag_cnt, const int t2_frag_cnt, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> *t3_frag,
const int shm_v2_offset, const int shm_t2_offset,
int rng_b, int rng_a,
int internal_upperbound,
int rng_c)
{
	const int ld_stride_a = TILE_B1 * TILE_E;
	const int ld_stride_b = TILE_E * TILE_A;

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, double, nvcuda::wmma::row_major> v2_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, double, nvcuda::wmma::row_major> t2_frag;

	const int lane = threadIdx.x & 31;
	const int v2_lane_offset = (lane & 3);
	const int v2_fragment_offset = (((lane >> 2) << 1) ^ ((lane >> 3) & 1));
	const int t2_lane_offset = ((lane >> 2) & 3);
	const int t2_fragment_offset = ((((lane & 3) << 1) ^ ((lane >> 1) & 1)) ^ (lane >> 4));

	for(int ll = 0; ll < TILE_UNIT - internal_upperbound; ll += 4)
	{
		#pragma unroll
		for(int iter_v2 = 0; iter_v2 < wmiter; iter_v2++)
		{
			#pragma unroll
			for(int cnt_v2 = 0; cnt_v2 < v2_frag_cnt; cnt_v2++)
			{
				int v2_offset = shm_v2_offset + ((wrow + iter_v2) * ld_stride_a) + (cnt_v2 << 6) + ((v2_fragment_offset ^ (ll >> 2)) << 2) + v2_lane_offset;
				v2_frag.x[0] = sm_v2[v2_offset];

				#pragma unroll
				for(int iter_t2 = 0; iter_t2 < wniter; iter_t2++)
				{
					int t2_offset = shm_t2_offset + ((wcol + iter_t2) * ld_stride_b) + (t2_fragment_offset << 2) + t2_lane_offset + (ll << 3);
					const int out_idx = ((iter_v2 * wniter) + iter_t2) * v2_frag_cnt + cnt_v2;
					t2_frag.x[0] = sm_t2[t2_offset];
					nvcuda::wmma::mma_sync(t3_frag[out_idx], v2_frag, t2_frag, t3_frag[out_idx]);
				}
			}
		}
	}
}

__device__ __forceinline__ void scatter_store_tail0(double *__restrict__ dev_t3, int m_base, int g_n_iter, int rng_b, int rng_c, int intra_stride,
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double>&t3_frag)
{
	const int lane = threadIdx.x & 31;
	const int frag_m = lane >> 2;
	const int frag_n = (lane & 3) << 1;
	double2 reg = *reinterpret_cast<const double2*>(&t3_frag);
	double x = reg.x;
	double y = __shfl_xor_sync(0xFFFFFFFF, reg.y, 4);
	int div4 = (lane & 7) >> 2;
	int id_m = frag_m - div4;
	int id_n = frag_n + div4;
	if((m_base + id_m) < rng_b && g_n_iter < rng_c)
	{
		dev_t3[id_m * intra_stride + id_n] = (div4 ? y : x);
	}

	id_m = frag_m + (div4 ^ 1);
	id_n = frag_n + (div4 ^ 1);
	if((m_base + id_m) < rng_b && g_n_iter < rng_c)
	{
		dev_t3[id_m * intra_stride + id_n] = ((div4 ^ 1) ? y : x);
	}
}

__device__ __forceinline__ void scatter_store_tail1(double *__restrict__ dev_t3, int m_base, int n_base, int rng_b, int rng_a, int intra_stride,
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double>&t3_frag)
{
	const int lane = threadIdx.x & 31;
	const int frag_m = lane >> 2;
	const int frag_n = (lane & 3) << 1;
	double2 reg = *reinterpret_cast<const double2*>(&t3_frag);
	double x = reg.x;
	double y = __shfl_xor_sync(0xFFFFFFFF, reg.y, 4);
	int div4 = (lane & 7) >> 2;
	int id_m = frag_m - div4;
	int id_n = frag_n + div4;
	if((m_base + id_m) < rng_b && (n_base + id_n) < rng_a)
	{
		dev_t3[id_m * intra_stride + id_n] = (div4 ? y : x);
	}

	id_m = frag_m + (div4 ^ 1);
	id_n = frag_n + (div4 ^ 1);
	if((m_base + id_m) < rng_b && (n_base + id_n) < rng_a)
	{
		dev_t3[id_m * intra_stride + id_n] = ((div4 ^ 1) ? y : x);
	}
}

__device__ __forceinline__ void scatter_store_tail2(double *__restrict__ dev_t3, int m_base, int g_n_iter, int n_base, int rng_b, int rng_c, int rng_a, int intra_stride,
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double>&t3_frag)
{
	const int lane = threadIdx.x & 31;
	const int frag_m = lane >> 2;
	const int frag_n = (lane & 3) << 1;
	double2 reg = *reinterpret_cast<const double2*>(&t3_frag);
	double x = reg.x;
	double y = __shfl_xor_sync(0xFFFFFFFF, reg.y, 4);
	int div4 = (lane & 7) >> 2;
	int id_m = frag_m - div4;
	int id_n = frag_n + div4;
	if((m_base + id_m) < rng_b && g_n_iter < rng_c && (n_base + id_n) < rng_a)
	{
		dev_t3[id_m * intra_stride + id_n] = (div4 ? y : x);
	}

	id_m = frag_m + (div4 ^ 1);
	id_n = frag_n + (div4 ^ 1);
	if((m_base + id_m) < rng_b && g_n_iter < rng_c && (n_base + id_n) < rng_a)
	{
		dev_t3[id_m * intra_stride + id_n] = ((div4 ^ 1) ? y : x);
	}
}

extern "C" __global__ void kernel_1(double *dev_t3, double *__restrict dev_v2, double *__restrict dev_t2, 
int size_a, int size_b, int size_c, int size_d, int size_e, 
int numBlk_a, int numBlk_b, int numBlk_c, 
int size_internal, int v2_tile_size, int t2_tile_size)
{
	// For Pipeline
	auto thread = cooperative_groups::this_thread();
	auto block = cooperative_groups::this_thread_block();

	#pragma nv_diag_suppress static_var_with_dynamic_init
	__shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, PIPELINE_STAGES> shared_state;
	auto pipeline = cuda::make_pipeline(block, &shared_state);

	// For Shared Memory
	extern __shared__ __align__(128) double shared_memory[];
	double *sm_v2 = shared_memory;
	double *sm_t2 = shared_memory + (PIPELINE_STAGES * v2_tile_size);

	// Index for each Thread Blocks
	int tmp_blkIdx;
	const int blk_idx_c = blockIdx.x / (numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_b * numBlk_a);

	const int blk_idx_b = tmp_blkIdx / (numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	const int blk_idx_a = tmp_blkIdx;

	// Warp related variables
	const int wmiter = TILE_B2 / 1;
	const int wniter = TILE_C / 4;
	const int wrow = ((threadIdx.x >> 5) / 4) * wmiter;
	const int wcol = ((threadIdx.x >> 5) % 4) * wniter;

	// Fragment Allocation
	const int v2_frag_cnt = TILE_B1 >> 3;
	const int t2_frag_cnt = TILE_A >> 3;

	// Size of fragment stride
	const int inter_m_stride = TILE_B1 * size_a;
	const int inter_n_stride = size_b * size_a;
	const int intra_stride = size_a;

	// Base index of each warps for result tensor
	const int t3_base_thread = (blk_idx_a * TILE_A + (blk_idx_b * TILE_B + (blk_idx_c * TILE_C) * size_b) * size_a)
								+ wrow * inter_m_stride + wcol * inter_n_stride;

	// WMMA fragment
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> t3_frag[wmiter * wniter * v2_frag_cnt * t2_frag_cnt];

	// Fill fragment
	#pragma unroll
	for(int i = 0; i < wmiter * wniter * v2_frag_cnt * t2_frag_cnt; i++)
	{
		nvcuda::wmma::fill_fragment(t3_frag[i], 0.0);
	}

	// Shared memory offset for pipelining
	int shm_offset = 0;

	// Kernel 1
	// Frag_mapped : Full, Reg_mapped : Full / Internal : Full

	// Tensor Contraction : [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'd', 'e', 'c']], [16, 'STR_SD2_V2_H7', '', 'v2', ['e', 'b', 'd']], '-']
	for(int iter_d = 0; iter_d < size_d; iter_d += TILE_D)
	{
		for(int l = -((int)(PIPELINE_STAGES - 1) * TILE_UNIT); l < size_internal; l += TILE_UNIT)
		{
			// Load
			int load_idx = l + (PIPELINE_STAGES - 1) * TILE_UNIT;

			if(load_idx < size_internal)
			{
				int next_shm_offset = (shm_offset + PIPELINE_STAGES - 1) % PIPELINE_STAGES;

				pipeline.producer_acquire();
				load_from_GM_1(dev_v2, dev_t2, sm_v2, sm_t2,
						size_a, size_b, size_c,
						size_d, size_e,
						blk_idx_a, blk_idx_b, blk_idx_c,
						pipeline, thread,
						next_shm_offset * v2_tile_size, next_shm_offset * t2_tile_size, load_idx, iter_d);
				pipeline.producer_commit();
			}

			// Compute
			if(l >= 0)
			{
				pipeline.consumer_wait();
				compute_MMA_1(sm_v2, sm_t2, wmiter, wniter, wrow, wcol,
						v2_frag_cnt, t2_frag_cnt,
						t3_frag,
						shm_offset * v2_tile_size, shm_offset * t2_tile_size);
				pipeline.consumer_release();
			}

			// Update shared memory offset
			shm_offset = (shm_offset + 1) % PIPELINE_STAGES;
		}

		// Update shared memory offset
		shm_offset = (shm_offset + 1) % PIPELINE_STAGES;
	}

	// Store result into global memory
	int frag_idx = 0;
	#pragma unroll
	for(int iter_v2 = 0; iter_v2 < wmiter; iter_v2++)
	{
		int base = t3_base_thread + iter_v2 * inter_m_stride;
		#pragma unroll
		for(int iter_t2 = 0; iter_t2 < wniter; iter_t2++)
		{
			int base_iter = base + iter_t2 * inter_n_stride;
			#pragma unroll
			for(int cnt_v2 = 0; cnt_v2 < v2_frag_cnt; cnt_v2++)
			{
				int base_v2 = base_iter + (cnt_v2 << 3) * intra_stride;
				#pragma unroll
				for(int cnt_t2 = 0; cnt_t2 < t2_frag_cnt; cnt_t2++, frag_idx++)
				{
					int n_base = (cnt_t2 << 3);
					int dst = base_v2 + (cnt_t2 << 3);
					nvcuda::wmma::store_matrix_sync(&dev_t3[dst], t3_frag[frag_idx], intra_stride, nvcuda::wmma::mem_row_major);
				}
			}
		}
	}
}

extern "C" __global__ void kernel_2(double *dev_t3, double *__restrict dev_v2, double *__restrict dev_t2, 
int size_a, int size_b, int size_c, int size_d, int size_e, 
int numBlk_a, int numBlk_b, int numBlk_c, 
int size_internal, int v2_tile_size, int t2_tile_size)
{
	// For Pipeline
	auto thread = cooperative_groups::this_thread();
	auto block = cooperative_groups::this_thread_block();

	#pragma nv_diag_suppress static_var_with_dynamic_init
	__shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, PIPELINE_STAGES> shared_state;
	auto pipeline = cuda::make_pipeline(block, &shared_state);

	// For Shared Memory
	extern __shared__ __align__(128) double shared_memory[];
	double *sm_v2 = shared_memory;
	double *sm_t2 = shared_memory + (PIPELINE_STAGES * v2_tile_size);

	// Index for each Thread Blocks
	int tmp_blkIdx;
	const int blk_idx_c = blockIdx.x / (numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_b * numBlk_a);

	const int blk_idx_b = tmp_blkIdx / (numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	const int blk_idx_a = tmp_blkIdx;

	// Warp related variables
	const int wmiter = TILE_B2 / 1;
	const int wniter = TILE_C / 4;
	const int wrow = ((threadIdx.x >> 5) / 4) * wmiter;
	const int wcol = ((threadIdx.x >> 5) % 4) * wniter;

	// Fragment Allocation
	const int v2_frag_cnt = TILE_B1 >> 3;
	const int t2_frag_cnt = TILE_A >> 3;

	// Size of fragment stride
	const int inter_m_stride = TILE_B1 * size_a;
	const int inter_n_stride = size_b * size_a;
	const int intra_stride = size_a;

	// Base index of each warps for result tensor
	const int t3_base_thread = (blk_idx_a * TILE_A + (blk_idx_b * TILE_B + (blk_idx_c * TILE_C) * size_b) * size_a)
								+ wrow * inter_m_stride + wcol * inter_n_stride;

	// WMMA fragment
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> t3_frag[wmiter * wniter * v2_frag_cnt * t2_frag_cnt];

	// Fill fragment
	#pragma unroll
	for(int i = 0; i < wmiter * wniter * v2_frag_cnt * t2_frag_cnt; i++)
	{
		nvcuda::wmma::fill_fragment(t3_frag[i], 0.0);
	}

	// Shared memory offset for pipelining
	int shm_offset = 0;

	// Kernel 2
	// Frag_mapped : Full, Reg_mapped : Full / Internal : Partial
	int iter_e_offset = 0;

	// Tensor Contraction : [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'd', 'e', 'c']], [16, 'STR_SD2_V2_H7', '', 'v2', ['e', 'b', 'd']], '-']
	for(int iter_d = 0; iter_d < size_d; iter_d += TILE_D)
	{
		for(int l = -((int)(PIPELINE_STAGES - 1) * TILE_UNIT); l < size_internal; l += TILE_UNIT)
		{
			// Load
			int load_idx = l + (PIPELINE_STAGES - 1) * TILE_UNIT;

			if(load_idx < size_internal)
			{
				int next_shm_offset = (shm_offset + PIPELINE_STAGES - 1) % PIPELINE_STAGES;
				int load_ub = max(0, (load_idx + TILE_UNIT - size_internal));

				pipeline.producer_acquire();
				load_from_GM_2(dev_v2, dev_t2, sm_v2, sm_t2,
						size_a, size_b, size_c,
						size_d, size_e,
						blk_idx_a, blk_idx_b, blk_idx_c,
						pipeline, thread,
						next_shm_offset * v2_tile_size, next_shm_offset * t2_tile_size, load_idx, iter_d,
						load_ub);
				pipeline.producer_commit();
			}

			// Compute
			if(l >= 0)
			{
				int compute_ub = max(0, (l + TILE_UNIT - size_internal));

				pipeline.consumer_wait();
				compute_MMA_2(sm_v2, sm_t2, wmiter, wniter, wrow, wcol,
						v2_frag_cnt, t2_frag_cnt,
						t3_frag,
						shm_offset * v2_tile_size, shm_offset * t2_tile_size,
						compute_ub);
				pipeline.consumer_release();
			}

			// Update shared memory offset
			shm_offset = (shm_offset + 1) % PIPELINE_STAGES;
		}

		// Update shared memory offset
		shm_offset = (shm_offset + 1) % PIPELINE_STAGES;
	}

	// Store result into global memory
	int frag_idx = 0;
	#pragma unroll
	for(int iter_v2 = 0; iter_v2 < wmiter; iter_v2++)
	{
		int base = t3_base_thread + iter_v2 * inter_m_stride;
		#pragma unroll
		for(int iter_t2 = 0; iter_t2 < wniter; iter_t2++)
		{
			int base_iter = base + iter_t2 * inter_n_stride;
			#pragma unroll
			for(int cnt_v2 = 0; cnt_v2 < v2_frag_cnt; cnt_v2++)
			{
				int base_v2 = base_iter + (cnt_v2 << 3) * intra_stride;
				#pragma unroll
				for(int cnt_t2 = 0; cnt_t2 < t2_frag_cnt; cnt_t2++, frag_idx++)
				{
					int n_base = (cnt_t2 << 3);
					int dst = base_v2 + (cnt_t2 << 3);
					nvcuda::wmma::store_matrix_sync(&dev_t3[dst], t3_frag[frag_idx], intra_stride, nvcuda::wmma::mem_row_major);
				}
			}
		}
	}
}

extern "C" __global__ void kernel_3(double *dev_t3, double *__restrict dev_v2, double *__restrict dev_t2, 
int size_a, int size_b, int size_c, int size_d, int size_e, 
int numBlk_a, int numBlk_b, int numBlk_c, 
int size_internal, int v2_tile_size, int t2_tile_size)
{
	// For Pipeline
	auto thread = cooperative_groups::this_thread();
	auto block = cooperative_groups::this_thread_block();

	#pragma nv_diag_suppress static_var_with_dynamic_init
	__shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, PIPELINE_STAGES> shared_state;
	auto pipeline = cuda::make_pipeline(block, &shared_state);

	// For Shared Memory
	extern __shared__ __align__(128) double shared_memory[];
	double *sm_v2 = shared_memory;
	double *sm_t2 = shared_memory + (PIPELINE_STAGES * v2_tile_size);

	// Index for each Thread Blocks
	int tmp_blkIdx;
	const int blk_idx_c = blockIdx.x / (numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_b * numBlk_a);

	const int blk_idx_b = tmp_blkIdx / (numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	const int blk_idx_a = tmp_blkIdx;

	// Warp related variables
	const int wmiter = TILE_B2 / 1;
	const int wniter = TILE_C / 4;
	const int wrow = ((threadIdx.x >> 5) / 4) * wmiter;
	const int wcol = ((threadIdx.x >> 5) % 4) * wniter;

	// Fragment Allocation
	const int v2_frag_cnt = TILE_B1 >> 3;
	const int t2_frag_cnt = TILE_A >> 3;

	// Size of fragment stride
	const int inter_m_stride = TILE_B1 * size_a;
	const int inter_n_stride = size_b * size_a;
	const int intra_stride = size_a;

	// Base index of each warps for result tensor
	const int t3_base_thread = (blk_idx_a * TILE_A + (blk_idx_b * TILE_B + (blk_idx_c * TILE_C) * size_b) * size_a)
								+ wrow * inter_m_stride + wcol * inter_n_stride;

	// WMMA fragment
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> t3_frag[wmiter * wniter * v2_frag_cnt * t2_frag_cnt];

	// Fill fragment
	#pragma unroll
	for(int i = 0; i < wmiter * wniter * v2_frag_cnt * t2_frag_cnt; i++)
	{
		nvcuda::wmma::fill_fragment(t3_frag[i], 0.0);
	}

	// Shared memory offset for pipelining
	int shm_offset = 0;

	// Kernel 3
	// Frag_mapped : Full, Reg_mapped : Partial / Internal : Full
	int rng_b, rng_c;
	if((size_b - (blk_idx_b * TILE_B)) >= TILE_B)
    {
        rng_b = TILE_B;
    }
    else
    {
        rng_b = size_b % TILE_B;
    }

	if((size_c - (blk_idx_c * TILE_C)) >= TILE_C)
    {
        rng_c = TILE_C;
    }
    else
    {
        rng_c = size_c % TILE_C;
    }

	// Tensor Contraction : [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'd', 'e', 'c']], [16, 'STR_SD2_V2_H7', '', 'v2', ['e', 'b', 'd']], '-']
	for(int iter_d = 0; iter_d < size_d; iter_d += TILE_D)
	{
		for(int l = -((int)(PIPELINE_STAGES - 1) * TILE_UNIT); l < size_internal; l += TILE_UNIT)
		{
			// Load
			int load_idx = l + (PIPELINE_STAGES - 1) * TILE_UNIT;

			if(load_idx < size_internal)
			{
				int next_shm_offset = (shm_offset + PIPELINE_STAGES - 1) % PIPELINE_STAGES;

				pipeline.producer_acquire();
				load_from_GM_3(dev_v2, dev_t2, sm_v2, sm_t2,
						size_a, size_b, size_c,
						size_d, size_e,
						blk_idx_a, blk_idx_b, blk_idx_c,
						pipeline, thread,
						next_shm_offset * v2_tile_size, next_shm_offset * t2_tile_size, load_idx, iter_d,
						rng_b, rng_c);
				pipeline.producer_commit();
			}

			// Compute
			if(l >= 0)
			{
				pipeline.consumer_wait();
				compute_MMA_3(sm_v2, sm_t2, wmiter, wniter, wrow, wcol,
						v2_frag_cnt, t2_frag_cnt,
						t3_frag,
						shm_offset * v2_tile_size, shm_offset * t2_tile_size,
						rng_b, rng_c);
				pipeline.consumer_release();
			}

			// Update shared memory offset
			shm_offset = (shm_offset + 1) % PIPELINE_STAGES;
		}

		// Update shared memory offset
		shm_offset = (shm_offset + 1) % PIPELINE_STAGES;
	}

	// Store result into global memory
	int frag_idx = 0;
	#pragma unroll
	for(int iter_v2 = 0; iter_v2 < wmiter; iter_v2++)
	{
		int base = t3_base_thread + iter_v2 * inter_m_stride;
		int g_m_iter = (wrow + iter_v2) * TILE_B1;
		#pragma unroll
		for(int iter_t2 = 0; iter_t2 < wniter; iter_t2++)
		{
			int base_iter = base + iter_t2 * inter_n_stride;
			int g_n_iter = wcol + iter_t2;
			#pragma unroll
			for(int cnt_v2 = 0; cnt_v2 < v2_frag_cnt; cnt_v2++)
			{
				int m_base = g_m_iter + (cnt_v2 << 3);
				int base_v2 = base_iter + (cnt_v2 << 3) * intra_stride;
				#pragma unroll
				for(int cnt_t2 = 0; cnt_t2 < t2_frag_cnt; cnt_t2++, frag_idx++)
				{
					int n_base = (cnt_t2 << 3);
					int dst = base_v2 + (cnt_t2 << 3);
					bool full = ((m_base + 8 <= rng_b) && (g_n_iter < rng_c) && (((uintptr_t)dst & 0x1F) == 0) && ((intra_stride & 1) == 0));
					if(full)
					{
						nvcuda::wmma::store_matrix_sync(&dev_t3[dst], t3_frag[frag_idx], intra_stride, nvcuda::wmma::mem_row_major);
					}
					else
					{
						scatter_store_tail0(&dev_t3[dst], m_base, g_n_iter, rng_b, rng_c, intra_stride, t3_frag[frag_idx]);
					}
				}
			}
		}
	}
}

extern "C" __global__ void kernel_4(double *dev_t3, double *__restrict dev_v2, double *__restrict dev_t2, 
int size_a, int size_b, int size_c, int size_d, int size_e, 
int numBlk_a, int numBlk_b, int numBlk_c, 
int size_internal, int v2_tile_size, int t2_tile_size)
{
	// For Pipeline
	auto thread = cooperative_groups::this_thread();
	auto block = cooperative_groups::this_thread_block();

	#pragma nv_diag_suppress static_var_with_dynamic_init
	__shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, PIPELINE_STAGES> shared_state;
	auto pipeline = cuda::make_pipeline(block, &shared_state);

	// For Shared Memory
	extern __shared__ __align__(128) double shared_memory[];
	double *sm_v2 = shared_memory;
	double *sm_t2 = shared_memory + (PIPELINE_STAGES * v2_tile_size);

	// Index for each Thread Blocks
	int tmp_blkIdx;
	const int blk_idx_c = blockIdx.x / (numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_b * numBlk_a);

	const int blk_idx_b = tmp_blkIdx / (numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	const int blk_idx_a = tmp_blkIdx;

	// Warp related variables
	const int wmiter = TILE_B2 / 1;
	const int wniter = TILE_C / 4;
	const int wrow = ((threadIdx.x >> 5) / 4) * wmiter;
	const int wcol = ((threadIdx.x >> 5) % 4) * wniter;

	// Fragment Allocation
	const int v2_frag_cnt = TILE_B1 >> 3;
	const int t2_frag_cnt = TILE_A >> 3;

	// Size of fragment stride
	const int inter_m_stride = TILE_B1 * size_a;
	const int inter_n_stride = size_b * size_a;
	const int intra_stride = size_a;

	// Base index of each warps for result tensor
	const int t3_base_thread = (blk_idx_a * TILE_A + (blk_idx_b * TILE_B + (blk_idx_c * TILE_C) * size_b) * size_a)
								+ wrow * inter_m_stride + wcol * inter_n_stride;

	// WMMA fragment
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> t3_frag[wmiter * wniter * v2_frag_cnt * t2_frag_cnt];

	// Fill fragment
	#pragma unroll
	for(int i = 0; i < wmiter * wniter * v2_frag_cnt * t2_frag_cnt; i++)
	{
		nvcuda::wmma::fill_fragment(t3_frag[i], 0.0);
	}

	// Shared memory offset for pipelining
	int shm_offset = 0;

	// Kernel 4
	// Frag_mapped : Full, Reg_mapped : Partial / Internal : Partial
	int iter_e_offset = 0;
	int rng_b, rng_c;
	if((size_b - (blk_idx_b * TILE_B)) >= TILE_B)
    {
        rng_b = TILE_B;
    }
    else
    {
        rng_b = size_b % TILE_B;
    }

	if((size_c - (blk_idx_c * TILE_C)) >= TILE_C)
    {
        rng_c = TILE_C;
    }
    else
    {
        rng_c = size_c % TILE_C;
    }

	// Tensor Contraction : [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'd', 'e', 'c']], [16, 'STR_SD2_V2_H7', '', 'v2', ['e', 'b', 'd']], '-']
	for(int iter_d = 0; iter_d < size_d; iter_d += TILE_D)
	{
		for(int l = -((int)(PIPELINE_STAGES - 1) * TILE_UNIT); l < size_internal; l += TILE_UNIT)
		{
			// Load
			int load_idx = l + (PIPELINE_STAGES - 1) * TILE_UNIT;

			if(load_idx < size_internal)
			{
				int next_shm_offset = (shm_offset + PIPELINE_STAGES - 1) % PIPELINE_STAGES;
				int load_ub = max(0, (load_idx + TILE_UNIT - size_internal));

				pipeline.producer_acquire();
				load_from_GM_4(dev_v2, dev_t2, sm_v2, sm_t2,
						size_a, size_b, size_c,
						size_d, size_e,
						blk_idx_a, blk_idx_b, blk_idx_c,
						pipeline, thread,
						next_shm_offset * v2_tile_size, next_shm_offset * t2_tile_size, load_idx, iter_d,
						load_ub,
						rng_b, rng_c);
				pipeline.producer_commit();
			}

			// Compute
			if(l >= 0)
			{
				int compute_ub = max(0, (l + TILE_UNIT - size_internal));

				pipeline.consumer_wait();
				compute_MMA_4(sm_v2, sm_t2, wmiter, wniter, wrow, wcol,
						v2_frag_cnt, t2_frag_cnt,
						t3_frag,
						shm_offset * v2_tile_size, shm_offset * t2_tile_size,
						compute_ub,
						rng_b, rng_c);
				pipeline.consumer_release();
			}

			// Update shared memory offset
			shm_offset = (shm_offset + 1) % PIPELINE_STAGES;
		}

		// Update shared memory offset
		shm_offset = (shm_offset + 1) % PIPELINE_STAGES;
	}

	// Store result into global memory
	int frag_idx = 0;
	#pragma unroll
	for(int iter_v2 = 0; iter_v2 < wmiter; iter_v2++)
	{
		int base = t3_base_thread + iter_v2 * inter_m_stride;
		int g_m_iter = (wrow + iter_v2) * TILE_B1;
		#pragma unroll
		for(int iter_t2 = 0; iter_t2 < wniter; iter_t2++)
		{
			int base_iter = base + iter_t2 * inter_n_stride;
			int g_n_iter = wcol + iter_t2;
			#pragma unroll
			for(int cnt_v2 = 0; cnt_v2 < v2_frag_cnt; cnt_v2++)
			{
				int m_base = g_m_iter + (cnt_v2 << 3);
				int base_v2 = base_iter + (cnt_v2 << 3) * intra_stride;
				#pragma unroll
				for(int cnt_t2 = 0; cnt_t2 < t2_frag_cnt; cnt_t2++, frag_idx++)
				{
					int n_base = (cnt_t2 << 3);
					int dst = base_v2 + (cnt_t2 << 3);
					bool full = ((m_base + 8 <= rng_b) && (g_n_iter < rng_c) && (((uintptr_t)dst & 0x1F) == 0) && ((intra_stride & 1) == 0));
					if(full)
					{
						nvcuda::wmma::store_matrix_sync(&dev_t3[dst], t3_frag[frag_idx], intra_stride, nvcuda::wmma::mem_row_major);
					}
					else
					{
						scatter_store_tail0(&dev_t3[dst], m_base, g_n_iter, rng_b, rng_c, intra_stride, t3_frag[frag_idx]);
					}
				}
			}
		}
	}
}

extern "C" __global__ void kernel_5(double *dev_t3, double *__restrict dev_v2, double *__restrict dev_t2, 
int size_a, int size_b, int size_c, int size_d, int size_e, 
int numBlk_a, int numBlk_b, int numBlk_c, 
int size_internal, int v2_tile_size, int t2_tile_size)
{
	// For Pipeline
	auto thread = cooperative_groups::this_thread();
	auto block = cooperative_groups::this_thread_block();

	#pragma nv_diag_suppress static_var_with_dynamic_init
	__shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, PIPELINE_STAGES> shared_state;
	auto pipeline = cuda::make_pipeline(block, &shared_state);

	// For Shared Memory
	extern __shared__ __align__(128) double shared_memory[];
	double *sm_v2 = shared_memory;
	double *sm_t2 = shared_memory + (PIPELINE_STAGES * v2_tile_size);

	// Index for each Thread Blocks
	int tmp_blkIdx;
	const int blk_idx_c = blockIdx.x / (numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_b * numBlk_a);

	const int blk_idx_b = tmp_blkIdx / (numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	const int blk_idx_a = tmp_blkIdx;

	// Warp related variables
	const int wmiter = TILE_B2 / 1;
	const int wniter = TILE_C / 4;
	const int wrow = ((threadIdx.x >> 5) / 4) * wmiter;
	const int wcol = ((threadIdx.x >> 5) % 4) * wniter;

	// Fragment Allocation
	const int v2_frag_cnt = TILE_B1 >> 3;
	const int t2_frag_cnt = TILE_A >> 3;

	// Size of fragment stride
	const int inter_m_stride = TILE_B1 * size_a;
	const int inter_n_stride = size_b * size_a;
	const int intra_stride = size_a;

	// Base index of each warps for result tensor
	const int t3_base_thread = (blk_idx_a * TILE_A + (blk_idx_b * TILE_B + (blk_idx_c * TILE_C) * size_b) * size_a)
								+ wrow * inter_m_stride + wcol * inter_n_stride;

	// WMMA fragment
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> t3_frag[wmiter * wniter * v2_frag_cnt * t2_frag_cnt];

	// Fill fragment
	#pragma unroll
	for(int i = 0; i < wmiter * wniter * v2_frag_cnt * t2_frag_cnt; i++)
	{
		nvcuda::wmma::fill_fragment(t3_frag[i], 0.0);
	}

	// Shared memory offset for pipelining
	int shm_offset = 0;

	// Kernel 5
	// Frag_mapped : Partial, Reg_mapped : Full / Internal : Full
	int rng_b, rng_a;
	if((size_b - (blk_idx_b * TILE_B)) >= TILE_B)
    {
        rng_b = TILE_B;
    }
    else
    {
        rng_b = size_b % TILE_B;
    }

	if((size_a - (blk_idx_a * TILE_A)) >= TILE_A)
    {
        rng_a = TILE_A;
    }
    else
    {
        rng_a = size_a % TILE_A;
    }

	// Tensor Contraction : [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'd', 'e', 'c']], [16, 'STR_SD2_V2_H7', '', 'v2', ['e', 'b', 'd']], '-']
	for(int iter_d = 0; iter_d < size_d; iter_d += TILE_D)
	{
		for(int l = -((int)(PIPELINE_STAGES - 1) * TILE_UNIT); l < size_internal; l += TILE_UNIT)
		{
			// Load
			int load_idx = l + (PIPELINE_STAGES - 1) * TILE_UNIT;

			if(load_idx < size_internal)
			{
				int next_shm_offset = (shm_offset + PIPELINE_STAGES - 1) % PIPELINE_STAGES;

				pipeline.producer_acquire();
				load_from_GM_5(dev_v2, dev_t2, sm_v2, sm_t2,
						size_a, size_b, size_c,
						size_d, size_e,
						blk_idx_a, blk_idx_b, blk_idx_c,
						pipeline, thread,
						next_shm_offset * v2_tile_size, next_shm_offset * t2_tile_size, load_idx, iter_d,
						rng_b, rng_a);
				pipeline.producer_commit();
			}

			// Compute
			if(l >= 0)
			{
				pipeline.consumer_wait();
				compute_MMA_5(sm_v2, sm_t2, wmiter, wniter, wrow, wcol,
						v2_frag_cnt, t2_frag_cnt,
						t3_frag,
						shm_offset * v2_tile_size, shm_offset * t2_tile_size,
						rng_b, rng_a);
				pipeline.consumer_release();
			}

			// Update shared memory offset
			shm_offset = (shm_offset + 1) % PIPELINE_STAGES;
		}

		// Update shared memory offset
		shm_offset = (shm_offset + 1) % PIPELINE_STAGES;
	}

	// Store result into global memory
	int frag_idx = 0;
	#pragma unroll
	for(int iter_v2 = 0; iter_v2 < wmiter; iter_v2++)
	{
		int base = t3_base_thread + iter_v2 * inter_m_stride;
		int g_m_iter = (wrow + iter_v2) * TILE_B1;
		#pragma unroll
		for(int iter_t2 = 0; iter_t2 < wniter; iter_t2++)
		{
			int base_iter = base + iter_t2 * inter_n_stride;
			#pragma unroll
			for(int cnt_v2 = 0; cnt_v2 < v2_frag_cnt; cnt_v2++)
			{
				int m_base = g_m_iter + (cnt_v2 << 3);
				int base_v2 = base_iter + (cnt_v2 << 3) * intra_stride;
				#pragma unroll
				for(int cnt_t2 = 0; cnt_t2 < t2_frag_cnt; cnt_t2++, frag_idx++)
				{
					int n_base = (cnt_t2 << 3);
					int dst = base_v2 + (cnt_t2 << 3);
					bool full = ((m_base + 8 <= rng_b) && (n_base + 8 <= rng_a) && (((uintptr_t)dst & 0x1F) == 0) && ((intra_stride & 1) == 0));
					if(full)
					{
						nvcuda::wmma::store_matrix_sync(&dev_t3[dst], t3_frag[frag_idx], intra_stride, nvcuda::wmma::mem_row_major);
					}
					else
					{
						scatter_store_tail1(&dev_t3[dst], m_base, n_base, rng_b, rng_a, intra_stride, t3_frag[frag_idx]);
					}
				}
			}
		}
	}
}

extern "C" __global__ void kernel_6(double *dev_t3, double *__restrict dev_v2, double *__restrict dev_t2, 
int size_a, int size_b, int size_c, int size_d, int size_e, 
int numBlk_a, int numBlk_b, int numBlk_c, 
int size_internal, int v2_tile_size, int t2_tile_size)
{
	// For Pipeline
	auto thread = cooperative_groups::this_thread();
	auto block = cooperative_groups::this_thread_block();

	#pragma nv_diag_suppress static_var_with_dynamic_init
	__shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, PIPELINE_STAGES> shared_state;
	auto pipeline = cuda::make_pipeline(block, &shared_state);

	// For Shared Memory
	extern __shared__ __align__(128) double shared_memory[];
	double *sm_v2 = shared_memory;
	double *sm_t2 = shared_memory + (PIPELINE_STAGES * v2_tile_size);

	// Index for each Thread Blocks
	int tmp_blkIdx;
	const int blk_idx_c = blockIdx.x / (numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_b * numBlk_a);

	const int blk_idx_b = tmp_blkIdx / (numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	const int blk_idx_a = tmp_blkIdx;

	// Warp related variables
	const int wmiter = TILE_B2 / 1;
	const int wniter = TILE_C / 4;
	const int wrow = ((threadIdx.x >> 5) / 4) * wmiter;
	const int wcol = ((threadIdx.x >> 5) % 4) * wniter;

	// Fragment Allocation
	const int v2_frag_cnt = TILE_B1 >> 3;
	const int t2_frag_cnt = TILE_A >> 3;

	// Size of fragment stride
	const int inter_m_stride = TILE_B1 * size_a;
	const int inter_n_stride = size_b * size_a;
	const int intra_stride = size_a;

	// Base index of each warps for result tensor
	const int t3_base_thread = (blk_idx_a * TILE_A + (blk_idx_b * TILE_B + (blk_idx_c * TILE_C) * size_b) * size_a)
								+ wrow * inter_m_stride + wcol * inter_n_stride;

	// WMMA fragment
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> t3_frag[wmiter * wniter * v2_frag_cnt * t2_frag_cnt];

	// Fill fragment
	#pragma unroll
	for(int i = 0; i < wmiter * wniter * v2_frag_cnt * t2_frag_cnt; i++)
	{
		nvcuda::wmma::fill_fragment(t3_frag[i], 0.0);
	}

	// Shared memory offset for pipelining
	int shm_offset = 0;

	// Kernel 6
	// Frag_mapped : Partial, Reg_mapped : Full / Internal : Partial
	int iter_e_offset = 0;
	int rng_b, rng_a;
	if((size_b - (blk_idx_b * TILE_B)) >= TILE_B)
    {
        rng_b = TILE_B;
    }
    else
    {
        rng_b = size_b % TILE_B;
    }

	if((size_a - (blk_idx_a * TILE_A)) >= TILE_A)
    {
        rng_a = TILE_A;
    }
    else
    {
        rng_a = size_a % TILE_A;
    }

	// Tensor Contraction : [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'd', 'e', 'c']], [16, 'STR_SD2_V2_H7', '', 'v2', ['e', 'b', 'd']], '-']
	for(int iter_d = 0; iter_d < size_d; iter_d += TILE_D)
	{
		for(int l = -((int)(PIPELINE_STAGES - 1) * TILE_UNIT); l < size_internal; l += TILE_UNIT)
		{
			// Load
			int load_idx = l + (PIPELINE_STAGES - 1) * TILE_UNIT;

			if(load_idx < size_internal)
			{
				int next_shm_offset = (shm_offset + PIPELINE_STAGES - 1) % PIPELINE_STAGES;
				int load_ub = max(0, (load_idx + TILE_UNIT - size_internal));

				pipeline.producer_acquire();
				load_from_GM_6(dev_v2, dev_t2, sm_v2, sm_t2,
						size_a, size_b, size_c,
						size_d, size_e,
						blk_idx_a, blk_idx_b, blk_idx_c,
						pipeline, thread,
						next_shm_offset * v2_tile_size, next_shm_offset * t2_tile_size, load_idx, iter_d,
						rng_b, rng_a,
						load_ub);
				pipeline.producer_commit();
			}

			// Compute
			if(l >= 0)
			{
				int compute_ub = max(0, (l + TILE_UNIT - size_internal));

				pipeline.consumer_wait();
				compute_MMA_6(sm_v2, sm_t2, wmiter, wniter, wrow, wcol,
						v2_frag_cnt, t2_frag_cnt,
						t3_frag,
						shm_offset * v2_tile_size, shm_offset * t2_tile_size,
						rng_b, rng_a,
						compute_ub);
				pipeline.consumer_release();
			}

			// Update shared memory offset
			shm_offset = (shm_offset + 1) % PIPELINE_STAGES;
		}

		// Update shared memory offset
		shm_offset = (shm_offset + 1) % PIPELINE_STAGES;
	}

	// Store result into global memory
	int frag_idx = 0;
	#pragma unroll
	for(int iter_v2 = 0; iter_v2 < wmiter; iter_v2++)
	{
		int base = t3_base_thread + iter_v2 * inter_m_stride;
		int g_m_iter = (wrow + iter_v2) * TILE_B1;
		#pragma unroll
		for(int iter_t2 = 0; iter_t2 < wniter; iter_t2++)
		{
			int base_iter = base + iter_t2 * inter_n_stride;
			#pragma unroll
			for(int cnt_v2 = 0; cnt_v2 < v2_frag_cnt; cnt_v2++)
			{
				int m_base = g_m_iter + (cnt_v2 << 3);
				int base_v2 = base_iter + (cnt_v2 << 3) * intra_stride;
				#pragma unroll
				for(int cnt_t2 = 0; cnt_t2 < t2_frag_cnt; cnt_t2++, frag_idx++)
				{
					int n_base = (cnt_t2 << 3);
					int dst = base_v2 + (cnt_t2 << 3);
					bool full = ((m_base + 8 <= rng_b) && (n_base + 8 <= rng_a) && (((uintptr_t)dst & 0x1F) == 0) && ((intra_stride & 1) == 0));
					if(full)
					{
						nvcuda::wmma::store_matrix_sync(&dev_t3[dst], t3_frag[frag_idx], intra_stride, nvcuda::wmma::mem_row_major);
					}
					else
					{
						scatter_store_tail1(&dev_t3[dst], m_base, n_base, rng_b, rng_a, intra_stride, t3_frag[frag_idx]);
					}
				}
			}
		}
	}
}

extern "C" __global__ void kernel_7(double *dev_t3, double *__restrict dev_v2, double *__restrict dev_t2, 
int size_a, int size_b, int size_c, int size_d, int size_e, 
int numBlk_a, int numBlk_b, int numBlk_c, 
int size_internal, int v2_tile_size, int t2_tile_size)
{
	// For Pipeline
	auto thread = cooperative_groups::this_thread();
	auto block = cooperative_groups::this_thread_block();

	#pragma nv_diag_suppress static_var_with_dynamic_init
	__shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, PIPELINE_STAGES> shared_state;
	auto pipeline = cuda::make_pipeline(block, &shared_state);

	// For Shared Memory
	extern __shared__ __align__(128) double shared_memory[];
	double *sm_v2 = shared_memory;
	double *sm_t2 = shared_memory + (PIPELINE_STAGES * v2_tile_size);

	// Index for each Thread Blocks
	int tmp_blkIdx;
	const int blk_idx_c = blockIdx.x / (numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_b * numBlk_a);

	const int blk_idx_b = tmp_blkIdx / (numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	const int blk_idx_a = tmp_blkIdx;

	// Warp related variables
	const int wmiter = TILE_B2 / 1;
	const int wniter = TILE_C / 4;
	const int wrow = ((threadIdx.x >> 5) / 4) * wmiter;
	const int wcol = ((threadIdx.x >> 5) % 4) * wniter;

	// Fragment Allocation
	const int v2_frag_cnt = TILE_B1 >> 3;
	const int t2_frag_cnt = TILE_A >> 3;

	// Size of fragment stride
	const int inter_m_stride = TILE_B1 * size_a;
	const int inter_n_stride = size_b * size_a;
	const int intra_stride = size_a;

	// Base index of each warps for result tensor
	const int t3_base_thread = (blk_idx_a * TILE_A + (blk_idx_b * TILE_B + (blk_idx_c * TILE_C) * size_b) * size_a)
								+ wrow * inter_m_stride + wcol * inter_n_stride;

	// WMMA fragment
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> t3_frag[wmiter * wniter * v2_frag_cnt * t2_frag_cnt];

	// Fill fragment
	#pragma unroll
	for(int i = 0; i < wmiter * wniter * v2_frag_cnt * t2_frag_cnt; i++)
	{
		nvcuda::wmma::fill_fragment(t3_frag[i], 0.0);
	}

	// Shared memory offset for pipelining
	int shm_offset = 0;

	// Kernel 7
	// Frag_mapped : Partial, Reg_mapped : Partial / Internal : Full
	int rng_b, rng_c;
	if((size_b - (blk_idx_b * TILE_B)) >= TILE_B)
    {
        rng_b = TILE_B;
    }
    else
    {
        rng_b = size_b % TILE_B;
    }

	if((size_c - (blk_idx_c * TILE_C)) >= TILE_C)
    {
        rng_c = TILE_C;
    }
    else
    {
        rng_c = size_c % TILE_C;
    }

	int rng_a;
	if((size_a - (blk_idx_a * TILE_A)) >= TILE_A)
    {
        rng_a = TILE_A;
    }
    else
    {
        rng_a = size_a % TILE_A;
    }

	// Tensor Contraction : [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'd', 'e', 'c']], [16, 'STR_SD2_V2_H7', '', 'v2', ['e', 'b', 'd']], '-']
	for(int iter_d = 0; iter_d < size_d; iter_d += TILE_D)
	{
		for(int l = -((int)(PIPELINE_STAGES - 1) * TILE_UNIT); l < size_internal; l += TILE_UNIT)
		{
			// Load
			int load_idx = l + (PIPELINE_STAGES - 1) * TILE_UNIT;

			if(load_idx < size_internal)
			{
				int next_shm_offset = (shm_offset + PIPELINE_STAGES - 1) % PIPELINE_STAGES;

				pipeline.producer_acquire();
				load_from_GM_7(dev_v2, dev_t2, sm_v2, sm_t2,
						size_a, size_b, size_c,
						size_d, size_e,
						blk_idx_a, blk_idx_b, blk_idx_c,
						pipeline, thread,
						next_shm_offset * v2_tile_size, next_shm_offset * t2_tile_size, load_idx, iter_d,
						rng_b, rng_a,
						rng_c);
				pipeline.producer_commit();
			}

			// Compute
			if(l >= 0)
			{
				pipeline.consumer_wait();
				compute_MMA_7(sm_v2, sm_t2, wmiter, wniter, wrow, wcol,
						v2_frag_cnt, t2_frag_cnt,
						t3_frag,
						shm_offset * v2_tile_size, shm_offset * t2_tile_size,
						rng_b, rng_a,
						rng_c);
				pipeline.consumer_release();
			}

			// Update shared memory offset
			shm_offset = (shm_offset + 1) % PIPELINE_STAGES;
		}

		// Update shared memory offset
		shm_offset = (shm_offset + 1) % PIPELINE_STAGES;
	}

	// Store result into global memory
	int frag_idx = 0;
	#pragma unroll
	for(int iter_v2 = 0; iter_v2 < wmiter; iter_v2++)
	{
		int base = t3_base_thread + iter_v2 * inter_m_stride;
		int g_m_iter = (wrow + iter_v2) * TILE_B1;
		#pragma unroll
		for(int iter_t2 = 0; iter_t2 < wniter; iter_t2++)
		{
			int base_iter = base + iter_t2 * inter_n_stride;
			int g_n_iter = wcol + iter_t2;
			#pragma unroll
			for(int cnt_v2 = 0; cnt_v2 < v2_frag_cnt; cnt_v2++)
			{
				int m_base = g_m_iter + (cnt_v2 << 3);
				int base_v2 = base_iter + (cnt_v2 << 3) * intra_stride;
				#pragma unroll
				for(int cnt_t2 = 0; cnt_t2 < t2_frag_cnt; cnt_t2++, frag_idx++)
				{
					int n_base = (cnt_t2 << 3);
					int dst = base_v2 + (cnt_t2 << 3);
					bool full = ((m_base + 8 <= rng_b) && (g_n_iter < rng_c) && (n_base + 8 <= rng_a) && (((uintptr_t)dst & 0x1F) == 0) && ((intra_stride & 1) == 0));
					if(full)
					{
						nvcuda::wmma::store_matrix_sync(&dev_t3[dst], t3_frag[frag_idx], intra_stride, nvcuda::wmma::mem_row_major);
					}
					else
					{
						scatter_store_tail2(&dev_t3[dst], m_base, g_n_iter, n_base, rng_b, rng_c, rng_a, intra_stride, t3_frag[frag_idx]);
					}
				}
			}
		}
	}
}

extern "C" __global__ void kernel_8(double *dev_t3, double *__restrict dev_v2, double *__restrict dev_t2, 
int size_a, int size_b, int size_c, int size_d, int size_e, 
int numBlk_a, int numBlk_b, int numBlk_c, 
int size_internal, int v2_tile_size, int t2_tile_size)
{
	// For Pipeline
	auto thread = cooperative_groups::this_thread();
	auto block = cooperative_groups::this_thread_block();

	#pragma nv_diag_suppress static_var_with_dynamic_init
	__shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, PIPELINE_STAGES> shared_state;
	auto pipeline = cuda::make_pipeline(block, &shared_state);

	// For Shared Memory
	extern __shared__ __align__(128) double shared_memory[];
	double *sm_v2 = shared_memory;
	double *sm_t2 = shared_memory + (PIPELINE_STAGES * v2_tile_size);

	// Index for each Thread Blocks
	int tmp_blkIdx;
	const int blk_idx_c = blockIdx.x / (numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_b * numBlk_a);

	const int blk_idx_b = tmp_blkIdx / (numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	const int blk_idx_a = tmp_blkIdx;

	// Warp related variables
	const int wmiter = TILE_B2 / 1;
	const int wniter = TILE_C / 4;
	const int wrow = ((threadIdx.x >> 5) / 4) * wmiter;
	const int wcol = ((threadIdx.x >> 5) % 4) * wniter;

	// Fragment Allocation
	const int v2_frag_cnt = TILE_B1 >> 3;
	const int t2_frag_cnt = TILE_A >> 3;

	// Size of fragment stride
	const int inter_m_stride = TILE_B1 * size_a;
	const int inter_n_stride = size_b * size_a;
	const int intra_stride = size_a;

	// Base index of each warps for result tensor
	const int t3_base_thread = (blk_idx_a * TILE_A + (blk_idx_b * TILE_B + (blk_idx_c * TILE_C) * size_b) * size_a)
								+ wrow * inter_m_stride + wcol * inter_n_stride;

	// WMMA fragment
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> t3_frag[wmiter * wniter * v2_frag_cnt * t2_frag_cnt];

	// Fill fragment
	#pragma unroll
	for(int i = 0; i < wmiter * wniter * v2_frag_cnt * t2_frag_cnt; i++)
	{
		nvcuda::wmma::fill_fragment(t3_frag[i], 0.0);
	}

	// Shared memory offset for pipelining
	int shm_offset = 0;

	// Kernel 8
	// Frag_mapped : Partial, Reg_mapped : Partial / Internal : Partial
	int iter_e_offset = 0;
	int rng_b, rng_c;
	if((size_b - (blk_idx_b * TILE_B)) >= TILE_B)
    {
        rng_b = TILE_B;
    }
    else
    {
        rng_b = size_b % TILE_B;
    }

	if((size_c - (blk_idx_c * TILE_C)) >= TILE_C)
    {
        rng_c = TILE_C;
    }
    else
    {
        rng_c = size_c % TILE_C;
    }

	int rng_a;
	if((size_a - (blk_idx_a * TILE_A)) >= TILE_A)
    {
        rng_a = TILE_A;
    }
    else
    {
        rng_a = size_a % TILE_A;
    }

	// Tensor Contraction : [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'd', 'e', 'c']], [16, 'STR_SD2_V2_H7', '', 'v2', ['e', 'b', 'd']], '-']
	for(int iter_d = 0; iter_d < size_d; iter_d += TILE_D)
	{
		for(int l = -((int)(PIPELINE_STAGES - 1) * TILE_UNIT); l < size_internal; l += TILE_UNIT)
		{
			// Load
			int load_idx = l + (PIPELINE_STAGES - 1) * TILE_UNIT;

			if(load_idx < size_internal)
			{
				int next_shm_offset = (shm_offset + PIPELINE_STAGES - 1) % PIPELINE_STAGES;
				int load_ub = max(0, (load_idx + TILE_UNIT - size_internal));

				pipeline.producer_acquire();
				load_from_GM_8(dev_v2, dev_t2, sm_v2, sm_t2,
						size_a, size_b, size_c,
						size_d, size_e,
						blk_idx_a, blk_idx_b, blk_idx_c,
						pipeline, thread,
						next_shm_offset * v2_tile_size, next_shm_offset * t2_tile_size, load_idx, iter_d,
						rng_b, rng_a,
						load_ub,
						rng_c);
				pipeline.producer_commit();
			}

			// Compute
			if(l >= 0)
			{
				int compute_ub = max(0, (l + TILE_UNIT - size_internal));

				pipeline.consumer_wait();
				compute_MMA_8(sm_v2, sm_t2, wmiter, wniter, wrow, wcol,
						v2_frag_cnt, t2_frag_cnt,
						t3_frag,
						shm_offset * v2_tile_size, shm_offset * t2_tile_size,
						rng_b, rng_a,
						compute_ub,
						rng_c);
				pipeline.consumer_release();
			}

			// Update shared memory offset
			shm_offset = (shm_offset + 1) % PIPELINE_STAGES;
		}

		// Update shared memory offset
		shm_offset = (shm_offset + 1) % PIPELINE_STAGES;
	}

	// Store result into global memory
	int frag_idx = 0;
	#pragma unroll
	for(int iter_v2 = 0; iter_v2 < wmiter; iter_v2++)
	{
		int base = t3_base_thread + iter_v2 * inter_m_stride;
		int g_m_iter = (wrow + iter_v2) * TILE_B1;
		#pragma unroll
		for(int iter_t2 = 0; iter_t2 < wniter; iter_t2++)
		{
			int base_iter = base + iter_t2 * inter_n_stride;
			int g_n_iter = wcol + iter_t2;
			#pragma unroll
			for(int cnt_v2 = 0; cnt_v2 < v2_frag_cnt; cnt_v2++)
			{
				int m_base = g_m_iter + (cnt_v2 << 3);
				int base_v2 = base_iter + (cnt_v2 << 3) * intra_stride;
				#pragma unroll
				for(int cnt_t2 = 0; cnt_t2 < t2_frag_cnt; cnt_t2++, frag_idx++)
				{
					int n_base = (cnt_t2 << 3);
					int dst = base_v2 + (cnt_t2 << 3);
					bool full = ((m_base + 8 <= rng_b) && (g_n_iter < rng_c) && (n_base + 8 <= rng_a) && (((uintptr_t)dst & 0x1F) == 0) && ((intra_stride & 1) == 0));
					if(full)
					{
						nvcuda::wmma::store_matrix_sync(&dev_t3[dst], t3_frag[frag_idx], intra_stride, nvcuda::wmma::mem_row_major);
					}
					else
					{
						scatter_store_tail2(&dev_t3[dst], m_base, g_n_iter, n_base, rng_b, rng_c, rng_a, intra_stride, t3_frag[frag_idx]);
					}
				}
			}
		}
	}
}

