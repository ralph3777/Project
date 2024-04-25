#include <curand_kernel.h>
#include <stdio.h>
#include <float.h>
#include <cuda_runtime.h>

#define SWARM_SIZE 256
#define MAX_ITER 100
#define DIMENSIONS 2
#define X_MAX 100.0f
#define X_MIN -100.0f
#define V_MAX 20.0f
#define BLOCK_SIZE 256

struct Particle {
    float position[DIMENSIONS];
    float velocity[DIMENSIONS];
    float personal_best_position[DIMENSIONS];
    float personal_best_value;
};

__device__ float fitness_function(float *position) {
    float sum = 0.0f;
    for (int i = 0; i < DIMENSIONS; i++) {
        sum += position[i] * position[i];
    }
    return sum;
}

__global__ void initialize_particles(Particle *swarm, curandState *states, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= SWARM_SIZE) return;

    curand_init(seed + idx, 0, 0, &states[idx]);

    for (int d = 0; d < DIMENSIONS; d++) {
        swarm[idx].position[d] = X_MIN + curand_uniform(&states[idx]) * (X_MAX - X_MIN);
        swarm[idx].velocity[d] = 0.5f * V_MAX * (curand_uniform(&states[idx]) * 2.0f - 1.0f);
        swarm[idx].personal_best_position[d] = swarm[idx].position[d];
    }
    swarm[idx].personal_best_value = fitness_function(swarm[idx].personal_best_position);
}

__global__ void update_particles(Particle *swarm, float *global_best_position, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= SWARM_SIZE) return;

    float r1, r2;
    for (int d = 0; d < DIMENSIONS; d++) {
        r1 = curand_uniform(&states[idx]);
        r2 = curand_uniform(&states[idx]);

        swarm[idx].velocity[d] = 0.5f * swarm[idx].velocity[d] +
                                 2.0f * r1 * (swarm[idx].personal_best_position[d] - swarm[idx].position[d]) +
                                 2.0f * r2 * (global_best_position[d] - swarm[idx].position[d]);

        if (swarm[idx].velocity[d] > V_MAX) swarm[idx].velocity[d] = V_MAX;
        if (swarm[idx].velocity[d] < -V_MAX) swarm[idx].velocity[d] = -V_MAX;

        swarm[idx].position[d] += swarm[idx].velocity[d];
        if (swarm[idx].position[d] > X_MAX) swarm[idx].position[d] = X_MAX;
        if (swarm[idx].position[d] < X_MIN) swarm[idx].position[d] = X_MIN;
    }

    float current_value = fitness_function(swarm[idx].position);
    if (current_value < swarm[idx].personal_best_value) {
        swarm[idx].personal_best_value = current_value;
        for (int d = 0; d < DIMENSIONS; d++) {
            swarm[idx].personal_best_position[d] = swarm[idx].position[d];
        }
    }
}

__device__ void atomicMin(float *address, float val) {
    int *address_as_i = (int *)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

__global__ void find_global_best(Particle *swarm, float *global_best_position, float *global_best_value) {
    __shared__ float best_values[BLOCK_SIZE];
    __shared__ int best_indices[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    best_values[tid] = FLT_MAX;
    best_indices[tid] = idx;

    if (idx < SWARM_SIZE) {
        best_values[tid] = swarm[idx].personal_best_value;
        best_indices[tid] = idx;
    }
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset && (idx + offset) < SWARM_SIZE) {
            if (best_values[tid + offset] < best_values[tid]) {
                best_values[tid] = best_values[tid + offset];
                best_indices[tid] = best_indices[tid + offset];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMin(global_best_value, best_values[0]);
        if (*global_best_value == best_values[0]) {
            int best_idx = best_indices[0];
            for (int d = 0; d < DIMENSIONS; d++) {
                global_best_position[d] = swarm[best_idx].personal_best_position[d];
            }
        }
    }
}

int main() {
    Particle *d_swarm;
    float *d_global_best_position;
    float h_global_best_position[DIMENSIONS];
    float global_best_value = FLT_MAX, *d_global_best_value;
    curandState *d_states;

    cudaMalloc(&d_swarm, SWARM_SIZE * sizeof(Particle));
    cudaMalloc(&d_global_best_position, DIMENSIONS * sizeof(float));
    cudaMalloc(&d_global_best_value, sizeof(float));
    cudaMalloc(&d_states, SWARM_SIZE * sizeof(curandState));

    cudaMemcpy(d_global_best_value, &global_best_value, sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = (SWARM_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    initialize_particles<<<numBlocks, BLOCK_SIZE>>>(d_swarm, d_states, time(NULL));

    for (int iter = 0; iter < MAX_ITER; iter++) {
        update_particles<<<numBlocks, BLOCK_SIZE>>>(d_swarm, d_global_best_position, d_states);
        find_global_best<<<numBlocks, BLOCK_SIZE>>>(d_swarm, d_global_best_position, d_global_best_value);

        cudaDeviceSynchronize();

        cudaMemcpy(&global_best_value, d_global_best_value, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_global_best_position, d_global_best_position, DIMENSIONS * sizeof(float), cudaMemcpyDeviceToHost);

        printf("Iteration %d: Best Value = %f\n", iter, global_best_value);
        printf("Best Position: [");
        for (int i = 0; i < DIMENSIONS; i++) {
            printf("%f ", h_global_best_position[i]);
        }
        printf("]\n");
    }

    cudaFree(d_swarm);
    cudaFree(d_global_best_position);
    cudaFree(d_global_best_value);
    cudaFree(d_states);

    return 0;
}
