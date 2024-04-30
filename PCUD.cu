#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <float.h>

#define SWARM_SIZE 30
#define MAX_ITER 100
#define DIMENSIONS 2
#define X_MAX 100
#define X_MIN -100
#define V_MAX 20

struct Particle {
    double position[DIMENSIONS];
    double velocity[DIMENSIONS];
    double personal_best_position[DIMENSIONS];
    double personal_best_value;
};

__device__ double fitness_function(double *position) {
    double sum = 0;
    for (int i = 0; i < DIMENSIONS; i++) {
        sum += position[i] * position[i];
    }
    return sum;
}

__global__ void init_particles(Particle *swarm, curandState *states, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= SWARM_SIZE) return;

    curand_init(seed, idx, 0, &states[idx]);
    for (int d = 0; d < DIMENSIONS; d++) {
        swarm[idx].position[d] = X_MIN + curand_uniform(&states[idx]) * (X_MAX - X_MIN);
        swarm[idx].velocity[d] = 0.5 * V_MAX * (curand_uniform(&states[idx]) * 2 - 1);
        swarm[idx].personal_best_position[d] = swarm[idx].position[d];
    }
    swarm[idx].personal_best_value = fitness_function(swarm[idx].personal_best_position);
}

__global__ void update_particles(Particle *swarm, double *global_best_position, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= SWARM_SIZE) return;

    double r1 = curand_uniform(&states[idx]);
    double r2 = curand_uniform(&states[idx]);

    for (int d = 0; d < DIMENSIONS; d++) {
        swarm[idx].velocity[d] = 0.5 * swarm[idx].velocity[d]
                                 + 2.0 * r1 * (swarm[idx].personal_best_position[d] - swarm[idx].position[d])
                                 + 2.0 * r2 * (global_best_position[d] - swarm[idx].position[d]);

        if (swarm[idx].velocity[d] > V_MAX) swarm[idx].velocity[d] = V_MAX;
        if (swarm[idx].velocity[d] < -V_MAX) swarm[idx].velocity[d] = -V_MAX;

        swarm[idx].position[d] += swarm[idx].velocity[d];
        if (swarm[idx].position[d] > X_MAX) swarm[idx].position[d] = X_MAX;
        if (swarm[idx].position[d] < X_MIN) swarm[idx].position[d] = X_MIN;
    }

    double current_value = fitness_function(swarm[idx].position);
    if (current_value < swarm[idx].personal_best_value) {
        swarm[idx].personal_best_value = current_value;
        for (int d = 0; d < DIMENSIONS; d++) {
            swarm[idx].personal_best_position[d] = swarm[idx].position[d];
        }
    }
}

int main() {
    Particle *d_swarm;
    double *d_global_best_position, global_best_position[DIMENSIONS];
    double *d_global_best_value, global_best_value = DBL_MAX;
    curandState *d_states;

    cudaMalloc(&d_swarm, sizeof(Particle) * SWARM_SIZE);
    cudaMalloc(&d_global_best_position, sizeof(double) * DIMENSIONS);
    cudaMalloc(&d_global_best_value, sizeof(double));
    cudaMalloc(&d_states, sizeof(curandState) * SWARM_SIZE);

    // Prepare CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // Start timing after initialization
    cudaEventRecord(start);

    dim3 blocks((SWARM_SIZE + 15) / 16);
    dim3 threads(16);

    init_particles<<<blocks, threads>>>(d_swarm, d_states, time(NULL));
    cudaDeviceSynchronize();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        update_particles<<<blocks, threads>>>(d_swarm, d_global_best_position, d_states);
        cudaDeviceSynchronize(); // Ensure all updates are completed
    }

    // Stop timing after computation is done
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy the global best position and value back to host
    cudaMemcpy(global_best_position, d_global_best_position, sizeof(double) * DIMENSIONS, cudaMemcpyDeviceToHost);
    cudaMemcpy(&global_best_value, d_global_best_value, sizeof(double), cudaMemcpyDeviceToHost);

    printf("Best Value = %f at position (%f, %f)\n", global_best_value, global_best_position[0], global_best_position[1]);


    // Cleanup
    cudaFree(d_swarm);
    cudaFree(d_global_best_position);
    cudaFree(d_global_best_value);
    cudaFree(d_states);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
