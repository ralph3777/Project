#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define SWARM_SIZE 30
#define MAX_ITER 100
#define DIMENSIONS 2
#define X_MAX 100
#define X_MIN -100
#define V_MAX 20
#define COMM_INTERVAL 10  // Update the global best every 5 iterations

typedef struct {
    double position[DIMENSIONS];
    double velocity[DIMENSIONS];
    double personal_best_position[DIMENSIONS];
    double personal_best_value;
} Particle;

double fitness_function(double position[]) {
    return position[0] * position[0] + position[1] * position[1];
}

void initialize_particles(Particle swarm[], int num_particles) {
    for (int i = 0; i < num_particles; i++) {
        for (int d = 0; d < DIMENSIONS; d++) {
            swarm[i].position[d] = X_MIN + (double)rand() / RAND_MAX * (X_MAX - X_MIN);
            swarm[i].velocity[d] = 0.5 * V_MAX * ((double)rand() / RAND_MAX * 2 - 1);
            swarm[i].personal_best_position[d] = swarm[i].position[d];
        }
        swarm[i].personal_best_value = fitness_function(swarm[i].personal_best_position);
    }
}

void update_particles(Particle swarm[], double global_best_position[], int num_particles) {
    const double w = 0.5;
    const double c1 = 2.0;
    const double c2 = 2.0;

    for (int i = 0; i < num_particles; i++) {
        for (int d = 0; d < DIMENSIONS; d++) {
            double r1 = (double)rand() / RAND_MAX;
            double r2 = (double)rand() / RAND_MAX;
            swarm[i].velocity[d] = w * swarm[i].velocity[d]
                + c1 * r1 * (swarm[i].personal_best_position[d] - swarm[i].position[d])
                + c2 * r2 * (global_best_position[d] - swarm[i].position[d]);

            if (swarm[i].velocity[d] > V_MAX) swarm[i].velocity[d] = V_MAX;
            if (swarm[i].velocity[d] < -V_MAX) swarm[i].velocity[d] = -V_MAX;

            swarm[i].position[d] += swarm[i].velocity[d];
            if (swarm[i].position[d] > X_MAX) swarm[i].position[d] = X_MAX;
            if (swarm[i].position[d] < X_MIN) swarm[i].position[d] = X_MIN;
        }

        double current_value = fitness_function(swarm[i].position);
        if (current_value < swarm[i].personal_best_value) {
            swarm[i].personal_best_value = current_value;
            for (int d = 0; d < DIMENSIONS; d++) {
                swarm[i].personal_best_position[d] = swarm[i].position[d];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    srand(time(NULL) + world_rank);

    int particles_per_proc = SWARM_SIZE / world_size + (world_rank < SWARM_SIZE % world_size ? 1 : 0);
    Particle local_swarm[particles_per_proc];
    double global_best_position[DIMENSIONS];
    double global_best_value = INFINITY;

    double start_time = MPI_Wtime();

    initialize_particles(local_swarm, particles_per_proc);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        update_particles(local_swarm, global_best_position, particles_per_proc);

        if (iter % COMM_INTERVAL == 0 || iter == MAX_ITER - 1) {
            double local_best_value = INFINITY;
            double local_best_position[DIMENSIONS];
            for (int i = 0; i < particles_per_proc; i++) {
                if (local_swarm[i].personal_best_value < local_best_value) {
                    local_best_value = local_swarm[i].personal_best_value;
                    memcpy(local_best_position, local_swarm[i].personal_best_position, DIMENSIONS * sizeof(double));
                }
            }

            double new_global_best_value;
            MPI_Allreduce(&local_best_value, &new_global_best_value, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

            if (new_global_best_value < global_best_value) {
                global_best_value = new_global_best_value;
                MPI_Bcast(local_best_position, DIMENSIONS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                memcpy(global_best_position, local_best_position, DIMENSIONS * sizeof(double));
            }
        }

        if (world_rank == 0 && (iter % COMM_INTERVAL == 0 || iter == MAX_ITER - 1)) {
            printf("Iteration %d: Best Value = %f\n", iter, global_best_value);
        }
    }

    double end_time = MPI_Wtime();

    if (world_rank == 0) {
        printf("Elapsed time: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
