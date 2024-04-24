#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define SWARM_SIZE 30
#define MAX_ITER 100
#define DIMENSIONS 2
#define X_MAX 100
#define X_MIN -100
#define V_MAX 20

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
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL) + rank);
    int num_particles = SWARM_SIZE / size;

    Particle swarm[num_particles];
    double global_best_position[DIMENSIONS];
    double global_best_value = INFINITY;
    double local_best_position[DIMENSIONS];
    double local_best_value = INFINITY;

    initialize_particles(swarm, num_particles);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        for (int i = 0; i < num_particles; i++) {
            if (swarm[i].personal_best_value < local_best_value) {
                local_best_value = swarm[i].personal_best_value;
                for (int d = 0; d < DIMENSIONS; d++) {
                    local_best_position[d] = swarm[i].personal_best_position[d];
                }
            }
        }

        MPI_Allreduce(&local_best_value, &global_best_value, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        double temp_best_position[DIMENSIONS];
        MPI_Allreduce(local_best_position, temp_best_position, DIMENSIONS, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (local_best_value == global_best_value) {
            for (int d = 0; d < DIMENSIONS; d++) {
                global_best_position[d] = temp_best_position[d];
            }
        }

        update_particles(swarm, global_best_position, num_particles);
        if (rank == 0) {
            printf("Iteration %d: Best Value = %f\n", iter, global_best_value);
        }
    }

    MPI_Finalize();
    return 0;
}
