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

void initialize_particles(Particle swarm[]) {
    for (int i = 0; i < SWARM_SIZE; i++) {
        for (int d = 0; d < DIMENSIONS; d++) {
            swarm[i].position[d] = X_MIN + (double)rand() / RAND_MAX * (X_MAX - X_MIN);
            swarm[i].velocity[d] = 0.5 * V_MAX * ((double)rand() / RAND_MAX * 2 - 1);
            swarm[i].personal_best_position[d] = swarm[i].position[d];
        }
        swarm[i].personal_best_value = fitness_function(swarm[i].personal_best_position);
    }
}

void update_particles(Particle swarm[], double global_best_position[]) {
    const double w = 0.5;
    const double c1 = 2.0;
    const double c2 = 2.0;

    for (int i = 0; i < SWARM_SIZE; i++) {
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

int main() {
    srand(time(NULL));
    Particle swarm[SWARM_SIZE];
    double global_best_position[DIMENSIONS];
    double global_best_value = INFINITY;

    initialize_particles(swarm);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        for (int i = 0; i < SWARM_SIZE; i++) {
            if (swarm[i].personal_best_value < global_best_value) {
                global_best_value = swarm[i].personal_best_value;
                for (int d = 0; d < DIMENSIONS; d++) {
                    global_best_position[d] = swarm[i].personal_best_position[d];
                }
            }
        }

        update_particles(swarm, global_best_position);

        printf("Iteration %d: Best Value = %f\n", iter, global_best_value);
    }

    return 0;
}
