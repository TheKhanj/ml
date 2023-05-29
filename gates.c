#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define model_size (3)

int32_t or_train[][model_size] = {
		{0, 0, 0},
		{0, 1, 0},
		{1, 0, 0},
		{1, 1, 1},
};

#define or_train_count (sizeof(or_train) / sizeof(or_train[0]))

float rand_float(void) { return (float)rand() / (float)RAND_MAX; }

float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

float forward(int model[model_size - 1], float *w, float bias) {
	float ret = bias;
	for (size_t i = 0; i < model_size - 1; i++) {
		ret += w[i] * model[i];
	}

	return sigmoidf(ret);
}

float cost(float *w, float bias) {
	float ret = 0.0f;

	for (size_t i = 0; i < or_train_count; i++) {
		float y = forward(or_train[i], w, bias);
		float value = or_train[i][2];

		float diff = y - value;

		ret += diff * diff;
	}
	return ret / or_train_count;
}

void print(float *w, float bias) {
	for (size_t i = 0; i < 2; i++) {
		printf("w%lu: %f ", i, w[i]);
	}

	printf("b: %f, c: %f\n", bias, cost(w, bias));
}

int32_t main() {
	srand(time(0));

	float bias = rand_float();
	float w[2] = {rand_float(), rand_float()};

	float eps = 1;
	float rate = 1;

	for (int i = 0; i < 10000000; i++) {
		float initial_cost = cost(w, bias);
		float dw[2];

		for (size_t i = 0; i < 2; i++) {
			w[i] += eps;
			dw[i] = (cost(w, bias) - initial_cost) / eps;
			w[i] -= eps;
		}

		float d_bias = (cost(w, bias + eps) - initial_cost) / eps;

		for (size_t i = 0; i < 2; i++) {
			w[i] -= dw[i] * rate;
		}
		bias -= rate * d_bias;
	}
	print(w, bias);

	for (size_t i = 0; i < 2; i++) {
		for (size_t j = 0; j < 2; j++) {

			int foo[] = {i, j};
			printf("%zu & %zu = %f\n", i, j, forward(foo, w, bias));
		}
	}
}
