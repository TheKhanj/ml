#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int32_t train[][2] = {
		{1, 2}, {2, 4}, {3, 6}, {4, 8}, {5, 10}, {6, 12},
};

#define train_count (sizeof(train) / sizeof(train[0]))

float rand_float(void) { return (float)rand() / (float)RAND_MAX; }

float cost(float w, float b) {
	float ret = 0.0f;

	for (size_t i = 0; i < train_count; i++) {
		float x = train[i][0];
		float y = x * w + b;
		float value = train[i][1];

		float diff = y - value;

		ret += diff * diff;
	}
	return ret / train_count;
}

int32_t main() {
	srand(time(0));

	float b = rand_float();
	float w = rand_float();

	float eps = 1e-5;
	float rate = 1e-5;

	for (int i = 0; i < 1000000; i++) {
		float initial_cost = cost(w, b);

		float dw = (cost(w + eps, b) - initial_cost) / eps;
		float db = (cost(w + b, b) - initial_cost) / b;

		w -= dw * rate;
		b -= db * rate;
	}

	printf("w: %f, b: %f, c: %f\n", w, b, cost(w, b));
}
