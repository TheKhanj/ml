#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

typedef float(activator_func)(float);

float rand_float(void) { return (float)rand() / (float)RAND_MAX; }

typedef enum input_t {
	INPUT_TYPE_NODE = 0,
	INPUT_TYPE_FLOAT,
	INPUT_TYPE_SIZE
} node_input_type_enum;

// float* | node_t*
typedef void *node_input_t;

typedef struct node_t {
	node_input_t inputs;
	node_input_type_enum input_type;
	size_t input_size;

	float *weights;
	float bias;
	activator_func *activator;
} node_t;

void node_init(node_t *node, node_input_type_enum input_type,
							 node_input_t inputs, size_t input_size,
							 activator_func *activator) {
	node->inputs = inputs;
	node->input_type = input_type;
	node->input_size = input_size;
	node->weights = malloc(input_size * sizeof(*node->weights));

	for (size_t i = 0; i < input_size; ++i) {
		node->weights[i] = rand_float();
	}

	node->bias = rand_float();
	node->activator = activator;
}

void node_print(node_t *node) {
	printf("  {\n");
	printf("    input_type: %d\n", node->input_type);
	printf("    weights: [ ");
	for (size_t i = 0; i < node->input_size; ++i) {
		printf("%f%s ", node->weights[i], (i == node->input_size - 1 ? "" : ","));
	}
	printf("]\n");
	printf("  }");
}

float node_calculate(node_t *node) {
	float ret = 0;
	assert(node->inputs != NULL);
	for (size_t i = 0; i < node->input_size; ++i) {
		node_input_t input = node->inputs + i;
		float prev_value = 1.f;

		assert(node->input_type < INPUT_TYPE_SIZE);
		if (node->input_type == INPUT_TYPE_FLOAT) {
			prev_value = *(float *)input;
		} else if (node->input_type == INPUT_TYPE_NODE) {
			prev_value = node_calculate((node_t *)input);
		}

		ret += node->weights[i] * prev_value;
	}

	ret += node->bias;
	return node->activator(ret);
}

float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

#define NETWORK_SIZE(net) ((net).partial_sum[(net).layers_size - 1])

typedef struct network_t {
	node_t *nodes;
	size_t layers_size;
	float *initials;
	size_t *layers;
	size_t *partial_sum;
} network_t;

network_t network_init(size_t *layers, size_t layers_size) {
	network_t net;

	net.partial_sum = malloc(layers_size * sizeof(*net.partial_sum));
	net.partial_sum[0] = layers[0];

	for (size_t i = 1; i < layers_size; ++i) {
		net.partial_sum[i] = net.partial_sum[i - 1] + layers[i];
	}

	const size_t sum = net.partial_sum[layers_size - 1];
	net.nodes = malloc(sum * sizeof(*net.nodes));

	assert(net.nodes != NULL);
	net.layers = layers;
	net.layers_size = layers_size;

	net.initials = malloc(net.layers[0] * sizeof(*net.initials));

	size_t prev_layer_size = net.layers[0];
	node_input_t prev_layer_nodes = net.initials;

	node_t *node = net.nodes;

	for (size_t layer = 0; layer < net.layers_size; ++layer) {
		node_input_type_enum input_type =
				(layer == 0 ? INPUT_TYPE_FLOAT : INPUT_TYPE_NODE);

		for (size_t i = 0; i < net.layers[layer]; ++i, ++node) {
			node_init(node, input_type, prev_layer_nodes, prev_layer_size, sigmoidf);
		}

		prev_layer_size = net.layers[layer];
		prev_layer_nodes = node - (prev_layer_size - 1);
	}

	return net;
}

void network_print(network_t *net)  {
	printf("size: %zu\n", NETWORK_SIZE(*net));

	printf("nodes: [\n");
	for (size_t i = 0; i < NETWORK_SIZE(*net); ++i) {
		node_t *node = net->nodes + i;

		node_print(node);
		printf(",\n");
	}
	printf("]\n");
}

int main(void) {
	size_t layers[] = {2, 1};
	network_t net = network_init(layers, sizeof(layers) / sizeof(layers[0]));
	network_print(&net);
}