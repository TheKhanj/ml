#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef float(activator_func)(float);

float rand_float(void) { return (float)rand() / (float)RAND_MAX; }

typedef enum input_t {
	INPUT_TYPE_NODE = 0,
	INPUT_TYPE_FLOAT,
	INPUT_TYPE_SIZE
} node_input_type_enum;

const float CACHE_NOT_SET = -1.f;

typedef struct node_t {
	size_t index;
	struct node_t *node_inputs;
	float *float_inputs;
	node_input_type_enum input_type;
	size_t input_size;

	float *weights;
	float *d_weights;
	float bias;
	float d_bias;
	activator_func *activator;
	float cache;
} node_t;

void node_init(node_t *node, node_input_type_enum input_type,
							 node_t *node_inputs, float *float_inputs, size_t input_size,
							 activator_func *activator) {
	node->node_inputs = node_inputs;
	node->float_inputs = float_inputs;
	node->input_type = input_type;
	node->input_size = input_size;
	node->weights = malloc(input_size * sizeof(*node->weights));
	node->d_weights = malloc(input_size * sizeof(*node->weights));

	for (size_t i = 0; i < input_size; ++i) {
		node->weights[i] = rand_float();
	}

	node->d_bias = 0.f;
	node->bias = rand_float();
	node->activator = activator;
	node->cache = CACHE_NOT_SET;
}

void node_print(node_t *node) {
	printf("  {\n");
	printf("    index: %zu\n", node->index);
	printf("    input_type: %d\n", node->input_type);
	printf("    inputs: [ ");
	for (size_t i = 0; i < node->input_size; ++i) {
		node_t *input = node->node_inputs + i;
		char foo[100];
		if (node->input_type == INPUT_TYPE_FLOAT) {
			sprintf(foo, "%f", node->float_inputs[i]);
		} else {
			sprintf(foo, "<%zu>", ((node_t *)input)->index);
		}

		printf("%s%s ", foo, (i == node->input_size - 1 ? "" : ","));
	}
	printf("]\n");
	printf("    weights: [ ");
	for (size_t i = 0; i < node->input_size; ++i) {
		printf("%f%s ", node->weights[i], (i == node->input_size - 1 ? "" : ","));
	}
	printf("]\n");
	printf("    bias: %f\n", node->bias);
	printf("    cache: %f\n", node->cache);
	printf("    d_bias: %f\n", node->d_bias);
	printf("    d_weights: [ ");
	for (size_t i = 0; i < node->input_size; ++i) {
		printf("%f%s ", node->d_weights[i], (i == node->input_size - 1 ? "" : ","));
	}
	printf("]\n");
	printf("  }");
}

float node_calculate(node_t *node) {
	if (node->cache >= 0.f) {
		return node->cache;
	}

	float ret = 0;
	assert(node->input_type < INPUT_TYPE_SIZE);
	if (node->input_type == INPUT_TYPE_FLOAT) {
		assert(node->float_inputs != NULL);
	} else {
		assert(node->node_inputs != NULL);
	}

	for (size_t i = 0; i < node->input_size; ++i) {
		float prev_value = 1.f;

		if (node->input_type == INPUT_TYPE_FLOAT) {
			prev_value = node->float_inputs[i];
		} else {
			node_t *node_input = node->node_inputs + i;
			prev_value = node_calculate(node_input);
		}

		ret += node->weights[i] * prev_value;
	}

	ret += node->bias;
	node->cache = node->activator(ret);
	return node->cache;
}

float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

#define NETWORK_SIZE(net) ((net).partial_sum[(net).layers_size - 1])
#define NETWORK_NODE_IN_LAYER_AT(net, layer, index)                            \
	((net).nodes + ((layer) == 0 ? 0 : (net).partial_sum[(layer)-1]) + (index))

typedef struct traning_set_t {
	size_t size;
	size_t initials_size;
	float **initials;
	float **expected;
} training_set_t;

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
	node_t *prev_layer_nodes = NULL;

	size_t index = 0;
	node_t *node = net.nodes;

	for (size_t layer = 0; layer < net.layers_size; ++layer) {
		node_input_type_enum input_type =
				(layer == 0 ? INPUT_TYPE_FLOAT : INPUT_TYPE_NODE);

		for (size_t i = 0; i < net.layers[layer]; ++i, ++node, index++) {
			node->index = index;
			if (layer == 0) {
				node_init(node, input_type, NULL, net.initials, prev_layer_size,
									sigmoidf);
			} else {
				node_init(node, input_type, prev_layer_nodes, NULL, prev_layer_size,
									sigmoidf);
			}
		}

		prev_layer_size = net.layers[layer];
		prev_layer_nodes = node - prev_layer_size;
	}

	return net;
}

void network_set_initials(network_t *net, float *initials,
													size_t initials_size) {
	const size_t net_initials_size = net->layers[0];

	assert(initials_size == net_initials_size);
	for (size_t i = 0; i < initials_size; ++i) {
		node_t *node = net->nodes + i;
		node->float_inputs = initials;
		node->input_type = INPUT_TYPE_FLOAT;
	}
}

void network_print(network_t *net) {
	printf("size: %zu\n", NETWORK_SIZE(*net));

	printf("nodes: [\n");
	for (size_t i = 0; i < NETWORK_SIZE(*net); ++i) {
		node_t *node = net->nodes + i;

		node_print(node);
		printf(",\n");
	}
	printf("]\n");
}

static void network_clear_cache(network_t *net) {
	for (size_t i = 0; i < NETWORK_SIZE(*net); ++i) {
		node_t *node = net->nodes + i;
		node->cache = CACHE_NOT_SET;
	}
}

float network_cost(network_t *net, training_set_t *set) {
	float ret = 0;

	const size_t initials_size = set->initials_size;
	const size_t last_layer = net->layers_size - 1;
	const size_t last_layer_size = net->layers[last_layer];

	for (size_t i = 0; i < set->size; i++) {
		network_clear_cache(net);
		float *initials = set->initials[i];
		float *expecteds = set->expected[i];

		network_set_initials(net, initials, initials_size);

		for (size_t j = 0; j < last_layer_size; j++) {
			node_t *node = NETWORK_NODE_IN_LAYER_AT(*net, last_layer, j);
			float guess = node_calculate(node);
			float expected = expecteds[j];
			float diff = guess - expected;
			ret += diff * diff;
		}
	}

	return ret / (set->size * last_layer_size);
}

void node_learn(network_t *net, training_set_t *set, node_t *node, float eps,
								float initial_cost) {
	for (size_t i = 0; i < node->input_size; ++i) {
		const float initial_weight = node->weights[i];
		node->weights[i] += eps;
		node->d_weights[i] = (network_cost(net, set) - initial_cost) / eps;
		node->weights[i] = initial_weight;
	}

	const float initial_bias = node->bias;
	node->bias += eps;
	node->d_bias = (network_cost(net, set) - initial_cost) / eps;
	node->bias = initial_bias;
}

// learn :)
void network_learn(network_t *net, training_set_t *set, float rate, float eps) {
	float initial_cost = network_cost(net, set);

	for (size_t i = 0; i < NETWORK_SIZE(*net); ++i) {
		node_t *node = net->nodes + i;

		node_learn(net, set, node, eps, initial_cost);
	}

	for (size_t i = 0; i < NETWORK_SIZE(*net); ++i) {
		node_t *node = net->nodes + i;

		for (size_t j = 0; j < node->input_size; ++j) {
			node->weights[j] -= node->d_weights[j] * rate;
		}

		node->bias -= node->d_bias * rate;
	}
}

int main(void) {
	srand(time(0));
	size_t layers[] = {2, 1};
	network_t net = network_init(layers, sizeof(layers) / sizeof(layers[0]));

	float model[][3] = {
			{0.f, 0.f, 0.f},
			{1.f, 0.f, 1.f},
			{0.f, 1.f, 1.f},
			{1.f, 1.f, 0.f},
	};

	float *initials[] = {model[0], model[1], model[2], model[3]};

	// I know it's shit :) I'm noob I admit it:)
	float *expected[] = {
			&model[0][2],
			&model[1][2],
			&model[2][2],
			&model[3][2],
	};

	training_set_t set = {.initials = initials,
												.initials_size = 2,
												.expected = expected,
												.size = 4};

	const float eps = 1e-1, rate = 1e-1;

	for (size_t i = 0; i < 100000; ++i) {
		network_learn(&net, &set, rate, eps);
	}

	network_print(&net);

	node_t *last_node = NETWORK_NODE_IN_LAYER_AT(net, net.layers_size - 1, 0);
	for (size_t i = 0; i < 2; ++i) {

		for (size_t j = 0; j < 2; ++j) {
			float initials[] = {(float)i, (float)j};
			network_set_initials(&net, initials, 2);
			network_clear_cache(&net);
			printf("%zu ^ %zu: %f\n", i, j, node_calculate(last_node));
		}
	}
}
