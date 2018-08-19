#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "n3_header.h"
#include "n3_logger.h"

void n3l_free(N3LData *state)
{
  N3LLogger *p_l = state->args->logger;
  uint64_t l_idx, n_idx, layers = state->args->h_layers + 2;

  N3L_LLOW_START(p_l);

  free(state->args);
  for( l_idx = 0; l_idx < layers; ++l_idx ) {
    if ( state->net[l_idx].ltype != N3LOutputLayer ) {
      for ( n_idx = 0; n_idx < state->net[l_idx].size; ++n_idx ) {
        free(state->net[l_idx].neurons[n_idx].weights);
      }
    }
    free(state->net[l_idx].neurons);
  }
  free(state->net);
  free(state);

  N3L_LLOW_END(p_l);
}

N3LData *n3l_clone(N3LData *net)
{
  N3LData *clone;
  uint64_t layers = 2 + net->args->h_layers;
  uint64_t l_idx, n_idx, w_idx;

  N3L_LHIGH_START(net->args->logger);

  N3L_LMEDIUM(net->args->logger, "Cloning attributes");
  clone = (N3LData *) malloc(sizeof(N3LData));
  clone->inputs = net->inputs;
  clone->outputs = net->outputs;
  clone->targets = net->targets;
  clone->get_rnd_weight = net->get_rnd_weight;

  N3L_LMEDIUM(net->args->logger, "Cloning arguments");
  clone->args = (N3LArgs *) malloc(sizeof(N3LArgs));
  memcpy(clone->args, net->args, sizeof(N3LArgs));

  clone->net = (N3LLayer *) malloc(layers * sizeof(N3LLayer));
  for ( l_idx = 0; l_idx < layers; ++l_idx ) {
    N3L_LMEDIUM(net->args->logger, "Cloning layer %ld", l_idx);
    clone->net[l_idx].size = net->net[l_idx].size;
    clone->net[l_idx].ltype = net->net[l_idx].ltype;

    clone->net[l_idx].neurons = (N3LNeuron *) malloc(net->net[l_idx].size * sizeof(N3LNeuron));
    for ( n_idx = 0; n_idx < net->net[l_idx].size; ++n_idx ) {
      N3L_LMEDIUM(net->args->logger, "Cloning neuron %ld", n_idx);
      clone->net[l_idx].neurons[n_idx].input = net->net[l_idx].neurons[n_idx].input;
      clone->net[l_idx].neurons[n_idx].outputs = net->net[l_idx].neurons[n_idx].outputs;
      clone->net[l_idx].neurons[n_idx].result = net->net[l_idx].neurons[n_idx].result;
      clone->net[l_idx].neurons[n_idx].act = net->net[l_idx].neurons[n_idx].act;
      clone->net[l_idx].neurons[n_idx].act_prime = net->net[l_idx].neurons[n_idx].act_prime;
      if ( net->net[l_idx].neurons[n_idx].weights != NULL ) {
        clone->net[l_idx].neurons[n_idx].weights = (double *) malloc(
          net->net[l_idx].neurons[n_idx].outputs * sizeof(double));
        memcpy(clone->net[l_idx].neurons[n_idx].weights, net->net[l_idx].neurons[n_idx].weights,
          net->net[l_idx].neurons[n_idx].outputs * sizeof(double));
      }
      else {
        clone->net[l_idx].neurons[n_idx].weights = NULL;
      }
    }
  }

  N3L_LHIGH_END(net->args->logger);
  return clone;
}
