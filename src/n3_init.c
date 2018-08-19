#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "n3_header.h"
#include "n3_logger.h"
#include "n3_act.h"

#define N3L_SET_NEURON_ACTS(n,_act,_act_prime) \
  (n).act = &(_act); \
  (n).act_prime = &(_act_prime)

void n3l_build_network(N3LData *state, FILE *of);
void n3l_build_layer(N3LData *s, FILE *of, uint64_t l_idx, N3LActType act);
void n3l_build_bias(N3LData *s, uint64_t l_idx, uint64_t n_idx);

double n3l_rnd_weight(N3LLayer l_ref)
{
  return ((double) rand()) / (RAND_MAX + 1.f);
}

N3LArgs n3l_get_default_args(void)
{
  N3LArgs args;

  args.read_file = false;
  args.in_filename = NULL;
  args.bias = 0.f;
  args.learning_rate = 1.f;
  args.in_size = 0;
  args.h_size = 0;
  args.h_layers = 0;
  args.out_size = 0;
  args.logger = NULL;
  args.act_in = N3LNone;
  args.act_h = N3LSigmoid;
  args.act_out = N3LSigmoid;

  return args;
}

N3LData *n3l_build(N3LArgs args, N3L_RND_WEIGHT(rnd_w))
{
  FILE *of = NULL;
  N3LData *n3_state = NULL;

  N3L_LCRITICAL_START(args.logger);

  n3_state = (N3LData *) malloc(sizeof(N3LData));
  n3_state->inputs = NULL;
  n3_state->outputs = NULL;
  n3_state->targets = NULL;
  n3_state->get_rnd_weight = rnd_w;

  N3L_LMEDIUM(args.logger, "Initializing arguments");
  n3_state->args = (N3LArgs *) malloc(sizeof(N3LArgs));
  memcpy(n3_state->args, &args, sizeof(N3LArgs));

  ++n3_state->args->in_size;
  ++n3_state->args->h_size;

  if ( n3_state->args->read_file ) {
    if ( n3_state->args->in_filename ) {
      if ( (of = fopen(n3_state->args->in_filename, "r")) ) {
        N3L_LMEDIUM(args.logger, "Initializing network from file.");

        fread(&(n3_state->args->in_size), sizeof(uint64_t), 1, of);
        fread(&(n3_state->args->h_size), sizeof(uint64_t), 1, of);
        fread(&(n3_state->args->out_size), sizeof(uint64_t), 1, of);
        fread(&(n3_state->args->h_layers), sizeof(uint64_t), 1, of);
        fread(&(n3_state->args->bias), sizeof(double), 1, of);
        fread(&(n3_state->args->act_in), sizeof(N3LActType), 1, of);
        fread(&(n3_state->args->act_h), sizeof(N3LActType), 1, of);
        fread(&(n3_state->args->act_out), sizeof(N3LActType), 1, of);
      }
      else {
        N3L_LCRITICAL(args.logger, "File opening failed. Initializing network with args provided.");
      }
    }
    else {
      N3L_LCRITICAL(args.logger, "Input file not specified. Initializing network with args provided.");
    }
  }

  n3l_build_network(n3_state, of);
  if ( of ) {
    fclose(of);
  }

  N3L_LCRITICAL_END(args.logger);
  return n3_state;
}

void n3l_build_network(N3LData *state, FILE *of)
{
  uint64_t layers = 2 + state->args->h_layers;
  uint64_t l_idx;
  N3L_LHIGH_START(state->args->logger);

  state->net = (N3LLayer *) malloc(layers * sizeof(N3LLayer));
  for ( l_idx = 0; l_idx < layers; ++l_idx ) {
    N3L_LHIGH(state->args->logger, "Building layer %d", l_idx);

    if ( !l_idx ) {
      N3L_LMEDIUM(state->args->logger, "Type: Input Layer - Size: %d", state->args->in_size);
      state->net[l_idx].size = state->args->in_size;
      state->net[l_idx].ltype = N3LInputLayer;
      n3l_build_layer(state, of, l_idx, state->args->act_in);
    }
    else if ( l_idx == (layers - 1) ) {
      N3L_LMEDIUM(state->args->logger, "Type: Output Layer - Size: %d", state->args->out_size);
      state->net[l_idx].size = state->args->out_size;
      state->net[l_idx].ltype = N3LOutputLayer;
      n3l_build_layer(state, of, l_idx, state->args->act_out);
    }
    else {
      N3L_LMEDIUM(state->args->logger, "Type: Hidden Layer - Size: %d", state->args->h_size);
      state->net[l_idx].size = state->args->h_size;
      state->net[l_idx].ltype = N3LHiddenLayer;
      n3l_build_layer(state, of, l_idx, state->args->act_h);
    }

    if ( l_idx != (layers - 1) ) {
      n3l_build_bias(state, l_idx, state->net[l_idx].size - 1);
    }
  }

  N3L_LHIGH_END(state->args->logger);
}

void n3l_build_layer(N3LData *s, FILE *of, uint64_t l_idx, N3LActType act)
{
  uint64_t n_idx, w_idx;
  N3L_LMEDIUM_START(s->args->logger);

  s->net[l_idx].neurons = (N3LNeuron *) malloc(s->net[l_idx].size * sizeof(N3LNeuron));
  for ( n_idx = 0; n_idx < s->net[l_idx].size; ++n_idx ) {
    N3L_LMEDIUM(s->args->logger, "Building neuron %d", n_idx);
    switch(act) {
      case N3LCustom:
        N3L_LLOW(s->args->logger, "Activation: Custom");
        N3L_LPEDANTIC(s->args->logger, "Note: At this point the activation function will be None.");
      case N3LNone:
        N3L_LLOW(s->args->logger, "Activation: None");
        N3L_SET_NEURON_ACTS(s->net[l_idx].neurons[n_idx], n3l_act_none, n3l_act_none);
        break;
      case N3LSigmoid:
        N3L_LLOW(s->args->logger, "Activation: Sigmoid");
        N3L_SET_NEURON_ACTS(s->net[l_idx].neurons[n_idx], n3l_act_sigmoid, n3l_act_sigmoid_prime);
        break;
      case N3LRelu:
        N3L_LLOW(s->args->logger, "Activation: ReLU");
        N3L_SET_NEURON_ACTS(s->net[l_idx].neurons[n_idx], n3l_act_relu, n3l_act_relu_prime);
        break;
      case N3LTanh:
        N3L_LLOW(s->args->logger, "Activation: Tanh");
        N3L_SET_NEURON_ACTS(s->net[l_idx].neurons[n_idx], n3l_act_tanh, n3l_act_tanh_prime);
        break;
      case N3LIdentity:
        N3L_LLOW(s->args->logger, "Activation: Identity");
        N3L_SET_NEURON_ACTS(s->net[l_idx].neurons[n_idx], n3l_act_identity, n3l_act_identity_prime);
        break;
      case N3LLeakyRelu:
        N3L_LLOW(s->args->logger, "Activation: Leaky ReLU");
        N3L_SET_NEURON_ACTS(s->net[l_idx].neurons[n_idx], n3l_act_leaky_relu, n3l_act_leaky_relu_prime);
        break;
      case N3LSoftPlus:
        N3L_LLOW(s->args->logger, "Activation: SoftPlus");
        N3L_SET_NEURON_ACTS(s->net[l_idx].neurons[n_idx], n3l_act_softplus, n3l_act_softplus_prime);
        break;
      case N3LSoftSign:
        N3L_LLOW(s->args->logger, "Activation: SoftSign");
        N3L_SET_NEURON_ACTS(s->net[l_idx].neurons[n_idx], n3l_act_softsign, n3l_act_softsign_prime);
        break;
      case N3LSwish:
        N3L_LLOW(s->args->logger, "Activation: Swish");
        N3L_SET_NEURON_ACTS(s->net[l_idx].neurons[n_idx], n3l_act_swish, n3l_act_swish_prime);
        break;
    }

    switch(s->net[l_idx].ltype) {
      case N3LInputLayer:
        if ( s->args->h_layers ) {
          s->net[l_idx].neurons[n_idx].outputs = s->args->h_size;
        }
        else {
          s->net[l_idx].neurons[n_idx].outputs = s->args->out_size;
        }
        break;
      case N3LHiddenLayer:
        if ( l_idx == s->args->h_layers ) {
          s->net[l_idx].neurons[n_idx].outputs = s->args->out_size;
        }
        else {
          s->net[l_idx].neurons[n_idx].outputs = s->args->h_size;
        }
        break;
      case N3LOutputLayer:
        s->net[l_idx].neurons[n_idx].outputs = 1;
        break;
    }

    if ( s->net[l_idx].ltype != N3LOutputLayer ) {
      s->net[l_idx].neurons[n_idx].weights =
        (double *) malloc(s->net[l_idx].neurons[n_idx].outputs * sizeof(double));

      for ( w_idx = 0; w_idx < s->net[l_idx].neurons[n_idx].outputs; ++w_idx ) {
        if ( of ) {
          fread(&(s->net[l_idx].neurons[n_idx].weights[w_idx]), sizeof(double), 1, of);
        }
        else {
          s->net[l_idx].neurons[n_idx].weights[w_idx] = s->get_rnd_weight(s->net[l_idx]);
        }
        N3L_LLOW(s->args->logger, "Weight(%d,%d): %lf [ --> (%d,%d) ]", l_idx, n_idx,
          s->net[l_idx].neurons[n_idx].weights[w_idx], l_idx + 1, w_idx);
      }
    }
  }

  N3L_LMEDIUM_END(s->args->logger);
}

void n3l_build_bias(N3LData *s, uint64_t l_idx, uint64_t n_idx)
{
  N3L_LMEDIUM_START(s->args->logger);

  N3L_LMEDIUM(s->args->logger, "Building bias neuron (%ld,%ld)", l_idx, n_idx);
  N3L_SET_NEURON_ACTS(s->net[l_idx].neurons[n_idx], n3l_act_none, n3l_act_none);
  s->net[l_idx].neurons[n_idx].input = s->args->bias;

  N3L_LMEDIUM_END(s->args->logger);
}

void n3l_set_custom_act(N3LData *state, uint64_t l_idx, N3L_ACT(act), N3L_ACT(act_prime))
{
  uint64_t n_idx;

  N3L_LHIGH_START(state->args->logger);

  N3L_LHIGH(state->args->logger, "Setting custom activation functions at layer %ld.", l_idx);

  for ( n_idx = 0; n_idx < state->net[l_idx].size; ++n_idx ) {
    state->net[l_idx].neurons[n_idx].act = act;
    state->net[l_idx].neurons[n_idx].act_prime = act_prime;
  }

  switch(state->net[l_idx].ltype) {
    case N3LInputLayer:
      N3L_LLOW(state->args->logger, "Changing Input layer activation reference.");
      state->args->act_in = N3LCustom;
      break;
    case N3LHiddenLayer:
      N3L_LLOW(state->args->logger, "Changing Hidden layer activation reference.");
      state->args->act_h = N3LCustom;
      break;
    case N3LOutputLayer:
      N3L_LLOW(state->args->logger, "Changing Output layer activation reference.");
      state->args->act_out = N3LCustom;
      break;
  }

  N3L_LHIGH_END(state->args->logger);
}
