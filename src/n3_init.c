#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "n3_header.h"
#include "n3_logger.h"
#include "n3_act.h"

void n3l_build_network(N3LData *state, FILE *of, N3LActType act_h, N3LActType act_o);
void n3l_build_layer(N3LData *s, FILE *of, uint64_t l_idx, N3LActType act);

double n3l_rnd_weight(N3LLayer l_ref)
{
  return ((double) rand()) / (RAND_MAX + 1.f);
}

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

N3LData *n3l_build(N3LArgs args, N3L_RND_WEIGHT(rnd_w), N3LActType act_h, N3LActType act_o)
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

  if ( n3_state->args->read_file ) {
    if ( n3_state->args->in_filename ) {
      if ( (of = fopen(n3_state->args->in_filename, "r")) ) {
        N3L_LMEDIUM(args.logger, "Initializing network from file.");

        fread(&(n3_state->args->in_size), sizeof(uint64_t), 1, of);
        fread(&(n3_state->args->h_size), sizeof(uint64_t), 1, of);
        fread(&(n3_state->args->out_size), sizeof(uint64_t), 1, of);
        fread(&(n3_state->args->h_layers), sizeof(uint64_t), 1, of);
      }
      else {
        N3L_LCRITICAL(args.logger, "File opening failed. Initializing network with args provided.");
      }
    }
    else {
      N3L_LCRITICAL(args.logger, "Input file not specified. Initializing network with args provided.");
    }
  }

  n3l_build_network(n3_state, of, act_h, act_o);
  if ( of ) {
    fclose(of);
  }

  N3L_LCRITICAL_END(args.logger);
  return n3_state;
}

void n3l_build_network(N3LData *state, FILE *of, N3LActType act_h, N3LActType act_o)
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
      n3l_build_layer(state, of, l_idx, N3LNone);
    }
    else if ( l_idx == (layers - 1) ) {
      N3L_LMEDIUM(state->args->logger, "Type: Output Layer - Size: %d", state->args->out_size);
      state->net[l_idx].size = state->args->out_size;
      state->net[l_idx].ltype = N3LOutputLayer;
      n3l_build_layer(state, of, l_idx, act_o);
    }
    else {
      N3L_LMEDIUM(state->args->logger, "Type: Hidden Layer - Size: %d", state->args->h_size);
      state->net[l_idx].size = state->args->h_size;
      state->net[l_idx].ltype = N3LHiddenLayer;
      n3l_build_layer(state, of, l_idx, act_h);
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
      case N3LNone:
        s->net[l_idx].neurons[n_idx].act = &n3l_act_none;
        s->net[l_idx].neurons[n_idx].act_prime = &n3l_act_none;
        break;
      case N3LSigmoid:
        s->net[l_idx].neurons[n_idx].act = &n3l_act_sigmoid;
        s->net[l_idx].neurons[n_idx].act_prime = &n3l_act_sigmoid_prime;
        break;
      case N3LTanh:
        s->net[l_idx].neurons[n_idx].act = &n3l_act_tanh;
        s->net[l_idx].neurons[n_idx].act_prime = &n3l_act_tanh_prime;
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
