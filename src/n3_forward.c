#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include "n3_header.h"
#include "n3_logger.h"

struct _n3l_in_to_out {
  uint64_t dst_idx;
  N3LLayer *l_ref;
  double *result;
  N3LLogger *logger;
};

void *    n3l_collect_outputs (void *arg);
void *    n3l_execute_neuron  (void *arg);
double *  n3l_forward_layer   (N3LData *, uint64_t, double *);

double *n3l_forward_propagation(N3LData *state)
{
  uint64_t layers = state->args->h_layers + 2;
  double *layer_in_data = NULL, *layer_out_data = NULL;
  uint64_t l_idx;

  N3L_LCRITICAL_START(state->args->logger);

  for ( l_idx = 0; l_idx < layers; ++l_idx ) {
    layer_out_data = n3l_forward_layer(state, l_idx, layer_in_data);

    if ( layer_in_data ) {
      free(layer_in_data);
    }

    layer_in_data = layer_out_data;
  }

  N3L_LCRITICAL_END(state->args->logger);

  return layer_out_data;
}

double *n3l_forward_layer(N3LData *state, uint64_t l_idx, double *in_data)
{
  uint64_t n_idx, next_layer_size;
  pthread_t *threads;
  double *out_data = NULL;
  struct _n3l_in_to_out *tdata = NULL;

  N3L_LHIGH_START(state->args->logger);

  threads = (pthread_t *) malloc(state->net[l_idx].size * sizeof(pthread_t));
  for ( n_idx = 0; n_idx < state->net[l_idx].size; ++n_idx ) {
    if ( state->net[l_idx].ltype == N3LOutputLayer || n_idx != (state->net[l_idx].size - 1) ) {
      if ( state->net[l_idx].ltype == N3LInputLayer ) {
        state->net[l_idx].neurons[n_idx].input = state->inputs[n_idx];
      }
      else {
        state->net[l_idx].neurons[n_idx].input = in_data[n_idx];
      }
    }
    else {
      N3L_LMEDIUM(state->args->logger, "Current neuron type: Bias");
    }

    N3L_LHIGH(state->args->logger, "Starting thread to execute neuron %ld from layer %ld.", n_idx, l_idx);
    pthread_create(&threads[n_idx], NULL, n3l_execute_neuron, (void *) &(state->net[l_idx].neurons[n_idx]));
  }

  for ( n_idx = 0; n_idx < state->net[l_idx].size; ++n_idx ) {
    pthread_join(threads[n_idx], NULL);
  }
  free(threads);

  N3L_LHIGH(state->args->logger, "End of neurons execution. Collecting outputs.");
  if ( state->net[l_idx].ltype != N3LOutputLayer ) {
    next_layer_size = state->net[l_idx + 1].size;
    N3L_LMEDIUM(state->args->logger, "Next layer size: %ld", next_layer_size);

    threads = (pthread_t *) malloc(next_layer_size * sizeof(pthread_t));
    out_data = (double *) malloc(next_layer_size * sizeof(double));
    tdata = (struct _n3l_in_to_out *) malloc(next_layer_size * sizeof(struct _n3l_in_to_out));
    for ( n_idx = 0; n_idx < next_layer_size; ++n_idx ) {
      tdata[n_idx].dst_idx = n_idx;
      tdata[n_idx].l_ref = &(state->net[l_idx]);
      tdata[n_idx].result = &(out_data[n_idx]);
      tdata[n_idx].logger = state->args->logger;

      N3L_LHIGH(state->args->logger, "Starting thread to collect outputs to neuron %ld.", n_idx);
      pthread_create(&threads[n_idx], NULL, n3l_collect_outputs, (void *) &(tdata[n_idx]));
    }

    for ( n_idx = 0; n_idx < next_layer_size; ++n_idx ) {
      pthread_join(threads[n_idx], NULL);
    }

    free(threads);
    free(tdata);
  }
  else {
    out_data = (double *) malloc(state->net[l_idx].size * sizeof(double));
    for ( n_idx = 0; n_idx < state->net[l_idx].size; ++n_idx ) {
      out_data[n_idx] = state->net[l_idx].neurons[n_idx].result;
      N3L_LHIGH(state->args->logger, "Output %ld: %lf", n_idx, out_data[n_idx]);
    }
  }

  N3L_LHIGH_END(state->args->logger);

  return out_data;
}

void *n3l_execute_neuron(void *arg)
{
  N3LNeuron *ref = (N3LNeuron *) arg;

  ref->result = ref->act(ref->input);
  return NULL;
}

void *n3l_collect_outputs(void *arg)
{
  struct _n3l_in_to_out *ref = (struct _n3l_in_to_out *) arg;
  uint64_t n_idx;

  *(ref->result) = 0;
  for ( n_idx = 0; n_idx < ref->l_ref->size; ++n_idx ) {
    N3L_LPEDANTIC(ref->logger, "[Thread] Result from neuron %ld: %lf", n_idx, ref->l_ref->neurons[n_idx].result);
    *(ref->result) += ref->l_ref->neurons[n_idx].result * ref->l_ref->neurons[n_idx].weights[ref->dst_idx];
  }

  N3L_LLOW(ref->logger, "[Thread] Input to neuron %ld: %lf", ref->dst_idx, *(ref->result));
  return NULL;
}
