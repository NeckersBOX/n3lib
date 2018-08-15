#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "n3_header.h"
#include "n3_logger.h"

struct _n3l_out_to_in {
  N3LData *state;
  uint64_t l_idx;
  uint64_t o_idx;
  double delta_w;
};

void *n3l_execute_backward_propagation(void *arg);

double n3l_evaluate_out_error(N3LData *state, uint64_t o_idx)
{
  double delta_E;
  double diff;
  N3LNeuron *p_n;

  N3L_LLOW_START(state->args->logger);
  p_n = &(state->net[state->args->h_layers + 1].neurons[o_idx]);

  diff = (state->targets[o_idx] - state->outputs[o_idx]);
  N3L_LLOW(state->args->logger, "Output %ld diff: %lf", o_idx, diff);

  delta_E = diff * p_n->act_prime(p_n->result);
  N3L_LLOW(state->args->logger, "Output %ld delta: %lf", o_idx, delta_E);

  N3L_LLOW_END(state->args->logger);
  return delta_E;
}

void n3l_backward_propagation(N3LData *state)
{
  pthread_t *threads;
  struct _n3l_out_to_in *tdata = NULL;
  uint64_t layers = 2 + state->args->h_layers;
  uint64_t o_idx;
  N3L_LCRITICAL_START(state->args->logger);

  threads = (pthread_t *) malloc(state->args->out_size * sizeof(pthread_t));
  tdata = (struct _n3l_out_to_in *) malloc(state->args->out_size * sizeof(struct _n3l_out_to_in));
  for ( o_idx = 0; o_idx < state->args->out_size; ++o_idx ) {
    tdata[o_idx].state = state;
    tdata[o_idx].l_idx = state->args->h_layers;
    tdata[o_idx].o_idx = o_idx;
    tdata[o_idx].delta_w = n3l_evaluate_out_error(state, o_idx);

    N3L_LHIGH(state->args->logger, "Starting thread to execute backward propagation "
      "from neuron (%ld,%ld) to layer %ld.", layers - 1, o_idx, state->args->h_layers);
    pthread_create(&threads[o_idx], NULL, n3l_execute_backward_propagation, (void *) &(tdata[o_idx]));
  }

  for ( o_idx = 0; o_idx < state->args->out_size; ++o_idx ) {
    pthread_join(threads[o_idx], NULL);
  }

  free(threads);
  free(tdata);

  N3L_LCRITICAL_END(state->args->logger);
}

void *n3l_execute_backward_propagation(void *arg)
{
  pthread_t thread;
  struct _n3l_out_to_in *tdata = (struct _n3l_out_to_in *) arg;
  struct _n3l_out_to_in next_tdata;
  N3LNeuron *p_n;
  N3LLogger *p_l = tdata->state->args->logger;
  uint64_t n_idx;
  N3L_LMEDIUM_START(p_l);

  N3L_LLOW(p_l, "Current layer index: %ld", tdata->l_idx);
  N3L_LMEDIUM(p_l, "Executing backward propagation with delta: %lf", tdata->delta_w);

  for ( n_idx = 0; n_idx < tdata->state->net[tdata->l_idx].size; ++n_idx ) {
    p_n = &tdata->state->net[tdata->l_idx].neurons[n_idx];
    if ( tdata->state->net[tdata->l_idx].ltype != N3LInputLayer ) {
      next_tdata.state = tdata->state;
      next_tdata.l_idx = tdata->l_idx - 1;
      next_tdata.o_idx = n_idx;
      next_tdata.delta_w = tdata->delta_w * p_n->weights[tdata->o_idx] * p_n->act_prime(p_n->result);

      N3L_LLOW(p_l, "Neuron(%ld,%ld) - Propagate new delta to previous layer.", tdata->l_idx, n_idx);
      pthread_create(&thread, NULL, n3l_execute_backward_propagation, (void *) &(next_tdata));
      pthread_join(thread, NULL);
    }

    N3L_LMEDIUM(p_l, "Weight (%ld,%ld) --> (%ld,%ld) - Old: %lf",
      tdata->l_idx, n_idx, tdata->l_idx + 1, tdata->o_idx, p_n->weights[tdata->o_idx]);

    p_n->weights[tdata->o_idx] += tdata->state->args->learning_rate * tdata->delta_w * p_n->result;

    N3L_LMEDIUM(p_l, "Weight (%ld,%ld) --> (%ld,%ld) - New: %lf",
      tdata->l_idx, n_idx, tdata->l_idx + 1, tdata->o_idx, p_n->weights[tdata->o_idx]);
  }

  N3L_LMEDIUM_END(p_l);
}
