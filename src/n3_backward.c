#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "n3_header.h"
#include "n3_logger.h"

struct _n3l_out_to_in {
  N3LData *state;
  uint64_t dst_l_idx;
  uint64_t o_idx;
  double delta_w;
};

void *n3l_execute_backward_propagation(void *arg);

double n3l_evaluate_out_error(N3LData *state, uint64_t out_idx)
{
  double d;
  N3L_LLOW_START(state->args->logger);

  d = (state->targets[out_idx] - state->outputs[out_idx]);
  N3L_LLOW(state->args->logger, "Output %ld diff: %lf", out_idx, d);

  d *= state->net[state->args->h_layers + 1].neurons[out_idx].act_prime(
    state->net[state->args->h_layers + 1].neurons[out_idx].result
  );
  N3L_LLOW(state->args->logger, "Output %ld delta: %lf", out_idx, d);

  N3L_LLOW_END(state->args->logger);
  return d;
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
    tdata[o_idx].dst_l_idx = state->args->h_layers;
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
  uint64_t src_n_idx;
  N3L_LMEDIUM_START(tdata->state->args->logger);

  N3L_LMEDIUM(tdata->state->args->logger, "Current layer index: %ld", tdata->dst_l_idx);
  N3L_LMEDIUM(tdata->state->args->logger, "Executing backward propagation with delta: %lf", tdata->delta_w);

  for ( src_n_idx = 0; src_n_idx < tdata->state->net[tdata->dst_l_idx].size; ++src_n_idx ) {
    if ( tdata->state->net[tdata->dst_l_idx].ltype != N3LInputLayer ) {
      next_tdata.state = tdata->state;
      next_tdata.dst_l_idx = tdata->dst_l_idx - 1;
      next_tdata.o_idx = src_n_idx;
      next_tdata.delta_w = tdata->delta_w
        * tdata->state->net[tdata->dst_l_idx].neurons[src_n_idx].weights[tdata->o_idx]
        * tdata->state->net[tdata->dst_l_idx].neurons[src_n_idx].act_prime(
          tdata->state->net[tdata->dst_l_idx].neurons[src_n_idx].result
        );

      N3L_LLOW(tdata->state->args->logger,
        "Neuron(%ld,%ld) - Propagate new delta to previous layer.",
        tdata->dst_l_idx, src_n_idx);

      pthread_create(&thread, NULL, n3l_execute_backward_propagation, (void *) &(next_tdata));
      pthread_join(thread, NULL);
    }

    N3L_LPEDANTIC(tdata->state->args->logger, "Weight (%ld,%ld) - Old: %lf", tdata->dst_l_idx, src_n_idx,
      tdata->state->net[tdata->dst_l_idx].neurons[src_n_idx].weights[tdata->o_idx]);

    tdata->state->net[tdata->dst_l_idx].neurons[src_n_idx].weights[tdata->o_idx] +=
      tdata->state->args->learning_rate * tdata->delta_w *
      tdata->state->net[tdata->dst_l_idx].neurons[src_n_idx].input;

    N3L_LMEDIUM(tdata->state->args->logger, "Weight (%ld,%ld) - New: %lf", tdata->dst_l_idx, src_n_idx,
      tdata->state->net[tdata->dst_l_idx].neurons[src_n_idx].weights[tdata->o_idx]);
  }

  N3L_LMEDIUM_END(tdata->state->args->logger);
}
