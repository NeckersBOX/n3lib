#include <stdio.h>
#include <stdint.h>
#include "n3_header.h"
#include "n3_logger.h"

void n3l_save(N3LData *state, FILE *of)
{
  uint64_t layers = state->args->h_layers + 1, l_idx, n_idx;
  N3L_LHIGH_START(state->args->logger);

  N3L_LHIGH(state->args->logger, "Writing header info..");
  fwrite(&(state->args->in_size), sizeof(uint64_t), 1, of);
  fwrite(&(state->args->h_size), sizeof(uint64_t), 1, of);
  fwrite(&(state->args->out_size), sizeof(uint64_t), 1, of);
  fwrite(&(state->args->h_layers), sizeof(uint64_t), 1, of);
  fwrite(&(state->args->bias), sizeof(double), 1, of);
  fwrite(&(state->args->act_in), sizeof(N3LActType), 1, of);
  fwrite(&(state->args->act_h), sizeof(N3LActType), 1, of);
  fwrite(&(state->args->act_out), sizeof(N3LActType), 1, of);

  for ( l_idx = 0; l_idx < layers; ++l_idx ) {
    N3L_LHIGH(state->args->logger, "Writing layer %ld weights.", l_idx);
    for ( n_idx = 0; n_idx < state->net[l_idx].size; ++n_idx ) {
      N3L_LLOW(state->args->logger, "Writing neuron %ld weights.", n_idx);
      fwrite(state->net[l_idx].neurons[n_idx].weights, sizeof(double),
        state->net[l_idx].neurons[n_idx].outputs, of);
    }
  }

  N3L_LHIGH_END(state->args->logger);
}
