#include <stdio.h>
#include <stdlib.h>
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
