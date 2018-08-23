#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "n3_header.h"
#include "n3_neuron.h"

struct __n3l_backward_data {
  uint64_t ref;
  N3LLayer *layer;
  double delta;
  double learning_rate;
};

void *__n3l_backward_execute(void *arg);

bool n3l_backward_propagation(N3LNetwork *net)
{
  pthread_t *threads;
  uint64_t nsize, j;
  N3LNeuron *out;
  struct __n3l_backward_data *tdata;

  if ( !net ) {
    return false;
  }

  if ( !(net->targets && net->ltail) ) {
    return false;
  }

  if ( !net->ltail->type != N3LOutputLayer ) {
    return false;
  }

  nsize = n3l_neuron_count(net->ltail->nhead);
  threads = (pthread_t *) malloc(nsize * sizeof(pthread_t));
  tdata = (struct __n3l_backward_data *) malloc(nsize * sizeof(struct __n3l_backward_data));
  for ( out = net->ltail->nhead, j = 0; out; out = out->next, ++j ) {
    tdata[j].ref = out->ref;
    tdata[j].layer = net->ltail->prev;
    tdata[j].delta = (net->targets[j] - out->result) * out->act_prime(out->input);
    tdata[j].learning_rate = net->learning_rate;

    pthread_create(&threads[j], NULL, __n3l_backward_execute, (void *) &(tdata[j]));
  }

  for ( j = 0; j < nsize; ++j ) {
    pthread_join(threads[j], NULL);
  }

  free(threads);
  free(tdata);

  return true;
}


void *__n3l_backward_execute(void *arg)
{
  pthread_t *threads;
  uint64_t nsize, j;
  struct __n3l_backward_data *tdata = (struct __n3l_backward_data *) arg;
  struct __n3l_backward_data *next_tdata;
  N3LNeuron *neuron;
  N3LWeight *weight;

  if ( !tdata->layer ) {
    for ( neuron = tdata->layer->nhead; neuron; neuron = neuron->next ) {
      for ( weight = neuron->whead; weight; weight = weight->next ) {
        if ( weight->target_ref == tdata->ref ) {
          break;
        }
      }

      if ( weight == NULL ) {
        continue;
      }

      weight->value += tdata->learning_rate * tdata->delta * neuron->result;
    }
  }
  else {
    nsize = n3l_neuron_count(tdata->layer->nhead);
    threads = (pthread_t *) malloc(nsize * sizeof(pthread_t));
    next_tdata = (struct __n3l_backward_data *) malloc(nsize * sizeof(struct __n3l_backward_data));

    for ( neuron = tdata->layer->nhead, j = 0; neuron; neuron = neuron->next, ++j ) {
      for ( weight = neuron->whead; weight; weight = weight->next ) {
        if ( weight->target_ref == tdata->ref ) {
          break;
        }
      }

      if ( weight == NULL ) {
        continue;
      }

      next_tdata[j].ref = neuron->ref;
      next_tdata[j].layer = tdata->layer->prev;
      next_tdata[j].learning_rate = tdata->learning_rate;
      next_tdata[j].delta = tdata->delta * weight->value * neuron->act_prime(neuron->input);

      pthread_create(&threads[j], NULL, __n3l_backward_execute, (void *) &(next_tdata[j]));
    }

    for ( j = 0; j < nsize; ++j ) {
      pthread_join(threads[j], NULL);
    }

    free(threads);
    free(next_tdata);
  }
}
