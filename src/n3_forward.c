#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <assert.h>
#include "n3_header.h"
#include "n3_neuron.h"

struct __n3l_forward_data {
  uint64_t ref;
  N3LLayer *layer;
  double *result;
};

void *    __n3l_forward_activate    (void *arg);
void *    __n3l_forward_get_outputs (void *arg);
double *  __n3l_forward_layer       (N3LLayer *layer, double *inputs);

double *n3l_forward_propagation(N3LNetwork *net)
{
  N3LLayer *layer;
  double *layer_in_data = NULL, *layer_out_data = NULL;

  for ( layer = net->lhead; layer; layer = layer->next ) {
    layer_out_data = __n3l_forward_layer(layer, layer_in_data ? : net->inputs);

    if ( layer_in_data ) {
      free(layer_in_data);
    }

    layer_in_data = layer_out_data;
  }

  return layer_out_data;
}

double *__n3l_forward_layer(N3LLayer *layer, double *inputs)
{
  pthread_t *threads = NULL;
  struct __n3l_forward_data *tdata = NULL;
  uint64_t nsize = n3l_neuron_count(layer->nhead), j;
  N3LNeuron *neuron = NULL;
  double *outputs = NULL;

  assert(inputs != NULL && nsize > 0);

  threads = (pthread_t *) malloc(nsize * sizeof(pthread_t));
  for ( neuron = layer->nhead, j = 0; neuron; neuron = neuron->next, ++j ) {
    if ( neuron->bias == false ) {
      neuron->input = inputs[j];
    }

    pthread_create(&threads[j], NULL, __n3l_forward_activate, (void *) neuron);
  }

  for ( j = 0; j < nsize; ++j ) {
    pthread_join(threads[j], NULL);
  }
  free(threads);

  if ( layer->type == N3LOutputLayer ) {
    outputs = (double *) malloc(nsize * sizeof(double));
    for ( neuron = layer->nhead, j = 0; neuron; neuron = neuron->next, ++j ) {
      outputs[j] = neuron->result;
    }
  }
  else {
    assert(layer->next != NULL);

    nsize = n3l_neuron_count(layer->next->nhead);
    assert(nsize > 0);

    outputs = (double *) malloc(nsize * sizeof(double));
    threads = (pthread_t *) malloc(nsize * sizeof(pthread_t));
    tdata = (struct __n3l_forward_data *) malloc(nsize * sizeof(struct __n3l_forward_data));
    for ( neuron = layer->next->nhead, j = 0; neuron; neuron = neuron->next, ++j ) {
      tdata[j].ref = neuron->ref;
      tdata[j].layer = layer;
      tdata[j].result = &outputs[j];

      pthread_create(&threads[j], NULL, __n3l_forward_get_outputs, (void *) &tdata[j]);
    }

    for ( j = 0; j < nsize; ++j ) {
      pthread_join(threads[j], NULL);
    }

    free(threads);
    free(tdata);
  }

  return outputs;
}

void *__n3l_forward_activate(void *arg)
{
  N3LNeuron *ref = (N3LNeuron *) arg;

  ref->result = ref->act(ref->input);

  return NULL;
}

void *__n3l_forward_get_outputs(void *arg)
{
  struct __n3l_forward_data *tdata = (struct __n3l_forward_data *) arg;
  N3LNeuron *neuron;
  N3LWeight *weight;

  *(tdata->result) = 0.0f;
  for ( neuron = tdata->layer->nhead; neuron; neuron = neuron->next ) {
    for ( weight = neuron->whead; weight; weight = weight->next ) {
      if ( weight->target_ref == tdata->ref ) {
        break;
      }
    }

    if ( !weight ) {
      continue;
    }

    *(tdata->result) += neuron->result * weight->value;
  }

  return NULL;
}
