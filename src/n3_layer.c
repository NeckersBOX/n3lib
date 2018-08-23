#include <stdio.h>
#include <stdlib.h>
#include "n3_header.h"
#include "n3_neuron.h"

N3LLayer *n3l_layer_build(N3LLayerType ltype)
{
  N3LLayer *layer;

  layer = (N3LLayer *) malloc(sizeof(N3LLayer));
  layer->type = ltype;
  layer->prev = NULL;
  layer->next = NULL;
  layer->nhead = NULL;
  layer->ntail = NULL;

  return layer;
}

N3LLayer *n3l_layer_build_after(N3LLayer *prev, N3LLayerType ltype)
{
  N3LLayer *layer;

  layer = n3l_layer_build(ltype);
  if ( prev ) {
    layer->prev = prev;
    layer->next = prev->next;
    prev->next = layer;
  }

  return layer;
}

N3LLayer *n3l_layer_build_before(N3LLayer *next, N3LLayerType ltype)
{
  N3LLayer *layer;

  layer = n3l_layer_build(ltype);
  if ( next ) {
    layer->next = next;
    layer->prev = next->prev;
    next->prev = layer;
  }

  return layer;
}

uint64_t n3l_layer_count(N3LLayer *head)
{
  uint64_t cnt = 0;
  N3LLayer *p;

  for ( p = head; p; p = p->next, ++cnt );
  return cnt;
}

void n3l_layer_free(N3LLayer *layer)
{
  N3LNeuron *p_neuron;

  if ( layer ) {
    while ( layer->nhead  ) {
      p_neuron = layer->nhead->next;
      n3l_neuron_free(layer->nhead);
      layer->nhead = p_neuron;
    }
    free(layer);
  }
}

void n3l_layer_set_custom_act(N3LLayer *layer, N3LAct act, N3LAct prime, bool ignore_bias)
{
  N3LNeuron *neuron = layer->nhead;

  while ( neuron ) {
    if ( neuron->bias && ignore_bias ) {
      neuron = neuron->next;
      continue;
    }

    n3l_neuron_set_custom_act(neuron, act, prime);
    neuron = neuron->next;
  }
}
