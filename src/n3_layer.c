/**
 * @file n3_layer.c
 * @author Davide Francesco Merico
 * @brief This file contains functions to work with N3Layer type.
 * @note You may not use these functions directly but use functions like
 *  n3l_network_build(), n3l_network_free(), n3l_file_import_network(), etc..
 */
#include <stdlib.h>
#include "n3_header.h"
#include "n3_neuron.h"

/**
 * @brief Build a layer.
 *
 * @param ltype Layer type.
 * @return The new built layer of type \p ltype.
 *
 * @note References to neurons or others layers are sets to NULL.
 * @see n3l_layer_build_after, n3l_layer_build_before, n3l_layer_free
 */
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

/**
 * @brief Build a layer linked to a previous one.
 *
 * @param prev  Previous layer to link the current one.
 * @param ltype Layer type.
 * @return The new built layer of type \p ltype.
 *
 * @note References to neurons or next layers are sets to NULL.
 * @note Reference to previous layer is set to \p prev
 * @note \p prev reference to the next layer is set to the current one.
 *
 * @see n3l_layer_build, n3l_layer_build_before, n3l_layer_free
 */
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

/**
 * @brief Build a layer linked to a next one.
 *
 * @param next  Next layer to link the current one.
 * @param ltype Layer type.
 * @return The new built layer of type \p ltype.
 *
 * @note References to neurons are sets to NULL.
 * @note Reference to previous layer is set to \p next->prev
 * @note \p next reference to the previous layer is set to the current one.
 *
 * @see n3l_layer_build, n3l_layer_build_after, n3l_layer_free
 */
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

/**
 * @brief Count the layers from the layer passed as argument.
 *
 * @param head Layer from which to start counting the next layers.
 * @return Number of layers from \p head ( it included ).
 * @note If \p head is NULL, the return value is 0.
 *
 * @see n3l_layer_build, n3l_layer_build_after, n3l_layer_build_before
 */
uint64_t n3l_layer_count(N3LLayer *head)
{
  uint64_t cnt = 0;
  N3LLayer *p;

  for ( p = head; p; p = p->next, ++cnt );
  return cnt;
}

/**
 * @brief Free the layer's allocated memory.
 *
 * @warning It also free the memory allocated from neurons into it.
 * @warning References to linked layers are not changed.
 *
 * @param layer Layer to free.
 *
 * @see n3l_layer_build, n3l_layer_build_after, n3l_layer_build_before, n3l_neuron_free
 */
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

/**
 * @brief Set custom activation functions to the layer's neurons.
 *
 * @param layer Layer to apply the customs activation functions.
 * @param act Custom activation function.
 * @param prime Custom activativation function primitive.
 * @param ignore_bias If TRUE the change is not applied to bias neurons.
 *
 * @see N3LAct, n3l_neuron_set_custom_act, n3l_act, n3l_act_prime
 */
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
