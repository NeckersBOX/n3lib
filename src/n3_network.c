/**
 * @file n3_network.c
 * @author Davide Francesco Merico
 * @brief This file contains functions to work with N3LNetwork type.
 */
#include <stdlib.h>
#include <string.h>
#include "n3_header.h"
#include "n3_layer.h"
#include "n3_neuron.h"

/**
 * @brief Build a new network.
 *
 * @note Layers are built in order: Input, Hidden, Output.
 * @note Bias neurons, if specified, are added as last neuron in neurons list.
 *
 * @param args Parameters to initialize the network.
 * @param learn_rate Initial learning rate when used backpropagation.
 * @return The new network if built successfully, otherwise NULL.
 *
 * @see N3LArgs, n3l_misc_init_arg, N3LNetwork, n3l_network_free, n3l_file_import_network
 */
N3LNetwork *n3l_network_build(N3LArgs args, double learn_rate)
{
  N3LNetwork *net;
  N3LLayer *layer;
  N3LNeuron *neuron = NULL, *bias;
  uint64_t h_idx;

  if ( !args.in_size || !args.out_size ) {
    return NULL;
  }

  net = (N3LNetwork *) malloc(sizeof(N3LNetwork));
  net->inputs = NULL;
  net->targets = NULL;
  net->learning_rate = learn_rate;

  layer = n3l_layer_build(N3LInputLayer);
  net->lhead = layer;
  while ( args.in_size-- ) {
    neuron = n3l_neuron_build_after(neuron, args.act_in);
    layer->nhead = layer->nhead ? : neuron;
    layer->ntail = neuron;
  }

  for ( h_idx = 0; h_idx < args.h_layers; ++h_idx ) {
    layer = n3l_layer_build_after(layer, N3LHiddenLayer);
    neuron = NULL;

    while ( args.h_size[h_idx]-- ) {
      neuron = n3l_neuron_build_after(neuron, args.act_h[h_idx]);
      layer->nhead = layer->nhead ? : neuron;
      layer->ntail = neuron;
    }
  }

  layer = n3l_layer_build_after(layer, N3LOutputLayer);
  net->ltail = layer;
  neuron = NULL;
  while ( args.out_size-- ) {
    neuron = n3l_neuron_build_after(neuron, args.act_out);
    layer->nhead = layer->nhead ? : neuron;
    layer->ntail = neuron;
  }

  for ( layer = net->lhead; layer; layer = layer->next ) {
    if ( layer->next ) {
      if ( args.bias != 0.0f ) {
        bias = n3l_neuron_build_after(layer->ntail, N3LNone);
        bias->bias = true;
        bias->input = args.bias;
        layer->ntail = bias;
      }

      for ( neuron = layer->nhead; neuron; neuron = neuron->next ) {
        n3l_neuron_build_weights(neuron, layer->next->nhead, args.rand_weight, args.rand_arg);
      }
    }
  }

  return net;
}

/**
 * @brief Free the network's allocated memory.
 *
 * @note It also free all network's layers and neurons.
 *
 * @see n3l_layer_free, n3l_neuron_free, n3l_network_build
 */
void n3l_network_free(N3LNetwork *net)
{
  N3LLayer *p;

  if ( net ) {
    while ( net->lhead ) {
      p = net->lhead->next;
      n3l_layer_free(net->lhead);
      net->lhead = p;
    }
    free(net);
  }
}
