/**
 * @file n3_file.c
 * @author Davide Francesco Merico
 * @brief This file contains functions to import and save a network state.
 */
#include <stdio.h>
#include <stdlib.h>
#include "n3_header.h"
#include "n3_neuron.h"
#include "n3_layer.h"

/**
 * @brief Internal function to read weight from file during network initialization.
 *
 * @param data Pointer to an already opened file. (FILE type)
 * @return The weight read from the FILE last read position. 
 *
 * @see n3l_file_import_network
 */
double __n3l_get_weight_from_file(void *data)
{
  double weight;
  FILE *n3_file = (FILE *) data;

  fread(&weight, sizeof(double), 1, n3_file);

  return weight;
}

/**
 * @brief Import the network state from a previously saved file.
 *
 * @param filename Previously saved file name with n3l_file_export_network().
 * @return The N3LNetwork saved if successfully read, otherwise NULL.
 *
 * @see n3l_file_export_network, _n3l_network, n3l_network_build
 */
N3LNetwork *n3l_file_import_network(char *filename)
{
  N3LNetwork *net;
  FILE *n3_file;
  N3LArgs args;
  double learning_rate;
  uint64_t lsize, nsize, wsize;
  N3LLayer *layer = NULL;
  N3LNeuron *neuron = NULL;
  N3LWeight *weight = NULL;
  N3LLayerType ltype;
  N3LActType act_type;

  if ( !(n3_file = fopen(filename, "r")) ) {
    return NULL;
  }

  net = (N3LNetwork *) malloc(sizeof(N3LNetwork));
  net->inputs = NULL;
  net->targets = NULL;
  net->lhead = NULL;
  net->ltail = NULL;
  fread(&(net->learning_rate), sizeof(double), 1, n3_file);
  fread(&lsize, sizeof(uint64_t), 1, n3_file);

  while ( lsize-- ) {
    fread(&ltype, sizeof(N3LLayerType), 1, n3_file);
    layer = n3l_layer_build_after(layer, ltype);

    net->lhead = net->lhead ? : layer;
    net->ltail = layer;

    fread(&nsize, sizeof(uint64_t), 1, n3_file);
    while ( nsize-- ) {
      fread(&act_type, sizeof(N3LActType), 1, n3_file);
      neuron = n3l_neuron_build_after(layer->ntail, act_type);
      fread(&(neuron->bias), sizeof(bool), 1, n3_file);
      fread(&(neuron->ref), sizeof(uint64_t), 1, n3_file);
      fread(&(neuron->input), sizeof(double), 1, n3_file);

      layer->nhead = layer->nhead ? : neuron;
      layer->ntail = neuron;

      fread(&wsize, sizeof(uint64_t), 1, n3_file);
      while ( wsize-- ) {
        weight = (N3LWeight *) malloc(sizeof(N3LWeight));
        fread(&(weight->value), sizeof(double), 1, n3_file);
        fread(&(weight->target_ref), sizeof(uint64_t), 1, n3_file);
        weight->next = neuron->whead;
        neuron->whead = weight;
      }
    }
  }

  fclose(n3_file);

  return net;
}

/**
 * @brief Export the current network state to the chosen file.
 *
 * @param net Initialized network state.
 * @param filename File name into write the current network state.
 * @return TRUE if correctly executed, otherwise FALSE.
 *
 * @see n3l_file_import_network, _n3l_network, n3l_network_free
 */
bool n3l_file_export_network(N3LNetwork *net, char *filename)
{
  FILE *n3_file;
  N3LLayer *layer;
  N3LNeuron *neuron;
  N3LWeight *weight;
  uint64_t size;

  if ( !(n3_file = fopen(filename, "w")) ) {
    return false;
  }

  fwrite(&(net->learning_rate), sizeof(double), 1, n3_file);

  size = n3l_layer_count(net->lhead);
  fwrite(&size, sizeof(uint64_t), 1, n3_file);
  for ( layer = net->lhead; layer; layer = layer->next ) {
    fwrite(&(layer->type), sizeof(N3LLayerType), 1, n3_file);

    size = n3l_neuron_count(layer->nhead);
    fwrite(&size, sizeof(uint64_t), 1, n3_file);
    for ( neuron = layer->nhead; neuron; neuron = neuron->next ) {
      fwrite(&(neuron->act_type), sizeof(N3LActType), 1, n3_file);
      fwrite(&(neuron->bias), sizeof(bool), 1, n3_file);
      fwrite(&(neuron->ref), sizeof(uint64_t), 1, n3_file);
      fwrite(&(neuron->input), sizeof(double), 1, n3_file);

      size = n3l_neuron_count_weights(neuron->whead);
      fwrite(&size, sizeof(uint64_t), 1, n3_file);
      for ( weight = neuron->whead; weight; weight = weight->next ) {
        fwrite(&(weight->value), sizeof(double), 1, n3_file);
        fwrite(&(weight->target_ref), sizeof(uint64_t), 1, n3_file);
      }
    }
  }

  return true;
}
