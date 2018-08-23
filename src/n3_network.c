#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "n3_header.h"
#include "n3_layer.h"
#include "n3_neuron.h"

N3LArgs __n3l_network_get_args_from_file(FILE *, double *);
double __n3l_get_weight_from_file(void *);

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

double __n3l_get_weight_from_file(void *data)
{
  double weight;
  FILE *n3_file = (FILE *) data;

  fread(&weight, sizeof(double), 1, n3_file);

  return weight;
}

N3LArgs __n3l_network_get_args_from_file(FILE *n3_file, double *learning_rate)
{
  N3LArgs args;

  fread(learning_rate, sizeof(double), 1, n3_file);
  fread(&(args.bias), sizeof(double), 1, n3_file);
  fread(&(args.in_size), sizeof(uint64_t), 1, n3_file);
  fread(&(args.h_layers), sizeof(uint64_t), 1, n3_file);
  args.h_size = (uint64_t *) malloc(args.h_layers * sizeof(uint64_t));
  fread(args.h_size, sizeof(uint64_t), args.h_layers, n3_file);
  fread(&(args.out_size), sizeof(uint64_t), 1, n3_file);
  fread(&(args.act_in), sizeof(N3LActType), 1, n3_file);
  args.act_h = (N3LActType *) malloc(args.h_layers * sizeof(N3LActType));
  fread(args.act_h, sizeof(N3LActType), args.h_layers, n3_file);
  fread(&(args.act_out), sizeof(N3LActType), 1, n3_file);
  args.rand_arg = (void *) n3_file;
  args.rand_weight = &__n3l_get_weight_from_file;

  return args;
}

N3LNetwork *n3l_network_build_from_file(char *filename)
{
  N3LNetwork *net;
  FILE *n3_file;
  N3LArgs args;
  double learning_rate;

  if ( !(n3_file = fopen(filename, "r")) ) {
    return NULL;
  }

  args = __n3l_network_get_args_from_file(n3_file, &learning_rate);
  net = n3l_network_build(args, learning_rate);
  free(args.h_size);
  free(args.act_h);
  fclose(n3_file);

  return net;
}

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

bool n3l_network_save(N3LNetwork *net, N3LArgs args, char *filename)
{
  FILE *n3_file;
  N3LLayer *layer;
  N3LNeuron *neuron;
  N3LWeight *weight;

  if ( !(n3_file = fopen(filename, "w")) ) {
    return false;
  }

  fwrite(&(net->learning_rate), sizeof(double), 1, n3_file);
  fwrite(&(args.bias), sizeof(double), 1, n3_file);
  fwrite(&(args.in_size), sizeof(uint64_t), 1, n3_file);
  fwrite(&(args.h_layers), sizeof(uint64_t), 1, n3_file);
  fwrite(args.h_size, sizeof(uint64_t), args.h_layers, n3_file);
  fwrite(&(args.out_size), sizeof(uint64_t), 1, n3_file);
  fwrite(&(args.act_in), sizeof(N3LActType), 1, n3_file);
  fwrite(args.act_h, sizeof(N3LActType), args.h_layers, n3_file);
  fwrite(&(args.act_out), sizeof(N3LActType), 1, n3_file);

  for ( layer = net->lhead; layer; layer = layer->next ) {
    for ( neuron = layer->nhead; neuron; neuron = neuron->next ) {
      for ( weight = neuron->whead; weight; weight = weight->next ) {
        fwrite(&(weight->value), sizeof(double), 1, n3_file);
      }
    }
  }

  return true;
}
