#ifndef _N3L_HEADER_
#define _N3L_HEADER_

#include <stdio.h>
#include <stdint.h>

#define N3L_VERSION "2.0.0"

typedef double (*N3LAct)(double);
typedef double (*N3LWeightGenerator)(void *);

typedef enum { false = 0, true } bool;

typedef enum {
  N3LInputLayer = 0,
  N3LHiddenLayer,
  N3LOutputLayer
} N3LLayerType;

typedef enum {
  N3LCustom = -1,
  N3LNone = 0,
  N3LSigmoid,
  N3LTanh,
  N3LRelu,
  N3LIdentity,
  N3LLeakyRelu,
  N3LSoftPlus,
  N3LSoftSign,
  N3LSwish
} N3LActType;

typedef struct _n3l_weight {
  double value;
  uint64_t target_ref;
  struct _n3l_weight *next;
} N3LWeight;

typedef struct _n3l_neuron {
  bool bias;
  uint64_t ref;
  double input;
  N3LWeight *whead;
  double result;
  N3LAct act;
  N3LAct act_prime;
  struct _n3l_neuron *next;
  struct _n3l_neuron *prev;
} N3LNeuron;

typedef struct _n3l_layer {
  N3LLayerType type;
  N3LNeuron *nhead;
  N3LNeuron *ntail;
  struct _n3l_layer *next;
  struct _n3l_layer *prev;
} N3LLayer;

typedef struct {
  double bias;
  uint64_t in_size;
  uint64_t h_layers;
  uint64_t *h_size;
  uint64_t out_size;
  N3LActType act_in;
  N3LActType *act_h;
  N3LActType act_out;
  void *rand_arg;
  N3LWeightGenerator rand_weight;
} N3LArgs;

typedef struct {
  double *inputs;
  double *targets;
  double learning_rate;
  N3LLayer *lhead;
  N3LLayer *ltail;
} N3LNetwork;

#endif
