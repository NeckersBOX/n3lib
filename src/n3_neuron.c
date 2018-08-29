#include <stdlib.h>
#include "n3_header.h"
#include "n3_act.h"

uint64_t n3l_neuron_count(N3LNeuron *head);

N3LNeuron *n3l_neuron_build(N3LActType act_type)
{
  static uint64_t ref = 0;
  N3LNeuron *neuron;

  neuron = (N3LNeuron *) malloc(sizeof(N3LNeuron));
  neuron->bias = false;
  neuron->ref = ++ref;
  neuron->act_type = act_type;
  neuron->act = n3l_act(act_type);
  neuron->act_prime = n3l_act_prime(act_type);
  neuron->next = NULL;
  neuron->prev = NULL;
  neuron->whead = NULL;

  return neuron;
}

N3LNeuron *n3l_neuron_build_after(N3LNeuron *prev, N3LActType type)
{
  N3LNeuron *neuron;

  neuron = n3l_neuron_build(type);
  if ( prev ) {
    neuron->prev = prev;
    neuron->next = prev->next;
    if ( prev->next ) {
      prev->next->prev = neuron;
    }
    prev->next = neuron;
  }

  return neuron;
}

N3LNeuron *n3l_neuron_build_before(N3LNeuron *next, N3LActType type)
{
  N3LNeuron *neuron;

  neuron = n3l_neuron_build(type);
  if ( next ) {
    neuron->next = next;
    neuron->prev = next->prev;
    if ( next->prev ) {
      next->prev->next = neuron;
    }
    next->prev = neuron;
  }

  return neuron;
}

void n3l_neuron_build_weights(N3LNeuron *src, N3LNeuron *t_list, N3LWeightGenerator weight_generator, void *weight_arg)
{
  N3LNeuron *target;
  N3LWeight *weight = NULL;
  uint64_t j, size;

  size = n3l_neuron_count(t_list);
  for ( target = t_list, j = 0; j < size; target = target->next, ++j ) {
    if ( weight ) {
      weight->next = (N3LWeight *) malloc(sizeof(N3LWeight));
      weight = weight->next;
    }
    else {
      weight = (N3LWeight *) malloc(sizeof(N3LWeight));
    }

    weight->next = NULL;
    weight->value = weight_generator(weight_arg);
    weight->target_ref = target->ref;
    src->whead = src->whead ? : weight;
  }
}

uint64_t n3l_neuron_count(N3LNeuron *head)
{
  uint64_t cnt = 0;
  N3LNeuron *p;

  for ( p = head; p; p = p->next, ++cnt );
  return cnt;
}

uint64_t n3l_neuron_count_weights(N3LWeight *head)
{
  uint64_t cnt = 0;
  N3LWeight *p;

  for ( p = head; p; p = p->next, ++cnt );
  return cnt;
}

void n3l_neuron_free(N3LNeuron *neuron)
{
  N3LWeight *p;

  if ( neuron ) {
    while ( neuron->whead ) {
      p = neuron->whead->next;
      free(neuron->whead);;
      neuron->whead = p;
    }
    free(neuron);
  }
}


extern N3LWeight *n3l_neuron_get_weight(N3LWeight *whead, uint64_t t_ref)
{
  N3LWeight *weight;

  for ( weight = whead; weight; weight = weight->next ) {
    if ( weight->target_ref == t_ref ) {
      break;
    }
  }

  return weight;
}

void n3l_neuron_set_custom_act(N3LNeuron *neuron, N3LAct act, N3LAct prime)
{
  neuron->act_type = N3LCustom;
  neuron->act = act;
  neuron->act_prime = prime;
}
