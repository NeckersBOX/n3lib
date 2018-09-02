/**
 * @file n3_neuron.c
 * @author Davide Francesco Merico
 * @brief This file contains functions to work with N3LNeuron type.
 * @note You may not use these functions directly but use functions like
 *  n3l_network_build(), n3l_network_free(), n3l_file_import_network(), etc..
 */
#include <stdlib.h>
#include "n3_header.h"
#include "n3_act.h"

uint64_t n3l_neuron_count(N3LNeuron *head);
N3LWeight *n3l_neuron_clone_weights(N3LWeight *whead);

/**
 * @brief Build a neuron.
 *
 * @param act_type Activation function used by the new neuron.
 * @return The new built neuron.
 *
 * @note References to weights or others neurons are sets to NULL.
 * @see n3l_neuron_build_after, n3l_neuron_build_before, n3l_neuron_free
 */
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

/**
 * @brief Build a neuron linked to a previous one.
 *
 * @param prev  Previous neuron to link the current one.
 * @param type Activation function used by the new neuron.
 * @return The new built neuron.
 *
 * @note References to weights or next neurons are sets to NULL.
 * @note Reference to previous neuron is set to \p prev
 * @note \p prev reference to the next neuron is set to the current one.
 *
 * @see n3l_neuron_build, n3l_neuron_build_before, n3l_neuron_free
 */
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

/**
 * @brief Build a neuron linked to a next one.
 *
 * @param next  Next neuron to link the current one.
 * @param type Activation function used by the new neuron.
 * @return The new built neuron.
 *
 * @note References to weights are sets to NULL.
 * @note Reference to previous neuron is set to \p next->prev
 * @note \p next reference to the previous neuron is set to the current one.
 *
 * @see n3l_neuron_build, n3l_neuron_build_after, n3l_neuron_free
 */
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

/**
 * @brief Build and initialize weights from a neuron to a list of neurons.
 *
 * @param src The neuron who has the weights with references to \p t_list
 * @param t_list Neuron's list head to be linked by \p src
 * @param weight_generator Weight initializing function.
 * @param weight_arg Argument to pass to \p weight_generator
 *
 * @see n3l_neuron_count, N3LWeightGenerator, n3l_neuron_count_weights, n3l_neuron_get_weight
 */
void n3l_neuron_build_weights(N3LNeuron *src, N3LNeuron *t_list, N3LWeightGenerator weight_generator, void *weight_arg)
{
  N3LNeuron *target;
  N3LWeight *weight = NULL;

  for ( target = t_list; target; target = target->next ) {
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

/**
 * @brief Count the neurons from the neuron passed as argument.
 *
 * @param head Neuron from which to start counting the next neurons.
 * @return Number of neurons from \p head ( it included ).
 * @note If \p head is NULL, the return value is 0.
 *
 * @see n3l_neuron_build, n3l_neuron_build_after, n3l_neuron_build_before
 */
uint64_t n3l_neuron_count(N3LNeuron *head)
{
  uint64_t cnt = 0;
  N3LNeuron *p;

  for ( p = head; p; p = p->next, ++cnt );
  return cnt;
}

/**
 * @brief Clone a neuron
 *
 * Clone the \p neuron with all weights and evaluated results.
 *
 * @param neuron Neuron to clone.
 * @return The new cloned neuron.
 * @see n3l_neuron_build, n3l_neuron_free, n3l_neuron_clone_weights
 */
N3LNeuron *n3l_neuron_clone(N3LNeuron *neuron)
{
  N3LNeuron *clone_neuron;

  clone_neuron = (N3LNeuron *) malloc(sizeof(N3LNeuron));
  clone_neuron->bias = neuron->bias;
  clone_neuron->ref = neuron->ref;
  clone_neuron->input = neuron->input;
  clone_neuron->whead = n3l_neuron_clone_weights(neuron->whead);
  clone_neuron->result = neuron->result;
  clone_neuron->act_type = neuron->act_type;
  clone_neuron->act = neuron->act;
  clone_neuron->act_prime = neuron->act_prime;
  clone_neuron->next = NULL;
  clone_neuron->prev = NULL;

  return clone_neuron;
}

/**
 * @brief Clone a weights list
 *
 * Clone the \p whead weights list with all the evaluated results.
 *
 * @param whead Weights list head to clone.
 * @return The new cloned weights list.
 * @see  n3l_neuron_free, n3l_neuron_clone
 */
N3LWeight *n3l_neuron_clone_weights(N3LWeight *whead)
{
  N3LWeight *clone_whead = NULL, *weight, *clone_weight = NULL;

  for ( weight = whead; weight; weight = weight->next ) {
    if ( clone_weight ) {
      clone_weight->next = (N3LWeight *) malloc(sizeof(N3LWeight));
      clone_weight = clone_weight->next;
    }
    else {
      clone_weight = (N3LWeight *) malloc(sizeof(N3LWeight));
    }

    clone_weight->value = weight->value;
    clone_weight->target_ref = weight->target_ref;
    clone_weight->next = NULL;

    clone_whead = clone_whead ? : clone_weight;
  }

  return clone_whead;
}

/**
 * @brief Count the weights from the weight passed as argument.
 *
 * @param head Weight from which to start counting the next weights.
 * @return Number of weights from \p head ( it included ).
 * @note If \p head is NULL, the return value is 0.
 *
 * @see n3l_neuron_build_weights, _n3l_weight, n3l_neuron_get_weight
 */
uint64_t n3l_neuron_count_weights(N3LWeight *head)
{
  uint64_t cnt = 0;
  N3LWeight *p;

  for ( p = head; p; p = p->next, ++cnt );
  return cnt;
}

/**
 * @brief Free the neuron's allocated memory.
 *
 * @warning It also free the memory allocated from weights into it.
 * @warning References to linked neurons are not changed.
 *
 * @param neuron Neuron to free.
 *
 * @see n3l_neuron_build, n3l_neuron_build_after, n3l_neuron_build_before
 */
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

/**
 * @brief Get the weight with specified target reference.
 *
 * @param whead Weight from which start to search the value \p t_ref
 * @param t_ref Target reference, equal to the linked neuron reference.
 * @return The weight with target reference searched if found, otherwise NULL.
 *
 * @see n3l_neuron_count_weights, n3l_neuron_build_weights, _n3l_weight
 */
N3LWeight *n3l_neuron_get_weight(N3LWeight *whead, uint64_t t_ref)
{
  N3LWeight *weight;

  for ( weight = whead; weight; weight = weight->next ) {
    if ( weight->target_ref == t_ref ) {
      break;
    }
  }

  return weight;
}

/**
 * @brief Set custom activation functions to the specified neuron.
 *
 * @note The act_type of \p neuron will be set to N3LCustom.
 *
 * @param neuron Neuron to apply the customs activation functions.
 * @param act Custom activation function.
 * @param prime Custom activativation function primitive.
 *
 * @see N3LAct, n3l_layer_set_custom_act, n3l_act, n3l_act_prime
 */
void n3l_neuron_set_custom_act(N3LNeuron *neuron, N3LAct act, N3LAct prime)
{
  neuron->act_type = N3LCustom;
  neuron->act = act;
  neuron->act_prime = prime;
}
