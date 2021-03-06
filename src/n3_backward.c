/**
 * @file n3_backward.c
 * @author Davide Francesco Merico
 * @brief This file contains functions to backpropagate the error and adjusts the weights.
 */
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include "n3_header.h"
#include "n3_neuron.h"
#include "n3_threads.h"

/**
 * @brief Internal struct to share data between threads.
 *
 * Initialized from the current layer to the previous one.
 */
struct __n3l_backward_data {
  uint64_t ref;					 /**< Out neuron reference id */
  N3LLayer *layer;			 /**< Previous layer */
  double delta;					 /**< Delta evaluated */
  double learning_rate;	 /**< Learning rate set to the net */
};

void *__n3l_backward_execute(void *arg);

/**
 * @brief Execute backward propagation on the whole network.
 *
 * Each call to the previous layer from the last layer is executed with concurrents threads.
 *
 * @note The member \p net->targets must be initialized before calling this function.
 * @note This function should be called after n3l_forward_execute()
 * @param net Initialized network
 * @return TRUE if was correctely executed, otherwise FALSE.
 *
 * @see n3l_forward_execute, _n3l_network, __n3l_backward_execute
 */
bool n3l_backward_propagation(N3LNetwork *net)
{
  uint64_t nsize, j;
  N3LNeuron *out;
  struct __n3l_backward_data *tdata;

  if ( !net ) {
    return false;
  }

  assert(net->ltail != NULL);
  if ( !net->targets || net->ltail->type != N3LOutputLayer ) {
    return false;
  }

  nsize = n3l_neuron_count(net->ltail->nhead);
  assert(nsize > 0);

  n3l_threads_init();
  tdata = (struct __n3l_backward_data *) malloc(nsize * sizeof(struct __n3l_backward_data));
  for ( out = net->ltail->nhead, j = 0; out; out = out->next, ++j ) {
    tdata[j].ref = out->ref;
    tdata[j].layer = net->ltail->prev;
    tdata[j].delta = (net->targets[j] - out->result) * out->act_prime(out->input);
    if ( isnan(tdata[j].delta) || isinf(tdata[j].delta) ) {
      if ( (out->act_prime(out->input) > 0) == ((net->targets[j] - out->result) > 0) ) {
        tdata[j].delta = DBL_MAX;
      }
      else {
        tdata[j].delta = DBL_MIN;
      }
    }
    tdata[j].learning_rate = net->learning_rate;

    n3l_threads_add(__n3l_backward_execute, (void *) &(tdata[j]));
  }

  n3l_threads_flush();
  free(tdata);

  return true;
}

/**
 * @brief Internal function to execute backward propagation from the current layer to the previous one.
 *
 * Recursive function to execute backpropagation from the current layer to the previous one, only if
 * the layer passed as thread data is not NULL. After backpropagate it adjusts the current layer's weights.
 *
 * @param arg Pointer to an initialized #__n3l_backward_data struct
 * @return No value returned.
 *
 * @see n3l_backward_execute, __n3l_backward_data
 */
void *__n3l_backward_execute(void *arg)
{
  struct __n3l_backward_data *tdata = (struct __n3l_backward_data *) arg;
  struct __n3l_backward_data next_tdata;
  N3LNeuron *neuron;
  N3LWeight *weight;

  if ( tdata->layer ) {
    for ( neuron = tdata->layer->nhead; neuron; neuron = neuron->next ) {
      if ( neuron->bias ) {
        continue;
      }

      if ( !(weight = n3l_neuron_get_weight(neuron->whead, tdata->ref)) ) {
        continue;
      }

      next_tdata.ref = neuron->ref;
      next_tdata.layer = tdata->layer->prev;
      next_tdata.learning_rate = tdata->learning_rate;
      next_tdata.delta = tdata->delta * weight->value * neuron->act_prime(neuron->input);

      if ( isnan(next_tdata.delta) || isinf(next_tdata.delta) ) {
        next_tdata.delta = ( weight->value < 0 ) ? DBL_MIN : DBL_MAX;
      }

      __n3l_backward_execute((void *) &next_tdata);
    }

    for ( neuron = tdata->layer->nhead; neuron; neuron = neuron->next ) {
      if ( !(weight = n3l_neuron_get_weight(neuron->whead, tdata->ref)) ) {
        continue;
      }

      weight->value += tdata->learning_rate * tdata->delta * neuron->result;

      if ( isnan(weight->value) || isinf(weight->value) ) {
        if ( (tdata->delta > 0) == (neuron->result > 0) ) {
          weight->value = DBL_MAX;
        }
        else {
          weight->value = DBL_MIN;
        }
      }
    }
  }
}
