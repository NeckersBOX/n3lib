/**
 * @file n3_misc.c
 * @author Davide Francesco Merico
 * @brief This file contains functions to simplify working with the library.
 */
#include <stdlib.h>
#include "n3_header.h"

double n3l_misc_rnd_wp1(void *);
double n3l_misc_rnd_wn1(void *);
double n3l_misc_rnd_wpn1(void *);

/**
 * @brief Get the defaults arguments to build the network.
 *
 * Defaults values:
 * - bias:        \p 0.0
 * - in_size:     \p 1
 * - h_size:      \p NULL
 * - h_layers:    \p 0
 * - out_size:    \p 1
 * - act_in:      \p N3LNone
 * - act_h:       \p NULL
 * - act_out:     \p N3LSigmoid
 * - rand_arg:    \p NULL
 * - rand_weight: \p &n3l_misc_rnd_wp1
 *
 * @return The defaults arguments listed above.
 *
 * @see N3LArgs, n3l_network_build, n3l_misc_rnd_wp1
 */
N3LArgs n3l_misc_init_arg(void)
{
  N3LArgs defaults;

  defaults.bias = 0.0f;
  defaults.in_size = 1;
  defaults.h_size = NULL;
  defaults.h_layers = 0;
  defaults.out_size = 1;
  defaults.act_in = N3LNone;
  defaults.act_h = NULL;
  defaults.act_out = N3LSigmoid;
  defaults.rand_arg = NULL;
  defaults.rand_weight = &n3l_misc_rnd_wp1;

  return defaults;
}

/**
 * @brief Get random values between 0 and 1
 *
 * @param data Not used. Required by N3LWeightGenerator.
 * @return A random value between 0 and 1.
 *
 * @see n3l_neuron_build_weights, N3LArgs
 */
double n3l_misc_rnd_wp1(void *data)
{
  return ((double) rand()) / (RAND_MAX + 1.f);
}

/**
 * @brief Get random values between -1 and 0
 *
 * @param data Not used. Required by N3LWeightGenerator.
 * @return A random value between -1 and 0.
 *
 * @see n3l_neuron_build_weights, N3LArgs
 */
double n3l_misc_rnd_wn1(void *data)
{
  return -(((double) rand()) / (RAND_MAX + 1.f));
}

/**
 * @brief Get random values between -1 and 1
 *
 * @param data Not used. Required by N3LWeightGenerator.
 * @return A random value between -1 and 1.
 *
 * @see n3l_neuron_build_weights, N3LArgs
 */
double n3l_misc_rnd_wpn1(void *data)
{
  return (((double) rand()) / (RAND_MAX + 1.f)) * 2.f - 1.f;
}
