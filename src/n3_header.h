/**
 * @file n3_header.h
 * @author Davide Francesco Merico
 * @brief This file contains types, enums and structs definitions.
 */
#ifndef _N3L_HEADER_
#define _N3L_HEADER_

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

/**
 * @brief N3 Library version
 */
#define N3L_VERSION "2.0.0"

/**
 * @brief Pointer to an activation function.
 *
 * @see n3l_act, n3l_act_prime, _n3l_neuron
 */
typedef double (*N3LAct)(double);

/**
 * @brief Pointer to a function to get the network weights.
 *
 * Used to get weights when a network is imported or built.
 *
 * @see n3l_misc_rnd_wp1, n3l_misc_rnd_wn1, n3l_misc_rnd_wpn1, __n3l_get_weight_from_file
 */
typedef double (*N3LWeightGenerator)(void *);

/**
 * @brief Identify the layer type.
 * @see _n3l_layer
 */
typedef enum {
  N3LInputLayer = 0, /**< Input layer, usually this type of layer doesn't have a previous layer linked. */
  N3LHiddenLayer,    /**< Hidden Layer, usually have both previous and next layer linked. */
  N3LOutputLayer     /**< Output Layer, usually this type of layer doesn't have a next layer linked. */
} N3LLayerType;

/**
 * @brief Activation function type.
 * @see n3l_act, n3l_act_prime, _n3l_neuron
 */
typedef enum {
  N3LCustom = -1, /**< Custom activation function. @see n3l_layer_set_custom_act, n3l_neuron_set_custom_act */
  N3LNone = 0,    /**< No activation function. @see n3l_act_none */
  N3LSigmoid,     /**< Sigmoid activation function. @see n3l_act_sigmoid */
  N3LTanh,        /**< Tanh activation function. @see n3l_act_tanh */
  N3LRelu,        /**< ReLU activation function. @see n3l_act_relu */
  N3LIdentity,    /**< Identity activation function. @see n3l_act_identity */
  N3LLeakyRelu,   /**< Leaky ReLU activation function. @see n3l_act_leaky_relu */
  N3LSoftPlus,    /**< SoftPlus activation function. @see n3l_act_softplus */
  N3LSoftSign,    /**< SoftSign activation function. @see n3l_act_softsign */
  N3LSwish        /**< Swish activation function. @see n3l_act_swish */
} N3LActType;

/**
 * @brief Single Linked List which contains weight's values.
 *
 * The list is built by n3l_network_build() or n3l_file_import_network().
 *
 * @see _n3l_neuron, n3l_neuron_get_weight, n3l_neuron_count_weights, n3l_neuron_build_weights
 */
typedef struct _n3l_weight {
  double value;             /**< Current weight value. */
  uint64_t target_ref;      /**< Reference to the next layer's neuron linked */
  struct _n3l_weight *next; /**< Next weight in the list or NULL if it's the last one. */
} N3LWeight;

/**
 * @brief Double Linked List which contains neuron's values.
 *
 * The list is built by n3l_network_build() or n3l_file_import_network().
 *
 * @see _n3l_layer, _n3l_weight, n3l_neuron_build, n3l_neuron_count, n3l_neuron_free
 */
typedef struct _n3l_neuron {
  /**
   * Identifies if the neuron is a bias neuron.
   * The bias neurons doesn't get inputs and have N3LNone as activation function.
   * @note Usually there is only one bias neuron for each layer, except for the output one,
   * and it's built as last neuron.
   *
   * @see n3l_act_none, N3LArgs, n3l_network_build
   */
  bool bias;
  /**
   * Current neuron reference.
   * @warning This value must be unique into the network. It is used to collecting
   * outputs in forward propagation and to evaluate delta in backward propagation.
   *
   * @see __n3l_forward_get_outputs, __n3l_backward_execute
   */
  uint64_t ref;
  /**
   * Neuron input value. This value is also used to apply activation fuction.
   *
   * @see __n3l_forward_layer
   */
  double input;
  /**
   * Weight's list head.
   * @note It is set to NULL if the neuron's layer is of type N3LOutputLayer
   *
   * @see _n3l_weight, _n3l_layer
   */
  N3LWeight *whead;
  /**
   * Neuron's result. This value is initialiazed after forward propagation.
   *
   * @see n3l_forward_propagation, __n3l_forward_activate
   */
  double result;
  N3LActType act_type;        /**< Activation function type. @see N3LActType */
  N3LAct act;                 /**< Activation function. @see N3LAct */
  N3LAct act_prime;           /**< Activation function primitive. @see N3LAct */
  struct _n3l_neuron *next;   /**< Next neuron in the list or NULL if it's the last one. */
  struct _n3l_neuron *prev;   /**< Previous neuron in the list or NULL if it's the first one. */
} N3LNeuron;

/**
 * @brief Double Linked List which contains layer's values.
 *
 * The list is built by n3l_network_build() or n3l_file_import_network().
 *
 * @see _n3l_neuron, N3LNetwork, n3l_layer_build, n3l_layer_count, n3l_layer_free
 */
typedef struct _n3l_layer {
  N3LLayerType type;        /**< Layer type. @see N3LLayerType */
  N3LNeuron *nhead;         /**< Layer's neuron list head. @see _n3l_neuron */
  N3LNeuron *ntail;         /**< Layer's neuron list tail. @see _n3l_neuron */
  struct _n3l_layer *next;  /**< Next layer in the list or NULL if it's the last one. */
  struct _n3l_layer *prev;  /**< Previous layer in the list or NULL if it's the first one. */
} N3LLayer;

/**
 * @brief Network arguments.
 *
 * These arguments are used to build network through the n3l_network_build() function.
 *
 * @see n3l_misc_init_arg, n3l_network_build
 */
typedef struct {
  double bias;        /**< Bias value. If 0 no bias neuron are built in the network. */
  uint64_t in_size;   /**< Number of neurons in the input layer. This must be greater than 0. */
  uint64_t h_layers;  /**< Number of hidden layers in the network. */
  uint64_t *h_size;   /**< Array of number of neurons in the hidden layers. */
  uint64_t out_size;  /**< Number of neurons in the output layer. This must be greater than 0. */
  N3LActType act_in;  /**< Activation function to use for input layer's neurons. */
  N3LActType *act_h;  /**< Array of activation functions to use for hidden layer's neurons. */
  N3LActType act_out; /**< Activation function to use for output layer's neurons. */
  void *rand_arg;     /**< Arguments to pass when \p rand_weight is called. */
  N3LWeightGenerator rand_weight; /**< Weights initialization function. */
} N3LArgs;

/**
 * @brief Network state.
 *
 * The main type used in the library to execute forward and backward propagation.
 * From this type you can have access to the whole network state.
 *
 * @see n3l_network_build, n3l_network_free, n3l_file_import_network, n3l_file_export_network
 */
typedef struct {
  double *inputs;       /**< Input values to start executing forward propagation. @see n3l_forward_propagation. */
  double *targets;      /**< Targets values to start executing backward propagation. @see n3l_backward_propagation. */
  double learning_rate; /**< Learning rate value. This value can be changed at any time. */
  N3LLayer *lhead;      /**< Network's layer list head. @see _n3l_layer */
  N3LLayer *ltail;      /**< Network's layer list tail. @see _n3l_layer */
} N3LNetwork;

#endif
