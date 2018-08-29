/**
 * @file n3_act.c
 * @author Davide Francesco Merico
 * @brief This file contains activation functions and their primitive.
 * 
 * @see <a href="https://en.wikipedia.org/wiki/Activation_function">Wikipedia - Activation Function</a> 
 * @note These functions, except for n3l_act or n3l_act_prime, shouldn't be used directly to improve 
 * 			 compatibility with future library versions.
 */
#include <assert.h>
#include <math.h>
#include "n3_header.h"

/**
 * @brief Get the absolute value of the argument passed.
 *
 * @param x value
 * @return x if x is positive, otherwise -x.
 */
#define N3L_ABS(x) (((x) < 0) ? -(x) : (x))

/**
 * @brief Doesn't change the value passed as argument.
 *
 * Used when no activation function is needed, by default is used for input layer's neurons.
 *
 *  \f[none(value) = value\f]
 *
 * @param val input value
 * @return the same value passed as argument.
 *
 * @see n3l_act_identity, n3l_act, N3LAct, N3LActTye
 */
double n3l_act_none(double val)
{
  return val;
}

/**
 * @brief Sigmoid activation function.
 *
 * \f[sigmoid(value) = \frac{1}{1+ e^{-value}}\f]
 *
 * @param val input value
 * @return the results from the formula above.
 *
 * @see n3l_act_sigmoid_prime, n3l_act, N3LAct, N3LActTye
 */
double n3l_act_sigmoid(double val)
{
  return 1 / (1 + exp(-val));
}

/**
 * @brief Sigmoid activation primitive function.
 *
 * \f[f'(value)= sigmoid(value) * (1 - sigmoid(value))\f]
 *
 * @param val input value
 * @return the results from the formula above.
 *
 * @see n3l_act_sigmoid, n3l_act_prime, N3LAct, N3LActTye
 */
double n3l_act_sigmoid_prime(double val)
{
  return n3l_act_sigmoid(val) * (1 - n3l_act_sigmoid(val));
}

/**
 * @brief Tanh activation function.
 *
 * \f[tanh(value)= \frac{e^{value}-e^{-value}}{e^{value}+e^{-value}}\f]
 *
 * @param val input value
 * @return the results from the formula above.
 *
 * @see n3l_act_tanh_prime, n3l_act, N3LAct, N3LActTye
 */
double n3l_act_tanh(double val)
{
  return tanh(val);
}

/**
 * @brief Tanh activation primitive function.
 *
 * \f[f'(value) = 1 - tanh(value)^{2}\f]
 *
 * @param val input value
 * @return the results from the formula above.
 *
 * @see n3l_act_tanh, n3l_act_prime, N3LAct, N3LActTye
 */
double n3l_act_tanh_prime(double val)
{
  return 1 - pow(tanh(val), 2);
}

/**
 * @brief ReLU activation function.
 *
 * \f[relu(value) = \begin{cases} 0 & \text{ if } value < 0 \\ value& \text{ if } value \geq 0 \end{cases}\f]
 *
 * @param val input value
 * @return the results from the formula above.
 *
 * @see n3l_act_relu_prime, n3l_act, N3LAct, N3LActTye
 */
double n3l_act_relu(double val)
{
  return ( val < 0 ) ? 0 : val;
}

/**
 * @brief ReLU activation primitive function.
 *
 * \f[(value)= \begin{cases} 0 & \text{ if } value < 0 \\ 1& \text{ if } value \geq 0 \end{cases}\f]
 *
 * @param val input value
 * @return the results from the formula above.
 *
 * @see n3l_act_relu, n3l_act_prime, N3LAct, N3LActTye
 */
double n3l_act_relu_prime(double val)
{
  return ( val < 0 ) ? 0 : 1;
}

/**
 * @brief Identity activation function.
 *
 * \f[identity(value) = value\f]
 *
 * @param val input value
 * @return the results from the formula above.
 *
 * @see n3l_act_identity_prime, n3l_act_none, n3l_act, N3LAct, N3LActTye
 */
double n3l_act_identity(double val)
{
  return val;
}

/**
 * @brief Identity activation primitive function.
 *
 * \f[f'(value)=1\f]
 *
 * @param val input value
 * @return the results from the formula above.
 *
 * @see n3l_act_identity, n3l_act_prime, N3LAct, N3LActTye
 */
double n3l_act_identity_prime(double val)
{
  return 1;
}

/**
 * @brief SoftSign activation function.
 *
 * \f[softsign(value)=\frac{value}{1+|value|}\f]
 *
 * @param val input value
 * @return the results from the formula above.
 *
 * @see n3l_act_softsign_prime, n3l_act, N3LAct, N3LActTye
 */
double n3l_act_softsign(double val)
{
  return val / (1 + N3L_ABS(val));
}

/**
 * @brief SoftSign activation primitive function.
 *
 * \f[f'(value)=\frac{1}{(1+|value|)^2}\f]
 *
 * @param val input value
 * @return the results from the formula above.
 *
 * @see n3l_act_softsign, n3l_act_prime, N3LAct, N3LActTye
 */
double n3l_act_softsign_prime(double val)
{
  return 1 / pow(1 + N3L_ABS(val), 2);
}

/**
 * @brief Leaky ReLU activation function.
 *
 * \f[leaky\_relu(value) = \begin{cases} 0.01value & \text{for } value < 0\\ value & \text{for } value \ge 0\end{cases}\f]
 *
 * @param val input value
 * @return the results from the formula above.
 *
 * @see n3l_act_leaky_relu_prime, n3l_act, N3LAct, N3LActTye
 */
double n3l_act_leaky_relu(double val)
{
  return val * ((val < 0) ? 0.01 : 1);
}

/**
 * @brief Leaky ReLU activation primitive function.
 *
 * \f[f'(value) = \begin{cases} 0.01 & \text{for } value < 0\\ 1 & \text{for } value \ge 0\end{cases}\f]
 *
 * @param val input value
 * @return the results from the formula above.
 *
 * @see n3l_act_leaky_relu, n3l_act_prime, N3LAct, N3LActTye
 */
double n3l_act_leaky_relu_prime(double val)
{
  return (val < 0) ? 0.01 : 1;
}

/**
 * @brief SoftPlus activation function.
 *
 * \f[softplus(value) = \ln(1 + e^{value})\f]
 *
 * @param val input value
 * @return the results from the formula above.
 *
 * @see n3l_act_softplus_prime, n3l_act, N3LAct, N3LActTye
 */
double n3l_act_softplus(double val)
{
  return log(1 + exp(val));
}

/**
 * @brief SoftPlus activation primitive function.
 *
 * \f[f'(value) = \frac{1}{1+ e^{-value}}\f]
 *
 * @param val input value
 * @return the results from the formula above.
 *
 * @see n3l_act_softplus, n3l_act_prime, N3LAct, N3LActTye
 */
double n3l_act_softplus_prime(double val)
{
  return 1 / (1 + exp(-val));
}

/**
 * @brief Swish activation function.
 *
 * \f[swish(value)=value * sigmoid(value)\f]
 *
 * @param val input value
 * @return the results from the formula above.
 *
 * @see n3l_act_swish_prime, n3l_act, N3LAct, N3LActTye
 */
double n3l_act_swish(double val)
{
  return val * n3l_act_sigmoid(val);
}

/**
 * @brief Swish activation primitive function.
 *
 * \f[f'(value)=swish(value) + sigmoid(value) * (1 - swish(value))\f]
 *
 * @param val input value
 * @return the results from the formula above.
 *
 * @see n3l_act_swish, n3l_act_prime, N3LAct, N3LActTye
 */
double n3l_act_swish_prime(double val)
{
  return n3l_act_swish(val) + n3l_act_sigmoid(val) * (1 - n3l_act_swish(val));
}

/**
 * @brief Get the pointer to an activation function.
 *
 * @param type Activation function type.
 * @return Pointer to the function chosen.
 *
 * @see n3l_act_prime, N3LAct, N3LActTye
 */
N3LAct n3l_act(N3LActType type)
{
  switch(type) {
    case N3LCustom:     return &n3l_act_none;
    case N3LNone:       return &n3l_act_none;
    case N3LSigmoid:    return &n3l_act_sigmoid;
    case N3LRelu:       return &n3l_act_relu;
    case N3LTanh:       return &n3l_act_tanh;
    case N3LIdentity:   return &n3l_act_identity;
    case N3LLeakyRelu:  return &n3l_act_leaky_relu;
    case N3LSoftPlus:   return &n3l_act_softplus;
    case N3LSoftSign:   return &n3l_act_softsign;
    case N3LSwish:      return &n3l_act_swish;
  }

  assert("Unmanaged N3LActType" && 0);
}

/**
 * @brief Get the pointer to an activation primitive function.
 *
 * @param type Activation function type.
 * @return Pointer to the function's primitive chosen.
 *
 * @see n3l_act, N3LAct, N3LActTye
 */
N3LAct n3l_act_prime(N3LActType type)
{
  switch(type) {
    case N3LCustom:     return &n3l_act_none;
    case N3LNone:       return &n3l_act_none;
    case N3LSigmoid:    return &n3l_act_sigmoid_prime;
    case N3LRelu:       return &n3l_act_relu_prime;
    case N3LTanh:       return &n3l_act_tanh_prime;
    case N3LIdentity:   return &n3l_act_identity_prime;
    case N3LLeakyRelu:  return &n3l_act_leaky_relu_prime;
    case N3LSoftPlus:   return &n3l_act_softplus_prime;
    case N3LSoftSign:   return &n3l_act_softsign_prime;
    case N3LSwish:      return &n3l_act_swish_prime;
  }

  assert("Unmanaged N3LActType" && 0);
}
