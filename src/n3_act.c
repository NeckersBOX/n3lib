#include <assert.h>
#include <math.h>
#include "n3_header.h"

#define N3L_ABS(x) (((x) < 0) ? -(x) : (x))

double n3l_act_none(double val)
{
  return val;
}

double n3l_act_sigmoid(double val)
{
  return 1 / (1 + exp(-val));
}

double n3l_act_sigmoid_prime(double val)
{
  return n3l_act_sigmoid(val) * (1 - n3l_act_sigmoid(val));
}

double n3l_act_tanh(double val)
{
  return tanh(val);
}

double n3l_act_tanh_prime(double val)
{
  return 1 - pow(tanh(val), 2);
}

double n3l_act_relu(double val)
{
  return ( val < 0 ) ? 0 : val;
}

double n3l_act_relu_prime(double val)
{
  return ( val < 0 ) ? 0 : 1;
}

double n3l_act_identity(double val)
{
  return val;
}

double n3l_act_identity_prime(double val)
{
  return 1;
}

double n3l_act_softsign(double val)
{
  return val / (1 + N3L_ABS(val));
}

double n3l_act_softsign_prime(double val)
{
  return 1 / pow(1 + N3L_ABS(val), 2);
}

double n3l_act_leaky_relu(double val)
{
  return val * ((val < 0) ? 0.01 : 1);
}

double n3l_act_leaky_relu_prime(double val)
{
  return (val < 0) ? 0.01 : 1;
}

double n3l_act_softplus(double val)
{
  return log(1 + exp(val));
}

double n3l_act_softplus_prime(double val)
{
  return 1 / (1 + exp(-val));
}

double n3l_act_swish(double val)
{
  return val * n3l_act_sigmoid(val);
}

double n3l_act_swish_prime(double val)
{
  return n3l_act_swish(val) + n3l_act_sigmoid(val) * (1 - n3l_act_swish(val));
}

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
