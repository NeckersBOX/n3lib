#include <math.h>

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
  return ( val <= 0 ) ? 0 : 1;
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
