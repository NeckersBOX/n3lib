#include <math.h>

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
