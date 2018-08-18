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
  return val * (1 - val);
}

double n3l_act_tanh(double val)
{
  return tanh(val);
}

double n3l_act_tanh_prime(double val)
{
  return 1 - (val * val);
}

double n3l_act_relu(double val)
{
  return ( val < 0 ) ? 0 : val;
}

double n3l_act_relu_prime(double val)
{
  return ( val <= 0 ) ? 0 : 1; 
}
