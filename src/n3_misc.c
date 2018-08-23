#include <stdlib.h>
#include "n3_header.h"

double n3l_misc_rnd_wp1(void *);
double n3l_misc_rnd_wn1(void *);
double n3l_misc_rnd_wpn1(void *);

N3LArgs n3l_misc_init_arg(void)
{
  N3LArgs defaults;

  defaults.bias = 0.0f;
  defaults.in_size = 1,
  defaults.h_size = NULL;
  defaults.h_layers = 0;
  defaults.out_size = 1;
  defaults.act_in = N3LNone;
  defaults.act_h = NULL;
  defaults.act_out = N3LSigmoid;
  defaults.rand_arg = NULL;
  defaults.rand_weight = &n3l_misc_rnd_wp1;
}

double n3l_misc_rnd_wp1(void *data)
{
  return ((double) rand()) / (RAND_MAX + 1.f);
}

double n3l_misc_rnd_wn1(void *data)
{
  return -(((double) rand()) / (RAND_MAX + 1.f));
}

double n3l_misc_rnd_wpn1(void *data)
{
  return (((double) rand()) / (RAND_MAX + 1.f)) * 2.f - 1.f;
}
