#ifndef _N3L_ACT_
#define _N3L_ACT_

extern N3LAct n3l_act(N3LActType);
extern N3LAct n3l_act_prime(N3LActType);

extern double n3l_act_identity(double);
extern double n3l_act_identity_prime(double);
extern double n3l_act_leaky_relu(double);
extern double n3l_act_leaky_relu_prime(double);
extern double n3l_act_none(double);
extern double n3l_act_relu(double);
extern double n3l_act_relu_prime(double);
extern double n3l_act_sigmoid(double);
extern double n3l_act_sigmoid_prime(double);
extern double n3l_act_softplus(double);
extern double n3l_act_softplus_prime(double);
extern double n3l_act_softsign(double);
extern double n3l_act_softsign_prime(double);
extern double n3l_act_swish(double);
extern double n3l_act_swish_prime(double);
extern double n3l_act_tanh(double);
extern double n3l_act_tanh_prime(double);

#endif
