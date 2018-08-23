#ifndef _N3L_NEURON_
#define _N3L_NEURON_

extern N3LNeuron *  n3l_neuron_build          (N3LActType);
extern N3LNeuron *  n3l_neuron_build_after    (N3LNeuron *, N3LActType type);
extern N3LNeuron *  n3l_neuron_build_before   (N3LNeuron *, N3LActType type);
extern void         n3l_neuron_build_weights  (N3LNeuron *, N3LNeuron *, N3LWeightGenerator, void *);
extern uint64_t     n3l_neuron_count          (N3LNeuron *);
extern void         n3l_neuron_free           (N3LNeuron *);
extern void         n3l_neuron_set_custom_act (N3LNeuron *, N3LAct, N3LAct);

#endif
