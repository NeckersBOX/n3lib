#ifndef _N3_NETWORK
#define _N3_NETWORK

extern N3LNetwork * n3l_network_build           (N3LArgs, double);
extern N3LNetwork * n3l_network_clone           (N3LNetwork *);
extern void         n3l_network_free            (N3LNetwork *);

#endif
