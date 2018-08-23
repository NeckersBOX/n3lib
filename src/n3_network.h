#ifndef _N3_NETWORK
#define _N3_NETWORK

extern N3LNetwork * n3l_network_build           (N3LArgs, double);
extern N3LNetwork * n3l_network_build_from_file (char *);
extern void         n3l_network_free            (N3LNetwork *);
extern bool         n3l_network_save            (N3LNetwork *, N3LArgs, char *);

#endif
