#ifndef _N3L_INIT_
#define _N3L_INIT_

extern N3LData *n3l_build           (N3LArgs, N3L_RND_WEIGHT(rnd_w));
extern void     n3l_free            (N3LData *);
extern double   n3l_rnd_weight      (N3LLayer);
extern void     n3l_set_custom_act  (N3LData *, uint64_t, N3L_ACT(act), N3L_ACT(act_prime));

#endif
