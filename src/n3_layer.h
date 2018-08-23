#ifndef _N3L_LAYER_
#define _N3L_LAYER_

extern N3LLayer * n3l_layer_build           (N3LLayerType);
extern N3LLayer * n3l_layer_build_after     (N3LLayer *, N3LLayerType);
extern N3LLayer * n3l_layer_build_before    (N3LLayer *, N3LLayerType);
extern uint64_t   n3l_layer_count           (N3LLayer *);
extern void       n3l_layer_free            (N3LLayer *);
extern void       n3l_layer_set_custom_act  (N3LLayer *, N3LAct, N3LAct, bool);

#endif
