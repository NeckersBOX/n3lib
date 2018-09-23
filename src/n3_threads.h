#ifndef _N3L_THREADS_
#define _N3L_THREADS_

extern int  n3l_threads_init  (void);
extern int  n3l_threads_add   (void *(*start_routine) (void *), void *arg);
extern int  n3l_threads_flush (void);

extern uint64_t N3L_THREADS_CORES;

#endif
