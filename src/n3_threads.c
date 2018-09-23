/**
 * @file n3_threads.c
 * @author Davide Francesco Merico
 * @brief This file contains functions to work with threads.
 * @note These functions are used to manage threads internally.
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>

/**
 * @brief Internal struct to manage a dynamic queue of threads
 *
 * @see __n3l_thread_queue, n3l_threads_flush, n3l_threads_init, n3l_threads_add
 */
struct __n3l_thread_controller {
  bool flush_call;        /**< True if n3l_threads_flush() is called */
  pthread_t *threads;     /**< Reference to running threads */
  pthread_mutex_t mutex;  /**< Used to lock the queue */
  pthread_cond_t cond;    /**< Used to signal a new element in the queue */
} N3L_THREADS_CTRL;

/**
 * @brief Maximum numbers of cores in use by the library
 * Default value: \p 1
 *
 * You can change this value before calling \p n3l_threads_init.
 *
 * @see n3l_threads_init, n3l_threads_add, n3l_threads_flush
 */
uint64_t N3L_THREADS_CORES = 1;

/**
 * @brief Internal struct to manage the routines queue
 *
 * It's a LIFO single linked list.
 *
 * @see __n3l_thread_controller, __n3l_threads_routine
 */
struct __n3l_thread_queue {
  void *(*start_routine) (void *);  /**< Thread routine */
  void *arg;                        /**< Routine argument */
  struct __n3l_thread_queue *next;  /**< Next element in the queue */
} *N3L_THREADS_QUEUE = NULL;

/**
 * @brief Internal thread routine to execute the queue routines.
 *
 * This function run in a separate thread and still alive until
 * n3l_threads_flush() is called. If no element are present in the
 * queue, it waits for a cond signal then start executing one-by-one
 * all elements in the queue.
 *
 * @note Each time an element in the queue is picked, the element routine
 * run inside this thread.
 *
 * @param arg NULL
 * @return NULL
 * @see n3l_threads_add, n3l_threads_flush, n3l_threads_init, __n3l_threads_queue
 */
void *__n3l_threads_routine(void *arg)
{
  int retval;
  struct __n3l_thread_queue *tq_item;

  while (true) {
    if ( (retval = pthread_mutex_lock(&N3L_THREADS_CTRL.mutex)) ) {
      fprintf(stderr, "N3Library: error on mutex lock in __n3l_threads_routine: %d\n", retval);
    }
    else if ( !N3L_THREADS_QUEUE ) {
      if ( N3L_THREADS_CTRL.flush_call ) {
        if ( (retval = pthread_mutex_unlock(&N3L_THREADS_CTRL.mutex)) ) {
          fprintf(stderr, "N3Library: error on mutex unlock in __n3l_threads_routine: %d\n", retval);
        }

        break;
      }
      else {
        pthread_cond_wait(&N3L_THREADS_CTRL.cond, &N3L_THREADS_CTRL.mutex);
      }

      if ( (retval = pthread_mutex_unlock(&N3L_THREADS_CTRL.mutex)) ) {
        fprintf(stderr, "N3Library: error on mutex unlock in __n3l_threads_routine: %d\n", retval);
      }
    }
    else {
      tq_item = N3L_THREADS_QUEUE;
      N3L_THREADS_QUEUE = N3L_THREADS_QUEUE->next;

      if ( (retval = pthread_mutex_unlock(&N3L_THREADS_CTRL.mutex)) ) {
        fprintf(stderr, "N3Library: error on mutex unlock in __n3l_threads_routine: %d\n", retval);
      }

      tq_item->start_routine(tq_item->arg);
      free(tq_item);
    }
  }

  return NULL;
}

/**
 * @brief Initialize the threads queue.
 *
 * @return 0 on success.
 * @see n3l_threads_add, n3l_threads_flush, __n3l_threads_routine
 */
int n3l_threads_init(void)
{
  int retval;
  uint64_t j;

  if ( !(retval = pthread_cond_init(&N3L_THREADS_CTRL.cond, NULL)) ) {
    if ( !(retval = pthread_mutex_init(&N3L_THREADS_CTRL.mutex, NULL)) ) {

      N3L_THREADS_CTRL.threads = (pthread_t *) malloc(N3L_THREADS_CORES * sizeof(pthread_t));
      N3L_THREADS_CTRL.flush_call = false;

      for ( j = 0; j < N3L_THREADS_CORES; ++j ) {
        retval = pthread_create(&(N3L_THREADS_CTRL.threads[j]), NULL, &__n3l_threads_routine, NULL);

        if ( retval ) {
          fprintf(stderr, "N3Library: error on pthread_create in n3l_threads_init: %d [%ld]\n", retval, j);
        }
      }
    }
  }

  return retval;
}

/**
 * @brief Add a new routine into the threads queue.
 *
 * @param start_routine Thread routine to call.
 * @param arg Routine argument.
 * @return 0 on success.
 *
 * @see n3l_threads_init, n3l_threads_flush, __n3l_threads_routine
 */
int n3l_threads_add(void *(*start_routine) (void *), void *arg)
{
  struct __n3l_thread_queue *tq_item;
  int retval;

  if ( (retval = pthread_mutex_lock(&N3L_THREADS_CTRL.mutex)) ) {
    fprintf(stderr, "N3Library: error on mutex lock in n3l_threads_add: %d\n", retval);
  }
  else {
    tq_item = (struct __n3l_thread_queue *) malloc(sizeof(struct __n3l_thread_queue));
    tq_item->start_routine = start_routine;
    tq_item->arg = arg;
    tq_item->next = N3L_THREADS_QUEUE;
    N3L_THREADS_QUEUE = tq_item;
    pthread_cond_signal(&N3L_THREADS_CTRL.cond);

    if ( (retval = pthread_mutex_unlock(&N3L_THREADS_CTRL.mutex)) ) {
      fprintf(stderr, "N3Library: error on mutex unlock in n3l_threads_add: %d\n", retval);
    }
  }

  return retval;
}

/**
 * @brief Execute all elements in the threads queue.
 *
 * Wait for all elements in the queue to be executed then return.
 *
 * @return 0 on success.
 *
 * @see n3l_threads_init, n3l_threads_add, __n3l_threads_routine
 */
int n3l_threads_flush(void)
{
  uint64_t j;
  int retval = 0;

  N3L_THREADS_CTRL.flush_call = true;

  if ( (retval = pthread_mutex_lock(&N3L_THREADS_CTRL.mutex)) ) {
    fprintf(stderr, "N3Library: error on mutex lock in n3l_threads_add: %d\n", retval);
  }
  else {
    /* Unlock waiting threads */
    for ( j = 0; j < N3L_THREADS_CORES; ++j ) {
      pthread_cond_signal(&N3L_THREADS_CTRL.cond);
    }

    if ( (retval = pthread_mutex_unlock(&N3L_THREADS_CTRL.mutex)) ) {
      fprintf(stderr, "N3Library: error on mutex unlock in n3l_threads_add: %d\n", retval);
    }

    for ( j = 0; j < N3L_THREADS_CORES; ++j ) {
      pthread_join(N3L_THREADS_CTRL.threads[j], NULL);
    }

    pthread_mutex_destroy(&N3L_THREADS_CTRL.mutex);
    pthread_cond_destroy(&N3L_THREADS_CTRL.cond);

    free(N3L_THREADS_CTRL.threads);
  }

  return retval;
}
