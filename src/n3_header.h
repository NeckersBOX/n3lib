#ifndef _N3L_HEADER_
#define _N3L_HEADER_

#include <stdint.h>

#define N3L_VERSION "1.2.7"

#define N3L_ACT(fun)          double (*fun)(double)
#define N3L_RND_WEIGHT(rnd_w) double (*rnd_w)(N3LLayer)

typedef enum { false = 0, true } bool;

typedef enum {
  N3LInputLayer = 0,
  N3LHiddenLayer,
  N3LOutputLayer
} N3LLayerType;

typedef enum {
  N3LLogNone = -1,
  N3LLogCritical = 0,
  N3LLogHigh,
  N3LLogMedium,
  N3LLogLow,
  N3LLogPedantic
} N3LLogType;

typedef enum {
  N3LNone = 0,
  N3LSigmoid,
  N3LTanh,
  N3LRelu,
} N3LActType;

typedef struct {
  FILE *log_file;
  N3LLogType verbosity;
} N3LLogger;

typedef struct {
  double input;
  double *weights;
  uint64_t outputs;
  double result;
  N3L_ACT(act);
  N3L_ACT(act_prime);
} N3LNeuron;

typedef struct {
  N3LLayerType ltype;
  uint64_t size;
  N3LNeuron *neurons;
} N3LLayer;

typedef struct {
  bool read_file;
  char *in_filename;
  char *out_filename;
  double bias;
  double learning_rate;
  uint64_t in_size;
  uint64_t h_size;
  uint64_t h_layers;
  uint64_t out_size;
  N3LLogger *logger;
  N3LActType act_h;
  N3LActType act_o;
} N3LArgs;

typedef struct {
  double *inputs;
  double *targets;
  double *outputs;
  N3L_RND_WEIGHT(get_rnd_weight);
  N3LArgs *args;
  N3LLayer *net;
} N3LData;

#endif
