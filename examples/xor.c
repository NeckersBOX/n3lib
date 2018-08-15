#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#include "../n3lib.h"

#define BOOL_STR(b) ((b) ? "True" : "False")

struct user_args {
  bool learning;
  bool read_result;
  bool save_result;
  char *save_filename;
  char *read_filename;
  double learning_rate;
  uint64_t iterations;
  N3LLogType verbose;
};

double *get_inputs_batch_mode(void);
double *get_targets(double *inputs);
void xor_operation(struct user_args);

int main(int argc, char *argv[])
{
  struct user_args arg = { false, false, false, "xor.n3l", "xor.n3l", 1.f, 1, N3LLogNone };
  int c, v, option_index = 0;
  struct option long_options[] = {
    { "iteration", required_argument, 0, 'i' },
    { "learning",  optional_argument, 0, 'l' },
    { "read",      optional_argument, 0, 'r' },
    { "save",      optional_argument, 0, 's' },
    { "verbose",   required_argument, 0, 'v' }
  };

  srand(time(NULL));
  fprintf(stdout, "XOR Example - N3L v. %s\n", N3L_VERSION);
  fprintf(stdout, "(c) 2018 - Davide Francesco Merico <hds619 [at] gmail [dot] com>\n\n");

  while(1) {
    c = getopt_long(argc, argv, "i:lrsv:", long_options, &option_index);
    if ( c == -1 ) {
      break;
    }

    switch(c) {
      case 'l':
        arg.learning = true;
        if ( optarg ) {
          sscanf(optarg, "%lf", &(arg.learning_rate));
        }
        break;
      case 'i':
        arg.iterations = atoi(optarg);
        if ( arg.iterations <= 0 ) {
          fprintf(stderr, "Wrong iteration argument: %s\n", optarg);
          arg.iterations = 1;
        }
        break;
     case 's':
        arg.save_result = true;
        if ( optarg ) {
          arg.save_filename = strdup(optarg);
        }
        break;
     case 'r':
        arg.read_result = true;
        if ( optarg ) {
          arg.read_filename = strdup(optarg);
        }
        break;
     case 'v':
        v = atoi(optarg);
        arg.verbose = v < 0 ? -1 : (v > N3LLogPedantic) ? N3LLogPedantic : v;
        break;
    }
  }

  xor_operation(arg);

  return 0;
}

void xor_operation(struct user_args args)
{
  N3LArgs n3_args;
  N3LData *n3_net;
  N3LLogger n3_logger = { stdout, args.verbose };
  uint64_t stats[2] = { 0, 0 };
  FILE *of;

  fprintf(stdout, "Simulation property:\n");
  fprintf(stdout, "Read from file: %s ( %s )\n", BOOL_STR(args.read_result), args.read_filename);
  fprintf(stdout, "  Save to file: %s ( %s )\n", BOOL_STR(args.save_result), args.save_filename);
  fprintf(stdout, " Learning rate: %lf\n", args.learning_rate);
  fprintf(stdout, "     Verbosity: %d\n", args.verbose);
  fprintf(stdout, "    Iterations: %ld\n", args.iterations);

  n3_args.read_file = args.read_result;
  n3_args.in_filename = args.read_filename;
  n3_args.out_filename = args.save_filename;
  n3_args.learning_rate = args.learning_rate;
  n3_args.in_size = 2;
  n3_args.out_size = 1;
  n3_args.h_size = 3;
  n3_args.h_layers = 1;

  n3_args.logger = &n3_logger;
  n3_net = n3l_build(n3_args, &n3l_rnd_weight, N3LTanh, N3LTanh);

  do {
    n3_net->inputs = get_inputs_batch_mode();
    fprintf(stdout, "[XOR]         Input 0: %.0lf\n", n3_net->inputs[0]);
    fprintf(stdout, "[XOR]         Input 1: %.0lf\n", n3_net->inputs[1]);
    n3_net->outputs = n3l_forward_propagation(n3_net);
    fprintf(stdout, "[XOR]          Output: %lf\n", n3_net->outputs[0]);
    n3_net->targets = get_targets(n3_net->inputs);
    fprintf(stdout, "[XOR]          Target: %.0lf\n", n3_net->targets[0]);
    ++stats[round(n3_net->outputs[0]) == n3_net->targets[0] ? 0 : 1];
    fprintf(stdout, "[XOR] Overall success: %.3lf%%\n", (stats[0] * 100.f) / (double) (stats[0] + stats[1]));

    if ( args.learning ) {
      n3l_backward_propagation(n3_net);
    }

    free(n3_net->inputs);
    free(n3_net->outputs);
    free(n3_net->targets);
  } while (--args.iterations);

  if ( args.save_result ) {
    if ( !(of = fopen(args.save_filename, "w")) ) {
      fprintf(stderr, "Cannot save results. Open failed ( %s )", args.save_filename);
    }
    else {
      n3l_save(n3_net, of);
      fclose(of);
    }
  }
}

double *get_inputs_batch_mode(void)
{
  double *inputs = NULL;
  static uint8_t step = 0;

  inputs = (double *) malloc(2 * sizeof(double));

  switch(step) {
    case 0:
      inputs[0] = 0; inputs[1] = 0;
      break;
    case 1:
      inputs[0] = 1; inputs[1] = 0;
      break;
    case 2:
      inputs[0] = 1; inputs[1] = 1;
      break;
    case 3:
      inputs[0] = 0; inputs[1] = 1;
      step = -1;
      break;
  }

  ++step;

  return inputs;
}

double *get_targets(double *inputs)
{
  double *targets = NULL;

  targets = (double *) malloc(sizeof(double));
  targets[0] = (double) ((((uint8_t) inputs[0]) ^ ((uint8_t) inputs[1])) & 1);

  return targets;
}
