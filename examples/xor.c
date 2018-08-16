#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "../n3lib.h"

#define BOOL_STR(b) ((b) ? "True" : "False")

struct user_args {
  bool learning;
  bool read_result;
  bool save_result;
  bool mute;
  bool progress;
  char *save_filename;
  char *read_filename;
  double learning_rate;
  double bias;
  uint64_t iterations;
  N3LLogType verbose;
};

double *get_inputs_batch_mode(void);
double *get_targets(double *inputs);
void xor_operation(struct user_args);

int main(int argc, char *argv[])
{
  struct user_args arg = { false, false, false, false, false, "xor.n3l", "xor.n3l", 1.f, 0, 1, N3LLogNone };
  int opt, v;
  bool free_save_filename = false;

  srand(time(NULL));
  fprintf(stdout, "XOR Example - N3L v. %s\n", N3L_VERSION);
  fprintf(stdout, "(c) 2018 - Davide Francesco Merico <hds619 [at] gmail [dot] com>\n\n");

  while ((opt = getopt(argc, argv, "b:hi:lmo:pr:sv:")) != -1) {
    switch(opt) {
      case 'b':
        sscanf(optarg, "%lf", &(arg.bias));
        break;
      case 'l':
        arg.learning = true;
        if ( optarg ) {
          sscanf(optarg, "%lf", &(arg.learning_rate));
        }
        break;
      case 'm':
        arg.mute = true;
        break;
      case 'p':
        arg.progress = true;
        break;
      case 'i':
        arg.iterations = atoi(optarg);
        if ( arg.iterations <= 0 ) {
          fprintf(stderr, "Wrong iteration argument: %s\n", optarg);
          arg.iterations = 1;
        }
        break;
     case 'o':
        free_save_filename = true;
        arg.save_filename = strdup(optarg);
        break;
     case 's':
        arg.save_result = true;
        break;
     case 'r':
        arg.read_result = true;
        arg.read_filename = strdup(optarg);
        break;
     case 'v':
        v = atoi(optarg);
        arg.verbose = v < 0 ? -1 : (v > N3LLogPedantic) ? N3LLogPedantic : v;
        break;
     case 'h':
        fprintf(stdout, "Usage: %s [options]\n\n", *argv);
        fprintf(stdout, "Options:\n");
        fprintf(stdout, "\t-b [n]         Set the bias term in the network. Default: 0\n");
        fprintf(stdout, "\t-h             Show this help with the options list.\n");
        fprintf(stdout, "\t-i [n]         Number of iterations. Default: 1\n");
        fprintf(stdout, "\t-l             Enable learning with backpropagation.\n");
        fprintf(stdout, "\t-m             No log at all. Note: Disable -v option.\n");
        fprintf(stdout, "\t-o [filename]  After the number of iterations provided, save the\n");
        fprintf(stdout, "\t               neural network state. Note: It works only if used\n");
        fprintf(stdout, "\t               with option -s.\n");
        fprintf(stdout, "\t-p             Enable the progress viewer. Active -m, Disable -v.\n");
        fprintf(stdout, "\t-r [filename]  Initialize the neural network reading the number of\n");
        fprintf(stdout, "\t               neurons, layers and weights from a previous state saved.\n");
        fprintf(stdout, "\t-s             After the number of iterations provided, save the\n");
        fprintf(stdout, "\t               neural network state. Default filename: xor.n3l\n");
        fprintf(stdout, "\t-v [n]         Enable N3 Library to log with specified verbosity.\n");
        fprintf(stdout, "\t               Value: 0 - Critical, 1 - High, 2 - Medium, 3 - Low,\n");
        fprintf(stdout, "\t                      4 - Pedantic.\n\n");
        exit(0);
        break;
    }
  }

  if ( arg.mute ) {
    arg.verbose = -1;
  }

  if ( arg.progress ) {
    arg.mute = true;
    arg.verbose = -1;
  }

  xor_operation(arg);
  if ( free_save_filename ) {
    free(arg.save_filename);
  }

  if ( arg.read_result ) {
    free(arg.read_filename);
  }

  return 0;
}

void xor_operation(struct user_args args)
{
  N3LArgs n3_args;
  N3LData *n3_net;
  N3LLogger n3_logger = { stdout, args.verbose };
  uint64_t stats[2] = { 0, 0 };
  FILE *of;

  if ( !args.mute ) {
    fprintf(stdout, "Simulation property:\n");
    fprintf(stdout, "Read from file: %s ( %s )\n", BOOL_STR(args.read_result), args.read_filename);
    fprintf(stdout, "  Save to file: %s ( %s )\n", BOOL_STR(args.save_result), args.save_filename);
    fprintf(stdout, " Learning rate: %lf\n", args.learning_rate);
    fprintf(stdout, "     Verbosity: %d\n", args.verbose);
    fprintf(stdout, "    Iterations: %ld\n\n", args.iterations);
  }

  n3_args.read_file = args.read_result;
  n3_args.in_filename = args.read_filename;
  n3_args.out_filename = args.save_filename;
  n3_args.learning_rate = args.learning_rate;
  n3_args.bias = args.bias;
  n3_args.in_size = 2;
  n3_args.out_size = 1;
  n3_args.h_size = 3;
  n3_args.h_layers = 1;
  n3_args.act_h = N3LSigmoid;
  n3_args.act_out = N3LRelu;

  n3_args.logger = &n3_logger;
  n3_net = n3l_build(n3_args, &n3l_rnd_weight);

  do {
    if ( !args.mute ) {
      fprintf(stdout, "[XOR] -- Iteration %ld on %ld --\n",
        stats[0] + stats[1] + 1, args.iterations + stats[0] + stats[1]);
    }

    if ( args.progress ) {
      fprintf(stdout, "\r[XOR] Iteration %ld on %ld - Overall: %.3lf%%",
        stats[0] + stats[1] + 1, args.iterations + stats[0] + stats[1],
        (stats[0] * 100.f) / (double) (stats[0] + stats[1]));
    }

    n3_net->inputs = get_inputs_batch_mode();
    if ( !args.mute ) {
      fprintf(stdout, "[XOR]         Input 0: %.0lf\n", n3_net->inputs[0]);
      fprintf(stdout, "[XOR]         Input 1: %.0lf\n", n3_net->inputs[1]);
    }
    n3_net->outputs = n3l_forward_propagation(n3_net);

    if ( !args.mute ) {
      fprintf(stdout, "[XOR]          Output: %lf\n", n3_net->outputs[0]);
    }

    n3_net->targets = get_targets(n3_net->inputs);
    ++stats[round(n3_net->outputs[0]) == n3_net->targets[0] ? 0 : 1];

    if ( !args.mute ) {
      fprintf(stdout, "[XOR]          Target: %.0lf\n", n3_net->targets[0]);
      fprintf(stdout, "[XOR] Overall success: %.3lf%%\n", (stats[0] * 100.f) / (double) (stats[0] + stats[1]));
    }

    if ( args.learning ) {
      n3l_backward_propagation(n3_net);
    }

    free(n3_net->inputs);
    free(n3_net->outputs);
    free(n3_net->targets);
  } while (--args.iterations);

  if ( args.progress ) {
    fprintf(stdout, "\n");
  }

  if ( args.save_result ) {
    if ( !(of = fopen(args.save_filename, "w")) ) {
      fprintf(stderr, "Cannot save results. Open failed ( %s )", args.save_filename);
    }
    else {
      n3l_save(n3_net, of);
      fclose(of);
    }
  }

  n3l_free(n3_net);
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
