#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <n3l/n3lib.h>
#include "../n3_example.h"
#ifdef N3L_ENABLE_STATS
#include "../n3_stats.h"
#endif

double *get_inputs_batch_mode(void);
double *get_targets(double *inputs);
void xor_operation(struct user_args);

int main(int argc, char *argv[])
{
  struct user_args args = { false, false, false, false, false, "xor.n3l", "xor.n3l", 1.f, 0, 1, N3LLogNone };
  bool free_save_filename = false;

  srand(time(NULL));
  fprintf(stdout, "XOR Example - N3L v. %s\n", N3L_VERSION);
  fprintf(stdout, "(c) 2018 - Davide Francesco Merico <hds619 [at] gmail [dot] com>\n\n");

  free_save_filename = n3_example_arguments_parser(argc, argv, &args);

  xor_operation(args);
  if ( free_save_filename ) {
    free(args.save_filename);
  }

  if ( args.read_result ) {
    free(args.read_filename);
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
#ifdef N3L_ENABLE_STATS
  struct _stats xor_stat;
#endif

  if ( !args.mute ) {
    fprintf(stdout, "Simulation property:\n");
    fprintf(stdout, "Read from file: %s ( %s )\n", BOOL_STR(args.read_result), args.read_filename);
    fprintf(stdout, "  Save to file: %s ( %s )\n", BOOL_STR(args.save_result), args.save_filename);
    fprintf(stdout, " Learning rate: %lf\n", args.learning_rate);
    fprintf(stdout, "     Verbosity: %d\n", args.verbose);
    fprintf(stdout, "    Iterations: %ld\n\n", args.iterations);
  }

  n3_args = n3l_get_default_args();
  n3_args.read_file = args.read_result;
  n3_args.in_filename = args.read_filename;
  n3_args.learning_rate = args.learning_rate;
  n3_args.bias = args.bias;
  n3_args.in_size = 2;
  n3_args.out_size = 1;
  n3_args.h_size = 3;
  n3_args.h_layers = 1;
  n3_args.act_h = N3LSigmoid;
  n3_args.act_out = N3LSigmoid;

  n3_args.logger = &n3_logger;
  n3_net = n3l_build(n3_args, &n3l_rnd_weight);

#ifdef N3L_ENABLE_STATS
  xor_stat = n3_stats_start(n3_net, 4, args.iterations);
#endif

  do {
    if ( !args.mute ) {
      fprintf(stdout, "[XOR] -- Iteration %ld on %ld --\n",
        stats[0] + stats[1] + 1, args.iterations + stats[0] + stats[1]);
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

#ifdef N3L_ENABLE_STATS
    n3_stats_cycle(&xor_stat, stats[0] + stats[1] - 1, round(n3_net->outputs[0]) == n3_net->targets[0]);

    if ( args.progress ) {
      fprintf(stdout, "\r[XOR] Iteration %ld on %ld - MNE: %lf - MNS: %.3lf%%",
        stats[0] + stats[1], args.iterations + stats[0] + stats[1] - 1,
        xor_stat.data[stats[0] + stats[1] - 1].mne,
        xor_stat.data[stats[0] + stats[1] - 1].mns * 100.f);
    }
#else
    if ( args.progress ) {
      fprintf(stdout, "\r[XOR] Iteration %ld on %ld - Overall: %.3lf%%",
        stats[0] + stats[1], args.iterations + stats[0] + stats[1] - 1,
        (stats[0] * 100.f) / (double) (stats[0] + stats[1]));
    }
#endif

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

#ifdef N3L_ENABLE_STATS
  n3_stats_end(&xor_stat);
  n3_stats_to_csv(&xor_stat, "xor_stats.csv");
  n3_stats_free(&xor_stat);
#endif

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
