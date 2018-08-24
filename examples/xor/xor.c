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
void xor_operation(struct n3_example_args);

int main(int argc, char *argv[])
{
  struct n3_example_args args = {
    false, false, false, 1.0, 0.0, 1, NULL, NULL, NULL
  };

  srand(time(NULL));

  n3_example_arguments_parser(argc, argv, &args);

  if ( !args.mute ) {
    fprintf(stdout, "XOR Example - N3L v. %s\n", N3L_VERSION);
    fprintf(stdout, "(C) 2018 - Davide Francesco Merico <hds619 [at] gmail [dot] com>\n\n");
  }

  xor_operation(args);

  if ( args.save_filename ) {
    free(args.save_filename);
  }

  if ( args.read_filename ) {
    free(args.read_filename);
  }

  return 0;
}

void xor_operation(struct n3_example_args args)
{
  N3LArgs n3_args;
  N3LNetwork *net;
  uint64_t stats[2] = { 0, 0 };
  uint64_t hsize[1] = { 3 };
  N3LActType act_h[1] = { N3LSigmoid };
  double *outs;

#ifdef N3L_ENABLE_STATS
  struct _stats xor_stat;
#endif

  if ( !args.mute ) {
    fprintf(stdout, "Simulation property:\n");
    fprintf(stdout, "Read from file: %s\n", args.read_filename ? : "NULL");
    fprintf(stdout, "  Save to file: %s\n", args.save_filename ? : "NULL");
    fprintf(stdout, " Learning rate: %lf\n", args.learning_rate);
    fprintf(stdout, "    Iterations: %ld\n\n", args.iterations);
  }

  n3_args = n3l_misc_init_arg();
  n3_args.bias = args.bias;
  n3_args.in_size = 2;
  n3_args.h_size = hsize;
  n3_args.h_layers = 1;
  n3_args.out_size = 1;
  n3_args.act_h = act_h;
  n3_args.act_out = N3LSigmoid;

  net = n3l_network_build(n3_args, args.learning_rate);

#ifdef N3L_ENABLE_STATS
  xor_stat = n3_stats_start(net, 4, args.iterations);
#endif

  do {
    if ( !args.mute ) {
      fprintf(stdout, "[XOR] -- Iteration %ld on %ld --\n",
        stats[0] + stats[1] + 1, args.iterations + stats[0] + stats[1]);
    }

    net->inputs = get_inputs_batch_mode();
    if ( !args.mute ) {
      fprintf(stdout, "[XOR]         Input 0: %.0lf\n", net->inputs[0]);
      fprintf(stdout, "[XOR]         Input 1: %.0lf\n", net->inputs[1]);
    }
    outs = n3l_forward_propagation(net);

    if ( !args.mute ) {
      fprintf(stdout, "[XOR]          Output: %lf\n", outs[0]);
    }

    net->targets = get_targets(net->inputs);
    ++stats[round(outs[0]) == net->targets[0] ? 0 : 1];

    if ( !args.mute ) {
      fprintf(stdout, "[XOR]          Target: %.0lf\n", net->targets[0]);
      fprintf(stdout, "[XOR] Overall success: %.3lf%%\n", (stats[0] * 100.f) / (double) (stats[0] + stats[1]));
    }

    if ( args.learning ) {
      n3l_backward_propagation(net);
    }

#ifdef N3L_ENABLE_STATS
    n3_stats_cycle(&xor_stat, net->targets, outs, stats[0] + stats[1] - 1, round(outs[0]) == net->targets[0]);

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

    free(net->inputs);
    free(outs);
    free(net->targets);
  } while (--args.iterations);

  if ( args.progress ) {
    fprintf(stdout, "\n");
  }

  if ( args.save_filename ) {
    n3l_network_save(net, n3_args, args.save_filename);
  }

#ifdef N3L_ENABLE_STATS
  n3_stats_end(&xor_stat);
  n3_stats_to_csv(&xor_stat, args.csv_filename);
  n3_stats_free(&xor_stat);
#endif

  n3l_network_free(net);
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
