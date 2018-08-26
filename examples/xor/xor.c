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
  N3LArgs params;
  N3LNetwork *net;
  uint64_t success = 0, fail = 0, iter;

  uint64_t   hidden_size[1] = { 3 };
  N3LActType hidden_act[1]  = { N3LSigmoid };
  double *outs;

#ifdef N3L_ENABLE_STATS
  struct _stats xor_stat;
#endif

  if ( !args.read_filename ) {
    params = n3l_misc_init_arg();
    params.bias = args.bias;
    params.in_size = 2;
    params.h_layers = 1;
    params.h_size = hidden_size;
    params.act_h = hidden_act;
    params.out_size = 1;
    params.act_out = N3LSigmoid;

    net = n3l_network_build(params, args.learning_rate);
  }
  else {
    net = n3l_file_import_network(args.read_filename);
    if ( !net ) {
      fprintf(stderr, "Error while opening file: %s\n", args.read_filename);
      exit(1);
    }
  }

#ifdef N3L_ENABLE_STATS
  xor_stat = n3_stats_start(net, 4, args.iterations);
#endif

  for ( iter = 0; iter < args.iterations; ++iter ) {
    net->inputs = get_inputs_batch_mode();
    if ( !args.mute ) {
      fprintf(stdout, "[XOR] Iter %ld on %ld\n", iter + 1, args.iterations);
      fprintf(stdout, "[XOR]    Input 0: %.0lf\n", net->inputs[0]);
      fprintf(stdout, "[XOR]    Input 1: %.0lf\n", net->inputs[1]);
    }

    outs = n3l_forward_propagation(net);
    net->targets = get_targets(net->inputs);
    if ( round(outs[0]) == net->targets[0] ) {
      ++success;
    }
    else {
      ++fail;
    }

    if ( !args.mute ) {
      fprintf(stdout, "[XOR]     Output: %lf\n", outs[0]);
      fprintf(stdout, "[XOR]     Target: %.0lf\n", net->targets[0]);
      fprintf(stdout, "[XOR]        TNS: %.3lf%%\n", (success * 100.f) / (double) iter);
    }

    if ( args.learning ) {
      n3l_backward_propagation(net);
    }

#ifdef N3L_ENABLE_STATS
    n3_stats_cycle(&xor_stat, net->targets, outs, iter, round(outs[0]) == net->targets[0]);

    if ( args.progress ) {
      fprintf(stdout, "\r[XOR] Iteration %ld on %ld - MNE: %lf - MNS: %.3lf%%",
        iter + 1, args.iterations, xor_stat.data[iter].mne, xor_stat.data[iter].mns * 100.f);
    }
#else
    if ( args.progress ) {
      fprintf(stdout, "\r[XOR] Iteration %ld on %ld - TNE: %.3lf - TNS: %.3lf%%",
        iter + 1, args.iterations, net->targets[0] - outs[0], (success * 100.f) / (double) (iter + 1));
    }
#endif

    free(net->inputs);
    free(outs);
    free(net->targets);
  }

  if ( args.progress ) {
    fprintf(stdout, "\n");
  }

  if ( args.save_filename ) {
    n3l_file_export_network(net, args.save_filename);
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

  inputs[0] = (double) (step & 1);
  inputs[1] = (double) ((step & 2) >> 1);
  step = (step == 3) ? 0 : (step + 1);

  return inputs;
}

double *get_targets(double *inputs)
{
  double *targets = NULL;

  targets = (double *) malloc(sizeof(double));
  targets[0] = (double) ((((uint8_t) inputs[0]) ^ ((uint8_t) inputs[1])) & 1);

  return targets;
}
