#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <n3l/n3lib.h>
#include "../n3_example.h"
#include "../n3_stats.h"

#ifndef IRIS_DATA_PATH_PREFIX
#define IRIS_DATA_PATH_PREFIX "./"
#endif

struct _iris_data {
  double *inputs;
  double *outputs;
  uint8_t result_code;
};

struct _iris_data get_data(FILE *in);
void iris_classification(struct n3_example_args);

int main(int argc, char *argv[])
{
  struct n3_example_args args = {
    false, false, false, 0.01, 0.0, 1, NULL, NULL, NULL
  };

  srand(time(NULL));

  n3_example_arguments_parser(argc, argv, &args);

  if ( !args.mute ) {
    fprintf(stdout, "IRIS Example - N3L v. %s\n", N3L_VERSION);
    fprintf(stdout, "(c) 2018 - Davide Francesco Merico <hds619 [at] gmail [dot] com>\n\n");
  }

  iris_classification(args);

  if ( args.save_filename ) {
    free(args.save_filename);
  }

  if ( args.read_filename ) {
    free(args.read_filename);
  }

  return 0;
}

void iris_classification(struct n3_example_args args)
{
  N3LArgs params;
  N3LNetwork *net;
  double *outs;
  uint8_t result_code;
  uint64_t success = 0, fail = 0, iter;
  uint64_t   hidden_size[2] = { 3, 3 };
  N3LActType hidden_act[2]  = { N3LSwish, N3LSwish };
  FILE *of;
  struct _iris_data data;

#ifdef N3L_ENABLE_STATS
  struct _stats iris_stat;
#endif

  if ( !args.read_filename ) {
    params = n3l_misc_init_arg();
    params.bias = args.bias;
    params.in_size = 4;
    params.h_layers = 2;
    params.h_size = hidden_size;
    params.act_h = hidden_act;
    params.out_size = 3;
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
    iris_stat = n3_stats_start(net, 150, args.iterations);
  #endif

  if ( !(of = fopen(IRIS_DATA_PATH_PREFIX "iris.data", "r")) ) {
    fprintf(stderr, "[IRIS] Cannot found the data file. Exit.");
    exit(1);
  }

  for ( iter = 0; iter < args.iterations; ++iter ) {
    data = get_data(of);
    net->inputs = data.inputs;
    net->targets = data.outputs;

    if ( !args.mute ) {
      fprintf(stdout, "[IRIS] Iter %ld on %ld\n", iter + 1, args.iterations);
      fprintf(stdout, "[IRIS]    Input 0: %lf\n", net->inputs[0]);
      fprintf(stdout, "[IRIS]    Input 1: %lf\n", net->inputs[1]);
      fprintf(stdout, "[IRIS]    Input 2: %lf\n", net->inputs[2]);
      fprintf(stdout, "[IRIS]    Input 3: %lf\n", net->inputs[3]);
    }

    outs = n3l_forward_propagation(net);

    if ( !args.mute ) {
      fprintf(stdout, "[IRIS]   Output 0: %lf\n", outs[0]);
      fprintf(stdout, "[IRIS]   Target 0: %.0lf\n", net->targets[0]);
      fprintf(stdout, "[IRIS]   Output 1: %lf\n", outs[1]);
      fprintf(stdout, "[IRIS]   Target 1: %.0lf\n", net->targets[1]);
      fprintf(stdout, "[IRIS]   Output 2: %lf\n", outs[2]);
      fprintf(stdout, "[IRIS]   Target 2: %.0lf\n",net->targets[2]);
    }

    if ( outs[0] > outs[1] && outs[0] > outs[2] ) {
      result_code = 0;
    }
    else if ( outs[1] > outs[0] && outs[1] > outs[2] ) {
      result_code = 1;
    }
    else if ( outs[2] > outs[0] && outs[2] > outs[1] ){
      result_code = 2;
    }
    else {
      result_code = 0xFF;
    }

    if ( !args.mute ) {
      fprintf(stdout, "[IRIS]   Result code: %d\n", result_code);
      fprintf(stdout, "[IRIS]   Target code: %d\n", data.result_code);
    }

    if ( result_code == data.result_code ) {
      success++;
    }
    else {
      fail++;
    }

    if ( !args.mute ) {
      fprintf(stdout, "[IRIS]           TNS: %.3lf%%\n", (success * 100.f) / (double) (iter + 1));
    }

    if ( args.learning ) {
      n3l_backward_propagation(net);
    }

    #ifdef N3L_ENABLE_STATS
        n3_stats_cycle(&iris_stat, net->targets, outs, iter, result_code == data.result_code);

        if ( args.progress ) {
          fprintf(stdout, "\r[IRIS] Iteration %ld on %ld - MNE: %lf - MNS: %.3lf%%",
            iter + 1, args.iterations, iris_stat.data[iter].mne, iris_stat.data[iter].mns * 100.f);
        }
    #else
        if ( args.progress ) {
          fprintf(stdout, "\r[IRIS] Iteration %ld on %ld - TNE: %.3lf - TNS: %.3lf%%",
            iter + 1, args.iterations,
            ((net->targets[0] - outs[0]) + (net->targets[1] - outs[1]) + (net->targets[2] - outs[2])) / 3,
            (success * 100.f) / (double) (iter + 1));
        }
    #endif

    free(net->inputs);
    free(outs);
    free(net->targets);
  }
  fclose(of);

  if ( args.progress ) {
    fprintf(stdout, "\n");
  }

  if ( args.save_filename ) {
    n3l_file_export_network(net, args.save_filename);
  }

  #ifdef N3L_ENABLE_STATS
    n3_stats_end(&iris_stat);
    n3_stats_to_csv(&iris_stat, args.csv_filename);
    n3_stats_free(&iris_stat);
  #endif

  n3l_network_free(net);
}

struct _iris_data get_data(FILE *in)
{
  struct _iris_data data;
  char name[32];
  static uint8_t lines_read = 0;
  uint8_t i = -1;

  if ( lines_read == 150 ) {
    fseek(in, 0, SEEK_SET);
    lines_read = 0;
  }
  lines_read++;

  data.inputs = (double *) malloc(4 * sizeof(double));
  fscanf(in, "%lf,%lf,%lf,%lf,", &data.inputs[0], &data.inputs[1], &data.inputs[2], &data.inputs[3]);
  while ( (name[++i] = getc(in)) != '\n');
  name[i] = '\0';

  data.outputs = (double *) calloc(3, sizeof(double));

  if ( !strcasecmp("Iris-virginica", name) ) {
    data.result_code = 0;
    data.outputs[0] = 1.f;
  }
  else if ( !strcasecmp("Iris-setosa", name) ) {
    data.result_code = 1;
    data.outputs[1] = 1.f;
  }
  else if ( !strcasecmp("Iris-versicolor", name) ) {
    data.result_code = 2;
    data.outputs[2] = 1.f;
  }
  else {
    data.result_code = 0xff;
  }

  return data;
}
