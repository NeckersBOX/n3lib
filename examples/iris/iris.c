#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <n3l/n3lib.h>
#include "../n3_example.h"
#include "../n3_stats.h"

struct _iris_data {
  double *inputs;
  double *outputs;
  uint8_t result_code;
};

struct _iris_data get_data(FILE *in);
void iris_classification(struct user_args);

int main(int argc, char *argv[])
{
  struct user_args args = {
    false, false, false, false, false,
    "iris.n3l", "iris.n3l",
    0.05f, 0, 1,
    "iris.csv",
    N3LLogNone
  };

  bool free_save_filename;

  srand(time(NULL));
  fprintf(stdout, "IRIS Example - N3L v. %s\n", N3L_VERSION);
  fprintf(stdout, "(c) 2018 - Davide Francesco Merico <hds619 [at] gmail [dot] com>\n\n");

  free_save_filename = n3_example_arguments_parser(argc, argv, &args);

  iris_classification(args);
  if ( free_save_filename ) {
    free(args.save_filename);
  }

  if ( args.read_result ) {
    free(args.read_filename);
  }

  return 0;
}

void iris_classification(struct user_args args)
{
  N3LArgs n3_args;
  N3LData *n3_net;
  N3LLogger n3_logger = { stdout, args.verbose };
  uint64_t stats[2] = { 0, 0 };
  uint8_t result_code;
  struct _iris_data data;
  FILE *of;
#ifdef N3L_ENABLE_STATS
  struct _stats iris_stat;
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
  n3_args.in_size = 4;
  n3_args.h_size = 3;
  n3_args.h_layers = 1;
  n3_args.out_size = 3;
  n3_args.act_h = N3LSigmoid;
  n3_args.act_out = N3LSigmoid;

  n3_args.logger = &n3_logger;
  n3_net = n3l_build(n3_args, &n3l_rnd_weight);

  #ifdef N3L_ENABLE_STATS
    iris_stat = n3_stats_start(n3_net, 150, args.iterations);
  #endif

  if ( !(of = fopen("iris.data", "r")) ) {
    fprintf(stderr, "[IRIS] Cannot found the data file. Exit.");
    exit(1);
  }

  do {
    if ( !args.mute ) {
      fprintf(stdout, "[IRIS] -- Iteration %ld on %ld --\n",
        stats[0] + stats[1] + 1, args.iterations + stats[0] + stats[1]);
    }

    data = get_data(of);
    n3_net->inputs = data.inputs;
    n3_net->targets = data.outputs;

    if ( !args.mute ) {
      fprintf(stdout, "[IRIS]         Input 0: %lf\n", n3_net->inputs[0]);
      fprintf(stdout, "[IRIS]         Input 1: %lf\n", n3_net->inputs[1]);
      fprintf(stdout, "[IRIS]         Input 2: %lf\n", n3_net->inputs[2]);
      fprintf(stdout, "[IRIS]         Input 3: %lf\n", n3_net->inputs[3]);
    }
    n3_net->outputs = n3l_forward_propagation(n3_net);

    if ( !args.mute ) {
      fprintf(stdout, "[IRIS]        Output 0: %lf\n", n3_net->outputs[0]);
      fprintf(stdout, "[IRIS]        Target 0: %.0lf\n", n3_net->targets[0]);
      fprintf(stdout, "[IRIS]        Output 1: %lf\n", n3_net->outputs[1]);
      fprintf(stdout, "[IRIS]        Target 1: %.0lf\n", n3_net->targets[1]);
      fprintf(stdout, "[IRIS]        Output 2: %lf\n", n3_net->outputs[2]);
      fprintf(stdout, "[IRIS]        Target 2: %.0lf\n", n3_net->targets[2]);
    }

    if ( n3_net->outputs[0] > n3_net->outputs[1] && n3_net->outputs[0] > n3_net->outputs[2] ) {
      result_code = 0;
    }
    else if ( n3_net->outputs[1] > n3_net->outputs[0] && n3_net->outputs[1] > n3_net->outputs[2] ) {
      result_code = 1;
    }
    else if ( n3_net->outputs[2] > n3_net->outputs[0] && n3_net->outputs[2] > n3_net->outputs[1] ){
      result_code = 2;
    }
    else {
      result_code = 0xFF;
    }

    if ( !args.mute ) {
      fprintf(stdout, "[IRIS]   Result code: %d\n", result_code);
      fprintf(stdout, "[IRIS]   Target code: %d\n", data.result_code);
    }

    ++stats[result_code == data.result_code ? 0 : 1];

    if ( !args.mute ) {
      fprintf(stdout, "[IRIS] Overall success: %.3lf%%\n", (stats[0] * 100.f) / (double) (stats[0] + stats[1]));
    }

    if ( args.learning ) {
      n3l_backward_propagation(n3_net);
    }

    #ifdef N3L_ENABLE_STATS
        n3_stats_cycle(&iris_stat, stats[0] + stats[1] - 1, result_code == data.result_code);

        if ( args.progress ) {
          fprintf(stdout, "\r[XOR] Iteration %ld on %ld - MNE: %lf - MNS: %.3lf%%",
            stats[0] + stats[1], args.iterations + stats[0] + stats[1] - 1,
            iris_stat.data[stats[0] + stats[1] - 1].mne,
            iris_stat.data[stats[0] + stats[1] - 1].mns * 100.f);
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
  fclose(of);

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
    n3_stats_end(&iris_stat);
    n3_stats_to_csv(&iris_stat, args.csv_filename);
    n3_stats_free(&iris_stat);
  #endif

  n3l_free(n3_net);
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
