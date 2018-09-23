#include <stdio.h>

struct n3_example_args {
  bool learning;
  bool mute;
  bool progress;
  double learning_rate;
  double bias;
  uint64_t iterations;
  char *save_filename;
  char *read_filename;
  char *csv_filename;
};

void n3_example_arguments_parser(int argc, char *argv[], struct n3_example_args *args)
{
  int opt;
  int cores;

  while ((opt = getopt(argc, argv, "b:j:hi:lmo:pr:t:")) != -1) {
    switch(opt) {
      case 'b':
        sscanf(optarg, "%lf", &(args->bias));
        break;
      case 'j':
        if ( (cores = atoi(optarg)) < 1 ) {
          fprintf(stderr, "Wrong threads number: %d\n", cores);
        }
        else {
          N3L_THREADS_CORES = cores;
        }
      case 'l':
        args->learning = true;
        break;
      case 'm':
        args->mute = true;
        break;
      case 'p':
        args->progress = true;
        break;
      case 'i':
        args->iterations = atoi(optarg);
        if ( args->iterations <= 0 ) {
          fprintf(stderr, "Wrong iteration argument: %s\n", optarg);
          args->iterations = 1;
        }
        break;
     case 'o':
        args->save_filename = strdup(optarg);
        break;
     case 't':
        args->csv_filename = strdup(optarg);
        break;
     case 'r':
        args->read_filename = strdup(optarg);
        break;
     case 'h':
        fprintf(stdout, "Usage: %s [options]\n\n", *argv);
        fprintf(stdout, "Options:\n");
        fprintf(stdout, "\t-b [n]         Set the bias term in the network. Default: 0\n");
        fprintf(stdout, "\t-h             Show this help with the options list.\n");
        fprintf(stdout, "\t-i [n]         Number of iterations. Default: 1\n");
        fprintf(stdout, "\t-l             Enable learning with backpropagation.\n");
        fprintf(stdout, "\t-m             No log at all. Note: Disable -v option.\n");
        fprintf(stdout, "\t-o [filename]  After the number of iterations provided, save the");
        fprintf(stdout, " neural network state. Note: It works only if used with option -s.\n");
        fprintf(stdout, "\t-p             Enable the progress viewer. Active -m, Disable -v.\n");
        fprintf(stdout, "\t-r [filename]  Initialize the neural network reading the number of");
        fprintf(stdout, " neurons, layers and weights from a previous state saved.\n");
        fprintf(stdout, "\t-t [filename]  Specified the CSV filename where the data are saved.\n");
        exit(0);
        break;
    }
  }

  if ( args->progress ) {
    args->mute = true;
  }
}

void n3_print_net(N3LNetwork *net)
{
  N3LLayer *layer;
  N3LNeuron *neuron;
  N3LWeight *weight;
  uint64_t size, j, k, z;

  printf("Network\n");
  printf("+ learning_rate: %lf\n", net->learning_rate);
  size = n3l_layer_count(net->lhead);
  printf("- layer_count: %ld\n", size);
  for ( layer = net->lhead, j = 0; layer; layer = layer->next, ++j ) {
    printf("- Layer %ld\n", j);
    printf("  + type: %s\n", layer->type == N3LInputLayer ? "N3LInputLayer" : (layer->type == N3LHiddenLayer ? "N3LHiddenLayer" : (layer->type == N3LOutputLayer ? "N3LOutputLayer" : "Unknown")) );
    size = n3l_neuron_count(layer->nhead);
    printf("  - neuron_count: %ld\n", size);

    for ( neuron = layer->nhead, k = 0; neuron; neuron = neuron->next, ++k ) {
      printf("  - Neuron %ld\n", k);
      printf("    + act_type: %d\n", (int) neuron->act_type);
      printf("    + bias: %s\n", neuron->bias ? "true" : "false");
      printf("    + ref: %ld\n", neuron->ref);
      printf("    + input: %lf\n", neuron->input);

      size = n3l_neuron_count_weights(neuron->whead);
      printf("    - weights_count: %ld\n", size);
      for ( weight = neuron->whead, z = 0; weight; weight = weight->next, ++z ) {
        printf("    - Weight %ld\n", z);
        printf("      + value: %lf\n", weight->value);
        printf("      + target_ref: %ld\n", weight->target_ref);
      }
    }
  }
}
