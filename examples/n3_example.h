#include <stdio.h>

#define BOOL_STR(b) ((b) ? "True" : "False")

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
  while ((opt = getopt(argc, argv, "b:hi:lmo:pr:t:")) != -1) {
    switch(opt) {
      case 'b':
        sscanf(optarg, "%lf", &(args->bias));
        break;
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
