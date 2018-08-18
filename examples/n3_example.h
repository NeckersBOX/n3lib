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

bool n3_example_arguments_parser(int argc, char *argv[], struct user_args *args)
{
  int opt, v;
  bool free_save_filename = false;

  while ((opt = getopt(argc, argv, "b:hi:lmo:pr:sv:")) != -1) {
    switch(opt) {
      case 'b':
        sscanf(optarg, "%lf", &(args->bias));
        break;
      case 'l':
        args->learning = true;
        if ( optarg ) {
          sscanf(optarg, "%lf", &(args->learning_rate));
        }
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
        free_save_filename = true;
        args->save_filename = strdup(optarg);
        break;
     case 's':
        args->save_result = true;
        break;
     case 'r':
        args->read_result = true;
        args->read_filename = strdup(optarg);
        break;
     case 'v':
        v = atoi(optarg);
        args->verbose = v < 0 ? -1 : (v > N3LLogPedantic) ? N3LLogPedantic : v;
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

  if ( args->mute ) {
    args->verbose = -1;
  }

  if ( args->progress ) {
    args->mute = true;
    args->verbose = -1;
  }

  return free_save_filename;
}
