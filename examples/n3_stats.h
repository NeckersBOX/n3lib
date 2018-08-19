#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <n3l/n3lib.h>

struct _stat_per_cycle {
  double mne, tne;
  double mns, tns;
  double tm_elapsed;
};

struct _stats {
  struct timeval tvstart, tvend;
  double *mne, tne;
  double *mns, tns;
  uint64_t sample_size, iterations;
  N3LData *status;
  struct _stat_per_cycle *data;
};

struct _stats n3_stats_start(N3LData *network, uint64_t train_size, uint64_t iterations)
{
  struct _stats stat;

  stat.status = network;
  stat.sample_size = train_size;
  stat.iterations = iterations;
  stat.data = (struct _stat_per_cycle *) malloc(iterations * sizeof(struct _stat_per_cycle));
  stat.mns = (double *) calloc(train_size, sizeof(double));
  stat.mne = (double *) calloc(train_size, sizeof(double));
  stat.tne = 0.0;
  stat.tns = 0.0;

  gettimeofday(&(stat.tvstart), NULL);

  return stat;
}

void n3_stats_cycle(struct _stats *stat, uint64_t iteration, bool success)
{
  double curr_err = 0.0;
  uint64_t j;

  gettimeofday(&(stat->tvend), NULL);

  for ( j = 0; j < stat->status->args->out_size; ++j ) {
    curr_err += pow(stat->status->targets[j] - stat->status->outputs[j], 2);
  }
  curr_err /= 2.0;

  stat->mne[iteration % stat->sample_size] = curr_err;
  stat->tne += curr_err;

  stat->mns[iteration % stat->sample_size] = success ? 1 : 0;
  stat->tns += success ? 1 : 0;

  stat->data[iteration].mne = 0.0;
  stat->data[iteration].mns = 0.0;
  for ( j = 0; j < stat->sample_size; ++j ) {
    stat->data[iteration].mne += stat->mne[j];
    stat->data[iteration].mns += stat->mns[j];
  }
  stat->data[iteration].mne /= (double) stat->sample_size;
  stat->data[iteration].mns /= (double) stat->sample_size;
  stat->data[iteration].tne = stat->tne / ((double) iteration + 1.f);
  stat->data[iteration].tns = stat->tns / ((double) iteration + 1.f);

  stat->data[iteration].tm_elapsed =
    ((double) (stat->tvend.tv_usec - stat->tvstart.tv_usec)) / 1000000.f +
    ((double) (stat->tvend.tv_sec  - stat->tvstart.tv_sec));
}

void n3_stats_end(struct _stats *stat)
{
  gettimeofday(&(stat->tvend), NULL);
  stat->tne /= (double) stat->iterations;
  stat->tns /= (double) stat->iterations;
}

void n3_stats_free(struct _stats *stat)
{
  free(stat->data);
  free(stat->mne);
  free(stat->mns);
}

bool n3_stats_to_csv(struct _stats *stat, char *filename)
{
  FILE *of;
  uint64_t row;

  if ( !(of = fopen(filename, "w")) ) {
    return false;
  }

  fprintf(of,
    "ITERATION,"
    "TNE %% ( Total Network Error ),"
    "TNS %% ( Total Network Success ),"
    "MNE %% ( Mobile Network Error ),"
    "MNS %% ( Mobile Network Success ),"
    "Time Elapsed (s)\n");

  for ( row = 0; row < stat->iterations; ++row ) {
    fprintf(of, "%ld,%lf,%lf,%lf,%lf,%lf\n",
      row + 1,
      stat->data[row].tne * 100.f,
      stat->data[row].tns * 100.f,
      stat->data[row].mne * 100.f,
      stat->data[row].mns * 100.f,
      stat->data[row].tm_elapsed);
  }
  fclose(of);

  return true;
}
