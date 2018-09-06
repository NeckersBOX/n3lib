/**
 * @file n3_file.c
 * @author Davide Francesco Merico
 * @brief This file contains functions to import and save a network state.
 */
#include <stdlib.h>
#include "n3_header.h"
#include "n3_neuron.h"
#include "n3_layer.h"

/**
 * @brief Internal function to read weight from file during network initialization.
 *
 * @param data Pointer to an already opened file. (FILE type)
 * @return The weight read from the FILE last read position.
 *
 * @see n3l_file_import_network
 */
double __n3l_get_weight_from_file(void *data)
{
  double weight;
  FILE *n3_file = (FILE *) data;

  fread(&weight, sizeof(double), 1, n3_file);

  return weight;
}

/**
 * @brief Import the network state from a previously saved file.
 *
 * @param filename Previously saved file name with n3l_file_export_network().
 * @return The N3LNetwork saved if successfully read, otherwise NULL.
 *
 * @see n3l_file_export_network, _n3l_network, n3l_network_build
 */
N3LNetwork *n3l_file_import_network(char *filename)
{
  N3LNetwork *net;
  FILE *n3_file;
  N3LArgs args;
  double learning_rate;
  uint64_t lsize, nsize, wsize;
  N3LLayer *layer = NULL;
  N3LNeuron *neuron = NULL;
  N3LWeight *weight = NULL;
  N3LLayerType ltype;
  N3LActType act_type;

  if ( !(n3_file = fopen(filename, "r")) ) {
    return NULL;
  }

  net = (N3LNetwork *) malloc(sizeof(N3LNetwork));
  net->inputs = NULL;
  net->targets = NULL;
  net->lhead = NULL;
  net->ltail = NULL;
  fread(&(net->learning_rate), sizeof(double), 1, n3_file);
  fread(&lsize, sizeof(uint64_t), 1, n3_file);

  while ( lsize-- ) {
    fread(&ltype, sizeof(N3LLayerType), 1, n3_file);
    layer = n3l_layer_build_after(layer, ltype);

    net->lhead = net->lhead ? : layer;
    net->ltail = layer;

    fread(&nsize, sizeof(uint64_t), 1, n3_file);
    while ( nsize-- ) {
      fread(&act_type, sizeof(N3LActType), 1, n3_file);
      neuron = n3l_neuron_build_after(layer->ntail, act_type);
      fread(&(neuron->bias), sizeof(bool), 1, n3_file);
      fread(&(neuron->ref), sizeof(uint64_t), 1, n3_file);
      fread(&(neuron->input), sizeof(double), 1, n3_file);

      layer->nhead = layer->nhead ? : neuron;
      layer->ntail = neuron;

      fread(&wsize, sizeof(uint64_t), 1, n3_file);
      while ( wsize-- ) {
        weight = (N3LWeight *) malloc(sizeof(N3LWeight));
        fread(&(weight->value), sizeof(double), 1, n3_file);
        fread(&(weight->target_ref), sizeof(uint64_t), 1, n3_file);
        weight->next = neuron->whead;
        neuron->whead = weight;
      }
    }
  }

  fclose(n3_file);

  return net;
}

/**
 * @brief Export the current network state to the chosen file.
 *
 * @param net Initialized network state.
 * @param filename File name into write the current network state.
 * @return TRUE if correctly executed, otherwise FALSE.
 *
 * @see n3l_file_import_network, _n3l_network, n3l_network_free
 */
bool n3l_file_export_network(N3LNetwork *net, char *filename)
{
  FILE *n3_file;
  N3LLayer *layer;
  N3LNeuron *neuron;
  N3LWeight *weight;
  uint64_t size;

  if ( !(n3_file = fopen(filename, "w")) ) {
    return false;
  }

  fwrite(&(net->learning_rate), sizeof(double), 1, n3_file);

  size = n3l_layer_count(net->lhead);
  fwrite(&size, sizeof(uint64_t), 1, n3_file);
  for ( layer = net->lhead; layer; layer = layer->next ) {
    fwrite(&(layer->type), sizeof(N3LLayerType), 1, n3_file);

    size = n3l_neuron_count(layer->nhead);
    fwrite(&size, sizeof(uint64_t), 1, n3_file);
    for ( neuron = layer->nhead; neuron; neuron = neuron->next ) {
      fwrite(&(neuron->act_type), sizeof(N3LActType), 1, n3_file);
      fwrite(&(neuron->bias), sizeof(bool), 1, n3_file);
      fwrite(&(neuron->ref), sizeof(uint64_t), 1, n3_file);
      fwrite(&(neuron->input), sizeof(double), 1, n3_file);

      size = n3l_neuron_count_weights(neuron->whead);
      fwrite(&size, sizeof(uint64_t), 1, n3_file);
      for ( weight = neuron->whead; weight; weight = weight->next ) {
        fwrite(&(weight->value), sizeof(double), 1, n3_file);
        fwrite(&(weight->target_ref), sizeof(uint64_t), 1, n3_file);
      }
    }
  }

  return true;
}

/**
 * @brief Get inputs from an already opened csv file.
 *
 * Get a single line, from the \p csv current position.
 *
 * @param csv Already opened in read mode csv file.
 * @param row_offset If 0 read from the current position, otherwise set the position
 * to the \p row_offset line.
 * @param col_offset Start column from 0 to get the first data column
 * @param size Number of data to read from the \p col_offset index
 * @param data_parser A custom data parser, each data will be passed as string to
 * this function. If you don't need custom parser this could be set to NULL.
 * @return A dynamic allocated array of inputs or NULL on error.
 *
 * @note The returned pointer should be free() manually.
 * @see N3LCSVData
 */
double *n3l_file_get_csv_data(FILE *csv, uint64_t row_offset, uint64_t col_offset, uint64_t size, N3LCSVData data_parser)
{
 	double *data;
 	uint64_t row_skipped = 0, col_skipped = 0, data_index;
 	uint64_t r_size = 1, line_start, line_end, tok_s, tok_e;
 	bool flag = false;
 	char c, *raw_data;

 	if ( !csv || !size ) {
 		return NULL;
 	}

 	/* Move row offset to the chosen one */
 	while ( row_skipped < row_offset && (c = fgetc(csv)) != EOF ) {
 		if ( c == '\n' ) {
 			++row_skipped;
 		}
 	}

 	if ( row_skipped < row_offset ) {
 		return NULL;
 	}

 	/* Validate fields number */
 	line_start = ftell(csv);
 	while ( (c = fgetc(csv)) != EOF ) {
 		if ( c == '\n' ) {
 			break;
 		}
 		else if ( c == '"' ) {
 			flag = !flag;
 		}
 		else if ( (c == ',' || c == '\t' || c == ';') && !flag ) {
 			++r_size;
 		}
 	}

 	line_end = ftell(csv);
 	if ( line_end == line_start || r_size < (size + col_offset) ) {
 		return NULL;
 	}

 	fseek(csv, line_start, SEEK_SET);

 	/* Move col offset to the chosen one */
 	for ( flag = false, col_skipped = 0; col_skipped < col_offset && (c = fgetc(csv)) != EOF; ) {
 		if ( c == '\n' ) {
 			break;
 		}
 		else if ( c == '"' ) {
 			flag = !flag;
 		}
 		else if ( (c == ',' || c == '\t' || c == ';') && !flag ) {
 			col_skipped++;
 		}
 	}

 	if ( col_skipped < col_offset ) {
 		return NULL;
 	}

 	/* Recover field's data */
 	data = (double *) malloc(size * sizeof(double));
 	for ( data_index = 0; data_index < size; ++data_index ) {
 		for ( flag = false, tok_s = tok_e = ftell(csv); (c = fgetc(csv)) != EOF; tok_e++ ) {
 			if ( c == '\n' || ((c == ',' || c == ';' || c == '\t') && !flag) ) {
 				break;
 			}
 			else if ( c == '"' ) {
 				flag = !flag;
 			}
 		}

 		if ( !(tok_e - tok_s) ) {
 			raw_data = (char *) malloc(sizeof(char));
 			*raw_data = 0x00;
 		}
 		else {
 			++tok_e;
 			raw_data = (char *) malloc(((tok_e - tok_s) + 1) * sizeof(char));
 			fseek(csv, tok_s, SEEK_SET);
 			fread(raw_data, sizeof(char), (tok_e - tok_s), csv);
 			raw_data[(tok_e - tok_s) - 1] = 0x00;
 		}

 		if ( data_parser ) {
 			data[data_index] = data_parser(raw_data);
 		}
 		else {
 			if ( !(*raw_data) ) {
 				data[data_index] = 0.0;
 			}
 			else {
 				sscanf(raw_data, "%lf", &(data[data_index]));
 			}
 		}
 	}

 	fseek(csv, line_end, SEEK_SET);

 	return data;
}
