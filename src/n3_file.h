#ifndef _N3_FILE_
#define _N3_FILE_

extern double *     n3l_file_get_csv_data       (FILE *, uint64_t, uint64_t, uint64_t, N3LCSVData);
extern N3LNetwork * n3l_file_import_network     (char *);
extern bool         n3l_file_export_network     (N3LNetwork *, char *);

#endif
