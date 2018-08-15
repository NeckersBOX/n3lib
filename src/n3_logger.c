#define _ISOC99_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include "n3_header.h"

#define return_if_fail(arg) if ( !arg ) { return; }

void n3l_log_start(N3LLogger *nl, const char *fname, N3LLogType type)
{
  return_if_fail(nl && type != N3LLogNone);

  if ( type <= nl->verbosity ) {
    fprintf(nl->log_file, "[N3Lib] [%d] [%s] -->>\n", type, fname);
  }
}

void n3l_log_end(N3LLogger *nl, const char *fname, N3LLogType type)
{
  return_if_fail(nl && type != N3LLogNone);

  if ( type <= nl->verbosity ) {
    fprintf(nl->log_file, "[N3Lib] [%d] [%s] <<--\n", type, fname);
  }
}

void n3l_log(N3LLogger *nl, const char *fname, N3LLogType type, const char *message, ...)
{
  return_if_fail(nl && type != N3LLogNone);

  char *str = NULL;
  va_list ap;

  if ( type <= nl->verbosity ) {
    str = malloc((2 << 12) * sizeof(char));
    va_start(ap, message);

    vsnprintf(str, 2 << 12, message, ap);
    fprintf(nl->log_file, "[N3Lib] [%d] [%s] %s\n", type, fname, str);
    free(str);
    fflush(nl->log_file);

    va_end(ap);
  }
}
