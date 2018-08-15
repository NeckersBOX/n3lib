#ifndef _N3L_LOGGER_
#define _N3L_LOGGER_

#define N3L_LCRITICAL(state,...)        n3l_log(state, __FUNCTION__, N3LLogCritical, __VA_ARGS__)
#define N3L_LHIGH(state,...)            n3l_log(state, __FUNCTION__, N3LLogHigh,     __VA_ARGS__)
#define N3L_LMEDIUM(state,...)          n3l_log(state, __FUNCTION__, N3LLogMedium,   __VA_ARGS__)
#define N3L_LLOW(state,...)             n3l_log(state, __FUNCTION__, N3LLogLow,      __VA_ARGS__)
#define N3L_LPEDANTIC(state,...)        n3l_log(state, __FUNCTION__, N3LLogPedantic, __VA_ARGS__)

#define N3L_LCRITICAL_START(state)      n3l_log_start(state, __FUNCTION__, N3LLogCritical)
#define N3L_LHIGH_START(state)          n3l_log_start(state, __FUNCTION__, N3LLogHigh)
#define N3L_LMEDIUM_START(state)        n3l_log_start(state, __FUNCTION__, N3LLogMedium)
#define N3L_LLOW_START(state)           n3l_log_start(state, __FUNCTION__, N3LLogLow)
#define N3L_LPEDANTIC_START(state)      n3l_log_start(state, __FUNCTION__, N3LLogPedantic)

#define N3L_LCRITICAL_END(state)        n3l_log_end(state, __FUNCTION__, N3LLogCritical)
#define N3L_LHIGH_END(state)            n3l_log_end(state, __FUNCTION__, N3LLogHigh)
#define N3L_LMEDIUM_END(state)          n3l_log_end(state, __FUNCTION__, N3LLogMedium)
#define N3L_LLOW_END(state)             n3l_log_end(state, __FUNCTION__, N3LLogLow)
#define N3L_LPEDANTIC_END(state)        n3l_log_end(state, __FUNCTION__, N3LLogPedantic)

extern void n3l_log(N3LLogger *, const char *, N3LLogType, const char *, ...);
extern void n3l_log_end(N3LLogger *, const char *, N3LLogType);
extern void n3l_log_start(N3LLogger *, const char *, N3LLogType);

#endif
