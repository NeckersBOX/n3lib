objects = n3_act.o n3_forward.o n3_init.o n3_logger.o n3_backward.o

all_flags = $(flags) -fPIC

ifeq ($(debug), true)
      all_flags += -g -Ddebug_enable -pg
endif

ifeq ($(extra), true)
      all_flags += -Wall -Wextra -ansi -pedantic
endif

%.o: src/%.c
	gcc -c $< -o $@ $(all_flags)

n3lib: $(objects)
	gcc -shared -o libn3l.so -lm $(objects)

clean:
	rm -vf $(objects)
