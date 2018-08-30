objects := $(patsubst %.c,%.o,$(wildcard src/*.c))

version = 1.2.9
all_flags := $(flags) -fPIC
soname := libn3l.so.$(version)
aname := libn3l.a.$(version)
pkgconfig = n3l.pc
header_files = $(wildcard src/*.h)
prefix := $(destdir)/usr
library_path := $(prefix)/lib
include_path := $(prefix)/include/n3l
man_path := $(prefix)/local/share/man/
private_lib = -lm -lpthread

ifeq ($(debug), true)
      all_flags += -g -Ddebug_enable -pg
endif

ifeq ($(extra), true)
      all_flags += -Wall -Wextra -ansi -pedantic
endif

src/%.o: src/%.c
	gcc -c $< -o $@ $(all_flags)

build: $(objects)
	gcc -shared -Wl,-soname,libn3l.so -o $(soname) $(objects) $(private_lib)
	ar rcs $(aname) $(objects)
	@echo -n "Generating .pc file... "
	@echo -e "prefix=$(prefix)\n"\
					 "exec_prefix=$(prefix)\n"\
					 "libdir=$(library_path)\n"\
					 "includedir=$(include_path)\n"\
					 "\n"\
					 "Name: N3 Library\n"\
					 "Description: A tiny C library for building neural network.\n"\
					 "Version: $(version)\n"\
					 "Requires: \n"\
					 "Libs: -L$(library_path) -ln3l $(private_lib)\n"\
					 "Libs.private: $(private_lib)\n"\
					 "Cflags: -I$(include_path)" > $(pkgconfig)
	@echo "Done"


doc:
	doxygen n3lib.doxygen.conf
	rm -vf docs/man/man3/_home_*
	rm -vf docs/man/man3/md*

clean:
	rm -vf $(objects) $(soname) $(aname) $(pkgconfig)

install: $(soname)
	 install -D $(soname) $(library_path)/$(soname)
	 link $(library_path)/$(soname) $(library_path)/libn3l.so
	 install -D $(aname) $(library_path)/$(aname)
	 link $(library_path)/$(aname) $(library_path)/libn3l.a
	 install -D n3lib.h $(include_path)/n3lib.h
	 @echo -n "Installing header... "
	 $(foreach header, $(header_files), $(shell install -D -t $(include_path)/src $(header)))
	 @echo "Done."
	 install -D $(pkgconfig) $(library_path)/pkgconfig/$(pkgconfig)
	 @echo -n "Installing manpages... "
	 mkdir $(man_path) -p
	 cp docs/man/man3 $(man_path) -r
	 @echo "Done."
	 mandb

uninstall:
	 unlink $(library_path)/libn3l.so
	 unlink $(library_path)/libn3l.a
	 rm -vr $(library_path)/$(soname) $(include_path) \
	  $(library_path)/$(aname) \
	  $(library_path)/pkgconfig/$(pkgconfig) \
		$(man_path)/man3/n3l* $(man_path)/man3/__n3* \
		$(man_path)/man3/_n3*
