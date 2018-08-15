# N3 Library
A tiny C library for building neural network.

**STATUS:** PRE-ALPHA

## Build the library
In the main folder:
`make n3lib`

## Use the library in your project ( local )
If your library build path is `/home/myuser/N3Lib` type the following command:
```
cd /home/myuser/N3Lib
```

Next you need to include the library header. If your file is `N3Lib/examples/xor.c` you should add:
```
#include ".../n3lib.h"
```

Finally, to compile your project:
```
gcc examples/xor.c -o xor -L/home/myuser/N3Lib -ln3l -lm -lpthread
```

**NOTE:** If it works, in the future, I will add the install procedure by Makefile.

## Documentation
Awww, you're so cute. No documentation at this point of development. Enjoy :)
