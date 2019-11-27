to compile sosfilt.so, do:

gcc -c -fPIC sosfilt.c -o sosfilt.o
gcc sosfilt.o -shared -o sosfilt.so
