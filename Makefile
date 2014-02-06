all: cuep test

cuep: cuep.cu cuep.h
	nvcc -shared -Xcompiler '-fPIC' -lcudart -O2 cuep.cu -o libcuep.so

test: test.cpp
	g++ -m64 test.cpp -o test libcuep.so

.PHONY: clean
clean: test.cpp
	rm -f *.o test

