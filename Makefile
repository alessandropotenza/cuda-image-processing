CXX=nvcc
INCLUDES=$(shell pkg-config --cflags opencv)
CXXFLAGS=$(INCLUDES)
LDLIBS=-L/usr/lib/x86_64-linux-gnu/ $(shell pkg-config --libs opencv)

files=$(wildcard *.cu)
programs=$(files:%.cu=%)

all: $(programs)

main: main.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm $(programs)

	