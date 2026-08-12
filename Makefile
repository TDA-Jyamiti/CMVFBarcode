CXX ?= g++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -pedantic -I.

.PHONY: all clean test generate-examples

all: build/cmvf_barcodes build/cmvf_barcodes_validated

build/cmvf_barcodes: cmvf_barcodes.cpp $(wildcard mvf_module/*.hpp)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) cmvf_barcodes.cpp -o $@

build/cmvf_barcodes_validated: cmvf_barcodes_validated.cpp $(wildcard mvf_module/*.hpp)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) cmvf_barcodes_validated.cpp -o $@

test: all
	bash test/run_tests.sh

generate-examples:
	python3 -B mvf_module/utils/generate_demo_sequences.py

clean:
	rm -rf build
