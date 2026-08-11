CXX ?= g++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -pedantic -I.

.PHONY: all clean test generate-examples

all: build/mvf_barcodes build/mvf_barcodes_validated

build/mvf_barcodes: mvf_barcodes.cpp $(wildcard mvf_module/*.hpp)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) mvf_barcodes.cpp -o $@

build/mvf_barcodes_validated: mvf_barcodes_validated.cpp $(wildcard mvf_module/*.hpp)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) mvf_barcodes_validated.cpp -o $@

test: all
	bash test/run_tests.sh

generate-examples:
	python3 -B mvf_module/utils/generate_demo_sequences.py

clean:
	rm -rf build
