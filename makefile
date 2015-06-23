CC = gcc
AOC = aoc -v
BOARD = pcie385n_d5
HOST_TARGET = lcr
TARGET = lcr_kernel
HOST_SOURCES = src/main.c src/load_pgm.c src/util.c src/cg.c
CL_SOURCES = src/cl_setup.c src/reconstruction_cl.c

GSL_HEADERS = -I/mnt/applications/gsl/1.16/include
GSL_LDFLAGS = -L/mnt/applications/gsl/1.16/lib
GSL_LDLIBS = -lgsl -lgslcblas -lm

AOCL_COMPILE_CONFIG = -I/mnt/applications/altera/aocl-sdk/host/include -I/mnt/applications/altera/aocl-sdk/board/nalla_pcie/include #$(aocl compile-config)
AOCL_LDFLAGS = -L/mnt/applications/altera/aocl-sdk/board/nalla_pcie/linux64/lib -L/mnt/applications/altera/aocl-sdk/host/linux64/lib #$(aocl ldflags)
AOCL_LDLIBS = -lalteracl -ldl -lacl_emulator_kernel_rt  -lalterahalmmd -lnalla_pcie_mmd -lelf -lrt -lstdc++ #$({aocl ldlibs)


bin/$(HOST_TARGET) : 
	mkdir -p bin
	$(CC) $(HOST_SOURCES) $(CL_SOURCES) -o $@ -DUSE_OPENCL $(GSL_HEADERS) $(GSL_LDFLAGS) $(GSL_LDLIBS) $(AOCL_COMPILE_CONFIG) $(AOCL_LDFLAGS) $(AOCL_LDLIBS)
	
host: bin/$(HOST_TARGET)

host_fpga:
	mkdir -p bin
	$(CC) $(HOST_SOURCES) $(CL_SOURCES) -o bin/lcr_fpga -DUSE_OPENCL -DUSE_FPGA $(GSL_HEADERS) $(GSL_LDFLAGS) $(GSL_LDLIBS) $(AOCL_COMPILE_CONFIG) $(AOCL_LDFLAGS) $(AOCL_LDLIBS)


c_only: 
	mkdir -p bin
	$(CC) $(HOST_SOURCES) -o bin/lcr_c $(GSL_HEADERS) $(GSL_LDFLAGS) $(GSL_LDLIBS)
	
simulation: bin/$(HOST_TARGET)
	$(AOC) -march=emulator src/$(TARGET).cl --board $(BOARD)
	bin/$(HOST_TARGET)

clean:
	rm -rf bin/$(TARGET)
	rm -rf bin/$(HOST_TARGET)
	rm -f bin/$(TARGET).aoco
	rm -f bin/$(TARGET).aocx
