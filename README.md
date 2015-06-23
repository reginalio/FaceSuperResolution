# FaceSuperResolution
Face super-resolution with LcR algorithm in C and OpenCL

### Project overview
This repository contains the C and OpenCL code for a face super-resolution algorithm called LcR proposed by Junjun Jiang, Ruimin Hu, Zhongyuan Wang, Zhen Han. Noise Robust Face Hallucination via
Locality-Constrained Representation. Multimedia, IEEE Transactions on 2014;16(5) 1268-1281.

The dataset has not been included in this repository.

### Makefile command for different compilation purposes
make host_fpga: compile host that launches kernels onto FPGA

run: bin/lcr_fpga

===

make simulation: compile kernel for emulator, compile host that emulates launching kernels on CPU

make host: compile host that emulates launching kernels on CPU

run: bin/lcr

===
make c_only: compile serial version of algorithm

run: bin/lcr_c
