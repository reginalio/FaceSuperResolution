//
// Created by Regina on 29/05/2015.
//

#include <assert.h>
#include <string.h>
#include <stdio.h>
#include "cl_setup.h"
#include "stdlib.h"
#include "util.h"

cl_platform_id setOpenCLPlatform(void) {

    cl_int ret;
    printf("CL set platform\n");

//***Creating Platform***
    cl_uint num_platforms;
    ret = clGetPlatformIDs(0, NULL, &num_platforms);
    assert(ret == CL_SUCCESS);

    cl_platform_id *platform_id = (cl_platform_id *) malloc(sizeof(cl_platform_id) * num_platforms);
    cl_platform_id platform = NULL;
    clGetPlatformIDs(num_platforms, platform_id, &num_platforms);
    unsigned int i;
    for (i = 0; i < num_platforms; ++i) {
        char pbuff[100];
        clGetPlatformInfo(platform_id[i], CL_PLATFORM_VENDOR, sizeof(pbuff), pbuff, NULL);
        platform = platform_id[i];
        if (!strcmp(pbuff, "Altera Corporation")) { break; }
    }

    free(platform_id);

    return platform;
}


//***Creating Context***
cl_context createOpenCLContext(cl_platform_id platform) {

    printf("CL create context\n");

    cl_int ret;
    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0};
    cl_context context = clCreateContextFromType(cps, CL_DEVICE_TYPE_ALL, NULL, NULL, &ret);
    assert(ret == CL_SUCCESS);

    return context;
}

//***Creating Device***
cl_device_id getOpenCLDevices(cl_context context) {
    printf("CL set device\n");

    cl_int ret;
    size_t deviceListSize;
    ret = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceListSize);
    assert(ret == CL_SUCCESS);

    cl_device_id *devices = (cl_device_id *) malloc(deviceListSize);
    clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceListSize, devices, NULL);
    cl_device_id device = devices[0];

    return device;
}

cl_program createOpenCLProgram(cl_context context, cl_device_id device){

    printf("CL create program\n");

    cl_int ret;
    char mm[3];
    char n[3];
    char clFilename[80];
    #ifdef USE_FPGA
    strcpy(clFilename, "fpga/lcr_");
    #else
    strcpy(clFilename, "simKernels/lcr_");
    #endif
    sprintf(mm, "%d", MM);
    sprintf(n, "%d", N);
    strcat(clFilename, mm);
    strcat(clFilename, "_");
    strcat(clFilename, n);
    strcat(clFilename, ".aocx\0");

    printf("opening kernel file %s \n", clFilename);

    FILE *fp=fopen(clFilename, "r");
    char *binary_buf = (char *)malloc(100000000);
    size_t binary_size = fread(binary_buf, 1, 100000000, fp);
    fclose(fp);
    cl_program program = clCreateProgramWithBinary(context, 1, &device, (const size_t *)&binary_size,(const unsigned char **)&binary_buf, NULL, &ret);
    assert(ret==CL_SUCCESS);

    ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    assert(ret==CL_SUCCESS);

    printf("lalala\n");
//    free(binary_buf);
    return program;
}




