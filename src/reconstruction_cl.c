//
// Created by Regina on 29/05/2015.
//

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "reconstruction_cl.h"
#include "cl_setup.h"


Patch *reconstruction_cl(Patch* testLRPatched, Patch* trainLRPatched, Patch* trainHRPatched, Parameters* p) {
    cl_platform_id platform = setOpenCLPlatform();
    cl_context context = createOpenCLContext(platform);
    cl_device_id device = getOpenCLDevices(context);
    cl_program program = createOpenCLProgram(context, device);

    cl_int ret;

    // create opencl kernel
    cl_kernel cov_matrix = clCreateKernel(program, "cov_matrix", &ret);
    cl_kernel cholesky_decomp = clCreateKernel(program, "cholesky_decomp", &ret);
    cl_kernel solver = clCreateKernel(program, "solver", &ret);
    assert(ret==CL_SUCCESS);

    // create opencl command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, &ret);
    assert(ret==CL_SUCCESS);

    // create opencl memory objects
    int byte_size = sizeof(float)*p->numTrainImagesToUse*p->numTrainImagesToUse;
    cl_mem G_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, byte_size, NULL, &ret);
    assert(ret==CL_SUCCESS);
    cl_mem L_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, byte_size, NULL, &ret);
    assert(ret==CL_SUCCESS);
    cl_mem w_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, byte_size, NULL, &ret);
    assert(ret==CL_SUCCESS);
    cl_mem y_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, byte_size, NULL, &ret);
    assert(ret==CL_SUCCESS);


    Patch* testHRPatched;

    printf("using opencl methods\n");

    printf("start - reconstruction analytic\n");
    assert(p->numTrainImagesToUse<=p->numTrainImages);
    int numPatch = testLRPatched->numPatchX * testLRPatched->numPatchX;

    testHRPatched = malloc(sizeof(Patch)*numPatch);

    int hrPW = p->hrPatchWidth;
    int patch;

//    float *C, *G;
//    C = malloc(sizeof(float) * testLRPatched->col * p->numTrainImagesToUse);
//    G = malloc(sizeof(float) * p->numTrainImagesToUse * p->numTrainImagesToUse);

    float **C, *G, *L;
    C = allocate_dynamic_matrix_float(testLRPatched->col, p->numTrainImagesToUse);
    G = malloc(sizeof(float)*p->numTrainImagesToUse*p->numTrainImagesToUse);
    L = malloc(byte_size);

    for(patch=0; patch < numPatch; patch++){

        printf("reconstructing patch %d\n", patch);
        testHRPatched[patch].matrix = allocate_dynamic_matrix_float(testLRPatched->row, hrPW*hrPW );
        testHRPatched[patch].col = trainHRPatched->col;
        testHRPatched[patch].row = testLRPatched->row;
        testHRPatched[patch].numPatchX = testLRPatched->numPatchX;
        printf("here? \n");

        // get distance between xLR(patch) and all yLR(patch)
        float* dist;        // size all_Y
        dist = calcDistance(&(testLRPatched[patch]), &(trainLRPatched[patch]));

        // sort dist and get sort index (an index array that contains the order of the sort)
        int* idx;
        idx = sortDistIndex(dist, trainLRPatched->row);

        // get weights
        float* w;
        w = malloc(sizeof(float)*p->numTrainImagesToUse);

        int i,j,k;
        // loop over the closest `numTrainImagesToUse' trainLR
        for(i=0; i<testLRPatched->col; i++) {
            for(j=0; j<p->numTrainImagesToUse; j++){
                C[i][j] = testLRPatched[patch].matrix[0][i] - trainLRPatched[patch].matrix[idx[j]][i];
            }
        }

        for(i=0; i<p->numTrainImagesToUse; i++) {
            for(j =0; j <p->numTrainImagesToUse; j++) {
                G[i * p->numTrainImagesToUse + j] = 0;
//                printf("i: %d, j: %d \n", i, j);
                for (k = 0; k < testLRPatched->col; k++) {
                    G[i * p->numTrainImagesToUse + j] += C[k][i] * C[k][j];     //TODO optimise the indices
//                    printf("i: %d, j: %d, k: %d \n", i, j, k);
                }
                if(j == i) {
                    G[i * p->numTrainImagesToUse + j] += p->tau * dist[idx[i]] * dist[idx[i]];
                }
            }
        }

        //---------- CHOLESKY DECOMPOSITION ------------------------------------------------------------

        // write to buffer
        cl_event write_events[2];
        ret = clEnqueueWriteBuffer(command_queue, G_buffer, CL_TRUE, 0, byte_size, G, 0, NULL, &write_events[0]);
        assert(ret==CL_SUCCESS);
        ret = clEnqueueWriteBuffer(command_queue, L_buffer, CL_TRUE, 0, byte_size, G, 0, NULL, &write_events[1]);
        assert(ret==CL_SUCCESS);

        // set kernel arguments
        ret = clSetKernelArg(cholesky_decomp, 0, sizeof(cl_mem), &G_buffer);
        assert(ret==CL_SUCCESS);
        ret = clSetKernelArg(cholesky_decomp, 1, sizeof(cl_mem), &L_buffer);
        assert(ret==CL_SUCCESS);
        ret = clSetKernelArg(cholesky_decomp, 2, sizeof(p->numTrainImagesToUse), &(p->numTrainImagesToUse));
        assert(ret==CL_SUCCESS);

        cl_event kernel_event[1];
        cl_event read_events[1];
        size_t local_size = 1;
        size_t global_size = 1;

        // run kernel
        ret = clEnqueueNDRangeKernel(command_queue, cholesky_decomp, (cl_uint) 1, // one dimension
                                     NULL, &global_size, &local_size, 2, write_events, kernel_event);
        assert(ret==CL_SUCCESS);

        // Read L
//        ret = clEnqueueReadBuffer(command_queue, L_buffer, CL_TRUE, 0, byte_size, L, 1, kernel_event, &read_events[0]);
//        assert(ret==CL_SUCCESS);


        //---------------- CHOLESKY SOLVER ---------------------------------------------------

        // set kernel arguments
        ret = clSetKernelArg(solver, 0, sizeof(cl_mem), &L_buffer);
        assert(ret==CL_SUCCESS);
        ret = clSetKernelArg(solver, 1, sizeof(p->numTrainImagesToUse), &(p->numTrainImagesToUse));
        assert(ret==CL_SUCCESS);
        ret = clSetKernelArg(solver, 2, sizeof(cl_mem), &w_buffer);
        assert(ret==CL_SUCCESS);
        ret = clSetKernelArg(solver, 3, sizeof(cl_mem), &y_buffer);

        // run kernel
        ret = clEnqueueNDRangeKernel(command_queue, solver, (cl_uint) 1, // one dimension
                                     NULL, &global_size, &local_size, 0, NULL, kernel_event);
        assert(ret==CL_SUCCESS);

        // Read w
        ret = clEnqueueReadBuffer(command_queue, w_buffer, CL_TRUE, 0, sizeof(float)*p->numTrainImagesToUse, w, 1, kernel_event, &read_events[0]);
        assert(ret==CL_SUCCESS);

        ret = clWaitForEvents(1,read_events);
        assert(ret==CL_SUCCESS);

        //------------------- construct HR patch ----------------------------------------------
        double sum_x = 0.0;
        for(i=0; i<p->numTrainImagesToUse; i++)
            sum_x += w[i];

        for(j=0;j<p->numTrainImagesToUse;j++){
            for(i=0;i<testHRPatched->col;i++){
                //printf ("testHRPatched[%d].matrix[0][%d] = %f\n", patch, i, testHRPatched[patch].matrix[0][i]);
                testHRPatched[patch].matrix[0][i] += (w[j]/sum_x)*trainHRPatched[patch].matrix[idx[j]][i];
                if(testHRPatched[patch].matrix[0][i]>1)
                    testHRPatched[patch].matrix[0][i] = 1;
                else if(testHRPatched[patch].matrix[0][i]<0)
                    testHRPatched[patch].matrix[0][i] = 0;
            }
        }



        printf("reconstructingggg\n");
        fflush(stdout);

        // free allocated memory

        //gsl_vector_free (x);
        free(dist);
        free(idx);
        free(w);
    }
    deallocate_dynamic_matrix_float(C,testLRPatched->col);
    free(G);
    free(L);


    return testHRPatched;



}
