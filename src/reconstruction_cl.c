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
    printf("here\n");

    cl_int ret;

    // create opencl kernel
    cl_kernel cov_matrix = clCreateKernel(program, "cov_matrix", &ret);
    assert(ret==CL_SUCCESS);

    cl_kernel cholesky_decomp = clCreateKernel(program, "cholesky_decomp", &ret);
    assert(ret==CL_SUCCESS);

    cl_kernel solver = clCreateKernel(program, "solver", &ret);
    assert(ret==CL_SUCCESS);

    // create opencl command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, &ret);
    assert(ret==CL_SUCCESS);

    // create opencl memory objects
    int byte_size = sizeof(float)* MM* MM;
    cl_mem testLR_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*N, NULL, &ret);
    assert(ret==CL_SUCCESS);
    cl_mem trainLR_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*N*NUMTRAINIMAGES, NULL, &ret);
    assert(ret==CL_SUCCESS);
    cl_mem idx_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*NUMTRAINIMAGES, NULL, &ret);
    assert(ret==CL_SUCCESS);
    cl_mem C_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)* MM*N, NULL, &ret);
    assert(ret==CL_SUCCESS);
    cl_mem dist_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NUMTRAINIMAGES, NULL, &ret);
    assert(ret==CL_SUCCESS);
    cl_mem G_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, byte_size, NULL, &ret);
    assert(ret==CL_SUCCESS);
    cl_mem L_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, byte_size, NULL, &ret);
    assert(ret==CL_SUCCESS);
    cl_mem w_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, byte_size, NULL, &ret);
    assert(ret==CL_SUCCESS);
    cl_mem y_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, byte_size, NULL, &ret);

    Patch* testHRPatched;

    printf("using opencl methods\n");

    printf("start - reconstruction analytic\n");
    assert(p->numTrainImagesToUse<=p->numTrainImages);
    int numPatch = testLRPatched->numPatchX * testLRPatched->numPatchX;

    testHRPatched = malloc(sizeof(Patch)*numPatch);

    int hrPW = p->hrPatchWidth;
    int patch;

    float **C, *G, *L;
    C = allocate_dynamic_matrix_float(N, MM);
    G = malloc(sizeof(float)*MM*MM);
    L = malloc(sizeof(float)*MM*MM);

    for(patch=0; patch < numPatch; patch++){

        printf("reconstructing patch %d\n", patch);
        testHRPatched[patch].matrix = allocate_dynamic_matrix_float(testLRPatched->row, hrPW*hrPW );
        testHRPatched[patch].col = trainHRPatched->col;
        testHRPatched[patch].row = testLRPatched->row;
        testHRPatched[patch].numPatchX = testLRPatched->numPatchX;

        // get distance between xLR(patch) and all yLR(patch)
        float* dist;        // size all_Y
        dist = calcDistance(&(testLRPatched[patch]), &(trainLRPatched[patch]));

        // sort dist and get sort index (an index array that contains the order of the sort)
        int* idx;
        idx = sortDistIndex(dist, trainLRPatched->row);

        // get weights
        int i,j,k;

        float* w;
        w = malloc(sizeof(float)* MM);

        // loop over the closest `numTrainImagesToUse' trainLR
        for(i=0; i<testLRPatched->col; i++) {
            for(j=0; j<p->numTrainImagesToUse; j++){
                C[i][j] = testLRPatched[patch].matrix[0][i] - trainLRPatched[patch].matrix[idx[j]][i];
            }
        }

        for(i=0; i<p->numTrainImagesToUse; i++) {
            for(j =0; j <p->numTrainImagesToUse; j++) {
                G[i * p->numTrainImagesToUse + j] = 0;
                for (k = 0; k < testLRPatched->col; k++) {
                    G[i * p->numTrainImagesToUse + j] += C[k][i] * C[k][j];     //TODO optimise the indices
                }
                if(j == i) {
                    G[i * p->numTrainImagesToUse + j] += p->tau * dist[idx[i]] * dist[idx[i]];
                }
            }
        }

        // ---------------------------------------------------------------------------------------------
        cl_int ret;
        cl_event kernel_event[1];
        cl_event read_events[1];
        size_t local_size = 1;
        size_t global_size = 1;

        //---------------------- COVARIANCE MATRIX -----------------------------------------------------
//        // write to buffer
//        cl_event write_events[5];
//        ret = clEnqueueWriteBuffer(command_queue, testLR_buffer, CL_TRUE, 0, sizeof(float)*N, testLRPatched[patch].matrix, 0, NULL, &write_events[0]);
//        assert(ret==CL_SUCCESS);
//        ret = clEnqueueWriteBuffer(command_queue, trainLR_buffer, CL_TRUE, 0, sizeof(float)*N*NUMTRAINIMAGES, trainLRPatched[patch].matrix, 0, NULL, &write_events[1]);
//        assert(ret==CL_SUCCESS);
//        ret = clEnqueueWriteBuffer(command_queue, idx_buffer, CL_TRUE, 0, sizeof(int)*NUMTRAINIMAGES, idx, 0, NULL, &write_events[2]);
//        assert(ret==CL_SUCCESS);
//        ret = clEnqueueWriteBuffer(command_queue, dist_buffer, CL_TRUE, 0, sizeof(float)*NUMTRAINIMAGES, dist, 0, NULL, &write_events[3]);
//        assert(ret==CL_SUCCESS);
//        ret = clEnqueueWriteBuffer(command_queue, G_buffer, CL_TRUE, 0, byte_size, G, 0, NULL, &write_events[4]);
//        assert(ret==CL_SUCCESS);
//
//        // set kernel arguments
//        ret = clSetKernelArg(cov_matrix, 0, sizeof(cl_mem), &testLR_buffer);
//        assert(ret==CL_SUCCESS);
//        ret = clSetKernelArg(cov_matrix, 1, sizeof(cl_mem), &trainLR_buffer);
//        assert(ret==CL_SUCCESS);
//        ret = clSetKernelArg(cov_matrix, 2, sizeof(cl_mem), &idx_buffer);
//        assert(ret==CL_SUCCESS);
//        ret = clSetKernelArg(cov_matrix, 3, sizeof(cl_mem), &C_buffer);
//        assert(ret==CL_SUCCESS);
//        ret = clSetKernelArg(cov_matrix, 4, sizeof(cl_mem), &G_buffer);
//        assert(ret==CL_SUCCESS);
//        ret = clSetKernelArg(cov_matrix, 5, sizeof(cl_mem), &dist_buffer);
//        assert(ret==CL_SUCCESS);
//
//
//        // run kernel
//        ret = clEnqueueNDRangeKernel(command_queue, cov_matrix, (cl_uint) 1, // one dimension
//                                     NULL, &global_size, &local_size, 2, write_events, kernel_event);
//        assert(ret==CL_SUCCESS);
//
//        // debug G
//        ret = clEnqueueReadBuffer(command_queue, G_buffer, CL_TRUE, 0, byte_size, G1, 1, kernel_event, &read_events[0]);
//        assert(ret==CL_SUCCESS);
//
//        for(i=0;i<N;++i){
//            for(j=0;j<N;j++)
//                printf("G:%f \t G1:%f\n",G[i*N+j],G1[i*N+j]);
//        }


        //---------- CHOLESKY DECOMPOSITION ------------------------------------------------------------

        // write to buffer
        cl_event write_events2[2];
        ret = clEnqueueWriteBuffer(command_queue, L_buffer, CL_TRUE, 0, byte_size, L, 0, NULL, &write_events2[0]);
        assert(ret==CL_SUCCESS);
        ret = clEnqueueWriteBuffer(command_queue, G_buffer, CL_TRUE, 0, byte_size, G, 0, NULL, &write_events2[1]);
        assert(ret==CL_SUCCESS);

        // set kernel arguments
        ret = clSetKernelArg(cholesky_decomp, 0, sizeof(cl_mem), &G_buffer);
        assert(ret==CL_SUCCESS);
        ret = clSetKernelArg(cholesky_decomp, 1, sizeof(cl_mem), &L_buffer);
        assert(ret==CL_SUCCESS);

        // run kernel
        ret = clEnqueueNDRangeKernel(command_queue, cholesky_decomp, (cl_uint) 1, // one dimension
                                     NULL, &global_size, &local_size, 2, write_events2, kernel_event);
        assert(ret==CL_SUCCESS);


        //---------------- CHOLESKY SOLVER ---------------------------------------------------

        // set kernel arguments
        ret = clSetKernelArg(solver, 0, sizeof(cl_mem), &L_buffer);
        assert(ret==CL_SUCCESS);
        ret = clSetKernelArg(solver, 1, sizeof(cl_mem), &w_buffer);
        assert(ret==CL_SUCCESS);
        ret = clSetKernelArg(solver, 2, sizeof(cl_mem), &y_buffer);

        // run kernel
        ret = clEnqueueNDRangeKernel(command_queue, solver, (cl_uint) 1, // one dimension
                                     NULL, &global_size, &local_size, 0, NULL, kernel_event);
        assert(ret==CL_SUCCESS);

        // Read w
        ret = clEnqueueReadBuffer(command_queue, w_buffer, CL_TRUE, 0, sizeof(float)* MM, w, 1, kernel_event, &read_events[0]);
        assert(ret==CL_SUCCESS);

        ret = clWaitForEvents(1,read_events);
        assert(ret==CL_SUCCESS);

        //------------------- construct HR patch ----------------------------------------------
        double sum_x = 0.0;
        for(i=0; i<p->numTrainImagesToUse; i++)
            sum_x += w[i];

        for(j=0;j<p->numTrainImagesToUse;j++){
            for(i=0;i<testHRPatched->col;i++){
                testHRPatched[patch].matrix[0][i] += (w[j]/sum_x)*trainHRPatched[patch].matrix[idx[j]][i];
            }
        }

        // free allocated memory
        free(dist);
        free(idx);
        free(w);
    }



    deallocate_dynamic_matrix_float(C,testLRPatched->col);
    free(G);
    free(L);
//    free(w);


    return testHRPatched;



}
