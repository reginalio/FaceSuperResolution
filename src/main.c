#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "load_pgm.h"
#include "util.h"
#include "time.h"
#include <sys/time.h>

#include <assert.h>

#ifdef USE_OPENCL
#include "reconstruction_cl.h"
#include "cl_setup.h"
#endif

#define AOCL_ALIGNMENT 64

/* Return 1 if the difference is negative, otherwise 0.  */
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
    result->tv_sec = diff / 1000000;
    result->tv_usec = diff % 1000000;

    return (diff<0);
}

int main() {

    static Parameters* p;
    p = setParameters();

    //----------------- read training images ------------------------------------------
    int i;
    char istr[3];
    PGMData trainSetLR[NUMTRAINIMAGES];
    PGMData trainSetHR[NUMTRAINIMAGES];

    printf("reading training set\n");
    fflush(stdout);

    for (i=0;i<NUMTRAINIMAGES;i++){
        char fnLR[80];
        char fnHR[80];

        sprintf(istr, "%d", i+1);     // convert i to string

        // get file name
        strcpy(fnLR, "dataset/trainLR/Face");
        strcat(fnLR, istr);
        strcat(fnLR, ".pgm\0");

        strcpy(fnHR, "dataset/trainHR/Face");
        strcat(fnHR, istr);
        strcat(fnHR, ".pgm\0");

        readPGM(fnLR, &(trainSetLR[i]));
        readPGM(fnHR, &(trainSetHR[i]));
    }

    //------------------ Training images: divide to patches ----------------------------
    Patch *trainLRPatched;
    Patch *trainHRPatched;
    trainLRPatched = divideToPatches(trainSetLR, NUMTRAINIMAGES, p, 'l');
    trainHRPatched = divideToPatches(trainSetHR, NUMTRAINIMAGES, p, 'h');

    printf("done dividing patch\n");
    fflush(stdout);

    double patchingTime = 0.0;
    double combineTime = 0.0;
    double runTime = 0.0;
    double total_time = 0.0;


    clock_t start;
    clock_t end;
    struct timeval tvBegin, tvEnd, tvDiff, tvPatch, tvRun, tvCombine;

#ifdef USE_OPENCL
    start = clock();
        gettimeofday(&tvBegin, NULL);

    cl_platform_id platform = setOpenCLPlatform();
    cl_context context = createOpenCLContext(platform);
    cl_device_id device = getOpenCLDevices(context);
    cl_program program = createOpenCLProgram(context, device);
    printf("here\n");

    cl_int ret;

    // create opencl kernel
    cl_kernel cholesky_decomp = clCreateKernel(program, "cholesky_decomp", &ret);
    assert(ret==CL_SUCCESS);

    cl_kernel solver = clCreateKernel(program, "solver", &ret);
    assert(ret==CL_SUCCESS);

    // create opencl command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &ret);
    assert(ret==CL_SUCCESS);

    cl_ulong time_start, time_end;

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
    assert(ret==CL_SUCCESS);

    end = clock();
    gettimeofday(&tvEnd, NULL);

    double clPrepTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
    printf("cl preptime: %ld.%06ld s\n", tvDiff.tv_sec, tvDiff.tv_usec);

    #endif


    int numPatch = trainLRPatched->numPatchX * trainLRPatched->numPatchX;
    int patch;
    Patch* testHRPatched = malloc(sizeof(Patch)*numPatch);;
    for(patch=0; patch < numPatch; patch++) {

        testHRPatched[patch].matrix = allocate_dynamic_matrix_float(1, p->hrPatchWidth * p->hrPatchWidth);
        testHRPatched[patch].col = trainHRPatched->col;
        testHRPatched[patch].row = 1;
        testHRPatched[patch].numPatchX = trainLRPatched->numPatchX;
    }

    start = clock();
    gettimeofday(&tvBegin, NULL);

    //------------------- LcR -----------------------------------------------------------
    for(i=0; i<NUMTESTFACE;i++) {

        //------------------- read test image -------------------------------------------
//        printf("reading test image\n");
        fflush(stdout);

        char fnTestLR[80];
        char fnTestHROut[80];

        // get file name
        sprintf(istr, "%d", i+1);     // convert i to string
        strcpy(fnTestLR, "dataset/testLR/Face");
        strcat(fnTestLR, istr);
        strcat(fnTestLR, ".pgm\0");

        char mm[3];
        char n[5];

        sprintf(mm, "%dx", MM);
        sprintf(n, "%do%d", N, LRO);

        strcpy(fnTestHROut, "result/");
        #ifdef USE_OPENCL
            #ifdef USE_FPGA
                strcat(fnTestHROut, "fpga/");
            #else
                strcat(fnTestHROut, "sim/");
            #endif
        #else
            strcat(fnTestHROut, "serial/");
        #endif
        strcat(fnTestHROut, mm);
        strcat(fnTestHROut, n);
        strcat(fnTestHROut, "_");
        strcat(fnTestHROut, istr);
        strcat(fnTestHROut, ".pgm\0");

//        strcpy(fnTestLR, "dataset/testLR/Face20.pgm");
        PGMData testLR;
        readPGM(fnTestLR, &(testLR));
        Patch *testLRPatched;

//        start = clock();
        testLRPatched = divideToPatches(&testLR, 1, p, 'l');
//        end = clock();
//        patchingTime += ((double) (end - start));


        //------------------------ reconstruction ----------------------------------------

        #ifdef USE_OPENCL
            //testHRPatched = reconstruction_cl(testLRPatched, trainLRPatched, trainSetHRPatched, p);

           // Patch* testHRPatched;

            printf("using opencl methods\n");

            printf("start - reconstruction analytic\n");
            assert(p->numTrainImagesToUse<=p->numTrainImages);

            int hrPW = p->hrPatchWidth;

            float **C, *G, *L;
            C = allocate_dynamic_matrix_float(N, MM);
            posix_memalign((void**)&G, AOCL_ALIGNMENT, sizeof(float)*MM*MM);
            posix_memalign((void**)&L, AOCL_ALIGNMENT, sizeof(float)*MM*MM);

//            G = malloc(sizeof(float)*MM*MM);
//            L = malloc (sizeof(float)*MM*MM);

            for(patch=0; patch < numPatch; patch++){

//                printf("reconstructing patch %d\n", patch);
//                testHRPatched[patch].matrix = allocate_dynamic_matrix_float(testLRPatched->row, hrPW*hrPW );
//                testHRPatched[patch].col = trainHRPatched->col;
//                testHRPatched[patch].row = testLRPatched->row;
//                testHRPatched[patch].numPatchX = testLRPatched->numPatchX;

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
                            G[i * p->numTrainImagesToUse + j] += C[k][i] * C[k][j];
                        }
                        if(j == i) {
                            G[i * p->numTrainImagesToUse + j] += p->tau * dist[idx[i]] * dist[idx[i]];
                        }
                    }
                }

                // ---------------------------------------------------------------------------------------------
                cl_int ret;
                cl_event kernel_event;
                cl_event read_events[1];
                size_t local_size = 1;
                size_t global_size = 1;

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
                                             NULL, &global_size, &local_size, 2, write_events2, &kernel_event);
                assert(ret==CL_SUCCESS);

                clWaitForEvents(1 , &kernel_event);
                clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
                clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
                total_time += (time_end - time_start)/ 1000000.0;
//                printf("\n1. Execution time in milliseconds = %0.3f ms\n", (total_time / 1000000.0) );

                //---------------- CHOLESKY SOLVER ---------------------------------------------------

                // set kernel arguments
                ret = clSetKernelArg(solver, 0, sizeof(cl_mem), &L_buffer);
                assert(ret==CL_SUCCESS);
                ret = clSetKernelArg(solver, 1, sizeof(cl_mem), &w_buffer);
                assert(ret==CL_SUCCESS);
                ret = clSetKernelArg(solver, 2, sizeof(cl_mem), &y_buffer);

                clFinish(command_queue);
                // run kernel
                ret = clEnqueueNDRangeKernel(command_queue, solver, (cl_uint) 1, // one dimension
                                             NULL, &global_size, &local_size, 0, NULL, &kernel_event);
                assert(ret==CL_SUCCESS);

                // Read w
                ret = clEnqueueReadBuffer(command_queue, w_buffer, CL_TRUE, 0, sizeof(float)* MM, w, 1, &kernel_event, &read_events[0]);
                assert(ret==CL_SUCCESS);

                ret = clWaitForEvents(1,read_events);
                assert(ret==CL_SUCCESS);

                clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
                clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
                total_time +=(time_end - time_start)/ 1000000.0;
//                printf("\n2. Execution time in milliseconds = %0.3f ms\n", (total_time / 1000000.0) );



                //------------------- construct HR patch ----------------------------------------------
                double sum_x = 0.0;
                for(i=0; i<p->numTrainImagesToUse; i++)
                    sum_x += w[i];

                for(i=0;i<testHRPatched->col;i++){
                    double tmp = 0.0;
                    for(j=0;j<p->numTrainImagesToUse;j++){
                         tmp += (w[j]/sum_x)*trainHRPatched[patch].matrix[idx[j]][i];
                    }
                    testHRPatched[patch].matrix[0][i] = tmp;
                }

//                for(j=0;j<p->numTrainImagesToUse;j++){
//                    for(i=0;i<testHRPatched->col;i++){
//                        testHRPatched[patch].matrix[0][i] += (w[j]/sum_x)*trainHRPatched[patch].matrix[idx[j]][i];
//                    }
//                }

                // free allocated memory
                free(dist);
                free(idx);
                free(w);
            }

            deallocate_dynamic_matrix_float(C,testLRPatched->col);
            free(G);
            free(L);

        #else
            testHRPatched = reconstruction(testLRPatched, trainLRPatched, trainHRPatched, p);
        #endif


//        printf("done resconstruction\n");
//        fflush(stdout);

        //------------------------ combine patches ---------------------------------------
        PGMData testRecombinedImage;
//        start = clock();
        testRecombinedImage = combinePatches(testHRPatched, p);
//        end = clock();
//        combineTime += ((double) (end - start));

        writePGM(fnTestHROut, &(testRecombinedImage));

        printf("wrote to %s\n", fnTestHROut);

        for(patch=0; patch < numPatch; patch++)
            deallocate_dynamic_matrix_float(testLRPatched[patch].matrix, testLRPatched[patch].row);

        free(testLRPatched);
    }
    end = clock();

    gettimeofday(&tvEnd, NULL);
    runTime += ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("\n  cpu runtime everything: %f\n", runTime);
    timeval_subtract(&tvRun, &tvEnd, &tvBegin);
    printf("everything: %ld.%06ld s\n", tvRun.tv_sec, tvRun.tv_usec);

    printf("\n Kernel Execution time in milliseconds = %0.3f ms\n", total_time );

    for(patch=0; patch < numPatch; patch++){
        deallocate_dynamic_matrix_float(testHRPatched[patch].matrix, testHRPatched[patch].row);
    }
    free(testHRPatched);

    free(p);

    patchingTime = patchingTime / CLOCKS_PER_SEC;
    combineTime = combineTime / CLOCKS_PER_SEC;
    printf("-- Timing for %d testing images -- \n", NUMTESTFACE);
    #ifdef USE_OPENCL
    printf("clPrepTime: %fs \n", clPrepTime);
    #endif
    printf("patchingTime: %fs\nrunTime: %fs\ncombinePatchTime: %fs\ntotal time: %fs\n", patchingTime, runTime, combineTime, patchingTime + runTime + combineTime);
    printf("Total time taken to process one test image: %fs \n", (patchingTime+runTime)/NUMTESTFACE);
    fflush(stdout);

    #ifdef USE_OPENCL
    // Clean up OpenCL stuff
    ret = clReleaseKernel(cholesky_decomp);
    assert(ret==CL_SUCCESS);
    ret = clReleaseKernel(solver);
    assert(ret==CL_SUCCESS);
    ret = clReleaseProgram(program);
    assert(ret==CL_SUCCESS);
    ret = clReleaseCommandQueue(command_queue);
    assert(ret==CL_SUCCESS);
    ret = clReleaseContext(context);
    assert(ret==CL_SUCCESS);
    ret = clReleaseMemObject(testLR_buffer);
    assert(ret==CL_SUCCESS);
    ret = clReleaseMemObject(trainLR_buffer);
    assert(ret==CL_SUCCESS);
    ret = clReleaseMemObject(idx_buffer);
    assert(ret==CL_SUCCESS);
    ret = clReleaseMemObject(C_buffer);
    assert(ret==CL_SUCCESS);
    ret = clReleaseMemObject(dist_buffer);
    assert(ret==CL_SUCCESS);
    ret = clReleaseMemObject(G_buffer);
    assert(ret==CL_SUCCESS);
    ret = clReleaseMemObject(L_buffer);
    assert(ret==CL_SUCCESS);
    ret = clReleaseMemObject(w_buffer);
    assert(ret==CL_SUCCESS);
    ret = clReleaseMemObject(y_buffer);
    assert(ret==CL_SUCCESS);
    #endif

    free(trainLRPatched);
    free(trainHRPatched);


    return 0;
}