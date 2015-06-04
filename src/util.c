//
// Created by Regina on 18/05/2015.
//

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "util.h"
#include "cg.h"
#include <gsl/gsl_linalg.h>

Parameters* setParameters(){
    Parameters* p;
    p = malloc(sizeof(Parameters));

    p->ratio = 4;
    p->tau = 0.04;

    p->lrWidth = 15;
    p->lrPatchWidth = 4;
    p->lrOverlap = 2;
    p->hrWidth = p->lrWidth * p->ratio;
    p->hrPatchWidth = p->lrPatchWidth * p->ratio;
    p->hrOverlap = p->lrOverlap * p->ratio;

    p->numTrainImages = NUMTRAINIMAGES;
    p->numTrainImagesToUse = 360;
    p->method = 'a';
    p->maxiter = 30;        // max cg iterations
    return p;
}

Patch* divideToPatches(PGMData *data, const int numImages, Parameters* p, const char mode) {
    int width,pw,overlap;

    if (mode == 'l'){
        width = p->lrWidth;
        pw = p->lrPatchWidth;
        overlap = p->lrOverlap;
    }else if (mode == 'h'){
        width = p->hrWidth;
        pw = p->hrPatchWidth;
        overlap = p->hrOverlap;
    }else{
        fprintf(stderr, "Divide to patches: wrong mode!\n");
        exit(EXIT_FAILURE);
    }

    int numPatchX = (int)ceil((double)(width-overlap)/(double)(pw-overlap)); // number of patches in x direction
    int numPatchXY = numPatchX*numPatchX;  // number of patches in one face image
    int step = pw - overlap;

//    printf("dividing to patches, mode: %c\n", mode);
//    printf("width: %d\n", width);
//    printf("overlap: %d\n", overlap);
//    printf("pw: %d\n", pw);
//
//    printf("num patch x: %d\n", numPatchX);
//    fflush(stdout);

    Patch* patches;
    int n,i,j, patchNumber;
    patches = malloc(numPatchXY*sizeof(Patch));

    for (patchNumber=0;patchNumber<numPatchXY;patchNumber++) {
//        printf("creating patch %d\n", patchNumber);

        patches[patchNumber].numPatchX = numPatchX;
        patches[patchNumber].row = numImages;
        patches[patchNumber].col = pw*pw;
        patches[patchNumber].matrix = allocate_dynamic_matrix_float(numImages, pw * pw);

        // start xy of each patch in each image
        int startX = ((patchNumber%numPatchX) * step) % width;
        int startY = floor(patchNumber / numPatchX) * step;

        // copy data from each original image to the patched version
        for (n = 0; n < numImages; n++) {
            for (j = 0; j < pw; j++) {
                for (i = 0; i < pw; i++) {
                    if (((startX + i) < width) && ((startY + j) < width))
                        patches[patchNumber].matrix[n][j * pw + i] = data[n].matrix[startX + i][startY + j];
                    else
                        patches[patchNumber].matrix[n][j * pw + i] = 0;     // zero pad right/bottom edges patches

                }
            }
        } // end for numImages
    } // end for patch number

    return patches;
}

// TODO perhaps consider numImages instead of just constrained to one patched image?
PGMData combinePatches(Patch* patches, Parameters* p) {
    printf("Combining patches\n");
    fflush(stdout);

    PGMData xHR;
    int width = p->hrWidth;
    int overlap = p->hrOverlap;
    int pw = p->hrPatchWidth;

    int numPatchX = (int)ceil((double)(width-overlap)/(double)(pw-overlap)); // number of patches in x direction
    int step = pw - overlap;

    int i, j, patch, startX, startY;
    int** counter;  // count how many patches contributed to the pixel, used as divider when averaging
    float** xHRmatrix;
    xHRmatrix = allocate_dynamic_matrix_float(width, width);
    counter = allocate_dynamic_matrix(width, width);

    for(patch=0;patch<numPatchX*numPatchX;patch++) {
//        printf("\n patch %d\n", patch);
        for (i = 0; i < pw * pw; i++) {
//            printf("i %d\n", i);

            startX = ((patch % numPatchX) * step) % width;
            startY = floor(patch / numPatchX) * step;

            int ystep = floor(i / pw);

//            printf("start X+i_pw is %d\n", startX + i % pw);
//            printf("startY + ystep is %d\n", startY + ystep);
//            printf("patches :%f\n",patches[patch].matrix[0][i]);
//            assert(startX + i % pw < width);
//            assert(startY + ystep < width);
            if ((startX + i % pw < width) && (startY + ystep < width)){
                xHRmatrix[startX + i % pw][startY + ystep] += patches[patch].matrix[0][i];
                counter[startX + i % pw][startY + ystep]++;
            }
        }
    }

    for(j=0; j<width; j++)
        for(i=0; i<width; i++){
            xHRmatrix[i][j] = xHRmatrix[i][j]/counter[i][j];
        }

    xHR.matrix = xHRmatrix;
    xHR.col = width;
    xHR.row = width;
    xHR.max_gray = 255;

    return xHR;
}

float* calcDistance(Patch* testLRPatched, Patch* trainLRPatched) {
    float* dist;
    dist = malloc(sizeof(float)*trainLRPatched->row);

    int n, i;
    double tmp;
    for(n = 0; n<trainLRPatched->row; n++){
        tmp = 0.0;
        for(i = 0; i<trainLRPatched->col; i++){
            tmp += (testLRPatched->matrix[0][i] - trainLRPatched->matrix[n][i])*
                   (testLRPatched->matrix[0][i] - trainLRPatched->matrix[n][i]);
        }
        dist[n] = sqrt(tmp);
    }

    return dist;
}

//int compare_doubles (const void *a, const void *b)
//{
//    const double *da = (const double *) a;
//    const double *db = (const double *) b;
//
//    return (*da > *db) - (*da < *db);
//}

static int compareDist (const void *a, const void *b)
{
    int aa = *((int *) a), bb = *((int *) b);
    if (baseDistArray[aa] < baseDistArray[bb])
        return -1;
    if (baseDistArray[aa] == baseDistArray[bb])
        return 0;
    if (baseDistArray[aa] > baseDistArray[bb])
        return 1;
}

int *sortDistIndex(float* dist, int numElements) {
    int *idx, i;

    idx = malloc(sizeof(int)*numElements);

    // initialise initial index permutation
    for(i=0; i<numElements; i++){
        idx[i] = i;
    }

    // assign address of original dist array to the static global pointer
    // used by the compare function
    baseDistArray = dist;

    qsort(idx, numElements, sizeof(int), compareDist);

    return idx;
}

Patch* reconstructionAnalytic(Patch* testLRPatched, Patch* trainLRPatched, Patch* trainHRPatched, Parameters* p){
    printf("start - reconstruction analytic\n");
    assert(p->numTrainImagesToUse<=p->numTrainImages);
    int numPatch = testLRPatched->numPatchX * testLRPatched->numPatchX;

    Patch* testHRPatched;
    testHRPatched = malloc(sizeof(Patch)*numPatch);

    int hrPW = p->hrPatchWidth;
    int patch;

    double **C, *G;
    C = allocate_dynamic_matrix_double(testLRPatched->col, p->numTrainImagesToUse);
    G = malloc(sizeof(double)*p->numTrainImagesToUse*p->numTrainImagesToUse);

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

        int i, j, k;

//        printf("reconstruction 1\n");
//        printf("numTrainImagesToUse: %d \n",p->numTrainImagesToUse );
//        printf("testLRPatched->col : %d\n", testLRPatched->col);
//
//        for(i=0;i<p->numTrainImages; i++){
//            printf("i: %d, dist : %f, position: %d\n", i, dist[i], idx[i]);
//        }
//        fflush(stdout);


        // loop over the closest `numTrainImagesToUse' trainLR
        for(i=0; i<testLRPatched->col; i++) {
            for(j=0; j<p->numTrainImagesToUse; j++){
                C[i][j] = testLRPatched[patch].matrix[0][i] - trainLRPatched[patch].matrix[idx[j]][i];
            }
        }

        printf("reconstruction 2\n");
        fflush(stdout);


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
        printf("reconstruction 3\n");
        fflush(stdout);
        ////////////////////// GSL stuff to solve Ax=b ////////////////////////
        // http://stackoverflow.com/questions/7949229/how-to-implement-a-left-matrix-division-on-c-using-gsl
        double* ones = malloc(sizeof(double)*p->numTrainImagesToUse);
        double* x_data = malloc(sizeof(double)*p->numTrainImagesToUse);
        for(i=0; i<p->numTrainImagesToUse;i++){
            ones[i] = 1.0;
            x_data[i] = 0.0;
        }

        gsl_matrix_view A
                = gsl_matrix_view_array (G, p->numTrainImagesToUse, p->numTrainImagesToUse);

        gsl_vector_view b
                = gsl_vector_view_array (ones, p->numTrainImagesToUse);

        gsl_vector_view x
                = gsl_vector_view_array (x_data, p->numTrainImagesToUse);
        //gsl_vector *x = gsl_vector_alloc (p->numTrainImagesToUse); // size equal to n
        gsl_vector *residual = gsl_vector_alloc (p->numTrainImagesToUse); // size equal to m
        gsl_vector *tau = gsl_vector_alloc (p->numTrainImagesToUse); //size equal to min(m,n)
        gsl_linalg_cholesky_decomp(&A.matrix);
        gsl_linalg_cholesky_solve(&A.matrix, &b.vector, &x.vector);
//        gsl_linalg_QR_decomp (&A.matrix, tau); //
//        gsl_linalg_QR_lssolve(&A.matrix, tau, &b.vector, &x.vector, residual);

//        printf ("x = \n");
//        gsl_vector_fprintf (stdout, &x.vector, "%g");
        gsl_vector_free (tau);
        gsl_vector_free (residual);

        double sum_x = 0.0;

        for(i=0; i<p->numTrainImagesToUse; i++)
            sum_x += x_data[i];

        //////////////////// construct HR patch /////////////////////////////////
        for(j=0;j<p->numTrainImagesToUse;j++){
            for(i=0;i<testHRPatched->col;i++){
                //printf ("testHRPatched[%d].matrix[0][%d] = %f\n", patch, i, testHRPatched[patch].matrix[0][i]);
                testHRPatched[patch].matrix[0][i] += (x_data[j]/sum_x)*trainHRPatched[patch].matrix[idx[j]][i];
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
        free(ones);
        free(x_data);
        free(dist);
        free(idx);
        free(w);
    }
    deallocate_dynamic_matrix_double(C,testLRPatched->col);
    free(G);

    return testHRPatched;
}


Patch* reconstructionIterative(Patch* testLRPatched, Patch* trainLRPatched, Patch* trainHRPatched, Parameters* p){
    printf("start - reconstruction iterative\n");
    int numPatch = testLRPatched->numPatchX * testLRPatched->numPatchX;

    Patch* testHRPatched;
    testHRPatched = malloc(sizeof(Patch)*numPatch);

    int hrPW = p->hrPatchWidth;
    int patch;

    double **C, **G;
    C = allocate_dynamic_matrix_double(testLRPatched->col, p->numTrainImages);
    G = allocate_dynamic_matrix_double(p->numTrainImages,p->numTrainImages);

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
        float* w;
        w = malloc(sizeof(float)*p->numTrainImages);

        int i, j, k;

//        printf("reconstruction 1\n");
//        printf("numTrainImagesToUse: %d \n",p->numTrainImagesToUse );
//        printf("testLRPatched->col : %d\n", testLRPatched->col);
//
//        for(i=0;i<p->numTrainImages; i++){
//            printf("i: %d, dist : %f, position: %d\n", i, dist[i], idx[i]);
//        }
//        fflush(stdout);


        // loop over the closest `numTrainImagesToUse' trainLR
        for(i=0; i<testLRPatched->col; i++) {
            for(j=0; j<p->numTrainImages; j++){
                C[i][j] = testLRPatched[patch].matrix[0][i] - trainLRPatched[patch].matrix[idx[j]][i];
            }
        }

        printf("reconstruction 2\n");
        fflush(stdout);


        for(i=0; i<p->numTrainImages; i++) {
            for(j =0; j <p->numTrainImages; j++) {
                G[i][j] = 0;
//                printf("i: %d, j: %d \n", i, j);
                for (k = 0; k < testLRPatched->col; k++) {
                    G[i][j] += C[k][i] * C[k][j];
                }
                if(j == i) {
                    G[i][j] += p->tau * dist[idx[i]] * dist[idx[i]];
                }
            }
        }
        printf("reconstruction 3\n");
        fflush(stdout);
        ////////////////////// solve Gx=ones(M,1) by iterative CG ////////////////////////
        // https://github.com/vasia/Conjugate-Gradient-C-implementation
        double *Mdata   = calloc(p->numTrainImages, sizeof(double));
        double *ones    = malloc(sizeof(double)*p->numTrainImages);
        double *x       = calloc(p->numTrainImages,sizeof(double));
        double rtol     = 1e-8;

        for(i=0; i<p->numTrainImages;i++){
            ones[i] = 1.0;
        }

        jacobi_precond(Mdata, G, p->numTrainImages);
        int iter = precond_cg(matvec, psolve, G, Mdata,
                              ones, x, rtol, p->numTrainImages, p->maxiter);


        //////////////////// construct HR patch /////////////////////////////////
        double sum_x = 0.0;
        for(i=0; i<p->numTrainImages; i++){
//            printf("x[%d]: %f\n", i, x[i]);
            sum_x += x[i];

        }

        printf("here\n");
        fflush(stdout);

        for(j=0;j<p->numTrainImages;j++){
            for(i=0;i<testHRPatched->col;i++){
                //printf ("testHRPatched[%d].matrix[0][%d] = %f\n", patch, i, testHRPatched[patch].matrix[0][i]);
                testHRPatched[patch].matrix[0][i] += (x[j]/sum_x)*trainHRPatched[patch].matrix[idx[j]][i];
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
        free(Mdata);
        free(ones);
        free(x);
        free(dist);
        free(idx);
        free(w);
    }
    deallocate_dynamic_matrix_double(C,testLRPatched->col);
    deallocate_dynamic_matrix_double(G,p->numTrainImages);

    return testHRPatched;
}

Patch *reconstruction(Patch* testLRPatched, Patch* trainSetLRPatched, Patch* trainSetHRPatched, Parameters* p) {

    Patch* testHRPatched;

    printf("using basic c methods\n");
    if(p->method == 'a')
        testHRPatched = reconstructionAnalytic(testLRPatched, trainSetLRPatched, trainSetHRPatched, p);
    else if(p->method == 'i')
        testHRPatched = reconstructionIterative(testLRPatched, trainSetLRPatched, trainSetHRPatched, p);
    else{
        fprintf(stderr, "Reconstruction: wrong method!\n");
        exit(EXIT_FAILURE);
    }

    return testHRPatched;
}