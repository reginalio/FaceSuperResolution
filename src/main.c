#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "load_pgm.h"
#include "util.h"

#ifdef USE_OPENCL
#include "reconstruction_cl.h"
#endif

int main() {
    static Parameters* p;
    p = setParameters();
//    printf("p lrwidth: %d\n", p->lrWidth);
//    printf("p lroverlap: %d\n", p->lrOverlap);
//    printf("p lrpw: %d\n", p->lrPatchWidth);

    // read training images
    int i;
    char istr[3];
    PGMData trainSetLR[NUMTRAINIMAGES];
    PGMData trainSetHR[NUMTRAINIMAGES];


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

    printf("hr[0] matrix [8][15]: %f\n", trainSetHR[0].matrix[8][15]);
    fflush(stdout);

    // Training images: divide to patches
    Patch *trainSetLRPatched;
    Patch *trainSetHRPatched;
    trainSetLRPatched = divideToPatches(trainSetLR, NUMTRAINIMAGES, p, 'l');
    trainSetHRPatched = divideToPatches(trainSetHR, NUMTRAINIMAGES, p, 'h');

    //printf("patch2 matrix2 %d\n", trainSetLRPatched[2].matrix[2][1]);
    printf("done dividing patch\n");
    fflush(stdout);

    // transform trainPatched to test combinePatches
//    Patch* testPatched;
//    testPatched = malloc(49*sizeof(Patch));
//    int patch;
//    for(patch=0; patch<49; patch++){
//        testPatched[patch].patchNumber = patch;
//        testPatched[patch].matrix = allocate_dynamic_matrix_float(p->hrPatchWidth*p->hrPatchWidth, 1);
//        for(i=0;i<p->hrPatchWidth*p->hrPatchWidth; i++)
//            testPatched[patch].matrix[i][0] = trainSetHRPatched[patch].matrix[i][0];
//    }
//    printf("here\n");
//    fflush(stdout);

    // read test image
    char fnTestLR[80];
    strcpy(fnTestLR, "dataset/testLR/Face20.pgm");
    PGMData testLR;
    readPGM(fnTestLR, &(testLR));
    Patch *testLRPatched, *testHRPatched;
    testLRPatched = divideToPatches(&testLR, 1, p, 'l');

    // reconstruction
    #ifdef USE_OPENCL
    testHRPatched = reconstruction_cl(testLRPatched, trainSetLRPatched, trainSetHRPatched, p);
    #else
    testHRPatched = reconstruction(testLRPatched, trainSetLRPatched, trainSetHRPatched, p);
    #endif


    printf("done resconstruction\n");
    fflush(stdout);




    // combine patches
    PGMData testRecombinedImage;
    testRecombinedImage = combinePatches(testHRPatched, p);
    writePGM("testCombined.pgm", &(testRecombinedImage));

    //////////////// debug //////////////////
//    int j;
//    for(i=10; i<20; i++){
//        for(j=10; j<20; j++){
//            printf("\t%f", testRecombinedImage.matrix[i][j]);
//        }
//        printf("\n");
//    }

//    printf("\n");
//    for(i=0; i<10; i++){
//        for(j=0; j<10; j++){
//            printf("\t%d", (uint)(testRecombinedImage.matrix[i][j]*testRecombinedImage.max_gray));
//        }
//        printf("\n");
//    }



    printf("wrote to testCombined.pgm\n PROGRAM PASSED\n");
    fflush(stdout);
//    free(fnLR);
//    free(fnHR);
    free(trainSetLRPatched);
    free(trainSetHRPatched);
    free(p);




    return 0;
}