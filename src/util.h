//
// Created by Regina on 18/05/2015.
//

#ifndef FYP_UTIL_H
#define FYP_UTIL_H

#include "load_pgm.h"

#define NUMTRAINIMAGES 360

#ifndef MM
    #define MM 30        // number of training images to use in reconstruction
#endif

#ifndef LRWIDTH
    #define LRWIDTH 15
#endif
#ifndef LRPW
    #define LRPW 4
#endif
#ifndef LRO
    #define LRO 1
#endif
#ifndef N
    #define N 16     // number of pixels in an LR patch
#endif

#define RATIO 4
#define TAU 0.04
#define MAXITER 10

#ifndef NUMTESTFACE
#define NUMTESTFACE 40
#endif

typedef struct _Parameters {
    int ratio;
    double tau;

    int lrWidth;
    int lrPatchWidth;
    int lrOverlap;
    int hrWidth;
    int hrPatchWidth;
    int hrOverlap;

    int numTrainImages;
    int numTrainImagesToUse;

    char method;    // a: analytic, i: iterative
    int maxiter;

} Parameters;

typedef struct _Patch{
    int numPatchX;  // number of patches in the x direction
    int row;        // number of images
    int col;        // patch width * patch width
    float **matrix;
}Patch;

// static global variable
float* baseDistArray;

Parameters* setParameters();
Patch * divideToPatches(PGMData *data, const int numImages, Parameters* p, const char mode);
PGMData combinePatches(Patch* patches, Parameters* p);
float * calcDistance(Patch* testLRPatched, Patch* trainLRPatched);
int * sortDistIndex(float* dist, int numElements);

Patch * reconstruction(Patch* testLRPatched, Patch* trainSetLR, Patch* trainSetHR, Parameters* p);


#endif //FYP_UTIL_H
