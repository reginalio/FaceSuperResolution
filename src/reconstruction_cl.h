//
// Created by Regina on 29/05/2015.
//

#ifndef FYP_CL_RECONSTRUCTION_H
#define FYP_CL_RECONSTRUCTION_H
#include "load_pgm.h"
#include "util.h"
#include "CL/opencl.h"

Patch * reconstruction_cl(Patch* testLRPatched, Patch* trainSetLRPatched, Patch* trainSetHRPatched, Parameters* p);


#endif //FYP_CL_RECONSTRUCTION_H
