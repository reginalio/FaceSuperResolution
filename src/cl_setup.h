//
// Created by Regina on 29/05/2015.
//

#ifndef FYP_CL_SETUP_H
#define FYP_CL_SETUP_H

#include "CL/opencl.h"
cl_platform_id setOpenCLPlatform();
cl_context createOpenCLContext(cl_platform_id platform);
cl_device_id getOpenCLDevices(cl_context context);
cl_program createOpenCLProgram(cl_context context, cl_device_id device);





#endif //FYP_CL_SETUP_H
