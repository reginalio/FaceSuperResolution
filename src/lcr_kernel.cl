__kernel void cov_matrix(global float* restrict testLRPatchedMatrix, 
							global float* restrict trainLRPatchedMatrix, 
							global float* restrict w, global int* restrict idx,
							int numTrainImagesToUse, int numPixelsInPatch, 
							global float* restrict C, global float* restrict G, 
							float tau, global float* restrict dist){

	int i, j, k;
	
	// loop over the closest `numTrainImagesToUse' trainLR
	for(i=0; i<numPixelsInPatch; i++) {
		for(j=0; j<numTrainImagesToUse; j++){
			C[i*numPixelsInPatch+j] = testLRPatchedMatrix[i] - trainLRPatchedMatrix[idx[j]*numPixelsInPatch+i];
		}
	}
	
	mem_fence(CLK_GLOBAL_MEM_FENCE);

	
	for(i=0; i<numTrainImagesToUse; i++) {
		for(j =0; j <numTrainImagesToUse; j++) {
			G[i * numTrainImagesToUse + j] = 0;
			for (k = 0; k < numPixelsInPatch; k++) {
				G[i * numTrainImagesToUse + j] += C[k*numPixelsInPatch+i] * C[k*numPixelsInPatch+j];     //TODO optimise the indices
			}
			if(j == i) {
				G[i * numTrainImagesToUse + j] += tau * dist[idx[i]] * dist[idx[i]];
			}
		}
	}
		

}

__kernel void cholesky_decomp(global float* restrict A, global float* restrict L, int n){

	int i,j,k;

	for (i = 0; i < n; i++){
        for (j = 0; j < (i+1); j++) {
            double s = 0;
            for (k = 0; k < j; k++)
                s += L[i * n + k] * L[j * n + k];
			
            L[i * n + j] = (i == j) ?
                           sqrt(A[i * n + i] - s) :
                           (1.0 / L[j * n + j] * (A[i * n + j] - s));
        }
	}
		
}
	
__kernel void solver(global float* restrict L, /*global float* b,*/ int n, 
						global float* restrict x, global float* restrict y){
	
	int i,j;
	
	// Forward solve Ly = b
	for (i = 0; i < n; i++)
	{
		y[i] = 1.0;	//b[i];
		for (j = 0; j < i; j++)
		{
			y[i] -= L[i*n + j] * y[j];
		}
		y[i] /= L[i*n + i];
	}
	// Backward solve L'x = y
	for (i = n - 1; i >= 0; i--)
	{
		x[i] = y[i];
		for (j = i + 1; j < n; j++)
		{
			x[i] -= L[j*n + i] * x[j];
		}
		x[i] /= L[i*n + i];
	}
	
}


#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void cholesky_decomp_double(global double* restrict A, global double* restrict L, int n){

	int i,j,k;

	for (i = 0; i < n; i++){
        for (j = 0; j < (i+1); j++) {
            double s = 0;
            for (k = 0; k < j; k++)
                s += L[i * n + k] * L[j * n + k];
			
            L[i * n + j] = (i == j) ?
                           sqrt(A[i * n + i] - s) :
                           (1.0 / L[j * n + j] * (A[i * n + j] - s));
        }
	}
		
}
	
__kernel void solver_double(global double* restrict L, /*global float* b,*/ int n, 
						global double* restrict x, global double* restrict y){
	
	int i,j;
	
	// Forward solve Ly = b
	for (i = 0; i < n; i++)
	{
		y[i] = 1.0;	//b[i];
		for (j = 0; j < i; j++)
		{
			y[i] -= L[i*n + j] * y[j];
		}
		y[i] /= L[i*n + i];
	}
	// Backward solve L'x = y
	for (i = n - 1; i >= 0; i--)
	{
		x[i] = y[i];
		for (j = i + 1; j < n; j++)
		{
			x[i] -= L[j*n + i] * x[j];
		}
		x[i] /= L[i*n + i];
	}
	
}