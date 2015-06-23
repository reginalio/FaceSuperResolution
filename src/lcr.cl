#ifndef M			// number of training images to use
 #define M 20
#endif
#ifndef N			// number of pixels in an LR patch
 #define N 16
#endif
#define TAU 0.04

// __kernel void cov_matrix(	global float* restrict testLRPatchedMatrix, 
							// global float* restrict trainLRPatchedMatrix, 
							// global int* restrict idx, global float* restrict C, 
							// global float* restrict G, global float* restrict dist){

	// int i, j, k;
	
	// //loop over the closest `M' trainLR
	// for(i=0; i<N; i++) {
		// for(j=0; j<M; j++){
			// C[i*N+j] = testLRPatchedMatrix[i] - trainLRPatchedMatrix[idx[j]*N+i];
		// }
	// }
	
	// for(i=0; i<M; i++) {
		// for(j=0; j <M; j++) {
			// G[i * M + j] = 0.0;
			// for (k = 0; k < N; k++) {
				// G[i * M + j] += C[k*N+i] * C[k*N+j];     //TODO optimise the indices
			// }
			// if(j == i) {
				// G[i * M + j] += TAU * dist[idx[i]] * dist[idx[i]];
			// }
		// }
	// }
		
// }

__kernel void cholesky_decomp(global float* restrict A, global float* restrict L){

	int i,j,k;

	for (i = 0; i < M; i++){
        for (j = 0; j < (i+1); j++) {
            double s = 0;
            for (k = 0; k < j; k++)
                s += L[i * M + k] * L[j * M + k];
			
            L[i * M + j] = (i == j) ?
                           sqrt(A[i * M + i] - s) :
                           (1.0 / L[j * M + j] * (A[i * M + j] - s));
        }
	}
		
}
	
__kernel void solver(global float* restrict L, global float* restrict x, global float* restrict y){
	
	int i,j;
	
	// Forward solve Ly = b
	for (i = 0; i < M; i++)
	{
		y[i] = 1.0;	//b[i];
		for (j = 0; j < i; j++)
		{
			y[i] -= L[i*M + j] * y[j];
		}
		y[i] /= L[i*M + i];
	}
	// Backward solve L'x = y
	for (i = M - 1; i >= 0; i--)
	{
		x[i] = y[i];
		for (j = i + 1; j < M; j++)
		{
			x[i] -= L[j*M + i] * x[j];
		}
		x[i] /= L[i*M + i];
	}
	
}

