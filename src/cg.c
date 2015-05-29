//
// Created by Regina on 29/05/2015.
//

#include "cg.h"

void matvec(double *Ax, double **Adata, double *xvect, int n) {
    int i, j;

    for (i = 0; i < n; ++i) {
        Ax[i] = 0;
        for (j = 0; j < n; ++j) {
            Ax[i] += Adata[i][j]*xvect[j];
        }
    }
}

void psolve(double *Minvx, double *Mdata, double *x, int n) {
    int i;

    for(i=0; i<n; i++){
        Minvx[i] = 1/Mdata[i]*x[i];
    }
}

void jacobi_precond(double *M, double **Adata, int n) {
    int i;

    for (i = 0; i < n; ++i) {
        M[i] += Adata[i][i];
    }
}

void axpy(double *dest, double a, double *x, double *y, int n)
{
    int i;
    for (i = 0; i < n; ++i)
        dest[i] = a * x[i] + y[i];
}

double ddot(double *x, double *y, int n)
{
    int i;
    double final_sum = 0;

    for (i = 0; i < n; ++i)
        final_sum += x[i] * y[i];

    return final_sum;
}

int precond_cg(void (*matvec) (double *Ax, double **Adata, double *x, int n),
               void (*psolve) (double *Minvx, double *Mdata, double *x, int n),
               double **Adata, double *Mdata, double *b,
               double *x, double rtol, int n, int maxiter) {
    const int nbytes = n * sizeof(double);

    double bnorm2;              /* ||b||^2 */
    double rnorm2;              /* ????? ????????? ??? ????????? */
    double rz, rzold;           /* r'*z ??? 2 ?????????? ??????????? */
    double alpha, beta;
    double rz_local,rnorm2_local,bnorm2_local;

    double *s;                  /* ?????????? ?????????? */
    double *r;                  /* ????????         */
    double *z;                  /* ????????? ???????? */

    int i = 0,j;                /* ???????? ????????? */

    s = (double *) malloc(nbytes);
    r = (double *) malloc(nbytes);
    z = (double *) malloc(nbytes);


    bnorm2    = ddot(b, b, n);

    memset(x, 0, nbytes);	//???????????? ?????
    memcpy(r, b, nbytes);	//??? ????????? - r0=b-A*x0 (x0=0)

    psolve(z, Mdata, r, n);	//???????? ??? preconditioner - z0 = (M ???? -1)*r0

    memcpy(s, z, nbytes);	//???????????? ??????????? ??????????	- p0 = z0

    /* ???????????? rz ??? rnorm2 */
    rz        = ddot(r, z, n);
    rnorm2    = ddot(r, r, n);


    // printf("rz=%2.15f,alpha=%2.15f,rnorm=%2.15f,bnorm=%f\n",rz,alpha,rnorm2,bnorm2);

    for (i = 0; i < maxiter ; ++i) {
        printf("#%d\n", i);

        matvec(z, Adata, s, n);	//z:=A*pk


        /* Ddot*/
        alpha = rz / ddot(s, z, n);	//ak = rkT*zk/pkT*A*pk
        axpy(x, alpha, s, x, n);	//xk+1 = xk + ak*pk
        axpy(r, -alpha, z, r, n);	//rk+1 = rk - ak*A*pk

        psolve(z, Mdata, r, n);		//zk+1 = (M ???? -1)*rk+1

        rzold = rz;


        rz = ddot(r, z, n);  		//rTk+1*zk+1
        beta = -rz / rzold;		//? = rTk+1*zk+1/rTk*zk
        axpy(s, -beta, s, z, n);	//pk+1 = zk+1+?k*pk

        printf("(%d)rz=%2.15f,alpha=%2.15f,rnorm=%2.15f\n",i,rz,alpha,rnorm2);


        /* check error */
        rnorm2     = ddot(r, r, n);
        if(rnorm2 <= bnorm2 * rtol * rtol)
            break;

    }

    free(z);
    free(r);
    free(s);

    if (i >= maxiter)
        return -1;

    return i;
}