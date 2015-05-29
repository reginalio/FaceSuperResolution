//
// Created by Regina on 29/05/2015.
//

#ifndef FYP_CG_H
#define FYP_CG_H

#include <stdlib.h>
#include <string.h>
#include <libintl.h>
#include <stdio.h>

void psolve(double *Minvx, double *Mdata, double *x, int n);

void matvec(double *Ax, double **Adata, double *xvect, int n);

void jacobi_precond(double *M, double **Adata, int n);

int precond_cg(void (*matvec) (double *Ax, double **Adata, double *x, int n),
               void (*psolve) (double *Minvx, double *Mdata, double *x, int n),
               double **Adata, double *Mdata, double *b,
               double *x, double rtol, int n, int maxiter);


#endif //FYP_CG_H
