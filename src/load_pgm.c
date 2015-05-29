//
// Created by Regina on 17/05/2015.
//
// adapted from https://ugurkoltuk.wordpress.com/2010/03/04/an-extreme-simple-pgm-io-api/

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include "load_pgm.h"

#define HI(num) (((num) & 0x0000FF00) >> 8)
#define LO(num) ((num) & 0x000000FF)

int **allocate_dynamic_matrix(int row, int col) {
    int **ret_val;
    int i;

    ret_val = (int **)calloc(row,sizeof(int *));
    if (ret_val == NULL) {
        perror("memory allocation failure");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < row; ++i) {
        ret_val[i] = (int *)calloc(col,sizeof(int));
        if (ret_val[i] == NULL) {
            perror("memory allocation failure");
            exit(EXIT_FAILURE);
        }
    }

    return ret_val;
}

void deallocate_dynamic_matrix(int **matrix, int row)
{
    int i;

    for (i = 0; i < row; ++i)
        free(matrix[i]);
    free(matrix);
}

float **allocate_dynamic_matrix_float(int row, int col) {
    float **ret_val;
    int i;

    ret_val = (float **)calloc(row, sizeof(float *));
    if (ret_val == NULL) {
        perror("memory allocation failure");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < row; ++i) {
        ret_val[i] = (float *)calloc(col, sizeof(float));
        if (ret_val[i] == NULL) {
            perror("memory allocation failure");
            exit(EXIT_FAILURE);
        }
    }

    return ret_val;
}


double **allocate_dynamic_matrix_double(int row, int col) {
    double **ret_val;
    int i;

    ret_val = (double **)calloc(row, sizeof(double *));
    if (ret_val == NULL) {
        perror("memory allocation failure");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < row; ++i) {
        ret_val[i] = (double *)calloc(col, sizeof(double));
        if (ret_val[i] == NULL) {
            perror("memory allocation failure");
            exit(EXIT_FAILURE);
        }
    }

    return ret_val;
}

void deallocate_dynamic_matrix_float(float **matrix, int row) {
    int i;

    for (i = 0; i < row; ++i)
        free(matrix[i]);
    free(matrix);
}


void deallocate_dynamic_matrix_double(double **matrix, int row) {
    int i;

    for (i = 0; i < row; ++i)
        free(matrix[i]);
    free(matrix);
}

void SkipComments(FILE *fp)
{
    int ch;
    char line[100];

    while ((ch = fgetc(fp)) != EOF && isspace(ch))
        ;
    if (ch == '#') {
        fgets(line, sizeof(line), fp);
        SkipComments(fp);
    } else
        fseek(fp, -1, SEEK_CUR);
}


void readPGM(const char *file_name, PGMData *data) {
    FILE *pgmFile;
    char version[3];
    int i, j;
    int lo, hi;

    printf("reading %s\n", file_name );
    fflush(stdout);
    pgmFile = fopen(file_name, "rb");
    if (pgmFile == NULL) {
        perror("cannot open file to read");
        exit(EXIT_FAILURE);
    }

    fgets(version, sizeof(version), pgmFile);
    if (strcmp(version, "P5")) {
        fprintf(stderr, "Wrong file type!\n");
        exit(EXIT_FAILURE);
    }

    SkipComments(pgmFile);
    fscanf(pgmFile, "%d", &data->col);
    SkipComments(pgmFile);
    fscanf(pgmFile, "%d", &data->row);
    SkipComments(pgmFile);
    fscanf(pgmFile, "%d", &data->max_gray);
    fgetc(pgmFile);

    data->matrix = allocate_dynamic_matrix_float(data->row, data->col);
    for (i = 0; i < data->row; ++i)
        for (j = 0; j < data->col; ++j) {
            lo = fgetc(pgmFile);
            data->matrix[i][j] = (float)lo/(float)data->max_gray;
        }

    fclose(pgmFile);

}


void writePGM(const char *filename, const PGMData *data) {
    FILE *pgmFile;
    int i, j;
    int hi, lo;

    pgmFile = fopen(filename, "wb");
    if (pgmFile == NULL) {
        perror("cannot open file to write");
        exit(EXIT_FAILURE);
    }

    fprintf(pgmFile, "P5 ");
    fprintf(pgmFile, "%d %d ", data->col, data->row);
    fprintf(pgmFile, "%d ", data->max_gray);

    for (i = 0; i < data->row; ++i)
        for (j = 0; j < data->col; ++j) {
            lo = LO((unsigned)(data->matrix[i][j]*data->max_gray));
            fputc(lo, pgmFile);
        }


    fclose(pgmFile);
    deallocate_dynamic_matrix_float(data->matrix, data->row);
}