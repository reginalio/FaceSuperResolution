//
// Created by Regina on 17/05/2015.
//

#ifndef FYP_LOAD_PGM_H
#define FYP_LOAD_PGM_H

typedef struct _PGMData {
    int row;
    int col;
    int max_gray;
    float **matrix;
} PGMData;

int ** allocate_dynamic_matrix(int row, int col);
float ** allocate_dynamic_matrix_float(int row, int col);
double ** allocate_dynamic_matrix_double(int row, int col);

void deallocate_dynamic_matrix_float(float **matrix, int row);
void deallocate_dynamic_matrix_double(double **matrix, int row);

void readPGM(const char *file_name, PGMData *data);
void writePGM(const char *filename, const PGMData *data);



#endif //FYP_LOAD_PGM_H
