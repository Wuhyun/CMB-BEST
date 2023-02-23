#ifndef ARRAYS_H
#define ARRAYS_H

double *create_array(int d1);
double **create_2D_array(int d1, int d2);
double ***create_3D_array(int d1, int d2, int d3);

double **create_2D_array_from_data(double *data, int d1, int d2);
double ***create_3D_array_from_data(double *data, int d1, int d2, int d3);

int *create_int_array(int d1);
int **create_2D_int_array(int d1, int d2);


void free_array(double *arr);
void free_2D_array(double **arr);
void free_3D_array(double ***arr);
void free_2D_array_from_data(double **arr);
void free_3D_array_from_data(double ***arr);


#endif /* ARRAYS_H */
