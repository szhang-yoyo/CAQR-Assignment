/*
 * Case Studies-Q2-TSQR implementation 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "mkl_lapacke.h"

/* Helper: compute Frobenius norm difference*/
double frob_diff(int n, double *A, double *B)
{
    double sum = 0.0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
        {
            double d = A[i + j*n] - B[i + j*n];
            sum += d*d;
        }
    return sqrt(sum);
}

/* Helper: Frobenius norm of one matrix */
double frob_norm(int n, double *A)
{
    double sum = 0.0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
        {
            double v = A[i + j*n];
            sum += v * v;
        }
    return sqrt(sum);
}

/* Compute G = A^T A
   A is m x n (column major) */
void compute_AtA(int m, int n, double *A, double *G)
{
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            G[i + j*n] = 0.0;

    for (int k = 0; k < m; k++)
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
                G[i + j*n] += A[k + i*m] * A[k + j*m];
}

/* Stack two R matrices and do QR again
   Input: R1, R2 (both n x n)
   Output: R_out (n x n) */
void combine_R(int n, double *R1, double *R2, double *R_out)
{
    int rows = 2*n;

    double *R_stack = malloc(rows*n*sizeof(double));
    double *tau     = malloc(n*sizeof(double));

    // stack R1 over R2
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < n; i++)
        {
            R_stack[i + j*rows]     = R1[i + j*n];
            R_stack[i+n + j*rows]   = R2[i + j*n];
        }
    }

    // QR on stacked matrix
    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, rows, n, R_stack, rows, tau);

    // take upper n x n part
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            R_out[i + j*n] = R_stack[i + j*rows];

    free(R_stack);
    free(tau);
}

/* MAIN PROGRAM */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = -1, size = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4)
    {
        if (rank == 0)
            printf("Please run with 4 processes!\n");
        MPI_Finalize();
        return 1;
    }

    /* Problem size (small test) */
    int m_global = 40;
    int n = 8;
    int m_local = m_global / size;

    /* Step 1: Create global matrix */
    double *A_global = malloc(m_global*n*sizeof(double));

    if (rank == 0)
    {
        srand(0);
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m_global; i++)
                A_global[i + j*m_global] =
                    (double)rand() / RAND_MAX;
    }

    MPI_Bcast(A_global, m_global*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Step 2: Extract local rows*/
    double *A_local = malloc(m_local*n*sizeof(double));

    for (int j = 0; j < n; j++)
        for (int i = 0; i < m_local; i++)
        {
            int global_row = rank*m_local + i;
            A_local[i + j*m_local] =
                A_global[global_row + j*m_global];
        }

    /* Step 3: Local QR */
    double *tau = malloc(n*sizeof(double));

    LAPACKE_dgeqrf(LAPACK_COL_MAJOR,
                   m_local, n,
                   A_local, m_local,
                   tau);

    // Extract R
    double *R_local = malloc(n*n*sizeof(double));

    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            R_local[i + j*n] =
                (i <= j) ? A_local[i + j*m_local] : 0.0;

    free(tau);

    /* Step 4: Tree reduction */

    // First level
    if (rank == 1 || rank == 3)
    {
        MPI_Send(R_local, n*n, MPI_DOUBLE,
                 rank-1, 0, MPI_COMM_WORLD);
    }

    if (rank == 0 || rank == 2)
    {
        double *R_recv = malloc(n*n*sizeof(double));
        MPI_Recv(R_recv, n*n, MPI_DOUBLE,
                 rank+1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double *R_new = malloc(n*n*sizeof(double));
        combine_R(n, R_local, R_recv, R_new);

        free(R_local);
        free(R_recv);
        R_local = R_new;
    }

    // Second level
    if (rank == 2)
    {
        MPI_Send(R_local, n*n, MPI_DOUBLE,
                 0, 1, MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
        double *R_recv = malloc(n*n*sizeof(double));
        MPI_Recv(R_recv, n*n, MPI_DOUBLE,
                 2, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double *R_final = malloc(n*n*sizeof(double));
        combine_R(n, R_local, R_recv, R_final);

        /* Step 5: Check correctness */

        double *G1 = malloc(n*n*sizeof(double));
        double *G2 = malloc(n*n*sizeof(double));

        compute_AtA(m_global, n, A_global, G1);
        compute_AtA(n, n, R_final, G2);  // R^T R via same routine

        double error = frob_diff(n, G1, G2);
        double normA = frob_norm(n, G1); 

        printf("\nTSQR Test (m=%d, n=%d)\n", m_global, n);
        printf("Relative error = %.3e\n",
               error / normA);

        free(G1);
        free(G2);
        free(R_recv);
        free(R_final);
    }

    free(R_local);
    free(A_local);
    free(A_global);

    MPI_Finalize();
    return 0;
}
