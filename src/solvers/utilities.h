/******************************************************************************/
/* Authors    : Hussam Al-Daas                                                */
/* Creation   : 25/06/2017                                                    */
/* Description: Enlarged GMRES                                                */
/******************************************************************************/

/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/

#ifndef UTILITIES_H
#define UTILITIES_H

#include <metis.h>

//#define EGMRESDEBUG 1
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/

void quickSortDoubleWithPerm(double **v, int m, int **p);
int PrintMat(double* A, int m, int n, int lda, char* name);
int PrintMatZ(MKL_Complex16* A, int m, int n, int lda, char* name);
int PrintIVec(int *piv, int s, char* name);
int PrintDVec(double* v, int s, char* name);
int PrintDVecZ(MKL_Complex16* v, int s, char* name);
int GetInfoFileCSR(const char* filename, int* M, int* nnz);
int AllocateMatCSR(int M, int nnz, int** rowPtr, int** colInd, double** val);
int LoadMatCSR( const char* filename, int* M, int* nnz, int** rowPtr, int** colInd, double** val);
idx_t* K_way(int M, int nnz, int* rowPtr, int* colInd, double* val, idx_t nbparts);
int K_wayPermutationVector(char structure, int M, int nnz, int* rowPtr, int* colInd, double* val, int** perm, int nblock, int** posB);
int PrepareNonSymmetricGraphForMetis(int M, int nnz1, int* rowPtr1, int* colInd1, double* val1, int* nnz2, int** rowPtr2, int** colInd2, double** val2);
int PrepareSymmetricGraphForMetis(int M, int nnz1, int* rowPtr1, int* colInd1, double* val1, int* nnz2, int** rowPtr2, int** colInd2, double** val2);
int GetBlockPosition(int size_parts, idx_t *parts, int npart, int** pos);
int GetIntPermArray(idx_t npart, idx_t size_parts, idx_t *parts, int** perm);
int PermuteCSR(int M, int nnz, int* rowPtr_in, int* colInd_in, double* val_in, int* rowPtr_out, int* colInd_out, double* val_out, int* perm);
int PermuteVec(double* v_in, double* v_out, int* perm, int m);
int GetRowPanel(int M_in, int nnz_in, int* rowPtr_in, int* colInd_in, double* val_in, int *M_out, int* nnz_out, int** rowPtr_out, int** colInd_out, double** val_out, int* pos, int numBlock);
int GetColBlockPos(int M_in, int nnz_in, int* rowPtr_in, int* colInd_in, double* val_in, int* pos, int* colPos, int nblock);
int GetCommDep(int* colPos, int M, int nblock, int rank, int** dep);
int GetDiagonalBlock(int M, int nnz, int* rowPtr, int* colInd, double* val, int* nnz_out, int* rowPtr_out, int* colInd_out, double* val_out, int* pos, int* colPos);
int GetBlock(int m, int n, int nnz, int* rowPtr, int* colInd, double* val, int n_out, int* lnnz_out, int* rowPtr_out, int* colInd_out, double* val_out, int* posB, int* colPos, int BlockNum, int size, int* work);
int TSQR(MPI_Comm comm, int m, int n, double* A, double* Q, double* R, double* work);
int ReadDVecFromFile(char* filename, int m, double* v);
int ReadIVecFromFile(char* filename, int m, int* v);

int flops_dgemm_sum(int m, int n, int k);
int flops_dgemm(int m, int n, int k);
int flops_dormqr(int m, int n, int k);
int flops_dgeqrf(int m, int n);
int flops_dpotrf(int n);
int flops_svd(int m, int n);
int flops_dgetrf(int m);
int flops_dtrsm(int m, int n);
int flops_spmm(int nnz, int n);
int flops_memset(int n);
int flops_allreduce(int n, int ND);
int flops_copy(int m, int n);
int flops_sum(int m, int n);
int flops_zunmqr(int m, int n, int k);
int flops_zgemm_sum(int m, int n, int k);
int flops_zgemm(int m, int n, int k);
int flops_zgeqrf(int m, int n);
int flops_zpotrf(int n);
int flops_zgesvd(int m, int n);
int flops_zgetrf(int m);
int flops_ztrsm(int m, int n);
int flops_spmm_z(int nnz, int n);
int flops_memset_z(int n);
int flops_allreduce_z(int n, int ND);
int flops_copy_z(int m, int n);
int flops_sum_z(int m, int n);
int flops_zgebal(int m);
int flops_zgehrd(int n, int ilo, int ihi);
int flops_zunghr(int ilo, int ihi);
int flops_zungqr(int m, int n);
int flops_zhseqr(int n);
int flops_ztrexc(int n, int ifst, int ilst);
int flops_zgebak(int m, int n);
int flops_zggbal(int n);
int flops_zgghd3(int n);
int flops_zhgeqz(int n);
int flops_ztgexc(int n, int ifst, int ilst);
int flops_zggbak(int m, int n);
#endif
