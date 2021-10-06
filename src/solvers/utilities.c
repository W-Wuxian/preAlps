/******************************************************************************/
/* Authors    : Hussam Al-Daas                                                */
/* Creation   : 25/06/2017                                                    */
/* Description: Enlarged GMRES                                                */
/******************************************************************************/

/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
/*	STD	 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <mpi.h>
#include <mkl.h>
#include "utilities.h"

#define CSR_MAX_LINE_LENGTH  1025
/******************************************************************************/
/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/
typedef struct{
	int col;
	double val;
}ColVal;

static int compare_absval_colval (void const *a, void const *b){
	ColVal const *pa = (ColVal const*)a;
	ColVal const *pb = (ColVal const*)b;
	return (fabs(pa->val) - fabs(pb->val) >= 0) ? 1 : -1;
}

static int compare (void const *a, void const *b){
   int const *pa = (const int*)a;
   int const *pb = (const int*)b;
   return *pa - *pb;
}

static int compare_col (void const *a, void const *b){
   ColVal const *pa = (ColVal const*)a;
   ColVal const *pb = (ColVal const*)b;
   return pa->col - pb->col;
}

void quickSortDoubleWithPerm(double **v, int m, int **p){
	ColVal *tab = NULL;
	tab = (ColVal*)malloc(m * sizeof(ColVal));
	if(tab == NULL){
		printf("Error during malloc for tab size = %d\n", m);
		exit(1);
	}
	for(int i = 0; i < m; i++){
		tab[i].col = (*p)[i];
		tab[i].val = (*v)[i];
	}
	qsort(tab, m, sizeof(ColVal), compare_absval_colval);
	for(int i = 0; i < m; i++){
		(*p)[i] = tab[i].col;
	}
	free(tab);
}

void quickSortWithValues(int array[], int begin, int end, double values[]){
	ColVal *tab;
	int size = end + 1 - begin;
	if((tab = (ColVal*) malloc(size * sizeof(ColVal))) == NULL){
		fprintf(stderr, "Error during malloc for tab size = %d = %d - %d\n", size, end - 1, begin);
		exit(1);
	}
	for(unsigned int i = 0; i < size; i++){
		tab[i].col = array[i + begin];
		tab[i].val = values[i + begin];
	}
	qsort(tab, size, sizeof(ColVal), compare_col);

	for(unsigned int i = 0; i < size; i++){
		array[i + begin]	= tab[i].col;
		values[i + begin] = tab[i].val;
	}
	free(tab);
}

void quickSort(int array[], int begin, int end){
	qsort(array + begin, end + 1 - begin, sizeof(int), compare);
}

int PrintMat(double* A, int m, int n, int lda, char* name){
	printf("Matrix %s of size %d x %d\n", name, m, n);
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			printf("%.4e\t", A[i + j * lda]);
		}
		printf("\n");
	}
	return 0;
}

int PrintMatZ(MKL_Complex16* A, int m, int n, int lda, char* name){
	printf("Matrix %s of size %d x %d\n", name, m, n);
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			printf("%.4e + i %.4e\t", A[i + j * lda].real, A[i + j * lda].imag);
		}
		printf("\n");
	}
	return 0;
}
int PrintIVec(int* v, int s, char* name){
	printf("Integer vector  %s \n", name);
	for(int i = 0; i < s; i++){
		printf("%d\t", v[i]);
	}
	printf("\n");
	return 0;
}

int PrintDVec(double* v, int s, char* name){
	printf("Double vector  %s \n", name);
	for(int i = 0; i < s; i++){
		printf("%f\t", v[i]);
	}
	printf("\n");
	return 0;
}

int PrintDVecZ(MKL_Complex16* v, int s, char* name){
	printf("Double vector  %s \n", name);
	for(int i = 0; i < s; i++){
		printf("%f + i %f\t", v[i].real, v[i].imag);
	}
	printf("\n");
	return 0;
}

int GetInfoFileCSR(const char* filename, int* M, int* nnz){
	int ierr = 0;
  FILE *fd;
  char buf[CSR_MAX_LINE_LENGTH];
	char* ptr = NULL;
	int blocksize = 0;

	fd = fopen(filename, "r");
	if(!fd)
	{
		printf("Matrix file can not be opened");
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
  // Get the header line
  ptr = fgets(buf, CSR_MAX_LINE_LENGTH, fd);
  sscanf(buf, "%d %d %d", M, &blocksize, nnz);
	fclose(fd);
	return ierr;
}

int AllocateMatCSR(int M, int nnz, int** rowPtr, int** colInd, double** val){
	*rowPtr	= (int*) malloc( (M + 1) * sizeof(int));
	*colInd	= (int*) malloc( nnz			* sizeof(int));
	*val		= (double*)				malloc( nnz			* sizeof(double));
	return 0;
}

int LoadMatCSR( const char* filename, int* M, int* nnz, int** rowPtr, int** colInd, double** val){
	int ierr = 0;
  FILE *fd;
  char buf[CSR_MAX_LINE_LENGTH];
	char* ptr = NULL;
	int blocksize, nread;
	fd = fopen(filename, "r");
	if(!fd)
	{
		printf("Matrix file can not be opened\n");
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
  // Get the header line
  ptr = fgets(buf, CSR_MAX_LINE_LENGTH, fd);
  sscanf(buf, "%d %d %d", M, &blocksize, nnz);
	printf("Matrix of size %d , blocksize = %d, nnz = %d \n", *M, blocksize, *nnz);
		
	if(*rowPtr == NULL)
		printf("rowPtr Not allocated\n");
	for(unsigned int i = 0; i <= (*M); i++){
		nread = fscanf(fd, "%d", *rowPtr + i);
		(*rowPtr)[i] = (*rowPtr)[i] - 1;
	}
	for(unsigned int i = 0; i < *nnz; i++){
		nread = fscanf(fd, "%d", *colInd + i);
		(*colInd)[i]--;
	}
	for(unsigned int i = 0; i < *nnz; i++){
		nread = fscanf(fd, "%lf", *val + i);
	}
	fclose(fd);
	return ierr;
}

idx_t* K_way(int M, int nnz, int* rowPtr, int* colInd, double* val, idx_t nbparts){
  int ierr			= 0;
  idx_t nvtxs   = 0;
	idx_t ncon    = 1;
  idx_t *parts  = NULL;
	idx_t *xadj   = NULL;
  idx_t *adjncy = NULL;
  idx_t objval  = 0;
	//	Casting of M to be used by METIS
	nvtxs = (idx_t) M;
	//	Pointers used by METIS
	parts   = (idx_t*) malloc(nvtxs							* sizeof(idx_t));
	xadj    = (idx_t*) malloc((nvtxs + 1)       * sizeof(idx_t));
	adjncy  = (idx_t*) malloc(nnz								* sizeof(idx_t));
	//	Copy rowPtr and colInd to xadj and adjncy (NECESSARY SINCE THEY DON'T HAVE THE SAME TYPE!)
	for(unsigned int i = 0; i < nvtxs + 1; i++)
		xadj[i] = rowPtr[i];
	for(unsigned int i = 0; i < nnz; i++)
		adjncy[i] = colInd[i];
	//	METIS interface call
	ierr = METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, NULL, NULL, NULL, &nbparts, NULL, NULL, NULL, &objval, parts);
	//	Match the return value returned by METIS_PartGraphKway
	switch(ierr){
	  case METIS_ERROR:
	  	fprintf(stderr, "Error\n");
	  	exit(1);
	  	break;
	  case METIS_ERROR_INPUT:
	  	fprintf(stderr, "Error INPUT\n");
	  	exit(1);
	  	break;
	  case METIS_ERROR_MEMORY:
	  	fprintf(stderr, "Error MEMORY\n");
	  	exit(1);
	  case METIS_OK:
	  	break;
	  default:
	  	fprintf(stderr, "Unknown value returned by METIS_PartGraphKway\n");
	  	exit(1);
	}
	free(xadj);
	free(adjncy);
	return parts;
}

/**
 * \fn int K_wayPermutationVector(char structure, int M, int nnz, int* rowPtr, int* colInd, double* val, int** perm, int nblock, int** posB
 * \brief Function calls Kway partitionning algorithm and return a new matrix permuted
 * \param (char structure, int M, int nnz, int* rowPtr, int* colInd, double* val) represents the original matrix
 * \param perm This vector contains data of permutation
 * \param nblock Number of partition for Kway
 * \param posB This vector contains first index of each block
 */
int K_wayPermutationVector(char structure, int M, int nnz, int* rowPtr, int* colInd, double* val, int** perm, int nblock, int** posB){
#if EGMRESDEBUG
	printf("Function K_wayPermutation\n");
#endif
	printf("Function K_wayPermutation\n");
  int ierr  = 0;
	int nnz2 = 0;
	int* rowPtr2 = NULL;
	int* colInd2 = NULL;
	double*				val2		=	NULL;
  idx_t nvtxs   = 0;
  idx_t *parts  = NULL;
  // If nothing to do
  if(nblock == 1){
    (*posB)[0]  = 0;
    (*posB)[1]  = M;//[WARNING] It could be with -1
    return ierr;
  }
  // The matrix has to be symmetric and without diagonal elements
  if(structure == 'N'){
#if EGMRESDEBUG
		printf("Prepare Matrix for Metis\n");
#endif
    ierr = PrepareNonSymmetricGraphForMetis(M, nnz, rowPtr, colInd, val, &nnz2, &rowPtr2, &colInd2, &val2);
#if EGMRESDEBUG
		printf("Prepare Matrix for Metis done\n");
#endif
	}
  else{
	  ierr = PrepareSymmetricGraphForMetis(M, nnz, rowPtr, colInd, val, &nnz2, &rowPtr2, &colInd2, &val2);
	}
  nvtxs = (idx_t) M;
#if EGMRESDEBUG
	printf("Call K way Metis\n");
#endif
  parts = K_way(M, nnz2, rowPtr2, colInd2, val2, nblock);
#if EGMRESDEBUG
	printf("Call K way Metis done\n");
#endif
#if EGMRESDEBUG
	printf("Get Block Pos\n");
#endif
  ierr = GetBlockPosition(M, parts, nblock, posB);
#if EGMRESDEBUG
	printf("Get Permutation array\n");
#endif
	ierr = GetIntPermArray(nblock, nvtxs, parts, perm);
  if(parts != NULL) 
    free(parts);
		/*
	free(colInd2);
	free(rowPtr2);
	if(structure == 'S');
		free(val2);
		*/
	mkl_free(colInd2);
	mkl_free(rowPtr2);
	if(structure == 'S');
		mkl_free(val2);
  return ierr;
}

int PrepareNonSymmetricGraphForMetis(int M, int nnz1, int* rowPtr1, int* colInd1, double* val1, int* nnz2, int** rowPtr2, int** colInd2, double** val2){
#if EGMRESDEBUG
	printf("Prepare nonsym Matrix to K way\n");
#endif
  int ierr = 0;
	*rowPtr2 = (int*) mkl_calloc((M + 1), sizeof(int), 32);
	//*rowPtr2 = (int*) calloc((M + 1), sizeof(int));
	//	Number of added values to the original matrix in order to symmetrize it
	int nbVal = 0;
	int valueAdded = 0;
  for(unsigned int i = 0; i < M; i++){
	  for(unsigned int j = rowPtr1[i]; j < rowPtr1[i + 1]; j++){
		  int tmp_col = colInd1[j];
		  //if not a diagonal element
		  if(tmp_col != i){
			  valueAdded = 0;
			  //For each nnz of the column
			  for(unsigned int k = rowPtr1[tmp_col]; k < rowPtr1[tmp_col + 1]; k++){
				  if(colInd1[k] == i){
					  valueAdded = 1;
					  break;
				  }
				  else if(colInd1[k] > i){
					  valueAdded = 1;
					  ((*rowPtr2)[tmp_col + 1])++;
					  nbVal++;
					  break;
				  }
				}
			  if(valueAdded == 0){
				  ((*rowPtr2)[tmp_col + 1])++;
				  nbVal++;
			  }
		  }
	  }
  }
	*nnz2 = nnz1 + nbVal - M;
  *colInd2 = (int*) mkl_malloc( *nnz2 * sizeof(int), 32);
  //*colInd2 = (int*) malloc( *nnz2 * sizeof(int));
	int valueIgnored = 0;
	//init by nnz2
	for(unsigned int i = 0; i < *nnz2; i++) (*colInd2)[i] = *nnz2;
	int sum = 0;
	for(unsigned int i = 0; i < M; i++){
		//compute the real number of values in the row i
		sum += (*rowPtr2)[i + 1] - 1;
		(*rowPtr2)[i + 1] = rowPtr1[i + 1] + sum;
	}
  for(unsigned int i = 0; i < M; i++){
	  //route the row i
	  for(unsigned int j = rowPtr1[i]; j < rowPtr1[i + 1]; j++){
		  int tmp_col = colInd1[j];
		  //if value is not the diagonal
		  if(tmp_col != i){
			  //copy values from original matrix
			  for(unsigned int ii = (*rowPtr2)[i]; ii < (*rowPtr2)[i + 1]; ii++){
				  if((*colInd2)[ii] == *nnz2){
					  (*colInd2)[ii] = tmp_col;
					  break;
				  }
			  }
			  valueIgnored = 0;
			  //looking for if i ( the row ) has to be added into tmp_col row
			  for(unsigned int k = rowPtr1[tmp_col]; k < rowPtr1[tmp_col + 1]; k++){
				  //if exists so continue
				  if(i == colInd1[k]){
					  valueIgnored = 1;
					  break;
				  }
				  else if(i < colInd1[k]){
					  break;
				  }
			  }
			  //if all values are less than the row index
			  if(valueIgnored == 0){
				  valueAdded = 0;
				  for(unsigned int ii = (*rowPtr2)[tmp_col]; ii < (*rowPtr2)[tmp_col + 1]; ii++){
						if((*colInd2)[ii] == *nnz2){
						  valueAdded = 1;
						  (*colInd2)[ii] = i;
						  break;
						}
					}
			  }
		  }
	  }
  }

	for(unsigned int i = 0; i < M; i++){
		quickSort(*colInd2, (*rowPtr2)[i], (*rowPtr2)[i + 1] - 1);
	}
	*val2 = NULL;
	return ierr;
}

int PrepareSymmetricGraphForMetis(int M, int nnz1, int* rowPtr1, int* colInd1, double* val1, int* nnz2, int** rowPtr2, int** colInd2, double** val2){
	int del				= 0;
  int ierr      = 0;
	int lnAdd     = 0;
  int nAdd      = 0;
	//Assumption: the diagonal elements are all non zero
  *nnz2					= nnz1 - M;
	*rowPtr2	= (int*) mkl_malloc((M + 1)	* sizeof(int), 32);
	*colInd2	= (int*) mkl_malloc( *nnz2		* sizeof(int), 32);
	*val2			= (double*)				mkl_malloc( *nnz2		* sizeof(double),				64);
	//*rowPtr2	= (int*) malloc((M + 1)	* sizeof(int));
	//*colInd2	= (int*) malloc( *nnz2		* sizeof(int));
	//*val2			= (double*)				malloc( *nnz2		* sizeof(double)			);
	(*rowPtr2)[0] = 0;
	for(unsigned int i = 0; i < M; i++){
		lnAdd   = 0;
		del			= 0;
		for(unsigned int j = rowPtr1[i]; j < rowPtr1[i + 1]; j++){
			if(colInd1[j] == i){
				del  = 1;
				continue;
			}
			(*colInd2)[nAdd]  = colInd1[j];
			(*val2)[nAdd]     = val1[j];
			nAdd++;
			lnAdd++;
		}
		if(del == 0){
			fprintf(stderr, "Error, no diagonal value on row %d\n", i);
			return 1;
		}
		(*rowPtr2)[i + 1] = (*rowPtr2)[i] + lnAdd;
	}
  if (*nnz2 != nAdd){
			fprintf(stderr, "Error during diagonal elimination\t Malloc of %d elements not equal to %d values added\n", *nnz2, nAdd);
			return 1;
  }
	return ierr;
}

int GetBlockPosition(int size_parts, idx_t *parts, int npart, int** pos){
	int* tmp	= NULL;
  int sum	= 0;
  int ierr  = 0;
	tmp = (int*) calloc(npart + 1, sizeof(int));
	for(unsigned int i = 0; i < size_parts; i++){
		tmp[parts[i]]++;
  }
  //	Allocation with one more cell to store the last interval
	for(unsigned int i = 0; i < npart + 1; i++) (*pos)[i] = 0;
	for(unsigned int i = 0; i < npart + 1; i++){
		(*pos)[i]	=   sum;
		sum			+=  tmp[i];
	}
	free(tmp);
	return ierr;
}

int GetIntPermArray(idx_t npart, idx_t size_parts, idx_t *parts, int** perm){
  int ierr = 0;
	int	current_row = 0;
	*perm = (int*) malloc( (int)size_parts * sizeof(int));
	//*perm = (int*) malloc( (int)size_parts * sizeof(int));
	//	[caution], if int is 64 loop has to be changed
	for(unsigned int i = 0; i < npart; i++){
		for(unsigned int j = 0; j < size_parts; j++){
			if(parts[j] == i)
				(*perm)[current_row++] = j;
    }
  }
	return ierr;
}

int PermuteCSR(int M, int nnz, int* rowPtr_in, int*  colInd_in, double* val_in, int* rowPtr_out, int* colInd_out, double* val_out, int* perm){
	int lnval = 0;
	int* tmp_s = NULL;
	int* iColPerm = NULL;
	tmp_s = perm;
	iColPerm = (int*) malloc( M * sizeof(int));
	for(unsigned int i = 0; i < M; i++){
		iColPerm[tmp_s[i]] = i;
  }
	rowPtr_out[0] = 0;
  //Copy data where rows are permuted too
  for(unsigned int i = 0; i < M; i++){
    lnval = rowPtr_in[ perm[i] + 1] - rowPtr_in[ perm[i] ];
    memcpy( colInd_out + rowPtr_out[i], colInd_in + rowPtr_in[perm[i]], lnval * sizeof(int));
  	rowPtr_out[i + 1] = rowPtr_out[i] + lnval;
  }
  for(int i = 0; i < M; i++){
    lnval = rowPtr_in[ perm[i] + 1] - rowPtr_in[ perm[i] ];
    memcpy( val_out + rowPtr_out[i], val_in + rowPtr_in[perm[i]], lnval * sizeof(double));
  }
  //Permute columns
  for(unsigned int i = 0; i < nnz; i++)
  	colInd_out[i] = iColPerm[colInd_out[i]];
  //Sort columns
	for(int i = 0; i < M; i++)
		quickSortWithValues(colInd_out, rowPtr_out[i], rowPtr_out[i + 1] - 1, val_out);
	free(iColPerm);
	return 0;
}

int PermuteVec(double* v_in, double* v_out, int* perm, int m){
	for(unsigned int i = 0; i < m; i++)
		v_out[perm[i]] = v_in[i];
	return 0;
}

int GetRowPanel(int M_in, int nnz_in, int* rowPtr_in, int* colInd_in, double* val_in, int *M_out, int* nnz_out, int** rowPtr_out, int** colInd_out, double** val_out, int* pos, int numBlock){
  int ierr    = 0;
  int lm      = 0;
  int lnnz    = 0;
	//	variable which adapts the values in rowPtr for local colInd
	int offset  = 0;

  lm      = pos[numBlock + 1] - pos[numBlock];
  lnnz    = rowPtr_in[pos[numBlock + 1]] - rowPtr_in[pos[numBlock]];
	offset  = rowPtr_in[pos[numBlock]];
	*M_out = lm;
	*nnz_out = lnnz;

  if(*val_out == NULL){
		*rowPtr_out = (int*) mkl_malloc((*M_out + 1)		* sizeof(int), 32);
		*colInd_out = (int*) mkl_malloc(*nnz_out				* sizeof(int), 32);
		*val_out		= (double*)				mkl_malloc(*nnz_out				* sizeof(double),				64);
		//*rowPtr_out = (int*) malloc((*M_out + 1)		* sizeof(int));
		//*colInd_out = (int*) malloc(*nnz_out				* sizeof(int));
		//*val_out		= (double*)				malloc(*nnz_out				* sizeof(double)			);
  }

  memcpy(*colInd_out, colInd_in + offset,					lnnz			* sizeof(int));
  memcpy(*val_out,    val_in + offset,						lnnz			* sizeof(double));
  memcpy(*rowPtr_out, rowPtr_in + pos[numBlock],	(lm + 1)  * sizeof(int));

  for(unsigned int i = 0; i < *M_out + 1; i++){
    (*rowPtr_out)[i]  -=  offset;
  }
	return ierr;
}


int GetColBlockPos(int M_in, int nnz_in, int* rowPtr_in, int* colInd_in, double* val_in, int* pos, int* colPos, int nblock){
  int numBlock    = 0;//By default, we consider the first block
  int newNumBlock = 0;
  int c           = 0;
  colPos[0]				= 0;
  colPos[nblock]	= rowPtr_in[M_in];
  for(unsigned int i = 0; i < M_in; i++){
    numBlock = 0;
    for(unsigned int j = rowPtr_in[i]; j < rowPtr_in[i + 1]; j++){
      c = colInd_in[j];
      while(c >= pos[numBlock + 1]){
        numBlock++;
        colPos[i * nblock + numBlock] = j;
      }
    }
    for(unsigned int k = numBlock + 1; k <= nblock; k++){
      colPos[i * nblock + k] = rowPtr_in[i + 1];
    }
  }
  return 0;
}

int GetCommDep(int* colPos, int M, int nblock, int rank, int** dep){
  int cpt   = 0;
	int* tmp = (int*) malloc( nblock * sizeof(int));
	for(unsigned int i = 0; i < nblock; i++) tmp[i] = 0;

  for(unsigned int i = 0; i < M; i++){
    for(unsigned int j = 0; j < nblock; j++){
      tmp[j] += colPos[i * nblock + j + 1] - colPos[i * nblock + j];
    }
  }

  for(unsigned int i = 0; i < nblock; i++){
    if(tmp[i] && i != rank)
      (*dep)[cpt++] = i;
	}

	//*dep = (int*) realloc(*dep, cpt * sizeof(int));
  if(cpt == 0){
    fprintf(stderr, "There is no dependencies between some blocks of A.\nProc %d is connected to no other proc", rank);
    MPI_Abort(MPI_COMM_WORLD, -1);
	}
	free(tmp);
  return cpt;
}


/**
 * \fn
 * \brief Function extracts the diagonal block matrix of a row panel using pos and colPos given after applying Metis_GraphPartKway
 */
int GetDiagonalBlock(int M, int nnz, int* rowPtr, int* colInd, double* val, int* nnz_out, int* rowPtr_out, int* colInd_out, double* val_out, int* pos, int* colPos){
	int offset    = 0;	//	variable which adapts the values in rowPtr for local colInd
  int rank			= 0;
  int size			= 0;
  int ind				= 0;
  int ptr   		= 0;
  int nvAdd 		= 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  //Count how many lnnz there will be
  int sum      = 0;
	int* rowSize = NULL;
	rowSize = (int*) malloc(M * sizeof(int));
  for(unsigned int i = 0; i < M; i++){
    rowSize[i]  = colPos[i * size + rank + 1] - colPos[i * size + rank];
    sum         += rowSize[i];
		if(rowSize[i] > M){
			printf("number of nnz in line larger than the matrix dimension rowSize = %d, M = %d!!!\n ", rowSize[i], M);
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Abort(MPI_COMM_WORLD, -1);
		}
  }
	*nnz_out = sum;
  rowPtr_out[0] = 0;
  for(unsigned int i = 0; i < M; i++){
    ptr   = colPos[i * size + rank];
    nvAdd = rowSize[i];
		if( ind > sum){
			printf("ind depasses max index with value ind = %d\n", ind);
			MPI_Abort(MPI_COMM_WORLD, -1);
		}
		if(ptr + nvAdd > nnz){
			printf("ptr + nvAdd depasses max index with value ptr = %d, nvAdd = %d\n", ptr, nvAdd);
			MPI_Abort(MPI_COMM_WORLD, -1);
		}
    memcpy(&(colInd_out[ind]),	&(colInd[ptr]),	nvAdd * sizeof(int));
		if(ptr + nvAdd > nnz){
			printf("ptr + nvAdd depasses max index with value ptr = %d, nvAdd = %d\n", ptr, nvAdd);
			MPI_Abort(MPI_COMM_WORLD, -1);
		}
    memcpy(&(val_out[ind])   ,	&(val[ptr]),		nvAdd * sizeof(double));
    rowPtr_out[i + 1]	=   rowPtr_out[i] + nvAdd;
    ind								+=  nvAdd;
  }
	MPI_Barrier(MPI_COMM_WORLD);
  offset = pos[rank];
  for(unsigned int i = 0; i < *nnz_out; i++)
    colInd_out[i] -= offset;
	free(rowSize);
	return 0;
}

int TSQR(MPI_Comm comm, int m, int n, double* A, double* Q, double* R, double* work){
	int ierr = 0;
	int	size		= -1;
	int	rank		= -1;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	if(m < size * n){
		printf("Number of local lines for TSQR is smaller than number of columns * number of procs.\nThis case is not treated in this subroutine\n");
		MPI_Abort(comm, -99);
	}
	double*				Q_s			= NULL;
	double*				Atau_s	= NULL;
	double*				Qtau_s	= NULL;
	double*				OQ_s		=	NULL;
	//	Symbolic pointer to tau of QR factorization of local matrix
	Atau_s = work;
	//	Symbolic pointer to the place to assemble the R matrices by Allgather
	Q_s = work + n * n;
	//	Symbolic pointer to tau of QR foctorization of the assembled matrix by Allgather
	Qtau_s = Q_s + n * n * size;
	//	Symbolic pointer to the reordered assembled matrix
	OQ_s = Qtau_s + n * n;

	//	QR factorization of local matrix
	ierr = LAPACKE_dgeqrt(LAPACK_COL_MAJOR, m, n, n, A, m, work, n);
	//	Set R to zeros
	memset(R, 0, n * n * sizeof(double));
	//	Copy the upper triangular of the factorized matrix
	ierr = LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', n, n, A, m, R, n);
	//	Assemble the matrix of concatenated R 
	ierr = MPI_Allgather(R, n * n, MPI_DOUBLE, Q_s, n * n, MPI_DOUBLE, comm);
	//	Reorder in COL MAJOR
	for(unsigned int i = 0; i < n; i++){
		for(unsigned int j = 0; j < size; j++){
			memcpy(OQ_s + i * n * size + j * n, Q_s + i * n + j * n * n, n * sizeof(double));
		}
	}
	//	QR factorization of the assembled matrix
	ierr = LAPACKE_dgeqrt(LAPACK_COL_MAJOR, n * size, n, n, OQ_s, n * size, Qtau_s, n);
	//	Copy the result R into the output parameter R
	ierr = LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', n, n, OQ_s, n * size, R, n);
	//	Set Q to 0
	memset(Q, 0, m * n * sizeof(double));
	//	Identity matrix in the first block of Q
	for(unsigned int i = 0; i < n; i++){
		Q[i + m * i] = 1.;
	}
	//	Apply the Q factor of the assembled matrix on Q
	ierr = LAPACKE_dgemqrt(LAPACK_COL_MAJOR, 'L', 'N', n * size, n, n, n, OQ_s, n * size, Qtau_s, n, Q, m);
	if(!ierr){
		printf("DGEMQRT error\n");
		MPI_Abort(comm, ierr);
	}
	//	Put the block corresponding to my rank on the top of the result matrix
	ierr = LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', n, n, Q + rank * n, m, Q, m);
	//	Set under first block to 0
	for(unsigned int i = 0; i < n; i++){
		memset(Q + n + i * m, 0, (m - n) * sizeof(double));
	}
	//	Apply the Q factor of the local matrix on Q to have the local factor Q of the matrix A
	ierr = LAPACKE_dgemqrt(LAPACK_COL_MAJOR, 'L', 'N', m, n, n, n, A, m, Atau_s, n, Q, m);
	if(!ierr){
		printf("DGEMQRT error\n");
		MPI_Abort(comm, ierr);
	}
	return ierr;
}

int ReadIVecFromFile(char* filename, int m, int* v){
	int ierr = 0, nread;
  FILE *fd;
	fd = fopen(filename, "r");
	if(!fd)
	{
		printf("IVec file can not be opened\n");
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
	for(unsigned int i = 0; i < m; i++){
		nread = fscanf(fd, "%d", v + i);
	}
	fclose(fd);
	return ierr;

}

int ReadDVecFromFile(char* filename, int m, double* v){
	int ierr = 0, nread;
  FILE *fd;
	fd = fopen(filename, "r");
	if(!fd)
	{
		printf("IVec file can not be opened\n");
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
	for(unsigned int i = 0; i < m; i++){
		nread = fscanf(fd, "%lf", v + i);
	}
	fclose(fd);
	return ierr;

}

int GetBlock(int m, int n, int nnz, int* rowPtr, int* colInd, double* val, int n_out, int* lnnz_out, int* rowPtr_out, int* colInd_out, double* val_out, int* posB, int* colPos, int BlockNum, int size, int* work){
  int nblock  = size;
  int sum     = 0;
  int ind     = 0;  //
  int ptr     = 0;  //Position of the first column of the block
  int ptrC    = 0;  //Pointer to the first column of the block
  int lnvAdd  = 0;  //Number of values on the current line to add
  int offset  = 0;  //Offset applied to the colInd array to shift the column 
                    // values such that the first column of the block in A 
                    // is numbered as 0 in B
  //	Count how many lnnz there will be
  for(int i = 0; i < m; i++){
    ptr			= i * nblock + BlockNum;
    work[i]	= colPos[ptr + 1] - colPos[ptr];
    sum			+= work[i];
  }
  *lnnz_out = sum;
  if(!sum){
    return 0;
  }
  /* ====================*
  *    copy of arrays    *
  * =====================*/
  rowPtr_out[0] = rowPtr[0];
  for(int i = 0; i < m; i++){
    ptrC   = colPos[i * nblock + BlockNum];
    lnvAdd = work[i];
    memcpy(colInd_out + ind,	colInd	+ ptrC, lnvAdd * sizeof(int));
    memcpy(val_out + ind,			val			+ ptrC, lnvAdd * sizeof(double));
    rowPtr_out[i + 1]  =   rowPtr_out[i] + lnvAdd;
    ind                +=  lnvAdd;
  }
  offset = posB[BlockNum];
  for(int i = 0; i < *lnnz_out; i++){
    colInd_out[i] -= offset;
  }
	return 0;
}
int flops_dormqr(int m, int n, int k){
	return m * n * k;
}
int flops_dgemm_sum(int m, int n, int k){
	return m * n * k + m * k;
}
int flops_dgemm(int m, int n, int k){
	return m * n * k;
}
int flops_dgeqrf(int m, int n){
	return (2 * n * n * (3 * m - n))/3;
}
int flops_dpotrf(int n){
	return (4 * n * n * n)/3;
}
int flops_svd(int m, int n){
	return 12 * m * n * n;
}
int flops_dgetrf(int m){
	return m * m * m;
}
int flops_dtrsm(int m, int n){
	return m * m * n;
}
int flops_spmm(int nnz, int n){
	return nnz * n;
}
int flops_memset(int n){
	return n;
}
int flops_allreduce(int n, int ND){
	return n * log(ND);
}
int flops_copy(int m, int n){
	return m * n;
}
int flops_sum(int m, int n){
	return m * n;
}
int flops_zunmqr(int m, int n, int k){
	return 6 * m * n * k;
}
int flops_zgemm_sum(int m, int n, int k){
	return 6 * m * n * k + 2 * m * k;
}
int flops_zgemm(int m, int n, int k){
	return 6 * m * n * k;
}
int flops_zgeqrf(int m, int n){
	return (8  * n * n * (3 * m - n))/3;
}
int flops_zpotrf(int n){
	return (16 * n * n * n)/3;
}
int flops_zgesvd(int m, int n){
	return 48 * m * n * n;
}
int flops_zgetrf(int m){
	return 4 * m * m * m;
}
int flops_ztrsm(int m, int n){
	return 6 * m * m * n;
}
int flops_spmm_z(int nnz, int n){
	return nnz * n * 6;
}
int flops_memset_z(int n){
	return 2 * n;
}
int flops_allreduce_z(int n, int ND){
	return 2 * n * log(ND);
}
int flops_copy_z(int m, int n){
	return m * n * 2;
}
int flops_sum_z(int m, int n){
	return m * n * 2;
}
int flops_zgebal(int m){
	return 4 * m * m;
}
int flops_zgehrd(int n, int ilo, int ihi){
	return (8 * (ihi - ilo) * (ihi - ilo) * (2 * (ihi + ilo) + 3 * n))/3;
}
int flops_zunghr(int ilo, int ihi){
	return (16 * (ihi - ilo) * (ihi - ilo) * (ihi - ilo) )/3;
}
int flops_zungqr(int m, int n){
	return (16 * m * n * n )/3;
}
int flops_zhseqr(int n){
	return 70 * n * n * n;
}
int flops_ztrexc(int n, int ifst, int ilst){
	return 40 * n * (ifst - ilst);
}
int flops_zgebak(int m, int n){
	return m * n;
}
int flops_zggbal(int n){
	return 8 * n * n;
}
int flops_zgghd3(int n){
	return 14 * 4 * n * n * n;
}
int flops_zhgeqz(int n){
	return 46 * 4 * n * n * n;
}
int flops_ztgexc(int n, int ifst, int ilst){
	return 80 * n * (ifst - ilst);
}
int flops_zggbak(int m, int n){
	return m * n;
}
