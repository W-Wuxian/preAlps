/******************************************************************************/
/* Authors    : Hussam Al-Daas                                                */
/* Creation   : 16/02/2018                                                    */
/* Description: Block Modified GCRODR	                                      */
/******************************************************************************/

/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/

#ifndef BGMRESMDR_H
#define BGMRESMDR_H

/*	MPI			*/
#include <mpi.h>
/*	MKL			*/
//#include <mkl.h>
/*	BGMRESMDR	        */
//#define ILU0
//#define LUDS
/******************************************************************************/
// iparam length 16
#define NRHS								       0
#define MAXITOUT							       1
#define MAXBASIS							       2
#define ORTHOSTRATEGY						               3
#define	INEXACTBREAKDOWN				                       4
#define	PRECONDITIONING					                       5
#define	GDOF								       6
#define	LDOF								       7
#define	MAXDEF								       8
#define	DEFLATION							       9
#define	ORTHOGONALIZATION				                      10
#define ACTUALDEFDIM						              11
#define NUMBEROFDOMAINS					                      12
#define	DEFSTRATEGY   					                      13
#define	NUMBEROFSYS							      14
#define	ISNEWSYSTEM							      15
// dparam length 4
#define	CONVERGENCETOL					                       0
#define EVTHRESHOLD							       1
#define	REDUCTIONOFCSTHRESHOLD	                                               2
#define	ACTUALRESNORM						               3
// debug printf info if 1
#define BGMRESMDRDEBUG 0
// data type
#define MYTYPE MKL_Complex16

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/
/* Preconditioner */
typedef enum {
  Prec,
  NoPrec
} Prec_t;
/* Deflation */
typedef enum {
  RITZ,
  HRITZ,
  SVD,
  NoDef
} Def_t;
/* Inexact breakdown type */
typedef enum {
  NoRed,
  SVDR,
  RRQR
} Red_t;
/* Orthogonalization type */
typedef enum {
  BCGS,
  DBCGS
} Ortho_t;
/* Timing structure */
typedef struct{
	double TotalTime;
	double PreconditioningTime;
	double HessenbergOperationsTime;
	double DeflationComputationTime;
	double DeflationComputationComm;
	double DeflationApplicationTime;
	double SPMMTime;
	double SPMMDiag;
	double SPMMOffDiag;
	double SPMMComm;
	double OrthogonalizationTime;
	double OrthogonalizationComm;
	double InexactBreakDownTime;
	double Recovering;
}	Timing_t;

typedef struct{
  /**	Basis									*/
  MYTYPE*     V; /**	Basis of Enlarged Krylov SS				*/
  int*	   VIdx; /**	Dimension of V					        */

  /**	Deflation 								*/
  MYTYPE*     Z; /**	Deflation basis	                                        */
  MYTYPE*    AZ; /**	A * Deflation basis                                     */
  MYTYPE* WorkZ; /**	Work space for Deflation basis				*/
  int	   dimZ; /**	Deflation dimension                                     */
  int MaxDefDim; /**	Maximal dimension of deflation subspace	                */

  /**	Hessenberg                  */
  MYTYPE*								H;					/**	Hessenberg Matrix	*/
  MYTYPE*								Htau;					/**	Hessenberg Householders coeff	*/
  MYTYPE*								H2;					/**	Hessenberg Matrix	*/
  MYTYPE*								H3;					/**	Hessenberg Matrix	*/
  MYTYPE*								GtG;					/**	Matrix A in generalized EVP		  */
  MYTYPE*								VtV;					/**	Matrix B in the generalized EVP	          */
  MYTYPE*								VtW;					/**	Factor of matrix A in the generalized EVP */
  MYTYPE*								ZtZ;					/**	Part of matrix B in the generalized EVP	  */
  MYTYPE*								D;					/**	Diagonal matrix in the Hessenberg */

	/**	Breakdown	*/
  MYTYPE*								Q;					/**	Breakdowns Roations	*/
  MYTYPE*								Qtau;					/**	Breakdowns Roations Householders coeff */
  int*									BIdx;				/**	Breakdown Indices           */
  int*									BSize;				/**	Block size over iterations  */
  int									IBreakdown;		/**	Number of inexact breakdown */
  int									ActualSize;		/**	Number of inexact breakdown */

	/**	Shared variables */
  MYTYPE*								KRHS;					/**	Kylov RHS */
  MYTYPE*								KSOL;					/**	Kylov RHS */
  MYTYPE*								rvec;					/**	Vector containing residual norm of iterations */

	/**	System vectors   */
  MYTYPE*								b;						/**	right hand side	*/
  MYTYPE*								Solution;			/**	Solution to be returned	*/
  MYTYPE*								Residual;			/**	Residual to be returned	*/
  int									ln;				/**	Local number of degrees of freedom */
  int										ldv;					/**	Local leading dimension of v																														*/
  int										nrhs;					/** Number of right hand sides	*/

	/**	Parameters for the solver				 */
  int										MaxBasis;			/**	Max dimension of enlarged Krylov subspace */
  int										ND;						/**	Number of domains	  */
  int										s;						/**	Number of vectors on which the user applies his operator or preconditioner */
  int										iteration;		/**	Actual number of interior iteration */
  int                   Cycle;				/**	Actual number of exterior iteration */
  int	        	MaxCycle;			/**	Maximum number of outer iterations  */
  int                   GIter;				/**	Static (over restarts) iteration    */
  double                MaxNormb;			/**	Max Norm of the rhs's */
  double*               normb;				/**	norm of the rhs	*/
  double								Ctol;					/**	Convergence threshold */
  Ortho_t								Ortho;				/**	Orthogonalization strategy */
  Prec_t								Prec;					/**	Flag for using a preconditioner	*/
  Def_t									Def;					/**	Flag for using deflation of eigenvalues	*/
  Red_t									Red;					/**	Flag for using deflation of eigenvalues */
  MPI_Comm							        comm;					/**	Communicator */
  int									rank;					/**	Rank of proc */

	/**	Arrays 	*/
  MYTYPE*								WorkV;				/**	Work space for basis vectors */
  MYTYPE*								WorkH;				/**	Work space for Enlarged Krylov subspace	*/

  MYTYPE*								U;						/**	Pointer to the matrix on which we apply the operator */
  MYTYPE*								AU;						/**	Pointer to the matrix that results from applying the operator on U (previous parameter)	*/
  unsigned long long 		flops;				/**	Number of flops	*/
  Timing_t							Timing;				/**	Structure for timings */
}BGMRESMDR_t;

int BGMRESMDRSetUp(MPI_Comm comm, MYTYPE* b, int* iparam, double* dparam, MYTYPE* v, int ldv, MYTYPE* workd, int* worki, BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRInexactBreakdownDetection(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRInexactBreakdownReduction(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRResSolInit(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRSetResidual(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRPrepareArnoldi(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRRCI(MPI_Comm comm, int* ido, MYTYPE* x, MYTYPE* b, int* iparam, double* dparam, MYTYPE* v, int ldv, MYTYPE* workd, int* worki, BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRDeflation(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDROrthogonalization(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRUpdateH(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRUpdateKRHS(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRRITZDeflation(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRRITZDeflationSimple(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRHRITZDeflation(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRHRITZDeflationSimple(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRSVDDeflation(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRSVDDeflationSimple(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRSVDDeflationSEV(BGMRESMDR_t* BGMRESMDR);
//int BGMRESMDRSVDDeflationTest(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRSetHessenbergDefPart(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRCriterionTest(BGMRESMDR_t* BGMRESMDR, int* flag);
int BGMRESMDRRecoverSolution(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRRecoverSolutionInX(BGMRESMDR_t* BGMRESMDR, MYTYPE* x);
int BGMRESMDRRecoverRealResidual(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRRecoverNormRealResidual(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRPrepareDeflationMatrix(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRReduceDeflationSS(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRRestart(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRTest(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRNormalizeD(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRUnNormalizeVecZ(BGMRESMDR_t* BGMRESMDR);
int BGMRESMDRDump(BGMRESMDR_t* BGMRESMDR, char* MatName);
int lap2dvaw(MPI_Comm comm, MYTYPE* w, MYTYPE* v, int n, int m);
/******************************************************************************/

#endif
