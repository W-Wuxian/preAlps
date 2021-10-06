/******************************************************************************/
/* Authors    : Hussam Al-Daas                                                */
/* Creation   : 16/02/2018                                                    */
/* Description: Block GMRES-MDR                                               */
/******************************************************************************/

/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mkl.h>
#include <math.h>
#include <time.h>
#include <omp.h>

/* BGMRESMDR */
#include "BGMRESMDR.h"
#include "utilities.h"

/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/

int BGMRESMDRSetUp(MPI_Comm comm, MYTYPE* b, int* iparam, double* dparam, MYTYPE* v, int ldv, MYTYPE* workd, int* worki, BGMRESMDR_t* BGMRESMDR){
	int myrank = -1;
	MPI_Comm_rank(comm, &myrank);
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!myrank) printf("Set Up\n");
	MPI_Barrier(comm);
#endif
	int ierr = 0;
	/*	Set up preconditioner flag					*/
	if(iparam[PRECONDITIONING] == 1){
		BGMRESMDR->Prec	= Prec;
	}else if(iparam[PRECONDITIONING] == 0){
		BGMRESMDR->Prec	= NoPrec;
	}else{
		fprintf(stderr, "Flag of Preconditioner has an unknown value  = %d \n", iparam[PRECONDITIONING]);
		MPI_Abort(comm, -1);
	}
	/*	Set up Inexact breakdown flag					*/
	if(iparam[INEXACTBREAKDOWN] == 0){
		BGMRESMDR->Red	= NoRed;
	}else if(iparam[INEXACTBREAKDOWN] == 1){
		BGMRESMDR->Red	= SVDR;
	}else if(iparam[INEXACTBREAKDOWN] == 2){
		BGMRESMDR->Red	= RRQR;
	}else{
		fprintf(stderr, "Flag of Inexact Breakdown has an unknown value  = %d \n", iparam[INEXACTBREAKDOWN]);
		MPI_Abort(comm, -1);
	}

	/*	Set up Deflation flag						 */
	if(iparam[DEFLATION] == 0){
		BGMRESMDR->Def       = NoDef;
		BGMRESMDR->MaxDefDim = 0;
		BGMRESMDR->dimZ	     = 0;
	}else if(iparam[DEFLATION] == 1){
		BGMRESMDR->MaxDefDim = iparam[MAXDEF];
		BGMRESMDR->Def       = SVD;
	}else if(iparam[DEFLATION] == 2){
		BGMRESMDR->MaxDefDim = iparam[MAXDEF];
		BGMRESMDR->Def       = RITZ;
	}else if(iparam[DEFLATION] == 3){
		BGMRESMDR->MaxDefDim = iparam[MAXDEF];
		BGMRESMDR->Def       = HRITZ;
	}else{
		fprintf(stderr, "Flag of Deflation has an unknown value  = %d \n", iparam[DEFLATION]);
		MPI_Abort(comm, -1);
	}

	/*	Set up orthogonalization strategy			          */
	if(iparam[ORTHOGONALIZATION] == 0){
		BGMRESMDR->Ortho = BCGS;
	}else if(iparam[ORTHOGONALIZATION] == 1){
		BGMRESMDR->Ortho = DBCGS;
	}else{
		fprintf(stderr, "Flag of Orthogonalization strategy has an unknown value  = %d \n", iparam[ORTHOGONALIZATION]);
		MPI_Abort(comm, -1);
	}
	/*	Set up Other integer params					   */
	MPI_Comm_rank(comm,		&BGMRESMDR->rank);
	BGMRESMDR->ln	      = iparam[LDOF];
	BGMRESMDR->ldv	      = ldv;
	BGMRESMDR->ND	      = iparam[NUMBEROFDOMAINS];
	BGMRESMDR->MaxBasis   = iparam[MAXBASIS];
	BGMRESMDR->dimZ	      = iparam[ACTUALDEFDIM];
	BGMRESMDR->iteration  = 0;
	BGMRESMDR->GIter      = 0;
	BGMRESMDR->Cycle      = 1;
	BGMRESMDR->MaxCycle   = iparam[MAXITOUT];
	BGMRESMDR->comm	      = comm;
	BGMRESMDR->nrhs	      = iparam[NRHS];
	BGMRESMDR->IBreakdown = 0;

	/*	Set up double params						    */
	BGMRESMDR->Ctol	    = dparam[CONVERGENCETOL];
	BGMRESMDR->b	    = b;
	BGMRESMDR->MaxNormb = 0.0;

	//	Do not forget to desallocate
	BGMRESMDR->normb = NULL;
	BGMRESMDR->normb = (double*) malloc(BGMRESMDR->nrhs * sizeof(double));
	if(BGMRESMDR->normb == NULL){
		fprintf(stderr, "normb is not allocated\n");
	}

	for(int i = 0; i < BGMRESMDR->nrhs; i++){
		BGMRESMDR->normb[i] = cblas_dznrm2 (iparam[LDOF], b + BGMRESMDR->ldv * i, 1);
		BGMRESMDR->normb[i] = BGMRESMDR->normb[i] * BGMRESMDR->normb[i];
	}
	ierr = MPI_Allreduce(MPI_IN_PLACE, BGMRESMDR->normb, BGMRESMDR->nrhs, MPI_DOUBLE, MPI_SUM, comm);
	for(int i = 0; i < BGMRESMDR->nrhs; i++){
		BGMRESMDR->normb[i] = sqrt(BGMRESMDR->normb[i]);
		BGMRESMDR->MaxNormb = (BGMRESMDR->MaxNormb < BGMRESMDR->normb[i]) ? BGMRESMDR->normb[i] : BGMRESMDR->MaxNormb;
	}
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!BGMRESMDR->rank) printf("Number of rhs = %d \n", BGMRESMDR->nrhs);
	for(int i = 0; i < BGMRESMDR->nrhs; i++){
		if(!BGMRESMDR->rank) printf("Norm of b[%d] = %.8e \n", i, BGMRESMDR->normb[i]);
	}
	MPI_Barrier(comm);
#endif

	/*	Check if the user supplied v and set up local and global matrices		       */
	if(v != NULL){
		if(!workd){
			fprintf(stderr, "v is supplied but not workd\n");
			MPI_Abort(comm, -1);
		}
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!BGMRESMDR->rank) printf("BGMRESMDR->MaxDefDim = %d \n", BGMRESMDR->MaxDefDim);
	if(!BGMRESMDR->rank) printf("ldv = %d \n ln = %d\n", ldv, BGMRESMDR->ln);
	MPI_Barrier(comm);
#endif
		int offset = 0;
		/*	Set up Z, AZ, V, WorkZ								*/
		BGMRESMDR->Z     = v + offset;
		offset          += ldv * BGMRESMDR->MaxDefDim;
		BGMRESMDR->AZ    = v + offset;
		offset          += ldv * BGMRESMDR->MaxDefDim;
		BGMRESMDR->V     = v + offset;
		offset          += ldv * iparam[MAXBASIS];
		BGMRESMDR->WorkZ = v + offset;
		offset          += ldv * BGMRESMDR->MaxDefDim;

		/*	Set up Work, Solution and Residual					       */
		BGMRESMDR->Residual = v + offset;
		offset             += ldv * BGMRESMDR->nrhs;
		BGMRESMDR->Solution = v + offset;
		offset		   += ldv * BGMRESMDR->nrhs;
		BGMRESMDR->WorkV    = v + offset;
		offset		   += 2 * ldv * BGMRESMDR->nrhs;
		/*	Set up T, H, H2, Htau, Q, Qtau, EKRHS, EKSolution and KRHS		       */

		int sum = (iparam[MAXBASIS] + iparam[MAXDEF]);
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!BGMRESMDR->rank) printf("sum = %d \n", sum);
	MPI_Barrier(comm);
#endif
		offset           = 0;
		BGMRESMDR->H	 = workd + offset;
		offset		+= sum * ( sum - BGMRESMDR->nrhs);
		BGMRESMDR->H2	 = workd + offset;
		offset		+= sum * ( sum - BGMRESMDR->nrhs);
		BGMRESMDR->H3	 = workd + offset;
		offset		+= sum * ( sum - BGMRESMDR->nrhs);
		BGMRESMDR->Htau	 = workd + offset;
		offset		+= BGMRESMDR->MaxBasis * BGMRESMDR->nrhs;
		BGMRESMDR->WorkH = workd + offset;
		offset		+= sum * (sum + 3);
		BGMRESMDR->GtG	 = workd + offset;
		offset		+= sum * sum;
		BGMRESMDR->VtV	 = workd + offset;
		offset		+= sum * sum;
		BGMRESMDR->VtW	 = workd + offset;
		offset		+= sum * sum;
		BGMRESMDR->ZtZ	 = workd + offset;
		offset		+= BGMRESMDR->MaxDefDim * BGMRESMDR->MaxDefDim;
		BGMRESMDR->D	 = workd + offset;
		offset		+= BGMRESMDR->MaxDefDim;
		BGMRESMDR->KSOL	 = workd + offset;
		offset		+= sum * BGMRESMDR->nrhs;
		BGMRESMDR->KRHS	 = workd + offset;
		offset		+= sum * BGMRESMDR->nrhs;
		BGMRESMDR->Q	 = workd + offset;
		offset		+= iparam[MAXBASIS] * BGMRESMDR->nrhs * BGMRESMDR->nrhs;
		BGMRESMDR->Qtau	 = workd + offset;
		offset		+= BGMRESMDR->nrhs * iparam[MAXBASIS];
		BGMRESMDR->rvec	 = workd + offset;
		//	Set Solution to 0
		memset(BGMRESMDR->Solution, 0, BGMRESMDR->ln * BGMRESMDR->nrhs * sizeof(MYTYPE));

		/*	Set up VIdx, TLUP							       */
		BGMRESMDR->VIdx	 = worki;
		BGMRESMDR->BIdx  = worki + iparam[MAXBASIS];
		BGMRESMDR->BSize = worki + iparam[MAXBASIS] * 2;
		memset(BGMRESMDR->BSize, 0, iparam[MAXBASIS] * iparam[MAXITOUT] * sizeof(int));
	}else{
		printf("Case without prior allocation not supported yet. Please preallocate memory for BGMRESMDR!");
		MPI_Abort(comm, -1);
	}
	//	Compute the Frobenius norm of the rhs
	BGMRESMDR->rvec[0].real = 0;
	BGMRESMDR->MaxNormb = 0;
	for(int i = 0; i < BGMRESMDR->nrhs; i++){
		BGMRESMDR->rvec[0].real += BGMRESMDR->normb[i] * BGMRESMDR->normb[i];
		BGMRESMDR->MaxNormb = (BGMRESMDR->MaxNormb < BGMRESMDR->normb[i]) ? BGMRESMDR->normb[i] : BGMRESMDR->MaxNormb;
	}
	BGMRESMDR->rvec[0].real = sqrt(BGMRESMDR->rvec[0].real);
	free(BGMRESMDR->normb);


	//	Reset timing to 0
	BGMRESMDR->Timing.TotalTime		   = 0;
	BGMRESMDR->Timing.SPMMTime		   = 0;
	BGMRESMDR->Timing.SPMMDiag		   = 0;
	BGMRESMDR->Timing.SPMMOffDiag		   = 0;
	BGMRESMDR->Timing.SPMMComm		   = 0;
	BGMRESMDR->Timing.PreconditioningTime	   = 0;
	BGMRESMDR->Timing.OrthogonalizationTime	   = 0;
	BGMRESMDR->Timing.OrthogonalizationComm	   = 0;
	BGMRESMDR->Timing.DeflationApplicationTime = 0;
	BGMRESMDR->Timing.DeflationComputationTime = 0;
	BGMRESMDR->Timing.DeflationComputationComm = 0;
	BGMRESMDR->Timing.HessenbergOperationsTime = 0;
	BGMRESMDR->Timing.Recovering		   = 0;
	BGMRESMDR->flops			   = 0;
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!myrank) printf("Set Up done\n");
	MPI_Barrier(comm);
#endif
	return ierr;
}

int BGMRESMDRResSolInit(BGMRESMDR_t* BGMRESMDR){
	int ierr	= 0;
	int t		= BGMRESMDR->nrhs;
	int dimZ	= BGMRESMDR->dimZ;
	int ln		= BGMRESMDR->ln;
	int ldkr	= BGMRESMDR->MaxBasis + BGMRESMDR->MaxDefDim;

	MYTYPE* AZ	= BGMRESMDR->AZ;
	MYTYPE* Z	= BGMRESMDR->Z;
	MYTYPE* R 	= BGMRESMDR->Residual;
	MYTYPE* b 	= BGMRESMDR->b;
	MYTYPE* x 	= BGMRESMDR->Solution;
	MYTYPE* KR	= BGMRESMDR->KRHS;

	MYTYPE const_alpha = {1., 0};
	MYTYPE const_beta  = {0, 0};

	memcpy(R, b, ln * t * sizeof(MYTYPE));
	BGMRESMDR->flops += flops_memset_z(ln * t);
	memset(KR, 0, ldkr * t * sizeof(MYTYPE));
	BGMRESMDR->flops += flops_memset_z(ldkr * t);

	if(dimZ > 0){
		int sum      = dimZ + BGMRESMDR->MaxBasis;
		MYTYPE* AZ   = BGMRESMDR->AZ;
		MYTYPE* work = BGMRESMDR->WorkH;

		//	Orthogonalize Residual against deflated vectors h_i = AZ' * R_i
		cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, dimZ, t, ln, &const_alpha, AZ, ln, R, ln, &const_beta, work, dimZ);
		BGMRESMDR->flops += flops_zgemm(dimZ, ln, t);

		//	h = Sum h_i
		ierr = MPI_Allreduce(MPI_IN_PLACE, work, dimZ * t, MPI_C_DOUBLE_COMPLEX, MPI_SUM, BGMRESMDR->comm);
		BGMRESMDR->flops += flops_allreduce_z(dimZ * t, BGMRESMDR->ND);

		//	R = R - AZ * h
		const_alpha.real = -1.;
		const_beta.real = 1.;
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ln, t, dimZ, &const_alpha, AZ, ln, work, dimZ, &const_beta, R, ln);
		BGMRESMDR->flops += flops_zgemm_sum(ln, dimZ, t);

		//	x = x + Z * (D^{-1} * h)
			// h_1 = (D^{-1} * h)
		for(int j = 0; j < t; j++){
			for(int i = 0; i < dimZ; i++){
				work[ (j * dimZ) + i].real = work[ (j * dimZ) + i].real / BGMRESMDR->D[i].real;
				work[ (j * dimZ) + i].imag = work[ (j * dimZ) + i].imag / BGMRESMDR->D[i].real;
			}
		}
		BGMRESMDR->flops += 2 * flops_dgemm(dimZ, dimZ, t);
			// x = x + Z * h_1
		const_alpha.real = 1.;
		const_beta.real = 0;
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ln, t, dimZ, &const_alpha, Z, ln, work, dimZ, &const_beta, x, ln);
		BGMRESMDR->flops += flops_zgemm_sum(BGMRESMDR->ln, dimZ, t);
	}
	return ierr;
}

/*	Preparation of Arnoldi iteration:
 *		Set V(1), H, H2, Htau, Q, Qtau.
 */
/** \fn int BGMRESMDRPrepareArnoldi(BGMRESMDR_t* BGMRESMDR){
 * \brief Prepares the Arnoldi procedure
 * \details Orthogonalizes the initial block residual
 * \param BGMRESMDR BGMRESMDR context
 * \remarks
 * \warning If the threshold of convergence is less than 1e-8 this function might induce error since it uses CholQR
*/
int BGMRESMDRPrepareArnoldi(BGMRESMDR_t* BGMRESMDR){
	int ierr	= 0;
	int t		= BGMRESMDR->nrhs;
	int cpt		= 0;
	int dimZ	= BGMRESMDR->dimZ;
	int ln		= BGMRESMDR->ln;
	int ND		= BGMRESMDR->ND;
	MYTYPE const_alpha = {1., 0};
	MYTYPE const_beta  = {0, 0};

	MYTYPE* temp = BGMRESMDR->WorkH;

	//	V is the first set of vectors in the Krylov basis
	MYTYPE* V    = BGMRESMDR->V;

	//	R is the block residual
	MYTYPE* R    = BGMRESMDR->Residual;

	MYTYPE* kr   = BGMRESMDR->KRHS;
	int ldkr     = BGMRESMDR->MaxBasis + dimZ;

	//	Copy R to V1
	memcpy(V , R, ln * t * sizeof(MYTYPE));
	BGMRESMDR->flops += flops_memset_z(ln * t);

	//	temp_i = V1_i' * V1_i
	cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, t, t, ln, &const_alpha, V, ln, V, ln, &const_beta, temp, t);
	BGMRESMDR->flops += flops_zgemm_sum(t, t, ln);

	//	temp = Sum temp_i
	ierr = MPI_Allreduce(MPI_IN_PLACE, temp, t * t, MPI_C_DOUBLE_COMPLEX, MPI_SUM, BGMRESMDR->comm);
	BGMRESMDR->flops += flops_allreduce_z(t * t, ND);

	//	temp = r' * r
	ierr = LAPACKE_zpotrf(LAPACK_COL_MAJOR, 'U', t, temp, t);
	if(ierr != 0){
		fprintf(stderr, "BGMRESMDRPrepareArnoldi::LAPACKE_zpotrf error %d\n", ierr);
		MPI_Abort(BGMRESMDR->comm, ierr);
	}
	BGMRESMDR->flops += flops_zpotrf(t);

	//	Normalize V1
	cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, ln, t, &const_alpha, temp, t, V, ln);
	BGMRESMDR->flops += flops_ztrsm(BGMRESMDR->ln, t);

	//	Set the Krylov residual to 0
	for(int i = 0; i < t; i++){
		memset(kr + i * ldkr, 0, ldkr * sizeof(MYTYPE));
	}
	BGMRESMDR->flops += flops_memset_z(ldkr * t);

	//	Set the Krylov residual to r
	ierr = LAPACKE_zlacpy (LAPACK_COL_MAJOR, 'U', t, t, temp, t, kr + dimZ, ldkr);
	BGMRESMDR->flops += flops_copy_z(t, t);

	return ierr;
}

/*	Reverse Communication interface for Enlarged Krylov GMRES
 *	User has to supply his own Matrix-matrix product and preconditioner.
 *
 *	Parameters:
 *		comm:		the communicator.
 *		x:			Vector of size LDOF that will contain the approximated solution.
 *		b:			Vector of size LDOF that contains the RHS.
 *		iparam:	Table of integers of size 64 containing integer parameters for the method.
 *				iparam[0]
 *		dparam:	Table of doubles of size 64 containing double parameters for the method.
 *		v:			(Optional) Allocated space for the global basis of the Enlarged subspace. At lease of size (LDOF * (MAXBASIS + 3 * EF)).
 *		ldv:		Not referenced if v is not supplied. Otherwise, the leading dimension of v, at least LDOF.
 *		workd:	Not referenced if v is not supplied. Otherwise allocated space for the local matrices. At least of size ().
 *		BGMRESMDR:	Structure for Enlarged GMRES method. Either the user initializes it or (v, ldv, workd are supplied). For more information see BGMRESMDR.h, EKrylov.h
 *		ido:	interface communicator.
 *				ido = 1		=>	The user has to apply his operator (including the preconditioner) on BGMRESMDR->U and	returns the result in BGMRESMDR->AU.
 *				ido = 3		=>	The user has to apply the preconditioner on BGMRESMDR->U and returns the result in BGMRESMDR->AU.
 */
/** \fn int BGMRESMDRRCI(MPI_Comm comm, int* ido, MYTYPE* x, MYTYPE* b, int* iparam, double* dparam, MYTYPE* v, int ldv, MYTYPE* workd, int* worki, BGMRESMDR_t* BGMRESMDR){
 * \brief Reverse Communication interface for Enlarged Krylov GMRES
 * \details User has to supply his own Matrix-matrix product and preconditioner.
 * \param comm Communicator between processes running BGMRESMDR
 * \param ido RCI indicator
 * \param x Pointer to the memory where to store the approximate solution
 * \param b Pointer to the right-hand side
 * \param iparam Integer parameters for BGMRESMDR
 * \param dparam Double precision parameters for BGMRESMDR
 * \param v pointer to a memory space for BGMRESMDR
 * \param ldv Leading dimension of v. For the moment it has to be equal to the size of the local problem
 * \param workd Pointer to work place for internal arrays of BGMRESMDR
 * \param worki Pointer to work place of internal indices of BGMRESMDR
 * \param BGMRESMDR BGMRESMDR context
 * \remarks
 * \warning
*/
int BGMRESMDRRCI(MPI_Comm comm, int* ido, MYTYPE* x, MYTYPE* b, int* iparam, double* dparam, MYTYPE* v, int ldv, MYTYPE* workd, int* worki, BGMRESMDR_t* BGMRESMDR){
	static int status = 0;
	int ierr = 0;
	int flag = 0;
	static double Timing = 0;
	/*	First Call: Set up BGMRESMDR */
	if(*ido == 0){
		ierr = BGMRESMDRSetUp( comm, b, iparam, dparam, v, ldv, workd, worki, BGMRESMDR);
		ierr = BGMRESMDRRestart(BGMRESMDR);
		if(BGMRESMDR->dimZ > 0 && iparam[ISNEWSYSTEM] == 1){
			iparam[ISNEWSYSTEM] = 0;
			ierr = BGMRESMDRUnNormalizeVecZ(BGMRESMDR);
			ierr = BGMRESMDRNormalizeD(BGMRESMDR);
			BGMRESMDR->U = BGMRESMDR->Z;
			BGMRESMDR->AU = BGMRESMDR->AZ;
			BGMRESMDR->s = BGMRESMDR->dimZ;
			status = -1;
			*ido = 1;
#if BGMRESMDRDEBUG
			MPI_Barrier(comm);
			if(!BGMRESMDR->rank){
				printf("Computing A * Z ...\n");
			}
			MPI_Barrier(comm);
#endif
			Timing = MPI_Wtime();
			return 0;
		}else{
			ierr = BGMRESMDRResSolInit(BGMRESMDR);
			ierr = BGMRESMDRSetHessenbergDefPart(BGMRESMDR);
			status = 1;
		}
	}
	/*	Stop RCI */
	if(*ido == 99)
		status = 99;
	switch(status){
		case -1:
#if BGMRESMDRDEBUG
			MPI_Barrier(comm);
			if(!BGMRESMDR->rank){
				printf("Preparing correction subspace matrix for new system ...\n");
			}
			MPI_Barrier(comm);
#endif
			ierr = BGMRESMDRPrepareDeflationMatrix(BGMRESMDR);
			ierr = BGMRESMDRSetHessenbergDefPart(BGMRESMDR);
#if BGMRESMDRDEBUG
			MPI_Barrier(comm);
			if(!BGMRESMDR->rank){
				printf("Preparing correction subspace matrix for new system done\n");
			}
			MPI_Barrier(comm);
#endif
			*ido = 40;
			status = 1;
			break;

	/*	Prepare Arnoldi cycle	*/
		case 1:
			ierr = BGMRESMDRPrepareArnoldi(BGMRESMDR);
			ierr = BGMRESMDRInexactBreakdownDetection(BGMRESMDR);
			ierr = BGMRESMDRInexactBreakdownReduction(BGMRESMDR);
			BGMRESMDR->s		= BGMRESMDR->ActualSize;
			/*	Set variables for OP. AU = OP * U				       */
			BGMRESMDR->U		= BGMRESMDR->V;
			BGMRESMDR->AU	= BGMRESMDR->V + BGMRESMDR->nrhs * BGMRESMDR->ldv;
			*ido = 1;
			status = 2;
			break;
		case 2:
			BGMRESMDR->GIter++;
			BGMRESMDR->iteration++;
			/*	Orthogonalize V(i+1) against [V(1) ... V(i)]	                       */
#if BGMRESMDRDEBUG
			MPI_Barrier(comm);
			if(BGMRESMDR->rank == 0)
				printf("Orthogonalization \n");
			MPI_Barrier(comm);
#endif
			ierr = BGMRESMDROrthogonalization(BGMRESMDR);

			/*	Update the QR factorization of H					*/
#if BGMRESMDRDEBUG
			MPI_Barrier(comm);
			if(BGMRESMDR->rank == 0)
				printf("Updating Hessenberg matrix \n");
			MPI_Barrier(comm);
#endif
			ierr = BGMRESMDRUpdateH(BGMRESMDR);

			/*	Update the enlarged Krylov residual					 */
#if BGMRESMDRDEBUG
			MPI_Barrier(comm);
			if(BGMRESMDR->rank == 0)
				printf("Updating the Krylov residual \n");
			MPI_Barrier(comm);
#endif
			ierr = BGMRESMDRUpdateKRHS(BGMRESMDR);

			/*	Test the convergence							  */
#if BGMRESMDRDEBUG
			MPI_Barrier(comm);
			if(BGMRESMDR->rank == 0)
				printf("Criterion test \n");
			MPI_Barrier(comm);
#endif
			ierr = BGMRESMDRCriterionTest(BGMRESMDR, &flag);

			if(flag == 1){
				ierr = BGMRESMDRInexactBreakdownDetection(BGMRESMDR);
				ierr = BGMRESMDRInexactBreakdownReduction(BGMRESMDR);
#if BGMRESMDRDEBUG
				MPI_Barrier(comm);
				if(BGMRESMDR->rank == 0) printf("Go to next ietration \n");
				MPI_Barrier(comm);
#endif
				BGMRESMDR->U		= BGMRESMDR->V + BGMRESMDR->ldv * BGMRESMDR->VIdx[BGMRESMDR->iteration];
				BGMRESMDR->AU		= BGMRESMDR->V + BGMRESMDR->ldv * (BGMRESMDR->VIdx[BGMRESMDR->iteration] + BGMRESMDR->nrhs);
				BGMRESMDR->s		= BGMRESMDR->ActualSize;
				*ido = 1;
				status = 2;
			}else{
				ierr = BGMRESMDRRecoverSolution(BGMRESMDR);
				ierr = BGMRESMDRDeflation(BGMRESMDR);
				iparam[ACTUALDEFDIM] = BGMRESMDR->dimZ;

				if(flag == 0 || flag == 3){
					if(BGMRESMDR->Prec == Prec){
						BGMRESMDR->U	= BGMRESMDR->Solution;
						BGMRESMDR->AU	= x;
						BGMRESMDR->s	= BGMRESMDR->nrhs;
						*ido = 3;
						status = 0;
					}else{
						ierr = BGMRESMDRRecoverSolutionInX(BGMRESMDR, x);
						*ido = 99;
						status = 0;
#if BGMRESMDRDEBUG
						MPI_Barrier(comm);
						if(BGMRESMDR->rank == 0) printf("BGMRESMDR done. Solution recovered in X\n");
						MPI_Barrier(comm);
#endif
					}
				}else if(flag == 2){
#if BGMRESMDRDEBUG
					MPI_Barrier(comm);
					if(!BGMRESMDR->rank) printf("Convergence not reached yet. Max interior iteration is reached but not the exterior one. Go to the next cycle \n");
					MPI_Barrier(comm);
#endif
					BGMRESMDR->U		= BGMRESMDR->Solution;
					BGMRESMDR->AU	= BGMRESMDR->Residual;
					BGMRESMDR->s		= BGMRESMDR->nrhs;
					*ido = 1;
					status = 3;
				}
			}
			break;
		//	Restart the method
		case 3:
			BGMRESMDR->Cycle++;
//#if BGMRESMDRREALRESIDUAL
			ierr = BGMRESMDRRecoverRealResidual(BGMRESMDR);
//#endif
			ierr = BGMRESMDRRestart(BGMRESMDR);
			ierr = BGMRESMDRSetHessenbergDefPart(BGMRESMDR);
#if BGMRESMDRDEBUG
			MPI_Barrier(comm);
			ierr = BGMRESMDRRecoverNormRealResidual(BGMRESMDR);
			if(!BGMRESMDR->rank) printf(" Restart : Real Res norm = %.14e \n", BGMRESMDR->rvec[BGMRESMDR->GIter].real);
			MPI_Barrier(comm);
#endif
			*ido = 40;
			status = 1;
			break;
		/*	Stop RCI */
		case 99:
			status = 0;
			*ido = 99;
			MPI_Barrier(comm);
			break;
		case 0:
			*ido = 99;
			break;
	}
	return ierr;
}

/** \fn int BGMRESMDROrthogonalization(BGMRESMDR_t* BGMRESMDR)
 * \brief
 * \details
 * \param BGMRESMDR
 * \remarks
 * \warning
*/
int BGMRESMDROrthogonalization(BGMRESMDR_t* BGMRESMDR){
	double time = MPI_Wtime();
	int ierr	= 0;
	int ND		= BGMRESMDR->ND;
	int dimZ	= BGMRESMDR->dimZ;
	int iter	= BGMRESMDR->iteration;
	int* vidx       = BGMRESMDR->VIdx;
	int t		= BGMRESMDR->nrhs;

	int nW = dimZ + vidx[iter - 1] + t;
	int mW = BGMRESMDR->ln;
	int nV = BGMRESMDR->ActualSize;

	MYTYPE const_alpha = {1., 0};
	MYTYPE const_beta  = {0, 0};

	MYTYPE* V = BGMRESMDR->V + (vidx[iter - 1] + t) * mW;
	MYTYPE* W = (dimZ > 0) ? BGMRESMDR->AZ : BGMRESMDR->V;

	MYTYPE* d   = BGMRESMDR->WorkH;
	MYTYPE* d_1 = d + nW * nV;
	int ldd	    = nW;

#if BGMRESMDRDEBUG
	MPI_Barrier(BGMRESMDR->comm);
	if(!BGMRESMDR->rank) printf("BGMRESMDROrthogonalization::S_i = [AZ V_i(:, 1 : j)]' * V_i(j+1) \n");
	if(!BGMRESMDR->rank) printf("\t nW = %d, nV = %d, mW = %d\n", nW, nV, mW);
	MPI_Barrier(BGMRESMDR->comm);
#endif
	//	S_i = [AZ V_i(:, 1 : j)]' * V_i(j+1)
	double timeComm = MPI_Wtime();
	cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nW, nV, mW, &const_alpha, W, mW, V, mW, &const_beta, d, ldd);
	BGMRESMDR->Timing.OrthogonalizationTime += MPI_Wtime() - timeComm;
	BGMRESMDR->flops += flops_zgemm(nW, nV, mW);

#if BGMRESMDRDEBUG
	MPI_Barrier(BGMRESMDR->comm);
	if(!BGMRESMDR->rank) printf("BGMRESMDROrthogonalization::S = Sum_i S_i \n");
	MPI_Barrier(BGMRESMDR->comm);
#endif
	//	S = Sum_i S_i
	timeComm = MPI_Wtime();
	ierr = MPI_Allreduce(MPI_IN_PLACE, d, nW * nV, MPI_C_DOUBLE_COMPLEX, MPI_SUM, BGMRESMDR->comm);
	BGMRESMDR->Timing.OrthogonalizationComm += MPI_Wtime() - timeComm;
	BGMRESMDR->flops += flops_allreduce_z(nW * nV, ND);

#if BGMRESMDRDEBUG
	MPI_Barrier(BGMRESMDR->comm);
	if(!BGMRESMDR->rank) printf("BGMRESMDROrthogonalization:: V_i(j+1) = V_i(j+1) - [AZ V_i(:,1:j)] * S\n");
	MPI_Barrier(BGMRESMDR->comm);
#endif
	//	V_i(j+1) = V_i(j+1) - [AZ V_i(:,1:j)] * S
	timeComm = MPI_Wtime();
	const_alpha.real = -1.;
	const_beta.real = 1.;

	cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, mW, nV, nW, &const_alpha, W, mW, d, ldd, &const_beta, V, mW);
	BGMRESMDR->flops += flops_zgemm_sum(mW, nV, nW);

	//	S1_i = 0
	memset(d_1, 0, (nV * ldd) * sizeof(MYTYPE));
	BGMRESMDR->flops += flops_memset_z(ldd * nV);
	BGMRESMDR->Timing.OrthogonalizationTime += MPI_Wtime() - timeComm;

	if(BGMRESMDR->Ortho == DBCGS){
		//	S_i = [AZ V_i(:, 1 : j)]' * V_i(j+1)
		timeComm = MPI_Wtime();
		const_alpha.real = 1.;
		const_beta.real = 0;
		cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nW, nV, mW, &const_alpha, W, mW, V, mW, &const_beta, d_1, ldd);
		BGMRESMDR->Timing.OrthogonalizationTime += MPI_Wtime() - timeComm;
		BGMRESMDR->flops += flops_zgemm(nW, nV, mW);

		//  S1 = Sum_i S1_i
		timeComm = MPI_Wtime();
		ierr = MPI_Allreduce(MPI_IN_PLACE, d_1, nW * nV, MPI_C_DOUBLE_COMPLEX, MPI_SUM, BGMRESMDR->comm);
		BGMRESMDR->Timing.OrthogonalizationComm += MPI_Wtime() - timeComm;
		BGMRESMDR->flops += flops_allreduce_z(nW * nV, ND);

		//	V_i(j+1) = V_i(j+1) - [AZ V_i(:,1:j)] * S
		timeComm = MPI_Wtime();
		const_alpha.real = -1.;
		const_beta.real = 1.;
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, mW, nV, nW, &const_alpha, W, mW, d_1, ldd, &const_beta, V, mW);
		BGMRESMDR->Timing.OrthogonalizationTime += MPI_Wtime() - timeComm;
		BGMRESMDR->flops += flops_zgemm_sum(mW, nV, nW);
	}
	//	H the pointer to the Hessenberg matrix
	int ldH	  = BGMRESMDR->MaxBasis + dimZ;
	MYTYPE* H = BGMRESMDR->H + (nW - t) * ldH;

	//	S = S + S1
	timeComm = MPI_Wtime();
	const_alpha.real = 1.;
	const_beta.real  = 1.;
	mkl_zomatadd ('C', 'N', 'N', nW, nV,
									const_alpha, d, ldd,
									const_beta, d_1, ldd,
									H, ldH);
	BGMRESMDR->Timing.OrthogonalizationTime += MPI_Wtime() - timeComm;
	BGMRESMDR->flops += flops_sum_z(nW, nV);

	//	d_i =  V_i(j+1)' * V_i(j+1)
	timeComm = MPI_Wtime();
	const_alpha.real = 1.;
	const_beta.real  = 0;
	cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nV, nV, mW, &const_alpha, V, mW, V, mW, &const_beta, d, nV);
	BGMRESMDR->Timing.OrthogonalizationTime += MPI_Wtime() - timeComm;
	BGMRESMDR->flops += flops_zgemm(nV, nV, mW);

	//	d = Sum_i temp
	timeComm = MPI_Wtime();
	ierr = MPI_Allreduce(MPI_IN_PLACE, d, nV * nV, MPI_C_DOUBLE_COMPLEX, MPI_SUM, BGMRESMDR->comm);
	BGMRESMDR->Timing.OrthogonalizationComm += MPI_Wtime() - timeComm;
	BGMRESMDR->flops += flops_allreduce_z(nV * nV, ND);

	//	Cholesky factorization of d
	timeComm = MPI_Wtime();
	ierr = LAPACKE_zpotrf (LAPACK_COL_MAJOR, 'U', nV, d, nV);
	if(ierr != 0){
		fprintf(stderr, "BGMRESMDROrthogonalization::LAPACKE_zpotrf error %d\n", ierr);
		MPI_Abort(BGMRESMDR->comm, ierr);
	}
	BGMRESMDR->Timing.OrthogonalizationTime += MPI_Wtime() - timeComm;
	BGMRESMDR->flops += flops_zpotrf(nV);

	//	Zeros under diagonal of d
	time = MPI_Wtime();
	for(int j = 0; j < nV; j++){
		for(int i = j + 1; i < nV; i++){
			d[ j * nV + i].real = 0;
			d[ j * nV + i].imag = 0;
		}
	}
	//	Copy d to the Hessenberg matrix
	timeComm = MPI_Wtime();
	H = H + nW;
	ierr = LAPACKE_zlacpy (LAPACK_COL_MAJOR, 'A', nV, nV, d, nV, H, ldH);
	BGMRESMDR->Timing.OrthogonalizationTime += MPI_Wtime() - timeComm;
	BGMRESMDR->flops += flops_copy_z(nV, nV);

	//	Normalize new basis vector
	timeComm = MPI_Wtime();
	const_alpha.real = 1.;
	cblas_ztrsm (CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, mW, nV, &const_alpha, d, nV, V, mW);
	BGMRESMDR->Timing.OrthogonalizationTime += MPI_Wtime() - timeComm;
	BGMRESMDR->flops += flops_ztrsm(mW, nV);

	//	Copy S to the Hessenberg matrices
	if(BGMRESMDR->Def != NoDef){
		timeComm = MPI_Wtime();
		H = H - nW;
		MYTYPE* H_2 = BGMRESMDR->H2 + (nW - t) * ldH;
		MYTYPE* H_3 = BGMRESMDR->H3 + (nW - t) * ldH;
		ierr = LAPACKE_zlacpy (LAPACK_COL_MAJOR, 'A', nW + nV, nV, H, ldH, H_2, ldH);
		ierr = LAPACKE_zlacpy (LAPACK_COL_MAJOR, 'A', nW + nV, nV, H, ldH, H_3, ldH);
		BGMRESMDR->flops += flops_copy_z((nW + nV), nV);
		BGMRESMDR->flops += flops_copy_z((nW + nV), nV);
		BGMRESMDR->Timing.OrthogonalizationTime += MPI_Wtime() - timeComm;
	}
	BGMRESMDR->Timing.OrthogonalizationTime += MPI_Wtime() - time;
	return ierr;
}


/** \fn int	BGMRESMDRUpdateH(BGMRESMDR_t* BGMRESMDR)
 * \brief
 * \details
 * \param BGMRESMDR
 * \remarks
 * \warning
*/
int	BGMRESMDRUpdateH(BGMRESMDR_t* BGMRESMDR){
	double time = MPI_Wtime();
	int ierr	= 0;
	int iter	= BGMRESMDR->iteration;
	int s		= BGMRESMDR->ActualSize;
	int t		= BGMRESMDR->nrhs;
	int dimZ	= BGMRESMDR->dimZ;
	int* vidx = BGMRESMDR->VIdx;
	int* BIdx = BGMRESMDR->BIdx;

	MYTYPE* Q	= BGMRESMDR->Q + t * t * (iter - 1);
	MYTYPE* Qtau	= BGMRESMDR->Qtau + t * (iter - 1);
	int Qoffset	= t * t * (iter - 1);
	int Qtauoffset	= t * (iter - 1);

	MYTYPE* H	= BGMRESMDR->H;
	int ldH		= BGMRESMDR->MaxBasis + dimZ;

	//	Apply inexact breakdown rotations if they exist
	if(BGMRESMDR->Red != NoRed){
		for(int j = iter - 1; j > 0; j--){
			if(BGMRESMDR->BIdx[j] == 1){
				ierr = LAPACKE_zunmqr (LAPACK_COL_MAJOR, 'L', 'N', t, s, t, Q, t, Qtau, H + (vidx[iter - 1] + dimZ) * ldH + dimZ + vidx[j], ldH);
				BGMRESMDR->flops += flops_zgemm(t, s, t);
			}
			Q	-= t * t;
			Qtau	-= t;
		}
#if BGMRESMDRDEBUG
		MPI_Barrier(BGMRESMDR->comm);
		if(Q != BGMRESMDR->Q){
			fprintf(stderr, "BGMRESMDRUpdateH:: Verify indexing of inexact breakdown rotations\n");
			MPI_Abort(BGMRESMDR->comm, -99);
		}
		MPI_Barrier(BGMRESMDR->comm);
#endif
	}
	BGMRESMDR->Timing.InexactBreakDownTime += MPI_Wtime() - time;
#if BGMRESMDRDEBUG
	MPI_Barrier(BGMRESMDR->comm);
	if(vidx[iter] - vidx[iter - 1] != s){
		fprintf(stderr, "BGMRESMDRUpdateH:: Verify dimensions iter = % d\n vidx[iter] = %d\n vidx[iter - 1] = %d\n, s = %d \n", iter, vidx[iter], vidx[iter - 1], s);
		MPI_Abort(BGMRESMDR->comm, -1000);
	}
	MPI_Barrier(BGMRESMDR->comm);
#endif

	time = MPI_Wtime();
	//	Apply Householder rotations in H
	for(int j = 0; j < iter - 1; j++){
		//	Pointer to H
		Q = H + (dimZ + vidx[j]) * ldH + dimZ + vidx[j];

		//	Pointer to Htau
		Qtau = BGMRESMDR->Htau + vidx[j];

		//	Q^T * H
		ierr = LAPACKE_zunmqr (LAPACK_COL_MAJOR, 'L', 'C', t + vidx[j + 1] - vidx[j], vidx[iter] - vidx[iter - 1], vidx[j + 1] - vidx[j], Q, ldH, Qtau, H + (dimZ + vidx[iter - 1]) * ldH + dimZ + vidx[j], ldH);
		BGMRESMDR->flops += flops_zgemm(t + vidx[j + 1] - vidx[j], vidx[iter] - vidx[iter - 1], vidx[j + 1] - vidx[j]);
	}

	Qtau = BGMRESMDR->Htau + vidx[iter - 1];

#if BGMRESMDRDEBUG
	MPI_Barrier(BGMRESMDR->comm);
	//if(!BGMRESMDR->rank) PrintMat(BGMRESMDR->H2, vidx[iter] + t, vidx[iter], ldH, "H");
	MPI_Barrier(BGMRESMDR->comm);
#endif
	ierr = LAPACKE_zgeqrf (LAPACK_COL_MAJOR, s + t, s, H + (dimZ + vidx[iter - 1]) * ldH + dimZ + vidx[iter - 1], ldH, Qtau);
	BGMRESMDR->flops += flops_zgeqrf(s + t, s);

	BGMRESMDR->Timing.HessenbergOperationsTime += MPI_Wtime() - time;
	return ierr;
}

/** \fn int BGMRESMDRUpdateKRHS(BGMRESMDR_t* BGMRESMDR)
 * \brief
 * \details
 * \param BGMRESMDR
 * \remarks
 * \warning
*/
int BGMRESMDRUpdateKRHS(BGMRESMDR_t* BGMRESMDR){
	double time = MPI_Wtime();
	int ierr  = 0;
	int iter  = BGMRESMDR->iteration;
	int t     = BGMRESMDR->nrhs;
	int dimZ  = BGMRESMDR->dimZ;
	int* vidx = BGMRESMDR->VIdx;

	MYTYPE* Q	= BGMRESMDR->H + (dimZ + BGMRESMDR->MaxBasis) * (dimZ + vidx[iter - 1]) + dimZ +  vidx[iter - 1];
	MYTYPE* Qtau	= BGMRESMDR->Htau + vidx[iter - 1];

	MYTYPE* F	= BGMRESMDR->KRHS + dimZ + vidx[iter - 1];

	ierr = LAPACKE_zunmqr (LAPACK_COL_MAJOR, 'L', 'C', t + vidx[iter] - vidx[iter - 1], t, vidx[iter] - vidx[iter - 1], Q, dimZ + BGMRESMDR->MaxBasis, Qtau, F, dimZ + BGMRESMDR->MaxBasis);
	BGMRESMDR->flops += flops_zgemm(t + vidx[iter] - vidx[iter - 1], t, vidx[iter] - vidx[iter - 1]);
	BGMRESMDR->Timing.HessenbergOperationsTime += MPI_Wtime() - time;
	return ierr;
}


/** \fn int BGMRESMDRCriterionTest(BGMRESMDR_t* BGMRESMDR, int* flag)
 * \brief
 * \details
 * \param BGMRESMDR
 * \param flag
 * \remarks
 * \warning
*/
int BGMRESMDRCriterionTest(BGMRESMDR_t* BGMRESMDR, int* flag){
	int iter  = BGMRESMDR->iteration;
	int dimZ  = BGMRESMDR->dimZ;
	int t	  = BGMRESMDR->nrhs;
	int* vidx = BGMRESMDR->VIdx;
	MYTYPE* r = BGMRESMDR->KRHS + dimZ + vidx[iter];
	int ldr	  = dimZ + BGMRESMDR->MaxBasis;

	double normF = 0;

	for(int j = 0; j < t; j++){
		for(int i = 0; i < t; i++){
			normF += r[j * ldr + i].real * r[j * ldr + i].real;
			normF += r[j * ldr + i].imag * r[j * ldr + i].imag;
		}
	}
	normF = sqrt(normF);
	BGMRESMDR->rvec[BGMRESMDR->GIter].real = normF;
	if(normF < BGMRESMDR->Ctol * BGMRESMDR->MaxNormb){
		*flag = 0;
#if BGMRESMDRDEBUG
		MPI_Barrier(BGMRESMDR->comm);
		if(!BGMRESMDR->rank) printf("Convergence is achieved:: Frobenius norm of residual = %.14e\n", normF);
		if(!BGMRESMDR->rank) printf("iter = %d, vidx[iter] =  %d\n", iter, vidx[iter]);
		MPI_Barrier(BGMRESMDR->comm);
#endif
	}
	else if(vidx[iter] + t + BGMRESMDR->ActualSize <= BGMRESMDR->MaxBasis){
		*flag = 1;
	}
	else if(BGMRESMDR->Cycle < BGMRESMDR->MaxCycle){
		*flag = 2;
#if BGMRESMDRDEBUG
		MPI_Barrier(BGMRESMDR->comm);
		if(!BGMRESMDR->rank) printf("iter = %d, vidx[iter] = %d \n", iter, vidx[iter]);
		if(!BGMRESMDR->rank) printf("Convergence is not achieved, RESTART\n");
		MPI_Barrier(BGMRESMDR->comm);
#endif
	}
	else{
		*flag = 3;
#if BGMRESMDRDEBUG
		MPI_Barrier(BGMRESMDR->comm);
		if(!BGMRESMDR->rank) printf("BGMRESMDR filed to achieve convergence\n");
		MPI_Barrier(BGMRESMDR->comm);
#endif
	}
	return 0;
}

/** \fn int BGMRESMDRRecoverSolution(BGMRESMDR_t* BGMRESMDR)
 * \brief
 * \details
 * \param BGMRESMDR
 * \remarks
 * \warning
*/
int BGMRESMDRRecoverSolution(BGMRESMDR_t* BGMRESMDR){
	double time = MPI_Wtime();
	int dimZ  = BGMRESMDR->dimZ;
	int iter  = BGMRESMDR->iteration;
	int* vidx = BGMRESMDR->VIdx;
	int t	  = BGMRESMDR->nrhs;

	MYTYPE* H = BGMRESMDR->H;
	int mH	  = dimZ + vidx[iter] + t;
	int nH	  = dimZ + vidx[iter];
	int ldH	  = dimZ + BGMRESMDR->MaxBasis;

	MYTYPE* kr = BGMRESMDR->KRHS;
	int ldkr   = ldH;

	MYTYPE* V = BGMRESMDR->V;
	int nV	  = vidx[iter];

	MYTYPE* Z = BGMRESMDR->Z;
	MYTYPE* x = BGMRESMDR->Solution;

	MYTYPE const_alpha = {1., 0};
	MYTYPE const_beta  = {0, 0};

	int ln = BGMRESMDR->ln;

#if BGMRESMDRDEBUG
		MPI_Barrier(BGMRESMDR->comm);
		if(!BGMRESMDR->rank) printf("BGMRESMDRRecoverSolution:: dimZ = %d\n", dimZ);
		if(!BGMRESMDR->rank) printf("BGMRESMDRRecoverSolution:: iter = %d\n", iter);
		if(!BGMRESMDR->rank) printf("BGMRESMDRRecoverSolution:: nH   = %d\n", nH);
		if(!BGMRESMDR->rank) printf("BGMRESMDRRecoverSolution:: BGMRESMDR->MaxBasis = %d\n", BGMRESMDR->MaxBasis);
		MPI_Barrier(BGMRESMDR->comm);
#endif
	//	Triangular solve of LSP
	int ierr = LAPACKE_ztrtrs (LAPACK_COL_MAJOR, 'U', 'N', 'N', nH, t, H, ldH, kr, ldkr);
	BGMRESMDR->flops += flops_ztrsm(nH, t);

	if(dimZ > 0){
		//	x = x + Z * kr(1:dimZ, :)
		const_alpha.real = 1.;
		const_beta.real = 1.;
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ln, t, dimZ, &const_alpha, Z, ln, kr, ldkr, &const_beta, x, ln);
		BGMRESMDR->flops += flops_zgemm_sum(ln, t, dimZ);
	}

	//	x = x + V * kr(dimZ + 1 : dimZ + nV, : )
	const_alpha.real = 1.;
	const_beta.real = 1.;
	cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ln, t, nV, &const_alpha, V, ln, kr + dimZ, ldkr, &const_beta, x, ln);
	BGMRESMDR->flops += flops_zgemm_sum(ln, t, nV);

	BGMRESMDR->Timing.Recovering += MPI_Wtime() - time;
#if BGMRESMDRDEBUG
		MPI_Barrier(BGMRESMDR->comm);
		if(!BGMRESMDR->rank) printf("BGMRESMDRRecoverSolution:: Well passed\n");
		MPI_Barrier(BGMRESMDR->comm);
#endif
	return ierr;
}

/** \fn int BGMRESMDRRecoverSolutionInX(BGMRESMDR_t* BGMRESMDR, MYTYPE* x)
 * \brief
 * \details
 * \param BGMRESMDR
 * \param x
 * \remarks
 * \warning
*/
int BGMRESMDRRecoverSolutionInX(BGMRESMDR_t* BGMRESMDR, MYTYPE* x){
	double time = MPI_Wtime();
	int ierr = 0;
	int t		 = BGMRESMDR->nrhs;
	int ln	 = BGMRESMDR->ln;
	MYTYPE* sol = BGMRESMDR->Solution;

	//	Copy the solution to x
	ierr = LAPACKE_zlacpy (LAPACK_COL_MAJOR, 'A', ln, t, sol, ln, x, ln);
	BGMRESMDR->flops += flops_copy_z(ln, t);

	BGMRESMDR->Timing.Recovering += MPI_Wtime() - time;
	return ierr;
}


/** \fn int BGMRESMDRRecoverRealResidual(BGMRESMDR_t* BGMRESMDR)
 * \brief
 * \details
 * \param BGMRESMDR
 * \remarks
 * \warning
*/
int BGMRESMDRRecoverRealResidual(BGMRESMDR_t* BGMRESMDR){
#if BGMRESMDRDEBUG
	MPI_Barrier(BGMRESMDR->comm);
	if(!BGMRESMDR->rank){
		printf("BGMRESMDRRecoverRealResidual::Begin\n");
	}
	MPI_Barrier(BGMRESMDR->comm);
#endif

	int ierr = 0;
	double time = MPI_Wtime();

	int t			= BGMRESMDR->nrhs;
	int ln		= BGMRESMDR->ln;

	MYTYPE* R = BGMRESMDR->Residual;
	MYTYPE* b = BGMRESMDR->b;

	//	R = b - Ax
	for(int j = 0; j < t; j++){
		for(int i = 0; i < ln; i++){
			R[j * ln + i].real = b[j * ln + i].real - R[j * ln + i].real;
			R[j * ln + i].imag = b[j * ln + i].imag - R[j * ln + i].imag;
		}
	}
	BGMRESMDR->flops += 4 * t * ln;

#if BGMRESMDRDEBUG
	MYTYPE* work = BGMRESMDR->WorkH;
	//	work_i = R' * R
	MYTYPE const_alpha = {1., 0};
	MYTYPE const_beta = {0, 0};
	cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, t, t, ln, &const_alpha, R, ln, R, ln, &const_beta, work, t);
	BGMRESMDR->flops += flops_zgemm(t, t, ln);

	//	work = sum work_i
	ierr = MPI_Allreduce(MPI_IN_PLACE, work, t * t, MPI_C_DOUBLE_COMPLEX, MPI_SUM, BGMRESMDR->comm);
	BGMRESMDR->flops += flops_allreduce_z(t * t, BGMRESMDR->ND);

	MPI_Barrier(BGMRESMDR->comm);
	for( int i = 0; i < t; i++){
		if(!BGMRESMDR->rank){
			printf("BGMRESMDRRecoverRealResidual::Norm of Real Residual(%d) = %.14e\n", i, sqrt(work[i + t * i].real + work[i + t * i].imag));
		}
	}
	MPI_Barrier(BGMRESMDR->comm);
#endif
	BGMRESMDR->Timing.Recovering += MPI_Wtime() - time;
#if BGMRESMDRDEBUG
	MPI_Barrier(BGMRESMDR->comm);
	if(!BGMRESMDR->rank){
		printf("BGMRESMDRRecoverRealResidual::End\n");
	}
	MPI_Barrier(BGMRESMDR->comm);
#endif
	return ierr;
}


/** \fn int BGMRESMDRRecoverNormRealResidual(BGMRESMDR_t* BGMRESMDR)
 * \brief
 * \details
 * \param BGMRESMDR
 * \remarks
 * \warning
*/
int BGMRESMDRRecoverNormRealResidual(BGMRESMDR_t* BGMRESMDR){
	int ierr = 0;
	int t			= BGMRESMDR->nrhs;
	int ln		= BGMRESMDR->ln;

	MYTYPE* R = BGMRESMDR->Residual;
	double norm = 0;

	MYTYPE* work = BGMRESMDR->WorkH;
	//	work_i = R' * R
	MYTYPE const_alpha = {1., 0};
	MYTYPE const_beta = {0, 0};
	cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, t, t, ln, &const_alpha, R, ln, R, ln, &const_beta, work, t);
	BGMRESMDR->flops += flops_zgemm(t, t, ln);

	//	work = sum work_i
	ierr = MPI_Allreduce(MPI_IN_PLACE, work, t * t, MPI_C_DOUBLE_COMPLEX, MPI_SUM, BGMRESMDR->comm);
	BGMRESMDR->flops += flops_allreduce_z(t * t, BGMRESMDR->ND);

	for( int i = 0; i < t; i++){
		norm += work[i + t * i].real;
	}
	BGMRESMDR->rvec[BGMRESMDR->GIter].real = sqrt(norm);
	return 0;
}

int BGMRESMDRDeflation(BGMRESMDR_t* BGMRESMDR){
	int ierr = 0;
	if(BGMRESMDR->Def == RITZ){
		ierr = BGMRESMDRRITZDeflation(BGMRESMDR);
	}else if(BGMRESMDR->Def == HRITZ){
		ierr = BGMRESMDRHRITZDeflation(BGMRESMDR);
	}else if(BGMRESMDR->Def == SVD){
		ierr = BGMRESMDRSVDDeflation(BGMRESMDR);
	}
	return ierr;
}

/** \fn int BGMRESMDRRITZDeflationSimple(BGMRESMDR_t* BGMRESMDR){
 * \brief First deflation based on Ritz values
 * \details Computes the smalles eigenvalues of the upper block of the Hessenberg matrix
 * \param BGMRESMDR The BGMRESMDR context
 * \remarks This routine is applied when no deflation vectors exist
 * \warning
*/
int BGMRESMDRRITZDeflationSimple(BGMRESMDR_t* BGMRESMDR){

	double timing = MPI_Wtime();
	int ierr = 0;
	int rank = BGMRESMDR->rank;
	MPI_Comm comm = BGMRESMDR->comm;
	int iter = BGMRESMDR->iteration;
	int t    = BGMRESMDR->nrhs;
	int cpt  = 0;

	//	Hessenberg matrices
	MYTYPE* H		= BGMRESMDR->H;
	MYTYPE* H2	= BGMRESMDR->H2;
	MYTYPE* H3	= BGMRESMDR->H3;

	MYTYPE* work= BGMRESMDR->WorkH;

	int* vidx   = BGMRESMDR->VIdx;
	int ldh			= BGMRESMDR->MaxBasis;
	int nH			= vidx[iter];
	int mH			=	vidx[iter] + t;

	//	eigenvalues
	double* wr	= (double*) malloc( nH * sizeof(double));
	MYTYPE* wi	= (MYTYPE*) malloc( nH * sizeof(MYTYPE));
	MYTYPE* tau = (MYTYPE*) malloc( nH * sizeof(MYTYPE));
	double* scale = (double*) malloc( nH * sizeof(double));

	//	selection of EVs
	int* Perm		= (int*) malloc( nH * sizeof(int));
	int* select	= (int*) malloc( nH * sizeof(int));
	int* ifailr	= (int*) malloc( nH * sizeof(int));

	int ilo = 1;
	int ihi = nH;

	//	Copy H2 into H3
	ierr = LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', mH, nH, H2, ldh, H3, mH);
	BGMRESMDR->flops += flops_copy_z(mH, nH);
	/***************************	Return to Hessenberg	********************************/
	ierr = LAPACKE_zgebal(LAPACK_COL_MAJOR, 'P', vidx[iter], H2, ldh, &ilo, &ihi, scale);
	BGMRESMDR->flops += flops_zgebal(vidx[iter]);
	if(ierr != 0){
		fprintf(stderr, "BGMRESMDRRITZDeflationSimple::LAPACKE_zgebal error\n");
		MPI_Abort(BGMRESMDR->comm, ierr);
	}
	//	Return H into Hessenberg matrix
	ierr = LAPACKE_zgehrd (LAPACK_COL_MAJOR, vidx[iter], ilo, ihi, H2, ldh, tau);
	BGMRESMDR->flops += flops_zgehrd(nH, ilo, ihi);
	if(ierr != 0){
		fprintf(stderr, "BGMRESMDRRITZDeflationSimple::LAPACKE_zgehrd error\n");
		MPI_Abort(BGMRESMDR->comm, ierr);
	}
	//	Copy the Hessenberg matrix to Work
	ierr = LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', vidx[iter], vidx[iter], H2, ldh, H, vidx[iter]);
	BGMRESMDR->flops += flops_copy_z(vidx[iter], vidx[iter]);

	ierr = LAPACKE_zunghr (LAPACK_COL_MAJOR, vidx[iter], ilo, ihi, H, vidx[iter], tau);
	BGMRESMDR->flops += flops_zunghr(ilo, ihi);

	if(ierr != 0){
		fprintf(stderr, "BGMRESMDRRITZDeflationSimple::LAPACKE_zunghr error %d \n", ierr);
		MPI_Abort(BGMRESMDR->comm, ierr);
	}
	/***************************	Computation of eigenvalues	********************************/
	//	Compute eigenvalues
	ierr = LAPACKE_zhseqr(LAPACK_COL_MAJOR, 'S', 'V', nH, ilo, ihi, H2, ldh, wi, H, nH);
	BGMRESMDR->flops += flops_zhseqr(nH);

	if(ierr != 0){
		fprintf(stderr, "BGMRESMDRRITZDeflationSimple::Error in computing eigenvalues in zhseqr\n");
		MPI_Abort(comm, ierr);
	}

	//	Set values of choosing evs
	for(unsigned int j = 0; j < nH; j++){ Perm[j]		= j; }
	for(unsigned int j = 0; j < nH; j++){ select[j]	= 0; }
	for(unsigned int j = 0; j < nH; j++){ ifailr[j]	= 0; }
	//	Compute the absolute values of eigenvalues
	for(unsigned int j = 0; j < nH; j++){
		wr[j] = wi[j].real * wi[j].real + wi[j].imag * wi[j].imag;
	}

	//	sort eigenvalues
	quickSortDoubleWithPerm(&wr, nH, &Perm);

	//	Select eigenvalues
	for(int j = 0; j < nH; j++){

		//	select if there is place for an eigenvector
		if(cpt < BGMRESMDR->MaxDefDim){
			cpt++;
			select[Perm[j]] = 1;
#if BGMRESMDRDEBUG
				MPI_Barrier(comm);
					if(!rank){
						printf("BGMRESMDRRITZDeflationSimple:: EV[%d] = %lf + %lf is deflated\n", Perm[j], wi[Perm[j]].real, wi[Perm[j]].imag);
					}
				MPI_Barrier(comm);
#endif
		}
	}

	int dimZ = BGMRESMDR->MaxDefDim;
	BGMRESMDR->dimZ = BGMRESMDR->MaxDefDim;
	if(dimZ != cpt){
		fprintf(stderr, "BGMRESMDRRITZDeflationSimple:: number of deflated eigenvalues is less than the size precised\n");
		MPI_Abort(comm, -dimZ);
	}
	      //  Reordering the Schur form
  int counter = 1;
  for(int i = 0; i < nH; i++){
    if(select[i] == 1){
      int ifst = i + 1;
      int ilst = counter;
      ierr = LAPACKE_ztrexc(LAPACK_COL_MAJOR, 'V', nH, H2, ldh, H, nH, ifst, ilst);
			BGMRESMDR->flops += flops_ztrexc(nH, ifst, ilst);
      if(ierr != 0){
        fprintf(stderr, "BGMRESMDRRITZDeflationSimple:: LAPACKE_ztrexc error %d \n", ierr);
        MPI_Abort(comm, ierr);
      }
      counter++;
    }
  }
	ierr = LAPACKE_zgebak( LAPACK_COL_MAJOR, 'P', 'R', vidx[iter], ilo, ihi, scale, cpt, H, vidx[iter]);
	BGMRESMDR->flops += flops_zgebak(vidx[iter], cpt);
	free(scale);
	/******************************	End of computation of eigenvalues	***********************************/
	/***********************************	Computation of local matrices  ********************************/
	MYTYPE* pk = H;
	MYTYPE* Pk = BGMRESMDR->WorkH;

	//	Compute P_k = H * p_k
	MYTYPE const_alpha = {1., 0};
	MYTYPE const_beta = {0, 0};
	cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, mH, dimZ, nH, &const_alpha, H3, mH, pk, nH, &const_beta, Pk, mH);
	BGMRESMDR->flops += flops_zgemm(mH, dimZ, nH);

	//	QR factorization of P_k = Q_P R_P
	ierr = LAPACKE_zgeqrf (LAPACK_COL_MAJOR, mH, cpt, Pk, mH, tau);
	BGMRESMDR->flops += flops_zgeqrf(mH, cpt);

	//	p_k = p_k R_P^{-1}
	cblas_ztrsm (CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, nH, cpt, &const_alpha, Pk, mH, pk, nH);
	BGMRESMDR->flops += flops_ztrsm(nH, cpt);

	/******************************** End of computation of local matrices ******************************/
	/***************************************	Setting ZtZ ZtAZ and D	***********************************/
	MYTYPE* ZtZ		=	BGMRESMDR->ZtZ;
	MYTYPE* D			=	BGMRESMDR->D;

	//	Set values of D
	for(int j = 0; j < dimZ; j++){
		//	Compute norm of p_k(: , j)
		double normpk = cblas_dznrm2(nH, pk +  j * nH, 1);
		D[j].real = 1./normpk;
		D[j].imag = 0;

		//	Normalize p_k(:, j)
		for(int i = 0; i < nH; i++){
			pk[j * nH + i].real = pk[j * nH + i].real * D[j].real;
			pk[j * nH + i].imag = pk[j * nH + i].imag * D[j].real;
		}
	}
	BGMRESMDR->flops += 6 * dimZ * nH;

	//	Prepares Z' * Z
	cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, dimZ, dimZ, nH, &const_alpha, pk, nH, pk, nH, &const_beta, ZtZ, dimZ);
	BGMRESMDR->flops += flops_zgemm(dimZ, dimZ, nH);

	//	Form the orthogonal matrix of P_k
	ierr =  LAPACKE_zungqr (LAPACK_COL_MAJOR, mH, dimZ, dimZ, Pk, mH,tau);
	BGMRESMDR->flops += flops_zungqr(mH, dimZ);

	/************************************	End	Setting ZtZ ZtAZ and D	***********************************/
	/******************************	Extension of vectors to be occupied in Z ****************************/
	MYTYPE* Z			= BGMRESMDR->Z;
	MYTYPE* AZ		= BGMRESMDR->AZ;
	MYTYPE* V			= BGMRESMDR->V;
	MYTYPE* workz	=	BGMRESMDR->WorkZ;

	int ln = BGMRESMDR->ln;

	//	Z = V * p_k
	cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ln, cpt, nH, &const_alpha, V, ln, pk, nH, &const_beta, Z, ln);
	BGMRESMDR->flops += flops_zgemm(ln, cpt, nH);

	//	[AZ] = [V] * Q_k
	if(V == AZ){
		fprintf(stderr, "BGMRESMDRRITZDeflationSimple:: zgemm in place does not work\n");
		MPI_Abort(comm, -69);
	}
	cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ln, cpt, mH, &const_alpha, V, ln, Pk, mH, &const_beta, AZ, ln);
	BGMRESMDR->flops += flops_zgemm(ln, cpt, mH);

	free(Perm);
	free(select);
	free(wr);
	free(wi);
	free(tau);

	BGMRESMDR->Timing.DeflationComputationTime += MPI_Wtime() - timing;

	return ierr;
}

/** \fn int BGMRESMDRRITZDeflation(BGMRESMDR_t* BGMRESMDR){
 * \brief Deflation update based on Ritz values approximation
 * \details Computes the eigenvalues of the GEVP \f$ (V' W G) u = \lambda (V' V) u \f$
 * \param BGMRESMDR The BGMRESMDR context
 * \remarks The following relation is supposed to hold \f$ A V = W G \f$, where \f$ V = [Z, \mathcal{V}_k], W = [\tilde{Z}, \mathcal{V}_{k+1}] \f$ and  \f$ G = [D, \tilde{Z}' \mathcal{V}_k; 0, H_k]\f$
  \warning
*/
int BGMRESMDRRITZDeflation(BGMRESMDR_t* BGMRESMDR){
	int ierr = 0;
	int dimZ = BGMRESMDR->dimZ;
	int iter = BGMRESMDR->iteration;
	MPI_Comm comm = BGMRESMDR->comm;
	int rank			= BGMRESMDR->rank;
	int t			    = BGMRESMDR->nrhs;

	if(dimZ == 0){
		ierr = BGMRESMDRRITZDeflationSimple(BGMRESMDR);
	}else{
		double timing = MPI_Wtime();
		double timing_comm	= 0;
		double timing_op		= 0;

		//	Variables of basis vectors
		MYTYPE* V			= BGMRESMDR->V;
		MYTYPE* Z			= BGMRESMDR->Z;
		MYTYPE* workZ	= BGMRESMDR->WorkZ;
		MYTYPE* AZ		= BGMRESMDR->AZ;

		int ln			=	BGMRESMDR->ln;

		//	Variables of Hessenberg matrices
		MYTYPE* H		= BGMRESMDR->H;
		MYTYPE* H2	= BGMRESMDR->H2;
		MYTYPE* H3	= BGMRESMDR->H3;

		int* vidx   = BGMRESMDR->VIdx;

		int ldh			= BGMRESMDR->MaxBasis + dimZ;
		int nH			= vidx[iter] + dimZ;
		int mH			= vidx[iter] + dimZ + t;
		int nV			= vidx[iter];

		//	Work space
		MYTYPE* work = BGMRESMDR->WorkH;

		//	Generalized EVP variables
		MYTYPE* VtW		= BGMRESMDR->VtW;
		MYTYPE*	VtV		= BGMRESMDR->VtV;
		MYTYPE* VtWG	= BGMRESMDR->GtG;	//	Real value inside is V' * W * G
		MYTYPE* ZtZ		= BGMRESMDR->ZtZ;
		MYTYPE* D			=	BGMRESMDR->D;

		//	scaling factors for balancing matrices
		double* lscale = NULL;
		double* rscale = NULL;

		//	vectors for eigenvalues
		double* alphar = NULL;
		MYTYPE* alphai = NULL;
		MYTYPE* beta   = NULL;

		//	vector for rotation scalars
		MYTYPE* tau    = NULL;

		double* wr = NULL;
		MYTYPE* wi = NULL;

		//	vectors for selection of eigenvalues
		int* Perm = NULL;
		int* select = NULL;
		int* ifailr = NULL;

		int cpt = 0;

		int ilo = 1;
		int ihi = nH;

		MYTYPE const_alpha = {1., 0};
		MYTYPE const_beta = {0, 0};
		 /*********************************************************
		 *	Prepare  A = [Z V_iter]' * [AZ V_iter v_{iter+1}]	* G	*
		 *********************************************************/
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank) printf("BGMRESMDRRITZDeflation:: Preparing A = [Z V_iter]' * [AZ V_iter v_{iter+1}] * G\n");
	MPI_Barrier(comm);
#endif

		//	Temp = Z' * [AZ V_k v_k+1]
		cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, dimZ, mH, ln, &const_alpha, Z, ln, AZ, ln, &const_beta, work, dimZ);
		BGMRESMDR->flops += flops_zgemm(dimZ, mH, ln);
		timing_op += MPI_Wtime() - timing;

		timing = MPI_Wtime();
		ierr = MPI_Allreduce(MPI_IN_PLACE, work, dimZ * mH, MPI_C_DOUBLE_COMPLEX, MPI_SUM, comm);
		BGMRESMDR->flops += flops_allreduce_z(dimZ * mH, BGMRESMDR->ND);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRRITZDeflation:: MPI_Allreduce error %d\n", ierr);
			MPI_Abort(comm, ierr);
		}
		timing_comm += MPI_Wtime() - timing;

		//	Concatenate Temp to [0_{iter, dimZ}, I_{iter}, 0_{iter, 1}]
		timing = MPI_Wtime();
		ierr = LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', dimZ, mH, work, dimZ, VtW, nH);
		BGMRESMDR->flops += flops_copy_z(dimZ, nH);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRRITZDeflation:: LAPACKE_zlacpy concatenate Temp to [0_{iter, dimZ}, I_{iter}, 0_{iter, 1}] error %d\n", ierr);
			MPI_Abort(comm, ierr);
		}

		//	Set space under Temp to 0
		for(int j = 0; j < mH; j++){
			for(int i = 0; i < nV; i++){
				VtW[j * nH + dimZ + i].real = 0;
				VtW[j * nH + dimZ + i].imag = 0;
			}
		}
		for(int j = 0; j < nV; j++){
			VtW[dimZ * nH + j * nH + dimZ + j].real = 1.;
			VtW[dimZ * nH + j * nH + dimZ + j].imag = 0;
		}
		//	Compute V' * W * G
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nH, nH, mH, &const_alpha, VtW, nH, H2, ldh, &const_beta, VtWG, nH);
		BGMRESMDR->flops += flops_zgemm(nH, nH, mH);
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank) printf("BGMRESMDRRITZDeflation:: A is ready \n");
	if(!rank) PrintMatZ(VtWG, nH, nH, nH, "VtWG");
	MPI_Barrier(comm);
#endif

		/*********************************/

		 /********************************
		 *	Prepare  B = [Z V]' * [Z V]	 *
		 ********************************/
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank) printf("BGMRESMDRRITZDeflation:: Preparing B = [Z V]' * [Z V]\n");
	MPI_Barrier(comm);
#endif

			//	Copy Z' * V
		ierr = LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', dimZ, nV, work + dimZ * dimZ, dimZ, VtV + dimZ * nH, nH);
		BGMRESMDR->flops += flops_copy_z(dimZ, nV);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRRITZDeflation:: LAPACKE_zlacpy Copy Z' * V  error %d\n", ierr);
			MPI_Abort(comm, ierr);
		}

			//	Copy V' * Z
		//mkl_domatcopy ('C', 'T', iter, dimZ, 1., work + dimZ * dimZ, dimZ, VtV + dimZ, dimZ + iter);
		for(int j = 0; j < dimZ; j++){
			for(int i = 0; i < nV; i++){
				VtV[j * nH + dimZ + i].real = work[dimZ * dimZ + i * dimZ + j].real;
				VtV[j * nH + dimZ + i].imag = -work[dimZ * dimZ + i * dimZ + j].imag;
			}
		}
		BGMRESMDR->flops += 2 * nV * dimZ;
			//	Copy of  Z' * Z
		ierr = LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', dimZ, dimZ, ZtZ, dimZ, VtV, nH);
		BGMRESMDR->flops += flops_copy_z(dimZ, dimZ);

			//	Set V' * V
		for(int j = 0; j < nV; j++){
			for(int i = 0; i < nV; i++){
				VtV[ j * nH + dimZ * nH + dimZ + i].real = (i == j) ? 1. : 0;
				VtV[ j * nH + dimZ * nH + dimZ + i].imag = 0;
			}
		}
		BGMRESMDR->flops += 2 * nV * nV;
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank) printf("BGMRESMDRRITZDeflation:: B is ready \n");
	//if(!rank) PrintMat(VtV, nH, nH, nH, "VtV");
	MPI_Barrier(comm);
#endif
		 /********************************/

		//	rotation scalars
		tau			= (MYTYPE*) malloc(mH * sizeof(MYTYPE));

		//	scaling factors for balancing matrices
		lscale	=	(double*) malloc( nH * sizeof(double));
		rscale	=	(double*) malloc( nH * sizeof(double));

		//	eigenvalues
		alphar	= (double*) malloc( nH * sizeof(double));
		alphai	= (MYTYPE*) malloc( nH * sizeof(MYTYPE));
		beta  	= (MYTYPE*) malloc( nH * sizeof(MYTYPE));

		//	QR factorization of B
		ierr = LAPACKE_zgeqrf (LAPACK_COL_MAJOR, nH, nH, VtV, nH, tau);
		BGMRESMDR->flops += flops_zgeqrf(nH, nH);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRRITZDeflation:: LAPACKE_zgeqrf error %d\n", ierr);
			if(!rank) PrintMatZ(VtV, nH, nH, nH, "VtV");
			MPI_Abort(comm, ierr);
		}

		//	Apply Q on GtG
		ierr = LAPACKE_zunmqr (LAPACK_COL_MAJOR, 'L', 'C', nH, nH, nH, VtV, nH, tau, VtWG, nH);
		BGMRESMDR->flops += flops_zgemm(nH, nH, nH);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRRITZDeflation:: LAPACKE_zunmqr error %d\n", ierr);
			MPI_Abort(comm, ierr);
		}

		//	Balance of A and B
		ierr = LAPACKE_zggbal( LAPACK_COL_MAJOR, 'P', nH, VtWG, nH, VtV, nH, &ilo, &ihi, lscale, rscale);
		BGMRESMDR->flops += flops_zggbal(nH);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRRITZDeflation:: LAPACKE_zggbal error %d\n", ierr);
			MPI_Abort(comm, ierr);
		}

#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank) PrintMatZ(VtV, nH, nH, nH, "VtV");
	MPI_Barrier(comm);
#endif
		//	Return to Generalized Hessenberg form
		//	//	Set work to unit
		memset(work, 0, nH * nH * sizeof(MYTYPE));
		BGMRESMDR->flops += flops_memset_z(nH * nH);
		//for(int j = 0; j < nH; j++){
		//	work[j + nH * j] = 1.;
		//}
		ierr = LAPACKE_zgghd3 (LAPACK_COL_MAJOR, 'N', 'I', nH, ilo, ihi, VtWG, nH, VtV, nH, NULL, nH, work, nH);
		BGMRESMDR->flops += flops_zgghd3(nH);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRRITZDeflation:: LAPACKE_zgghd3 error %d\n", ierr);
			if(!rank) PrintMatZ(work, nH, nH, nH, "work param 13");
			MPI_Abort(comm, ierr);
		}

		//	QZ algorithm
		ierr = LAPACKE_zhgeqz( LAPACK_COL_MAJOR, 'S', 'N', 'V', nH, ilo, ihi, VtWG, nH, VtV, nH, alphai, beta, NULL, 1, work, nH);
		BGMRESMDR->flops += flops_zhgeqz(nH);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRRITZDeflation:: LAPACKE_zhgeqz error %d\n", ierr);
			MPI_Abort(comm, ierr);
		}
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		for(int j = 0; j < nH; j++){
			if(!rank){
				printf("BGMRESMDRRITZDeflation:: EV[%d] = (%.6e + i %.6e)/(%.6e + i %.6e) \n", j, alphai[j].real, alphai[j].imag, beta[j].real, beta[j].imag);
			}
		}
		MPI_Barrier(comm);
#endif

		//	Select eigenspace of deflation
		/******************************************/
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank) printf("BGMRESMDRRITZDeflation:: Selecting eigenvalues\n");
	MPI_Barrier(comm);
#endif
		Perm		= (int*) malloc( nH * sizeof(int));
		select	= (int*) malloc( nH * sizeof(int));

		wr			=	(double*) malloc( nH * sizeof(double));

		//	Compute magnitude of EVs
		for(int i = 0; i < nH; i++){
			if(sqrt(beta[i].real * beta[i].real + beta[i].imag * beta[i].imag) < 1e-15){
				fprintf(stderr, "BGMRESMDRRITZDeflation:: Overflow beta[%d] = %.6e + i %.6e \n", i, beta[i].real, beta[i].imag);
				wr[i] = 1000.;
				//MPI_Abort(comm, 69);
			}else{
				wr[i] = sqrt(alphai[i].real * alphai[i].real + alphai[i].imag * alphai[i].imag)/sqrt(beta[i].real * beta[i].real + beta[i].imag * beta[i].imag);
#if BGMRESMDRDEBUG
				if(!rank){
					printf("BGMRESMDRRITZDeflation:: wr[%d] = %.6e \n", i, wr[i]);
				}
#endif
			}
		}

		for(unsigned int j = 0; j < nH; j++){ Perm[j]		= j; }
		for(unsigned int j = 0; j < nH; j++){ select[j]	= 0; }

		//	Get Perm to sort the real part of eigenvalues in ascending way
		quickSortDoubleWithPerm(&wr, nH, &Perm);

#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		for(int j = 0; j < nH; j++){
			if(!rank){
				printf("BGMRESMDRRITZDeflation:: EV[%d] = (%.6e + %.6e i)/(%.6e + i %.6e) \n", Perm[j], alphai[Perm[j]].real, alphai[Perm[j]].imag, beta[Perm[j]].real, beta[Perm[j]].imag);
			}
		}
		MPI_Barrier(comm);
#endif
		for(int j = 0; j < nH; j++){
			if(cpt < BGMRESMDR->MaxDefDim){
				cpt++;
				select[Perm[j]] = 1;
			}
		}
		/******************************************/
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRRITZDeflation:: Reordering Schur\n");
		MYTYPE temp;
		for(int i = 0; i < nH * nH; i++){
			temp.real = VtWG[i].real;
			temp.imag = VtWG[i].imag;
		}
		if(!rank)	printf("BGMRESMDRRITZDeflation:: VtWG ok\n");
		MPI_Barrier(comm);

		for(int i = 0; i < (nH) * (nH); i++){
			temp.real = VtV[i].real;
			temp.imag = VtV[i].imag;
		}
		if(!rank)	printf("BGMRESMDRRITZDeflation:: VtV ok\n");
		MPI_Barrier(comm);

		for(int i = 0; i < (nH) * (nH); i++){
			temp.real = work[i].real;
			temp.imag = work[i].imag;
		}
		if(!rank)	printf("BGMRESMDRRITZDeflation:: work ok\n");
		MPI_Barrier(comm);
#endif

    int counter = 1;
    for(int i = 0; i < nH; i++){
      if(select[i] == 1){
        int ifst = i + 1;
        int ilst = counter;
				ierr = LAPACKE_ztgexc ( LAPACK_COL_MAJOR, 0, 1, nH, VtWG, nH, VtV, nH, tau, 1, work, nH, ifst, ilst);
				BGMRESMDR->flops += flops_ztgexc(nH, ifst, ilst);
        if(ierr != 0){
          fprintf(stderr, "BGMRESMDRRITZDeflation:: LAPACKE_ztgexc error %d \n", ierr);
          MPI_Abort(comm, ierr);
        }
        counter++;
      }
    }
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRRITZDeflation:: Reordering Schur done\n");
		MPI_Barrier(comm);
#endif

		//	Get the schur vectors of the original problem
		ierr = LAPACKE_zggbak( LAPACK_COL_MAJOR, 'P', 'R', nH, ilo, ihi, lscale, rscale, cpt, work, nH);
		BGMRESMDR->flops += flops_zggbak(nH, cpt);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRRITZDeflation:: LAPACKE_zggbak error %d\n", ierr);
			MPI_Abort(comm, ierr);
		}

		//	Set the dimension of the deflation subspace
		BGMRESMDR->dimZ = BGMRESMDR->MaxDefDim;
		cpt = BGMRESMDR->MaxDefDim;

		/**********************
		*	Set up p_k and P_k	*
		*	Set up Z and AZ			*
		**********************/
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRRITZDeflation:: Set up p_k and P_k \n");
		MPI_Barrier(comm);
#endif

		//	Compute P_k = H * p_k
		const_alpha.real = 1.;
		const_beta.real= 0;
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, mH, cpt, nH, &const_alpha, H2, ldh, work, nH, &const_beta, H3, mH);
		BGMRESMDR->flops += flops_zgemm(mH, cpt, nH);
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRRITZDeflation:: Compute H  p_k done \n");
		MPI_Barrier(comm);
#endif

		//	QR factorization of P_k = Q_P R_P
		ierr = LAPACKE_zgeqrf (LAPACK_COL_MAJOR, mH, cpt, H3, mH, tau);
		BGMRESMDR->flops += flops_zgeqrf(mH, cpt);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRRITZDeflation:: LAPACKE_zgeqrf QR factorization of P_k = Q_P R_P error %d\n", ierr);
			MPI_Abort(comm, ierr);
		}
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRRITZDeflation:: Compute Q_k = P_k r^{-1} done \n");
		MPI_Barrier(comm);
#endif

		//	p_k = p_k R_P^{-1}
		cblas_ztrsm (CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, nH, cpt, &const_alpha, H3, mH, work, nH);
		BGMRESMDR->flops += flops_ztrsm(nH, cpt);
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRRITZDeflation:: Compute p_k = p_k r^{-1} done \n");
		MPI_Barrier(comm);
#endif

		//	Temp = Z * p_k
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ln, cpt, dimZ, &const_alpha, Z, ln, work, nH, &const_beta, workZ, ln);
		BGMRESMDR->flops += flops_zgemm(ln, cpt, dimZ);
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRRITZDeflation:: Compute Z_k = Z p_k done \n");
		MPI_Barrier(comm);
#endif

		//	Temp = Temp + V * p_k
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ln, cpt, nV, &const_alpha, V, ln, work + dimZ, nH, &const_alpha, workZ, ln);
		BGMRESMDR->flops += flops_zgemm_sum(ln, cpt, nV);
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRRITZDeflation:: Compute Z_k += V p_k(dimZ + 1 : nH, :)  done \n");
		MPI_Barrier(comm);
#endif

		//	[AZ V] = [AZ V] * Q_k
		ierr = LAPACKE_zunmqr (LAPACK_COL_MAJOR, 'R', 'N', ln, mH, cpt, H3, mH, tau, AZ, ln);
		BGMRESMDR->flops += flops_zgemm(ln, mH, cpt);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRRITZDeflation:: LAPACKE_zunmqr [AZ V] = [AZ V] * Q_k error %d\n", ierr);
			MPI_Abort(comm, ierr);
		}
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRRITZDeflation:: Compute AZ_k = [AZ V] P_k  done \n");
		MPI_Barrier(comm);
#endif

		//	Z = Temp
		memcpy(Z, workZ, cpt * ln * sizeof(MYTYPE));
		BGMRESMDR->flops += flops_memset_z(cpt * ln);
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRRITZDeflation:: Copy Z_k  done \n");
		MPI_Barrier(comm);
#endif

		//	Prepares Z' * Z
		cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, dimZ, dimZ, ln, &const_alpha, Z, ln, Z, ln, &const_beta, ZtZ, dimZ);
		BGMRESMDR->flops += flops_zgemm(dimZ, dimZ, ln);
		ierr = MPI_Allreduce(MPI_IN_PLACE, ZtZ, dimZ * dimZ, MPI_C_DOUBLE_COMPLEX, MPI_SUM, comm);
		BGMRESMDR->flops += flops_allreduce_z(dimZ * dimZ, BGMRESMDR->ND);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRRITZDeflation:: MPI_Allreduce Prepares Z' * Z error %d\n", ierr);
			MPI_Abort(comm, ierr);
		}
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRRITZDeflation:: Compute ZtZ  done \n");
		MPI_Barrier(comm);
#endif

		//	Set values of D
		for(int i = 0; i < dimZ; i++){
			D[i].real = 1. / sqrt(ZtZ[i + i * dimZ].real);
			D[i].imag = 0;
		}
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRRITZDeflation:: Compute D  done \n");
		MPI_Barrier(comm);
#endif

		//	Normalize columns of Z
		for(int j = 0; j < dimZ; j++){
			for(int i = 0; i < ln; i++){
				Z[i + j * ln].real = Z[i + j * ln].real * D[j].real;
				Z[i + j * ln].imag = Z[i + j * ln].imag * D[j].real;
			}
		}
		BGMRESMDR->flops += 2 * ln * dimZ;
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRRITZDeflation:: Update Z done \n");
		MPI_Barrier(comm);
#endif

		//	Update ZtZ
		for(int j = 0; j < dimZ; j++){
			for(int i = 0; i < dimZ; i++){
				ZtZ[i + j * dimZ].real = ZtZ[i + j * dimZ].real * D[j].real;
				ZtZ[i + j * dimZ].imag = ZtZ[i + j * dimZ].imag * D[j].real;
			}
		}
		for(int j = 0; j < dimZ; j++){
			for(int i = 0; i < dimZ; i++){
				ZtZ[j + i * dimZ].real = ZtZ[j + i * dimZ].real * D[j].real;
				ZtZ[j + i * dimZ].imag = ZtZ[j + i * dimZ].imag * D[j].real;
			}
		}
		BGMRESMDR->flops += 4 * dimZ * dimZ;
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRRITZDeflation:: Update ZtZ done \n");
		MPI_Barrier(comm);
#endif

		free(Perm);
		free(select);
		free(wr);
		free(alphar);
		free(alphai);
		free(lscale);
		free(rscale);
		free(beta);
		free(tau);
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRRITZDeflation:: Set up p_k and P_k done \n");
		MPI_Barrier(comm);
#endif
		timing_op += MPI_Wtime() - timing;
		BGMRESMDR->Timing.DeflationComputationTime += timing_op;
		BGMRESMDR->Timing.DeflationComputationComm += timing_comm;
	}

	return ierr;
}

/** \fn int BGMRESMDRHRITZDeflationSimple(BGMRESMDR_t* BGMRESMDR){
 * \brief First deflation based on Harmonic Ritz values approximation
 * \details Computes the eigenvalues of the GEVP: \f$ H' H u = \lambda \bar{H}' u \f$
 * \param BGMRESMDR The BGMRESMDR context
 * \remarks Supppsed no deflation vectors exist
 * \warning
*/
int BGMRESMDRHRITZDeflationSimple(BGMRESMDR_t* BGMRESMDR){

	double timing = MPI_Wtime();
	int ierr = 0;
	int rank = BGMRESMDR->rank;
	MPI_Comm comm = BGMRESMDR->comm;

	int dimZ = BGMRESMDR->dimZ;
	int iter = BGMRESMDR->iteration;
	int t    = BGMRESMDR->nrhs;

	MYTYPE* H		= BGMRESMDR->H;
	MYTYPE* H2	= BGMRESMDR->H2;
	MYTYPE* H3	= BGMRESMDR->H3;

	int* vidx = BGMRESMDR->VIdx;

	int nH	= vidx[iter];
	int mH	=	nH + t;
	int ldh = BGMRESMDR->MaxBasis + dimZ;

	//	Basis variables
	MYTYPE* V			= BGMRESMDR->V;
	MYTYPE* Z			= BGMRESMDR->Z;
	MYTYPE* AZ		= BGMRESMDR->AZ;
	MYTYPE* workZ	= BGMRESMDR->WorkZ;

	int ln				= BGMRESMDR->ln;

	MYTYPE* work	= BGMRESMDR->WorkH;

	//	GEVP variables
	MYTYPE* GtG		= BGMRESMDR->GtG;
	MYTYPE* VtV		=	BGMRESMDR->VtV;
	MYTYPE* ZtZ		= BGMRESMDR->ZtZ;
	MYTYPE* D			=	BGMRESMDR->D;

	//	scaling factors for balancing matrices
	double* lscale = NULL;
	double* rscale = NULL;

	//	vectors for eigenvalues
	double* alphar = NULL;
	MYTYPE* alphai = NULL;
	MYTYPE* beta   = NULL;

	//	vector for rotation scalars
	MYTYPE* tau    = NULL;

	double* wr = NULL;
	MYTYPE* wi = NULL;

	//	vectors for selection of eigenvalues
	int* Perm = NULL;
	int* select = NULL;
	int* ifailr = NULL;

	int cpt = 0;

	MYTYPE const_alpha = {1., 0};
	MYTYPE const_beta = {0, 0};

	//	Prepare GtG The A matrix in GEVP
	cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nH, nH, mH, &const_alpha, H2, ldh, H2, ldh, &const_beta, GtG, nH);
	BGMRESMDR->flops += flops_zgemm(nH, nH, mH);

	//	Prepare VtV	The B matrix in GEVP
	for(int j = 0; j < nH; j++){
		for(int i = 0; i < nH; i++){
			VtV[j + i * nH].real = H2[i + j * ldh].real;
			VtV[j + i * nH].imag = -H2[i + j * ldh].imag;
		}
	}
	BGMRESMDR->flops += 2 * nH * nH;

	//	rotation scalars
	tau			= (MYTYPE*) malloc( mH * sizeof(MYTYPE));

	//	scaling factors for balancing matrices
	lscale	=	(double*) malloc( nH * sizeof(double));
	rscale	=	(double*) malloc( nH * sizeof(double));

	//	eigenvalues
	alphar	= (double*) malloc( nH * sizeof(double));
	alphai	= (MYTYPE*) malloc( nH * sizeof(MYTYPE));
	beta  	= (MYTYPE*) malloc( nH * sizeof(MYTYPE));

	int ilo = 1;
	int ihi = nH;

	//	QR factorization of B
	ierr = LAPACKE_zgeqrf (LAPACK_COL_MAJOR, nH, nH, VtV, nH, tau);
	BGMRESMDR->flops += flops_zgeqrf(nH, nH);
	if(ierr != 0){
		fprintf(stderr, "BGMRESMDRHRITZDeflationSimple:: LAPACKE_dgeqrf error %d\n", ierr);
		MPI_Abort(comm, ierr);
	}

	//	Apply Q on GtG
	ierr = LAPACKE_zunmqr (LAPACK_COL_MAJOR, 'L', 'C', nH, nH, nH, VtV, nH, tau, GtG, nH);
	BGMRESMDR->flops += flops_zgemm(nH, nH, nH);
	if(ierr != 0){
		fprintf(stderr, "BGMRESMDRHRITZDeflationSimple:: LAPACKE_zunmqr error %d\n", ierr);
		MPI_Abort(comm, ierr);
	}

	//	Balance of A and B
	ierr = LAPACKE_zggbal( LAPACK_COL_MAJOR, 'P', nH, GtG, nH, VtV, nH, &ilo, &ihi, lscale, rscale);
	if(ierr != 0){
		fprintf(stderr, "BGMRESMDRHRITZDeflationSimple:: LAPACKE_zggbal error %d\n", ierr);
		MPI_Abort(comm, ierr);
	}

	//	Return to Generalized Hessenberg form
	ierr = LAPACKE_zgghrd (LAPACK_COL_MAJOR, 'N', 'I', nH, ilo, ihi, GtG, nH, VtV, nH, NULL, 1, work, nH);
	if(ierr != 0){
		fprintf(stderr, "BGMRESMDRHRITZDeflationSimple:: LAPACKE_zgghrd error %d\n", ierr);
		MPI_Abort(comm, ierr);
	}

	//	QZ algorithm
	ierr = LAPACKE_zhgeqz( LAPACK_COL_MAJOR, 'S', 'N', 'V', nH, ilo, ihi, GtG, nH, VtV, nH, alphai, beta, NULL, 1, work, nH);
	BGMRESMDR->flops += flops_zhgeqz(nH);
	if(ierr != 0){
		fprintf(stderr, "BGMRESMDRHRITZDeflationSimple:: LAPACKE_zhgeqz error %d\n", ierr);
		MPI_Abort(comm, ierr);
	}
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	for(int j = 0; j < nH; j++){
		if(!rank){
			printf("BGMRESMDRHRITZDeflationSimple:: EV[%d] = (%.6e + %.6e i)/(%.6e + i %.6e) \n", j, alphai[j].real, alphai[j].imag, beta[j].real, beta[j].imag);
		}
	}
	MPI_Barrier(comm);
#endif

		//	Select eigenspace of deflation
		/******************************************/
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank) printf("BGMRESMDRHRITZDeflationSimple:: Selecting eigenvalues\n");
	MPI_Barrier(comm);
#endif
	Perm		= (int*) malloc( nH * sizeof(int));
	select	= (int*) malloc( nH * sizeof(int));

	wr			=	(double*) malloc( nH * sizeof(double));

		//	Compute magnitude of EVs
		for(int i = 0; i < nH; i++){
			if(sqrt(beta[i].real * beta[i].real + beta[i].imag * beta[i].imag) < 1e-15){
				fprintf(stderr, "BGMRESMDRHRITZDeflationSimple:: Overflow beta[%d] = %.6e + i %.6e \n", i, beta[i].real, beta[i].imag);
				MPI_Abort(comm, 69);
			}
			wr[i] = sqrt(alphai[i].real * alphai[i].real + alphai[i].imag * alphai[i].imag)/sqrt(beta[i].real * beta[i].real + beta[i].imag * beta[i].imag);
#if BGMRESMDRDEBUG
			if(!rank){
				printf("BGMRESMDRHRITZDeflationSimple:: wr[%d] = %.6e \n", i, wr[i]);
			}
#endif
		}

	for(unsigned int j = 0; j < nH; j++){ Perm[j]		= j; }
	for(unsigned int j = 0; j < nH; j++){ select[j]	= 0; }

	//	Get Perm to sort the real part of eigenvalues in ascending way
	quickSortDoubleWithPerm(&wr, nH, &Perm);
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	for(int j = 0; j < nH; j++){
		if(!rank){
			printf("BGMRESMDRHRITZDeflationSimple:: EV[%d] = (%.6e + %.6e i)/(%.6e + i %.6e) \n", Perm[j], alphai[Perm[j]].real, alphai[Perm[j]].imag, beta[Perm[j]].real, beta[Perm[j]].imag);
		}
	}
	MPI_Barrier(comm);
#endif
	for(int j = 0; j < nH; j++){
		if(cpt < BGMRESMDR->MaxDefDim){
			cpt++;
			select[Perm[j]] = 1;
		}
	}
	/******************************************/

#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank)	printf("BGMRESMDRHRITZDeflationSimple:: Reordering Schur\n");
	MYTYPE temp;
	for(int i = 0; i < nH * nH; i++){
		temp.real = GtG[i].real;
		temp.imag = GtG[i].imag;
	}
	if(!rank)	printf("BGMRESMDRHRITZDeflationSimple:: VtWG ok\n");
	MPI_Barrier(comm);

	for(int i = 0; i < (nH) * (nH); i++){
		temp.real = VtV[i].real;
		temp.imag = VtV[i].imag;
	}
	if(!rank)	printf("BGMRESMDRHRITZDeflationSimple:: VtV ok\n");
	MPI_Barrier(comm);

	for(int i = 0; i < (nH) * (nH); i++){
		temp.real = work[i].real;
		temp.imag = work[i].imag;
	}
	if(!rank)	printf("BGMRESMDRHRITZDeflationSimple:: work ok\n");
	MPI_Barrier(comm);
#endif

  int counter = 1;
  for(int i = 0; i < nH; i++){
    if(select[i] == 1){
      int ifst = i + 1;
      int ilst = counter;
			ierr = LAPACKE_ztgexc ( LAPACK_COL_MAJOR, 0, 1, nH, GtG, nH, VtV, nH, tau, 1, work, nH, ifst, ilst);
			BGMRESMDR->flops += flops_ztgexc(nH, ifst, ilst);
      if(ierr != 0){
				fprintf(stderr, "BGMRESMDRHRITZDeflationSimple:: LAPACKE_dtgexc error %d \n", ierr);
        MPI_Abort(comm, ierr);
      }
      counter++;
    }
  }

	if(cpt != BGMRESMDR->MaxDefDim){
		cpt = BGMRESMDR->MaxDefDim;
		fprintf(stdout, "BGMRESMDRHRITZDeflationSimple:: LAPACKE_dtgsen attention : The number of returned subspace = %d  doesn't correspond to the number of deflation subspace = %d \n", cpt, BGMRESMDR->MaxDefDim);
		//MPI_Abort(comm, -cpt);
	}
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank)	printf("BGMRESMDRHRITZDeflationSimple:: Reordering Schur done\n");
	MPI_Barrier(comm);
#endif

	//	Get the schur vectors of the original problem
	ierr = LAPACKE_zggbak( LAPACK_COL_MAJOR, 'P', 'R', nH, ilo, ihi, lscale, rscale, cpt, work, nH);
	BGMRESMDR->flops += flops_zggbak(nH, cpt);
	if(ierr != 0){
		fprintf(stderr, "BGMRESMDRHRITZDeflationSimple:: LAPACKE_zggbak error %d\n", ierr);
		MPI_Abort(comm, ierr);
	}

	//	Set the dimension of the deflation subspace
	BGMRESMDR->dimZ = BGMRESMDR->MaxDefDim;
	cpt = BGMRESMDR->MaxDefDim;
	dimZ = cpt;

	/**********************
	*	Set up p_k and P_k	*
	*	Set up Z and AZ			*
	**********************/
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank)	printf("BGMRESMDRHRITZDeflationSimple:: Set up p_k and P_k \n");
	MPI_Barrier(comm);
#endif

	//	Compute P_k = H * p_k
	cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, mH, cpt, nH, &const_alpha, H2, ldh, work, nH, &const_beta, H3, mH);
	BGMRESMDR->flops += flops_zgemm(mH, cpt, nH);
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank)	printf("BGMRESMDRHRITZDeflationSimple:: Compute H  p_k done \n");
	MPI_Barrier(comm);
#endif

		//	QR factorization of P_k = Q_P R_P
	ierr = LAPACKE_zgeqrf (LAPACK_COL_MAJOR, mH, cpt, H3, mH, tau);
	BGMRESMDR->flops += flops_zgeqrf(mH, cpt);
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank)	printf("BGMRESMDRHRITZDeflationSimple:: Compute Q_k = P_k r^{-1} done  \n");
	MPI_Barrier(comm);
#endif

	//	p_k = p_k R_P^{-1}
	cblas_ztrsm (CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, nH, cpt, &const_alpha, H3, mH, work, nH);
	BGMRESMDR->flops += flops_ztrsm(nH, cpt);
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank)	printf("BGMRESMDRHRITZDeflationSimple:: Compute p_k = p_k r^{-1} done \n");
	MPI_Barrier(comm);
#endif

	//	Normalize p_k and set D
	double norm_pk = 0;
	for(int j = 0; j < cpt; j++){
		norm_pk = cblas_dznrm2(nH, work + j * nH, 1);
		D[j].real = 1./norm_pk;
		D[j].imag = 0;
		for(int i = 0; i < nH; i++){
			work[ j * nH + i].real = work[ j * nH + i].real * D[j].real;
			work[ j * nH + i].imag = work[ j * nH + i].imag * D[j].real;
		}
	}
	BGMRESMDR->flops += 6 * cpt * nH;

	//	Prepares Z' * Z
	cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, dimZ, dimZ, nH, &const_alpha, work, nH, work, nH, &const_beta, ZtZ, dimZ);
	BGMRESMDR->flops += flops_zgemm(dimZ, dimZ, nH);
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank)	printf("BGMRESMDRHRITZDeflationSimple:: Compute ZtZ  done \n");
	MPI_Barrier(comm);
#endif

	//	Temp = V * p_k
	cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ln, cpt, nH, &const_alpha, V, ln, work, nH, &const_beta, Z, ln);
	BGMRESMDR->flops += flops_zgemm(ln, cpt, nH);
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);

	if(!rank)	printf("BGMRESMDRHRITZDeflationSimple:: Compute Z_k = V p_k(1 : nH, :)  done \n");
	MPI_Barrier(comm);
#endif

	//	[AZ V] = [AZ V] * Q_k
	ierr = LAPACKE_zunmqr (LAPACK_COL_MAJOR, 'R', 'N', ln, mH, cpt, H3, mH, tau, V, ln);
	BGMRESMDR->flops += flops_zgemm(ln, mH, cpt);
	if(ierr != 0){
		fprintf(stderr, "BGMRESMDRHRITZDeflationSimple:: zunmqr error %d \n", ierr);
		MPI_Abort(comm, 69);
	}
	//	Copy the result to AZ
	memcpy(AZ, V, ln * cpt * sizeof(MYTYPE));
	BGMRESMDR->flops += flops_memset_z(cpt * ln);
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank)	printf("BGMRESMDRHRITZDeflationSimple:: Compute AZ_k = [AZ V] P_k  done \n");
	MPI_Barrier(comm);
#endif

	free(Perm);
	free(select);
	free(wr);
	free(alphar);
	free(alphai);
	free(lscale);
	free(rscale);
	free(beta);
	free(tau);
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank)	printf("BGMRESMDRHRITZDeflationSimple:: Set up p_k and P_k done \n");
	MPI_Barrier(comm);
#endif

	BGMRESMDR->Timing.DeflationComputationTime += MPI_Wtime() - timing;

	return ierr;
}


/** \fn int BGMRESMDRHRITZDeflation(BGMRESMDR_t* BGMRESMDR){
 * \brief Update deflation vectors based on Harmonic Ritz values approximation
 * \details Computes the eigenvalues of the GEVP:  \f$ G' W V u = \lambda G' G u \f$
 * \param BGMRESMDR The BGMRESMDR context
 * \remarks The following relation is supposed to hold \f$ A V = W G \f$, where \f$ V = [Z, \mathcal{V}_k], W = [\tilde{Z}, \mathcal{V}_{k+1}] \f$ and \f$ G = [D, \tilde{Z}' \mathcal{V}_k;  0, H_k] \f$
 * \warning
*/
int BGMRESMDRHRITZDeflation(BGMRESMDR_t* BGMRESMDR){
	int ierr = 0;
	int dimZ = BGMRESMDR->dimZ;
	if(dimZ == 0){
		ierr = BGMRESMDRHRITZDeflationSimple(BGMRESMDR);
	}else{

		double timing = MPI_Wtime();
		int rank		= BGMRESMDR->rank;
		MPI_Comm comm = BGMRESMDR->comm;
		int iter		= BGMRESMDR->iteration;
		int t       = BGMRESMDR->nrhs;

		MYTYPE* H		= BGMRESMDR->H;
		MYTYPE* H2	= BGMRESMDR->H2;
		MYTYPE* H3	= BGMRESMDR->H3;

		int* vidx = BGMRESMDR->VIdx;
		int mH			= vidx[iter] + dimZ + t;
		int nH			= vidx[iter] + dimZ;
		int ldh			= BGMRESMDR->MaxBasis + dimZ;

		//	Basis variables
		MYTYPE* V			= BGMRESMDR->V;
		MYTYPE* Z			= BGMRESMDR->Z;
		MYTYPE* AZ		= BGMRESMDR->AZ;
		MYTYPE* workZ	= BGMRESMDR->WorkZ;

		int ln				= BGMRESMDR->ln;
		int nV        = vidx[iter];

		MYTYPE* work	= BGMRESMDR->WorkH;

		//	GEVP variables
		MYTYPE* GtG		= BGMRESMDR->GtG;
		MYTYPE* GtWtV	=	BGMRESMDR->VtV;
		MYTYPE* ZtZ		= BGMRESMDR->ZtZ;
		MYTYPE* D			=	BGMRESMDR->D;

		//	scaling factors for balancing matrices
		double* lscale = NULL;
		double* rscale = NULL;

		//	vectors for eigenvalues
		double* alphar = NULL;
		MYTYPE* alphai = NULL;
		MYTYPE* beta   = NULL;

		//	vector for rotation scalars
		MYTYPE* tau    = NULL;

		double* wr = NULL;
		MYTYPE* wi = NULL;

		//	vectors for selection of eigenvalues
		int* Perm = NULL;
		int* select = NULL;
		int* ifailr = NULL;

		int cpt = 0;

		int ilo = 1;
		int ihi = nH;

		tau			= (MYTYPE*) malloc( nH * sizeof(MYTYPE));
		lscale	= (double*) malloc( nH * sizeof(double));
		rscale	= (double*) malloc( nH * sizeof(double));

		alphar	= (double*) malloc( nH * sizeof(double));
		alphai	= (MYTYPE*) malloc( nH * sizeof(MYTYPE));
		beta  	= (MYTYPE*) malloc( nH * sizeof(MYTYPE));

		MYTYPE const_alpha = {1., 0};
		MYTYPE const_beta = {0, 0};

		//	Prepare GtG The A matrix in GEVP
		cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nH, nH, mH, &const_alpha, H2, ldh, H2, ldh, &const_beta, GtG, nH);
		BGMRESMDR->flops += flops_zgemm(nH, nH, mH);

		//	Prepatre W^TV
			//	work_i = W^T * Z
		cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, mH, dimZ, ln, &const_alpha, AZ, ln, Z, ln, &const_beta, work, mH);
		BGMRESMDR->flops += flops_zgemm(mH, dimZ, ln);
		BGMRESMDR->Timing.DeflationComputationTime += MPI_Wtime() - timing;

			//	work = sum_i work_i
		timing = MPI_Wtime();
		ierr = MPI_Allreduce(MPI_IN_PLACE, work, mH * dimZ, MPI_C_DOUBLE_COMPLEX, MPI_SUM, comm);
		BGMRESMDR->flops += flops_allreduce_z(dimZ * mH, BGMRESMDR->ND);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRHRITZDeflation:: MPI_Allreduce work = sum_i work_i error %d\n", ierr);
			MPI_Abort(comm, ierr);
		}
		BGMRESMDR->Timing.DeflationComputationComm += MPI_Wtime() - timing;

			//	G^t * W^t * Z
		timing = MPI_Wtime();
		cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nH, dimZ, mH, &const_alpha, H2, ldh, work, mH, &const_beta, GtWtV, nH);
		BGMRESMDR->flops += flops_zgemm(nH, dimZ, mH);

			//	G' * [0_{dimz, nH-dimz}; I_{nH - dimz}; 0_{1, nH - dimz}] (i.e., copy \bar{H}^t to the place of the Identity)
		for(int j = 0; j < nH - dimZ; j++){
			for(int i = 0; i < nH - dimZ; i++){
				GtWtV[dimZ * nH + dimZ + j * nH + i].real = H2[dimZ * ldh + dimZ + i * ldh + j].real;
				GtWtV[dimZ * nH + dimZ + j * nH + i].imag = -H2[dimZ * ldh + dimZ + i * ldh + j].imag;

#if BGMRESMDRDEBUG
				if(dimZ * nH + dimZ + j * nH + i < 0 || dimZ * nH + dimZ + j * nH + i > nH * nH - 1){
					fprintf(stderr, "BGMRESMDRHRITZDeflation:: You are depassing GtWtV matrix area\n");
					MPI_Abort(comm, 69);
				}
				if(dimZ * ldh + dimZ + i * ldh + j < 0 || dimZ * ldh + dimZ + i * ldh + j > ldh * nH - 1){
					fprintf(stderr, "BGMRESMDRHRITZDeflation:: You are depassing H2 matrix area \n");
					MPI_Abort(comm, 69);
				}
#endif
			}
		}
		BGMRESMDR->flops += 2 * (nH - dimZ) * (nH - dimZ);

		/**************************************************/
		/*	Ready for the generalized eigenvalue problem	*/
		/**************************************************/

		//	QR factorization of G' * W' * V
		ierr = LAPACKE_zgeqrf (LAPACK_COL_MAJOR, nH, nH, GtWtV, nH, tau);
		BGMRESMDR->flops += flops_zgeqrf(nH, nH);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRHRITZDeflation:: LAPACKE_dgeqrf error %d\n", ierr);
			MPI_Abort(comm, ierr);
		}

		//	Apply Q^t on GtG
		ierr = LAPACKE_zunmqr (LAPACK_COL_MAJOR, 'L', 'C', nH, nH, nH, GtWtV, nH, tau, GtG, nH);
		BGMRESMDR->flops += flops_zgemm(nH, nH, nH);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRHRITZDeflation:: LAPACKE_zunmqr error %d\n", ierr);
			MPI_Abort(comm, ierr);
		}

		//	Balance of A and B
		ierr = LAPACKE_zggbal( LAPACK_COL_MAJOR, 'P', nH, GtG, nH, GtWtV, nH, &ilo, &ihi, lscale, rscale);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRHRITZDeflation:: LAPACKE_zggbal error %d\n", ierr);
			MPI_Abort(comm, ierr);
		}

		memset(work, 0, nH * nH * sizeof(MYTYPE));
		BGMRESMDR->flops += flops_memset_z(nH * nH);
		//	Return to Generalized Hessenberg form
		ierr = LAPACKE_zgghd3 (LAPACK_COL_MAJOR, 'N', 'I', nH, ilo, ihi, GtG, nH, GtWtV, nH, NULL, 1, work, nH);
		BGMRESMDR->flops += flops_zgghd3(nH);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRHRITZDeflation:: LAPACKE_zgghd3 error %d\n", ierr);
			MPI_Abort(comm, ierr);
		}

		//	QZ algorithm
		ierr = LAPACKE_zhgeqz( LAPACK_COL_MAJOR, 'S', 'N', 'V', nH, ilo, ihi, GtG, nH, GtWtV, nH, alphai, beta, NULL, 1, work, nH);
		BGMRESMDR->flops += flops_zhgeqz(nH);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRHRITZDeflation:: LAPACKE_zhgeqz error %d\n", ierr);
			MPI_Abort(comm, ierr);
		}
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		for(int j = 0; j < nH; j++){
			if(!rank){
				printf("BGMRESMDRHRITZDeflation:: EV[%d] = (%.6e + %.6e i)/(%.6e + i %.6e) \n", j, alphai[j].real, alphai[j].imag, beta[j].real, beta[j].imag);
			}
		}
		MPI_Barrier(comm);
#endif

		//	Select eigenspace of deflation
		/******************************************/
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank) printf("BGMRESMDRHRITZDeflation:: Selecting eigenvalues\n");
	MPI_Barrier(comm);
#endif
		Perm		= (int*) malloc( nH * sizeof(int));
		select	= (int*) malloc( nH * sizeof(int));

		wr			=	(double*) malloc( nH * sizeof(double));

		//	Compute magnitude of EVs
		for(int i = 0; i < nH; i++){
			if(sqrt(beta[i].real * beta[i].real + beta[i].imag * beta[i].imag) < 1e-15){
				fprintf(stderr, "BGMRESMDRHRITZDeflation:: Overflow beta[%d] = %.6e + i %.6e \n", i, beta[i].real, beta[i].imag);
				wr[i] = 1000.;
				//MPI_Abort(comm, 69);
			}else{
				wr[i] = sqrt(alphai[i].real * alphai[i].real + alphai[i].imag * alphai[i].imag)/sqrt(beta[i].real * beta[i].real + beta[i].imag * beta[i].imag);
#if BGMRESMDRDEBUG
				if(!rank){
					printf("BGMRESMDRHRITZDeflation:: wr[%d] = %.6e \n", i, wr[i]);
				}
#endif
			}
		}

		for(unsigned int j = 0; j < nH; j++){ Perm[j]		= j; }
		for(unsigned int j = 0; j < nH; j++){ select[j]	= 0; }

		//	Get Perm to sort the real part of eigenvalues in ascending way
		quickSortDoubleWithPerm(&wr, nH, &Perm);

#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		for(int j = 0; j < nH; j++){
			if(!rank){
				printf("BGMRESMDRHRITZDeflation:: EV[%d] = (%.6e + %.6e i)/(%.6e + i %.6e) \n", Perm[j], alphai[Perm[j]].real, alphai[Perm[j]].imag, beta[Perm[j]].real, beta[Perm[j]].imag);
			}
		}
		MPI_Barrier(comm);
#endif
	for(int j = 0; j < nH; j++){
		if(cpt < BGMRESMDR->MaxDefDim){
			cpt++;
			select[Perm[j]] = 1;
		}
	}
		/******************************************/

    int counter = 1;
    for(int i = 0; i < nH; i++){
      if(select[i] == 1){
        int ifst = i + 1;
        int ilst = counter;
				ierr = LAPACKE_ztgexc ( LAPACK_COL_MAJOR, 0, 1, nH, GtG, nH, GtWtV, nH, tau, 1, work, nH, ifst, ilst);
				BGMRESMDR->flops += flops_ztgexc(nH, ifst, ilst);
        if(ierr != 0){
          fprintf(stderr, "BGMRESMDRHRITZDeflation:: LAPACKE_ztgexc error %d \n", ierr);
          MPI_Abort(comm, ierr);
        }
        counter++;
      }
    }

		cpt = BGMRESMDR->MaxDefDim;
		//	Reodering of the Schur form
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRHRITZDeflation:: LAPACKE_dtgsen error %d consult the documentation of mkl to understand what happened \n", ierr);
			MPI_Abort(comm, ierr);
		}
		if(cpt != BGMRESMDR->MaxDefDim){
			fprintf(stdout, "BGMRESMDRHRITZDeflation:: LAPACKE_dtgsen attention : The number of returned subspace = %d  doesn't correspond to the number of deflation subspace = %d \n", cpt, BGMRESMDR->MaxDefDim);
			MPI_Abort(comm, -cpt);
		}
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRHRITZDeflation:: Reordering Schur done\n");
		MPI_Barrier(comm);
#endif

		//	Get the schur vectors of the original problem
		ierr = LAPACKE_zggbak( LAPACK_COL_MAJOR, 'P', 'R', nH, ilo, ihi, lscale, rscale, cpt, work, nH);
		BGMRESMDR->flops += flops_zggbak(nH, cpt);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRHRITZDeflation:: LAPACKE_zggbak error %d\n", ierr);
			MPI_Abort(comm, ierr);
		}

		//	Set the dimension of the deflation subspace
		BGMRESMDR->dimZ = BGMRESMDR->MaxDefDim;
		cpt = BGMRESMDR->MaxDefDim;

		/**********************
		*	Set up p_k and P_k	*
		*	Set up Z and AZ			*
		**********************/
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRHRITZDeflation:: Set up p_k and P_k \n");
		MPI_Barrier(comm);
#endif

		//	Compute P_k = H * p_k
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, mH, cpt, nH, &const_alpha, H2, ldh, work, nH, &const_beta, H3, mH);
		BGMRESMDR->flops += flops_zgemm(mH, cpt, nH);
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRHRITZDeflation:: Compute H  p_k done \n");
		MPI_Barrier(comm);
#endif

		//	QR factorization of P_k = Q_P R_P
		ierr = LAPACKE_zgeqrf (LAPACK_COL_MAJOR, mH, cpt, H3, mH, tau);
		BGMRESMDR->flops += flops_zgeqrf(mH, cpt);
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRHRITZDeflation:: Compute Q_k = P_k r^{-1} done \n");
		MPI_Barrier(comm);
#endif

		//	p_k = p_k R_P^{-1}
		cblas_ztrsm (CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, nH, cpt, &const_alpha, H3, mH, work, nH);
		BGMRESMDR->flops += flops_ztrsm(nH, cpt);
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRHRITZDeflation:: Compute p_k = p_k r^{-1} done \n");
		MPI_Barrier(comm);
#endif

		//	Temp = Z * p_k
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ln, cpt, dimZ, &const_alpha, Z, ln, work, nH, &const_beta, workZ, ln);
		BGMRESMDR->flops += flops_zgemm(ln, cpt, dimZ);
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRHRITZDeflation:: Compute Z_k = Z p_k done \n");
		MPI_Barrier(comm);
#endif

		//	Temp = Temp + V * p_k
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ln, cpt, nV, &const_alpha, V, ln, work + dimZ, nH, &const_alpha, workZ, ln);
		BGMRESMDR->flops += flops_zgemm_sum(ln, cpt, nV);
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRHRITZDeflation:: Compute Z_k += V p_k(dimZ + 1 :nV + dimZ, :)  done \n");
		MPI_Barrier(comm);
#endif

		//	[AZ V] = [AZ V] * Q_k
		ierr = LAPACKE_zunmqr (LAPACK_COL_MAJOR, 'R', 'N', ln, mH, cpt, H3, mH, tau, AZ, ln);
		BGMRESMDR->flops += flops_zgemm(ln, mH, cpt);
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRHRITZDeflation:: Compute AZ_k = [AZ V] P_k  done \n");
		MPI_Barrier(comm);
#endif

		//	Z = Temp
		memcpy(Z, workZ, cpt * ln * sizeof(MYTYPE));
		BGMRESMDR->flops += flops_memset_z(cpt * ln);
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRHRITZDeflation:: Copy Z_k  done \n");
		MPI_Barrier(comm);
#endif

		//	Prepares Z' * Z
		cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, dimZ, dimZ, ln, &const_alpha, Z, ln, Z, ln, &const_beta, ZtZ, dimZ);
		BGMRESMDR->flops += flops_zgemm(dimZ, dimZ, ln);
		ierr = MPI_Allreduce(MPI_IN_PLACE, ZtZ, dimZ * dimZ, MPI_C_DOUBLE_COMPLEX, MPI_SUM, comm);
		BGMRESMDR->flops += flops_allreduce_z(dimZ * dimZ, BGMRESMDR->ND);
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRHRITZDeflation:: Compute ZtZ  done \n");
		MPI_Barrier(comm);
#endif

		//	Set values of D
		for(int i = 0; i < dimZ; i++){
			D[i].real = 1. / sqrt(ZtZ[i + i * dimZ].real);
			D[i].imag = 0;
		}
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRHRITZDeflation:: Compute D  done \n");
		MPI_Barrier(comm);
#endif

		//	Normalize columns of Z
		for(int j = 0; j < dimZ; j++){
			for(int i = 0; i < ln; i++){
				Z[i + j * ln].real = Z[i + j * ln].real * D[j].real;
				Z[i + j * ln].imag = Z[i + j * ln].imag * D[j].real;
			}
		}
		BGMRESMDR->flops += 2 * ln * dimZ;
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRHRITZDeflation:: Update Z done \n");
		MPI_Barrier(comm);
#endif

		//	Update ZtZ
		for(int j = 0; j < dimZ; j++){
			for(int i = 0; i < dimZ; i++){
				ZtZ[i + j * dimZ].real = ZtZ[i + j * dimZ].real * D[j].real;
				ZtZ[i + j * dimZ].imag = ZtZ[i + j * dimZ].imag * D[j].real;
			}
		}
		for(int j = 0; j < dimZ; j++){
			for(int i = 0; i < dimZ; i++){
				ZtZ[j + i * dimZ].real = ZtZ[j + i * dimZ].real * D[j].real;
				ZtZ[j + i * dimZ].imag = ZtZ[j + i * dimZ].imag * D[j].real;
			}
		}
		BGMRESMDR->flops += 4 * dimZ * dimZ;
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRHRITZDeflation:: Update ZtZ done \n");
		MPI_Barrier(comm);
#endif

		free(Perm);
		free(select);
		free(wr);
		free(alphar);
		free(alphai);
		free(lscale);
		free(rscale);
		free(beta);
		free(tau);
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank)	printf("BGMRESMDRHRITZDeflation:: Set up p_k and P_k done \n");
		MPI_Barrier(comm);
#endif

		BGMRESMDR->Timing.DeflationComputationTime += MPI_Wtime() - timing;

	}
	return ierr;
}

/** \fn int BGMRESMDRSVDDeflation(BGMRESMDR_t* BGMRESMDR){
 * \brief Update the deflation vectors based on the approximation of singular values
 * \details Compute the eigenvalues of the SGEVP \f$ G' G u = \lambda V' V u \f$
 * \param BGMRESMDR The BGMRESMDR context
 * \remarks The following relation is supposed to hold \f$ A V = W G \f$, where \f$ V = [Z, \mathcal{V}_k], W = [\tilde{Z}, \mathcal{V}_{k+1}] \f$ and \f$ \f$ G = [D, \tilde{Z}' \mathcal{V}_k; 0, H_k] \f$
 * \warning
*/
int BGMRESMDRSVDDeflation(BGMRESMDR_t* BGMRESMDR){
	int ierr = 0;
	MPI_Comm comm = BGMRESMDR->comm;
	int rank			= BGMRESMDR->rank;

	int dimZ = BGMRESMDR->dimZ;
	int iter = BGMRESMDR->iteration;

	int t = BGMRESMDR->nrhs;
	int* vidx = BGMRESMDR->VIdx;
	//	Hessenberg matrices
	MYTYPE* H = BGMRESMDR->H;
	MYTYPE* H2 = BGMRESMDR->H2;
	MYTYPE* H3 = BGMRESMDR->H3;

	//	Hessenberg dimensions
	int nH = dimZ + vidx[iter];
	int mH = dimZ + vidx[iter] + t;
	int ldh = BGMRESMDR->MaxBasis + dimZ;

	//	Basis vectors
	MYTYPE* V				= BGMRESMDR->V;
	MYTYPE* Z				= BGMRESMDR->Z;
	MYTYPE* AZ			= BGMRESMDR->AZ;
	MYTYPE* workZ		= BGMRESMDR->WorkZ;

	//	Basis dimension
	int ln					= BGMRESMDR->ln;
	int nV = vidx[iter];

	//	work space
	MYTYPE* work		= BGMRESMDR->WorkH;

	//	Generalized EVP variables
	MYTYPE* VtV			= BGMRESMDR->VtV;
	MYTYPE* GtG			= BGMRESMDR->GtG;
	MYTYPE* ZtZ			= BGMRESMDR->ZtZ;
	MYTYPE* D				= BGMRESMDR->D;

	double* alphar = NULL;
	MYTYPE* alphai = NULL;
	MYTYPE* beta   = NULL;
	MYTYPE* tau    = NULL;
	double* wr     = NULL;
	double* d			 = NULL;
	double* e			 = NULL;
	MYTYPE* VP		 = NULL;

	int* isplit    = NULL;
	int* iblock    = NULL;
	int* ifailv    = NULL;


	int* suppz		 = NULL;
	int* Perm			 = NULL;
	int* select		 = NULL;

	int cpt = 0;

	MYTYPE const_alpha = {1., 0};
	MYTYPE const_beta = {0, 0};
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank) printf("BGMRESMDRSVDDeflation:: iter = %d\n", iter);
	MPI_Barrier(comm);
#endif

	if(dimZ == 0){
		ierr = BGMRESMDRSVDDeflationSimple(BGMRESMDR);
	}else{
		double Timing = MPI_Wtime();
		/************************************************************************************
		 *                           Begin of Generalized EVP case                          *
		 ************************************************************************************/
		 /********************************
		 *	Prepare  B = [Z V]' * [Z V]	 *
		 ********************************/

		//	Compute V' * Z
		cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nV, dimZ, ln, &const_alpha, V, ln, Z, ln, &const_beta, work, nV);
		BGMRESMDR->Timing.DeflationComputationTime += MPI_Wtime() - Timing;
		BGMRESMDR->flops += flops_zgemm(nV, dimZ, ln);

		Timing = MPI_Wtime();
		ierr = MPI_Allreduce(MPI_IN_PLACE, work, nV * dimZ, MPI_C_DOUBLE_COMPLEX, MPI_SUM, comm);
		BGMRESMDR->flops += flops_allreduce_z(dimZ * nV, BGMRESMDR->ND);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRSVDDeflation::MPI_Allreduce error %d\n", ierr);
			MPI_Abort(comm, ierr);
		}
		BGMRESMDR->Timing.DeflationComputationComm += MPI_Wtime() - Timing;

			//	Copy V' * Z
		Timing = MPI_Wtime();
		ierr = LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', nV, dimZ, work, nV, VtV + dimZ, dimZ + nV);
		BGMRESMDR->flops += flops_copy_z(nV, dimZ);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRSVDDeflation::LAPACKE_dlacpy error %d\n", ierr);
			MPI_Abort(comm, ierr);
		}
			//	Copy Z' * V
		for(int j = 0; j < dimZ; j++){
			for(int i = 0; i < nV; i++){
				VtV[dimZ * (dimZ + nV) + i * (dimZ + nV) + j].real = work[ j * nV + i ].real;
				VtV[dimZ * (dimZ + nV) + i * (dimZ + nV) + j].imag = -1. * work[ j * nV + i ].imag;
			}
		}
		BGMRESMDR->flops += 2 * nV * dimZ;

			//	Copy of  Z' * Z
		ierr = LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', dimZ, dimZ, ZtZ, dimZ, VtV, dimZ + nV);
		BGMRESMDR->flops += flops_copy_z(dimZ, dimZ);

			//	Set V' * V
		for(int j = 0; j < nV; j++){
			for(int i = 0; i < nV; i++){
				VtV[ (dimZ + j) * (dimZ + nV) + dimZ + i].real = (i == j) ? 1. : 0;
				VtV[ (dimZ + j) * (dimZ + nV) + dimZ + i].imag = 0;
			}
		}
		BGMRESMDR->flops += 2 * nV * nV;
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank) printf("BGMRESMDRSVDDeflation::Preparation of VtV well passed\n");
		MPI_Barrier(comm);
#endif
		 /********************************/

		tau			= (MYTYPE*) malloc((dimZ + nV + 1) * sizeof(MYTYPE));
		alphar	= (double*) malloc((dimZ + nV) * sizeof(double));
		alphai	= (MYTYPE*) malloc((dimZ + nV) * sizeof(MYTYPE));
		beta  	= (MYTYPE*) malloc((dimZ + nV) * sizeof(MYTYPE));
		wr    	= (double*) malloc((dimZ + nV) * sizeof(double));

		d				= (double*) malloc((nV + dimZ) * sizeof(double));
		e 			= (double*) malloc((nV + dimZ) * sizeof(double));

		isplit 	= (int*) malloc((dimZ + nV) * sizeof(int));
		iblock 	= (int*) malloc((dimZ + nV) * sizeof(int));
		ifailv 	= (int*) malloc((dimZ + nV) * sizeof(int));
		suppz   = (int*) malloc(2 * (nV + dimZ) * sizeof(int));

		/******************	SGEVP	*********************/
			/******************************
			*		Solve B x = \lambda A x		*
			*   Find largest EV           *
			******************************/
		//	Return the problem into standard form C = R' \ (A / R)
		ierr = LAPACKE_zhegst (LAPACK_COL_MAJOR, 1, 'U', nV + dimZ, VtV, nV + dimZ, H, ldh);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRSVDDeflation:: LAPACKE_zhegst error %d consult the documentation of mkl to understand what happened \n", ierr);
			MPI_Abort(comm, ierr);
		}
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank) printf("BGMRESMDRSVDDeflation:: Return to the standard SEVP well passed\n");
		MPI_Barrier(comm);
#endif

		//	Reduce the matrix C to tridiagonal form
		ierr = LAPACKE_zhetrd (LAPACK_COL_MAJOR, 'U', nV + dimZ, VtV, nV + dimZ, d, e, tau);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRSVDDeflation:: LAPACKE_zhetrd error %d consult the documentation of mkl to understand what happened \n", ierr);
			if(!rank) PrintMatZ(BGMRESMDR->VtV, dimZ + nV, dimZ + nV, dimZ + nV, "VtV");
			MPI_Barrier(comm);
			MPI_Abort(comm, ierr);
		}
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank) printf("BGMRESMDRSVDDeflation::Reduction to tridiagonal matrix well passed\n");
		MPI_Barrier(comm);
#endif

		//	Solution of GEVP by hseqr
			//	Construct the orthogonal factor of tridiagonalization
		ierr = LAPACKE_zungtr( LAPACK_COL_MAJOR, 'U', dimZ + nV, VtV, dimZ + nV, tau);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRSVDDeflation:: LAPACKE_zungtr error %d consult the documentation of mkl to understand what happened \n", ierr);
			MPI_Abort(comm, ierr);
		}

		//	QR algorithm to compute eigenvalues and eigenvectors
		ierr = LAPACKE_zsteqr( LAPACK_COL_MAJOR, 'V', dimZ + nV, d, e, VtV, dimZ + nV );
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRSVDDeflation:: LAPACKE_zsteqr error %d consult the documentation of mkl to understand what happened \n", ierr);
			MPI_Abort(comm, ierr);
		}

#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank) PrintDVec(d, dimZ + nV, "Evs");
	MPI_Barrier(comm);
#endif
	  VP	= VtV + (nV + dimZ) * nV;
		cpt = dimZ;

		//	Construct the eigenvectors of the original problem
		ierr = LAPACKE_ztrtrs (LAPACK_COL_MAJOR, 'U', 'N', 'N', nV + dimZ, cpt, H, ldh, VP, dimZ + nV);
		BGMRESMDR->flops += flops_ztrsm(nV + dimZ, cpt);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRSVDDeflation:: LAPACKE_ztrtrs error %d consult the documentation of mkl to understand what happened \n", ierr);
			MPI_Abort(comm, ierr);
		}

		/**************** End SGEVP ******************/

		//	Set the dimension of the deflation subspace
		BGMRESMDR->dimZ = BGMRESMDR->MaxDefDim;
		cpt = BGMRESMDR->MaxDefDim;

		//	Compute P_k = H * p_k
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank){ printf("H*p_k:: mH = %d, cpt = %d, nH = %d, ldh = %d, nV + dimZ = %d\n", mH, cpt, nH, ldh, nV + dimZ); }
		MPI_Barrier(comm);
#endif
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, mH, cpt, nH, &const_alpha, H2, ldh, VP, nV + dimZ, &const_beta, H3, mH);
		BGMRESMDR->flops += flops_zgemm(mH, cpt, nH);
#if BGMRESMDRDEBUG
					MPI_Barrier(comm);
						if(!rank){
							//PrintMat(H2, nV + dimZ + 1, nV + dimZ, ldh, "H");
						}
					MPI_Barrier(comm);
#endif

		//	QR factorization of P_k = Q_P R_P
		ierr = LAPACKE_zgeqrf (LAPACK_COL_MAJOR, mH, cpt, H3, mH, tau);
		BGMRESMDR->flops += flops_zgeqrf(mH, cpt);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRSVDDeflation:: LAPACKE_zgeqrf error %d consult the documentation of mkl to understand what happened \n", ierr);
			MPI_Abort(comm, ierr);
		}

		//	p_k = p_k R_P^{-1}
		cblas_ztrsm (CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, nH, cpt, &const_alpha, H3, mH, VP, nH);
		BGMRESMDR->flops += flops_ztrsm(nH, cpt);

		//	Temp = Z * p_k
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank){ printf("Z*p_k::here ln = %d\n", ln); }
		MPI_Barrier(comm);
#endif
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ln, cpt, dimZ, &const_alpha, Z, ln, VP, nH, &const_beta, workZ, ln);
		BGMRESMDR->flops += flops_zgemm(ln, cpt, dimZ);

		//	Temp = Temp + V * p_k
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank){ printf("+V*p_k::here ln = %d\n", ln); }
		MPI_Barrier(comm);
#endif
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ln, cpt, nV, &const_alpha, V, ln, VP + dimZ, nH, &const_alpha, workZ, ln);
		BGMRESMDR->flops += flops_zgemm_sum(ln, cpt, nV);

		//	[AZ V] = [AZ V] * Q_k
		ierr = LAPACKE_zunmqr (LAPACK_COL_MAJOR, 'R', 'N', ln, mH, cpt, H3, mH, tau, AZ, ln);
		BGMRESMDR->flops += flops_zgemm(ln, mH, cpt);
		if(ierr != 0){
			fprintf(stderr, "BGMRESMDRSVDDeflation:: LAPACKE_zunmqr error %d consult the documentation of mkl to understand what happened \n", ierr);
			MPI_Abort(comm, ierr);
		}

		//	Z = Temp
		memcpy(Z, workZ, cpt * ln * sizeof(MYTYPE));
		BGMRESMDR->flops += flops_memset_z(cpt * ln);

		//	Prepares Z' * Z
#if BGMRESMDRDEBUG
		MPI_Barrier(comm);
		if(!rank){ printf("Z^H*Z::here ln = %d\n", ln); }
		MPI_Barrier(comm);
#endif
		cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, dimZ, dimZ, ln, &const_alpha, workZ, ln, workZ, ln, &const_beta, ZtZ, dimZ);
		BGMRESMDR->flops += flops_zgemm(dimZ, dimZ, ln);
		ierr = MPI_Allreduce(MPI_IN_PLACE, ZtZ, dimZ * dimZ, MPI_C_DOUBLE_COMPLEX, MPI_SUM, comm);
		BGMRESMDR->flops += flops_allreduce_z(dimZ * dimZ, BGMRESMDR->ND);

		//	Set values of D
		for(int i = 0; i < dimZ; i++){
			D[i].real = 1./sqrt(ZtZ[i + i * dimZ].real);
			D[i].imag = 0;
		}
#if BGMRESMDRDEBUG
					MPI_Barrier(comm);
						if(!rank){
							printf("here cpt = %d\n", cpt);
							PrintDVecZ(D, dimZ, "D");
						}
					MPI_Barrier(comm);
#endif

		//	Normalize columns of Z
		for(int j = 0; j < dimZ; j++){
			for(int i = 0; i < ln; i++){
				Z[i + j * ln].real = Z[i + j * ln].real * D[j].real;
				Z[i + j * ln].imag = Z[i + j * ln].imag * D[j].real;
			}
		}
		BGMRESMDR->flops += 2 * ln * dimZ;

		//	Update ZtZ
		for(int j = 0; j < dimZ; j++){
			for(int i = 0; i < dimZ; i++){
				ZtZ[i + j * dimZ].real = ZtZ[i + j * dimZ].real * D[j].real;
				ZtZ[i + j * dimZ].imag = ZtZ[i + j * dimZ].imag * D[j].real;
			}
		}
		for(int j = 0; j < dimZ; j++){
			for(int i = 0; i < dimZ; i++){
				ZtZ[j + i * dimZ].real = ZtZ[j + i * dimZ].real * D[j].real;
				ZtZ[j + i * dimZ].imag = ZtZ[j + i * dimZ].imag * D[j].real;
			}
		}
		BGMRESMDR->flops += 4 * dimZ * dimZ;

#if BGMRESMDRDEBUG
					MPI_Barrier(comm);
						if(!rank){
							printf("done\n");
						}
					MPI_Barrier(comm);
#endif
		free(wr);
		free(alphar);
		free(alphai);
		free(beta);
		free(tau);
		free(d);
    free(e);
		free(isplit);
		free(iblock);
		free(ifailv);
    free(suppz);
		BGMRESMDR->Timing.DeflationComputationTime += MPI_Wtime() - Timing;
	}
	return ierr;
}

/** \fn int BGMRESMDRSVDDeflationSimple(BGMRESMDR_t* BGMRESMDR){
 * \brief First deflation based on singular values approximation
 * \details Computes the singular values of the Hessenberg matrix
 * \param BGMRESMDR The BGMRESMDR context
 * \remarks No deflation vectors are supposed to exist
 * \warning
*/
int BGMRESMDRSVDDeflationSimple(BGMRESMDR_t* BGMRESMDR){

	double timing = MPI_Wtime();
	int ierr = 0;
	MPI_Comm comm = BGMRESMDR->comm;
	int rank			= BGMRESMDR->rank;

	int iter = BGMRESMDR->iteration;
	int t    = BGMRESMDR->nrhs;

	int* vidx = BGMRESMDR->VIdx;

	MYTYPE* H		= BGMRESMDR->H;
	MYTYPE* H2	= BGMRESMDR->H2;
	MYTYPE* H3	= BGMRESMDR->H3;

	int nH = vidx[iter];
	int mH = vidx[iter] + t;
	int ldh = BGMRESMDR->MaxBasis;

	MYTYPE* work	= BGMRESMDR->WorkH;
	MYTYPE* ZtZ		= BGMRESMDR->ZtZ;
	MYTYPE* D			= BGMRESMDR->D;

	MYTYPE* V			=	BGMRESMDR->V;
	MYTYPE* Z			=	BGMRESMDR->Z;
	MYTYPE* AZ		=	BGMRESMDR->AZ;
	MYTYPE* workZ	=	BGMRESMDR->WorkZ;

	int ln				=	BGMRESMDR->ln;

	double* alphar = (double*) malloc( mH * sizeof(double));

	BGMRESMDR->dimZ	= BGMRESMDR->MaxDefDim;
	int dimZ				= BGMRESMDR->dimZ;

	MYTYPE const_alpha = {1., 0};
	MYTYPE const_beta = {0, 0};

	//	SVD decomposition of H
	ierr = LAPACKE_zgesdd( LAPACK_COL_MAJOR, 'S', mH, nH, H2, ldh, alphar, H3, mH, work, nH);
	BGMRESMDR->flops += flops_zgesvd(mH, nH);
	if(ierr != 0){
		fprintf(stderr, "BGMRESMDRSVDDeflationSimple:: LAPACKE_zgesdd error %d\n", ierr);
		MPI_Abort(comm, ierr);
	}
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank) printf("BGMRESMDRSVDDeflationSimple::Computing SVD decomposition well passed\n");
	MPI_Barrier(comm);
#endif

	for(int i = 0; i < dimZ; i++){
		D[i].real = alphar[i + nH - dimZ];
		D[i].imag = 0;
	}

	//	Prepare AZ
	cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ln, dimZ, mH, &const_alpha, V, ln, H3 + (nH - dimZ) * mH, mH, &const_beta, AZ, ln);
	BGMRESMDR->flops += flops_zgemm(ln, dimZ, mH);
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank) printf("BGMRESMDRSVDDeflationSimple::Preparation of AZ well passed\n");
	MPI_Barrier(comm);
#endif

	//	Copy and translate vt returned by svd
	for(int j = nH - dimZ; j < nH; j++){
		for(int i = 0; i < nH; i++){
			H2[ j * nH + i ].real = work[ nH * i + j ].real;
			H2[ j * nH + i ].imag = -work[ nH * i + j ].imag;
		}
	}
	BGMRESMDR->flops += 2 * nH * (nH - dimZ);
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank) printf("BGMRESMDRSVDDeflationSimple::Copying and translating vt well passed\n");
	MPI_Barrier(comm);
#endif

	//	Prepare Z
	cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ln, dimZ, nH, &const_alpha, V, ln, H2 + (nH - dimZ) * nH , nH, &const_beta, Z, ln);
	BGMRESMDR->flops += flops_zgemm(ln, dimZ, nH);
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank) printf("BGMRESMDRSVDDeflationSimple::Preparation of Z well passed\n");
	MPI_Barrier(comm);
#endif

	//	Prepare Z' * Z
	for(int j = 0; j < dimZ; j++){
		for(int i = 0; i < dimZ; i++){
			ZtZ[i + j * dimZ].real = (i == j) ? 1. : 0;
			ZtZ[i + j * dimZ].imag = 0;
		}
	}
	BGMRESMDR->flops += 2 * dimZ * dimZ;
#if BGMRESMDRDEBUG
	MPI_Barrier(comm);
	if(!rank) printf("BGMRESMDRSVDDeflationSimple::Preparation of ZtZ well passed\n");
	MPI_Barrier(comm);
#endif
	free(alphar);

	BGMRESMDR->Timing.DeflationComputationTime += MPI_Wtime() - timing;

	return ierr;
}

int BGMRESMDRSetHessenbergDefPart(BGMRESMDR_t* BGMRESMDR){
	double Timing = MPI_Wtime();
	int ierr = 0;
	int dimZ = BGMRESMDR->dimZ;
	if(dimZ > 0){
		for(int i = 0; i < dimZ; i++){
			BGMRESMDR->H[i + (dimZ + BGMRESMDR->MaxBasis) * i].real = BGMRESMDR->D[i].real;
			BGMRESMDR->H2[i + (dimZ + BGMRESMDR->MaxBasis) * i].real = BGMRESMDR->D[i].real;
			BGMRESMDR->H3[i + (dimZ + BGMRESMDR->MaxBasis) * i].real = BGMRESMDR->D[i].real;

			BGMRESMDR->H[i + (dimZ + BGMRESMDR->MaxBasis) * i].imag = 0;
			BGMRESMDR->H2[i + (dimZ + BGMRESMDR->MaxBasis) * i].imag = 0;
			BGMRESMDR->H3[i + (dimZ + BGMRESMDR->MaxBasis) * i].imag = 0;
		}
	}
#if BGMRESMDRDEBUG
	MPI_Barrier(BGMRESMDR->comm);
	if(!BGMRESMDR->rank){
		PrintMatZ(BGMRESMDR->H, dimZ, dimZ, dimZ + BGMRESMDR->MaxBasis, "BGMRESMDRSetHessenbergDefPart::D^{-1}");
	}
	MPI_Barrier(BGMRESMDR->comm);
#endif
	BGMRESMDR->flops += dimZ * 3;
	BGMRESMDR->Timing.DeflationComputationTime += MPI_Wtime() - Timing;
	return ierr;
}



/** \fn int BGMRESMDRPrepareDeflationMatrix(BGMRESMDR_t* BGMRESMDR)
 * \brief
 * \details
 * \param BGMRESMDR
 * \remarks
 * \warning
*/
int BGMRESMDRPrepareDeflationMatrix(BGMRESMDR_t* BGMRESMDR){
	double time = MPI_Wtime();
	int ierr = 0;
	int ND = BGMRESMDR->ND;
	int dimZ = BGMRESMDR->dimZ;
	MYTYPE* temp = BGMRESMDR->WorkH;

	//	temp_i = AZ' * AZ
	MYTYPE const_alpha = {1., 0};
	MYTYPE const_beta = {0, 0};
	cblas_zgemm (CblasColMajor, CblasConjTrans, CblasNoTrans, BGMRESMDR->dimZ, BGMRESMDR->dimZ, BGMRESMDR->ln, &const_alpha, BGMRESMDR->AZ, BGMRESMDR->ln, BGMRESMDR->AZ, BGMRESMDR->ln, &const_beta, temp, BGMRESMDR->dimZ);
	BGMRESMDR->flops += flops_zgemm(dimZ, BGMRESMDR->ln, dimZ);

	//	temp = sum_i temp_i
	ierr = MPI_Allreduce(MPI_IN_PLACE, temp, BGMRESMDR->dimZ * BGMRESMDR->dimZ, MPI_C_DOUBLE_COMPLEX, MPI_SUM, BGMRESMDR->comm);
	BGMRESMDR->flops += flops_allreduce_z(dimZ * dimZ, ND);

#if BGMRESMDRDEBUG
	MPI_Barrier(BGMRESMDR->comm);
	if(!BGMRESMDR->rank) PrintMatZ(temp, dimZ, dimZ, dimZ, "BGMRESMDRPrepareDeflationMatrix::AZtAZ System changed");
	MPI_Barrier(BGMRESMDR->comm);
#endif

	//	temp = r' * r
	ierr = LAPACKE_zpotrf(LAPACK_COL_MAJOR, 'U', dimZ, temp, dimZ);
	if(ierr != 0){
		fprintf(stderr, "BGMRESMDRPrepareDeflationMatrix::LAPACKE_zpotrf error %d\n", ierr);
		MPI_Abort(BGMRESMDR->comm, ierr);
	}
	BGMRESMDR->flops += flops_zpotrf(dimZ);

	if(ierr != 0){
		fprintf(stderr, "BGMRESMDRPrepareDeflationMatrix::LAPACKE_dpotrf error %d\n", ierr);
		MPI_Abort(BGMRESMDR->comm, ierr);
	}

	//	AZ = AZ r^{-1}
	cblas_ztrsm (CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, BGMRESMDR->ln, dimZ, &const_alpha, temp, dimZ, BGMRESMDR->AZ, BGMRESMDR->ldv);
	BGMRESMDR->flops += flops_ztrsm(BGMRESMDR->ln, dimZ);

	//	Z = Z r^{-1}
	cblas_ztrsm (CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, BGMRESMDR->ln, dimZ, &const_alpha, BGMRESMDR->WorkH, dimZ, BGMRESMDR->Z, BGMRESMDR->ldv);
	BGMRESMDR->flops += flops_ztrsm(BGMRESMDR->ln, dimZ);
	ierr = BGMRESMDRResSolInit(BGMRESMDR);

	//	temp_i = Z' * Z
	cblas_zgemm (CblasColMajor, CblasConjTrans, CblasNoTrans, BGMRESMDR->dimZ, BGMRESMDR->dimZ, BGMRESMDR->ln, &const_alpha, BGMRESMDR->Z, BGMRESMDR->ln, BGMRESMDR->Z, BGMRESMDR->ln, &const_beta, BGMRESMDR->ZtZ, BGMRESMDR->dimZ);
	BGMRESMDR->flops += flops_zgemm(dimZ, BGMRESMDR->ln, dimZ);

	//	temp = sum temp_i
	ierr = MPI_Allreduce(MPI_IN_PLACE, BGMRESMDR->ZtZ, BGMRESMDR->dimZ * BGMRESMDR->dimZ, MPI_C_DOUBLE_COMPLEX, MPI_SUM, BGMRESMDR->comm);
	BGMRESMDR->flops += flops_allreduce_z(dimZ * dimZ, ND);

	//	D[j] = ZtZ[j, j]
	for(int j = 0; j < dimZ; j++){
		BGMRESMDR->D[j].real = 1./sqrt(BGMRESMDR->ZtZ[j + dimZ * j].real);
	}
	BGMRESMDR->flops += flops_allreduce_z(dimZ * dimZ, ND);

	//	Z = Z * D
	for(int j = 0; j < dimZ; j++){
		for(int i = 0; i < BGMRESMDR->ln; i++){
			BGMRESMDR->Z[i + j * BGMRESMDR->ln].real = BGMRESMDR->Z[i + j * BGMRESMDR->ln].real * BGMRESMDR->D[j].real;
			BGMRESMDR->Z[i + j * BGMRESMDR->ln].imag = BGMRESMDR->Z[i + j * BGMRESMDR->ln].imag * BGMRESMDR->D[j].real;
		}
	}
	BGMRESMDR->flops += 2 * BGMRESMDR->ln * dimZ;

	//	ZtZ = D * ZtZ * D
		// ZtZ = ZtZ * D
	for(int j = 0; j < dimZ; j++){
		for(int i = 0; i < dimZ; i++){
			BGMRESMDR->ZtZ[i + j * dimZ].real = BGMRESMDR->ZtZ[i + j * dimZ].real * BGMRESMDR->D[j].real;
			BGMRESMDR->ZtZ[i + j * dimZ].imag = BGMRESMDR->ZtZ[i + j * dimZ].imag * BGMRESMDR->D[j].real;
		}
	}
	BGMRESMDR->flops += 2 * dimZ * dimZ;
		//	ZtZ = D * ZtZ
	for(int j = 0; j < dimZ; j++){
		for(int i = 0; i < dimZ; i++){
			BGMRESMDR->ZtZ[j + i * dimZ].real = BGMRESMDR->ZtZ[j + i * dimZ].real * BGMRESMDR->D[j].real;
			BGMRESMDR->ZtZ[j + i * dimZ].imag = BGMRESMDR->ZtZ[j + i * dimZ].imag * BGMRESMDR->D[j].real;
		}
	}
	BGMRESMDR->flops += 2 * dimZ * dimZ;
#if BGMRESMDRDEBUG
			MPI_Barrier(BGMRESMDR->comm);
			if(!BGMRESMDR->rank) PrintMatZ(BGMRESMDR->ZtZ, dimZ, dimZ, dimZ, "BGMRESMDRPrepareDeflationMatrix::ZtZ");
			if(!BGMRESMDR->rank) PrintDVecZ(BGMRESMDR->D, dimZ, "BGMRESMDRPrepareDeflationMatrix::D");
			MPI_Barrier(BGMRESMDR->comm);
#endif

	BGMRESMDR->Timing.DeflationComputationTime += MPI_Wtime() - time;
	return ierr;
}


/** \fn int BGMRESMDRRestart(BGMRESMDR_t* BGMRESMDR)
 * \brief
 * \details
 * \param BGMRESMDR
 * \remarks
 * \warning
*/
int BGMRESMDRRestart(BGMRESMDR_t* BGMRESMDR){
	int ierr = 0;
	int temp = BGMRESMDR->MaxBasis + BGMRESMDR->dimZ;
	int t = BGMRESMDR->nrhs;
#if BGMRESMDRDEBUG
	MPI_Barrier(BGMRESMDR->comm);
	if(!BGMRESMDR->rank) printf("Resetting values ...\n");
	if(!BGMRESMDR->rank) printf("Resetting values H\n");
	MPI_Barrier(BGMRESMDR->comm);
#endif
	memset(BGMRESMDR->H,			0, temp * (temp - t) * sizeof(MYTYPE));
	BGMRESMDR->flops += flops_memset_z(temp * (temp - t));

	memset(BGMRESMDR->H2,		0, temp * (temp - t) * sizeof(MYTYPE));
	BGMRESMDR->flops += flops_memset_z(temp * (temp - t));

	memset(BGMRESMDR->H3,		0, temp * (temp - t) * sizeof(MYTYPE));
	BGMRESMDR->flops += flops_memset_z(temp * (temp - t));

#if BGMRESMDRDEBUG
	MPI_Barrier(BGMRESMDR->comm);
	if(!BGMRESMDR->rank) printf("Resetting values KRHS\n");
	MPI_Barrier(BGMRESMDR->comm);
#endif
	memset(BGMRESMDR->KRHS,	0, temp * t * sizeof(MYTYPE));
	BGMRESMDR->flops += flops_memset_z(temp * t);

	temp = BGMRESMDR->MaxBasis;
	memset(BGMRESMDR->Htau,	0, temp * t * sizeof(MYTYPE));
	BGMRESMDR->flops += flops_memset_z(temp * t);

#if BGMRESMDRDEBUG
	MPI_Barrier(BGMRESMDR->comm);
	if(!BGMRESMDR->rank) printf("Resetting values Q\n");
	MPI_Barrier(BGMRESMDR->comm);
#endif

	memset(BGMRESMDR->Q,			0, temp * t * t * sizeof(MYTYPE));
	BGMRESMDR->flops += flops_memset_z(temp * t * t);

	memset(BGMRESMDR->Qtau,	0, temp * t * sizeof(MYTYPE));
	BGMRESMDR->flops += flops_memset_z(temp * t);

#if BGMRESMDRDEBUG
	MPI_Barrier(BGMRESMDR->comm);
	if(!BGMRESMDR->rank) PrintIVec(BGMRESMDR->BSize, temp, "Bsize b4 resetting BIdx\n");
	MPI_Barrier(BGMRESMDR->comm);
#endif
	memset(BGMRESMDR->BIdx,	0, temp * sizeof(int));
	BGMRESMDR->flops += flops_memset(temp);
#if BGMRESMDRDEBUG
	MPI_Barrier(BGMRESMDR->comm);
	if(!BGMRESMDR->rank) PrintIVec(BGMRESMDR->BSize, temp, "Bsize after resetting BIdx\n");
	MPI_Barrier(BGMRESMDR->comm);
#endif

#if BGMRESMDRDEBUG
	MPI_Barrier(BGMRESMDR->comm);
	if(!BGMRESMDR->rank) printf("Resetting values VIdx\n");
	MPI_Barrier(BGMRESMDR->comm);
#endif
	memset(BGMRESMDR->VIdx,	0, temp * sizeof(int));
	BGMRESMDR->flops += flops_memset(temp);

	BGMRESMDR->VIdx[0] = 0;
	BGMRESMDR->VIdx[1] = t;
	BGMRESMDR->iteration = 0;
#if BGMRESMDRDEBUG
	MPI_Barrier(BGMRESMDR->comm);
	if(!BGMRESMDR->rank) printf("Resetting values done\n");
	MPI_Barrier(BGMRESMDR->comm);
#endif
	return 0;
}

int BGMRESMDRUnNormalizeVecZ(BGMRESMDR_t* BGMRESMDR){
	int ierr = 0;
	for(int j = 0; j < BGMRESMDR->dimZ; j++){
		for(int i = 0; i < BGMRESMDR->ln; i++){
			BGMRESMDR->Z[i + j * BGMRESMDR->ln].real = BGMRESMDR->Z[i + j * BGMRESMDR->ln].real * BGMRESMDR->D[j].real;
			BGMRESMDR->Z[i + j * BGMRESMDR->ln].imag = BGMRESMDR->Z[i + j * BGMRESMDR->ln].imag * BGMRESMDR->D[j].real;
		}
	}
	BGMRESMDR->flops += 2 * BGMRESMDR->ln * BGMRESMDR->dimZ;
	return ierr;
}

int BGMRESMDRNormalizeD(BGMRESMDR_t* BGMRESMDR){
	for(int i = 0; i < BGMRESMDR->dimZ; i++){
		BGMRESMDR->D[i].real = 1.;
		BGMRESMDR->H[i + (BGMRESMDR->dimZ + BGMRESMDR->MaxBasis) * i].real = 1.;
		BGMRESMDR->H2[i + (BGMRESMDR->dimZ + BGMRESMDR->MaxBasis) * i].real = 1.;
		BGMRESMDR->H3[i + (BGMRESMDR->dimZ + BGMRESMDR->MaxBasis) * i].real = 1.;

		BGMRESMDR->H[i + (BGMRESMDR->dimZ + BGMRESMDR->MaxBasis) * i].imag = 0;
		BGMRESMDR->H2[i + (BGMRESMDR->dimZ + BGMRESMDR->MaxBasis) * i].imag = 0;
		BGMRESMDR->H3[i + (BGMRESMDR->dimZ + BGMRESMDR->MaxBasis) * i].imag = 0;
	}
	BGMRESMDR->flops += BGMRESMDR->dimZ * 7;
	return 0;
}


// Laplacian 1D matrix parallel spmm: n is the number of local unknowns, m is the number of vectors
int lap2dvaw(MPI_Comm comm, MYTYPE* w, MYTYPE* v, int n, int m){
	//printf("lap2\n");
	int ierr = 0;
	int rank = -1;
	int size = -1;
	MYTYPE xp = {-1., 0};
	MYTYPE xl = {-1., 0};
	MYTYPE* b = NULL;
	MYTYPE* x = NULL;
	MPI_Request request[4];
	MPI_Status  status[4];
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	for(int j = 0; j < m; j++){
		x = w + n * j;
		b = v + n * j;
		if(rank == 0){
			MPI_Isend(&x[n - 1], 1, MPI_C_DOUBLE_COMPLEX, 1, 0, comm, request);
			MPI_Irecv(&xl, 1, MPI_C_DOUBLE_COMPLEX, 1, 0, comm, request + 1);
		}else if(rank != size - 1){
			MPI_Isend(&x[n - 1], 1, MPI_C_DOUBLE_COMPLEX, rank + 1, 0, comm, request );
			MPI_Isend(&x[0]	 , 1, MPI_C_DOUBLE_COMPLEX, rank - 1, 0, comm, request + 1);
			MPI_Irecv(&xl, 1, MPI_C_DOUBLE_COMPLEX, rank + 1, 0, comm, request + 2);
			MPI_Irecv(&xp, 1, MPI_C_DOUBLE_COMPLEX, rank - 1, 0, comm, request + 3);
		}else if(rank == size - 1){
			MPI_Isend(&x[0], 1, MPI_C_DOUBLE_COMPLEX, rank - 1, 0, comm, request);
			MPI_Irecv(&xp, 1, MPI_C_DOUBLE_COMPLEX, rank - 1, 0, comm, request + 1);
		}
		if(rank == 0){
			MPI_Waitall(2, request, status);
			b[0].real = 2 * x[0].real - x[1].real;
			b[0].imag = 2 * x[0].imag - x[1].imag;
			for(int i = 1; i < n - 1; i++){
				b[i].real = 2 * x[i].real - x[i + 1].real - x[i - 1].real;
				b[i].imag = 2 * x[i].imag - x[i + 1].imag - x[i - 1].imag;
			}
			b[n - 1].real = 2 * x[n - 1].real - x[n - 2].real - xl.real;
			b[n - 1].imag = 2 * x[n - 1].imag - x[n - 2].imag - xl.imag;
		}else if(rank != size - 1){
			MPI_Waitall(4, request, status);
			b[0].real = 2 * x[0].real - x[1].real - xp.real;
			b[0].imag = 2 * x[0].imag - x[1].imag - xp.imag;
			for(int i = 1; i < n - 1; i++){
				b[i].real = 2 * x[i].real - x[i + 1].real - x[i - 1].real;
				b[i].imag = 2 * x[i].imag - x[i + 1].imag - x[i - 1].imag;
			}
			b[n - 1].real = 2 * x[n - 1].real - x[n - 2].real - xl.real;
			b[n - 1].imag = 2 * x[n - 1].imag - x[n - 2].imag - xl.imag;
		}else if(rank == size - 1){
			MPI_Waitall(2, request, status);
			b[0].real = 2 * x[0].real - x[1].real - xp.real;
			b[0].imag = 2 * x[0].imag - x[1].imag - xp.imag;
			for(int i = 1; i < n - 1; i++){
				b[i].real = 2 * x[i].real - x[i + 1].real - x[i - 1].real;
				b[i].imag = 2 * x[i].imag - x[i + 1].imag - x[i - 1].imag;
			}
			b[n - 1].real = 2 * x[n - 1].real - x[n - 2].real;
			b[n - 1].imag = 2 * x[n - 1].imag - x[n - 2].imag;
		}
	}
	return 0;
}


int BGMRESMDRInexactBreakdownDetection(BGMRESMDR_t* BGMRESMDR){
	double Timing = MPI_Wtime();
	int ierr	= 0;
	int t			= BGMRESMDR->nrhs;
	int iter	= BGMRESMDR->iteration;
	int ln		= BGMRESMDR->ln;
	int* vidx	= BGMRESMDR->VIdx;
	int dimZ	= BGMRESMDR->dimZ;
	int ldh		= dimZ + BGMRESMDR->MaxBasis;
#if BGMRESMDRDEBUG
	MPI_Barrier(BGMRESMDR->comm);
	if(!BGMRESMDR->rank) printf("iteration: %d, vidx[iter] = %d\n", iter, vidx[iter]);
	MPI_Barrier(BGMRESMDR->comm);
#endif

	if(BGMRESMDR->Red != NoRed){
		//	work is a work space
		MYTYPE* work	= BGMRESMDR->WorkH;
		int	ldwork		= BGMRESMDR->MaxBasis;

		//	Set workH space to 0 this is necessary
		memset(work, 0, ldwork * t * sizeof(MYTYPE));
		BGMRESMDR->flops += flops_memset_z(ldwork * t);

		//	r points to the Krylov residual
		MYTYPE* r = BGMRESMDR->KRHS;
		int ldr		= ldh;

		//	G points to the rest of the Krylov residual
		MYTYPE* G = r + dimZ + vidx[iter];
		int ldG		= ldh;

		//	Q points to the place where the rest of residual will exists
		MYTYPE* Q = work + vidx[iter];
		int ldQ		= ldwork;

		//	Qtau will contain at the begining the singular values
		MYTYPE* Qtau = BGMRESMDR->Qtau + iter * t;
		double* double_Qtau = (double*) malloc(2 * t * sizeof(double));

		//	Copy G to Q
		ierr = LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', t, t, G, ldG, Q, ldQ);
		BGMRESMDR->flops += flops_copy_z(t, t);

		//	Compute SVD factorization of G
		ierr = LAPACKE_zgesvd( LAPACK_COL_MAJOR, 'O', 'N', t, t, Q, ldQ, double_Qtau, NULL, 1, NULL, 1, double_Qtau + t);
		BGMRESMDR->flops += flops_zgesvd(t, t);

		if(ierr != 0){
			if(!BGMRESMDR->rank){
				fprintf(stderr, "BGMRESMDRInexactBreakdownDetection:: LAPACKE_dgesvd error %d\n", ierr);
			}
		}

#if BGMRESMDRDEBUG
		MPI_Barrier(BGMRESMDR->comm);
		if(BGMRESMDR->rank == 0)
			PrintDVec(double_Qtau, t, "Singular Values");
		MPI_Barrier(BGMRESMDR->comm);
#endif

		//	Determin the rank of G
		double threshold = BGMRESMDR->Ctol;
		for(int i = 0; i < t; i++){
			if(double_Qtau[i] * sqrt((double) t) > threshold) BGMRESMDR->ActualSize = i + 1;
		}

		//	Determin t_{iter+1}
		BGMRESMDR->BSize[BGMRESMDR->GIter] = BGMRESMDR->ActualSize;

		//	Set VIdx and BIdx
		vidx[iter + 1] = vidx[iter] + BGMRESMDR->ActualSize;
		BGMRESMDR->IBreakdown = t - BGMRESMDR->ActualSize;
		if( BGMRESMDR->ActualSize != t ){
			BGMRESMDR->BIdx[iter] = 1;
		}

		MYTYPE* F			= NULL;
		int nF				= 0;
		MYTYPE* Ftau	= NULL;
		MYTYPE* C			= NULL;
		int ldC				=	0;
		int mC				=	0;
		int nC				= 0;

		//	Compute the matrix of rotation if there is an inexact breakdown
		if(BGMRESMDR->BIdx[iter] == 1){
			//	Apply the Householder rotaions on the rest of the residual
			//	From down to up
			for(int j = iter - 1; j >= 0; j--){
#if BGMRESMDRDEBUG
				if(ldh * (dimZ + vidx[j]) + dimZ + vidx[j] < 0 || ldh * (dimZ + vidx[j]) + dimZ + vidx[j] > ldh * (ldh - t)){
					fprintf(stderr, "BGMRESMDRInexactBreakdownDetection:: apply H Householder, pointer depasses memory of H");
				}
#endif
				//	F points to the Householder matrix of H
				F			= BGMRESMDR->H + ldh * (dimZ + vidx[j]) + dimZ + vidx[j];

				//	Ftau points to the Householder factors associated to F
				Ftau	= BGMRESMDR->Htau + vidx[j];

				//	number of columns in F
				nF = vidx[j + 1] - vidx[j];

				//	C is the part on which F is applied
				C = work + vidx[j];
				ldC		= ldwork;

				//	size of C
				mC = t + vidx[j + 1] - vidx[j];
				nC = t;

				//	Apply F on C
				ierr = LAPACKE_zunmqr (LAPACK_COL_MAJOR, 'L', 'N', mC, nC, nF, F, ldh, Ftau, C, ldC);
				BGMRESMDR->flops += flops_zunmqr(mC, nC, nF);
			}
			//	Apply the Householder rotaions of last inexact breakdown on the rest of the residual
			//	From up to down
			for(int j = 1 ; j < iter; j++){
				if(BGMRESMDR->BIdx[j] == 1){
					//	F points to the rotation matrix in the form of Householder rotations
					F			= BGMRESMDR->Q + j * t * t;

					//	Ftau points to the Householder factors
					Ftau	= BGMRESMDR->Qtau + j * t;

					//	C is the part on which F is applied
					C = work + vidx[j];
					ldC		= ldwork;

					//	Application of F on C
					ierr = LAPACKE_zunmqr (LAPACK_COL_MAJOR, 'L', 'C', t, t, t, F, t, Ftau, C, ldC);
					BGMRESMDR->flops += flops_zunmqr(t, t, t);
				}
			}

			//	Copy the matrix to its last place
			ierr = LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', t, t, Q, ldQ, BGMRESMDR->Q + iter * t * t, t);
			BGMRESMDR->flops += flops_copy_z(t, t);
		}
		free(double_Qtau);
	}else{
		BGMRESMDR->BIdx[iter] = 0;
		BGMRESMDR->ActualSize = t;
		vidx[iter + 1] = vidx[iter] + BGMRESMDR->ActualSize;
		BGMRESMDR->BSize[BGMRESMDR->GIter] = BGMRESMDR->ActualSize;
	  BGMRESMDR->IBreakdown = 0;
	}

#if BGMRESMDRDEBUG
		MPI_Barrier(BGMRESMDR->comm);
		if(BGMRESMDR->rank == 0){
			printf("BGMRESMDRInexactBreakdownDetection:: iter = %d, vidx[iter] = %d, vidx[iter + 1] = %d \n", iter, vidx[iter], vidx[iter + 1]);
		}
		MPI_Barrier(BGMRESMDR->comm);
#endif

	BGMRESMDR->Timing.InexactBreakDownTime += MPI_Wtime() - Timing;
	return ierr;
}

int BGMRESMDRInexactBreakdownReduction(BGMRESMDR_t* BGMRESMDR){
	double Timing = MPI_Wtime();
	int ierr = 0;
#if BGMRESMDRDEBUG
	int rank, size;
	MPI_Comm_rank(BGMRESMDR->comm, &rank);
	MPI_Comm_size(BGMRESMDR->comm, &size);
#endif

	if(BGMRESMDR->IBreakdown != 0){
		int t			= BGMRESMDR->nrhs;
		int iter	= BGMRESMDR->iteration;
		int ln		= BGMRESMDR->ln;
		int* vidx = BGMRESMDR->VIdx;
		int dimZ	= BGMRESMDR->dimZ;
		int ldh		= BGMRESMDR->MaxBasis + dimZ;

		//	Q is the rotation matrix computed in the inexact breakdown test
		//	it will be in the form of Householder rotations
		MYTYPE* Q = BGMRESMDR->Q + iter * t * t;

		//	Qtau will be the Householder factors of Q
		MYTYPE* Qtau = BGMRESMDR->Qtau + iter * t;

		//	Compute QR factorization PS: We do this in order to apply it on a set of vectors in place
		ierr = 	LAPACKE_zgeqrf (LAPACK_COL_MAJOR, t, t, Q, t, Qtau);
		BGMRESMDR->flops += flops_zgeqrf(t, t);

		//	Apply Q on [D_{iter - 1}, \tilde{V}_{iter + 1}]
		MYTYPE* V = BGMRESMDR->V + ln * vidx[iter];
		ierr = LAPACKE_zunmqr (LAPACK_COL_MAJOR, 'R', 'N', ln, t, t, Q, t, Qtau, V, ln);
		BGMRESMDR->flops += flops_zunmqr(ln, t, t);

		if(iter == 0){
			//	Apply Q^t on Krylov residual if it is the restart test
			ierr = LAPACKE_zunmqr (LAPACK_COL_MAJOR, 'L', 'C', t, t, t, Q, t, Qtau, BGMRESMDR->KRHS + dimZ, ldh);
			BGMRESMDR->flops += flops_zunmqr(t, t, t);
		}
		if(BGMRESMDR->Def != NoDef){
			//	Apply Q^t on Hessenberg matrix
			MYTYPE* H = BGMRESMDR->H2 + ldh * dimZ + vidx[iter] + dimZ;
			ierr = LAPACKE_zunmqr (LAPACK_COL_MAJOR, 'L', 'C', t, vidx[iter], t, Q, t, Qtau, H, ldh);
			BGMRESMDR->flops += flops_zunmqr(vidx[iter], t, t);
		}
	}
	BGMRESMDR->Timing.InexactBreakDownTime += MPI_Wtime() - Timing;
	return ierr;
}

/** \fn int BGMRESMDRDump(BGMRESMDR_t* BGMRESMDR, char* MatName)
 * \brief
 * \details
 * \param BGMRESMDR
 * \param MatName
 * \remarks
 * \warning
*/
int BGMRESMDRDump(BGMRESMDR_t* BGMRESMDR, char* MatName){
	int ierr = 0;
	int rank, size, nthreads;
  char date[20];
  time_t now = time(NULL);
  strftime(date, 20, "%Y-%m-%d_%H-%M-%S", localtime(&now));
  char rootName[100] = "BGMRESMDR_";
	char filename[200];
	nthreads = omp_get_max_threads();
  sprintf(filename,
          "%s%d_%d__%d__%s_DEF_%d.txt",
					rootName,
					BGMRESMDR->ND,
					nthreads,
					BGMRESMDR->MaxBasis,
          date,
					BGMRESMDR->dimZ);
  MPI_Comm_rank(BGMRESMDR->comm, &rank);
  MPI_Comm_size(BGMRESMDR->comm, &size);

	double spmm[3], prec[3], defapp[3], defcomp[3], ortho[3], hess[3], spmmD[3], spmmOD[3], spmmComm[3], orthocomm[3];
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.PreconditioningTime,				prec,			1, MPI_DOUBLE, MPI_MAX, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.SPMMTime,									spmm,			1, MPI_DOUBLE, MPI_MAX, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.SPMMDiag,									spmmD,		1, MPI_DOUBLE, MPI_MAX, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.SPMMOffDiag,								spmmOD,		1, MPI_DOUBLE, MPI_MAX, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.SPMMComm,									spmmComm,	1, MPI_DOUBLE, MPI_MAX, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.HessenbergOperationsTime,	hess,			1, MPI_DOUBLE, MPI_MAX, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.OrthogonalizationTime,			ortho,		1, MPI_DOUBLE, MPI_MAX, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.OrthogonalizationComm,			orthocomm,1, MPI_DOUBLE, MPI_MAX, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.DeflationComputationTime,	defcomp,	1, MPI_DOUBLE, MPI_MAX, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.DeflationApplicationTime,	defapp,		1, MPI_DOUBLE, MPI_MAX, BGMRESMDR->comm);

	ierr = MPI_Allreduce(&BGMRESMDR->Timing.PreconditioningTime,				1 + prec,			1, MPI_DOUBLE, MPI_MIN, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.SPMMTime,									1 + spmm,			1, MPI_DOUBLE, MPI_MIN, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.SPMMDiag,									1 + spmmD,		1, MPI_DOUBLE, MPI_MIN, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.SPMMOffDiag,								1 + spmmOD,		1, MPI_DOUBLE, MPI_MIN, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.SPMMComm,									1 + spmmComm,	1, MPI_DOUBLE, MPI_MIN, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.HessenbergOperationsTime,	1 + hess,			1, MPI_DOUBLE, MPI_MIN, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.OrthogonalizationTime,			1 + ortho,		1, MPI_DOUBLE, MPI_MIN, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.OrthogonalizationComm,			1 + orthocomm,1, MPI_DOUBLE, MPI_MIN, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.DeflationComputationTime,	1 + defcomp,	1, MPI_DOUBLE, MPI_MIN, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.DeflationApplicationTime,	1 + defapp,		1, MPI_DOUBLE, MPI_MIN, BGMRESMDR->comm);

	ierr = MPI_Allreduce(&BGMRESMDR->Timing.PreconditioningTime,				2 + prec,			1, MPI_DOUBLE, MPI_SUM, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.SPMMTime,									2 + spmm,			1, MPI_DOUBLE, MPI_SUM, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.SPMMDiag,									2 + spmmD,		1, MPI_DOUBLE, MPI_SUM, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.SPMMOffDiag,								2 + spmmOD,		1, MPI_DOUBLE, MPI_SUM, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.SPMMComm,									2 + spmmComm,	1, MPI_DOUBLE, MPI_SUM, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.HessenbergOperationsTime,	2 + hess,			1, MPI_DOUBLE, MPI_SUM, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.OrthogonalizationTime,			2 + ortho,		1, MPI_DOUBLE, MPI_SUM, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.OrthogonalizationComm,			2 + orthocomm,1, MPI_DOUBLE, MPI_SUM, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.DeflationComputationTime,	2 + defcomp,	1, MPI_DOUBLE, MPI_SUM, BGMRESMDR->comm);
	ierr = MPI_Allreduce(&BGMRESMDR->Timing.DeflationApplicationTime,	2 + defapp,		1, MPI_DOUBLE, MPI_SUM, BGMRESMDR->comm);
	prec[2]			= prec[2]/size;
	spmm[2]			= spmm[2]/size;
	spmmD[2]		=	spmmD[2]/size;
	spmmOD[2]		=	spmmOD[2]/size;
	spmmComm[2]	=	spmmComm[2]/size;
	orthocomm[2]= orthocomm[2]/size;
	hess[2]			= hess[2]/size;
	ortho[2]		= ortho[2]/size;
	defcomp[2]	= defcomp[2]/size;
	defapp[2]		= defapp[2]/size;

  if (rank == 0) {
    FILE* oFile = fopen(filename,"a+");
    fprintf(oFile, "########################################################\n");
		fprintf(oFile, "#                       BGMRESMDR	                      #\n");
    fprintf(oFile, "########################################################\n");
		fprintf(oFile, "#	Implemented by: Hussam AL DAAS                       #\n");
		fprintf(oFile, "# Email:          aldaas.hussam@gmail.com              #\n");
		fprintf(oFile, "# Date:           26/01/2018                           #\n");
    fprintf(oFile, "########################################################\n");
    //fprintf(oFile, "%s\n", MatName);
    if (BGMRESMDR->rvec[BGMRESMDR->GIter].real < (BGMRESMDR->Ctol * BGMRESMDR->MaxNormb)) {
      fprintf(oFile, "The method converged!\n");
		}else{
			fprintf(oFile, "/!\\ The method didn't converge! /!\\\n");
		}
		fprintf(oFile,		"Solver information:\n");
    fprintf(oFile, 		"Number of processors:															\t %d\n", size);
    fprintf(oFile, 		"Number of threads:																	\t %d\n", nthreads);
#ifdef ILU0
    fprintf(oFile, 		"Preconditioner:																		\t ILU0 of block Jacobi \n");
#elif defined(LUDS)
    fprintf(oFile, 		"Preconditioner:																		\t LU of block Jacobi \n");
#endif
		if(BGMRESMDR->Ortho == DBCGS)
			fprintf(oFile,	"Orthogonalization strategy: 												\t Double Block Classical Gram-Schmidt\n");
		else
			fprintf(oFile, 	"Orthogonalization strategy: 												\t Block Classical Gram-Schmidt\n");
		if(BGMRESMDR->Def != NoDef){
			if(BGMRESMDR->Def == RITZ)
							fprintf(oFile,  "Deflation strategy:                \t Ritz values\n");
			if(BGMRESMDR->Def == HRITZ)
							fprintf(oFile,  "Deflation strategy:                \t Harmonic Ritz values\n");
			if(BGMRESMDR->Def == SVD)
							fprintf(oFile,  "Deflation strategy:                \t SVD values\n");
			fprintf(oFile,	"Max number of deflated eigenvalues:								\t %d\n", BGMRESMDR->dimZ);
		}
    fprintf(oFile, "########################################################\n");
		fprintf(oFile, "Post analysis:\n");
    fprintf(oFile, "########################################################\n");
    fprintf(oFile, "Number of iterations							:\t\t\t %d\n", BGMRESMDR->GIter);
    fprintf(oFile, "Number of cycles									:\t\t\t %d\n", BGMRESMDR->Cycle);
    fprintf(oFile, "Residual norm											:\t\t\t %.14e\n", BGMRESMDR->rvec[BGMRESMDR->GIter].real);
    fprintf(oFile, "Relative residual norm						:\t\t\t %.14e\n", BGMRESMDR->rvec[BGMRESMDR->GIter].real / BGMRESMDR->MaxNormb);
    fprintf(oFile, "Relative tolerance of convergence	:\t\t\t %.14e\n", BGMRESMDR->Ctol * BGMRESMDR->MaxNormb);
    // Total time on proc 0 ! I could take the max over all processes (average, min also possible)...
    fprintf(oFile, "Times on root\n");
    fprintf(oFile, "--------------------------------------------------------------------------------------\n");
    fprintf(oFile, "Times over processors			:\t\t\t MAX \t\t\t\t|\t MIN \t\t\t\t|\t AVERAGE\n");
    fprintf(oFile, "  Total                 	:\t\t\t %f \t|\t %f \t|\t %f s\n", BGMRESMDR->Timing.TotalTime, BGMRESMDR->Timing.TotalTime, BGMRESMDR->Timing.TotalTime);
    fprintf(oFile, "  Preconditioning       	:\t\t\t %f \t|\t %f \t|\t %f s\n", prec[0],				prec[1],			prec[2]);
    fprintf(oFile, "  SPMM                  	:\t\t\t %f \t|\t %f \t|\t %f s\n", spmm[0],				spmm[1],			spmm[2]);
    fprintf(oFile, "  SPMM Diag             	:\t\t\t %f \t|\t %f \t|\t %f s\n", spmmD[0],			spmmD[1],			spmmD[2]);
    fprintf(oFile, "  SPMM Off Diag         	:\t\t\t %f \t|\t %f \t|\t %f s\n", spmmOD[0],			spmmOD[1],		spmmOD[2]);
    fprintf(oFile, "  SPMM Comm             	:\t\t\t %f \t|\t %f \t|\t %f s\n", spmmComm[0],		spmmComm[1],	spmmComm[2]);
    fprintf(oFile, "  DeflationComputation  	:\t\t\t %f \t|\t %f \t|\t %f s\n", defcomp[0],		defcomp[1], 	defcomp[2]);
    fprintf(oFile, "  DeflationApplication  	:\t\t\t %f \t|\t %f \t|\t %f s\n", defapp[0],			defapp[1],		defapp[2]);
    fprintf(oFile, "  Orthogonalization     	:\t\t\t %f \t|\t %f \t|\t %f s\n", ortho[0],			ortho[1],			ortho[2]);
    fprintf(oFile, "  Orthogonalization Comm	:\t\t\t %f \t|\t %f \t|\t %f s\n", orthocomm[0],	orthocomm[1],	orthocomm[2]);
    fprintf(oFile, "  HessenbergOperations  	:\t\t\t %f \t|\t %f \t|\t %f s\n", hess[0],				hess[1],			hess[2]);
    fprintf(oFile, "--------------------------------------------------------------------------------------\n");
    fprintf(oFile, "Flops/sec: %f GFlops/s\n", ((double)BGMRESMDR->flops)/BGMRESMDR->Timing.TotalTime * (1e-9));
    fprintf(oFile, "--------------------------------------------------------------------------------------\n");
    for (int j = 0; j < BGMRESMDR->GIter; j++)
	    fprintf(oFile, "RelResidual(%d) =\t\t\t %.14e \n", j + 1, BGMRESMDR->rvec[j].real);
    for (int j = 0; j < BGMRESMDR->GIter; j++)
	    fprintf(oFile, "BlockSize(%d) =\t\t\t %d \n", j + 1, BGMRESMDR->BSize[j]);
    fprintf(oFile,"########################################################\n");
    fclose(oFile);
  }
	return 0;
}
