/**
 * \file   test_ecg_petsc_ela.c
 * \author Olivier Tissot
 * \date   2018/07/26
 * \brief  Benchmark with an elasticity matrix generated by PETSc
 * \note   Modification of hpddm/examples/petsc/ex56.c file that can be found
 *         at https://github.com/hpddm/hpddm/blob/master/examples/petsc/ex56.c
 *         and whose original author is Pierre Jolivet - himself has modified
 *         the file src/ksp/ksp/examples/tutorials/ex56.c located in PETSc main
 *         directory
 * \details Be carefull to set MATSOLVERPACKAGE to MATSOLVERMKL_PARDISO
 */

/*****************************************************************************/
/*                                  INCLUDE                                  */
/*****************************************************************************/
/* STD */
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <math.h>
/* MPI */
#include <mpi.h>
/* MKL */
#include <mkl.h>

/* CPaLAMeM */
//#include <cpalamem_macro.h>
//#include <cpalamem_instrumentation.h>
#ifdef PETSC
#include <preAlps_cplm_petsc_interface.h>
/* Petsc */
#include <petscksp.h>
#endif
/* preAlps */
#include "operator.h"
#include "block_jacobi.h"
#include "ecg.h"
/*****************************************************************************/

/*****************************************************************************/
/*                            AUXILIARY FUNCTIONS                            */
/*****************************************************************************/
#ifdef PETSC
/** \brief Simple wrapper to PETSc MatMatMult */
void petsc_operator_apply(Mat A, double* V, double* AV, int M, int m, int n) {
  Mat V_petsc, AV_petsc;
  MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,M,n,V,&V_petsc);
  MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,M,n,AV,&AV_petsc);
  MatMatMult(A, V_petsc, MAT_REUSE_MATRIX, PETSC_DEFAULT, &AV_petsc);
  MatDestroy(&V_petsc);MatDestroy(&AV_petsc);
}
/** \brief Simple wrapper for applying PETSc preconditoner */
void petsc_precond_apply(Mat P, double* V, double* W, int M, int m, int n) {
  Mat V_petsc, W_petsc;
  MatCreateSeqDense(PETSC_COMM_SELF,m,n,V,&V_petsc);
  MatCreateSeqDense(PETSC_COMM_SELF,m,n,W,&W_petsc);
  MatMatSolve(P,V_petsc,W_petsc);
  MatDestroy(&V_petsc);MatDestroy(&W_petsc);
}
/** \brief PETSc utils */
PetscErrorCode elem_3d_elast_v_25(PetscScalar* dd)
{
    PetscErrorCode ierr;
    PetscScalar DD[] = {
        0.18981481481481474,       5.27777777777777568E-002,  5.27777777777777568E-002,  -5.64814814814814659E-002,
        -1.38888888888889072E-002, -1.38888888888889089E-002, -8.24074074074073876E-002, -5.27777777777777429E-002,
        1.38888888888888725E-002,  4.90740740740740339E-002,  1.38888888888889124E-002,  4.72222222222222071E-002,
        4.90740740740740339E-002,  4.72222222222221932E-002,  1.38888888888888968E-002,  -8.24074074074073876E-002,
        1.38888888888888673E-002,  -5.27777777777777429E-002, -7.87037037037036785E-002, -4.72222222222221932E-002,
        -4.72222222222222071E-002, 1.20370370370370180E-002,  -1.38888888888888742E-002, -1.38888888888888829E-002,
        5.27777777777777568E-002,  0.18981481481481474,       5.27777777777777568E-002,  1.38888888888889124E-002,
        4.90740740740740269E-002,  4.72222222222221932E-002,  -5.27777777777777637E-002, -8.24074074074073876E-002,
        1.38888888888888725E-002,  -1.38888888888889037E-002, -5.64814814814814728E-002, -1.38888888888888985E-002,
        4.72222222222221932E-002,  4.90740740740740478E-002,  1.38888888888888968E-002,  -1.38888888888888673E-002,
        1.20370370370370058E-002,  -1.38888888888888742E-002, -4.72222222222221932E-002, -7.87037037037036785E-002,
        -4.72222222222222002E-002, 1.38888888888888742E-002,  -8.24074074074073598E-002, -5.27777777777777568E-002,
        5.27777777777777568E-002,  5.27777777777777568E-002,  0.18981481481481474,       1.38888888888889055E-002,
        4.72222222222222002E-002,  4.90740740740740269E-002,  -1.38888888888888829E-002, -1.38888888888888829E-002,
        1.20370370370370180E-002,  4.72222222222222002E-002,  1.38888888888888985E-002,  4.90740740740740339E-002,
        -1.38888888888888985E-002, -1.38888888888888968E-002, -5.64814814814814520E-002, -5.27777777777777568E-002,
        1.38888888888888777E-002,  -8.24074074074073876E-002, -4.72222222222222002E-002, -4.72222222222221932E-002,
        -7.87037037037036646E-002, 1.38888888888888794E-002,  -5.27777777777777568E-002, -8.24074074074073598E-002,
        -5.64814814814814659E-002, 1.38888888888889124E-002,  1.38888888888889055E-002,  0.18981481481481474,
        -5.27777777777777568E-002, -5.27777777777777499E-002, 4.90740740740740269E-002,  -1.38888888888889072E-002,
        -4.72222222222221932E-002, -8.24074074074073876E-002, 5.27777777777777568E-002,  -1.38888888888888812E-002,
        -8.24074074074073876E-002, -1.38888888888888742E-002, 5.27777777777777499E-002,  4.90740740740740269E-002,
        -4.72222222222221863E-002, -1.38888888888889089E-002, 1.20370370370370162E-002,  1.38888888888888673E-002,
        1.38888888888888742E-002,  -7.87037037037036785E-002, 4.72222222222222002E-002,  4.72222222222222071E-002,
        -1.38888888888889072E-002, 4.90740740740740269E-002,  4.72222222222222002E-002,  -5.27777777777777568E-002,
        0.18981481481481480,       5.27777777777777568E-002,  1.38888888888889020E-002,  -5.64814814814814728E-002,
        -1.38888888888888951E-002, 5.27777777777777637E-002,  -8.24074074074073876E-002, 1.38888888888888881E-002,
        1.38888888888888742E-002,  1.20370370370370232E-002,  -1.38888888888888812E-002, -4.72222222222221863E-002,
        4.90740740740740339E-002,  1.38888888888888933E-002,  -1.38888888888888812E-002, -8.24074074074073876E-002,
        -5.27777777777777568E-002, 4.72222222222222071E-002,  -7.87037037037036924E-002, -4.72222222222222140E-002,
        -1.38888888888889089E-002, 4.72222222222221932E-002,  4.90740740740740269E-002,  -5.27777777777777499E-002,
        5.27777777777777568E-002,  0.18981481481481477,       -4.72222222222222071E-002, 1.38888888888888968E-002,
        4.90740740740740131E-002,  1.38888888888888812E-002,  -1.38888888888888708E-002, 1.20370370370370267E-002,
        5.27777777777777568E-002,  1.38888888888888812E-002,  -8.24074074074073876E-002, 1.38888888888889124E-002,
        -1.38888888888889055E-002, -5.64814814814814589E-002, -1.38888888888888812E-002, -5.27777777777777568E-002,
        -8.24074074074073737E-002, 4.72222222222222002E-002,  -4.72222222222222002E-002, -7.87037037037036924E-002,
        -8.24074074074073876E-002, -5.27777777777777637E-002, -1.38888888888888829E-002, 4.90740740740740269E-002,
        1.38888888888889020E-002,  -4.72222222222222071E-002, 0.18981481481481480,       5.27777777777777637E-002,
        -5.27777777777777637E-002, -5.64814814814814728E-002, -1.38888888888889037E-002, 1.38888888888888951E-002,
        -7.87037037037036785E-002, -4.72222222222222002E-002, 4.72222222222221932E-002,  1.20370370370370128E-002,
        -1.38888888888888725E-002, 1.38888888888888812E-002,  4.90740740740740408E-002,  4.72222222222222002E-002,
        -1.38888888888888951E-002, -8.24074074074073876E-002, 1.38888888888888812E-002,  5.27777777777777637E-002,
        -5.27777777777777429E-002, -8.24074074074073876E-002, -1.38888888888888829E-002, -1.38888888888889072E-002,
        -5.64814814814814728E-002, 1.38888888888888968E-002,  5.27777777777777637E-002,  0.18981481481481480,
        -5.27777777777777568E-002, 1.38888888888888916E-002,  4.90740740740740339E-002,  -4.72222222222222210E-002,
        -4.72222222222221932E-002, -7.87037037037036924E-002, 4.72222222222222002E-002,  1.38888888888888742E-002,
        -8.24074074074073876E-002, 5.27777777777777429E-002,  4.72222222222222002E-002,  4.90740740740740269E-002,
        -1.38888888888888951E-002, -1.38888888888888846E-002, 1.20370370370370267E-002,  1.38888888888888916E-002,
        1.38888888888888725E-002,  1.38888888888888725E-002,  1.20370370370370180E-002,  -4.72222222222221932E-002,
        -1.38888888888888951E-002, 4.90740740740740131E-002,  -5.27777777777777637E-002, -5.27777777777777568E-002,
        0.18981481481481480,       -1.38888888888888968E-002, -4.72222222222221932E-002, 4.90740740740740339E-002,
        4.72222222222221932E-002,  4.72222222222222071E-002,  -7.87037037037036646E-002, -1.38888888888888742E-002,
        5.27777777777777499E-002,  -8.24074074074073737E-002, 1.38888888888888933E-002,  1.38888888888889020E-002,
        -5.64814814814814589E-002, 5.27777777777777568E-002,  -1.38888888888888794E-002, -8.24074074074073876E-002,
        4.90740740740740339E-002,  -1.38888888888889037E-002, 4.72222222222222002E-002,  -8.24074074074073876E-002,
        5.27777777777777637E-002,  1.38888888888888812E-002,  -5.64814814814814728E-002, 1.38888888888888916E-002,
        -1.38888888888888968E-002, 0.18981481481481480,       -5.27777777777777499E-002, 5.27777777777777707E-002,
        1.20370370370370180E-002,  1.38888888888888812E-002,  -1.38888888888888812E-002, -7.87037037037036785E-002,
        4.72222222222222002E-002,  -4.72222222222222071E-002, -8.24074074074073876E-002, -1.38888888888888742E-002,
        -5.27777777777777568E-002, 4.90740740740740616E-002,  -4.72222222222222002E-002, 1.38888888888888846E-002,
        1.38888888888889124E-002,  -5.64814814814814728E-002, 1.38888888888888985E-002,  5.27777777777777568E-002,
        -8.24074074074073876E-002, -1.38888888888888708E-002, -1.38888888888889037E-002, 4.90740740740740339E-002,
        -4.72222222222221932E-002, -5.27777777777777499E-002, 0.18981481481481480,       -5.27777777777777568E-002,
        -1.38888888888888673E-002, -8.24074074074073598E-002, 5.27777777777777429E-002,  4.72222222222222002E-002,
        -7.87037037037036785E-002, 4.72222222222222002E-002,  1.38888888888888708E-002,  1.20370370370370128E-002,
        1.38888888888888760E-002,  -4.72222222222222002E-002, 4.90740740740740478E-002,  -1.38888888888888951E-002,
        4.72222222222222071E-002,  -1.38888888888888985E-002, 4.90740740740740339E-002,  -1.38888888888888812E-002,
        1.38888888888888881E-002,  1.20370370370370267E-002,  1.38888888888888951E-002,  -4.72222222222222210E-002,
        4.90740740740740339E-002,  5.27777777777777707E-002,  -5.27777777777777568E-002, 0.18981481481481477,
        1.38888888888888829E-002,  5.27777777777777707E-002,  -8.24074074074073598E-002, -4.72222222222222140E-002,
        4.72222222222222140E-002,  -7.87037037037036646E-002, -5.27777777777777707E-002, -1.38888888888888829E-002,
        -8.24074074074073876E-002, -1.38888888888888881E-002, 1.38888888888888881E-002,  -5.64814814814814589E-002,
        4.90740740740740339E-002,  4.72222222222221932E-002,  -1.38888888888888985E-002, -8.24074074074073876E-002,
        1.38888888888888742E-002,  5.27777777777777568E-002,  -7.87037037037036785E-002, -4.72222222222221932E-002,
        4.72222222222221932E-002,  1.20370370370370180E-002,  -1.38888888888888673E-002, 1.38888888888888829E-002,
        0.18981481481481469,       5.27777777777777429E-002,  -5.27777777777777429E-002, -5.64814814814814659E-002,
        -1.38888888888889055E-002, 1.38888888888889055E-002,  -8.24074074074074153E-002, -5.27777777777777429E-002,
        -1.38888888888888760E-002, 4.90740740740740408E-002,  1.38888888888888968E-002,  -4.72222222222222071E-002,
        4.72222222222221932E-002,  4.90740740740740478E-002,  -1.38888888888888968E-002, -1.38888888888888742E-002,
        1.20370370370370232E-002,  1.38888888888888812E-002,  -4.72222222222222002E-002, -7.87037037037036924E-002,
        4.72222222222222071E-002,  1.38888888888888812E-002,  -8.24074074074073598E-002, 5.27777777777777707E-002,
        5.27777777777777429E-002,  0.18981481481481477,       -5.27777777777777499E-002, 1.38888888888889107E-002,
        4.90740740740740478E-002,  -4.72222222222221932E-002, -5.27777777777777568E-002, -8.24074074074074153E-002,
        -1.38888888888888812E-002, -1.38888888888888846E-002, -5.64814814814814659E-002, 1.38888888888888812E-002,
        1.38888888888888968E-002,  1.38888888888888968E-002,  -5.64814814814814520E-002, 5.27777777777777499E-002,
        -1.38888888888888812E-002, -8.24074074074073876E-002, 4.72222222222221932E-002,  4.72222222222222002E-002,
        -7.87037037037036646E-002, -1.38888888888888812E-002, 5.27777777777777429E-002,  -8.24074074074073598E-002,
        -5.27777777777777429E-002, -5.27777777777777499E-002, 0.18981481481481474,       -1.38888888888888985E-002,
        -4.72222222222221863E-002, 4.90740740740740339E-002,  1.38888888888888829E-002,  1.38888888888888777E-002,
        1.20370370370370249E-002,  -4.72222222222222002E-002, -1.38888888888888933E-002, 4.90740740740740339E-002,
        -8.24074074074073876E-002, -1.38888888888888673E-002, -5.27777777777777568E-002, 4.90740740740740269E-002,
        -4.72222222222221863E-002, 1.38888888888889124E-002,  1.20370370370370128E-002,  1.38888888888888742E-002,
        -1.38888888888888742E-002, -7.87037037037036785E-002, 4.72222222222222002E-002,  -4.72222222222222140E-002,
        -5.64814814814814659E-002, 1.38888888888889107E-002,  -1.38888888888888985E-002, 0.18981481481481474,
        -5.27777777777777499E-002, 5.27777777777777499E-002,  4.90740740740740339E-002,  -1.38888888888889055E-002,
        4.72222222222221932E-002,  -8.24074074074074153E-002, 5.27777777777777499E-002,  1.38888888888888829E-002,
        1.38888888888888673E-002,  1.20370370370370058E-002,  1.38888888888888777E-002,  -4.72222222222221863E-002,
        4.90740740740740339E-002,  -1.38888888888889055E-002, -1.38888888888888725E-002, -8.24074074074073876E-002,
        5.27777777777777499E-002,  4.72222222222222002E-002,  -7.87037037037036785E-002, 4.72222222222222140E-002,
        -1.38888888888889055E-002, 4.90740740740740478E-002,  -4.72222222222221863E-002, -5.27777777777777499E-002,
        0.18981481481481469,       -5.27777777777777499E-002, 1.38888888888889072E-002,  -5.64814814814814659E-002,
        1.38888888888889003E-002,  5.27777777777777429E-002,  -8.24074074074074153E-002, -1.38888888888888812E-002,
        -5.27777777777777429E-002, -1.38888888888888742E-002, -8.24074074074073876E-002, -1.38888888888889089E-002,
        1.38888888888888933E-002,  -5.64814814814814589E-002, 1.38888888888888812E-002,  5.27777777777777429E-002,
        -8.24074074074073737E-002, -4.72222222222222071E-002, 4.72222222222222002E-002,  -7.87037037037036646E-002,
        1.38888888888889055E-002,  -4.72222222222221932E-002, 4.90740740740740339E-002,  5.27777777777777499E-002,
        -5.27777777777777499E-002, 0.18981481481481474,       4.72222222222222002E-002,  -1.38888888888888985E-002,
        4.90740740740740339E-002,  -1.38888888888888846E-002, 1.38888888888888812E-002,  1.20370370370370284E-002,
        -7.87037037037036785E-002, -4.72222222222221932E-002, -4.72222222222222002E-002, 1.20370370370370162E-002,
        -1.38888888888888812E-002, -1.38888888888888812E-002, 4.90740740740740408E-002,  4.72222222222222002E-002,
        1.38888888888888933E-002,  -8.24074074074073876E-002, 1.38888888888888708E-002,  -5.27777777777777707E-002,
        -8.24074074074074153E-002, -5.27777777777777568E-002, 1.38888888888888829E-002,  4.90740740740740339E-002,
        1.38888888888889072E-002,  4.72222222222222002E-002,  0.18981481481481477,       5.27777777777777429E-002,
        5.27777777777777568E-002,  -5.64814814814814659E-002, -1.38888888888888846E-002, -1.38888888888888881E-002,
        -4.72222222222221932E-002, -7.87037037037036785E-002, -4.72222222222221932E-002, 1.38888888888888673E-002,
        -8.24074074074073876E-002, -5.27777777777777568E-002, 4.72222222222222002E-002,  4.90740740740740269E-002,
        1.38888888888889020E-002,  -1.38888888888888742E-002, 1.20370370370370128E-002,  -1.38888888888888829E-002,
        -5.27777777777777429E-002, -8.24074074074074153E-002, 1.38888888888888777E-002,  -1.38888888888889055E-002,
        -5.64814814814814659E-002, -1.38888888888888985E-002, 5.27777777777777429E-002,  0.18981481481481469,
        5.27777777777777429E-002,  1.38888888888888933E-002,  4.90740740740740339E-002,  4.72222222222222071E-002,
        -4.72222222222222071E-002, -4.72222222222222002E-002, -7.87037037037036646E-002, 1.38888888888888742E-002,
        -5.27777777777777568E-002, -8.24074074074073737E-002, -1.38888888888888951E-002, -1.38888888888888951E-002,
        -5.64814814814814589E-002, -5.27777777777777568E-002, 1.38888888888888760E-002,  -8.24074074074073876E-002,
        -1.38888888888888760E-002, -1.38888888888888812E-002, 1.20370370370370249E-002,  4.72222222222221932E-002,
        1.38888888888889003E-002,  4.90740740740740339E-002,  5.27777777777777568E-002,  5.27777777777777429E-002,
        0.18981481481481474,       1.38888888888888933E-002,  4.72222222222222071E-002,  4.90740740740740339E-002,
        1.20370370370370180E-002,  1.38888888888888742E-002,  1.38888888888888794E-002,  -7.87037037037036785E-002,
        4.72222222222222071E-002,  4.72222222222222002E-002,  -8.24074074074073876E-002, -1.38888888888888846E-002,
        5.27777777777777568E-002,  4.90740740740740616E-002,  -4.72222222222222002E-002, -1.38888888888888881E-002,
        4.90740740740740408E-002,  -1.38888888888888846E-002, -4.72222222222222002E-002, -8.24074074074074153E-002,
        5.27777777777777429E-002,  -1.38888888888888846E-002, -5.64814814814814659E-002, 1.38888888888888933E-002,
        1.38888888888888933E-002,  0.18981481481481477,       -5.27777777777777568E-002, -5.27777777777777637E-002,
        -1.38888888888888742E-002, -8.24074074074073598E-002, -5.27777777777777568E-002, 4.72222222222222002E-002,
        -7.87037037037036924E-002, -4.72222222222222002E-002, 1.38888888888888812E-002,  1.20370370370370267E-002,
        -1.38888888888888794E-002, -4.72222222222222002E-002, 4.90740740740740478E-002,  1.38888888888888881E-002,
        1.38888888888888968E-002,  -5.64814814814814659E-002, -1.38888888888888933E-002, 5.27777777777777499E-002,
        -8.24074074074074153E-002, 1.38888888888888812E-002,  -1.38888888888888846E-002, 4.90740740740740339E-002,
        4.72222222222222071E-002,  -5.27777777777777568E-002, 0.18981481481481477,       5.27777777777777637E-002,
        -1.38888888888888829E-002, -5.27777777777777568E-002, -8.24074074074073598E-002, 4.72222222222222071E-002,
        -4.72222222222222140E-002, -7.87037037037036924E-002, 5.27777777777777637E-002,  1.38888888888888916E-002,
        -8.24074074074073876E-002, 1.38888888888888846E-002,  -1.38888888888888951E-002, -5.64814814814814589E-002,
        -4.72222222222222071E-002, 1.38888888888888812E-002,  4.90740740740740339E-002,  1.38888888888888829E-002,
        -1.38888888888888812E-002, 1.20370370370370284E-002,  -1.38888888888888881E-002, 4.72222222222222071E-002,
        4.90740740740740339E-002,  -5.27777777777777637E-002, 5.27777777777777637E-002,  0.18981481481481477,
    };
    PetscFunctionBeginUser;
    ierr = PetscMemcpy(dd, DD, sizeof(PetscScalar) * 576);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/** \brief Assemble the matrix */
PetscErrorCode AssembleSystem(Mat A, Vec b, PetscScalar soft_alpha, PetscScalar x_r,
                              PetscScalar y_r, PetscScalar z_r, PetscScalar r, PetscInt ne,
                              PetscMPIInt npe, PetscMPIInt rank, PetscInt nn, PetscInt m)
{
    PetscErrorCode ierr;
    PetscReal h = 1.0 / ne;
    PetscScalar DD[24][24], DD2[24][24];
    PetscScalar DD1[24][24];
    const PetscInt NP = (PetscInt)(PetscPowReal((PetscReal)npe, 1.0 / 3.0) + 0.5);
    const PetscInt ipx = rank % NP, ipy = (rank % (NP * NP)) / NP, ipz = rank / (NP * NP);
    const PetscInt Ni0 = ipx * (nn / NP), Nj0 = ipy * (nn / NP), Nk0 = ipz * (nn / NP);
    const PetscInt Ni1 = Ni0 + (m > 0 ? (nn / NP) : 0), Nj1 = Nj0 + (nn / NP), Nk1 = Nk0 + (nn / NP);
    const PetscInt NN = nn / NP, id0 = ipz * nn * nn * NN + ipy * nn * NN * NN + ipx * NN * NN * NN;
    PetscScalar vv[24], v2[24];
    PetscInt i, j, k;
    {
        ierr = elem_3d_elast_v_25((PetscScalar*)DD1);CHKERRQ(ierr);
        for (i = 0; i < 24; i++) {
            for (j = 0; j < 24; j++) {
                if (i < 12 || j < 12) {
                    if (i == j)
                        DD2[i][j] = 0.1 * DD1[i][j];
                    else
                        DD2[i][j] = 0.0;
                }
                else
                    DD2[i][j] = DD1[i][j];
            }
        }
        for (i = 0; i < 24; i++) {
            if (i % 3 == 0)
                vv[i] = h * h;
            else if (i % 3 == 1)
                vv[i] = 2.0 * h * h;
            else
                vv[i] = 0.0;
        }
        for (i = 0; i < 24; i++) {
            if (i % 3 == 0 && i >= 12)
                v2[i] = h * h;
            else if (i % 3 == 1 && i >= 12)
                v2[i] = 2.0 * h * h;
            else
                v2[i] = 0.0;
        }
    }
    ierr = MatZeroEntries(A);CHKERRQ(ierr);
    ierr = VecZeroEntries(b);CHKERRQ(ierr);
    PetscInt ii, jj, kk;
    for (i = Ni0, ii = 0; i < Ni1; i++, ii++) {
        for (j = Nj0, jj = 0; j < Nj1; j++, jj++) {
            for (k = Nk0, kk = 0; k < Nk1; k++, kk++) {
                PetscReal x = h * (PetscReal)i;
                PetscReal y = h * (PetscReal)j;
                PetscReal z = h * (PetscReal)k;
                PetscInt id = id0 + ii + NN * jj + NN * NN * kk;
                if (i < ne && j < ne && k < ne) {
                    // Insert some bubles with differents E coefficients
                    PetscReal radius1 = PetscSqrtReal((x - 0.25 + h / 2) * (x - 0.25 + h / 2) + (y - 0.25 + h / 2) * (y - 0.25 + h / 2) +
                                                      (z - 0.25 + h / 2) * (z - 0.25 + h / 2));
                    PetscReal radius2 = PetscSqrtReal((x - 0.75 + h / 2) * (x - 0.75 + h / 2) + (y - 0.25 + h / 2) * (y - 0.25 + h / 2) +
                                                      (z - 0.25 + h / 2) * (z - 0.25 + h / 2));
                    PetscReal radius3 = PetscSqrtReal((x - 0.25 + h / 2) * (x - 0.25 + h / 2) + (y - 0.75 + h / 2) * (y - 0.75 + h / 2) +
                                                      (z - 0.25 + h / 2) * (z - 0.25 + h / 2));
                    PetscReal radius4 = PetscSqrtReal((x - 0.75 + h / 2) * (x - 0.75 + h / 2) + (y - 0.75 + h / 2) * (y - 0.75 + h / 2) +
                                                      (z - 0.25 + h / 2) * (z - 0.25 + h / 2));
                    PetscReal radius5 = PetscSqrtReal((x - 0.25 + h / 2) * (x - 0.25 + h / 2) + (y - 0.25 + h / 2) * (y - 0.25 + h / 2) +
                                                      (z - 0.75 + h / 2) * (z - 0.75 + h / 2));
                    PetscReal radius6 = PetscSqrtReal((x - 0.75 + h / 2) * (x - 0.75 + h / 2) + (y - 0.25 + h / 2) * (y - 0.25 + h / 2) +
                                                      (z - 0.75 + h / 2) * (z - 0.75 + h / 2));
                    PetscReal radius7 = PetscSqrtReal((x - 0.25 + h / 2) * (x - 0.25 + h / 2) + (y - 0.75 + h / 2) * (y - 0.75 + h / 2) +
                                                      (z - 0.75 + h / 2) * (z - 0.75 + h / 2));
                    PetscReal radius8 = PetscSqrtReal((x - 0.75 + h / 2) * (x - 0.75 + h / 2) + (y - 0.75 + h / 2) * (y - 0.75 + h / 2) +
                                                      (z - 0.75 + h / 2) * (z - 0.75 + h / 2));

                    PetscReal alpha = 1.E0;
                    PetscInt jx, ix, idx[8];
                    idx[0] = id;
                    idx[1] = id + 1;
                    idx[2] = id + NN + 1;
                    idx[3] = id + NN;
                    idx[4] = id + NN * NN;
                    idx[5] = id + 1 + NN * NN;
                    idx[6] = id + NN + 1 + NN * NN;
                    idx[7] = id + NN + NN * NN;
                    if (i == Ni1 - 1 && Ni1 != nn) {
                        idx[1] += NN * (NN * NN - 1);
                        idx[2] += NN * (NN * NN - 1);
                        idx[5] += NN * (NN * NN - 1);
                        idx[6] += NN * (NN * NN - 1);
                    }
                    if (j == Nj1 - 1 && Nj1 != nn) {
                        idx[2] += NN * NN * (nn - 1);
                        idx[3] += NN * NN * (nn - 1);
                        idx[6] += NN * NN * (nn - 1);
                        idx[7] += NN * NN * (nn - 1);
                    }
                    if (k == Nk1 - 1 && Nk1 != nn) {
                        idx[4] += NN * (nn * nn - NN * NN);
                        idx[5] += NN * (nn * nn - NN * NN);
                        idx[6] += NN * (nn * nn - NN * NN);
                        idx[7] += NN * (nn * nn - NN * NN);
                    }
                    if (radius1 < r || radius5 < r) alpha = 1e-5;
                    if (radius2 < r || radius6 < r) alpha = 1e5;
                    if (radius3 < r || radius7 < r) alpha = 1e-5;
                    if (radius4 < r || radius8 < r) alpha = 1e5;

                    for (ix = 0; ix < 24; ix++) {
                        for (jx = 0; jx < 24; jx++) DD[ix][jx] = alpha * DD1[ix][jx];
                    }
                    if (k > 0) {
                        ierr = MatSetValuesBlocked(A, 8, idx, 8, idx, (const PetscScalar*)DD, ADD_VALUES);CHKERRQ(ierr);
                        ierr = VecSetValuesBlocked(b, 8, idx, (const PetscScalar*)vv, ADD_VALUES);CHKERRQ(ierr);
                    }
                    else {
                        for (ix = 0; ix < 24; ix++) {
                            for (jx = 0; jx < 24; jx++) DD[ix][jx] = alpha * DD2[ix][jx];
                        }
                        ierr = MatSetValuesBlocked(A, 8, idx, 8, idx, (const PetscScalar*)DD, ADD_VALUES);CHKERRQ(ierr);
                        ierr = VecSetValuesBlocked(b, 8, idx, (const PetscScalar*)v2, ADD_VALUES);CHKERRQ(ierr);
                    }
                }
            }
        }
    }
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
#endif

/******************************************************************************/

/******************************************************************************/
/*                                   MAIN                                     */
/******************************************************************************/
int main(int argc, char** argv) {
#ifdef PETSC
  /*================ Initialize ================*/
  PetscErrorCode ierr;
  PetscInt m, nn, M, j, k, ne = 4;
  PetscReal* coords;
  Vec x, rhs;
  Mat A;
  KSP ksp;
  PetscMPIInt npe, rank;
  PetscInitialize(&argc, &argv, NULL, NULL);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &npe);CHKERRQ(ierr);
  // Set default global parameters for both PETSc and ECG
  PetscBool verb;
  double tol = 1e-5;
  int maxIter = 1000;
  int enlFac = 1, bs_red = 0;
  // Timings
  double trash_t, trash_tg;
  double petsc_t = 0.E0;
  // double buildop_t = 0.E0, buildprec_t = 0.E0;
  double tot_t = 0.E0, op_t = 0.E0, prec_t = 0.E0;
  double totf_t = 0.E0, opf_t = 0.E0, precf_t = 0.E0;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Linear elasticity in 3D", "");
  /*================ Command line parameters ================*/
  {
    char nestring[256],trash[1024];
    PetscSNPrintf(nestring, sizeof nestring, "number of elements in each direction, ne+1 must be a multiple of %D (sizes^{1/3})",
                  (PetscInt)(PetscPowReal((PetscReal)npe, 1.0 / 3.0) + 0.5));
    PetscOptionsInt("-ne", nestring, "", ne, &ne, NULL);
    PetscOptionsGetString(NULL,NULL,"-ksp_monitor",trash,PETSC_MAX_PATH_LEN,&verb);
    PetscOptionsInt("-e", "enlarging factor", "", enlFac, &enlFac, NULL);
    PetscOptionsInt("-r", "dynamic reduction", "", bs_red, &bs_red, NULL);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  nn = ne + 1;
  M = 3 * nn * nn * nn;
  if (npe == 2) {
    if (rank == 1)
      m = 0;
    else
      m = nn * nn * nn;
    npe = 1;
  }
  else {
    m = nn * nn * nn / npe;
    if (rank == npe - 1) m = nn * nn * nn - (npe - 1) * m;
  }
  m *= 3;
  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  int i;
  {
    PetscInt Istart, Iend, jj, ic;
    const PetscInt NP = (PetscInt)(PetscPowReal((PetscReal)npe, 1.0 / 3.0) + 0.5);
    const PetscInt ipx = rank % NP, ipy = (rank % (NP * NP)) / NP, ipz = rank / (NP * NP);
    const PetscInt Ni0 = ipx * (nn / NP), Nj0 = ipy * (nn / NP), Nk0 = ipz * (nn / NP);
    const PetscInt Ni1 = Ni0 + (m > 0 ? (nn / NP) : 0), Nj1 = Nj0 + (nn / NP), Nk1 = Nk0 + (nn / NP);
    PetscInt *d_nnz, *o_nnz, osz[4] = {0, 9, 15, 19}, nbc;
    if (npe != NP * NP * NP) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "npe=%d: npe^{1/3} must be integer", npe);
    if (nn != NP * (nn / NP)) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "-ne %d: (ne+1)%(npe^{1/3}) must equal zero", ne);
    ierr = PetscMalloc1(m + 1, &d_nnz);CHKERRQ(ierr);
    ierr = PetscMalloc1(m + 1, &o_nnz);CHKERRQ(ierr);
    for (i = Ni0, ic = 0; i < Ni1; i++) {
      for (j = Nj0; j < Nj1; j++) {
        for (k = Nk0; k < Nk1; k++) {
          nbc = 0;
          if (i == Ni0 || i == Ni1 - 1) nbc++;
          if (j == Nj0 || j == Nj1 - 1) nbc++;
          if (k == Nk0 || k == Nk1 - 1) nbc++;
          for (jj = 0; jj < 3; jj++, ic++) {
            d_nnz[ic] = 3 * (27 - osz[nbc]);
            o_nnz[ic] = 3 * osz[nbc];
          }
        }
      }
    }
    if (ic != m) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "ic %D does not equal m %D", ic, m);
    ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
    ierr = MatSetSizes(A, m, m, M, M);CHKERRQ(ierr);
    ierr = MatSetBlockSize(A, 3);CHKERRQ(ierr);
    ierr = MatSetType(A, MATAIJ);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(A, 0, d_nnz);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(A, 0, d_nnz, 0, o_nnz);CHKERRQ(ierr);
    ierr = PetscFree(d_nnz);CHKERRQ(ierr);
    ierr = PetscFree(o_nnz);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A, &Istart, &Iend);CHKERRQ(ierr);
    if (m != Iend - Istart) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "m %D does not equal Iend %D - Istart %D", m, Iend, Istart);
    ierr = VecCreate(PETSC_COMM_WORLD, &x);CHKERRQ(ierr);
    ierr = VecSetSizes(x, m, M);CHKERRQ(ierr);
    ierr = VecSetBlockSize(x, 3);CHKERRQ(ierr);
    ierr = VecSetFromOptions(x);CHKERRQ(ierr);
    ierr = VecDuplicate(x, &rhs);CHKERRQ(ierr);
    ierr = PetscMalloc1(m + 1, &coords);CHKERRQ(ierr);
    coords[m] = -99.0;
    PetscReal h = 1.0 / ne;
    for (i = Ni0, ic = 0; i < Ni1; i++) {
      for (j = Nj0; j < Nj1; j++) {
        for (k = Nk0; k < Nk1; k++, ic++) {
          coords[3 * ic] = h * (PetscReal)i;
          coords[3 * ic + 1] = h * (PetscReal)j;
          coords[3 * ic + 2] = h * (PetscReal)k;
        }
      }
    }
  }
  PetscReal s_r =  1e7;
  PetscReal x_r = 0.5;
  PetscReal y_r = 0.5;
  PetscReal z_r = 0.5;
  PetscReal r   = 0.05;
  AssembleSystem(A, rhs, s_r, x_r, y_r, z_r, r, ne, npe, rank, nn, m);
  // MatScale(A, 100000.0);
  ierr = KSPSetOperators(ksp, A, A);CHKERRQ(ierr);

  /*================ Petsc solve ================*/
  Mat prec;
  // KSP ksp;
  KSP *subksp;
  PC pc, subpc;
  int first,nlocal;

  KSPSetType(ksp,KSPCG);
  KSPSetTolerances(ksp,tol,PETSC_DEFAULT,PETSC_DEFAULT,maxIter);
  KSPSetPCSide(ksp,PC_LEFT);
  KSPCGSetType(ksp,KSP_CG_SYMMETRIC);
  KSPSetNormType(ksp,KSP_NORM_UNPRECONDITIONED);
  KSPSetFromOptions(ksp);
  KSPSetUp(ksp);
  KSPGetPC(ksp,&pc);
  PCBJacobiGetSubKSP(pc,&nlocal,&first,&subksp);

  // Loop over the local blocks, setting various KSP options
  // for each block.
  // printf("nlocal: %d\n",nlocal);
  for (int i=0; i<nlocal; i++) {
    KSPGetPC(subksp[i],&subpc);
    PCSetUp(subpc);
    PCFactorGetMatrix(subpc,&prec);
    /* PCFactorSetMatSolverPackage(subpc,MATSOLVERMKL_PARDISO); */
  }

  VecSet(x,0e0);
  trash_t = MPI_Wtime();
  KSPSolve(ksp,rhs,x);
  petsc_t += MPI_Wtime() - trash_t;
  // VecSet(x,0e0);
  // KSPSolve(ksp,rhs,x);

  int its = -1;
  double rnorm = 0e0;
  KSPGetIterationNumber(ksp,&its);
  KSPGetResidualNorm(ksp,&rnorm);
  if (rank == 0) {
    printf("=== Petsc ===\n");
    printf("\titerations: %d\n",its);
    printf("\tnorm(res) : %e\n",rnorm);
    printf("\ttime      : %f s\n",petsc_t);
  }

  /*================ ECG solve ================*/
  // Restore the pointer
  double* rhs_s = NULL;
  VecGetArray(rhs,&rhs_s);

  preAlps_ECG_t ecg;
  // Set parameters
  ecg.comm = MPI_COMM_WORLD;
  ecg.globPbSize = M;
  ecg.locPbSize = m;
  ecg.maxIter = maxIter;
  ecg.enlFac = enlFac;
  ecg.tol = tol;
  ecg.ortho_alg = ORTHODIR;
  ecg.bs_red = (bs_red == 0 ? NO_BS_RED : ADAPT_BS);
  int rci_request = 0, stop = 0;
  double* sol = NULL;
  int* bs = NULL; // block size
  double* res = NULL; // residual
  sol = (double*) malloc(m*sizeof(double));
  bs = (int*) calloc(maxIter,sizeof(int));
  res = (double*) calloc(maxIter,sizeof(double));
  // Allocate memory and initialize variables
  MPI_Barrier(MPI_COMM_WORLD);
  trash_tg = MPI_Wtime();
  preAlps_ECGInitialize(&ecg,rhs_s,&rci_request);
  // Finish initialization
  if (rank == 0 && verb == PETSC_TRUE)
   printf("%3d ECG Residual norm %.12e\n",ecg.iter, ecg.res);
  // Finish initialization
  trash_t = MPI_Wtime();
  petsc_precond_apply(prec,ecg.R_p,ecg.P_p, M, m, enlFac);
  prec_t += MPI_Wtime() - trash_t;
  trash_t = MPI_Wtime();
  petsc_operator_apply(A, ecg.P_p, ecg.AP_p, M, m, enlFac);
  op_t += MPI_Wtime() - trash_t;
  // Main loop
  while (stop != 1) {
    preAlps_ECGIterate(&ecg,&rci_request);
    if (rci_request == 0) {
      trash_t = MPI_Wtime();
      petsc_operator_apply(A, ecg.P_p, ecg.AP_p, M, m, ecg.bs);
      op_t += MPI_Wtime() - trash_t;
    }
    else if (rci_request == 1) {
      preAlps_ECGStoppingCriterion(&ecg,&stop);
      bs[ecg.iter] = ecg.bs;
      res[ecg.iter] = ecg.res/ecg.normb;
      if (rank == 0 && verb == PETSC_TRUE)
        printf("%3d ECG Residual norm %.12e\n",ecg.iter, ecg.res);
      if (stop == 1) break;
      trash_t = MPI_Wtime();
      petsc_precond_apply(prec, ecg.AP_p, ecg.Z_p, M, m, ecg.bs);
      prec_t += MPI_Wtime() - trash_t;
    }
  }
  // Retrieve solution and free memory
  preAlps_ECGFinalize(&ecg,sol);
  MPI_Barrier(MPI_COMM_WORLD);
  tot_t += MPI_Wtime() - trash_tg;

  if (rank == 0) {
    printf("=== ECG ===\n");
    printf("\titerations: %d\n",ecg.iter);
    printf("\tnorm(res) : %e\n",ecg.res);
    printf("\tblock size: %d\n",ecg.bs);
    printf("\ttime      : %f s\n",tot_t);
  }

  /*================== Fused-ECG solve ==================*/
  preAlps_ECG_t ecg_f;
  // Set parameters
  ecg_f.comm = MPI_COMM_WORLD;
  ecg_f.globPbSize = M;
  ecg_f.locPbSize = m;
  ecg_f.maxIter = maxIter;
  ecg_f.enlFac = enlFac;
  ecg_f.tol = tol;
  ecg_f.ortho_alg = ORTHODIR_FUSED;
  ecg_f.bs_red = (bs_red == 0 ? NO_BS_RED : ADAPT_BS);
  stop = 0; rci_request = 0;
  if (sol != NULL) free(sol);
  int* bs_f = NULL; // block size
  double* res_f = NULL; // residual
  sol = (double*) malloc(m*sizeof(double));
  bs_f = (int*) calloc(maxIter,sizeof(int));
  res_f = (double*) calloc(maxIter,sizeof(double));

  // Allocate memory and initialize variables
  MPI_Barrier(MPI_COMM_WORLD);
  trash_tg = MPI_Wtime();
  preAlps_ECGInitialize(&ecg_f,rhs_s,&rci_request);
  // Finish initialization
  if (rank == 0 && verb == PETSC_TRUE)
    printf("%3d F-ECG Residual norm %.12e\n",ecg_f.iter, ecg_f.res);
  trash_t = MPI_Wtime();
  petsc_precond_apply(prec,ecg_f.R_p,ecg_f.P_p, M, m, enlFac);
  precf_t += MPI_Wtime() - trash_t;
  // Main loop
  while (rci_request != 1) {
    trash_t = MPI_Wtime();
    petsc_operator_apply(A, ecg_f.P_p, ecg_f.AP_p, M, m, ecg_f.bs);
    opf_t += MPI_Wtime() - trash_t;
    trash_t = MPI_Wtime();
    petsc_precond_apply(prec, ecg_f.AP_p, ecg_f.Z_p, M, m, ecg_f.bs);
    precf_t += MPI_Wtime() - trash_t;
    preAlps_ECGIterate(&ecg_f,&rci_request);
    bs_f[ecg_f.iter] = ecg_f.bs;
    res_f[ecg_f.iter] = ecg_f.res/ecg_f.normb;
    if (rank == 0 && verb == PETSC_TRUE)
      printf("%3d F-ECG Residual norm %.12e\n",ecg_f.iter, ecg_f.res);
  }
  // Retrieve solution and free memory
  preAlps_ECGFinalize(&ecg_f,sol);
  MPI_Barrier(MPI_COMM_WORLD);
  totf_t += MPI_Wtime() - trash_tg;

  if (rank == 0) {
    printf("=== F-ECG ===\n");
    printf("\titerations: %d\n",ecg_f.iter);
    printf("\tnorm(res) : %e\n",ecg_f.res);
    printf("\tblock size: %d\n",ecg_f.bs);
    printf("\ttime      : %f s\n",totf_t);
  }

  /*================ Finalize ================*/

  // Free PETSc structures
  // ierr = MatDestroy(&A);CHKERRQ(ierr);
  // ierr = VecDestroy(&x);CHKERRQ(ierr);
  // KSPDestroy(&ksp);

  // Free arrays
  if (sol != NULL) free(sol);

  PetscFinalize();
#endif
  return 0;
}
/******************************************************************************/
