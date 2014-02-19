///////////////////////////////////////////////////////////////////////////////
//
// The MIT License
//
// Copyright (c) 2006 Scientific Computing and Imaging Institute,
// University of Utah (USA)
//
// License for the specific language governing rights and limitations under
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef _PXOPTIXCUDACOMMON_CU
#define _PXOPTIXCUDACOMMON_CU

#include <float.h>
#include <optix_cuda.h>
#include <optix_math.h>
#include <ElVis/Core/matrix.cu>
#include <optixu/optixu_aabb.h>
#include <ElVis/Core/CutSurfacePayloads.cu>
#include <ElVis/Core/typedefs.cu>
#include <ElVis/Core/util.cu>

#include <Fundamentals/PX.h>
#include <Fundamentals/PXError.h>

#include <ElVis/Extensions/ProjectXExtension/PXStructDefinitions.h>


#ifdef RESTRICT
#undef RESTRICT
#define RESTRICT __restrict__
#endif

#ifdef unlikely
#undef unlikely
#define unlikely
#endif

#ifdef likely
#undef likely
#define likely
#endif

// #define PXErrorReturn(X) (X)
// #define PXError(X) (X)
// #define PXErrorBreak(X) (X)
// #define PXErrorReturnCode(X) (X)
// #define PXErrorReturnSilent(X) (X)

ELVIS_DEVICE void PXErrorReport( const char *file, const int line, const char *call, const int ierr){
  //ELVIS_PRINTF("Error %d has occured.\n File : %s  Line : %d\n Call : %s\n", ierr, file, line, call);
  ELVIS_PRINTF("Error %d has occured.\n Line : %d\n", ierr, line);
}

#if PX_DEBUG_MODE == 1
#define PXErrorDebug(X) PXError(X)
#else
#define PXErrorDebug(X) (X)
#endif

#include "PXShape_Elvis.cu"
#include "PXCoordinates_Elvis.cu"
#include "PXNormal_Elvis.cu"



ELVIS_DEVICE void EvaluateFieldGradient(PX_EgrpData const * egrpData, PX_SolutionOrderData const *attachData, ElVisFloat const * localSolution, ElVisFloat const * localCoord, int StateRank, int fieldId, const ElVisFloat3& worldPoint, ElVisFloat3& gradient){

  //PXErrorReturn( PXJacobianElementGivenGradient2(pg->ElementGroup[egrp].type, Dim, nbfQ, ResElemData->nodeCoordinates, xref, NULL, &J, ijacp, gphiQStart+Dim*nbfQ*iquad) );
  PX_REAL xref[3];// = {0.25, 0.25, 0.25};
  PX_REAL xglobal[3] = {worldPoint.x, worldPoint.y, worldPoint.z};
  PX_REAL iJac[9];
  PX_REAL gphi[DIM3D*SOLN_MAX_NBF];
  PX_REAL phix[DIM3D*SOLN_MAX_NBF];

  if(egrpData->cutCellFlag == (char) 1){
    LinearSimplexGlob2Ref(3, localCoord, xglobal, xref);
  }else{
    PXError(PXGlob2RefFromCoordinates2(egrpData->elemData, localCoord, xglobal, xref, PXE_False, PXE_False));
  }

  int nbfQ = (int) egrpData->elemData.nbf;
  enum PXE_SolutionOrder orderQ = (enum PXE_SolutionOrder) egrpData->elemData.order;
  int qorder = (int) egrpData->elemData.qorder;
  PXGradientsElem_Solution<PX_REAL>(orderQ, qorder, xref, gphi );
  PXError(PXJacobianElementFromCoordinatesGivenGradient2<PX_REAL>((enum PXE_ElementType) egrpData->elemData.type, nbfQ, localCoord, xref, NULL, NULL, iJac, gphi, PXE_False));

  enum PXE_SolutionOrder order = (enum PXE_SolutionOrder) egrpData->solData.order;
  int porder = (int) egrpData->solData.porder;
  int nbf = (int) egrpData->solData.nbf;
  int index = fieldId;
  if(fieldId < 0){
    //may need different parameters for plotting attachments
    order = (enum PXE_SolutionOrder) attachData->order;
    nbf = (int) attachData->nbf;
    porder = (int) attachData->porder;
    index = 0;
    StateRank = 1;
  }

  if(orderQ != order)
    PXGradientsElem_Solution<PX_REAL>(order, porder, xref, gphi );

  PXPhysicalGradientsGivenGradients<PX_REAL>(order, nbf, iJac, gphi, phix);

  gradient.x = 0.0;
  gradient.y = 0.0;
  gradient.z = 0.0;
  if(fieldId < MAX_STATERANK){  
    //int index = fieldId;
    int j;
    PX_REAL const * phixx = &(phix[0]);
    PX_REAL const * phixy = &(phix[0]) + nbf;
    PX_REAL const * phixz = &(phix[0]) + 2*nbf;
    for(j=0; j<nbf; j++){
      gradient.x += localSolution[j*StateRank + index]*phixx[j];
      gradient.y += localSolution[j*StateRank + index]*phixy[j];
      gradient.z += localSolution[j*StateRank + index]*phixz[j];
    }
  }else if(fieldId != 10){
    //10 is geometric jacobian; no gradient info available for that

    /* for now, only supporting computations on basic
       flow quantities (i.e., NOT rho*nutilde NOR artifical visc */
    PX_REAL stateGradients[DIM3D*FLOW_RANK] = {0.0};
    PX_REAL *stateGradx = stateGradients;
    PX_REAL *stateGrady = stateGradients + FLOW_RANK;
    PX_REAL *stateGradz = stateGradients + 2*FLOW_RANK;
    PX_REAL const * phixx = &(phix[0]);
    PX_REAL const * phixy = &(phix[0]) + nbf;
    PX_REAL const * phixz = &(phix[0]) + 2*nbf;
    PX_REAL *phi = gphi; //reuse storage
    PXShapeElem_Solution<PX_REAL>(order, porder, xref, phi );

    /* compute state, state gradients */
    PX_REAL state[FLOW_RANK] = {0.0};
    int jbf, iState;
    for(iState=0; iState<FLOW_RANK; iState++){
      for(jbf=0; jbf<nbf; jbf++){
        state[iState] += localSolution[jbf*StateRank + iState]*phi[jbf];
        stateGradx[iState] += localSolution[jbf*StateRank + iState]*phixx[jbf];
        stateGrady[iState] += localSolution[jbf*StateRank + iState]*phixy[jbf];
        stateGradz[iState] += localSolution[jbf*StateRank + iState]*phixz[jbf];
      }
    }

    /* compute derived quantities */
    PX_REAL SpecificHeatRatio = 1.4;
    PX_REAL gmi = SpecificHeatRatio - 1.0;
    PX_REAL irho = 1.0/state[0];
    PX_REAL vmag, p; //magnitude of velocity, pressure, speed of sound
    PX_REAL v2, q, E, e, a, a2, ia;
    PX_REAL vel[DIM3D] = {state[1]*irho, state[2]*irho, state[3]*irho};
    
    //q = sqrt(state[1]*state[1]+state[2]*state[2]+state[3]*state[3])/state[0];
    //p = gmi*(state[FLOW_RANK-1]-q*q*state[0]/2.0);
    v2 = vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2];
    q = 0.5*v2;
    E = irho*state[FLOW_RANK-1];
    e = E-q;
   
    p = state[0]*gmi*e; //pressure

    a2 = SpecificHeatRatio*gmi*e;
    a = sqrt(a2);
    ia = 1.0/a;
    vmag = sqrt(v2);

    //M = vmag*ia;

    /* done if physicality check fails */
    if(p>0.0 && state[0] > 0.0){ //if not, leave result as 0.0
      //c = sqrt(SpecificHeatRatio*p/state[0]);
      PX_REAL a_U[FLOW_RANK];
      PX_REAL result_U[FLOW_RANK] = {0.0};
      PX_REAL temp;

      /* compute gradients */
      switch(fieldId){
      case 7: //mach number
	temp = 0.5*ia*SpecificHeatRatio*gmi*irho;
	a_U[0] = temp*(v2-E);
	a_U[1] = -temp*vel[0];
	a_U[2] = -temp*vel[1];
	a_U[3] = -temp*vel[2];
	a_U[4] = temp;

	for(iState=0; iState < FLOW_RANK; iState++){
	  result_U[iState] = -vmag*ia*ia*a_U[iState];
	}
	result_U[0] -= ia*vmag*irho;
	result_U[1] += ia*irho*vel[0]/vmag;
	result_U[2] += ia*irho*vel[1]/vmag;
	result_U[3] += ia*irho*vel[2]/vmag;

	//result = M;
	break;
      case 8: //magnitude of velocity
	result_U[0] = -2*v2*irho;
	result_U[1] = 2*vel[0]*irho;
	result_U[2] = 2*vel[1]*irho;
	result_U[3] = 2*vel[2]*irho;
	result_U[4] = 0.0;
	break;
      case 9: //pressure 
	//result = p;
	result_U[0] = gmi*q;
	result_U[1] = -gmi*vel[0];
	result_U[2] = -gmi*vel[1];
	result_U[3] = -gmi*vel[2];
	result_U[4] = gmi;
	break;
      default:
	//result = -10.0;
	break;
      }
      /* fill in gradient */
      for(iState=0; iState<FLOW_RANK; iState++){
        gradient.x += stateGradx[iState]*result_U[iState];
        gradient.y += stateGrady[iState]*result_U[iState];
        gradient.z += stateGradz[iState]*result_U[iState];
      }
    }

  }

}

ELVIS_DEVICE ElVisFloat EvaluateField(PX_EgrpData const& egrpData, PX_SolutionOrderData const *attachData, ElVisFloat const * localSolution, ElVisFloat const * localCoord, int StateRank, int fieldId, const ElVisFloat3& worldPoint, const ElVisFloat3& refPoint){
  PX_REAL xref[3] = {refPoint.x, refPoint.y, refPoint.z};
  PX_REAL phi[SOLN_MAX_NBF];
  for(int j = 0; j < SOLN_MAX_NBF; j++ ) phi[j] = 0;

  ELVIS_PRINTF("EvaluateField: xref=%f, yref=%f, zref=%f\n", refPoint.x, refPoint.y, refPoint.z);

/*
  if(egrpData.cutCellFlag == (char) 1){
      PX_REAL xglobal[3] = {worldPoint.x, worldPoint.y, worldPoint.z};
      //for cut elements, localCoord contains shadow coordinates
      //PXError(PXGlob2RefFromCoordinates2(&(egrpData->elemData), localCoord, xglobal, xref, PXE_False));
      //shadow element assumed to be PXE_UniformTetQ1
      LinearSimplexGlob2Ref(3, localCoord, xglobal, xref);
  }
*/
  /* set up basis parameters */
  enum PXE_SolutionOrder order = egrpData.solData.order;
  int nbf = egrpData.solData.nbf;
  int porder = egrpData.solData.porder;

  /*
  if(fieldId < 0 && attachData != NULL){
      //may need different parameters for plotting attachments
      order = attachData->order;
      nbf = attachData->nbf;
      porder = attachData->porder;
  }
*/
  /* evaluate basis */
  PXShapeElem_Solution<PX_REAL>(order, porder, xref, phi);

  ELVIS_PRINTF("EvaluateField: fieldId = %d, SOLN_MAX_NBF=%d, nbf=%d\n", fieldId, SOLN_MAX_NBF, nbf);
  ElVisFloat result = MAKE_FLOAT(0.0);
  for(int j=0; j<nbf; j++)
      result += localSolution[j*StateRank + fieldId]*phi[j];

  return result;

/*
  if(fieldId >= 0){
      if(fieldId < MAX_STATERANK){
          int index = fieldId;
          for(int j=0; j<nbf; j++){
              result += localSolution[j*StateRank + index]*phi[j];
          }
      }else if(fieldId == 10){
          //geometric jacobian
          PX_REAL gphi[DIM3D*MAX_NBF];
          PX_REAL J;
          enum PXE_SolutionOrder orderQ = egrpData.elemData.order;
          int qorder = egrpData.elemData.qorder;
          PXGradientsElem<PX_REAL>(orderQ, qorder, xref, gphi);
          int nbfQ = egrpData.elemData.nbf;

          PXJacobianElementFromCoordinatesGivenGradient2<PX_REAL>(egrpData.elemData.type, nbfQ, localCoord, xref, NULL, &J, NULL, gphi, PXE_True);
          result = J;
      }else{
          PX_REAL state[FLOW_RANK] = {0.0};
          int jbf, iState;
          for(iState=0; iState<FLOW_RANK; iState++){
              for(jbf=0; jbf<nbf; jbf++){
                  state[iState] += localSolution[jbf*StateRank + iState]*phi[jbf];
              }
          }

          PX_REAL SpecificHeatRatio = 1.4;
          PX_REAL gmi = SpecificHeatRatio - 1.0;
          PX_REAL irho = 1.0/state[0];
          PX_REAL vmag, p, M; //magnitude of velocity, pressure, speed of sound
          PX_REAL v2, q, E, e, a, ia;
          PX_REAL vel[DIM3D] = {state[1]*irho, state[2]*irho, state[3]*irho};

          v2 = vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2];
          q = 0.5*v2;
          E = irho*state[FLOW_RANK-1];
          e = E-q;

          p = state[0]*gmi*e; //pressure

          a = sqrt(SpecificHeatRatio*gmi*e);
          vmag = sqrt(v2);
          ia = 1.0/a;

          M = vmag*ia;


          //q = sqrt(state[1]*state[1]+state[2]*state[2]+state[3]*state[3])/state[0];
          //p = gmi*(state[FLOW_RANK-1]-q*q*state[0]/2.0);
          if(p>0.0 && state[0] > 0.0){ //if not, leave result as 0.0
              //c = sqrt(SpecificHeatRatio*p/state[0]);

              switch(fieldId){
                  case 7: //mach number
                      //result = q/c;
                      result = M;
                      break;
                  case 8:
                      //result = q;
                      result = vmag;
                      break;
                  case 9:
                      //result = p;
                      result = p;
                      break;
                  default:
                      result = -10.0;
              }
          }

      }

  }else{
      //distance function
      for(int j=0; j<nbf; j++){
          result += localSolution[j]*phi[j];
      }

  }

  return result; 
  */
}

#endif //_PXOPTIXCUDACOMMON_CU
