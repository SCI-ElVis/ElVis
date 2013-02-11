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

#ifndef _PX_EXTENSION_CUDA_INTERFACE_CU
#define _PX_EXTENSION_CUDA_INTERFACE_CU

//#include "PXSimplex.cu"
#include <optix_cuda.h>
#include <optix_math.h>
#include <ElVis/Core/matrix.cu>
#include <optixu/optixu_aabb.h>
#include <ElVis/Core/CutSurfacePayloads.cu>
#include <ElVis/Core/typedefs.cu>
#include <ElVis/Core/util.cu>


#include <ElVis/Core/Interval.hpp>
#include <Fundamentals/PX.h>

__device__ int Dim;
__device__ int StateRank;
__device__ int nFaceVertex;
__device__ int nbfQFace;
__device__ int faceType;
__device__ int faceOrder;
__device__ int porderFace;

// physical constants
__device__ PX_REAL SpecificHeatRatio;
__device__ PX_REAL GasConstant;


#include "PXOptiXCudaCommon.cu"


__device__ ElVisFloat* PXSimplexSolutionBuffer;
__device__ ElVisFloat* PXSimplexCoordinateBuffer;
__device__ ElVisFloat* PXSimplexFaceCoordinateBuffer;
__device__ PX_FaceData* PXSimplexFaceDataBuffer;
__device__ ElVisFloat* PXSimplexBoundingBoxBuffer;
__device__ PX_EgrpData* PXSimplexEgrpDataBuffer;
__device__ unsigned int* PXSimplexGlobalElemToEgrpElemBuffer;

__device__ PX_SolutionOrderData* PXSimplexAttachDataBuffer;
__device__ ElVisFloat* PXSimplexAttachmentBuffer;

__device__ ElVisFloat* PXSimplexShadowCoordinateBuffer;
__device__ unsigned int* PXSimplexEgrpToShadowIndexBuffer;
__device__ ElVisFloat* PXSimplexPatchCoordinateBuffer;
__device__ PX_REAL* PXSimplexKnownPointBuffer;
__device__ PX_REAL* PXSimplexBackgroundCoordinateBuffer;
__device__ char* PXSimplexCutCellBuffer;
__device__ unsigned int* PXSimplexGlobalElemToCutCellBuffer;



/// Evaluate the given field at the specified location in world space coordinates.
///
/// \param elementId The element's id.
/// \param elementType the element's type.
/// \param fieldId The field to evaluate.
/// \param worldPoint The point in world space at which to evaluate the point.l
/// \param initialGuessProvided Flags whether initialGuess is a valid starting point for the world point's reference space coordinates.
/// \param initalGuess On input, an initial guess for the world point's reference coordinates. On output, the reference coordinates calculated for worldPoint.
__device__ ElVisFloat EvaluateFieldCuda(unsigned int elementId, unsigned int elementType, int fieldId, const ElVisFloat3& worldPoint, ElVis::ReferencePointParameterType referenceType, ElVisFloat3& initialGuess )
{
  int egrp = PXSimplexGlobalElemToEgrpElemBuffer[2*elementId];
  int elem = PXSimplexGlobalElemToEgrpElemBuffer[2*elementId+1];

  int solnIndexStart = PXSimplexEgrpDataBuffer[egrp].egrpSolnCoeffStartIndex;
  //int nbfQ;
  int nbf = (int) PXSimplexEgrpDataBuffer[egrp].orderData.nbf;

  int nbfQ = (int) PXSimplexEgrpDataBuffer[egrp].typeData.nbf;
  int geomIndexStart = PXSimplexEgrpDataBuffer[egrp].egrpGeomCoeffStartIndex;
  ElVisFloat* localCoord = &PXSimplexCoordinateBuffer[Dim*geomIndexStart + elem*Dim*nbfQ];


  //LinearSimplexGlob2Ref(Dim, localCoord, xglobal, xref);

    bool initialGuessProvided = (referenceType == ElVis::eReferencePointIsInitialGuess);
  // ELVIS_PRINTF("egrp=%d,elem=%d,orderQ = %d, nbfQ = %d, result=%.8E\n",egrp,elem,orderQ, nbfQ, result);
  //ELVIS_PRINTF("egrp=%d,elem=%d,order = %d, nbf = %d, result=%.8E\n",egrp,elem,order, nbf, result);
  PX_EgrpData const *egrpData = &(PXSimplexEgrpDataBuffer[egrp]);
  if(egrpData->cutCellFlag != (char) 1){
    PX_REAL xref[3] = {initialGuess.x, initialGuess.y, initialGuess.z};
    PX_REAL xglobal[3] = {worldPoint.x, worldPoint.y, worldPoint.z};
    PXError(PXGlob2RefFromCoordinates2(&(egrpData->typeData), localCoord, xglobal, xref, (enum PXE_Boolean) initialGuessProvided, PXE_False));
    //ElVisFloat3 refPoint = MakeFloat3(xref[0],xref[1],xref[2]);
    initialGuess.x = xref[0];
    initialGuess.y = xref[1];
    initialGuess.z = xref[2];
  }else{
    int shadowIndexStart = PXSimplexEgrpToShadowIndexBuffer[egrp];
    localCoord = ((ElVisFloat*)&PXSimplexShadowCoordinateBuffer[0]) + shadowIndexStart + elem*DIM3D*SHADOW_NBF;
  }

  PX_SolutionOrderData *attachData = NULL;
  ElVisFloat* localSolution = &PXSimplexSolutionBuffer[StateRank*solnIndexStart + elem*StateRank*nbf];
  if(fieldId < 0){
    attachData = &PXSimplexAttachDataBuffer[egrp];
    localSolution = &PXSimplexAttachmentBuffer[elementId*((int)attachData->nbf)];
  }

    
  return EvaluateField(egrpData, attachData, localSolution, localCoord, StateRank, fieldId, worldPoint, initialGuess);
  //return MAKE_FLOAT(0.0);
  //return CalculateFieldValueCuda(elementId, elementType, 0, worldPoint);
}



__device__ ElVisFloat3 EvaluateNormalCuda(unsigned int elementId, unsigned int elementType, int fieldId, const ElVisFloat3& worldPoint)
{
  int egrp = PXSimplexGlobalElemToEgrpElemBuffer[2*elementId];
  int elem = PXSimplexGlobalElemToEgrpElemBuffer[2*elementId+1];

  int nbfQ = (int) PXSimplexEgrpDataBuffer[egrp].typeData.nbf;
  int geomIndexStart = PXSimplexEgrpDataBuffer[egrp].egrpGeomCoeffStartIndex;
  ElVisFloat* localCoord = &PXSimplexCoordinateBuffer[Dim*geomIndexStart + elem*Dim*nbfQ];

  if(PXSimplexEgrpDataBuffer[egrp].cutCellFlag == (char) 1){
    int shadowIndexStart = PXSimplexEgrpToShadowIndexBuffer[egrp];
    localCoord = ((ElVisFloat*)&PXSimplexShadowCoordinateBuffer[0]) + shadowIndexStart + elem*DIM3D*SHADOW_NBF;
  }


  int solnIndexStart = PXSimplexEgrpDataBuffer[egrp].egrpSolnCoeffStartIndex;
  //int nbfQ;
  int nbf = (int) PXSimplexEgrpDataBuffer[egrp].orderData.nbf;
  ElVisFloat* localSolution = &PXSimplexSolutionBuffer[StateRank*solnIndexStart + elem*StateRank*nbf];
  PX_SolutionOrderData *attachData = NULL;
  if(fieldId < 0){
    attachData = &PXSimplexAttachDataBuffer[egrp];
    localSolution = &PXSimplexAttachmentBuffer[elementId*((int)attachData->nbf)];
  }

  ElVisFloat3 result;

  EvaluateFieldGradient(&(PXSimplexEgrpDataBuffer[egrp]), attachData, localSolution, localCoord, StateRank, fieldId, worldPoint, result);

  return result;
  //return MakeFloat3(0.0,0.0,0.0);
}

__device__
void EstimateRangeCuda(unsigned int elementId, unsigned int elementType,  int fieldId,
                                          const ElVisFloat3& p0, const ElVisFloat3& p1,
                                          ElVis::Interval<ElVisFloat>& result)
{
}

#endif //end _PX_EXTENSION_INTERFACE_CU
