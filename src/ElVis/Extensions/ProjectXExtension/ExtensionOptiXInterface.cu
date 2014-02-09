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

#ifndef _PX_EXTENSION_OPTIX_INTERFACE_CU
#define _PX_EXTENSION_OPTIX_INTERFACE_CU

#include <ElVis/Core/VolumeRenderingPayload.cu>
#include <Fundamentals/PX.h>
#include <ElVis/Core/OptixVariables.cu>

rtDeclareVariable(int, Dim, , );
rtDeclareVariable(int, StateRank, , );
//rtDeclareVariable(int, nFaceVertex, , );
//rtDeclareVariable(int, nbfQFace, , );
//rtDeclareVariable(int, faceType, , );
//rtDeclareVariable(int, faceOrder, , );
//rtDeclareVariable(int, porderFace, , );

/* physical constants */
//rtDeclareVariable(PX_REAL, SpecificHeatRatio, , );
//rtDeclareVariable(PX_REAL, GasConstant, , );



#include "PXOptiXCudaCommon.cu"
#include "PXCutCell_ElVis.cu"
#include <ElVis/Core/Interval.hpp>

rtBuffer<int> egrp2GlobalElemIndex;
rtBuffer<PX_EgrpData> PXSimplexEgrpDataBuffer;

rtBuffer<ElVisFloat> PXSimplexCoordinateBuffer;
rtBuffer<ElVisFloat> PXSimplexSolutionBuffer;

//rtBuffer<ElVisFloat> PXSimplexFaceCoordinateBuffer;
//rtBuffer<PX_FaceData> PXSimplexFaceDataBuffer;
//rtBuffer<uint> PXSimplexGlobalElemToEgrpElemBuffer;

//rtBuffer<PX_SolutionOrderData> PXSimplexAttachDataBuffer;
//rtBuffer<ElVisFloat> PXSimplexAttachmentBuffer;

//rtBuffer<ElVisFloat> PXSimplexShadowCoordinateBuffer;
//rtBuffer<uint> PXSimplexEgrpToShadowIndexBuffer;
//rtBuffer<ElVisFloat> PXSimplexPatchCoordinateBuffer;
//rtBuffer<PX_REAL> PXSimplexBackgroundCoordinateBuffer;
//rtBuffer<PX_REAL> PXSimplexKnownPointBuffer;
//rtBuffer<char> PXSimplexCutCellBuffer;
//rtBuffer<uint> PXSimplexGlobalElemToCutCellBuffer;

#define PX_USE_ISOSURF 1



ELVIS_DEVICE ElVisError EvaluateFace(GlobalFaceIdx faceId, const FaceReferencePoint& refPoint, WorldPoint& result)
{
  ELVIS_PRINTF("MCG EvaluateFace: Didn't know this was called yet!\n");
	return eConvergenceFailure;

//    ElVisFloat r = refPoint.x;
//    ElVisFloat s = refPoint.y;
//    PX_REAL phi[MAX_NBF_FACE];
//    PX_REAL xref[2] = {r,s};
    //PX_REAL xreflocal[2];
    //PX_REAL *nodeCoord = NULL; // = &PXSimplexFaceCoordinateBuffer[DIM3D*nbfQFace*faceId];
    //int i;

    // PX_FaceData * faceData = &PXSimplexFaceDataBuffer[faceId];
    // PXErrorDebug( PXFaceRef2ElemFaceRef<PX_REAL>( (enum PXE_Shape) faceData->shape, (int) faceData->orientation, xref, xreflocal) );

    //PXShapeFace<PX_REAL>((enum PXE_SolutionOrder) faceOrder, porderFace, xref, phi);

//    result.x = 0.0;
//    result.y = 0.0;
//    result.z = 0.0;
//    for(i=0; i<nbfQFace; i++)
//    {
//        result.x += nodeCoord[i*DIM3D+0]*phi[i];
//        result.y += nodeCoord[i*DIM3D+1]*phi[i];
//        result.z += nodeCoord[i*DIM3D+2]*phi[i];
//    }

//    return eNoError;
}


//ELVIS_DEVICE ElVisFloat EvaluateFieldOptiX(unsigned int elementId, unsigned int elementType, int fieldId, const ElVisFloat3& worldPoint, ElVis::ReferencePointParameterType referenceType, ElVisFloat3& referencePoint)
//{
//    int egrp = PXSimplexGlobalElemToEgrpElemBuffer[2*elementId];
//    int elem = PXSimplexGlobalElemToEgrpElemBuffer[2*elementId+1];

//    int nbfQ = (int) PXSimplexEgrpDataBuffer[egrp].typeData.nbf;
//    int geomIndexStart = PXSimplexEgrpDataBuffer[egrp].egrpGeomCoeffStartIndex;

//    ElVisFloat* localCoord = &PXSimplexCoordinateBuffer[Dim*geomIndexStart + elem*Dim*nbfQ];
//    if(PXSimplexEgrpDataBuffer[egrp].cutCellFlag == (char) 1)
//    {
//        int shadowIndexStart = PXSimplexEgrpToShadowIndexBuffer[egrp];
//        localCoord = ((ElVisFloat*)&PXSimplexShadowCoordinateBuffer[0]) + shadowIndexStart + elem*DIM3D*SHADOW_NBF;
//    }
//    else if(referenceType != ElVis::eReferencePointIsValid)
//    {
//        // The reference point is not valid, so calculate it.
//        PX_REAL xglobal[3] = {worldPoint.x, worldPoint.y, worldPoint.z};
//        PX_REAL xref[3];
//        PXError(PXGlob2RefFromCoordinates2(&(PXSimplexEgrpDataBuffer[egrp].typeData), localCoord, xglobal, xref, PXE_False, PXE_False));
//        referencePoint.x = xref[0];
//        referencePoint.y = xref[1];
//        referencePoint.z = xref[2];
//    }

//    int solnIndexStart = PXSimplexEgrpDataBuffer[egrp].egrpSolnCoeffStartIndex;
//    int nbf = (int) PXSimplexEgrpDataBuffer[egrp].orderData.nbf;
//    ElVisFloat* localSolution = &PXSimplexSolutionBuffer[StateRank*solnIndexStart + elem*StateRank*nbf];
//    PX_SolutionOrderData *attachData = NULL;
//    if(fieldId < 0){
//      attachData = &PXSimplexAttachDataBuffer[egrp];
//      localSolution = &PXSimplexAttachmentBuffer[elementId*((int)attachData->nbf)];
//    }
    
//    return EvaluateField(&(PXSimplexEgrpDataBuffer[egrp]), attachData, localSolution, localCoord, StateRank, fieldId, worldPoint, referencePoint);
//}









/////////////////////////////////////////////
// Required ElVis Interface
/////////////////////////////////////////////
/// \brief Converts a point from world space to the given element's reference space.
/// \param elementId The element's id.
/// \param elementType The element's type.
/// \param wp The world point to be converted.
/// \param referenceType Describes the meaning of result.  If eReferencePointIsInvalid then
///        result will only be used for output. If eReferencePointIsInitialGuess, then result is
///        a guess for the actual location of the reference point.
/// \param result On input, it can be an initial guess.  On output, the actual reference point corresponding to wp.
/// \returns
ELVIS_DEVICE ElVisError ConvertWorldToReferenceSpaceOptiX(int elementId, int elementType, const WorldPoint& worldPoint,
                                                          ElVis::ReferencePointParameterType referenceType, ReferencePoint& referencePoint)
{
    ELVIS_PRINTF("MCG ConvertWorldToReferenceSpaceOptiX: Element Id %d, intersection point (%f, %f, %f)\n",
                 elementId, worldPoint.x, worldPoint.y, worldPoint.z);

    if( referenceType != ElVis::eReferencePointIsValid )
    {
        int egrp = 0;
        while( elementId > egrp2GlobalElemIndex[egrp+1] ) { egrp++; }
        int elem = elementId - egrp2GlobalElemIndex[egrp];

        int nbfQ = PXSimplexEgrpDataBuffer[egrp].elemData.nbf;
        int geomIndexStart = PXSimplexEgrpDataBuffer[egrp].egrpGeomCoeffStartIndex;

        ELVIS_PRINTF("MCG ConvertWorldToReferenceSpaceOptiX: Dim=%d, geomIndexStart=%d, elem=%d, nbfQ=%d, idx=%d\n",
            Dim, geomIndexStart, elem, nbfQ, geomIndexStart + Dim*elem*nbfQ);

        ElVisFloat* localCoord = &PXSimplexCoordinateBuffer[geomIndexStart + Dim*elem*nbfQ];

        PX_REAL xglobal[3] = {worldPoint.x, worldPoint.y, worldPoint.z};
        PX_REAL xref[3] = {0,0,0};
        PXGlob2RefFromCoordinates2(PXSimplexEgrpDataBuffer[egrp].elemData, localCoord, xglobal, xref, PXE_False, PXE_False);
        referencePoint.x = xref[0];
        referencePoint.y = xref[1];
        referencePoint.z = xref[2];
    }

    return eNoError;
}

/// \brief Evaluates a scalar field at the given point.
/// \param elementId The element's id.
/// \param elementType The element's type.
/// \param fieldId The field to be evaluated.
/// \param point The point at which the field will be evaluated.
/// \param result The result of the evaluation.
/// \returns
///
/// This method is designed to be used for evaluation at a single reference point (x0, x1, x2) and at
/// an interval of the reference space ([x0_low, x0_high], [x1_low, x1_high], [x2_low, x2_high]).
template<typename PointType, typename ResultType>
ELVIS_DEVICE ElVisError SampleScalarFieldAtReferencePointOptiX(int elementId, int elementType, int fieldId,
                                                               const PointType& worldPoint,
                                                               const PointType& referencePoint,
                                                               ResultType& result)
{

    ELVIS_PRINTF("MCG SampleScalarFieldAtReferencePointOptiX: Element Id %d, intersection point (%f, %f, %f)\n",
                 elementId, worldPoint.x, worldPoint.y, worldPoint.z);

    int egrp = 0;
    while( elementId > egrp2GlobalElemIndex[egrp+1] ) { egrp++; }
    int elem = elementId - egrp2GlobalElemIndex[egrp];


//    int nbfQ = (int) PXSimplexEgrpDataBuffer[egrp].elemData.nbf;
//    int geomIndexStart = PXSimplexEgrpDataBuffer[egrp].egrpGeomCoeffStartIndex;

    ElVisFloat* localCoord = NULL; //&PXSimplexCoordinateBuffer[Dim*geomIndexStart + elem*Dim*nbfQ];
//    if(PXSimplexEgrpDataBuffer[egrp].cutCellFlag == (char) 1)
//    {
//        int shadowIndexStart = PXSimplexEgrpToShadowIndexBuffer[egrp];
//        localCoord = ((ElVisFloat*)&PXSimplexShadowCoordinateBuffer[0]) + shadowIndexStart + elem*DIM3D*SHADOW_NBF;
//    }

    int solnIndexStart = PXSimplexEgrpDataBuffer[egrp].egrpSolnCoeffStartIndex;
    int nbf = PXSimplexEgrpDataBuffer[egrp].solData.nbf;

    ELVIS_PRINTF("MCG SampleScalarFieldAtReferencePointOptiX: fieldId = %d, SOLN_MAX_NBF=%d, nbf=%d, idx=%d\n", fieldId, SOLN_MAX_NBF, nbf, solnIndexStart + elem*StateRank*nbf);

    ElVisFloat* localSolution = &PXSimplexSolutionBuffer[solnIndexStart + elem*StateRank*nbf];
    PX_SolutionOrderData *attachData = NULL;
//    if(fieldId < 0){
//      attachData = &PXSimplexAttachDataBuffer[egrp];
//      localSolution = &PXSimplexAttachmentBuffer[elementId*((int)attachData->nbf)];
//    }

    result = EvaluateField(PXSimplexEgrpDataBuffer[egrp], attachData, localSolution, localCoord, StateRank, fieldId, worldPoint, referencePoint);

    //result = localSolution[fieldId*nbf];
/*
    PX_REAL xref[3] = {referencePoint.x, referencePoint.y, referencePoint.z};
    PX_REAL phi[SOLN_MAX_NBF];
    for(int j = 0; j < SOLN_MAX_NBF; j++ ) phi[j] = 0;

    enum PXE_SolutionOrder order = PXSimplexEgrpDataBuffer[egrp].solData.order;
    int porder = PXSimplexEgrpDataBuffer[egrp].solData.porder;

    PXShapeElem_Solution<PX_REAL>(order, porder, xref, phi);

    result = 0;
    for(int j=0; j<nbf; j++)
        result += localSolution[j*StateRank + fieldId]*phi[j];
*/
    ELVIS_PRINTF("MCG SampleScalarFieldAtReferencePointOptiX: result=%f\n", result);

    return eNoError;

}


ELVIS_DEVICE ElVisError IsValidFaceCoordinate(GlobalFaceIdx faceId, const FaceReferencePoint& p, bool& result)
{
  ELVIS_PRINTF("MCG IsValidFaceCoordinate: Didn't know this was called yet!\n");
    ElVisFloat r = p.x;
    ElVisFloat s = p.y;
    result = r >= MAKE_FLOAT(0.0) &&
            s >= MAKE_FLOAT(0.0) &&
            (r+s) <= MAKE_FLOAT(1.0);

    return eNoError;
}

template<typename T>
ELVIS_DEVICE ElVisError EvaluateFaceJacobian(GlobalFaceIdx faceId, const FaceReferencePoint& p,
                                             T& dx_dr, T& dx_ds,
                                             T& dy_dr, T& dy_ds,
                                             T& dz_dr, T& dz_ds)
{

  ELVIS_PRINTF("MCG EvaluateFaceJacobian: Didn't know this was called yet!\n");
	   dx_dr = 0.0; dx_ds = 0.0;
	   dy_dr = 0.0; dy_ds = 0.0;
	   dz_dr = 0.0; dz_ds = 0.0;

   return eNoError;
/*
   ELVIS_PRINTF("EvaluateFaceJacobian");
   const T& r = p.x;
   const T& s = p.y;

#if PX_USE_ISOSURF
   T gphi[2*MAX_NBF_FACE];
   T *gphir = gphi;
   T *gphis = gphir + MAX_NBF_FACE;
   T xref[2] = {r,s};
   //T xreflocal[2];
   PX_REAL *nodeCoord = &PXSimplexFaceCoordinateBuffer[DIM3D*nbfQFace*faceId];
   int i;

   // PX_FaceData * faceData = &PXSimplexFaceDataBuffer[faceId];
   // PXErrorDebug( PXFaceRef2ElemFaceRef<T>( (enum PXE_Shape) faceData->shape, (int) faceData->orientation, xref, xreflocal) );

   PXGradientsFace< T >((enum PXE_SolutionOrder) faceOrder, porderFace, xref, gphi);


   dx_dr = 0.0; dx_ds = 0.0;
   dy_dr = 0.0; dy_ds = 0.0;
   dz_dr = 0.0; dz_ds = 0.0;
   for(i=0; i<nbfQFace; i++){
       dx_dr += nodeCoord[i*DIM3D+0]*gphir[i];
       dy_dr += nodeCoord[i*DIM3D+1]*gphir[i];
       dz_dr += nodeCoord[i*DIM3D+2]*gphir[i];

       dx_ds += nodeCoord[i*DIM3D+0]*gphis[i];
       dy_ds += nodeCoord[i*DIM3D+1]*gphis[i];
       dz_ds += nodeCoord[i*DIM3D+2]*gphis[i];
   }
#endif
   return eNoError;
   */
}

// This function calculates the normal at the given point on a face.
// This function assumes it will only be called for planar faces.
ELVIS_DEVICE ElVisError GetFaceNormal(const ElVisFloat3& pointOnFace, GlobalFaceIdx globalFaceIdx, ElVisFloat3& result)
{
  PlanarFaceIdx planarFaceIdx = ConvertToPlanarFaceIdx(globalFaceIdx);
  ELVIS_PRINTF("MCG GetFaceNormal: Didn't know this was called yet!\n");
  ELVIS_PRINTF("MCG GetFaceNormal: normal=(%f, %f, %f)\n", PlanarFaceNormalBuffer[planarFaceIdx.Value].x, PlanarFaceNormalBuffer[planarFaceIdx.Value].y, PlanarFaceNormalBuffer[planarFaceIdx.Value].z);
  if( planarFaceIdx.Value > 0 )
  {
    result = MakeFloat3(PlanarFaceNormalBuffer[planarFaceIdx.Value]);
    return eNoError;
  }
  else
  {
    return eInvalidFaceId;
  }
}

//ELVIS_DEVICE ElVisError GetFaceNormal(const WorldPoint& pointOnFace, GlobalFaceIdx faceId, ElVisFloat3& result)
ELVIS_DEVICE ElVisError GetFaceNormal(const ElVisFloat2& referencePointOnFace, const ElVisFloat3& worldPointOnFace, GlobalFaceIdx faceId, ElVisFloat3& result)
{
  ELVIS_PRINTF("MCG GetFaceNormal: CURVED ELEMENTS Didn't know this was called yet!\n");
  result.x = 1;
  result.y = 0;
  result.z = 0;
	return eNoError;
    //PX_REAL xface[2] = {referencePointOnFace.x, referencePointOnFace.y};
    //PX_REAL nvec[3];

    /* compute normal w/orientation correction */
    /* this is guaranteed to point from left->right element */
    //PXError(PXOutwardNormal(faceOrder, porderFace, nbfQFace, &PXSimplexFaceDataBuffer[faceId], &PXSimplexFaceCoordinateBuffer[DIM3D*nbfQFace*faceId], xface, nvec));

    /* compute normal at the physical location corresponding to the input ref coords */
    /* EvaluateFace() & EvaluateFaceJacobian() do not perform the orientation
       correction.  Thus the normal *must* be evaluated without the correction too! */
    //PX_REAL nvec2[3];
    //PX_FaceData tempFace = {.orientation = 0, .side = 0, .shape = PXE_Shape_Triangle};
    //PXError(PXOutwardNormal(faceOrder, porderFace, nbfQFace, &tempFace, &PXSimplexFaceCoordinateBuffer[DIM3D*nbfQFace*faceId], xface, nvec2));

    /* Ensure that non-orientation-corrected normal points in the proper direction
       (=same direction as orientation-corrected normal) */
    /*
    PX_REAL temp = nvec[0]*nvec2[0]+nvec[1]*nvec2[1]+nvec[2]*nvec2[2];
    if(temp < 0){
      nvec2[0] *= -1;
      nvec2[1] *= -1;
      nvec2[2] *= -1;
    }

    result.x = nvec2[0];
    result.y = nvec2[1];
    result.z = nvec2[2];
*/
//    return eNoError;
}

ELVIS_DEVICE ElVisError SampleReferenceGradientOptiX(int elementId, int elementType, int fieldId, const ReferencePoint& refPoint, ElVisFloat3& gradient)
{
  ELVIS_PRINTF("MCG SampleReferenceGradientOptiX: CURVED ELEMENTS Didn't know this was called yet!\n");
  gradient.x = 1;
  gradient.y = 0;
  gradient.z = 0;

  return eNoError;
}

ELVIS_DEVICE ElVisError SampleGeometryMappingJacobianOptiX(int elementId, int elementType, const ReferencePoint& refPoint, ElVisFloat* J)
{
  ELVIS_PRINTF("MCG SampleGeometryMappingJacobianOptiX: CURVED ELEMENTS Didn't know this was called yet!\n");
  J[0] = 1; J[1] = 0; J[2] = 0;
  J[3] = 0; J[4] = 1; J[5] = 0;
  J[6] = 0; J[7] = 0; J[8] = 1;

  return eNoError;
}

#endif //end _PX_EXTENSION_OPTIX_INTERFACE_CU
