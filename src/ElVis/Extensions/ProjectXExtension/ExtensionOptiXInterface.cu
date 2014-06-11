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
rtDeclareVariable(int, nCurvedFace, , );
//rtDeclareVariable(int, nFaceVertex, , );
//rtDeclareVariable(int, nbfQFace, , );
//rtDeclareVariable(int, faceType, , );
//rtDeclareVariable(int, faceOrder, , );
//rtDeclareVariable(int, porderFace, , );

/* physical constants */
//rtDeclareVariable(PX_REAL, SpecificHeatRatio, , );
//rtDeclareVariable(PX_REAL, GasConstant, , );



#include <ElVis/Core/Interval.hpp>
#include "PXOptiXCudaCommon.cu"
#include "PXCutCell_ElVis.cu"

rtBuffer<int> egrp2GlobalElemIndex;
rtBuffer<PX_EgrpData> PXEgrpDataBuffer;

rtBuffer<PX_FaceTypeData> PXFaceDataBuffer;
rtBuffer<ElVisFloat> PXFaceCoordBuffer;

rtBuffer<ElVisFloat> PXElemCoordBuffer;
rtBuffer<ElVisFloat> PXSolutionBuffer;

//rtBuffer<ElVisFloat> PXFaceCoordinateBuffer;
//rtBuffer<uint> PXGlobalElemToEgrpElemBuffer;

//rtBuffer<PX_SolutionOrderData> PXAttachDataBuffer;
//rtBuffer<ElVisFloat> PXAttachmentBuffer;

//rtBuffer<ElVisFloat> PXShadowCoordinateBuffer;
//rtBuffer<uint> PXEgrpToShadowIndexBuffer;
//rtBuffer<ElVisFloat> PXPatchCoordinateBuffer;
//rtBuffer<PX_REAL> PXBackgroundCoordinateBuffer;
//rtBuffer<PX_REAL> PXKnownPointBuffer;
//rtBuffer<char> PXCutCellBuffer;
//rtBuffer<uint> PXGlobalElemToCutCellBuffer;

#define PX_USE_ISOSURF 1



ELVIS_DEVICE ElVisError EvaluateFace(GlobalFaceIdx faceId, const FaceReferencePoint& refPoint, WorldPoint& result)
{
  CurvedFaceIdx Idx = ConvertToCurvedFaceIdx(faceId);

  //ELVIS_PRINTF("MCG EvaluateFace: Didn't know this was called yet! Idx = %d\n",Idx.Value);

  ElVisFloat r = refPoint.x;
  ElVisFloat s = refPoint.y;
  PX_REAL phi[MAX_NBF_FACE];
  PX_REAL xref[2] = {r,s};
  //PX_REAL xreflocal[2];

  if( Idx.Value >= nCurvedFace ) {
    rtPrintf("############ EvaluateFace Idx.Value(%d) >= nCurvedFace(%d)", Idx.Value, nCurvedFace);
    return eFieldNotDefinedOnFace;
  }

  PX_FaceTypeData * faceData = &PXFaceDataBuffer[Idx.Value];

  PX_REAL *nodeCoord = &PXFaceCoordBuffer[faceData->idx];
  int nbfQFace = faceData->nbf;

  //PXErrorDebug( PXFaceRef2ElemFaceRef<PX_REAL>( faceData->shape, faceData->orientation, xref, xreflocal) );

  PXShapeFace<PX_REAL>(faceData->order, faceData->qorder, xref, phi);

  //ELVIS_PRINTF("MCG EvaluateFace: xref[0]=%f, xref[1]=%f, xreflocal[0]=%f, xreflocal[1]=%f\n",xref[0],xref[1],xreflocal[0],xreflocal[1]);
  //ELVIS_PRINTF("MCG EvaluateFace: xref[0]=%f, xref[1]=%f\n",xref[0],xref[1]);

  result.x = 0.0;
  result.y = 0.0;
  result.z = 0.0;
  for(int i=0; i<nbfQFace; i++)
  {
      result.x += nodeCoord[i*Dim+0]*phi[i];
      result.y += nodeCoord[i*Dim+1]*phi[i];
      result.z += Dim == 3 ? nodeCoord[i*Dim+2]*phi[i] : 0.0;
      //ELVIS_PRINTF("MCG EvaluateFace: phi[%d]=%f\n",i,phi[i]);
  }

  //ELVIS_PRINTF("MCG EvaluateFace: result.x=%f, result.y=%f, result.z=%f\n",result.x,result.y,result.z);

  return eNoError;

}


template<typename T>
ELVIS_DEVICE ElVisError EvaluateFaceJacobian(GlobalFaceIdx faceId, const FaceReferencePoint& p,
                                             T& dx_dr, T& dx_ds,
                                             T& dy_dr, T& dy_ds,
                                             T& dz_dr, T& dz_ds)
{
   CurvedFaceIdx Idx = ConvertToCurvedFaceIdx(faceId);

   if( Idx.Value >= nCurvedFace ) {
     rtPrintf("############ EvaluateFaceJacobian Idx.Value(%d) >= nCurvedFace(%d)", Idx.Value, nCurvedFace);
     return eFieldNotDefinedOnFace;
   }

   const T& r = p.x;
   const T& s = p.y;

   PX_FaceTypeData * faceData = &PXFaceDataBuffer[Idx.Value];

   PX_REAL *nodeCoord = &PXFaceCoordBuffer[faceData->idx];
   int nbfQFace = faceData->nbf;

   T gphi[2*MAX_NBF_FACE];
   T *gphir = gphi;
   T *gphis = gphir + nbfQFace;
   T xref[2] = {r,s};
   //T xreflocal[2];

   //rtPrintf("MCG EvaluateFaceJacobian: qorder=%d, xref[0]=%f, xref[1]=%f\n", faceData->qorder,xref[0],xref[1]);

   //PXErrorDebug( PXFaceRef2ElemFaceRef<T>( faceData->shape, faceData->orientation, xref, xreflocal) );

   PXGradientsFace< T >(faceData->order, faceData->qorder, xref, gphi);

   dx_dr = 0.0; dx_ds = 0.0;
   dy_dr = 0.0; dy_ds = 0.0;
   dz_dr = 0.0; dz_ds = 0.0;
   for(int i=0; i<nbfQFace; i++){
     dx_dr += nodeCoord[i*Dim+0]*gphir[i];
     dy_dr += nodeCoord[i*Dim+1]*gphir[i];
     dz_dr += Dim == 3 ? nodeCoord[i*Dim+2]*gphir[i] : 0.0;

     dx_ds += nodeCoord[i*Dim+0]*gphis[i];
     dy_ds += nodeCoord[i*Dim+1]*gphis[i];
     dz_ds += Dim == 3 ? nodeCoord[i*Dim+2]*gphis[i] : 0.0;
     //ELVIS_PRINTF("MCG EvaluateFaceJacobian: gphir[%d]=%f, gphis[%d]=%f\n",i,gphir[i],i,gphis[i]);
   }

   //ELVIS_PRINTF("MCG EvaluateFaceJacobian: dx_dr=%f, dy_dr=%f, dz_dr=%f\n",dx_dr,dy_dr,dz_dr);
   //ELVIS_PRINTF("MCG EvaluateFaceJacobian: dx_ds=%f, dy_ds=%f, dz_ds=%f\n",dx_ds,dy_ds,dz_ds);

   return eNoError;

}



//ELVIS_DEVICE ElVisFloat EvaluateFieldOptiX(unsigned int elementId, unsigned int elementType, int fieldId, const ElVisFloat3& worldPoint, ElVis::ReferencePointParameterType referenceType, ElVisFloat3& referencePoint)
//{
//    int egrp = PXGlobalElemToEgrpElemBuffer[2*elementId];
//    int elem = PXGlobalElemToEgrpElemBuffer[2*elementId+1];

//    int nbfQ = (int) PXEgrpDataBuffer[egrp].typeData.nbf;
//    int geomIndexStart = PXEgrpDataBuffer[egrp].egrpGeomCoeffStartIndex;

//    ElVisFloat* localCoord = &PXCoordinateBuffer[Dim*geomIndexStart + elem*Dim*nbfQ];
//    if(PXEgrpDataBuffer[egrp].cutCellFlag == (char) 1)
//    {
//        int shadowIndexStart = PXEgrpToShadowIndexBuffer[egrp];
//        localCoord = ((ElVisFloat*)&PXShadowCoordinateBuffer[0]) + shadowIndexStart + elem*DIM3D*SHADOW_NBF;
//    }
//    else if(referenceType != ElVis::eReferencePointIsValid)
//    {
//        // The reference point is not valid, so calculate it.
//        PX_REAL xglobal[3] = {worldPoint.x, worldPoint.y, worldPoint.z};
//        PX_REAL xref[3];
//        PXError(PXGlob2RefFromCoordinates2(&(PXEgrpDataBuffer[egrp].typeData), localCoord, xglobal, xref, PXE_False, PXE_False));
//        referencePoint.x = xref[0];
//        referencePoint.y = xref[1];
//        referencePoint.z = xref[2];
//    }

//    int solnIndexStart = PXEgrpDataBuffer[egrp].egrpSolnCoeffStartIndex;
//    int nbf = (int) PXEgrpDataBuffer[egrp].orderData.nbf;
//    ElVisFloat* localSolution = &PXSolutionBuffer[StateRank*solnIndexStart + elem*StateRank*nbf];
//    PX_SolutionOrderData *attachData = NULL;
//    if(fieldId < 0){
//      attachData = &PXAttachDataBuffer[egrp];
//      localSolution = &PXAttachmentBuffer[elementId*((int)attachData->nbf)];
//    }
    
//    return EvaluateField(&(PXEgrpDataBuffer[egrp]), attachData, localSolution, localCoord, StateRank, fieldId, worldPoint, referencePoint);
//}









/////////////////////////////////////////////
// Required ElVis Interface
/////////////////////////////////////////////
/// \brief Converts a point from world space to the given element's reference space.
/// \param elementId The element's id.  The id is a global id in the range [0, Model::GetNumberOfElements())
/// \param elementType The element's type.
/// \param wp The world point to be converted.
/// \param referenceType Describes the result parameter.
///                      if referenceType == eReferencePointIsInvalid then the result parameter is invalid on input,
///                                          and contains the conversion result on output.
///                      if referenceType == eReferencePointIsInitialGuess, then the result parameter is an initial guess on input,
///                                          and contains the conversion result on output.
/// \param result On input, it can be an initial guess.  On output, the actual reference point corresponding to wp.
/// \returns eNoError if a valid reference point is found
///          ePointOutsideElement if the world point does not fall inside this element
///          eInvalidElementId if the requested element id is invalid
///          eInvalidElementType if the requested element type is invalid
///          eConvergenceFailure if the reference point cannot be found due to convergence errors.
///
/// This method is responsible for transforming a given world-space coordinate to
/// the corresponding reference space coordinate for the specified element.  It
/// is assumed that an iterative method will be used to find the reference space coordinate,
/// although this is not required if analytic methods are available.  To aid convergence
/// of iterative methods, ElVis will provide an initial guess to this method when possible.
ELVIS_DEVICE ElVisError ConvertWorldToReferenceSpaceOptiX(int elementId, int elementType, const WorldPoint& worldPoint,
                                                          ElVis::ReferencePointParameterType referenceType, ReferencePoint& referencePoint)
{
    //ELVIS_PRINTF("MCG ConvertWorldToReferenceSpaceOptiX: Element Id %d, intersection point (%f, %f, %f)\n",
    //             elementId, worldPoint.x, worldPoint.y, worldPoint.z);

    if( referenceType != ElVis::eReferencePointIsValid )
    {
        int egrp = 0;
        while( elementId > egrp2GlobalElemIndex[egrp+1] ) { egrp++; }
        int elem = elementId - egrp2GlobalElemIndex[egrp];

        int nbfQ = PXEgrpDataBuffer[egrp].elemData.nbf;
        int geomIndexStart = PXEgrpDataBuffer[egrp].egrpGeomCoeffStartIndex;

        ELVIS_PRINTF("MCG ConvertWorldToReferenceSpaceOptiX: Dim=%d, geomIndexStart=%d, elem=%d, nbfQ=%d, idx=%d\n",
            Dim, geomIndexStart, elem, nbfQ, geomIndexStart + Dim*elem*nbfQ);

        ElVisFloat *localCoord = &PXElemCoordBuffer[geomIndexStart + Dim*elem*nbfQ];

        PX_REAL xglobal[3] = {worldPoint.x, worldPoint.y, worldPoint.z};
        PX_REAL xref[3] = {0.,0.,0.}; //Initial guess is overwritten anyways
        int result = PXGlob2RefFromCoordinates2(PXEgrpDataBuffer[egrp].elemData, localCoord, xglobal, xref, PXE_False, PXE_False);
        referencePoint.x = xref[0];
        referencePoint.y = xref[1];
        referencePoint.z = xref[2];
        ELVIS_PRINTF("MCG ConvertWorldToReferenceSpaceOptiX: Element Id %d, xref[0]=%f, xref[1]=%f, xref[2]=%f\n", elementId, xref[0], xref[1], xref[2]);

        //Bad point...
        if( result == PX_NOT_CONVERGED )
          return ePointOutsideElement;
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

    ELVIS_PRINTF("MCG SampleScalarFieldAtReferencePointOptiX: Element Id %d, reference point (%f, %f, %f)\n",
                 elementId, referencePoint.x, referencePoint.y, referencePoint.z);

    int egrp = 0;
    while( elementId > egrp2GlobalElemIndex[egrp+1] ) { egrp++; }
    int elem = elementId - egrp2GlobalElemIndex[egrp];


//    int nbfQ = (int) PXEgrpDataBuffer[egrp].elemData.nbf;
//    int geomIndexStart = PXEgrpDataBuffer[egrp].egrpGeomCoeffStartIndex;

    ElVisFloat* localCoord = NULL; //&PXCoordinateBuffer[Dim*geomIndexStart + elem*Dim*nbfQ];
//    if(PXEgrpDataBuffer[egrp].cutCellFlag == (char) 1)
//    {
//        int shadowIndexStart = PXEgrpToShadowIndexBuffer[egrp];
//        localCoord = ((ElVisFloat*)&PXShadowCoordinateBuffer[0]) + shadowIndexStart + elem*DIM3D*SHADOW_NBF;
//    }

    int solnIndexStart = PXEgrpDataBuffer[egrp].egrpSolnCoeffStartIndex;
    int nbf = PXEgrpDataBuffer[egrp].solData.nbf;

    //ELVIS_PRINTF("MCG SampleScalarFieldAtReferencePointOptiX: fieldId = %d, SOLN_MAX_NBF=%d, nbf=%d, idx=%d\n", fieldId, SOLN_MAX_NBF, nbf, solnIndexStart + elem*StateRank*nbf);

    ElVisFloat* localSolution = &PXSolutionBuffer[solnIndexStart + elem*StateRank*nbf];
    PX_SolutionOrderData *attachData = NULL;
//    if(fieldId < 0){
//      attachData = &PXAttachDataBuffer[egrp];
//      localSolution = &PXAttachmentBuffer[elementId*((int)attachData->nbf)];
//    }

    result = EvaluateField(PXEgrpDataBuffer[egrp], attachData, localSolution, localCoord, StateRank, fieldId, worldPoint, referencePoint);

    //result = localSolution[fieldId*nbf];
/*
    PX_REAL xref[3] = {referencePoint.x, referencePoint.y, referencePoint.z};
    PX_REAL phi[SOLN_MAX_NBF];
    for(int j = 0; j < SOLN_MAX_NBF; j++ ) phi[j] = 0;

    enum PXE_SolutionOrder order = PXEgrpDataBuffer[egrp].solData.order;
    int porder = PXEgrpDataBuffer[egrp].solData.porder;

    PXShapeElem_Solution<PX_REAL>(order, porder, xref, phi);

    result = 0;
    for(int j=0; j<nbf; j++)
        result += localSolution[j*StateRank + fieldId]*phi[j];
*/
    //ELVIS_PRINTF("MCG SampleScalarFieldAtReferencePointOptiX: result=%f\n", result);

    return eNoError;

}


ELVIS_DEVICE ElVisError IsValidFaceCoordinate(GlobalFaceIdx faceId, const FaceReferencePoint& p, bool& result)
{
    ElVisFloat r = p.x;
    ElVisFloat s = p.y;
    result = r >= MAKE_FLOAT(0.0) &&
            s >= MAKE_FLOAT(0.0) &&
            (r+s) <= MAKE_FLOAT(1.0);

    //ELVIS_PRINTF("MCG IsValidFaceCoordinate: r=%f, s=%f, result=%d!\n", r, s, result);

    return eNoError;
}

ELVIS_DEVICE ElVisError getStartingReferencePointForNewtonIteration(const CurvedFaceIdx& idx, ElVisFloat2& startingPoint)
{
  PX_REAL xref[2];
  PX_FaceTypeData * faceData = &PXFaceDataBuffer[idx.Value];

  PXElementCentroidReference(faceData->shape, xref);

  startingPoint.x = xref[0];
  startingPoint.y = xref[1];
  return eNoError;
}

/// \brief Adjust the step size in a Newton root-finding iteration to keep the current test point on the element's face.
/// \param curPoint The current test point.  curPoint.x and curPoint.y represent the current reference point on the
///                 face.  curPoint.z is the current t value along the ray.
/// \param idx The face being tested for intersection.
/// \param step The calculated adjustment that will be applied to curPoint for the next iteration.  If this adjustment
///             will cause the reference parameter to leave the element face, then step must be adjusted to prevent this.
/// Ray-face intersection is performed using Newton's method to numerically find the intersection.  The intersection
/// if found in term's of the face's reference coordinates.  In many systems, the mapping from reference to world space
/// is invalid outside the face's bounds, so we must keep the iteration from leaving the face).
ELVIS_DEVICE ElVisError adjustNewtonStepToKeepReferencePointOnFace(const ElVisFloat3& curPoint, const CurvedFaceIdx& idx, ElVisFloat3& step)
{
  if( curPoint.x - step.x < 0 )
    step.x = curPoint.x;

  if( curPoint.y - step.y < 0 )
    step.y = curPoint.y;

  if( curPoint.x - step.x + curPoint.y - step.y > 1 )
  {
    if( curPoint.x - step.x > 1)
      step.x = -(1 - curPoint.x);

    if( curPoint.x - step.x + curPoint.y - step.y > 1 )
      step.y = -(1 - curPoint.x + step.x - curPoint.y);
  }

  //ELVIS_PRINTF("MCG New Point: r=%f, s=%f\n", curPoint.x - step.x, curPoint.y - step.y);

  return eNoError;
}

// This function calculates the normal at the given point on a face.
// This function assumes it will only be called for planar faces.
ELVIS_DEVICE ElVisError GetFaceNormal(const ElVisFloat3& pointOnFace, GlobalFaceIdx globalFaceIdx, ElVisFloat3& result)
{
  PlanarFaceIdx planarFaceIdx = ConvertToPlanarFaceIdx(globalFaceIdx);
  //ELVIS_PRINTF("MCG GetFaceNormal: Didn't know this was called yet!\n");
  //ELVIS_PRINTF("MCG GetFaceNormal: normal=(%f, %f, %f)\n", PlanarFaceNormalBuffer[planarFaceIdx.Value].x, PlanarFaceNormalBuffer[planarFaceIdx.Value].y, PlanarFaceNormalBuffer[planarFaceIdx.Value].z);
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
    //PXError(PXOutwardNormal(faceOrder, porderFace, nbfQFace, &PXFaceDataBuffer[faceId], &PXFaceCoordinateBuffer[DIM3D*nbfQFace*faceId], xface, nvec));

    /* compute normal at the physical location corresponding to the input ref coords */
    /* EvaluateFace() & EvaluateFaceJacobian() do not perform the orientation
       correction.  Thus the normal *must* be evaluated without the correction too! */
    //PX_REAL nvec2[3];
    //PX_FaceData tempFace = {.orientation = 0, .side = 0, .shape = PXE_Shape_Triangle};
    //PXError(PXOutwardNormal(faceOrder, porderFace, nbfQFace, &tempFace, &PXFaceCoordinateBuffer[DIM3D*nbfQFace*faceId], xface, nvec2));

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
