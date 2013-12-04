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

#include <Fundamentals/PX.h>
#include <ElVis/Core/OptixVariables.cu>

rtDeclareVariable(int, Dim, , );
rtDeclareVariable(int, StateRank, , );
rtDeclareVariable(int, nFaceVertex, , );
rtDeclareVariable(int, nbfQFace, , );
rtDeclareVariable(int, faceType, , );
rtDeclareVariable(int, faceOrder, , );
rtDeclareVariable(int, porderFace, , );

/* physical constants */
rtDeclareVariable(PX_REAL, SpecificHeatRatio, , );
rtDeclareVariable(PX_REAL, GasConstant, , );



#include "PXOptiXCudaCommon.cu"
#include "PXCutCell_ElVis.cu"
#include <ElVis/Core/Interval.hpp>

rtBuffer<ElVisFloat> PXSimplexBoundingBoxBuffer;
rtBuffer<ElVisFloat> PXSimplexCoordinateBuffer;
rtBuffer<ElVisFloat> PXSimplexFaceCoordinateBuffer;
rtBuffer<PX_FaceData> PXSimplexFaceDataBuffer;
rtBuffer<ElVisFloat> PXSimplexSolutionBuffer;
rtBuffer<PX_EgrpData> PXSimplexEgrpDataBuffer;
rtBuffer<uint> PXSimplexGlobalElemToEgrpElemBuffer;

rtBuffer<PX_SolutionOrderData> PXSimplexAttachDataBuffer;
rtBuffer<ElVisFloat> PXSimplexAttachmentBuffer;

rtBuffer<ElVisFloat> PXSimplexShadowCoordinateBuffer;
rtBuffer<uint> PXSimplexEgrpToShadowIndexBuffer;
rtBuffer<ElVisFloat> PXSimplexPatchCoordinateBuffer;
rtBuffer<PX_REAL> PXSimplexBackgroundCoordinateBuffer;
rtBuffer<PX_REAL> PXSimplexKnownPointBuffer;
rtBuffer<char> PXSimplexCutCellBuffer;
rtBuffer<uint> PXSimplexGlobalElemToCutCellBuffer;

#define PX_USE_ISOSURF 1

/*
ELVIS_DEVICE TensorPoint ConvertToTensorSpace(unsigned int elementId, unsigned int elementType, const WorldPoint& wp)
{
  ELVIS_PRINTF("ConvertToTensorSpace NOT IMPLEMENTED!%d\n",1);
  return MakeFloat3(MAKE_FLOAT(0.0),MAKE_FLOAT(0.0),MAKE_FLOAT(0.0));
}

ELVIS_DEVICE ElVisFloat EvaluateFieldAtTensorPoint(unsigned int elementId, unsigned int elementType, const TensorPoint& tp)
{
  ELVIS_PRINTF("EvaluateFieldAtTensorPoint NOT IMPLEMENTED!%d\n ",1);
  return MAKE_FLOAT(0.0);
}
*/

/*
ELVIS_DEVICE
ElVis::Interval<ElVisFloat> EstimateRangeFromTensorPoints(unsigned int elementId, unsigned int elementType,
                                   const ElVisFloat3& p0, const ElVisFloat3& p1)
{
  ELVIS_PRINTF("EstimateRangeFromTensorPoints NOT IMPLEMENTED!%d\n ",1);
  return ElVis::Interval<ElVisFloat>();
}

ELVIS_DEVICE 
void EstimateRange(unsigned int elementId, unsigned int elementType, 
                                          const ElVisFloat3& p0, const ElVisFloat3& p1,
                                          ElVis::Interval<ElVisFloat>& result)
{
  ELVIS_PRINTF("EstimateRange NOT IMPLEMENTED! ");
}
*/




ELVIS_DEVICE ElVisError EvaluateFace(int faceId, const FaceReferencePoint& refPoint,
                               WorldPoint& result)
{
	return eConvergenceFailure;
#if PX_USE_ISOSURF
    ElVisFloat r = refPoint.x;
    ElVisFloat s = refPoint.y;
    PX_REAL phi[MAX_NBF_FACE];
    PX_REAL xref[2] = {r,s};
    PX_REAL xreflocal[2];
    PX_REAL *nodeCoord = &PXSimplexFaceCoordinateBuffer[DIM3D*nbfQFace*faceId];
    int i, d;

    // PX_FaceData * faceData = &PXSimplexFaceDataBuffer[faceId];
    // PXErrorDebug( PXFaceRef2ElemFaceRef<PX_REAL>( (enum PXE_Shape) faceData->shape, (int) faceData->orientation, xref, xreflocal) );

    PXShapeFace<PX_REAL>((enum PXE_SolutionOrder) faceOrder, porderFace, xref, phi);

    result.x = 0.0;
    result.y = 0.0;
    result.z = 0.0;
    for(i=0; i<nbfQFace; i++)
    {
        result.x += nodeCoord[i*DIM3D+0]*phi[i];
        result.y += nodeCoord[i*DIM3D+1]*phi[i];
        result.z += nodeCoord[i*DIM3D+2]*phi[i];
    }
#endif
}



RT_PROGRAM void PXSimplexContainsOriginByCheckingPoint(int PXSimplexId)
{
	return;

    ELVIS_PRINTF("PXSimplexContainsOriginByCheckingPoint: Checking element %d\n", PXSimplexId);
    PX_REAL xglobal[3] = {ray.origin.x, ray.origin.y, ray.origin.z};
    int boundingBoxFlag = 0;

    boundingBoxFlag = (xglobal[0] >= PXSimplexBoundingBoxBuffer[BBOX_SIZE*PXSimplexId + 0]) &&
            (xglobal[1] >= PXSimplexBoundingBoxBuffer[BBOX_SIZE*PXSimplexId + 1]) &&
            (xglobal[2] >= PXSimplexBoundingBoxBuffer[BBOX_SIZE*PXSimplexId + 2]) &&
            (xglobal[0] <= PXSimplexBoundingBoxBuffer[BBOX_SIZE*PXSimplexId + 3]) &&
            (xglobal[1] <= PXSimplexBoundingBoxBuffer[BBOX_SIZE*PXSimplexId + 4]) &&
            (xglobal[2] <= PXSimplexBoundingBoxBuffer[BBOX_SIZE*PXSimplexId + 5]);

    if(boundingBoxFlag == 1){
        PX_REAL xref[3];
        //ElVisFloat3 origin = MakeFloat3(ray.origin);
        int egrp = PXSimplexGlobalElemToEgrpElemBuffer[2*PXSimplexId];
        int elem = PXSimplexGlobalElemToEgrpElemBuffer[2*PXSimplexId+1];
        enum PXE_ElementType type = (enum PXE_ElementType) PXSimplexEgrpDataBuffer[egrp].typeData.type;
        char intersectionFoundFlag = 0;
        //enum PXE_SolutionOrder orderQ;

        if(PXSimplexEgrpDataBuffer[egrp].cutCellFlag == (char) 1){
            enum PXE_3D_ZeroDTypeOnBack zeroDType = (enum PXE_3D_ZeroDTypeOnBack) -1;
            //need to check background elements of *ALL* patch groups associated w/this element

            int curPatchGroup;
            unsigned int localCutCellIndex = PXSimplexGlobalElemToCutCellBuffer[PXSimplexId];
            char tempIntersectionFlag = 0;
            //rtPrintf("localCutCellIndex[%d] = %d\n",PXSimplexId,localCutCellIndex);
            PX_CutCellElVis* cutCell = (PX_CutCellElVis*)((char*)(&PXSimplexCutCellBuffer[0]) + localCutCellIndex);

            PX_REAL *localCoord;
            int *patchList;
            PX_PatchGroup* patchGroup = GetFirstPatchGroup(cutCell);
            intersectionFoundFlag = 0;
            for(curPatchGroup=0; curPatchGroup<cutCell->nPatchGroup; curPatchGroup++){
                //localCoord points to the coordinates of background element
                //background element assumed to be PXE_UniformTetQ1

                localCoord = &PXSimplexBackgroundCoordinateBuffer[0] + patchGroup->threeDId*DIM3D*BACK_NBF;

                //PXError( LinearSimplexGlob2Ref(3, localCoord, xglobal, xref) );

                PXError(PXDetermineInsideTet<PX_REAL>(localCoord, xglobal, zeroDType));

#if 1 //this case means in/out test only checks background element	
                if(zeroDType == PXE_3D_0DBackNull){
                    intersectionFoundFlag = 0;
                }else{
                    intersectionFoundFlag = 1;
                    break;
                }

#else //this case means in/out test will do a full computation
                //this case currently DOES NOT COMPILE

                if(zeroDType == PXE_3D_0DBackNull){
                    tempIntersectionFlag = 0;
                }else{
                    tempIntersectionFlag = 1;
                }

                if(tempIntersectionFlag == 1){
                    patchList = GetPatchList(patchGroup);
                    //PX_REAL lineNode[2][DIM3D];
                    PX_REAL const* lineNode[2];
                    int nIntersect;
                    int ierr;

                    lineNode[0] = &PXSimplexKnownPointBuffer[patchGroup->threeDId*DIM3D];
                    lineNode[1] = xglobal;
                    ierr = PXCountSegmentPatchesIntersect(&PXSimplexPatchCoordinateBuffer[0], patchList, patchGroup->nPatch, lineNode, nIntersect);

                    if(ierr != PX_NO_ERROR){
                        ELVIS_PRINTF("CountSegment FAILED\n");
                        intersectionFoundFlag = 0;
                        break;
                    }


                    if(patchGroup->knownPointFlag == (enum PXE_KnownPointType) PXE_KnownPoint_Outside){
                        if(nIntersect % 2 != 0){
                            intersectionFoundFlag == 1;
                            break;
                        }
                    }else{ // point type is PXE_KnownPoint_Inside
                        if(nIntersect % 2 == 0){
                            intersectionFoundFlag == 1;
                            break;
                        }
                    }
                }

#endif

                patchGroup = GetNextPatchGroup(patchGroup);
            }//loop over patch groups

        }else{
            int nbfQ = (int) PXSimplexEgrpDataBuffer[egrp].typeData.nbf;
            int geomIndexStart = PXSimplexEgrpDataBuffer[egrp].egrpGeomCoeffStartIndex;
            ElVisFloat* localCoord = &PXSimplexCoordinateBuffer[Dim*geomIndexStart + elem*Dim*nbfQ];
            //LinearSimplexGlob2Ref(Dim, localCoord, xglobal, xref);
            PXError(PXGlob2RefFromCoordinates2(&(PXSimplexEgrpDataBuffer[egrp].typeData), localCoord, xglobal, xref, PXE_False, PXE_False));

            PX_REAL sum = xref[0] + xref[1] + xref[2];
            intersectionFoundFlag = xref[0] >= -0.001 && xref[0] <= 1.001 &&
                    xref[1] >= -0.001 && xref[1] <= 1.001 &&
                    xref[2] >= -0.001 && xref[2] <= 1.001 &&
                    sum     >= -0.001 && sum     <= 1.001;
        }

        /* Test if xref is inside Q1 Tet */

        if( intersectionFoundFlag == (char) 1){
            /* If so, report intersection */
            if( rtPotentialIntersection(.1) ){
                /* potential intersect at t = 0.1 along the ray.
    0.1 is arbitrary b/c only ONE element should
    contain a point.  If unlucky and ray.origin
    is a corner/face node, then the first queried
    element is the winner. */

                intersectionPointPayload.IntersectionPoint.x = ray.origin.x;
                intersectionPointPayload.IntersectionPoint.y = ray.origin.y;
                intersectionPointPayload.IntersectionPoint.z = ray.origin.z;

                intersectionPointPayload.elementId = PXSimplexId;
                intersectionPointPayload.elementType = (int) type; // TEMPORARY

                intersectionPointPayload.ReferenceIntersectionPoint.x = xref[0];
                intersectionPointPayload.ReferenceIntersectionPoint.y = xref[1];
                intersectionPointPayload.ReferenceIntersectionPoint.z = xref[2];

                intersectionPointPayload.ReferencePointType = ElVis::eReferencePointIsValid;
                /* intersection.  call closest hit using material 0 */
                rtReportIntersection(0);
            }


        }
    }


}


RT_PROGRAM void PXSimplex_bounding (int id, float result[6])
{
  //ELVIS_PRINTF("I'm ray %d in PXSimplex_bounding\n", id);
  // int nbfQ;
  // int egrp = PXSimplexGlobalElemToEgrpElemBuffer[id].x;
  // int elem = PXSimplexGlobalElemToEgrpElemBuffer[id].y;
  // enum PXE_ElementType type = PXSimplexEgrpDataBuffer[egrp].type;
  // int geomIndexStart = PXSimplexEgrpDataBuffer[egrp].egrpGeomCoeffStartIndex;

  // PXType2nbf(type, &nbfQ);
  // ElVisFloat* localCoord = &PXSimplexCoordinateBuffer[Dim*geomIndexStart + elem*Dim*nbfQ];

  // if(type != PXE_UniformTetQ1){
  //   rtPrintf("ERROR, %d is an invalid element type!\n",type);
  // }
    
  optix::Aabb* aabb = (optix::Aabb*)result;

  /* Following code is ONLY VALID for UniformTetQ1, nbfQ = 4 */

  // aabb->m_min.x = fminf(localCoord[0*Dim + 0], fminf(localCoord[1*Dim + 0], fminf(localCoord[2*Dim + 0], localCoord[3*Dim + 0])));
  // aabb->m_min.y = fminf(localCoord[0*Dim + 1], fminf(localCoord[1*Dim + 1], fminf(localCoord[2*Dim + 1], localCoord[3*Dim + 1])));
  // aabb->m_min.z = fminf(localCoord[0*Dim + 2], fminf(localCoord[1*Dim + 2], fminf(localCoord[2*Dim + 2], localCoord[3*Dim + 2])));

  // aabb->m_max.x = fmaxf(localCoord[0*Dim + 0], fmaxf(localCoord[1*Dim + 0], fmaxf(localCoord[2*Dim + 0], localCoord[3*Dim + 0])));
  // aabb->m_max.y = fmaxf(localCoord[0*Dim + 1], fmaxf(localCoord[1*Dim + 1], fmaxf(localCoord[2*Dim + 1], localCoord[3*Dim + 1])));
  // aabb->m_max.z = fmaxf(localCoord[0*Dim + 2], fmaxf(localCoord[1*Dim + 2], fmaxf(localCoord[2*Dim + 2], localCoord[3*Dim + 2])));

  // PXSimplexBoundingBoxBuffer[BBOX_SIZE*id + 0] = aabb->m_min.x;  
  // PXSimplexBoundingBoxBuffer[BBOX_SIZE*id + 1] = aabb->m_min.y;  
  // PXSimplexBoundingBoxBuffer[BBOX_SIZE*id + 2] = aabb->m_min.z;  

  // PXSimplexBoundingBoxBuffer[BBOX_SIZE*id + 3] = aabb->m_max.x;  
  // PXSimplexBoundingBoxBuffer[BBOX_SIZE*id + 4] = aabb->m_max.y;  
  // PXSimplexBoundingBoxBuffer[BBOX_SIZE*id + 5] = aabb->m_max.z;  

  aabb->m_min.x = PXSimplexBoundingBoxBuffer[BBOX_SIZE*id + 0];  
  aabb->m_min.y = PXSimplexBoundingBoxBuffer[BBOX_SIZE*id + 1];  
  aabb->m_min.z = PXSimplexBoundingBoxBuffer[BBOX_SIZE*id + 2];  

  aabb->m_max.x = PXSimplexBoundingBoxBuffer[BBOX_SIZE*id + 3];
  aabb->m_max.y = PXSimplexBoundingBoxBuffer[BBOX_SIZE*id + 4];
  aabb->m_max.z = PXSimplexBoundingBoxBuffer[BBOX_SIZE*id + 5];
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
    ELVIS_PRINTF("ConvertWorldToReferenceSpaceOptiX: Element Id %d, intersection point (%f, %f, %f)\n",
                 elementId, worldPoint.x, worldPoint.y, worldPoint.z);
    return eConvergenceFailure;

    if( referenceType != ElVis::eReferencePointIsValid )
    {
        int egrp = PXSimplexGlobalElemToEgrpElemBuffer[2*elementId];
        int elem = PXSimplexGlobalElemToEgrpElemBuffer[2*elementId+1];

        int nbfQ = (int) PXSimplexEgrpDataBuffer[egrp].typeData.nbf;
        int geomIndexStart = PXSimplexEgrpDataBuffer[egrp].egrpGeomCoeffStartIndex;

        ElVisFloat* localCoord = &PXSimplexCoordinateBuffer[Dim*geomIndexStart + elem*Dim*nbfQ];

        PX_REAL xglobal[3] = {worldPoint.x, worldPoint.y, worldPoint.z};
        PX_REAL xref[3];
        PXError(PXGlob2RefFromCoordinates2(&(PXSimplexEgrpDataBuffer[egrp].typeData), localCoord, xglobal, xref, PXE_False, PXE_False));
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
    ELVIS_PRINTF("SampleScalarFieldAtReferencePointOptiX: Element Id %d, intersection point (%f, %f, %f)\n",
                 elementId, worldPoint.x, worldPoint.y, worldPoint.z);
    int egrp = PXSimplexGlobalElemToEgrpElemBuffer[2*elementId];
    int elem = PXSimplexGlobalElemToEgrpElemBuffer[2*elementId+1];

    int nbfQ = (int) PXSimplexEgrpDataBuffer[egrp].typeData.nbf;
    int geomIndexStart = PXSimplexEgrpDataBuffer[egrp].egrpGeomCoeffStartIndex;

    ElVisFloat* localCoord = &PXSimplexCoordinateBuffer[Dim*geomIndexStart + elem*Dim*nbfQ];
    if(PXSimplexEgrpDataBuffer[egrp].cutCellFlag == (char) 1)
    {
        int shadowIndexStart = PXSimplexEgrpToShadowIndexBuffer[egrp];
        localCoord = ((ElVisFloat*)&PXSimplexShadowCoordinateBuffer[0]) + shadowIndexStart + elem*DIM3D*SHADOW_NBF;
    }

    int solnIndexStart = PXSimplexEgrpDataBuffer[egrp].egrpSolnCoeffStartIndex;
    int nbf = (int) PXSimplexEgrpDataBuffer[egrp].orderData.nbf;
    ElVisFloat* localSolution = &PXSimplexSolutionBuffer[StateRank*solnIndexStart + elem*StateRank*nbf];
    PX_SolutionOrderData *attachData = NULL;
    if(fieldId < 0){
      attachData = &PXSimplexAttachDataBuffer[egrp];
      localSolution = &PXSimplexAttachmentBuffer[elementId*((int)attachData->nbf)];
    }

    result = EvaluateField(&(PXSimplexEgrpDataBuffer[egrp]), attachData, localSolution, localCoord, StateRank, fieldId, worldPoint, referencePoint);
    return eNoError;
}

ELVIS_DEVICE ElVisError GetNumberOfVerticesForFace(int faceId, int& result)
{
    result = nFaceVertex;
    return eNoError;
}


ELVIS_DEVICE ElVisError GetFaceVertex(int faceId, int vertexId, ElVisFloat4& result)
{
    result.x = PXSimplexFaceCoordinateBuffer[DIM3D*nbfQFace*faceId+vertexId*DIM3D];
    result.y = PXSimplexFaceCoordinateBuffer[DIM3D*nbfQFace*faceId+vertexId*DIM3D+1];
    result.z = PXSimplexFaceCoordinateBuffer[DIM3D*nbfQFace*faceId+vertexId*DIM3D+2];
    return eNoError;
}

ELVIS_DEVICE ElVisError IsValidFaceCoordinate(int faceId, const FaceReferencePoint& p, bool& result)
{
    ElVisFloat r = p.x;
    ElVisFloat s = p.y;
    result = r >= MAKE_FLOAT(0.0) &&
            s >= MAKE_FLOAT(0.0) &&
            (r+s) <= MAKE_FLOAT(1.0);

    return eNoError;
}

template<typename T>
ELVIS_DEVICE ElVisError EvaluateFaceJacobian(int faceId, const FaceReferencePoint& p,
                                             T& dx_dr, T& dx_ds,
                                             T& dy_dr, T& dy_ds,
                                             T& dz_dr, T& dz_ds)
{

	   dx_dr = 0.0; dx_ds = 0.0;
	   dy_dr = 0.0; dy_ds = 0.0;
	   dz_dr = 0.0; dz_ds = 0.0;

   return eNoError;

   ELVIS_PRINTF("EvaluateFaceJacobian");
   const T& r = p.x;
   const T& s = p.y;

#if PX_USE_ISOSURF
   T gphi[2*MAX_NBF_FACE];
   T *gphir = gphi;
   T *gphis = gphir + MAX_NBF_FACE;
   T xref[2] = {r,s};
   T xreflocal[2];
   PX_REAL *nodeCoord = &PXSimplexFaceCoordinateBuffer[DIM3D*nbfQFace*faceId];
   int i, d;

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
}

// This function calculates the normal at the given point on a face.
// This function assumes it will only be called for planar faces.
ELVIS_DEVICE ElVisError GetFaceNormal(const ElVisFloat3& pointOnFace, int faceId, ElVisFloat3& result)
{
	//return eConvergenceFailure;
    PX_REAL xface[2] = {MAKE_FLOAT(0.0), MAKE_FLOAT(0.0)};
    PX_REAL nvec[3];

    PXError(PXOutwardNormal((enum PXE_SolutionOrder) faceOrder, porderFace, nbfQFace, &PXSimplexFaceDataBuffer[faceId], &PXSimplexFaceCoordinateBuffer[DIM3D*nbfQFace*faceId], xface, nvec));

    result.x = nvec[0];
    result.y = nvec[1];
    result.z = nvec[2];
    return eNoError;
}

ELVIS_DEVICE ElVisError GetFaceNormal(const ElVisFloat2& referencePointOnFace, const ElVisFloat3& worldPointOnFace, int faceId, ElVisFloat3& result)
{
	return eConvergenceFailure;
#if PX_USE_ISOSURF
    PX_REAL xface[2] = {referencePointOnFace.x, referencePointOnFace.y};
    PX_REAL nvec[3];

    /* compute normal w/orientation correction */
    /* this is guaranteed to point from left->right element */
    PXError(PXOutwardNormal((enum PXE_SolutionOrder) faceOrder, porderFace, nbfQFace, &PXSimplexFaceDataBuffer[faceId], &PXSimplexFaceCoordinateBuffer[DIM3D*nbfQFace*faceId], xface, nvec));

    /* compute normal at the physical location corresponding to the input ref coords */
    /* EvaluateFace() & EvaluateFaceJacobian() do not perform the orientation
       correction.  Thus the normal *must* be evaluated without the correction too! */
    PX_REAL nvec2[3];
    PX_FaceData tempFace = {.orientation = 0, .side = 0, .shape = (enum PXE_Shape) PXE_Shape_Triangle};
    PXError(PXOutwardNormal((enum PXE_SolutionOrder) faceOrder, porderFace, nbfQFace, &tempFace, &PXSimplexFaceCoordinateBuffer[DIM3D*nbfQFace*faceId], xface, nvec2));

    /* Ensure that non-orientation-corrected normal points in the proper direction
       (=same direction as orientation-corrected normal) */
    PX_REAL temp = nvec[0]*nvec2[0]+nvec[1]*nvec2[1]+nvec[2]*nvec2[2];
    if(temp < 0){
      nvec2[0] *= -1;
      nvec2[1] *= -1;
      nvec2[2] *= -1;
    }

    result.x = nvec2[0];
    result.y = nvec2[1];
    result.z = nvec2[2];

#endif
    return eNoError;
}

#endif //end _PX_EXTENSION_OPTIX_INTERFACE_CU
