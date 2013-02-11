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

#ifndef _PX_EXTENSION_INTERFACE_CU
#define _PX_EXTENSION_INTERFACE_CU

#include "PXSimplex.cu"
#include <ElVis/Core/Interval.hpp>

ELVIS_DEVICE TensorPoint ConvertToTensorSpace(unsigned int elementId, unsigned int elementType, const WorldPoint& wp)
{
  rtPrintf("ConvertToTensorSpace NOT IMPLEMENTED! ");
  return MakeFloat3(MAKE_FLOAT(0.0),MAKE_FLOAT(0.0),MAKE_FLOAT(0.0));
}

ELVIS_DEVICE ElVisFloat EvaluateFieldAtTensorPoint(unsigned int elementId, unsigned int elementType, const TensorPoint& tp)
{
  rtPrintf("EvaluateFieldAtTensorPoint NOT IMPLEMENTED! ");
  return MAKE_FLOAT(0.0);
}

ELVIS_DEVICE
ElVis::Interval<ElVisFloat> EstimateRangeFromTensorPoints(unsigned int elementId, unsigned int elementType,
                                   const ElVisFloat3& p0, const ElVisFloat3& p1)
{
  rtPrintf("EstimateRangeFromTensorPoints NOT IMPLEMENTED! ");
  return ElVis::Interval<ElVisFloat>();
}

ELVIS_DEVICE 
void EstimateRange(unsigned int elementId, unsigned int elementType, 
                                          const ElVisFloat3& p0, const ElVisFloat3& p1,
                                          ElVis::Interval<ElVisFloat>& result)
{
  rtPrintf("EstimateRange NOT IMPLEMENTED! ");
}


ELVIS_DEVICE ElVisFloat EvaluateField(unsigned int elementId, unsigned int elementType, const ElVisFloat3& worldPoint)
{
 
  PX_REAL xref[3];// = {0.25, 0.25, 0.25};
  PX_REAL phi[56];
  int egrp = PXSimplexGlobalElemToEgrpElemBuffer[elementId].x;
  int elem = PXSimplexGlobalElemToEgrpElemBuffer[elementId].y;
  enum PXE_SolutionOrder order = PXSimplexEgrpDataBuffer[egrp].order;
  //enum PXE_ElementType type = PXSimplexEgrpDataBuffer[egrp].type;
  //enum PXE_SolutionOrder orderQ;

  int solnIndexStart = PXSimplexEgrpDataBuffer[egrp].egrpSolnCoeffStartIndex;
  //int nbfQ;
  int nbf;
  int i,j;

  PXOrder2nbf(order, &nbf);
  //PXType2nbf(type, &nbfQ);
  //PXType2Interpolation(type, &orderQ);

  //ElVisFloat* localCoord = &PXSimplexCoordinateBuffer[Dim*geomIndexStart + elem*Dim*nbfQ];

  /* TEMPORARY */
  PX_REAL xglobal[3] = {worldPoint.x, worldPoint.y, worldPoint.z};
  enum PXE_ElementType type = PXSimplexEgrpDataBuffer[egrp].type;
  int nbfQ;
  PXType2nbf(type, &nbfQ);
  int geomIndexStart = PXSimplexEgrpDataBuffer[egrp].egrpGeomCoeffStartIndex;
  ElVisFloat* localCoord = &PXSimplexCoordinateBuffer[Dim*geomIndexStart + elem*Dim*nbfQ];
  LinearSimplexGlob2Ref(Dim, localCoord, xglobal, xref);

  /* WANT TO DO IT THIS WAY INSTEAD */
  // xref[0] = intersectedPointRef.x;
  // xref[1] = intersectedPointRef.y;
  // xref[2] = intersectedPointRef.z;
  PXShape2(order, xref, phi);

  ElVisFloat* localSolution = &PXSimplexSolutionBuffer[StateRank*solnIndexStart + elem*StateRank*nbf];

  /* centroid location is (x,y,z) in global coords
     set result = x+y+z */
  ElVisFloat result = MAKE_FLOAT(0.0);

  /* plot density */
  int index = 1;
  for(j=0; j<nbf; j++){
    result += localSolution[j*StateRank + index]*phi[j];
  }
  

  // for(i=0; i<nbfQ; i++){
  //   printf("phi[%d]=%.8E\n",i,phi[i]);
  // }

  // for(i=0; i<Dim; i++){
  //   for(j=0; j<nbfQ; j++){
  //     result += localCoord[j*Dim + i]*phi[j];
  //   }
  // }

  // for(j=0; j<nbfQ; j++){
  //   for(i=0; i<Dim; i++){
  //     printf("localCoord[%d,%d]=%.8E\n",j,i,localCoord[j*Dim+i]);
  //   }
  // }

  // rtPrintf("egrp=%d,elem=%d,orderQ = %d, nbfQ = %d, result=%.8E\n",egrp,elem,orderQ, nbfQ, result);
  //rtPrintf("egrp=%d,elem=%d,order = %d, nbf = %d, result=%.8E\n",egrp,elem,order, nbf, result);
    
  return result;
}

#endif //end _PX_EXTENSION_INTERFACE_CU
