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

#ifndef ELVIS_JACOBI_EXTENSION_CUDA_INTERFACE_CU
#define ELVIS_JACOBI_EXTENSION_CUDA_INTERFACE_CU

#include <ElVis/Core/Float.cu>
#include <ElVis/Extensions/JacobiExtension/CudaHexahedron.cu>
#include <ElVis/Extensions/JacobiExtension/CudaPrism.cu>
#include <ElVis/Core/IntervalPoint.cu>

__device__ ElVisFloat4* FaceVertexBuffer;
__device__ ElVisFloat4* FaceNormalBuffer;



ELVIS_DEVICE void CalculateTransposedInvertedMappingJacobianCuda(unsigned int elementId, unsigned int elementType, int fieldId, const TensorPoint& tp, ElVisFloat* J)
{
    ElVisFloat JInv[9];

    if( elementType == 0 )
    {
        calculateInverseJacobian(HexVertexBuffer, elementId, tp, JInv);
    }
    else if( elementType == 1 )
    {
        CalculateInversePrismJacobian(PrismVertexBuffer, elementId, tp, JInv);
    }

    J[0] = JInv[0];
    J[1] = JInv[3];
    J[2] = JInv[6];

    J[3] = JInv[1];
    J[4] = JInv[4];
    J[5] = JInv[7];

    J[6] = JInv[2];
    J[7] = JInv[5];
    J[8] = JInv[8];
}

ELVIS_DEVICE ElVisFloat3 CalculateTensorGradient(unsigned int elementId, unsigned int elementType, int fieldId, const TensorPoint& p)
{
    ElVisFloat3 result = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
    if( elementType == 0 )
    {
        uint3 degree = HexDegrees[elementId];

        uint coefficientIndex = HexCoefficientIndices[elementId];
        ElVisFloat* coeffs = &(HexCoefficients[coefficientIndex]);

        result.x = EvaluateHexGradientDir1AtTensorPoint(degree, p.x, p.y, p.z, coeffs);
        result.y = EvaluateHexGradientDir2AtTensorPoint(degree, p.x, p.y, p.z, coeffs);
        result.z = EvaluateHexGradientDir3AtTensorPoint(degree, p.x, p.y, p.z, coeffs);
    }
    else if( elementType == 1 )
    {
        uint3 degree = PrismDegrees[elementId];

        uint coefficientIndex = PrismCoefficientIndices[elementId];
        ElVisFloat* coeffs = &(PrismCoefficients[coefficientIndex]);

        result.x = EvaluatePrismGradientDir1AtTensorPoint<ElVisFloat>(degree, p.x, p.y, p.z, coeffs);
        result.y = EvaluatePrismGradientDir2AtTensorPoint<ElVisFloat>(degree, p.x, p.y, p.z, coeffs);
        result.z = EvaluatePrismGradientDir3AtTensorPoint<ElVisFloat>(degree, p.x, p.y, p.z, coeffs);
    }
    return result;
}


////////////////////////
// Interface
///////////////////////
ELVIS_DEVICE ElVisError ConvertWorldToReferenceSpaceCuda(int elementId, int elementType, const WorldPoint& wp,
                                                          ElVis::ReferencePointParameterType referenceType, ReferencePoint& result)
{
    ElVisError returnVal = eNoError;
    if( elementType == 0 )
    {
        result = TransformWorldToTensor(HexVertexBuffer, elementId, wp);
    }
    else if( elementType == 1 )
    {
        result = TransformPrismWorldToTensor(PrismVertexBuffer,elementId, wp);
    }
    else
    {
        returnVal = eInvalidElementType;
    }
    return returnVal;
}

template<typename PointType, typename ResultType>
ELVIS_DEVICE ElVisError SampleScalarFieldAtReferencePointCuda(int elementId, int elementType, int fieldId,
                                                               const PointType& worldPoint,
                                                               const PointType& tp,
                                                               ResultType& result)
{
    ElVisError returnVal = eNoError;
    if( elementType == 0 )
    {
        uint3 degree = HexDegrees[elementId];

        uint coefficientIndex = HexCoefficientIndices[elementId];
        ElVisFloat* coeffs = &(HexCoefficients[coefficientIndex]);

        result = EvaluateHexFieldAtTensorPoint(degree, tp.x, tp.y, tp.z, coeffs);
    }
    else if( elementType == 1 )
    {
        uint3 degree = PrismDegrees[elementId];

        uint coefficientIndex = PrismCoefficientIndices[elementId];
        ElVisFloat* coeffs = &(PrismCoefficients[coefficientIndex]);

        result = EvaluatePrismFieldAtTensorPoint(degree, tp.x, tp.y, tp.z, coeffs);
    }
    else
    {
        returnVal = eInvalidElementType;
    }
    return returnVal;
}

ELVIS_DEVICE ElVisError SampleReferenceGradientCuda(int elementId, int elementType, int fieldId, const ReferencePoint& refPoint, ElVisFloat3& gradient)
{
    gradient = CalculateTensorGradient(elementId, elementType, fieldId, refPoint);
    return eNoError;
}

ELVIS_DEVICE ElVisError SampleGeometryMappingJacobian(int elementId, int elementType, const ReferencePoint& refPoint, ElVisFloat* J)
{
    if( elementType == 0 )
    {
        calculateTensorToWorldSpaceMappingJacobian(HexVertexBuffer, elementId, refPoint, J);
    }
    else
    {
        GetPrismWorldToReferenceJacobian(PrismVertexBuffer, elementId, refPoint, J);
    }
    return eNoError;
}



//ELVIS_DEVICE ElVisFloat3 EvaluateNormalCuda(unsigned int elementId, unsigned int elementType, int fieldId, const ElVisFloat3& worldPoint)
//{
//    ElVisFloat3 result = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));

//    ReferencePoint tp;
//    ConvertWorldToReferenceSpaceCuda(elementId, elementType, worldPoint, ElVis::eReferencePointIsInvalid, tp);

//    ElVisFloat3 tv = CalculateTensorGradient(elementId, elementType, fieldId, tp);

//    ElVisFloat J[9];
//    CalculateTransposedInvertedMappingJacobianCuda(elementId, elementType, fieldId, tp, J);

//    result.x = tv.x*J[0] + tv.y*J[1] + tv.z*J[2];
//    result.y = tv.x*J[3] + tv.y*J[4] + tv.z*J[5];
//    result.z = tv.x*J[6] + tv.y*J[7] + tv.z*J[8];

////    ELVIS_PRINTF("Normal Vector %f, %f, %f\n", result.x, result.y, result.z);
//    return result;
//}

#endif


