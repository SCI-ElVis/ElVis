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

#ifndef ELVIS_NEKTAR_MODEL_CU
#define ELVIS_NEKTAR_MODEL_CU

#include <ElVis/Core/Float.cu>
#include <LibUtilities/Foundations/BasisType.h>


// All coefficients for all fields for all elements.
// Nektar++ numbers fields 0...n.  Since the number of coefficients 
// per element is variable, to obtain the coefficients for an element,
// we must use the CoefficientOffsets.
//
// ElVisFloat* startIndex = Coefficients[FieldId*
rtBuffer<ElVisFloat> Coefficients;

rtBuffer<uint> SumPrefixNumberOfFieldCoefficients;

// Index of the start of the coefficient buffer by element.
rtBuffer<uint> CoefficientOffsets;

// The vertices associated with this hex.
rtBuffer<ElVisFloat4> Vertices;

rtDeclareVariable(uint, NumElements, ,);

rtBuffer<uint3> FieldModes;
rtBuffer<Nektar::LibUtilities::BasisType> FieldBases;

__device__ uint GetNumberOfFields()
{
    return SumPrefixNumberOfFieldCoefficients.size();
}

// Returns a pointer to the beginning of the coefficients for the field.
__device__ ElVisFloat* GetFieldCoefficientStart(int fieldId)
{
    return &Coefficients[SumPrefixNumberOfFieldCoefficients[fieldId]];
}

__device__ ElVisFloat* GetFieldCoefficients(int fieldId, int elementId)
{
    ElVisFloat* base = GetFieldCoefficientStart(fieldId);
    int offset = CoefficientOffsets[fieldId*NumElements+elementId];
    return base+offset;
}

__device__ uint3 GetModes(int fieldId, int elementId)
{
    return FieldModes[fieldId*NumElements + elementId];
}

__device__ Nektar::LibUtilities::BasisType GetBasis(int fieldId, int elementId, int dir)
{
    return FieldBases[fieldId*NumElements + elementId*3 + dir];
}

#endif 
