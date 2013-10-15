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

#ifndef ELVIS_CUT_SURFACE_PAYLOADS_H
#define ELVIS_CUT_SURFACE_PAYLOADS_H

#include <optix_cuda.h>
#include <optix_math.h>
#include <ElVis/Core/util.cu>
#include <ElVis/Core/Float.cu>
#include <ElVis/Core/ReferencePointParameter.h>

rtDeclareVariable(ElVisFloat3, BGColor, , );

// TODO - Update the payloads.
// Used by rays that query scalar at specific points.
// Set the found value with
// *scalarValue = value;
// If a scalar value is not found, then the background color is used 
// and specified in result;
struct CutSurfaceScalarValuePayload
{      
    ELVIS_DEVICE void Initialize()
    {
        isValid = false;
        ReferencePointSet = false;
        scalarValue = ELVIS_FLOAT_MAX;
        IntersectionT = -1.0f;
        Normal = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
        Color = BGColor;
        elementId = -1;
        elementType = -1;
    }

    int isValid;
    int elementId;
    int elementType;
    ElVisFloat3 ReferenceIntersectionPoint;
    ElVisFloat3 result;
    ElVisFloat3 IntersectionPoint;
    float IntersectionT;
    ElVisFloat3 Normal;
    ElVisFloat3 Color;
    ElVisFloat scalarValue;
    int ReferencePointSet;
};


/// \brief Ray payload designed for use with the element finder routines
///        in FineElement.cu
struct ElementFinderPayload
{
    ELVIS_DEVICE ElementFinderPayload() 
    {
    }

    /// This method takes the place of a constructor, as constructors in payload objects
    /// are not support by OptiX 2.5 and earlier.
    ELVIS_DEVICE void Initialize(const ElVisFloat3& p)
    {
        IntersectionPoint = p;
        elementId = -1;
        elementType = -1;
        ReferencePointType = ElVis::eReferencePointIsInvalid;
        ReferenceIntersectionPoint = MakeFloat3(ELVIS_FLOAT_MAX, ELVIS_FLOAT_MAX, ELVIS_FLOAT_MAX);
    }

    ELVIS_DEVICE ElementFinderPayload(const ElementFinderPayload& rhs) :
        IntersectionPoint(rhs.IntersectionPoint),
        elementId(rhs.elementId),
        elementType(rhs.elementType),
        ReferencePointType(rhs.ReferencePointType),
        ReferenceIntersectionPoint(rhs.ReferenceIntersectionPoint)
    {
    }

    ELVIS_DEVICE ElementFinderPayload& operator=(const ElementFinderPayload& rhs)
    {
        IntersectionPoint = MakeFloat3(rhs.IntersectionPoint.x, rhs.IntersectionPoint.y, rhs.IntersectionPoint.z);
        elementId = rhs.elementId;
        elementType = rhs.elementType;
        ReferencePointType = rhs.ReferencePointType;
        ReferenceIntersectionPoint = MakeFloat3(rhs.ReferenceIntersectionPoint.x, rhs.ReferenceIntersectionPoint.y, rhs.ReferenceIntersectionPoint.z);
        return *this;
    }
    /// \brief The point for which the enclosing element is sought.
    ElVisFloat3 IntersectionPoint;

    /// \brief The element id that encloses IntersectionPoint.
    int elementId;

    /// \brief The element type that encloses IntersectionPoint.
    int elementType;

    /// \brief In some cases, the determination of the enclosing element also calculates the reference
    ///        points associated with IntersectionPoint.  If that occurs, this can be stored to prevent
    ///        further processing later.
    ///
    /// If a reference point is calculated as part of the find element procecdure, set this value to
    /// eReferencePointIsValid, otherwise leave it as eReferencePointIsInvalid,
    ElVis::ReferencePointParameterType ReferencePointType;

    /// \brief If ReferencePointType is set to eReferencePointIsValid, then this contains the calculated reference
    ///        point.
    ElVisFloat3 ReferenceIntersectionPoint;
};

#endif //ELVIS_CUT_SURFACE_PAYLOADS_H
