////////////////////////////////////////////////////////////////////////////////
//
//  File: hoPointTransformations.hpp
//
//
//  The MIT License
//
//  Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
//  Department of Aeronautics, Imperial College London (UK), and Scientific
//  Computing and Imaging Institute, University of Utah (USA).
//
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
//  Description:
//
//
////////////////////////////////////////////////////////////////////////////////

#ifndef ELVIS_JACOBI_EXTENSION___POINT_TRANSFORMATIONS_HPP__
#define ELVIS_JACOBI_EXTENSION___POINT_TRANSFORMATIONS_HPP__

#include <ElVis/Extensions/JacobiExtension/Hexahedron.h>
#include <ElVis/Extensions/JacobiExtension/Prism.h>
#include <ElVis/Extensions/JacobiExtension/Pyramid.h>
#include <ElVis/Extensions/JacobiExtension/Tetrahedron.h>

namespace ElVis
{
    namespace JacobiExtension
    {
        namespace PointTransformations
        {
            ElVis::TensorPoint
                transformHexReferenceToTensor(const ElVis::ReferencePoint& p);

            ElVis::ReferencePoint
                transformHexTensorToReference(const ElVis::TensorPoint& p);

            ElVis::WorldPoint
                transformReferenceToWorld(const Hexahedron& hex, const ElVis::ReferencePoint& p);


            ElVis::TensorPoint
                transformPrismReferenceToTensor(const ElVis::ReferencePoint& p);


            ElVis::ReferencePoint
                transformPrismTensorToReference(const ElVis::TensorPoint& p);


            ElVis::WorldPoint
                transformReferenceToWorld(
                const Prism& prism, const ElVis::ReferencePoint& p);

            ElVis::TensorPoint
                transformPyramidReferenceToTensor(const ElVis::ReferencePoint& p);


            ElVis::ReferencePoint
                transformPyramidTensorToReference(const ElVis::TensorPoint& p);
            // ElVis::WorldPoint transformReferenceToWorld(
            //  const Pyramid& pyramid, const ElVis::ReferencePoint& p);


            ElVis::TensorPoint
                transformTetReferenceToTensor(const ElVis::ReferencePoint& p);


            ElVis::ReferencePoint
                transformTetTensorToReference(const ElVis::TensorPoint& p);


            //ElVis::WorldPoint transformReferenceToWorld(
            //    const Tetrahedron& tet, const ElVis::ReferencePoint& p);
        }

    }
}

#endif

