////////////////////////////////////////////////////////////////////////////////
//
//  File: hoPointTransformations.cpp
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
#include <ElVis/Extensions/JacobiExtension/Isosurface.h>
#include <ElVis/Extensions/JacobiExtension/PointTransformations.hpp>

namespace ElVis
{
    namespace JacobiExtension
    {
        namespace PointTransformations
        {
            ElVis::TensorPoint
                transformHexReferenceToTensor(
                const ElVis::ReferencePoint& p)
            {
                return ElVis::TensorPoint(p.r(), p.s(), p.t());
            }

            ElVis::ReferencePoint transformHexTensorToReference(
                const ElVis::TensorPoint& p)
            {
                return ElVis::ReferencePoint(p.a(), p.b(), p.c());
            }

            ElVis::WorldPoint transformReferenceToWorld(
                const Hexahedron& hex, const ElVis::ReferencePoint& p)
            {
                const ElVis::ReferencePoint::DataType& r = p.r();
                const ElVis::ReferencePoint::DataType& s = p.s();
                const ElVis::ReferencePoint::DataType& t = p.t();

                double t1 = (1.0-r)*(1.0-s)*(1.0-t);
                double t2 = (1.0+r)*(1.0-s)*(1.0-t);
                double t3 = (1.0+r)*(1.0+s)*(1.0-t);
                double t4 = (1.0-r)*(1.0+s)*(1.0-t);
                double t5 = (1.0-r)*(1.0-s)*(1.0+t);
                double t6 = (1.0+r)*(1.0-s)*(1.0+t);
                double t7 = (1.0+r)*(1.0+s)*(1.0+t);
                double t8 = (1.0-r)*(1.0+s)*(1.0+t);

                double x = .125 * (t1*hex.vertex(0).x() + t2*hex.vertex(1).x() +
                    t3*hex.vertex(2).x() + t4*hex.vertex(3).x() +
                    t5*hex.vertex(4).x() + t6*hex.vertex(5).x() +
                    t7*hex.vertex(6).x() + t8*hex.vertex(7).x());

                double y = .125 * (t1*hex.vertex(0).y() + t2*hex.vertex(1).y() +
                    t3*hex.vertex(2).y() + t4*hex.vertex(3).y() +
                    t5*hex.vertex(4).y() + t6*hex.vertex(5).y() +
                    t7*hex.vertex(6).y() + t8*hex.vertex(7).y());

                double z = .125 * (t1*hex.vertex(0).z() + t2*hex.vertex(1).z() +
                    t3*hex.vertex(2).z() + t4*hex.vertex(3).z() +
                    t5*hex.vertex(4).z() + t6*hex.vertex(5).z() +
                    t7*hex.vertex(6).z() + t8*hex.vertex(7).z());

                return ElVis::WorldPoint(x,y,z);
            }

            ElVis::TensorPoint transformPrismReferenceToTensor(const ElVis::ReferencePoint& p)
            {
                const ElVis::ReferencePoint::DataType& r = p.r();
                const ElVis::ReferencePoint::DataType& s = p.s();
                const ElVis::ReferencePoint::DataType& t = p.t();

                if( t != 1 )
                {
                    double a = 2.0*(r+1.0)/(1.0-t) - 1.0;
                    double b = s;
                    double c = t;

                    return ElVis::TensorPoint(a,b,c);
                }
                else
                {
                    // In this case we're on the collapsed edge.
                    // Pick a tensor point on the corresponding
                    // face.
                    // So just pick a.
                    return ElVis::TensorPoint(static_cast<ElVis::ReferencePoint::DataType>(0.0), s, t);
                }
            }

            ElVis::ReferencePoint transformPrismTensorToReference(const ElVis::TensorPoint& p)
            {
                const ElVis::ReferencePoint::DataType& a = p.a();
                const ElVis::ReferencePoint::DataType& b = p.b();
                const ElVis::ReferencePoint::DataType& c = p.c();

                double r = (1.0+a)/2.0 * (1.0-c) - 1.0;
                double s = b;
                double t = c;

                return ElVis::ReferencePoint(r,s,t);
            }

            ElVis::WorldPoint transformReferenceToWorld(
                const Prism& prism, const ElVis::ReferencePoint& p)
            {
                const ElVis::ReferencePoint::DataType& r = p.r();
                const ElVis::ReferencePoint::DataType& s = p.s();
                const ElVis::ReferencePoint::DataType& t = p.t();

                double t1 = -(r+t)*(1.-s);
                double t2 = (1.+r)*(1.-s);
                double t3 = (1.+r)*(1.+s);
                double t4 = -(r+t)*(1.+s);
                double t5 = (1.-s)*(1.+t);
                double t6 = (1.+s)*(1.+t);

                double x = .25 * (t1*prism.vertex(0).x() + t2*prism.vertex(1).x() +
                    t3*prism.vertex(2).x() + t4*prism.vertex(3).x() +
                    t5*prism.vertex(4).x() + t6*prism.vertex(5).x());

                double y = .25 * (t1*prism.vertex(0).y() + t2*prism.vertex(1).y() +
                    t3*prism.vertex(2).y() + t4*prism.vertex(3).y() +
                    t5*prism.vertex(4).y() + t6*prism.vertex(5).y());

                double z = .25 * (t1*prism.vertex(0).z() + t2*prism.vertex(1).z() +
                    t3*prism.vertex(2).z() + t4*prism.vertex(3).z() +
                    t5*prism.vertex(4).z() + t6*prism.vertex(5).z());

                return ElVis::WorldPoint(x,y,z);
            }

            ElVis::TensorPoint transformPyramidReferenceToTensor(const ElVis::ReferencePoint& p)
            {
                const ElVis::ReferencePoint::DataType& r = p.r();
                const ElVis::ReferencePoint::DataType& s = p.s();
                const ElVis::ReferencePoint::DataType& t = p.t();

                double c = t;
                if( t != 1.0 )
                {
                    double b = 2.0*(s+1.0)/(1.0-t) - 1.0;
                    double a = 2.0*(r+1.0)/(1.0-t) - 1.0;

                    return ElVis::TensorPoint(a,b,c);
                }
                else
                {
                    // Pick a point.
                    return ElVis::TensorPoint(0.0, 0.0, c);
                }
            }

            ElVis::ReferencePoint transformPyramidTensorToReference(const ElVis::TensorPoint& p)
            {
                const ElVis::ReferencePoint::DataType& a = p.a();
                const ElVis::ReferencePoint::DataType& b = p.b();
                const ElVis::ReferencePoint::DataType& c = p.c();

                double r = (1.0+a)*(1.0-c)/2.0 - 1.0;
                double s = (1.0+b)*(1.0-c)/2.0 - 1.0;
                double t = c;

                return ElVis::ReferencePoint(r,s,t);
            }

            // ElVis::WorldPoint transformReferenceToWorld(
            //  const Pyramid& pyramid, const ElVis::ReferencePoint& p)
            //{
            //  const DataType& r = p.r();
            //  const DataType& s = p.s();
            //  const DataType& t = p.t();
            //
            //  DataType t1 = ((r+t)*(s+t))/((1.0-t)*2.0);
            //  DataType t2 = ((1.0+r)*(s+t))/((1.0-t)*2.0);
            //  DataType t3 = ((1.0+r)*(1.0+s))/((1.0-t)*2.0);
            //  DataType t4 = ((r+t)*(1.0+s))/((1.0-t)*2.0);
            //  DataType t5 = (1.0+t)/2.0;
            //
            //  DataType x = t1*pyramid.vertex(0).x() - t2*pyramid.vertex(1).x() +
            //      t3*pyramid.vertex(2).x() - t4*pyramid.vertex(3).x() + t5*pyramid.vertex(4).x();
            //  DataType y = t1*pyramid.vertex(0).y() - t2*pyramid.vertex(1).y() +
            //      t3*pyramid.vertex(2).y() - t4*pyramid.vertex(3).y() + t5*pyramid.vertex(4).y();
            //  DataType z = t1*pyramid.vertex(0).z() - t2*pyramid.vertex(1).z() +
            //      t3*pyramid.vertex(2).z() - t4*pyramid.vertex(3).z() + t5*pyramid.vertex(4).z();
            //
            //  return ElVis::WorldPoint(x,y,z);
            //}
            //
            ElVis::TensorPoint transformTetReferenceToTensor(const ElVis::ReferencePoint& p)
            {
                const ElVis::ReferencePoint::DataType& r = p.r();
                const ElVis::ReferencePoint::DataType& s = p.s();
                const ElVis::ReferencePoint::DataType& t = p.t();

                double c = t;
                double b = 0.0;
                double a = 0.0;

                if( t != 1.0 )
                {
                    b = 2.0*(s+1.0)/(1.0-t) - 1.0;
                }

                if( t != s )
                {
                    a = -2.0*(r+1.0)/(t+s)-1.0;
                }

                return ElVis::TensorPoint(a,b,c);
            }

            ElVis::ReferencePoint transformTetTensorToReference(const ElVis::TensorPoint& p)
            {
                const ElVis::ReferencePoint::DataType& a = p.a();
                const ElVis::ReferencePoint::DataType& b = p.b();
                const ElVis::ReferencePoint::DataType& c = p.c();

                double r = ( (1.0+a)*(1.0-b) )/4.0 * (1.0-c) - 1.0;
                double s = (1.0+b)/2.0 * (1.0-c) - 1.0;
                double t = c;

                return ElVis::ReferencePoint(r,s,t);
            }

            // ElVis::WorldPoint transformReferenceToWorld(
            //    const Tetrahedron& tet, const ElVis::ReferencePoint& p)
            //{
            //    const ElVis::ReferencePoint::DataType& r = p.r();
            //    const ElVis::ReferencePoint::DataType& s = p.s();
            //    const ElVis::ReferencePoint::DataType& t = p.t();

            //    ElVis::ReferencePoint::DataType t1 = -(1.0+r+s+t)/2.0;
            //    ElVis::ReferencePoint::DataType t2 = (1.0+r)/2.0;
            //    ElVis::ReferencePoint::DataType t3 = (1.0+s)/2.0;
            //    ElVis::ReferencePoint::DataType t4 = (1.0+t)/2.0;

            //    ElVis::ReferencePoint::DataType x = t1*tet.vertex(0).x() + t2*tet.vertex(1).x() +
            //        t3*tet.vertex(2).x() + t4*tet.vertex(3).x();
            //    ElVis::ReferencePoint::DataType y = t1*tet.vertex(0).y() + t2*tet.vertex(1).y() +
            //        t3*tet.vertex(2).y() + t4*tet.vertex(3).y();
            //    ElVis::ReferencePoint::DataType z = t1*tet.vertex(0).z() + t2*tet.vertex(1).z() +
            //        t3*tet.vertex(2).z() + t4*tet.vertex(3).z();

            //    return ElVis::WorldPoint(x,y,z);
            //}
        }
    }
}

