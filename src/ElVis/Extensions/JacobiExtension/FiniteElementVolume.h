////////////////////////////////////////////////////////////////////////////////
//
//  File: hoFiniteElementVolume.h
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

#ifndef ELVIS_JACOBI_EXTENSION__FINITE_ELEMENT_VOLUME_H_
#define ELVIS_JACOBI_EXTENSION__FINITE_ELEMENT_VOLUME_H_

#include <ElVis/Extensions/JacobiExtension/Polyhedra.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>

#include <ElVis/Core/Vector.hpp>
//#include <ray_tracer/rtDomUtil.h>
//#include <ray_tracer/rtRayTraceableObjectGroup.h>
//
//#include <ray_tracer/rtIntersectionInfo.h>
//#include <ray_tracer/rtMaterial.h>

#include <boost/foreach.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <ElVis/Extensions/JacobiExtension/CastAndFilterIterator.hpp>

using std::endl;

namespace ElVis
{
    namespace JacobiExtension
    {
        class PolyhedraBoundingBox;
        /// \brief Represents a volume consisting of finite elements.
        class FiniteElementVolume 
        {
        public:
            /// \brief Initializes the volume from a file.
            /// \param matl The material of the volume.  Used to color the image.
            /// \param fileName The name of the file containing the volume.
            /// \param createGrid true if you wish to create a grid to accelerate the ray tracing.
            /// \param polyDegree The degree of the interpolating polynomial.
            JACOBI_EXTENSION_EXPORT FiniteElementVolume(const char* fileName);


            // FiniteElementVolume(XERCES_CPP_NAMESPACE_QUALIFIER DOMElement* materialElement, boost::shared_ptr<Scene> scene);


            JACOBI_EXTENSION_EXPORT FiniteElementVolume();
            JACOBI_EXTENSION_EXPORT ~FiniteElementVolume();

            JACOBI_EXTENSION_EXPORT static bool CanLoadVolume(const char* fileName);

            /// \brief Calculates the scalar value at the given point.
            /// \param p The point you want evaluated.
            /// \throw OutOfBoundsPoint If p is not inside the volume.
            /// This function will find the element in the volume which
            /// contains the point p and then will evaluate the interpolation
            /// function in that element at p to get the scalar value.
            JACOBI_EXTENSION_EXPORT  double calculateScalarValue(const ElVis::WorldPoint& p);

            /// \brief Calculates the scalar value, normal, and hessian at the given point.
            /// \param p The world space point to be evaluated.
            /// \param scalarValue The scalar value of the volume at p.
            /// \param normal The world space normal vector at p (un-normalized).
            /// \param hessian The world space hessian at p.
            /// \return true if a valid point is found, false otherwise.
            JACOBI_EXTENSION_EXPORT  bool sample(const ElVis::WorldPoint& p,
                double& scalarValue, ElVis::WorldVector& worldNormal,
                ElVis::TensorVector& tensorGradient,
                JacobiExtension::Matrix<double, 3, 3>& scalarHessian,
                JacobiExtension::Matrix<double, 3, 3>& mappingJacobian,
                JacobiExtension::Matrix<double, 3, 3>& mappingRHessian,
                JacobiExtension::Matrix<double, 3, 3>& mappingSHessian,
                JacobiExtension::Matrix<double, 3, 3>& mappingTHessian);

            /// \brief Get the element at the specified location.
            JACOBI_EXTENSION_EXPORT  boost::shared_ptr<Polyhedron> getElement(int index);
            JACOBI_EXTENSION_EXPORT  size_t numElements() const { return elements.size(); }

            // Find the closest polyhedra.
            //JACOBI_EXTENSION_EXPORT  boost::shared_ptr<Polyhedron> findIntersectedPolyhedron(const rt::Ray& ray);

            // Calculate the normal at the given ElVis::WorldPoint.
            //JACOBI_EXTENSION_EXPORT  virtual ElVis::WorldVector normal(const ElVis::WorldPoint&, const rt::IntersectionInfo& hit);

            /// \brief Returns the minimum and maximum scalar values in the volume.
            JACOBI_EXTENSION_EXPORT  virtual void get_minmax(double& min, double& max);
            JACOBI_EXTENSION_EXPORT  virtual double get_min();
            JACOBI_EXTENSION_EXPORT  virtual double get_max();

            JACOBI_EXTENSION_EXPORT void setMaterialForElements();
            /// \brief Calculates the overall min and max scalar value of the volume.
            JACOBI_EXTENSION_EXPORT  void calcOverallMinMaxScalar();

            /// \brief Calculates the bounds of the smallest axis aligned box
            /// which encloses the volume.
            JACOBI_EXTENSION_EXPORT  void calcOverallBoundingBox(ElVis::WorldPoint& min, ElVis::WorldPoint& max) const;
            JACOBI_EXTENSION_EXPORT  virtual double smallestX() const;
            JACOBI_EXTENSION_EXPORT  virtual double smallestY() const;
            JACOBI_EXTENSION_EXPORT  virtual double smallestZ() const;
            JACOBI_EXTENSION_EXPORT  virtual double largestX() const;
            JACOBI_EXTENSION_EXPORT  virtual double largestY() const;
            JACOBI_EXTENSION_EXPORT  virtual double largestZ() const;



            // TODO - This won't work in rtrt.
            boost::shared_ptr<Polyhedron> intersectedObject;
            double errorAtIntersection;
            double cn;
            virtual void print(std::ostream& traceFile) const
            {
                traceFile << "Finite Element Volume" << endl;
                traceFile << "Name: " << m_fileName << endl;
                //traceFile << "Polyhedron id " << intersectedObject->id();
            }

            //void setStartMaterial(const rt::Material& rhs) { m_startMaterial = rhs; }
            //void setEndMaterial(const rt::Material& rhs) { m_endMaterial = rhs; }
            void setLowerBound(double newBound) { if( newBound < m_upperBound ) m_lowerBound = newBound; }
            void setUpperBound(double newBound) { if( newBound > m_lowerBound ) m_upperBound = newBound; }
            void enableColorMap() { m_enableColorMap = true; }
            bool colorMapEnabled() const { return m_enableColorMap; }

            double isovalue() const { return m_isoval; }
            double isovalue(double newVal) { return m_isoval = newVal; }

            void addAbsoluteError(double error);
            void addRelativeError(double error);
            void addPolynomialError(double error);

            /// Writers
            /// These functions allow you to write the volume to various formats.

            /// \brief Writes a binary file.
            void writeVolume(FILE* outFile);

            JACOBI_EXTENSION_EXPORT void WriteCellVolume(unsigned int elementId, const std::string& filePrefix);
            JACOBI_EXTENSION_EXPORT void WriteCellVolumeForVTK(unsigned int elementId, const std::string& filePrefix);

            /// \brief Writes an unstructured grid for VTK using the cell types and vertex values of the elements.
            JACOBI_EXTENSION_EXPORT void writeCellVolumeForVTK(const char* fileName);

            /// \brief Writes the volume header to the given file.
            JACOBI_EXTENSION_EXPORT static bool writeHeader(FILE* file);

            template<typename T>
            CastAndFilterIterator<Polyhedron, T, typename PolyhedraVector::iterator>
                IterateElementsOfType()
            {
                CastAndFilterIterator<Polyhedron, T, typename PolyhedraVector::iterator> result(
                    elements.begin(), elements.end());
                return result;
            }

            template<typename T>
            int NumElementsOfType()
            {
                int result = 0;
                BOOST_FOREACH(boost::shared_ptr<Polyhedron> iter, IterateElementsOfType<T>() )
                {
                    ++result;
                }
                return result;
            }

        private:
            //virtual bool do_findIntersection(const rt::Ray& theRay, rt::IntersectionInfo& info) const;
            //virtual ElVis::WorldVector do_calculateNormal(const rt::IntersectionInfo& info) const;
            //virtual void do_addObject(boost::shared_ptr<rt::RayTraceableObject> obj);
            //virtual void do_addObjectGroup(boost::shared_ptr<const rt::RayTraceableObjectGroup> group);

            void init();

            void printStats(std::ostream& os, bool printHeader);

            /// \brief The smallest scalar value in the dataset.
            double m_minValue;

            /// \brief The largest scalar value in the dataset.
            double m_maxValue;

            /// \brief A list of all elements in the volume.
            PolyhedraVector elements;

            // calculateScalarValue does a linear search.  Assuming points 
            // will be somewhat adjacent, we'll start the search at the 
            // last element the return sucessfully.
            boost::shared_ptr<JacobiExtension::Polyhedron> LastFoundElement;

            /// \brief The file name of the volume file used to construct this volume.
            std::string m_fileName;

            /// \brief An acceleration structure for ray tracing.
            PolyhedraBoundingBox* itsBBoxes;

            // The following allows us to create a color map
            // based on the scalar value of the intersection ElVis::WorldPoint.
            // The color map is based on magnitude and is clamped at the ends.
            //rt::Material m_startMaterial;
            //rt::Material m_endMaterial;

            double m_lowerBound;
            double m_upperBound;
            bool m_enableColorMap;

            // Allows the user to specify the degree of the interpolating polynomial.
            // If this is not set then defaults will be used for all elements.
            unsigned int m_polynomialDegree;
            double m_tolerance;

            /// \brief The isovalue desired.
            /// Do we want/need to be able to do multiple values?
            double m_isoval;

            /// \brief The header for the finite element binary file.
            static const char* headerData;

            bool m_createGrid;
        public:
            /// \brief Thrown when an operation is requested for the volume which is out of bounds.
            class OutOfBoundsPoint : public std::runtime_error
            {
            public:
                OutOfBoundsPoint(const std::string& what_arg);
            };
        };
    }
}

#endif
