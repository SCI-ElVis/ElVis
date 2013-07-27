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

#ifndef ELVIS_JACOBI_EXTENSION_ELVIS_MODEL_H
#define ELVIS_JACOBI_EXTENSION_ELVIS_MODEL_H

#include <ElVis/Extensions/JacobiExtension/FiniteElementVolume.h>
#include <ElVis/Extensions/JacobiExtension/Polyhedra.h>
#include <ElVis/Extensions/JacobiExtension/Declspec.h>

#include <ElVis/Core/Model.h>
#include <ElVis/Core/Float.h>
#include <ElVis/Core/OptiXBuffer.hpp>

#include <optixu/optixpp.h>
#include <ElVis/Core/InteropBuffer.hpp>
#include <ElVis/Core/ElementId.h>
#include <ElVis/Core/FaceDef.h>

#include <ElVis/Extensions/JacobiExtension/Hexahedron.h>
#include <ElVis/Extensions/JacobiExtension/Tetrahedron.h>
#include <ElVis/Extensions/JacobiExtension/Prism.h>

#include <map>
#include <set>

namespace ElVis
{
    namespace JacobiExtension
    {
        // Temporary face structure to find unique faces among all elements.
        struct JacobiFace
        {
//            JacobiFace(const WorldPoint& point0, const WorldPoint& point1, const WorldPoint& point2) :
////                p0(point0),
////                p1(point1),
////                p2(point2),
////                p3(std::numeric_limits<ElVisFloat>::max(), std::numeric_limits<ElVisFloat>::max(), std::numeric_limits<ElVisFloat>::max()),
//                NumEdges(3),
//                normal()
//            {
//                p[0] = point0;
//                p[1] = point1;
//                p[2] = point2;
//                p[3] = WorldPoint(std::numeric_limits<ElVisFloat>::max(), std::numeric_limits<ElVisFloat>::max(), std::numeric_limits<ElVisFloat>::max());

//                for(int i = 0; i < 4; ++i)
//                {
//                    sorted[i] = p[i];
//                }

//                std::sort(sorted, sorted+4);
//            }

            JacobiFace(const WorldPoint& point0, const WorldPoint& point1, const WorldPoint& point2, const WorldPoint& point3, int numEdges, const WorldVector& n) :
//                p0(point0),
//                p1(point1),
//                p2(point2),
//                p3(point3),
                NumEdges(numEdges),
                normal(n)
            {
                p[0] = point0;
                p[1] = point1;
                p[2] = point2;
                p[3] = point3;

                for(int i = 0; i < 4; ++i)
                {
                    sorted[i] = p[i];
                }

                std::sort(sorted, sorted+4);
            }

            JacobiFace(const JacobiFace& rhs) :
//                p0(rhs.p0),
//                p1(rhs.p1),
//                p2(rhs.p2),
//                p3(rhs.p3),
                NumEdges(rhs.NumEdges),
                normal(rhs.normal)
            {
                for(int i = 0; i < 4; ++i)
                {
                    p[i] = rhs.p[i];
                }

                for(int i = 0; i < 4; ++i)
                {
                    sorted[i] = rhs.sorted[i];
                }
            }

            JacobiFace& operator=(const JacobiFace& rhs)
            {
                for(int i = 0; i < 4; ++i)
                {
                    p[i] = rhs.p[i];
                    sorted[i] = rhs.sorted[i];
                }
                NumEdges = rhs.NumEdges;
                normal = rhs.normal;
                return *this;
            }

            WorldPoint MinExtent() const;
            WorldPoint MaxExtent() const;

            int NumVertices() const;

            WorldPoint p[4];
            WorldPoint sorted[4];
            int NumEdges;
//            WorldPoint p0;
//            WorldPoint p1;
//            WorldPoint p2;
//            WorldPoint p3;
            WorldVector normal;
        };

        bool operator<(const JacobiFace& lhs, const JacobiFace& rhs);

        class JacobiExtensionModel : public ElVis::Model
        {
        public:
            JACOBI_EXTENSION_EXPORT JacobiExtensionModel();
            JACOBI_EXTENSION_EXPORT virtual ~JacobiExtensionModel();

            JACOBI_EXTENSION_EXPORT void LoadVolume(const std::string& filePath);
            JACOBI_EXTENSION_EXPORT void writeCellVolumeForVTK(const char* fileName);
            JACOBI_EXTENSION_EXPORT boost::shared_ptr<FiniteElementVolume> Volume() const { return m_volume; }

            // Number of copies is a temporary idea for testing purposes.  I needed large 
            // volumes to test, but Nektar++ models use far too much memory, so I couldn't
            // create anything large, nor did I have any volumes from before that were larger
            // than 10,000 elements.  So I can arbitrarily inflate the sizes by creating 
            // extra copies.
            JACOBI_EXTENSION_EXPORT void SetNumberOfCopies(unsigned int n) 
            {
                m_numberOfCopies = n;
            }

            // Number of modes is temporary and allows the user to make the volume 
            // one with an arbitrary number of modes.  This is for testing purposes for 
            // evaluating high-order data.
            JACOBI_EXTENSION_EXPORT void SetNumberOfModes(int n)
            {
                m_numberOfModes = n;
            }

        protected:
            virtual std::vector<optixu::GeometryGroup> DoGetPointLocationGeometry(Scene* scene, optixu::Context context);
            virtual void DoGetFaceGeometry(Scene* scene, optixu::Context context, optixu::Geometry& faces );
            virtual std::vector<optixu::GeometryInstance> DoGet2DPrimaryGeometry(Scene* scene, optixu::Context context);
            virtual optixu::Material DoGet2DPrimaryGeometryMaterial(SceneView* view);

            virtual int DoGetNumberOfBoundarySurfaces() const;
            virtual void DoGetBoundarySurface(int surfaceIndex, std::string& name, std::vector<int>& faceIds);

            virtual void DoCalculateExtents(WorldPoint& min, WorldPoint& max);
            //virtual unsigned int DoGetNumberOfPoints() const;
            virtual WorldPoint DoGetPoint(unsigned int id) const;
            virtual const std::string& DoGetPTXPrefix() const;
            virtual unsigned int DoGetNumberOfElements() const;

            virtual int DoGetNumFields() const;
            virtual FieldInfo DoGetFieldInfo(unsigned int index) const;

        private:
            JacobiExtensionModel(const JacobiExtensionModel& rhs);

            static const std::string HexahedronIntersectionProgramName;
            static const std::string HexahedronPointLocationProgramName;
            static const std::string HexahedronBoundingProgramName;

            static const std::string PrismIntersectionProgramName;
            static const std::string PrismPointLocationProgramName;
            static const std::string PrismBoundingProgramName;

            JacobiExtensionModel& operator=(const JacobiExtensionModel& rhs);

            virtual void DoMapInteropBufferForCuda();
            virtual void DoUnMapInteropBufferForCuda();
            virtual int DoGetModelDimension() const { return 3; }

            unsigned int RequiredCoefficientStorage(unsigned int numberOfCoefficients, unsigned int alignment) const
            {
                return numberOfCoefficients;
                //unsigned int baseline = numberOfCoefficients/alignment;
                //if( baseline%alignment != 0 )
                //{
                //    baseline += 1;
                //}
                //return baseline * alignment;
            }
            template<typename T>
            int NumCoefficientsForElementType(unsigned int alignment) const 
            {
                int result = 0;
                if( m_numberOfModes > 0 )
                {
                    BOOST_FOREACH(boost::shared_ptr<Polyhedron> iter, m_volume->IterateElementsOfType<T>() )
                    {
                        result += RequiredCoefficientStorage(iter->NumberOfCoefficientsForOrder(m_numberOfModes-1), alignment);
                    }
                }
                else
                {
                    BOOST_FOREACH(boost::shared_ptr<Polyhedron> iter, m_volume->IterateElementsOfType<T>() )
                    {
                        result += RequiredCoefficientStorage(iter->basisCoefficients().size(), alignment);
                    }
                }
                return result;
            }

            template<typename T> 
            ElVis::OptiXBuffer<int>& GetCoefficientIndexBuffer() ;

            template<typename T> 
            ElVis::OptiXBuffer<ElVisFloat>& GetCoefficientBuffer() ;

            template<typename T>
            ElVis::OptiXBuffer<ElVisFloat4>& GetPlaneBuffer() ;

            template<typename T>
            optixu::GeometryInstance CreateGeometryForElementType(boost::shared_ptr<FiniteElementVolume> volume, optixu::Context context, const std::string& variablePrefix)
            {
                unsigned int coefficientAlignment = 8;
                int numElements = volume->NumElementsOfType<T>()*m_numberOfCopies;
                int numCoefficients = NumCoefficientsForElementType<T>(coefficientAlignment)*m_numberOfCopies;
                optixu::GeometryInstance instance;

                std::cout << "Number of elements = " << numElements << std::endl;
                std::cout << "Number of Coefficients = " << numCoefficients << std::endl;

//                if( numElements == 0 )
//                {
//                    return optixu::GeometryInstance();
//                }
                std::string vertexBufferName = variablePrefix + "VertexBuffer";
                OptiXBuffer<ElVisFloat4> VertexBuffer(vertexBufferName);
                VertexBuffer.SetContext(context);
                VertexBuffer.SetDimensions(numElements*T::VertexCount);
                
                BOOST_AUTO(vertexData, VertexBuffer.Map());

                ElVis::OptiXBuffer<int>& CoefficientIndicesBuffer = GetCoefficientIndexBuffer<T>();
                CoefficientIndicesBuffer.SetContext(context);
                CoefficientIndicesBuffer.SetDimensions(numElements);
                BOOST_AUTO(coefficientIndicesData, CoefficientIndicesBuffer.Map());

                ElVis::OptiXBuffer<ElVisFloat>& CoefficientBuffer = GetCoefficientBuffer<T>();
                CoefficientBuffer.SetContext(context);
                CoefficientBuffer.SetDimensions(numCoefficients);
                BOOST_AUTO(coefficientData, CoefficientBuffer.Map());

                std::string degreesName = variablePrefix + "Degrees";
                optixu::Buffer DegreesBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, numElements);
                context[degreesName.c_str()]->set(DegreesBuffer);
                unsigned int* degreeData = static_cast<unsigned int*>(DegreesBuffer->map());


                std::string planeBufferName = variablePrefix + "PlaneBuffer";
                ElVis::OptiXBuffer<ElVisFloat4>& PlaneBuffer = GetPlaneBuffer<T>();
                PlaneBuffer.SetContext(context);
                PlaneBuffer.SetDimensions(8*numElements);
                BOOST_AUTO(planeData, PlaneBuffer.Map());

                const ElVis::WorldPoint& min = this->MinExtent();
                const ElVis::WorldPoint& max = this->MaxExtent();
                double range = (max.y() - min.y());

                int curElementId = 0;
                int curCoefficientIndex = 0;
                for(unsigned int copyId = 0; copyId < m_numberOfCopies; ++copyId)
                {
                    BOOST_FOREACH( boost::shared_ptr<Polyhedron> element, volume->IterateElementsOfType<T>() )
                    {
                        boost::shared_ptr<T> castElement = boost::dynamic_pointer_cast<T>(element);

                        // Degrees
                        int degreeIdx = curElementId*3;
                        if( m_numberOfModes > 0 )
                        {
                            degreeData[degreeIdx] = m_numberOfModes-1;
                            degreeData[degreeIdx+1] = m_numberOfModes-1;
                            degreeData[degreeIdx+2] = m_numberOfModes-1;
                        }
                        else
                        {
                            degreeData[degreeIdx] = element->degree(0);
                            degreeData[degreeIdx+1] = element->degree(1);
                            degreeData[degreeIdx+2] = element->degree(2);
                        }

                        // Vertices
                        // Each vertex is a float4.

                        for(int i = 0; i < T::VertexCount; ++i)
                        {
                            int vertexIdx = curElementId*T::VertexCount + i;
                            const ElVis::WorldPoint& p = element->vertex(i);
                            vertexData[vertexIdx].x = (float)p.x();
                            vertexData[vertexIdx].y = static_cast<ElVisFloat>(p.y() + copyId*range);
                            vertexData[vertexIdx].z = static_cast<ElVisFloat>(p.z());
                            vertexData[vertexIdx].w = 1.0;
                        }

                        // Coefficgeometryients
                        coefficientIndicesData[curElementId] = curCoefficientIndex;
                        unsigned int numCoefficients = element->basisCoefficients().size();
                        if( m_numberOfModes > 0 )
                        {
                            numCoefficients = element->NumberOfCoefficientsForOrder(m_numberOfModes-1);
                        }

                        unsigned int tempIndex = curCoefficientIndex;
                        for(unsigned int i = 0; i < numCoefficients; ++i)
                        {
                            if( i < element->basisCoefficients().size() )
                            {
                                coefficientData[tempIndex] = static_cast<float>(element->basisCoefficients()[i]);
                            }
                            else
                            {
                                coefficientData[tempIndex] = 0;
                            }
                            ++tempIndex;
                        }
                        unsigned int storageRequired = RequiredCoefficientStorage(numCoefficients, coefficientAlignment);
                        curCoefficientIndex += storageRequired;

                        // Faces and planes
                        int planeIdx = 8*curElementId;
                        for(int i = 0; i < T::NumFaces; ++i)
                        {
                            ElVisFloat4* base = planeData.get() + planeIdx + i;
                            castElement->GetFace(i, base[0].x, base[0].y, base[0].z, base[0].w);

                            // Adjust as needed for copies.
                            base[0].w = base[0].w - static_cast<ElVisFloat>(2.0*copyId*base[0].y);
                        }

                        ++curElementId;
                    }
                }

                DegreesBuffer->unmap();

                if( numElements == 0 ) return instance;
                optixu::Geometry geometry = context->createGeometry();
                geometry->setPrimitiveCount(numElements);

                instance = context->createGeometryInstance();
                instance->setGeometry(geometry);


                // Setup the per element type vertex/face mappings.
                std::string vertex_face_indexName = variablePrefix + "vertex_face_index";
                optixu::Buffer vertexFaceBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT4, T::NumFaces);
                instance[vertex_face_indexName.c_str()]->set(vertexFaceBuffer);
                unsigned int* vertexFaceBufferData = static_cast<unsigned int*>(vertexFaceBuffer->map()); 
                std::copy(T::VerticesForEachFace, T::VerticesForEachFace + 
                    4*T::NumFaces, vertexFaceBufferData);
                vertexFaceBuffer->unmap();

                return instance;

            }

            template<typename T>
            void UpdateFaceBuffersForElementType()
            {
            }

            template<typename T>
            void PopulateFaces(boost::shared_ptr<FiniteElementVolume> volume, std::map<JacobiFace, FaceDef>& values)
            {
                int id = 0;
                BOOST_FOREACH( boost::shared_ptr<Polyhedron> element, volume->IterateElementsOfType<T>() )
                {
                    for(unsigned int i = 0; i < T::NumFaces; ++i)
                    {
                        const WorldPoint& p0 = element->vertex(T::VerticesForEachFace[i*4]);
                        const WorldPoint& p1 = element->vertex(T::VerticesForEachFace[i*4+1]);
                        const WorldPoint& p2 = element->vertex(T::VerticesForEachFace[i*4+2]);
                        const WorldPoint& p3 = element->vertex(T::VerticesForEachFace[i*4+3]);

                        boost::shared_ptr<T> asT = boost::dynamic_pointer_cast<T>(element);

                        // Since the jacobi extension provides normals as inward facing normals, we need to invert
                        // them so they point out for the types of intersection tests we will be doing.
                        ElVis::WorldVector normal = -(asT->GetFaceNormal(i));
                        JacobiFace quadFace(p0, p1, p2, p3, T::NumEdgesForEachFace[i], normal);

                        std::map<JacobiFace, FaceDef>::iterator found = values.find(quadFace);
                        ElementId curElement;
                        curElement.Id = id;
                        curElement.Type = T::TypeId;

                        if( found != values.end() )
                        {
                            const JacobiFace& key = (*found).first;
                            FaceDef& value = (*found).second;
                            (*found).second.CommonElements[1] = curElement;
                        }
                        else
                        {
                            FaceDef value;
                            value.CommonElements[0] = curElement;

                            ElementId nextElement;
                            nextElement.Id = -1;
                            nextElement.Type = -1;
                            value.CommonElements[1] = nextElement;

                            values[quadFace] = value;
                        }
                    }
                    ++id;
                }
            }

            boost::shared_ptr<FiniteElementVolume> m_volume;
            unsigned m_numberOfCopies;
            int m_numberOfModes;

            // This set of vertices is used for external queries for points and isn't used for 
            // rendering.  I tried a boost::bimap, but couldn't figure out how to get a 
            // non-const WorldPoint reference out (which is required for vtkDataSet::GetPoint).
            std::set<WorldPoint> m_verticesLookupMap;
            std::vector<WorldPoint> m_vertices;

            ElVis::OptiXBuffer<int> HexCoefficientBufferIndices;
            ElVis::OptiXBuffer<int> PrismCoefficientBufferIndices;

            ElVis::OptiXBuffer<ElVisFloat> HexCoefficientBuffer;
            ElVis::OptiXBuffer<ElVisFloat> PrismCoefficientBuffer;

            ElVis::OptiXBuffer<ElVisFloat4> HexPlaneBuffer;
            ElVis::OptiXBuffer<ElVisFloat4> PrismPlaneBuffer;

            ElVis::InteropBuffer<ElVisFloat4> FaceVertexBuffer;
            ElVis::InteropBuffer<ElVisFloat4> FaceNormalBuffer;
        };

        template<>
        ElVis::OptiXBuffer<int>& JacobiExtensionModel::GetCoefficientIndexBuffer<Hexahedron>();
        template<>
        ElVis::OptiXBuffer<int>& JacobiExtensionModel::GetCoefficientIndexBuffer<Prism>();

        template<>
        ElVis::OptiXBuffer<ElVisFloat>& JacobiExtensionModel::GetCoefficientBuffer<Hexahedron>();

        template<>
        ElVis::OptiXBuffer<ElVisFloat>& JacobiExtensionModel::GetCoefficientBuffer<Prism>();

        template<>
        ElVis::OptiXBuffer<ElVisFloat4>& JacobiExtensionModel::GetPlaneBuffer<Hexahedron>();
        template<>
        ElVis::OptiXBuffer<ElVisFloat4>& JacobiExtensionModel::GetPlaneBuffer<Prism>();
    }
}

#endif //ELVIS_HIGH_ORDER_ISOSURFACE_MODEL_H
