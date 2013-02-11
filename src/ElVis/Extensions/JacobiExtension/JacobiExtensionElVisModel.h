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
#include <ElVis/Core/Buffer.h>
#include <ElVis/Core/Float.h>

#include <optixu/optixpp.h>
#include <ElVis/Core/CudaGlobalBuffer.hpp>
#include <ElVis/Core/CudaGlobalVariable.hpp>
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
            virtual std::vector<optixu::GeometryGroup> DoGetCellGeometry(Scene* scene, optixu::Context context, CUmodule module);
            virtual void DoGetFaceGeometry(Scene* scene, optixu::Context context, CUmodule module, optixu::Geometry& faces );

            virtual int DoGetNumberOfBoundarySurfaces() const;
            virtual void DoGetBoundarySurface(int surfaceIndex, std::string& name, std::vector<int>& faceIds);

            virtual void DoCalculateExtents(WorldPoint& min, WorldPoint& max);
            virtual unsigned int DoGetNumberOfPoints() const;
            virtual WorldPoint DoGetPoint(unsigned int id) const;
            virtual void DoSetupCudaContext(CUmodule module) const;
            virtual const std::string& DoGetCUBinPrefix() const;
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
            ElVis::InteropBuffer<int>& GetCoefficientIndexBuffer() ;



            template<typename T> 
            ElVis::InteropBuffer<ElVisFloat>& GetCoefficientBuffer() ;

            template<typename T>
            ElVis::InteropBuffer<ElVisFloat4>& GetPlaneBuffer() ;



            template<typename T>
            void CreateCudaGeometryForElementType(boost::shared_ptr<FiniteElementVolume> volume, CUmodule module, const std::string& variablePrefix) const
            {
                unsigned int coefficientAlignment = 8;
                unsigned int numElements = volume->NumElementsOfType<T>();
                unsigned int numVertices = numElements*T::VertexCount;
                unsigned int numCoefficients = NumCoefficientsForElementType<T>(coefficientAlignment);

                if( numElements == 0 )
                {
                    return;
                }
                std::string vertexBufferName = variablePrefix + "VertexBuffer";
                CudaGlobalBuffer<ElVisFloat4> vertexBuffer(vertexBufferName, numVertices, module);
                ElVisFloat* vertexData = static_cast<ElVisFloat*>(vertexBuffer.map());

                //std::string coefficientsName = variablePrefix + "Coefficients";
                //CudaGlobalBuffer<ElVisFloat> coefficientBuffer(coefficientsName, numCoefficients, module);
                //ElVisFloat* coefficientData = static_cast<ElVisFloat*>(coefficientBuffer.map());

                std::string degreesName = variablePrefix + "Degrees";
                CudaGlobalBuffer<uint3> degreeBuffer(degreesName, numElements, module);
                unsigned int* degreeData = static_cast<unsigned int*>(degreeBuffer.map());

//                std::string planeBufferName = variablePrefix + "PlaneBuffer";
//                CudaGlobalBuffer<ElVisFloat4> planeBuffer(planeBufferName, 8*numElements, module);
//                ElVisFloat* planeData = static_cast<ElVisFloat*>(planeBuffer.map());

                std::string numElementsName = variablePrefix + "NumElements";
                CudaGlobalVariable<unsigned int> numElementsVariable((numElementsName), (module));
                numElementsVariable.WriteToDevice(numElements);

                unsigned int curElementId = 0;
                unsigned int curCoefficientIndex = 0;
                BOOST_FOREACH( boost::shared_ptr<Polyhedron> element, volume->IterateElementsOfType<T>() )
                {
                    boost::shared_ptr<T> castElement = boost::dynamic_pointer_cast<T>(element);
        
                    ///////////////////////////////////////////////////////////////////////
                    // Degrees
                    ///////////////////////////////////////////////////////////////////////
                    int degreeIdx = curElementId*3;
                    degreeData[degreeIdx] = element->degree(0);
                    degreeData[degreeIdx+1] = element->degree(1);
                    degreeData[degreeIdx+2] = element->degree(2);

                    ///////////////////////////////////////////////////////////////////////
                    // Vertices
                    ///////////////////////////////////////////////////////////////////////
                    for(unsigned int i = 0; i < T::VertexCount; ++i)
                    {
                        int vertexIdx = curElementId*T::VertexCount*4 + 4*i;
                        const ElVis::WorldPoint& p = element->vertex(i);
                        vertexData[vertexIdx] = static_cast<ElVisFloat>(p.x());
                        vertexData[vertexIdx+1] = static_cast<ElVisFloat>(p.y());
                        vertexData[vertexIdx+2] = static_cast<ElVisFloat>(p.z());
                        vertexData[vertexIdx+3] = static_cast<ElVisFloat>(1.0);
                    }

                    ///////////////////////////////////////////////////////////////////////
                    // Coefficients
                    ///////////////////////////////////////////////////////////////////////
                    //unsigned int numCoefficients = element->basisCoefficients().size();

                    //unsigned int tempIndex = curCoefficientIndex;
                    //for(unsigned int i = 0; i < numCoefficients; ++i)
                    //{
                    //    coefficientData[tempIndex] = static_cast<float>(element->basisCoefficients()[i]);
                    //    ++tempIndex;
                    //}
                    //unsigned int storageRequired = RequiredCoefficientStorage(numCoefficients, coefficientAlignment);
                    //curCoefficientIndex += storageRequired;

                    ///////////////////////////////////////////////////////////////////////
                    // Faces and planes
                    ///////////////////////////////////////////////////////////////////////
//                    int planeIdx = 8*curElementId*4;
//                    for(int i = 0; i < T::NumFaces; ++i)
//                    {
//                        ElVisFloat* base = planeData + planeIdx + i*4;
//                        castElement->GetFace(i, base[0], base[1], base[2], base[3]);
//                    }

                    ++curElementId;
                }

                //planeBuffer.unmap();
                degreeBuffer.unmap();
                //coefficientBuffer.unmap();
                vertexBuffer.unmap();

                std::string vertex_face_indexName = variablePrefix + "vertex_face_index";
                CudaGlobalBuffer<uint4> vertexFaceBuffer(vertex_face_indexName, T::NumFaces, module);
                unsigned int* vertexFaceBufferData = static_cast<unsigned int*>(vertexFaceBuffer.map()); 
                std::copy(T::VerticesForEachFace, T::VerticesForEachFace + 
                    4*T::NumFaces, vertexFaceBufferData);
                vertexFaceBuffer.unmap();

            }

            template<typename T>
            optixu::GeometryInstance CreateGeometryForElementType(boost::shared_ptr<FiniteElementVolume> volume, optixu::Context context, CUmodule module, const std::string& variablePrefix)
            {
                unsigned int coefficientAlignment = 8;
                int numElements = volume->NumElementsOfType<T>()*m_numberOfCopies;
                int numCoefficients = NumCoefficientsForElementType<T>(coefficientAlignment)*m_numberOfCopies;
                optixu::GeometryInstance instance;

                std::cout << "Number of elements = " << numElements << std::endl;
                std::cout << "Number of Coefficients = " << numCoefficients << std::endl;

                if( numElements == 0 )
                {
                    return optixu::GeometryInstance();
                }
                std::string vertexBufferName = variablePrefix + "VertexBuffer";
                FloatingPointBuffer VertexBuffer(vertexBufferName.c_str(), 4);
                VertexBuffer.Create(context, RT_BUFFER_INPUT, numElements*T::VertexCount);
                context[VertexBuffer.Name().c_str()]->set(*VertexBuffer);
                ElVisFloat* vertexData = static_cast<ElVisFloat*>(VertexBuffer->map());

                ElVis::InteropBuffer<int>& CoefficientIndicesBuffer = GetCoefficientIndexBuffer<T>();
                CoefficientIndicesBuffer.SetContextInfo(context, module);
                CoefficientIndicesBuffer.SetDimensions(numElements);
                int* coefficientIndicesData = static_cast<int*>(CoefficientIndicesBuffer.map());

                ElVis::InteropBuffer<ElVisFloat>& CoefficientBuffer = GetCoefficientBuffer<T>();
                CoefficientBuffer.SetContextInfo(context, module);
                CoefficientBuffer.SetDimensions(numCoefficients);
                ElVisFloat* coefficientData = static_cast<ElVisFloat*>(CoefficientBuffer.map());

                std::string degreesName = variablePrefix + "Degrees";
                optixu::Buffer DegreesBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, numElements);
                context[degreesName.c_str()]->set(DegreesBuffer);
                unsigned int* degreeData = static_cast<unsigned int*>(DegreesBuffer->map());


                std::string planeBufferName = variablePrefix + "PlaneBuffer";
                ElVis::InteropBuffer<ElVisFloat4>& PlaneBuffer = GetPlaneBuffer<T>();
                PlaneBuffer.SetContextInfo(context, module);
                PlaneBuffer.SetDimensions(8*numElements);
                ElVisFloat* planeData = static_cast<ElVisFloat*>(PlaneBuffer.map());

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
                            int vertexIdx = curElementId*T::VertexCount*4 + 4*i;
                            const ElVis::WorldPoint& p = element->vertex(i);
                            vertexData[vertexIdx] = (float)p.x();
                            vertexData[vertexIdx+1] = static_cast<ElVisFloat>(p.y() + copyId*range);
                            vertexData[vertexIdx+2] = static_cast<ElVisFloat>(p.z());
                            vertexData[vertexIdx+3] = 1.0;
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
                        int planeIdx = 8*curElementId*4;
                        for(int i = 0; i < T::NumFaces; ++i)
                        {
                            ElVisFloat* base = planeData + planeIdx + i*4;
                            castElement->GetFace(i, base[0], base[1], base[2], base[3]);

                            // Adjust as needed for copies.
                            base[3] = base[3] - static_cast<ElVisFloat>(2.0*copyId*base[1]);
                        }

                        ++curElementId;
                    }
                }

                PlaneBuffer.unmap();
                CoefficientBuffer.unmap();
                CoefficientIndicesBuffer.unmap();
                DegreesBuffer->unmap();
                VertexBuffer->unmap();

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

            ElVis::InteropBuffer<int> HexCoefficientBufferIndices;
            ElVis::InteropBuffer<int> PrismCoefficientBufferIndices;

            ElVis::InteropBuffer<ElVisFloat> HexCoefficientBuffer;
            ElVis::InteropBuffer<ElVisFloat> PrismCoefficientBuffer;

            ElVis::InteropBuffer<ElVisFloat4> HexPlaneBuffer;
            ElVis::InteropBuffer<ElVisFloat4> PrismPlaneBuffer;

            ElVis::InteropBuffer<ElVisFloat4> FaceVertexBuffer;
            ElVis::InteropBuffer<ElVisFloat4> FaceNormalBuffer;
        };

        template<>
        ElVis::InteropBuffer<int>& JacobiExtensionModel::GetCoefficientIndexBuffer<Hexahedron>();
        template<>
        ElVis::InteropBuffer<int>& JacobiExtensionModel::GetCoefficientIndexBuffer<Prism>();

        template<>
        ElVis::InteropBuffer<ElVisFloat>& JacobiExtensionModel::GetCoefficientBuffer<Hexahedron>();

        template<>
        ElVis::InteropBuffer<ElVisFloat>& JacobiExtensionModel::GetCoefficientBuffer<Prism>();

        template<>
        ElVis::InteropBuffer<ElVisFloat4>& JacobiExtensionModel::GetPlaneBuffer<Hexahedron>();
        template<>
        ElVis::InteropBuffer<ElVisFloat4>& JacobiExtensionModel::GetPlaneBuffer<Prism>();
    }
}

#endif //ELVIS_HIGH_ORDER_ISOSURFACE_MODEL_H
