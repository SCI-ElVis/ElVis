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
#include <ElVis/Core/Point.hpp>
#include <ElVis/Core/Util.hpp>

#include <optixu/optixpp.h>
#include <ElVis/Core/ElementId.h>
#include <ElVis/Core/FaceInfo.h>

#include <ElVis/Extensions/JacobiExtension/Hexahedron.h>
#include <ElVis/Extensions/JacobiExtension/Tetrahedron.h>
#include <ElVis/Extensions/JacobiExtension/Prism.h>
#include <ElVis/Extensions/JacobiExtension/JacobiFace.h>

#include <map>
#include <set>
#include <boost/typeof/typeof.hpp>

namespace ElVis
{
    namespace JacobiExtension
    {
        class JacobiExtensionModel : public ElVis::Model
        {
        public:
            JACOBI_EXTENSION_EXPORT JacobiExtensionModel(const std::string& modelPath);
            JACOBI_EXTENSION_EXPORT virtual ~JacobiExtensionModel();

            JACOBI_EXTENSION_EXPORT void LoadVolume(const std::string& filePath);
            JACOBI_EXTENSION_EXPORT void writeCellVolumeForVTK(const char* fileName);
            JACOBI_EXTENSION_EXPORT boost::shared_ptr<FiniteElementVolume> Volume() const { return m_volume; }

        protected:
            virtual std::vector<optixu::GeometryInstance> DoGet2DPrimaryGeometry(boost::shared_ptr<Scene> scene, optixu::Context context);
            virtual optixu::Material DoGet2DPrimaryGeometryMaterial(SceneView* view);

            virtual int DoGetNumberOfBoundarySurfaces() const;
            virtual void DoGetBoundarySurface(int surfaceIndex, std::string& name, std::vector<int>& faceIds);

            virtual void DoCalculateExtents(WorldPoint& min, WorldPoint& max);
            virtual const std::string& DoGetPTXPrefix() const;
            virtual unsigned int DoGetNumberOfElements() const;

            virtual int DoGetNumFields() const;
            virtual FieldInfo DoGetFieldInfo(unsigned int index) const;
            virtual void DoCopyExtensionSpecificDataToOptiX(optixu::Context context);

        private:
            JacobiExtensionModel(const JacobiExtensionModel& rhs);
            JacobiExtensionModel& operator=(const JacobiExtensionModel& rhs);

            virtual int DoGetModelDimension() const { return 3; }

            virtual size_t DoGetNumberOfFaces() const;

            virtual FaceInfo DoGetFaceDefinition(size_t globalFaceId) const;

            virtual size_t DoGetNumberOfPlanarFaceVertices() const;

            virtual WorldPoint DoGetPlanarFaceVertex(size_t vertexIdx) const;

            virtual size_t DoGetNumberOfVerticesForPlanarFace(size_t localFaceIdx) const;

            virtual size_t DoGetPlanarFaceVertexIndex(size_t localFaceIdx, size_t vertexId);

            virtual WorldVector DoGetPlanarFaceNormal(size_t localFaceIdx) const;

            template<typename T>
            int NumCoefficientsForElementType(unsigned int alignment) const 
            {
                int result = 0;
                if( m_numberOfModes > 0 )
                {
                    BOOST_FOREACH(boost::shared_ptr<Polyhedron> iter, m_volume->IterateElementsOfType<T>() )
                    {
                        result += iter->NumberOfCoefficientsForOrder(m_numberOfModes-1);
                    }
                }
                else
                {
                    BOOST_FOREACH(boost::shared_ptr<Polyhedron> iter, m_volume->IterateElementsOfType<T>() )
                    {
                        result += iter->basisCoefficients().size();
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
            void CopyFieldsForElementType(boost::shared_ptr<FiniteElementVolume> volume, optixu::Context context, const std::string& variablePrefix)
            {
                unsigned int coefficientAlignment = 8;
                int numElements = volume->NumElementsOfType<T>()*m_numberOfCopies;
                int numCoefficients = NumCoefficientsForElementType<T>(coefficientAlignment)*m_numberOfCopies;

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
                        unsigned int storageRequired = numCoefficients;
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


                // Setup the per element type vertex/face mappings.
                std::string vertex_face_indexName = variablePrefix + "vertex_face_index";
                optixu::Buffer vertexFaceBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT4, T::NumFaces);
                unsigned int* vertexFaceBufferData = static_cast<unsigned int*>(vertexFaceBuffer->map()); 
                std::copy(T::VerticesForEachFace, T::VerticesForEachFace + 
                    4*T::NumFaces, vertexFaceBufferData);
                vertexFaceBuffer->unmap();
            }


            template<typename T>
            void PopulateFaces(boost::shared_ptr<FiniteElementVolume> volume, std::map<JacobiFaceKey, JacobiFace>& faceMap)
            {
                int id = 0;
                BOOST_FOREACH( boost::shared_ptr<Polyhedron> element, volume->IterateElementsOfType<T>() )
                {
                    for(unsigned int i = 0; i < T::NumFaces; ++i)
                    {
                        WorldPoint p[] = 
                        {
                          element->vertex(T::VerticesForEachFace[i*4]),
                          element->vertex(T::VerticesForEachFace[i*4+1]),
                          element->vertex(T::VerticesForEachFace[i*4+2]),
                          element->vertex(T::VerticesForEachFace[i*4+3])
                        };
                        
                        boost::shared_ptr<T> asT = boost::dynamic_pointer_cast<T>(element);

                        // Since the jacobi extension provides normals as inward facing normals, we need to invert
                        // them so they point out for the types of intersection tests we will be doing.
                        ElVis::WorldVector normal = -(asT->GetFaceNormal(i));
                        JacobiFaceKey key(p[0], p[1], p[2], p[3]);

                        BOOST_AUTO(found, faceMap.find(key));
                        ElementId curElement(id, T::TypeId);

                        if( found != faceMap.end() )
                        {
                            JacobiFace& face = (*found).second;
                            face.info.CommonElements[0] = curElement;
                        }
                        else
                        {
                            JacobiFace face(normal);
                            face.info.CommonElements[1] = curElement;
                            face.info.Type = ePlanar;
                            ElementId nextElement(-1, -1);
                            face.info.CommonElements[0] = nextElement;

                            face.info.MinExtent = MakeFloat3(key.MinExtent());
                            face.info.MaxExtent = MakeFloat3(key.MaxExtent());

                            if( T::NumEdgesForEachFace[i] == 4 )
                            {
                              face.planarInfo.Type = eQuad;
                            }
                            else
                            {
                              face.planarInfo.Type = eTriangle;
                            }

                            BOOST_AUTO(insertValue, std::make_pair(key, face));
                            faceMap.insert(insertValue);
                        }
                    }
                    ++id;
                }
            }

            boost::shared_ptr<FiniteElementVolume> m_volume;
            unsigned m_numberOfCopies;
            int m_numberOfModes;

            std::vector<WorldPoint> m_vertices;
            std::vector<JacobiFace> m_faces;

            ElVis::OptiXBuffer<int> HexCoefficientBufferIndices;
            ElVis::OptiXBuffer<int> PrismCoefficientBufferIndices;

            ElVis::OptiXBuffer<ElVisFloat> HexCoefficientBuffer;
            ElVis::OptiXBuffer<ElVisFloat> PrismCoefficientBuffer;

            ElVis::OptiXBuffer<ElVisFloat4> HexPlaneBuffer;
            ElVis::OptiXBuffer<ElVisFloat4> PrismPlaneBuffer;
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
