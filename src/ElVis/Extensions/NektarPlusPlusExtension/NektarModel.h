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

#ifndef ELVIS_NEKTAR_MODEL_H
#define ELVIS_NEKTAR_MODEL_H


#include <SpatialDomains/MeshGraph3D.h>
#include <SpatialDomains/MeshGraph2D.h>

#include <MultiRegions/ExpList.h>
#include <MultiRegions/ExpList3D.h>

#include <LibUtilities/LinearAlgebra/NekVector.hpp>

#include <ElVis/Core/Model.h>
#include <ElVis/Core/Float.h>
#include <ElVis/Core/OptiXBuffer.hpp>
#include <ElVis/Core/OptiXBuffer.hpp>
#include <ElVis/Core/FaceInfo.h>
#include <ElVis/Core/Util.hpp>

#include <ElVis/Extensions/NektarPlusPlusExtension/Declspec.h>

#include <boost/filesystem/path.hpp>
#include <boost/utility.hpp>

namespace ElVis
{
    namespace NektarPlusPlusExtension
    {
        class NektarModel;

        /// \brief Implementation of the Nektar model, broken up into subclasses
        /// depending of if the mesh is 2D or 3D
        class NektarModelImpl : public boost::noncopyable
        {
            public:
                explicit NektarModelImpl(NektarModel* model);
                virtual ~NektarModelImpl();

                virtual Nektar::SpatialDomains::MeshGraphSharedPtr GetMesh() const = 0;
            private:
                NektarModel* m_model;
        };

        class ThreeDNektarModel : public NektarModelImpl
        {
            public:
                ThreeDNektarModel(NektarModel* model, Nektar::SpatialDomains::MeshGraph3DSharedPtr mesh);
                virtual ~ThreeDNektarModel();

                virtual Nektar::SpatialDomains::MeshGraphSharedPtr GetMesh() const
                {
                    return m_mesh; 
                }

            private:
                Nektar::SpatialDomains::MeshGraph3DSharedPtr m_mesh;
        };

        class TwoDNektarModel : public NektarModelImpl
        {
            public:
                TwoDNektarModel(NektarModel* model, Nektar::SpatialDomains::MeshGraph2DSharedPtr mesh);
                virtual ~TwoDNektarModel();

                virtual Nektar::SpatialDomains::MeshGraphSharedPtr GetMesh() const
                {
                    return m_mesh; 
                }

            private:
                Nektar::SpatialDomains::MeshGraph2DSharedPtr m_mesh;
        };

        class NektarModel : public Model
        {
            public:
                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT explicit NektarModel(const std::string& modelPrefix);
                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT virtual ~NektarModel();

                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT void Initialize(const boost::filesystem::path& geomFile,
                    const boost::filesystem::path& fieldFile);

                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT Nektar::SpatialDomains::MeshGraphSharedPtr GetMesh() const { return m_graph; }
                //NEKTAR_PLUS_PLUS_EXTENSION_EXPORT Nektar::MultiRegions::ExpListSharedPtr GetExpansion() const { return m_globalExpansion; }

                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT LibUtilities::SessionReaderSharedPtr GetSession() const;

            protected:
                virtual std::vector<optixu::GeometryGroup> DoGetPointLocationGeometry(boost::shared_ptr<Scene> scene, optixu::Context context);

                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT virtual void DoGetFaceGeometry(boost::shared_ptr<Scene> scene, optixu::Context context, optixu::Geometry& faces);
                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT virtual std::vector<optixu::GeometryInstance> DoGet2DPrimaryGeometry(boost::shared_ptr<Scene> scene, optixu::Context context);
                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT virtual optixu::Material DoGet2DPrimaryGeometryMaterial(SceneView* view);

                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT virtual int DoGetNumberOfBoundarySurfaces() const;
                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT virtual void DoGetBoundarySurface(int surfaceIndex, std::string& name, std::vector<int>& faceIds);

                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT void DoCalculateExtents(ElVis::WorldPoint &,ElVis::WorldPoint &);
                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT virtual const std::string& DoGetPTXPrefix() const;

                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT virtual unsigned int DoGetNumberOfElements() const;

                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT virtual int DoGetNumFields() const;

                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT virtual FieldInfo DoGetFieldInfo(unsigned int index) const;

                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT virtual int DoGetModelDimension() const;

                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT virtual void DoInitializeOptiX(boost::shared_ptr<Scene> scene, optixu::Context context) {}

                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT virtual size_t DoGetNumberOfFaces() const;

                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT virtual FaceInfo DoGetFaceDefinition(size_t globalFaceId) const;

                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT virtual size_t DoGetNumberOfPlanarFaceVertices() const;

                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT virtual WorldPoint DoGetPlanarFaceVertex(size_t vertexIdx) const;

                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT virtual size_t DoGetNumberOfVerticesForPlanarFace(size_t globalFaceId) const;

                NEKTAR_PLUS_PLUS_EXTENSION_EXPORT virtual size_t DoGetPlanarFaceVertexIndex(size_t globalFaceId, size_t vertexId);

            private:
                NektarModel(const NektarModel& rhs);

                static const std::string HexahedronIntersectionProgramName;
                static const std::string HexahedronPointLocationProgramName;
                static const std::string HexahedronBoundingProgramName;

                NektarModel& operator=(NektarModel& rhs);

                void SetupOptixCoefficientBuffers(optixu::Context context);
                void SetupOptixVertexBuffers(optixu::Context context);

                template<typename FaceContainer>
                void CreateLocalToGlobalIdxMap(const FaceContainer& faces, std::vector<int>& idxMap)
                {
                  typedef typename FaceContainer::const_iterator Iterator;
                  for(Iterator iter = faces.begin(); iter != faces.end(); ++iter)
                  {
                    idxMap.push_back((*iter).second->GetGlobalID());
                  }
                }

                template<typename FaceContainer>
                void AddFaces(const FaceContainer& faces, ElVisFloat3* minBuffer, ElVisFloat3* maxBuffer, ElVisFloat4* faceVertexBuffer, FaceInfo*, ElVisFloat4* normalBuffer)
                {
                    int faceIndex = 0;
                    typedef typename FaceContainer::const_iterator Iterator;
                    for(Iterator iter = faces.begin(); iter != faces.end(); ++iter)
                    {
                        
                        WorldPoint minExtent(std::numeric_limits<ElVisFloat>::max(), std::numeric_limits<ElVisFloat>::max(), std::numeric_limits<ElVisFloat>::max());
                        WorldPoint maxExtent(-std::numeric_limits<ElVisFloat>::max(), -std::numeric_limits<ElVisFloat>::max(), -std::numeric_limits<ElVisFloat>::max());

                        boost::shared_ptr<Nektar::SpatialDomains::Geometry2D> geom = boost::dynamic_pointer_cast<Nektar::SpatialDomains::Geometry2D>( (*iter).second );
                        
                        for(int i = 0; i < geom->GetNumVerts(); ++i)
                        {
                            Nektar::SpatialDomains::VertexComponentSharedPtr rawVertex = geom->GetVertex(i);
                            WorldPoint v(rawVertex->x(), rawVertex->y(), rawVertex->z());
                            minExtent = CalcMin(minExtent, v);
                            maxExtent = CalcMax(maxExtent, v);

                            faceVertexBuffer[4*faceIndex+i] = MakeFloat4(v);

                            if( geom->GetNumVerts() == 3 && i == 2 )
                            {
                                faceVertexBuffer[4*faceIndex+i+1] = MakeFloat4(v);
                            }
                        }

                        // There is no proof that OptiX can't handle degenerate boxes,
                        // but just in case...
                        if( minExtent.x() == maxExtent.x() )
                        {
                            minExtent.SetX(minExtent.x() - .0001);
                            maxExtent.SetX(maxExtent.x() + .0001);
                        }

                        if( minExtent.y() == maxExtent.y() )
                        {
                            minExtent.SetY(minExtent.y() - .0001);
                            maxExtent.SetY(maxExtent.y() + .0001);
                        }

                        if( minExtent.z() == maxExtent.z() )
                        {
                            minExtent.SetZ(minExtent.z() - .0001);
                            maxExtent.SetZ(maxExtent.z() + .0001);
                        }

                        Nektar::SpatialDomains::MeshGraph3DSharedPtr castPtr = boost::dynamic_pointer_cast<Nektar::SpatialDomains::MeshGraph3D>(m_graph);
                        if( castPtr )
                        {
                            Nektar::SpatialDomains::ElementFaceVectorSharedPtr elements = castPtr->GetElementsFromFace(geom);
                            assert(elements->size() <= 2 );
                            //faceDefs[faceIndex].CommonElements[0].Id = -1;
                            //faceDefs[faceIndex].CommonElements[0].Type = -1;
                            //faceDefs[faceIndex].CommonElements[1].Id = -1;
                            //faceDefs[faceIndex].CommonElements[1].Type = -1;
                            //for(int elementId = 0; elementId < elements->size(); ++elementId)
                            //{
                            //    faceDefs[faceIndex].CommonElements[elementId].Id = (*elements)[elementId]->m_Element->GetGlobalID();
                            //    faceDefs[faceIndex].CommonElements[elementId].Type = (*elements)[elementId]->m_Element->GetGeomShapeType();
                            //}
                        }
                        
                        // TODO - Normal.
                        Nektar::NekVector<double> normal;
                        if( geom->GetNumVerts() == 3 )
                        {
                            Nektar::NekVector<double> v0 = Nektar::createVectorFromPoints(*geom->GetVertex(0), *geom->GetVertex(1));
                            Nektar::NekVector<double> v1 = Nektar::createVectorFromPoints(*geom->GetVertex(0), *geom->GetVertex(2));
                            normal = Cross(v0, v1);
                        }
                        else if( geom->GetNumVerts() == 4 )
                        {
                            Nektar::NekVector<double> v0 = Nektar::createVectorFromPoints(*geom->GetVertex(0), *geom->GetVertex(1));
                            Nektar::NekVector<double> v1 = Nektar::createVectorFromPoints(*geom->GetVertex(0), *geom->GetVertex(3));
                            normal = Cross(v0, v1);
                        }
                        normalBuffer[faceIndex].x = normal.x();
                        normalBuffer[faceIndex].y = normal.y();
                        normalBuffer[faceIndex].z = normal.z();
                        normalBuffer[faceIndex].w = 0.0;
                        ++faceIndex;

                    }
                }
                
                template<typename T>
                optixu::GeometryInstance CreateGeometryForElementType(optixu::Context context, const std::string& variablePrefix)
                {
                    unsigned int numElements = m_graph->GetAllElementsOfType<T>().size();

                    optixu::Geometry geometry = context->createGeometry();
                    geometry->setPrimitiveCount(numElements);

                    optixu::GeometryInstance instance = context->createGeometryInstance();
                    instance->setGeometry(geometry);

                    m_deviceHexPlaneBuffer.SetContext(context);
                    m_deviceHexPlaneBuffer.SetDimensions(numElements*8);
                    //FloatingPointBuffer hexPlaneBuffer("HexPlaneBuffer", 4);
                    //hexPlaneBuffer.Create(context, RT_BUFFER_INPUT, numElements*8);
                    //context[hexPlaneBuffer.Name().c_str()]->set(*hexPlaneBuffer);
                    BOOST_AUTO(hexPlaneData, m_deviceHexPlaneBuffer.Map());
                    
                    m_deviceHexVertexIndices.SetContext(context);
                    m_deviceHexVertexIndices.SetDimensions(numElements*8);
                    //optixu::Buffer vertexIndexBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, numElements*8);
                    //context["HexVertexIndices"]->set(vertexIndexBuffer);
                    BOOST_AUTO(coefficientIndicesData, m_deviceHexVertexIndices.Map());

                    typedef typename std::map<int, boost::shared_ptr<T> >::const_iterator IterType;
                    unsigned int i = 0;
                    for(IterType iter = m_graph->GetAllElementsOfType<T>().begin(); iter != m_graph->GetAllElementsOfType<T>().end(); ++iter)
                    {
                        // TODO - Check the correspondence between vertex id and global id.
                        boost::shared_ptr<T> hex = (*iter).second;
                        for(unsigned int j = 0; j < 8; ++j)
                        {
                            unsigned int vid = hex->GetVid(j);
                            coefficientIndicesData[i*8 + j] = vid;
                        }

                        BOOST_AUTO(v0, m_graph->GetVertex(hex->GetVid(0)));
                        BOOST_AUTO(v1, m_graph->GetVertex(hex->GetVid(1)));
                        BOOST_AUTO(v2, m_graph->GetVertex(hex->GetVid(2)));
                        BOOST_AUTO(v3, m_graph->GetVertex(hex->GetVid(3)));
                        BOOST_AUTO(v4, m_graph->GetVertex(hex->GetVid(4)));
                        BOOST_AUTO(v5, m_graph->GetVertex(hex->GetVid(5)));
                        BOOST_AUTO(v6, m_graph->GetVertex(hex->GetVid(6)));
                        BOOST_AUTO(v7, m_graph->GetVertex(hex->GetVid(7)));

                        Nektar::NekVector<double> faceNormals[6];

                        Nektar::NekVector<double> d1 = Nektar::createVectorFromPoints(*v3, *v0);
                        Nektar::NekVector<double> d2 = Nektar::createVectorFromPoints(*v1, *v0);
                        faceNormals[0] = d1.Cross(d2);

                        d1 = Nektar::createVectorFromPoints(*v5, *v4);
                        d2 = Nektar::createVectorFromPoints(*v7, *v4);
                        faceNormals[1] = d1.Cross(d2);

                        d1 = Nektar::createVectorFromPoints(*v6, *v7);
                        d2 = Nektar::createVectorFromPoints(*v3, *v7);
                        faceNormals[2] = d1.Cross(d2);

                        d1 = Nektar::createVectorFromPoints(*v7, *v4);
                        d2 = Nektar::createVectorFromPoints(*v0, *v4);
                        faceNormals[3] = d1.Cross(d2);

                        d1 = Nektar::createVectorFromPoints(*v0, *v4);
                        d2 = Nektar::createVectorFromPoints(*v5, *v4);
                        faceNormals[4] = d1.Cross(d2);

                        d1 = Nektar::createVectorFromPoints(*v1, *v5);
                        d2 = Nektar::createVectorFromPoints(*v6, *v5);
                        faceNormals[5] = d1.Cross(d2);

                        for(unsigned int c = 0; c < 6; ++c)
                        {
                            faceNormals[c].Normalize();
                        }

                        ElVisFloat D[6];
                        D[0] = -(faceNormals[0].x()*(*v0).x() + faceNormals[0].y()*(*v0).y() +
                            faceNormals[0].z()*(*v0).z());

                        D[1] = -(faceNormals[1].x()*(*v4).x() + faceNormals[1].y()*(*v4).y() +
                            faceNormals[1].z()*(*v4).z());


                        D[2] = -(faceNormals[2].x()*(*v3).x() + faceNormals[2].y()*(*v3).y() +
                            faceNormals[2].z()*(*v3).z());

                        D[3] = -(faceNormals[3].x()*(*v4).x() + faceNormals[3].y()*(*v4).y() +
                            faceNormals[3].z()*(*v4).z());

                        D[4] = -(faceNormals[4].x()*(*v4).x() + faceNormals[4].y()*(*v4).y() +
                            faceNormals[4].z()*(*v4).z());

                        D[5] = -(faceNormals[5].x()*(*v5).x() + faceNormals[5].y()*(*v5).y() +
                            faceNormals[5].z()*(*v5).z());

                        for(unsigned int c = 0; c < 6; ++c)
                        {
                            int indexBase = i*8 + c;
                            hexPlaneData[indexBase].x = faceNormals[c].x();
                            hexPlaneData[indexBase].y = faceNormals[c].y();
                            hexPlaneData[indexBase].z = faceNormals[c].z();
                            hexPlaneData[indexBase].x = D[c];
                            //std::cout << "(" << hexPlaneData[indexBase] << ", " << hexPlaneData[indexBase+1] << ", " << hexPlaneData[indexBase+2] << ", " << hexPlaneData[indexBase+3] << ")" << std::endl;
                        }
                        ++i;
                    }

                    m_deviceNumberOfModes.SetContext(context);
                    m_deviceNumberOfModes.SetDimensions(numElements*3);
                    //optixu::Buffer numberOfModesBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, numElements*3);
                    //context["NumberOfModes"]->set(numberOfModesBuffer);
                    BOOST_AUTO(modesData, m_deviceNumberOfModes.Map());

                    i = 0;
                    for(IterType iter = m_graph->GetAllElementsOfType<T>().begin(); iter != m_graph->GetAllElementsOfType<T>().end(); ++iter)
                    {
                        // TODO - Check the correspondence between vertex id and global id.
                        boost::shared_ptr<T> hex = (*iter).second; 
                        BOOST_AUTO(localExpansion, m_globalExpansions[0]->GetExp(hex->GetGlobalID()));

                        modesData[i].x = localExpansion->GetBasis(0)->GetNumModes();
                        modesData[i].y = localExpansion->GetBasis(1)->GetNumModes();
                        modesData[i].z = localExpansion->GetBasis(2)->GetNumModes();
                        ++i;
                    }

                    const unsigned int VerticesForEachFace[] = 
                    {0, 1, 2, 3, 
                    4, 5, 6, 7,
                    3, 2, 6, 7,
                    0, 4, 7, 3,
                    0, 1, 5, 4, 
                    1, 5, 6, 2 };

                    m_deviceHexVertexFaceIndex.SetContext(context);
                    m_deviceHexVertexFaceIndex.SetDimensions(6);

                    //std::string vertex_face_indexName = variablePrefix + "vertex_face_index";
                    //optixu::Buffer vertexFaceBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT4, 6);
                    //instance[vertex_face_indexName.c_str()]->set(vertexFaceBuffer);
                    BOOST_AUTO(vertexFaceBufferData, m_deviceHexVertexFaceIndex.Map());
                    std::copy(VerticesForEachFace, VerticesForEachFace + 4*6, (uint*)vertexFaceBufferData.get());

                    return instance;
                }

                // Initialization methods.
                void LoadFields(const boost::filesystem::path& fieldFile);
                void SetupFaces();

                void SetupSumPrefixNumberOfFieldCoefficients(optixu::Context context);
                void SetupCoefficientOffsetBuffer(optixu::Context context);
                void SetupFieldModes(optixu::Context context);
                void SetupFieldBases(optixu::Context context);

                // Data Members
                boost::shared_ptr<NektarModelImpl> m_impl;

                Nektar::SpatialDomains::MeshGraphSharedPtr m_graph;
                std::vector<Nektar::MultiRegions::ExpListSharedPtr> m_globalExpansions;
                std::vector<Nektar::SpatialDomains::FieldDefinitionsSharedPtr> m_fieldDefinitions;
                LibUtilities::SessionReaderSharedPtr m_session;

                optixu::Program m_hexGeometryIntersectionProgram;
                optixu::Program m_hexPointLocationProgram;
                optixu::Program m_2DTriangleIntersectionProgram;
                optixu::Program m_2DTriangleBoundingBoxProgram;
                optixu::Program m_2DQuadIntersectionProgram;
                optixu::Program m_2DQuadBoundingBoxProgram;
                optixu::Program m_2DElementClosestHitProgram;
                optixu::Program m_TwoDClosestHitProgram;

                ElVis::OptiXBuffer<Nektar::LibUtilities::BasisType> m_FieldBases;
                ElVis::OptiXBuffer<uint3> m_FieldModes;
                ElVis::OptiXBuffer<uint> m_SumPrefixNumberOfFieldCoefficients;
                ElVis::OptiXBuffer<ElVisFloat4> m_deviceVertexBuffer;
                ElVis::OptiXBuffer<ElVisFloat> m_deviceCoefficientBuffer;
                ElVis::OptiXBuffer<uint> m_deviceCoefficientOffsetBuffer;
                
                ElVis::OptiXBuffer<uint> m_deviceHexVertexIndices;
                ElVis::OptiXBuffer<ElVisFloat4> m_deviceHexPlaneBuffer;
                ElVis::OptiXBuffer<uint4> m_deviceHexVertexFaceIndex;
                ElVis::OptiXBuffer<uint3> m_deviceNumberOfModes;

                ElVis::OptiXBuffer<ElVisFloat4> PlanarFaceVertexBuffer;
                ElVis::OptiXBuffer<ElVisFloat4> FaceNormalBuffer;

                ElVis::OptiXBuffer<uint> m_deviceTriangleVertexIndexMap;
                ElVis::OptiXBuffer<uint2> m_TriangleModes;
                ElVis::OptiXBuffer<ElVisFloat> m_TriangleMappingCoeffsDir0;
                ElVis::OptiXBuffer<ElVisFloat> m_TriangleMappingCoeffsDir1;
                ElVis::OptiXBuffer<uint> m_TriangleCoeffMappingDir0;
                ElVis::OptiXBuffer<uint> m_TriangleCoeffMappingDir1;
                ElVis::OptiXBuffer<uint> m_TriangleGlobalIdMap;

                ElVis::OptiXBuffer<uint> m_deviceQuadVertexIndexMap;
                ElVis::OptiXBuffer<uint2> m_QuadModes;
                ElVis::OptiXBuffer<ElVisFloat> m_QuadMappingCoeffsDir0;
                ElVis::OptiXBuffer<ElVisFloat> m_QuadMappingCoeffsDir1;
                ElVis::OptiXBuffer<uint> m_QuadCoeffMappingDir0;
                ElVis::OptiXBuffer<uint> m_QuadCoeffMappingDir1;

                std::vector<int> m_triLocalToGlobalIdxMap;
                std::vector<int> m_quadLocalToGlobalIdxMap;
        };
    }

}


#endif //ELVIS_NEKTAR_MODEL_H
