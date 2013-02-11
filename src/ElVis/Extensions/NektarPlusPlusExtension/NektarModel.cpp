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


#include "NektarModel.h"
#include <SpatialDomains/MeshGraph1D.h>
#include <SpatialDomains/MeshGraph2D.h>
#include <SpatialDomains/MeshGraph3D.h>
#include <MultiRegions/ExpList3D.h>
#include <ElVis/Core/Util.hpp>
#include <ElVis/Core/PtxManager.h>
#include <ElVis/Core/Buffer.h>
#include <ElVis/Core/Float.h>
#include <ElVis/Core/Point.hpp>
#include <ElVis/Core/Scene.h>

#ifdef Max
#undef Max
#endif

namespace ElVis
{
    namespace NektarPlusPlusExtension
    {

        const std::string NektarModel::HexahedronIntersectionProgramName("HexahedronIntersection");
        const std::string NektarModel::HexahedronPointLocationProgramName("HexahedronContainsOriginByCheckingPoint");
        const std::string NektarModel::HexahedronBoundingProgramName("HexahedronBounding");

        NektarModel::NektarModel(const std::string& modelPrefix) :
            m_graph(),
            m_session(),
            m_deviceVertexBuffer("Vertices"),
            m_deviceCoefficientBuffer("Coefficients"),
            m_deviceCoefficientIndexBuffer("CoefficientIndices"),
            m_deviceHexVertexIndices("HexVertexIndices"),
            m_deviceHexPlaneBuffer("HexPlaneBuffer"),
            m_deviceHexVertexFaceIndex("Hexvertex_face_index"),
            m_deviceNumberOfModes("NumberOfModes"),
            FaceVertexBuffer("FaceVertexBuffer"),
            FaceNormalBuffer("FaceNormalBuffer")
        {
            std::string geometryFile = modelPrefix + ".xml";
            std::string fieldFile = modelPrefix + ".fld";
            InitializeWithGeometryAndField(geometryFile, fieldFile);
        }

        NektarModel::~NektarModel()
        {
        }

        unsigned int NektarModel::DoGetNumberOfPoints() const
        {
            return m_graph->GetNvertices();
        }

        void NektarModel::DoCalculateExtents(ElVis::WorldPoint& minResult, ElVis::WorldPoint& maxResult)
        {
            for(unsigned int i = 0; i < m_graph->GetNvertices(); ++i)
            {
                BOOST_AUTO(vertex, m_graph->GetVertex(i));
                
                ElVis::WorldPoint p(vertex->x(), vertex->y(), vertex->z());
                minResult = CalcMin(minResult, p);
                maxResult = CalcMax(maxResult, p);
            }
        }

        WorldPoint NektarModel::DoGetPoint(unsigned int id) const
        {
            BOOST_AUTO(vertex, m_graph->GetVertex(id));
            ElVis::WorldPoint result(vertex->x(), vertex->y(), vertex->z());
            return result;
        }

        const std::string& NektarModel::DoGetPTXPrefix() const
        {
            static std::string prefix("NektarPlusPlusExtension");
            return prefix;
        }



        const std::string& NektarModel::DoGetCUBinPrefix() const
        {
            static std::string prefix("NektarPlusPlusExtension");
            return prefix;
        }

        void NektarModel::InitializeWithGeometryAndField(const std::string& geomFileName,
            const std::string& fieldFileName)
        {
            int argc = 3;
            char* arg1 = "ElVis";
            char* arg2 = strdup(geomFileName.c_str());
            char* arg3 = strdup(fieldFileName.c_str());
            char* argv[] = {arg1, arg2, arg3};
            m_session = LibUtilities::SessionReader::CreateInstance(argc, argv);
            free(arg2);
            free(arg3);

            arg2 = 0;
            arg3 = 0;

            m_graph = boost::dynamic_pointer_cast<Nektar::SpatialDomains::MeshGraph3D>(SpatialDomains::MeshGraph::Read(m_session->GetFilename(), false));
            CalculateExtents();

            std::vector<Nektar::SpatialDomains::FieldDefinitionsSharedPtr> fieldDefinitions;
            std::vector<std::vector<double> > fieldData;

            m_graph->Import(fieldFileName, fieldDefinitions, fieldData);

            //----------------------------------------------
            // Set up Expansion information
            for(int i = 0; i < fieldDefinitions.size(); ++i)
            {
                vector<LibUtilities::PointsType> ptype;
                for(int j = 0; j < 3; ++j)
                {
                    ptype.push_back(LibUtilities::ePolyEvenlySpaced);
                }
                
                fieldDefinitions[i]->m_pointsDef = true;
                fieldDefinitions[i]->m_points    = ptype; 
                
                vector<unsigned int> porder;
                if(fieldDefinitions[i]->m_numPointsDef == false)
                {
                    for(int j = 0; j < fieldDefinitions[i]->m_numModes.size(); ++j)
                    {
                        porder.push_back(fieldDefinitions[i]->m_numModes[j]);
                    }
                    
                    fieldDefinitions[i]->m_numPointsDef = true;
                }
                else
                {
                    for(int j = 0; j < fieldDefinitions[i]->m_numPoints.size(); ++j)
                    {
                        porder.push_back(fieldDefinitions[i]->m_numPoints[j]);
                    }
                }
                fieldDefinitions[i]->m_numPoints = porder;
                
            }

            // It probably is possible to figure out which coefficients belong to which element
            // at this point, but it will be easier to setup the expansions in the mesh graph
            // and then have Nektar++ copy them over.  The SetExpansions call appears to be
            // incomplete, so we'll just do it ourselves.
            m_graph->SetExpansions(fieldDefinitions);

            m_globalExpansion = MemoryManager<Nektar::MultiRegions::ExpList3D>
                ::AllocateSharedPtr(m_session, m_graph);
            for(unsigned int i = 0; i < fieldDefinitions.size(); ++i)
            {
                m_globalExpansion->ExtractDataToCoeffs(fieldDefinitions[i], fieldData[i], fieldDefinitions[i]->m_fields[0]);
            }

            m_globalExpansion->BwdTrans(m_globalExpansion->GetCoeffs(), m_globalExpansion->UpdatePhys());

            m_globalExpansion->PutCoeffsInToElmtExp();
            m_globalExpansion->PutPhysInToElmtExp();
        }
                
        void NektarModel::SetupOptixCoefficientBuffers(optixu::Context context, CUmodule module)
        {
            // Coefficients are stored in a global array for all element types.
            m_deviceCoefficientBuffer.SetContextInfo(context, module);
            m_deviceCoefficientBuffer.SetDimensions(m_globalExpansion->GetNcoeffs());

            ElVisFloat* coeffData = static_cast<ElVisFloat*>(m_deviceCoefficientBuffer.map());
            for(unsigned int i = 0; i < m_globalExpansion->GetNcoeffs(); ++i)
            {
                coeffData[i] = m_globalExpansion->GetCoeff(i);
            }
            m_deviceCoefficientBuffer.unmap();

            m_deviceCoefficientIndexBuffer.SetContextInfo(context, module);
            m_deviceCoefficientIndexBuffer.SetDimensions(m_globalExpansion->GetNumElmts());
            int* coefficientIndicesData = static_cast<int*>(m_deviceCoefficientIndexBuffer.map());

            for(unsigned int i = 0; i < m_globalExpansion->GetNumElmts(); ++i)
            {
                coefficientIndicesData[i] = m_globalExpansion->GetCoeff_Offset(i);
            }
            m_deviceCoefficientIndexBuffer.unmap();

        }

        void NektarModel::SetupOptixVertexBuffers(optixu::Context context, CUmodule module)
        {
            m_deviceVertexBuffer.SetContextInfo(context, module);
            m_deviceVertexBuffer.SetDimensions(m_graph->GetNvertices());

            ElVisFloat* vertexData = static_cast<ElVisFloat*>(m_deviceVertexBuffer.map());
            for(unsigned int i = 0; i < m_graph->GetNvertices(); ++i)
            {
                BOOST_AUTO(vertex, m_graph->GetVertex(i));
                //std::cout << "(" << vertex->x() << ", " << vertex->y() << ", " << vertex->z() << ")" << std::endl;
                vertexData[i*4] = vertex->x();
                vertexData[i*4 + 1] = vertex->y();
                vertexData[i*4 + 2] = vertex->z();
                vertexData[i*4 + 3] = 1.0;
            }
            m_deviceVertexBuffer.unmap();
        }

        void NektarModel::DoSetupCudaContext(CUmodule module) const
        {
            //CreateCudaGeometryForElementType<Hexahedron>(m_volume, module, "Hex");
        }

        // Geometry for volume rendering, need ray/element intersections.
        std::vector<optixu::GeometryGroup> NektarModel::DoGetCellGeometry(Scene* scene, optixu::Context context, CUmodule module)
        {
           
            try
            {
                std::vector<optixu::GeometryGroup> result;
                if( !m_graph ) return result;

                SetupOptixCoefficientBuffers(context, module);
                SetupOptixVertexBuffers(context, module);

                optixu::Material m_hexCutSurfaceMaterial = context->createMaterial();
 
                optixu::Program hexBoundingProgram = PtxManager::LoadProgram(context, GetPTXPrefix(), "NektarHexahedronBounding");
                optixu::Program hexIntersectionProgram = PtxManager::LoadProgram(context, GetPTXPrefix(), HexahedronIntersectionProgramName);

                optixu::GeometryInstance hexInstance = CreateGeometryForElementType<Nektar::SpatialDomains::HexGeom>(context, module, "Hex");
                hexInstance->setMaterialCount(1);
                hexInstance->setMaterial(0, m_hexCutSurfaceMaterial);

                optixu::Geometry hexGeometry = hexInstance->getGeometry();
                hexGeometry->setBoundingBoxProgram( hexBoundingProgram );
                hexGeometry->setIntersectionProgram( hexIntersectionProgram );

                optixu::GeometryGroup group = context->createGeometryGroup();
                group->setChildCount(1);
                group->setChild(0, hexInstance);

                group->setAcceleration( context->createAcceleration("Sbvh","Bvh") );
                result.push_back(group);

                return result;
            }
            catch(optixu::Exception& e)
            {
                std::cerr << e.getErrorString() << std::endl;
                throw;
            }
            catch(std::exception& f)
            {
                std::cerr << f.what() << std::endl;
                throw;
            }
        }

        void NektarModel::DoGetFaceGeometry(Scene* scene, optixu::Context context, CUmodule module, optixu::Geometry& faceGeometry)
        {
            int numFaces = 0;
            numFaces += m_graph->GetAllTriGeoms().size();
            numFaces += m_graph->GetAllQuadGeoms().size();


            scene->GetFaceMinExtentBuffer()->setSize(numFaces);
            scene->GetFaceMaxExtentBuffer()->setSize(numFaces);

            ElVisFloat3* minBuffer = static_cast<ElVisFloat3*>(scene->GetFaceMinExtentBuffer()->map());
            ElVisFloat3* maxBuffer = static_cast<ElVisFloat3*>(scene->GetFaceMaxExtentBuffer()->map());
            FaceVertexBuffer.SetContextInfo(context, module);
            FaceVertexBuffer.SetDimensions(numFaces*4);
            FaceNormalBuffer.SetContextInfo(context, module);
            FaceNormalBuffer.SetDimensions(numFaces);

            scene->GetFaceIdBuffer()->setSize(numFaces);
            FaceDef* faceDefs = static_cast<FaceDef*>(scene->GetFaceIdBuffer()->map());
            ElVisFloat4* faceVertexBuffer = static_cast<ElVisFloat4*>(FaceVertexBuffer.map());
            ElVisFloat4* normalBuffer = static_cast<ElVisFloat4*>(FaceNormalBuffer.map());

            AddFaces(m_graph->GetAllTriGeoms(), minBuffer, maxBuffer, faceVertexBuffer, faceDefs, normalBuffer);

            int offset = m_graph->GetAllTriGeoms().size();
            AddFaces(m_graph->GetAllQuadGeoms(), minBuffer+offset, maxBuffer+offset, faceVertexBuffer+offset, faceDefs+offset, normalBuffer+offset);


            scene->GetFaceMinExtentBuffer()->unmap();
            scene->GetFaceMaxExtentBuffer()->unmap();
            scene->GetFaceIdBuffer()->unmap();
            FaceVertexBuffer.unmap();
            FaceNormalBuffer.unmap();

            faceGeometry->setPrimitiveCount(numFaces);
            //curvedFaces->setPrimitiveCount(faces.size());
        }

        int NektarModel::DoGetNumberOfBoundarySurfaces() const
        {
            return 0;
        }

        void NektarModel::DoGetBoundarySurface(int surfaceIndex, std::string& name, std::vector<int>& faceIds)
        {
        }

        void NektarModel::DoMapInteropBufferForCuda()
        {
            m_deviceVertexBuffer.GetMappedCudaPtr();
            m_deviceCoefficientBuffer.GetMappedCudaPtr();
            m_deviceCoefficientIndexBuffer.GetMappedCudaPtr();

            m_deviceHexVertexIndices.GetMappedCudaPtr();
            m_deviceHexPlaneBuffer.GetMappedCudaPtr();
            m_deviceHexVertexFaceIndex.GetMappedCudaPtr();
            m_deviceNumberOfModes.GetMappedCudaPtr();
        }

        void NektarModel::DoUnMapInteropBufferForCuda()
        {
            m_deviceVertexBuffer.UnmapCudaPtr();
            m_deviceCoefficientBuffer.UnmapCudaPtr();
            m_deviceCoefficientIndexBuffer.UnmapCudaPtr();

            m_deviceHexVertexIndices.UnmapCudaPtr();
            m_deviceHexPlaneBuffer.UnmapCudaPtr();
            m_deviceHexVertexFaceIndex.UnmapCudaPtr();
            m_deviceNumberOfModes.UnmapCudaPtr();
        }
    }

    
}
