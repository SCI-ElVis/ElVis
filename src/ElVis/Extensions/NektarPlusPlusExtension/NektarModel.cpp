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
#include <MultiRegions/ExpList2D.h>
#include <ElVis/Core/Util.hpp>
#include <ElVis/Core/PtxManager.h>
#include <ElVis/Core/Float.h>
#include <ElVis/Core/Point.hpp>
#include <ElVis/Core/Scene.h>
#include <ElVis/Core/Jacobi.hpp>
#include <ElVis/Core/SceneView.h>
#include <ElVis/Core/Float.h>

#include <boost/range.hpp>
#include <boost/range/algorithm/for_each.hpp>
#include <boost/range/adaptor/map.hpp>

#include <algorithm>

#include <math.h>

#include <exception>

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
            Model(modelPrefix),
            m_impl()
            ,m_graph()
            ,m_globalExpansions()
            ,m_fieldDefinitions()
            ,m_session()
            ,m_hexGeometryIntersectionProgram(0)
            ,m_hexPointLocationProgram(0)
            ,m_2DTriangleIntersectionProgram(0)
            ,m_2DTriangleBoundingBoxProgram(0)
            ,m_2DQuadIntersectionProgram(0)
            ,m_2DQuadBoundingBoxProgram(0)
            ,m_2DElementClosestHitProgram(0)
            ,m_TwoDClosestHitProgram(0)
            ,m_FieldBases("FieldBases")
            ,m_FieldModes("FieldModes")
            ,m_SumPrefixNumberOfFieldCoefficients("SumPrefixNumberOfFieldCoefficients")
            ,m_deviceVertexBuffer("Vertices")
            ,m_deviceCoefficientBuffer("Coefficients")
            ,m_deviceCoefficientOffsetBuffer("CoefficientOffsets")
            ,m_deviceHexVertexIndices("HexVertexIndices")
            ,m_deviceHexPlaneBuffer("HexPlaneBuffer")
            ,m_deviceHexVertexFaceIndex("Hexvertex_face_index")
            ,m_deviceNumberOfModes("NumberOfModes")
            ,PlanarFaceVertexBuffer("PlanarFaceVertexBuffer")
            ,FaceNormalBuffer("FaceNormalBuffer")
            ,m_deviceTriangleVertexIndexMap("TriangleVertexIndices")
            ,m_TriangleModes("TriangleModes")
            ,m_TriangleMappingCoeffsDir0("TriangleMappingCoeffsDir0")
            ,m_TriangleMappingCoeffsDir1("TriangleMappingCoeffsDir1")
            ,m_TriangleCoeffMappingDir0("TriangleCoeffMappingDir0")
            ,m_TriangleCoeffMappingDir1("TriangleCoeffMappingDir1")
            ,m_TriangleGlobalIdMap("TriangleGlobalIdMap")
            ,m_QuadModes("QuadModes")
            ,m_QuadMappingCoeffsDir0("QuadMappingCoeffsDir0")
            ,m_QuadMappingCoeffsDir1("QuadMappingCoeffsDir1")
            ,m_QuadCoeffMappingDir0("QuadCoeffMappingDir0")
            ,m_QuadCoeffMappingDir1("QuadCoeffMappingDir1")
            ,m_deviceQuadVertexIndexMap("QuadVertexIndices")
            ,m_triLocalToGlobalIdxMap()
            ,m_quadLocalToGlobalIdxMap()
        {
            boost::filesystem::path geometryFile(modelPrefix + ".xml");
            boost::filesystem::path fieldFile(modelPrefix + ".fld");

            if( !boost::filesystem::exists(geometryFile) )
            {
                std::string message = std::string("File ") + geometryFile.string() + " does not exist.";
                throw std::runtime_error(message.c_str());
            }

            Initialize(geometryFile, fieldFile);
        }

        NektarModel::~NektarModel()
        {
        }

        void NektarModel::DoCalculateExtents(ElVis::WorldPoint& minResult, ElVis::WorldPoint& maxResult)
        {
            for(unsigned int i = 0; i < m_graph->GetNvertices(); ++i)
            {
                BOOST_AUTO(vertex, m_graph->GetVertex(i));
                
                ElVis::WorldPoint p(vertex->x(), vertex->y(), vertex->z());
                minResult = CalcMin(minResult, p);
                maxResult = CalcMax(maxResult, p);
            };
        }

        const std::string& NektarModel::DoGetPTXPrefix() const
        {
            static std::string prefix("NektarPlusPlusExtension");
            return prefix;
        }

        Nektar::Array<Nektar::OneD, double> Create2DArray(double v0, double v1)
        {
            double values[] = {v0, v1};
            Nektar::Array<Nektar::OneD, double> result(2, values);
            return result;
        }

        // The desired result of this method is to load all of the data from the field
        // so I can easily access the expansions defined for each element.  The 
        // code as written seems to work OK for 3D hexes, but seems to be failing 
        // for 2D elements
        void NektarModel::LoadFields(const boost::filesystem::path& fieldFile)
        {
            std::vector<std::vector<double> > fieldData;

            // I assume that Import populates fieldDefinitions and fieldData with 
            // one entry per defined field (so a simulation with pressure and 
            // temperature would create 2 elements in each of these vectors).
            m_graph->Import(fieldFile.string(), m_fieldDefinitions, fieldData);

            // I got this code from one of the demos a long time ago.  
            // It doesn't seem to make sense in this context, but if I comment 
            // it out, Nektar++ throws an exception later because the points 
            // are not defined.  I'm not sure why I need to be defining them
            // (shouldn't this be part of reading the field?).
            for(int i = 0; i < m_fieldDefinitions.size(); ++i)
            {
                vector<LibUtilities::PointsType> ptype;
                for(int j = 0; j < 3; ++j)
                {
                    ptype.push_back(LibUtilities::ePolyEvenlySpaced);
                }
                
                m_fieldDefinitions[i]->m_pointsDef = true;
                m_fieldDefinitions[i]->m_points    = ptype; 
                
                vector<unsigned int> porder;
                if(m_fieldDefinitions[i]->m_numPointsDef == false)
                {
                    for(int j = 0; j < m_fieldDefinitions[i]->m_numModes.size(); ++j)
                    {
                        porder.push_back(m_fieldDefinitions[i]->m_numModes[j]);
                    }
                    
                    m_fieldDefinitions[i]->m_numPointsDef = true;
                }
                else
                {
                    for(int j = 0; j < m_fieldDefinitions[i]->m_numPoints.size(); ++j)
                    {
                        porder.push_back(m_fieldDefinitions[i]->m_numPoints[j]);
                    }
                }
                m_fieldDefinitions[i]->m_numPoints = porder;
                
            }

            m_graph->SetExpansions(m_fieldDefinitions);

            int expdim  = m_graph->GetMeshDimension();
            int nfields = m_fieldDefinitions[0]->m_fields.size();

            Nektar::SpatialDomains::MeshGraph3DSharedPtr as3d = boost::dynamic_pointer_cast<
                Nektar::SpatialDomains::MeshGraph3D>(m_graph);
            Nektar::SpatialDomains::MeshGraph2DSharedPtr as2d = boost::dynamic_pointer_cast<
                Nektar::SpatialDomains::MeshGraph2D>(m_graph);

            if( as3d )
            {
                Nektar::MultiRegions::ExpList3DSharedPtr exp = MemoryManager<Nektar::MultiRegions::ExpList3D>
                    ::AllocateSharedPtr(m_session, m_graph);
                m_globalExpansions.push_back(exp);
                for(int i = 1; i < nfields; ++i)
                {
                    m_globalExpansions.push_back(MemoryManager<Nektar::MultiRegions::ExpList3D>
                        ::AllocateSharedPtr(*exp));
                }
            }
            else
            {
                Nektar::MultiRegions::ExpList2DSharedPtr exp = MemoryManager<Nektar::MultiRegions::ExpList2D>
                    ::AllocateSharedPtr(m_session, m_graph);
                m_globalExpansions.push_back(exp);
                for(int i = 1; i < nfields; ++i)
            {
                    m_globalExpansions.push_back(MemoryManager<Nektar::MultiRegions::ExpList2D>
                        ::AllocateSharedPtr(*exp));
            }
        }
                
            for(int j = 0; j < nfields; ++j)
        {
                for(int i = 0; i < fieldData.size(); ++i)
            {
                    m_globalExpansions[j]->ExtractDataToCoeffs(m_fieldDefinitions[i],
                                                fieldData[i],
                     m_fieldDefinitions[i]->m_fields[j], m_globalExpansions[j]->UpdateCoeffs());
                }
                m_globalExpansions[j]->BwdTrans(
                    m_globalExpansions[j]->GetCoeffs(),
                    m_globalExpansions[j]->UpdatePhys());
            }

            //for(unsigned int i = 0; i < fieldDefinitions.size(); ++i)
            //{
            //    m_globalExpansion->ExtractDataToCoeffs(fieldDefinitions[i], fieldData[i], fieldDefinitions[i]->m_fields[0]);
            //}

            //m_globalExpansion->BwdTrans(m_globalExpansion->GetCoeffs(), m_globalExpansion->UpdatePhys());

            //m_globalExpansion->PutCoeffsInToElmtExp();
            //m_globalExpansion->PutPhysInToElmtExp();
        }

        void NektarModel::Initialize(const boost::filesystem::path& geomFile,
            const boost::filesystem::path& fieldFile)
        {
            int argc = 3;
            char* arg1 = strdup("ElVis");
            char* arg2 = strdup(geomFile.string().c_str());
            char* arg3 = strdup(fieldFile.string().c_str());
            char* argv[] = {arg1, arg2, arg3};
            m_session = LibUtilities::SessionReader::CreateInstance(argc, argv);
            free(arg1);
            free(arg2);
            free(arg3);

            arg1 = 0;
            arg2 = 0;
            arg3 = 0;

            m_graph = SpatialDomains::MeshGraph::Read(m_session);

            Nektar::SpatialDomains::MeshGraph3DSharedPtr as3d = boost::dynamic_pointer_cast<
                Nektar::SpatialDomains::MeshGraph3D>(m_graph);
            Nektar::SpatialDomains::MeshGraph2DSharedPtr as2d = boost::dynamic_pointer_cast<
                Nektar::SpatialDomains::MeshGraph2D>(m_graph);

            if( as3d )
            {
                m_impl.reset(new ThreeDNektarModel(this, as3d));
            }
            else
        {
                m_impl.reset(new TwoDNektarModel(this, as2d));
        }
            CalculateExtents();
            LoadFields(fieldFile);
        }

        void NektarModel::SetupFaces()
        {
          CreateLocalToGlobalIdxMap(m_graph->GetAllTriGeoms(), m_triLocalToGlobalIdxMap);
          CreateLocalToGlobalIdxMap(m_graph->GetAllQuadGeoms(), m_quadLocalToGlobalIdxMap);
        }

        std::vector<optixu::GeometryGroup> NektarModel::DoGetPointLocationGeometry(boost::shared_ptr<Scene> scene, optixu::Context context)
        {
           
            try
            {
                std::vector<optixu::GeometryGroup> result;
                if( !m_graph ) return result;

                SetupOptixCoefficientBuffers(context);
                SetupOptixVertexBuffers(context);

                optixu::Material m_hexCutSurfaceMaterial = context->createMaterial();
                optixu::Program hexBoundingProgram = PtxManager::LoadProgram(context, GetPTXPrefix(), "NektarHexahedronBounding");
                optixu::Program hexIntersectionProgram = PtxManager::LoadProgram(context, GetPTXPrefix(), HexahedronIntersectionProgramName);
                optixu::GeometryInstance hexInstance = CreateGeometryForElementType<Nektar::SpatialDomains::HexGeom>(context, "Hex");
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

        void NektarModel::SetupSumPrefixNumberOfFieldCoefficients(optixu::Context context)
        {
            m_SumPrefixNumberOfFieldCoefficients.SetContext(context);
            m_SumPrefixNumberOfFieldCoefficients.SetDimensions(m_globalExpansions.size());
            BOOST_AUTO(data, m_SumPrefixNumberOfFieldCoefficients.Map());

            data[0] = 0;

            for(int i = 0; i < m_globalExpansions.size()-1; ++i)
            {
                data[i+1] = data[i] + m_globalExpansions[i]->GetNcoeffs();
            }
        }

        void NektarModel::SetupCoefficientOffsetBuffer(optixu::Context context)
        {
            m_deviceCoefficientOffsetBuffer.SetContext(context);
            int numElements = m_globalExpansions[0]->GetNumElmts();
            m_deviceCoefficientOffsetBuffer.SetDimensions(m_globalExpansions.size()*numElements);
            
            for(unsigned int i = 0; i < m_globalExpansions.size(); ++i)
            {
                BOOST_AUTO(exp, m_globalExpansions[i]);
                if( exp->GetNumElmts() != numElements )
                {
                    throw std::runtime_error("Expansions must have the same number of elements.");
                }
            }

            // The offset is at elementId*numField + fieldId

            BOOST_AUTO(coefficientIndicesData, m_deviceCoefficientOffsetBuffer.Map());

            for(unsigned int expansionIndex = 0; expansionIndex < m_globalExpansions.size(); ++ expansionIndex)
            {
                for(unsigned int elementId = 0; elementId < numElements; ++elementId)
                {
                    BOOST_AUTO(exp, m_globalExpansions[expansionIndex]);
                    BOOST_AUTO(element, exp->GetExp(elementId));
                    BOOST_AUTO(id, element->GetGeom()->GetGlobalID());
                    BOOST_AUTO(offset, exp->GetCoeff_Offset(elementId));

                    coefficientIndicesData[expansionIndex*numElements + id] = offset;
                }
            }
        }

        void NektarModel::SetupFieldModes(optixu::Context context)
        {
            int numElements = m_globalExpansions[0]->GetNumElmts();
            int numFields = m_globalExpansions.size();
            m_FieldModes.SetContext(context);
            m_FieldModes.SetDimensions(numElements*numFields);

            BOOST_AUTO(data, m_FieldModes.Map());

            for(int fieldId = 0; fieldId < m_globalExpansions.size(); ++fieldId)
            {
                BOOST_AUTO(exp, m_globalExpansions[fieldId]);
                for(int elementId = 0; elementId < exp->GetNumElmts(); ++elementId)
                {
                    BOOST_AUTO( element, exp->GetExp(elementId));
                    BOOST_AUTO( id, element->GetGeom()->GetGlobalID());
                    BOOST_AUTO( bases, element->GetBase());
                    uint3 value;
                    value.x = bases[0]->GetNumModes();
                    if( bases.num_elements() > 1 )
                    {
                        value.y = bases[1]->GetNumModes();
                    }
                    if( bases.num_elements() > 2 )
                    {
                        value.z = bases[2]->GetNumModes();
                    }
                    data[fieldId*numElements + id] = value;
                }
            }
        }

        void NektarModel::SetupFieldBases(optixu::Context context)
        {
            int numElements = m_globalExpansions[0]->GetNumElmts();
            int numFields = m_globalExpansions.size();
            m_FieldBases.SetContext(context);
            m_FieldBases.SetDimensions(numElements*numFields*3);

            BOOST_AUTO(data, m_FieldBases.Map());

            for(int fieldId = 0; fieldId < m_globalExpansions.size(); ++fieldId)
            {
                BOOST_AUTO( exp, m_globalExpansions[fieldId]);
                for(int elementId = 0; elementId < exp->GetNumElmts(); ++elementId)
                {
                    BOOST_AUTO(element, exp->GetExp(elementId));
                    BOOST_AUTO( id, element->GetGeom()->GetGlobalID());
                    BOOST_AUTO( bases, element->GetBase());

                    int idx = fieldId*numElements + 3*id;

                    uint3 value;
                    data[idx] = bases[0]->GetBasisType();
                    if( bases.num_elements() > 1 )
                    {
                        data[idx+1] = bases[1]->GetBasisType();
                    }
                    if( bases.num_elements() > 2 )
                    {
                        data[idx+2] = bases[2]->GetBasisType();
                    }
                }
            }
        }

        int calculateNumCoeffs(int oldValue, Nektar::MultiRegions::ExpListSharedPtr exp)
        {
            return oldValue + exp->GetNcoeffs();
        };

        void NektarModel::SetupOptixCoefficientBuffers(optixu::Context context)
        {
            context["NumElements"]->setUint(m_globalExpansions[0]->GetNumElmts());
            SetupSumPrefixNumberOfFieldCoefficients(context);
            SetupCoefficientOffsetBuffer(context);
            SetupFieldBases(context);
            SetupFieldModes(context);

            int numCoeffs = std::accumulate(m_globalExpansions.begin(),
                m_globalExpansions.end(), 0,
                calculateNumCoeffs);

            m_deviceCoefficientBuffer.SetContext(context);
            m_deviceCoefficientBuffer.SetDimensions(numCoeffs);
            
            BOOST_AUTO(coeffData, m_deviceCoefficientBuffer.Map());
            int coeffIndex = 0;

            for(unsigned int i = 0; i < m_globalExpansions.size(); ++i)
            //std::for_each(m_globalExpansions.begin(), m_globalExpansions.end(),
            //    [&](Nektar::MultiRegions::ExpListSharedPtr exp)
            //{
            {
                BOOST_AUTO(exp, m_globalExpansions[i]);
                for(unsigned int i = 0; i < exp->GetNcoeffs(); ++i)
                {
                    coeffData[coeffIndex] = exp->GetCoeff(i);
                    ++coeffIndex;
                }    
            }
        }

        void NektarModel::SetupOptixVertexBuffers(optixu::Context context)
        {
            m_deviceVertexBuffer.SetContext(context);
            m_deviceVertexBuffer.SetDimensions(m_graph->GetNvertices());

            BOOST_AUTO(vertexData, m_deviceVertexBuffer.Map());
            for(unsigned int i = 0; i < m_graph->GetNvertices(); ++i)
            {
                BOOST_AUTO( vertex, m_graph->GetVertex(i));
                vertexData[i] = ::MakeFloat4(vertex->x(), vertex->y(), vertex->z(), 1.0);
            }
        }

        


        LibUtilities::SessionReaderSharedPtr NektarModel::GetSession() const
        {
            return m_session;
        }
        

        void NektarModel::DoGetFaceGeometry(boost::shared_ptr<Scene> scene, optixu::Context context, optixu::Geometry& faceGeometry)
        {
            int numFaces = 0;
            numFaces += m_graph->GetAllTriGeoms().size();
            numFaces += m_graph->GetAllQuadGeoms().size();

            PlanarFaceVertexBuffer.SetContext(context);
            PlanarFaceVertexBuffer.SetDimensions(numFaces*4);
            FaceNormalBuffer.SetContext(context);
            FaceNormalBuffer.SetDimensions(numFaces);

            BOOST_AUTO(faceVertexBuffer, PlanarFaceVertexBuffer.Map());
            BOOST_AUTO(normalBuffer, FaceNormalBuffer.Map());

            AddFaces(m_graph->GetAllTriGeoms(), 0, 0, faceVertexBuffer.get(), 0/* faceDefs.get()*/, normalBuffer.get());

            int offset = m_graph->GetAllTriGeoms().size();
            AddFaces(m_graph->GetAllQuadGeoms(), 0, 0, faceVertexBuffer.get()+offset, /*faceDefs.get()+offset*/ 0, normalBuffer.get()+offset);

            faceGeometry->setPrimitiveCount(numFaces);
            //curvedFaces->setPrimitiveCount(faces.size());
        }

        int NektarModel::DoGetNumberOfBoundarySurfaces() const
        {
            return 1;
        }

        void NektarModel::DoGetBoundarySurface(int surfaceIndex, std::string& name, std::vector<int>& faceIds)
        {
            name = std::string("All Faces");
            for(int i = 0; i < GetNumberOfFaces(); ++i)
            {
                faceIds.push_back(i);
            }
        }

        size_t NektarModel::DoGetNumberOfFaces() const
        {
            int numFaces = 0;
            if( m_graph )
            {
                numFaces += m_graph->GetAllTriGeoms().size();
                numFaces += m_graph->GetAllQuadGeoms().size();
            }
            return numFaces;
        }

        namespace
        {
          void setupFaceAdjacency(Nektar::SpatialDomains::MeshGraphSharedPtr m_graph,
            boost::shared_ptr<Nektar::SpatialDomains::Geometry2D> face,
            FaceInfo& result)
          {
            boost::shared_ptr<Nektar::SpatialDomains::Geometry2D> geom = boost::dynamic_pointer_cast<Nektar::SpatialDomains::Geometry2D>( face );

            Nektar::SpatialDomains::MeshGraph3DSharedPtr castPtr = boost::dynamic_pointer_cast<Nektar::SpatialDomains::MeshGraph3D>(m_graph);
            if( castPtr )
            {
              Nektar::SpatialDomains::ElementFaceVectorSharedPtr elements = castPtr->GetElementsFromFace(geom);
              assert(elements->size() <= 2 );
              result.CommonElements[0].Id = -1;
              result.CommonElements[0].Type = -1;
              result.CommonElements[1].Id = -1;
              result.CommonElements[1].Type = -1;
              for(int elementId = 0; elementId < elements->size(); ++elementId)
              {
                  result.CommonElements[elementId].Id = (*elements)[elementId]->m_Element->GetGlobalID();
                  result.CommonElements[elementId].Type = (*elements)[elementId]->m_Element->GetGeomShapeType();
              }
            }
          }

          void calculateBoundingBox(boost::shared_ptr<Nektar::SpatialDomains::Geometry2D> face,
            FaceInfo& result)
          {
            WorldPoint minExtent(std::numeric_limits<ElVisFloat>::max(), std::numeric_limits<ElVisFloat>::max(), std::numeric_limits<ElVisFloat>::max());
            WorldPoint maxExtent(-std::numeric_limits<ElVisFloat>::max(), -std::numeric_limits<ElVisFloat>::max(), -std::numeric_limits<ElVisFloat>::max());

            boost::shared_ptr<Nektar::SpatialDomains::Geometry2D> geom = boost::dynamic_pointer_cast<Nektar::SpatialDomains::Geometry2D>( face );

            for(int i = 0; i < geom->GetNumVerts(); ++i)
            {
              Nektar::SpatialDomains::VertexComponentSharedPtr rawVertex = geom->GetVertex(i);
              WorldPoint v(rawVertex->x(), rawVertex->y(), rawVertex->z());
              minExtent = CalcMin(minExtent, v);
              maxExtent = CalcMax(maxExtent, v);
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

            result.MinExtent = MakeFloat3(minExtent);
            result.MaxExtent = MakeFloat3(maxExtent);
          }
        }

        FaceInfo NektarModel::DoGetFaceDefinition(size_t globalFaceId) const
        {
          boost::shared_ptr<Nektar::SpatialDomains::Geometry2D> face;
          BOOST_AUTO(foundTri, m_graph->GetAllTriGeoms().find(globalFaceId));
          if( foundTri != m_graph->GetAllTriGeoms().end() )
          {
            face = (*foundTri).second;
          }
          else
          {
            BOOST_AUTO(foundQuad, m_graph->GetAllQuadGeoms().find(globalFaceId));
            if( foundQuad != m_graph->GetAllQuadGeoms().end() )
            {
              face = (*foundQuad).second;
            }
          }

          if( !face )
          {
            std::string msg = "Unable to find Nektar++ face with global id " +
              boost::lexical_cast<std::string>(globalFaceId);
            throw new std::runtime_error(msg.c_str());
          }

          FaceInfo result;

          // TODO - Update to detect curved faces.
          result.Type = ePlanar;
          calculateBoundingBox(face, result);
          setupFaceAdjacency(m_graph, face, result);
          return result;
        }

        size_t NektarModel::DoGetNumberOfPlanarFaceVertices() const
        {
          return 0;
        }

        WorldPoint NektarModel::DoGetPlanarFaceVertex(size_t vertexIdx) const
        {
          return WorldPoint();
        }

        size_t NektarModel::DoGetNumberOfVerticesForPlanarFace(size_t globalFaceId) const
        {
          return 0;
        }

        size_t NektarModel::DoGetPlanarFaceVertexIndex(size_t globalFaceId, size_t vertexId)
        {
          return 0;
        }

        namespace detail
        {
            
            void SetupTriangleModes(Nektar::SpatialDomains::TriGeomSharedPtr tri,
              uint2* modeArray, uint* expansion0Sizes, uint* expansion1Sizes, 
              int* expansionSize, int& idx)
            {      
                Nektar::StdRegions::StdExpansion2DSharedPtr x0 = tri->GetXmap(0);
                Nektar::StdRegions::StdExpansion2DSharedPtr x1 = tri->GetXmap(1);
                const Nektar::Array<Nektar::OneD, const Nektar::NekDouble>& u0 = x0->GetCoeffs();
                const Nektar::Array<Nektar::OneD, const Nektar::NekDouble>& u1 = x1->GetCoeffs();
                Nektar::LibUtilities::BasisSharedPtr b0 = x0->GetBasis(0);
                Nektar::LibUtilities::BasisSharedPtr b1 = x0->GetBasis(1);

                modeArray[idx].x = b0->GetNumModes();
                modeArray[idx].y = b1->GetNumModes();
                    
                expansionSize[0] += u0.num_elements();
                expansionSize[1] += u1.num_elements();

                expansion0Sizes[idx] = u0.num_elements();
                expansion1Sizes[idx] = u1.num_elements();
                    
                ++idx;
            }



            void CopyTriangleCoefficients(Nektar::SpatialDomains::TriGeomSharedPtr tri,
              ElVisFloat*& coeffs0, ElVisFloat*& coeffs1)
            {
                Nektar::StdRegions::StdExpansion2DSharedPtr x0 = tri->GetXmap(0);
                Nektar::StdRegions::StdExpansion2DSharedPtr x1 = tri->GetXmap(1);
                const Nektar::Array<Nektar::OneD, const Nektar::NekDouble>& u0 = x0->GetCoeffs();
                const Nektar::Array<Nektar::OneD, const Nektar::NekDouble>& u1 = x1->GetCoeffs();

                std::copy(u0.begin(), u0.end(), coeffs0);
                std::copy(u1.begin(), u1.end(), coeffs1);
                coeffs0 += u0.num_elements();
                coeffs1 += u1.num_elements();
            }

        }

        std::vector<optixu::GeometryInstance> NektarModel::DoGet2DPrimaryGeometry(boost::shared_ptr<Scene> scene, optixu::Context context)
        {
            std::vector<optixu::GeometryInstance> result;
            if( m_graph->GetMeshDimension() != 2 )
            {
                return result;
            }

            int numChildren = 0;
            int numTriangles = m_graph->GetAllTriGeoms().size();
            int numQuads = m_graph->GetAllQuadGeoms().size();

            if( numTriangles > 0 ) ++numChildren;
            if( numQuads > 0 ) ++numChildren;

            if( numChildren == 0 )
            {
                return result;
            }

            if( numTriangles > 0 )
            {               
                optixu::GeometryInstance instance = context->createGeometryInstance();
                optixu::Geometry geometry = context->createGeometry();
                geometry->setPrimitiveCount(numTriangles);
                instance->setGeometry(geometry);

                // Intersection Program
                if( !m_2DTriangleIntersectionProgram )
                {
                    m_2DTriangleIntersectionProgram = PtxManager::LoadProgram(context, GetPTXPrefix(), "NektarTriangleIntersection");
                }
                if( !m_2DTriangleBoundingBoxProgram )
                {
                    m_2DTriangleBoundingBoxProgram = PtxManager::LoadProgram(context, GetPTXPrefix(), "NektarTriangleBounding");
                }
                geometry->setBoundingBoxProgram(m_2DTriangleBoundingBoxProgram);
                geometry->setIntersectionProgram(m_2DTriangleIntersectionProgram);

                result.push_back(instance);

                // Setup the vertex map.
                m_deviceTriangleVertexIndexMap.SetContext(context);
                m_TriangleModes.SetContext(context);
                m_TriangleMappingCoeffsDir0.SetContext(context);
                m_TriangleMappingCoeffsDir1.SetContext(context);
                m_TriangleCoeffMappingDir0.SetContext(context);
                m_TriangleCoeffMappingDir1.SetContext(context);
                m_TriangleGlobalIdMap.SetContext(context);

                m_deviceTriangleVertexIndexMap.SetDimensions(3*numTriangles);
                m_TriangleGlobalIdMap.SetDimensions(numTriangles);
                BOOST_AUTO(globalElementIdMap, m_TriangleGlobalIdMap.Map());
                BOOST_AUTO(data, m_deviceTriangleVertexIndexMap.Map());
                int i = 0;
                for( Nektar::SpatialDomains::TriGeomMap::const_iterator iter = m_graph->GetAllTriGeoms().begin();
                    iter != m_graph->GetAllTriGeoms().end(); ++iter)
                {
                    BOOST_AUTO( tri, (*iter).second);
                    int id0 = tri->GetVid(0);
                    BOOST_AUTO( v0, tri->GetVertex(0));
                    BOOST_AUTO( gv0, m_graph->GetVertex(id0));
                    data[3*i] = tri->GetVid(0);
                    data[3*i+1] = tri->GetVid(1);
                    data[3*i+2] = tri->GetVid(2);

                    globalElementIdMap[i] = tri->GetGlobalID();
                    ++i;
                }

                // Setup the geometric expansion information.
                m_TriangleModes.SetDimensions(numTriangles);
                m_TriangleCoeffMappingDir0.SetDimensions(numTriangles);
                m_TriangleCoeffMappingDir1.SetDimensions(numTriangles);
                BOOST_AUTO(modeArray, m_TriangleModes.Map());
                BOOST_AUTO(coeffMapping0Array, m_TriangleCoeffMappingDir0.Map());
                BOOST_AUTO(coeffMapping1Array, m_TriangleCoeffMappingDir1.Map());

                int expansionSize[] = {0, 0};

                uint* expansion0Sizes = new uint[numTriangles];
                uint* expansion1Sizes = new uint[numTriangles];
                int idx = 0;

                boost::for_each(m_graph->GetAllTriGeoms() | boost::adaptors::map_values, 
                    boost::bind(&detail::SetupTriangleModes, _1,
                    modeArray.get(), expansion0Sizes, expansion1Sizes, expansionSize, boost::ref(idx)));

                //    [&](Nektar::SpatialDomains::TriGeomSharedPtr tri)
                //{
                //    Nektar::StdRegions::StdExpansion2DSharedPtr x0 = tri->GetXmap(0);
                //    Nektar::StdRegions::StdExpansion2DSharedPtr x1 = tri->GetXmap(1);
                //    const Nektar::Array<Nektar::OneD, const Nektar::NekDouble>& u0 = x0->GetCoeffs();
                //    const Nektar::Array<Nektar::OneD, const Nektar::NekDouble>& u1 = x1->GetCoeffs();
                //    Nektar::LibUtilities::BasisSharedPtr b0 = x0->GetBasis(0);
                //    Nektar::LibUtilities::BasisSharedPtr b1 = x0->GetBasis(1);
                    
                //    modeArray[idx].x = b0->GetNumModes();
                //    modeArray[idx].y = b1->GetNumModes();
                //    
                //    expansionSize[0] += u0.num_elements();
                //    expansionSize[1] += u1.num_elements();

                //    expansion0Sizes[idx] = u0.num_elements();
                //    expansion1Sizes[idx] = u1.num_elements();
                //    
                //    ++idx;
                //});

                coeffMapping0Array[0] = 0;
                coeffMapping1Array[0] = 0;
                for(int i = 1; i < numTriangles; ++i)
                {
                    coeffMapping0Array[i] = coeffMapping0Array[i-1] + expansion0Sizes[i-1];
                    coeffMapping1Array[i] = coeffMapping1Array[i-1] + expansion1Sizes[i-1];
                }

                delete [] expansion0Sizes;
                delete [] expansion1Sizes;

                m_TriangleMappingCoeffsDir0.SetDimensions(expansionSize[0]);
                m_TriangleMappingCoeffsDir1.SetDimensions(expansionSize[1]);
                BOOST_AUTO(coeffs0Array,  m_TriangleMappingCoeffsDir0.Map());
                BOOST_AUTO(coeffs1Array, m_TriangleMappingCoeffsDir1.Map());

                BOOST_AUTO(coeffs0, coeffs0Array.get());
                BOOST_AUTO(coeffs1, coeffs1Array.get());

                ElVisFloat* base0 = coeffs0;
                ElVisFloat* base1 = coeffs1;

                boost::for_each(m_graph->GetAllTriGeoms() | boost::adaptors::map_values, 
                    boost::bind(&detail::CopyTriangleCoefficients, _1, boost::ref(coeffs0),
                    boost::ref(coeffs1)));

                //    [&](Nektar::SpatialDomains::TriGeomSharedPtr tri)
                //{
                //    Nektar::StdRegions::StdExpansion2DSharedPtr x0 = tri->GetXmap(0);
                //    Nektar::StdRegions::StdExpansion2DSharedPtr x1 = tri->GetXmap(1);
                //    const Nektar::Array<Nektar::OneD, const Nektar::NekDouble>& u0 = x0->GetCoeffs();
                //    const Nektar::Array<Nektar::OneD, const Nektar::NekDouble>& u1 = x1->GetCoeffs();

                //    std::copy(u0.begin(), u0.end(), coeffs0);
                //    std::copy(u1.begin(), u1.end(), coeffs1);
                //    coeffs0 += u0.num_elements();
                //    coeffs1 += u1.num_elements();
                //});


                //boost::for_each(m_graph->GetAllTriGeoms() | boost::adaptors::map_values, 
                //    [&](Nektar::SpatialDomains::TriGeomSharedPtr tri)
                //{
                //    // TODO - Check the correspondence between vertex id and global id. 
                //    BOOST_AUTO(localExpansion, m_globalExpansion->GetExp(hex->GetGlobalID()));

                //    modesData[i*3] = localExpansion->GetBasis(0)->GetNumModes();
                //    modesData[i*3+1] = localExpansion->GetBasis(1)->GetNumModes();
                //    modesData[i*3+2] = localExpansion->GetBasis(2)->GetNumModes();
                //    ++i;
                //});
            }

            //if( numQuads > 0 )
            //{
            //    optixu::GeometryInstance instance = context->createGeometryInstance();
            //    optixu::Geometry geometry = context->createGeometry();
            //    geometry->setPrimitiveCount(numQuads);
            //    instance->setGeometry(geometry);

            //    // Intersection Program
            //    if( !m_2DQuadIntersectionProgram )
            //    {
            //        m_2DQuadIntersectionProgram = PtxManager::LoadProgram(context, GetPTXPrefix(), "NektarQuadIntersection");
            //    }
            //    if( !m_2DQuadBoundingBoxProgram )
            //    {
            //        m_2DQuadBoundingBoxProgram = PtxManager::LoadProgram(context, GetPTXPrefix(), "NektarQuadBounding");
            //    }
            //    geometry->setBoundingBoxProgram(m_2DQuadBoundingBoxProgram);
            //    geometry->setIntersectionProgram(m_2DQuadIntersectionProgram);

            //    result.push_back(instance);

            //    // Setup the vertex map.
            //    m_deviceQuadVertexIndexMap.SetContext(context, module);
            //    m_QuadModes.SetContext(context, module);
            //    m_QuadMappingCoeffsDir0.SetContext(context, module);
            //    m_QuadMappingCoeffsDir1.SetContext(context, module);
            //    m_QuadCoeffMappingDir0.SetContext(context, module);
            //    m_QuadCoeffMappingDir1.SetContext(context, module);

            //    m_deviceQuadVertexIndexMap.SetDimensions(4*numQuads);
            //    uint* data = m_deviceQuadVertexIndexMap.Map();
            //    int i = 0;
            //    for( Nektar::SpatialDomains::QuadGeomMap::const_iterator iter = m_graph->GetAllQuadGeoms().begin();
            //        iter != m_graph->GetAllQuadGeoms().end(); ++iter)
            //    {
            //        auto tri = (*iter).second;
            //        int id0 = tri->GetVid(0);
            //        auto v0 = tri->GetVertex(0);
            //        auto gv0 = m_graph->GetVertex(id0);
            //        data[4*i] = tri->GetVid(0);
            //        data[4*i+1] = tri->GetVid(1);
            //        data[4*i+2] = tri->GetVid(2);
            //        data[4*i+3] = tri->GetVid(3);
            //        ++i;
            //    }

            //    // Setup the geometric expansion information.
            //    m_QuadModes.SetDimensions(numQuads);
            //    m_QuadCoeffMappingDir0.SetDimensions(numQuads);
            //    m_QuadCoeffMappingDir1.SetDimensions(numQuads);
            //    uint2* modeArray = m_QuadModes.Map();
            //    uint* coeffMapping0Array = m_QuadCoeffMappingDir0.Map();
            //    uint* coeffMapping1Array = m_QuadCoeffMappingDir1.Map();

            //    int expansionSize[] = {0, 0};

            //    uint* expansion0Sizes = new uint[numQuads];
            //    uint* expansion1Sizes = new uint[numQuads];
            //    int idx = 0;
            //    boost::for_each(m_graph->GetAllQuadGeoms() | boost::adaptors::map_values, 
            //        [&](Nektar::SpatialDomains::QuadGeomSharedPtr tri)
            //    {
            //        Nektar::StdRegions::StdExpansion2DSharedPtr x0 = tri->GetXmap(0);
            //        Nektar::StdRegions::StdExpansion2DSharedPtr x1 = tri->GetXmap(1);
            //        const Nektar::Array<Nektar::OneD, const Nektar::NekDouble>& u0 = x0->GetCoeffs();
            //        const Nektar::Array<Nektar::OneD, const Nektar::NekDouble>& u1 = x1->GetCoeffs();
            //        Nektar::LibUtilities::BasisSharedPtr b0 = x0->GetBasis(0);
            //        Nektar::LibUtilities::BasisSharedPtr b1 = x0->GetBasis(1);

            //        modeArray[idx].x = b0->GetNumModes();
            //        modeArray[idx].y = b1->GetNumModes();
            //        
            //        expansionSize[0] += u0.num_elements();
            //        expansionSize[1] += u1.num_elements();

            //        expansion0Sizes[idx] = u0.num_elements();
            //        expansion1Sizes[idx] = u1.num_elements();

            //        ++idx;
            //    });

            //    coeffMapping0Array[0] = 0;
            //    coeffMapping1Array[0] = 0;
            //    for(int i = 1; i < numQuads; ++i)
            //    {
            //        coeffMapping0Array[i] = coeffMapping0Array[i-1] + expansion0Sizes[i-1];
            //        coeffMapping1Array[i] = coeffMapping1Array[i-1] + expansion1Sizes[i-1];
            //    }

            //    delete [] expansion0Sizes;
            //    delete [] expansion1Sizes;

            //    m_QuadMappingCoeffsDir0.SetDimensions(expansionSize[0]);
            //    m_QuadMappingCoeffsDir1.SetDimensions(expansionSize[1]);
            //    ElVisFloat* coeffs0 = m_QuadMappingCoeffsDir0.Map();
            //    ElVisFloat* coeffs1 = m_QuadMappingCoeffsDir1.Map();

            //    ElVisFloat* base0 = coeffs0;
            //    ElVisFloat* base1 = coeffs1;
            //    boost::for_each(m_graph->GetAllQuadGeoms() | boost::adaptors::map_values, 
            //        [&](Nektar::SpatialDomains::QuadGeomSharedPtr tri)
            //    {
            //        Nektar::StdRegions::StdExpansion2DSharedPtr x0 = tri->GetXmap(0);
            //        Nektar::StdRegions::StdExpansion2DSharedPtr x1 = tri->GetXmap(1);
            //        const Nektar::Array<Nektar::OneD, const Nektar::NekDouble>& u0 = x0->GetCoeffs();
            //        const Nektar::Array<Nektar::OneD, const Nektar::NekDouble>& u1 = x1->GetCoeffs();

            //        std::copy(u0.begin(), u0.end(), coeffs0);
            //        std::copy(u1.begin(), u1.end(), coeffs1);
            //        coeffs0 += u0.num_elements();
            //        coeffs1 += u1.num_elements();
            //    });

            //}
            return result;
        }

        optixu::Material NektarModel::DoGet2DPrimaryGeometryMaterial(SceneView* view)
        {
            if( !m_TwoDClosestHitProgram.get() )
            {
                m_TwoDClosestHitProgram = PtxManager::LoadProgram(this->GetPTXPrefix(), "TwoDClosestHitProgram");
            }
            BOOST_AUTO( context, view->GetContext());
            optixu::Material result = context->createMaterial();
            result->setClosestHitProgram(0, m_TwoDClosestHitProgram);
            return result;
        }

        unsigned int NektarModel::DoGetNumberOfElements() const
        {
            return 0;
        }

        int NektarModel::DoGetNumFields() const
        {
            return m_fieldDefinitions[0]->m_fields.size();
        }

        FieldInfo NektarModel::DoGetFieldInfo(unsigned int index) const
        {
            FieldInfo result;
            result.Name =  m_fieldDefinitions[0]->m_fields[index];
            result.Id = index;
            result.Shortcut = "";
            return result;
        }

        int NektarModel::DoGetModelDimension() const
        {
            if( boost::dynamic_pointer_cast<ThreeDNektarModel>(m_impl) )
            {
                return 3;
            }
            else
            {
                return 2;
            }
    }

        NektarModelImpl::NektarModelImpl(NektarModel* model) :
            boost::noncopyable()
            ,m_model(model)
        {
        }

        NektarModelImpl::~NektarModelImpl()
        {
        }

                

        ThreeDNektarModel::ThreeDNektarModel(NektarModel* model, Nektar::SpatialDomains::MeshGraph3DSharedPtr mesh) :
            NektarModelImpl(model)
            , m_mesh(mesh)
        {
        }
    
        ThreeDNektarModel::~ThreeDNektarModel()
        {
        }

        TwoDNektarModel::TwoDNektarModel(NektarModel* model, Nektar::SpatialDomains::MeshGraph2DSharedPtr mesh) :
            NektarModelImpl(model)
            , m_mesh(mesh)
        {
        }
             
        TwoDNektarModel::~TwoDNektarModel()
        {
}
    }
   
}





// Old Code
//        template<typename T>
//        T ModifiedA(unsigned int i, const T& x)
//        {
//            if( i == 0 )
//            {
//                return (1.0-x)/2.0;
//            }
//            else if( i == 1 )
//            {
//                return (1.0+x)/2.0;
//            }
//            else
//            {
//                return (1.0-x)/2.0 * 
//                    (1.0 + x)/2.0 * 
//                    ElVis::OrthoPoly::P(i-2, 1, 1, x);
//            }
//        }
//
//        template<typename T>
//        T dModifiedA(unsigned int i, const T& x)
//        {
//            if( i == 0 )
//            {
//                return -.5;
//            }
//            else if( i == 1 )
//            {
//                return .5;
//            }
//            else
//            {
//                double poly = ElVis::OrthoPoly::P(i-2, 1, 1, x);
//                double dpoly = ElVis::OrthoPoly::dP(i-2, 1, 1, x);
//
//                return .25*(1-x)*poly - .25 * (1+x)*poly + 
//                    .25 * (1-x)*(1+x)*dpoly;
//            }
//        }
//
//        template<typename T>
//        T ModifiedB(unsigned int i, unsigned int j, const T& x)
//        {
//            if( i == 0 )
//            {
//                return ModifiedA(j, x);
//            }
//            else if( j == 0 )
//            {
//                return pow((1.0-x)/2.0, (double)i);
//            }
//            else
//            {
//                double result = 1.0;
//                result = pow((1.0-x)/2.0, (double)i);
//                result *= (1.0+x)/2.0;
//                result *= ElVis::OrthoPoly::P(j-1, 2*i-1, 1, x);
//                return result;
//            }
//        }
//
//        
//        template<typename T>
//        T dModifiedB(unsigned int i, unsigned int j, const T& x)
//        {
//            if( i == 0 )
//            {
//                return dModifiedA(j, x);
//            }
//            else if( j == 0 )
//            {
//                double result = 1.0/pow(-2.0, (double)i);
//                result *= i;
//                result *= pow((1-x), (double)(i-1));
//                return result;
//            }
//            else
//            {
//                double scale = 1.0/pow(2.0, (double)(i+1));
//                double poly = ElVis::OrthoPoly::P(j-1, 2*i-1, 1, x);
//                double dpoly = ElVis::OrthoPoly::dP(j-1, 2*i-1, 1, x);
//                double result = pow(1.0-x, (double)i) * poly -
//                    i*pow(1.0-x, (double)(i-1))*(1+x)*poly +
//                    pow(1.0-x, (double)i)*(1+x)*dpoly;
//                result *= scale;
//                return result;
//            }
//        }
//
//void RefToWorldQuad(const ElVisFloat* u0,
//    const ElVisFloat* u1, 
//    int numModes1, int numModes2,
//    const ElVisFloat2& local,
//    ElVisFloat2& global)
//{
//    global.x = MAKE_FLOAT(0.0);
//    global.y = MAKE_FLOAT(0.0);
//    
//    int idx = 0;
//    for(int i = 0; i < numModes1; ++i)
//    {
//        ElVisFloat accum[] = {MAKE_FLOAT(0.0), MAKE_FLOAT(0.0)};
//
//        for(int j = 0; j < numModes2; ++j)
//        {
//            ElVisFloat poly = ModifiedA(j, local.y);
//            accum[0] += u0[idx]*poly;
//            accum[1] += u1[idx]*poly;
//            ++idx;
//        }
//
//        ElVisFloat outerPoly = ModifiedA(i, local.x);
//        global.x += accum[0]*outerPoly;
//        global.y += accum[1]*outerPoly;
//    }
//    //global.x += ModifiedA(1, local.x) * ModifiedB(0, 1, local.y) *
//    //    u0[1];
//    //global.y += ModifiedA(1, local.x) * ModifiedB(0, 1, local.y) *
//    //    u1[1];
//}
//
// ElVisFloat RefToWorldQuad_df_dr(const ElVisFloat* u, 
//    int numModes1, int numModes2, 
//    const ElVisFloat2& local)
//{
//    //ELVIS_PRINTF("RefToWorldQuad_df_dr Modes (%d, %d)\n", numModes1, numModes2);
//    ElVisFloat result = MAKE_FLOAT(0.0);
//    int idx = 0;
//    for(int i = 0; i < numModes1; ++i)
//    {
//        ElVisFloat accum = MAKE_FLOAT(0.0);
//        for(int j = 0; j < numModes2; ++j)
//        {
//            ElVisFloat poly = ModifiedA(j, local.y);
//            accum += u[idx]*poly;
//            //ELVIS_PRINTF("RefToWorldQuad_df_dr Poly (%2.15f) u (%2.15f)\n", poly, u[idx]);
//            ++idx;
//        }
//
//        ElVisFloat outerPoly = dModifiedA(i, local.x);
//        //ELVIS_PRINTF("RefToWorldTriangle_df_dr Outer poly (%2.15f)\n", outerPoly);
//        result += accum*outerPoly;
//    }
//    //result += ModifiedAPrime(1, local.x) * ModifiedB(0, 1, local.y) *
//    //    u[1];
//    //ELVIS_PRINTF("RefToWorldQuad_df_dr Result (%2.15f)\n", result);
//    return result;
//}
//
//ElVisFloat RefToWorldQuad_df_ds(const ElVisFloat* u,
//    int numModes1, int numModes2, 
//    const ElVisFloat2& local)
//{
//    ElVisFloat result = MAKE_FLOAT(0.0);
//    int idx = 0;
//    for(int i = 0; i < numModes1; ++i)
//    {
//        ElVisFloat accum = MAKE_FLOAT(0.0);
//        for(int j = 0; j < numModes2; ++j)
//        {
//            ElVisFloat poly = dModifiedA(j, local.y);
//            accum += u[idx]*poly;
//            //ELVIS_PRINTF("RefToWorldTriangle_df_ds poly(%f) accum (%f) u(%f)\n", poly, accum, u[idx]);
//            ++idx;
//        }
//
//        ElVisFloat outerPoly = ModifiedA(i, local.x);
//        result += accum*outerPoly;
//        //ELVIS_PRINTF("RefToWorldTriangle_df_ds outerPoly(%f) result (%f)\n", outerPoly, result);
//    }
//    //result += ModifiedA(1, local.x) * ModifiedBPrime(0, 1, local.y) *
//    //    u[1];
//    return result;
//}
//
//        void RefToWorldTriangle(const Nektar::Array<Nektar::OneD, const Nektar::NekDouble>& u0,
//            const Nektar::Array<Nektar::OneD, const Nektar::NekDouble>& u1, 
//            int numModes1, int numModes2, 
//            const Nektar::Array<Nektar::OneD, const Nektar::NekDouble>& local,
//            Nektar::Array<Nektar::OneD, Nektar::NekDouble>& global)
//        {
//            global[0] = 0.0;
//            global[1] = 0.0;
//            int idx = 0;
//            for(int i = 0; i < numModes1; ++i)
//            {
//                double accum[] = {0.0, 0.0};
//
//                for(int j = 0; j < numModes2-i; ++j)
//                {
//                    double poly = ModifiedB(i, j, local[1]);
//                    accum[0] += u0[idx]*poly;
//                    accum[1] += u1[idx]*poly;
//                    ++idx;
//                }
//
//                double outerPoly = ModifiedA(i, local[0]);
//                global[0] += accum[0]*outerPoly;
//                global[1] += accum[1]*outerPoly;
//            }
//            global[0] += ModifiedA(1, local[0]) * ModifiedB(0, 1, local[1]) *
//                u0[1];
//            global[1] += ModifiedA(1, local[0]) * ModifiedB(0, 1, local[1]) *
//                u1[1];
//        }
//
//        double RefToWorldTriangle_df_dr(const Nektar::Array<Nektar::OneD, const Nektar::NekDouble>& u, 
//            int numModes1, int numModes2, 
//            const Nektar::Array<Nektar::OneD, const Nektar::NekDouble>& local)
//        {
//            double result = 0.0;
//            int idx = 0;
//            for(int i = 0; i < numModes1; ++i)
//            {
//                double accum = 0.0;
//                for(int j = 0; j < numModes2-i; ++j)
//                {
//                    double poly = ModifiedB(i, j, local[1]);
//                    accum += u[idx]*poly;
//                    ++idx;
//                }
//
//                double outerPoly = dModifiedA(i, local[0]);
//                result += accum*outerPoly;
//            }
//            result += dModifiedA(1, local[0]) * ModifiedB(0, 1, local[1]) *
//                u[1];
//            return result;
//        }
//
//        double RefToWorldTriangle_df_ds(const Nektar::Array<Nektar::OneD, const Nektar::NekDouble>& u,
//            int numModes1, int numModes2, 
//            const Nektar::Array<Nektar::OneD, const Nektar::NekDouble>& local)
//        {
//            double result = 0.0;
//            int idx = 0;
//            for(int i = 0; i < numModes1; ++i)
//            {
//                double accum = 0.0;
//                for(int j = 0; j < numModes2-i; ++j)
//                {
//                    double poly = dModifiedB(i, j, local[1]);
//                    accum += u[idx]*poly;
//                    ++idx;
//                }
//
//                double outerPoly = ModifiedA(i, local[0]);
//                result += accum*outerPoly;
//            }
//            result += ModifiedA(1, local[0]) * dModifiedB(0, 1, local[1]) *
//                u[1];
//            return result;
//        }
//
//        ElVisFloat2 NektarQuadWorldPointToReference(ElVisFloat* u0, ElVisFloat* u1,
//            uint2 modes,
//            const ElVisFloat3& intersectionPoint)
//        {
//            // Now test the inverse and make sure I can get it right.
//            // Start the search in the middle of the element.
//            ElVisFloat2 local = MakeFloat2(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
//            ElVisFloat2 curGlobalPoint = MakeFloat2(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
//
//            ElVisFloat J[4];
//            ElVisFloat2 global = MakeFloat2(intersectionPoint.x, intersectionPoint.y);
//
//            unsigned int numIterations = 0;
//            ElVisFloat tolerance = MAKE_FLOAT(1e-5);
//            do
//            {
//                //exp->GetCoord(local, curGlobalPoint);
//                RefToWorldQuad(u0, u1, modes.x, modes.y,
//                    local, curGlobalPoint);
//                ElVisFloat2 f;
//                f.x = curGlobalPoint.x-global.x;
//                f.y = curGlobalPoint.y-global.y;
//
//                J[0] = RefToWorldQuad_df_dr(u0, modes.x, modes.y, local);
//                J[1] = RefToWorldQuad_df_ds(u0, modes.x, modes.y, local);
//                J[2] = RefToWorldQuad_df_dr(u1, modes.x, modes.y, local);
//                J[3] = RefToWorldQuad_df_ds(u1, modes.x, modes.y, local);
//     
//                ElVisFloat inverse[4];
//                ElVisFloat denom = J[0]*J[3] - J[1]*J[2];
//                ElVisFloat determinant = MAKE_FLOAT(1.0)/(denom);
//                inverse[0] = determinant*J[3];
//                inverse[1] = -determinant*J[1];
//                inverse[2] = -determinant*J[2];
//                inverse[3] = determinant*J[0];
//                double r_adjust = inverse[0]*f.x + inverse[1]*f.y;
//                double s_adjust = inverse[2]*f.x + inverse[3]*f.y;
//
//                if( fabsf(r_adjust) < tolerance &&
//                    fabsf(s_adjust) < tolerance )
//                {
//                    break;
//                }
//
//                local.x -= r_adjust;
//                local.y -= s_adjust;
//
//                ++numIterations;
//            }
//            while( numIterations < 50);
//            return local;
//        }
