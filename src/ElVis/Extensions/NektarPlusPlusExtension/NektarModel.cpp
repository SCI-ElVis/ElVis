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

#include <ElVis/Extensions/NektarPlusPlusExtension/NektarModel.h>

#include <SpatialDomains/MeshGraph3D.h>
#include <LocalRegions/TriExp.h>
#include <LocalRegions/QuadExp.h>
#include <MultiRegions/ExpList3D.h>

namespace ElVis
{
namespace NektarPlusPlusExtension
{
    NektarModel::NektarModel(const std::string& modelPath) :
        Model(modelPath)
    {

    }

    NektarModel::~NektarModel()
    {
        
    }

    void NektarModel::LoadVolume(const std::string &path)
    {
        int i, j;

        boost::filesystem::path geometryFile(path + ".xml");
        boost::filesystem::path fieldFile(path + ".fld");

        if (!boost::filesystem::exists(geometryFile))
        {
            std::string message = std::string("Geometry file ")
                + geometryFile.string() + " does not exist.";
            throw std::runtime_error(message.c_str());
        }

        if (!boost::filesystem::exists(fieldFile))
        {
            std::string message = std::string("Field file ")
                + fieldFile.string() + " does not exist.";
            throw std::runtime_error(message.c_str());
        }

        // Create session reader
        char* arg1 = strdup("ElVis");
        char* arg2 = strdup(geometryFile.string().c_str());
        char* arg3 = strdup(fieldFile.string().c_str());
        char* argv[] = {arg1, arg2, arg3};
        int argc = 3;
        m_session = LibUtilities::SessionReader::CreateInstance(argc, argv);
        free(arg1);
        free(arg2);
        free(arg3);
        arg1 = arg2 = arg3 = 0;

        // Create MeshGraph object
        m_graph = SpatialDomains::MeshGraph::Read(m_session);

        if (m_graph->GetMeshDimension() != 3)
        {
            throw std::runtime_error("Only 3D supported for Nektar++");
        }

        // Set up expansion lists
        int nvariables = m_session->GetVariables().size();
        m_fields.resize(nvariables);

        for (i = 0; i < m_fields.size(); i++)
        {
            m_fields[i] = MemoryManager<MultiRegions::ExpList3D>
                ::AllocateSharedPtr(
                    m_session, m_graph, 
                    m_session->GetVariable(i));
        }

        // Load field data
        LoadFields(fieldFile);

        // Set up map which takes element global IDs to expansion index
        for (i = 0; i < m_fields[0]->GetExpSize(); ++i)
        {
            m_idToElmt[m_fields[0]->GetExp(i)->GetGeom()->GetGlobalID()] = i;
        }

        // Populate list of faces
        SpatialDomains::QuadGeomMap::const_iterator qIt;
        const SpatialDomains::QuadGeomMap &quadMap = m_graph->GetAllQuadGeoms();

        SpatialDomains::TriGeomMap::const_iterator tIt;
        const SpatialDomains::TriGeomMap &triMap = m_graph->GetAllTriGeoms();
        
        m_faces.resize(quadMap.size() + triMap.size());
        m_faceNormalFlip.resize(quadMap.size() + triMap.size());

        for (i = 0, qIt = quadMap.begin(); qIt != quadMap.end(); ++qIt)
        {
            m_faces[i++] = qIt->second;
        }

        for (tIt = triMap.begin(); tIt != triMap.end(); ++tIt)
        {
            m_faces[i++] = tIt->second;
        }

        m_fields[0]->SetUpPhysNormals();

        // Initialise with large values (fix me later)
        ElVisFloat extentMaxX = -ELVIS_FLOAT_MAX;
        ElVisFloat extentMaxY = -ELVIS_FLOAT_MAX;
        ElVisFloat extentMaxZ = -ELVIS_FLOAT_MAX;
        ElVisFloat extentMinX =  ELVIS_FLOAT_MAX;
        ElVisFloat extentMinY =  ELVIS_FLOAT_MAX;
        ElVisFloat extentMinZ =  ELVIS_FLOAT_MAX;

        set<int> seenPlanarVerts;
        
        for (i = 0; i < m_faces.size(); ++i)
        {
            SpatialDomains::Geometry2DSharedPtr face = m_faces[i];
            SpatialDomains::ElementFaceVectorSharedPtr connectedElmt =
                boost::dynamic_pointer_cast<SpatialDomains::MeshGraph3D>(
                    m_graph)->GetElementsFromFace(face);

            FaceInfo fInfo;
            fInfo.CommonElements[1] = ElementId(-1,-1);
            for (int i = 0; i < connectedElmt->size(); ++i)
            {
                fInfo.CommonElements[i] = ElementId(
                    m_idToElmt[connectedElmt->at(i)->m_Element->GetGlobalID()],
                    (int)connectedElmt->at(i)->m_Element->GetShapeType());
            }

            StdRegions::Orientation orient =
                boost::dynamic_pointer_cast<SpatialDomains::Geometry3D>(
                    connectedElmt->at(0)->m_Element)->GetForient(
                        connectedElmt->at(0)->m_FaceIndx);
            LibUtilities::ShapeType shapeType =
                boost::dynamic_pointer_cast<SpatialDomains::Geometry3D>(
                    connectedElmt->at(0)->m_Element)->GetShapeType();

            bool flipThisFace;

            if (shapeType == LibUtilities::eHexahedron)
            {
                bool flipFace[6] = {true, false, false, true, true, false};
                flipThisFace = flipFace[connectedElmt->at(0)->m_FaceIndx];

            }
            else if (shapeType == LibUtilities::ePrism)
            {
                bool flipFace[5] = {true, false, false, true, true};
                flipThisFace = flipFace[connectedElmt->at(0)->m_FaceIndx];
            }

            if (orient == StdRegions::eDir1BwdDir1_Dir2FwdDir2 ||
                orient == StdRegions::eDir1FwdDir1_Dir2BwdDir2)
            {
                flipThisFace = !flipThisFace;
            }

            m_faceNormalFlip[i] = flipThisFace;

            //fInfo.Type = eCurved;
            if (face->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed)
            {
                fInfo.Type = eCurved;
            }
            else
            {
                fInfo.Type = ePlanar;

                // Construct face normal
                m_planarFaces.push_back(face);

                const Array<Nektar::OneD, const Array<Nektar::OneD, NekDouble> >
                    &normal = m_fields[0]->GetExp(
                        m_idToElmt[
                            connectedElmt->at(0)->m_Element->GetGlobalID()])
                    ->GetFaceNormal(connectedElmt->at(0)->m_FaceIndx);

                m_planarFaceNormals.push_back(
                    WorldVector(normal[0][0], normal[1][0], normal[2][0]));

                for (j = 0; j < face->GetNumVerts(); ++j)
                {
                    int vId = face->GetVid(j);

                    if (seenPlanarVerts.count(vId) > 0)
                    {
                        continue;
                    }

                    SpatialDomains::PointGeomSharedPtr v = face->GetVertex(j);
                    m_planarVerts.push_back(
                        WorldPoint((*v)(0), (*v)(1), (*v)(2)));
                    m_planarVertIdMap[vId] = m_planarVerts.size() - 1;

                    seenPlanarVerts.insert(vId);
                }
            }

            // Calculate estimate of face extent
            LocalRegions::Expansion2DSharedPtr faceExp;
            StdRegions::StdExpansionSharedPtr xmap = face->GetXmap();
            if (face->GetShapeType() == LibUtilities::eQuadrilateral)
            {
                faceExp = MemoryManager<LocalRegions::QuadExp>::
                    AllocateSharedPtr(xmap->GetBasis(0)->GetBasisKey(),
                                      xmap->GetBasis(1)->GetBasisKey(),
                                      boost::dynamic_pointer_cast<
                                          SpatialDomains::QuadGeom>(face));
            }
            else if (face->GetShapeType() == LibUtilities::eTriangle)
            {
                faceExp = MemoryManager<LocalRegions::TriExp>::
                    AllocateSharedPtr(xmap->GetBasis(0)->GetBasisKey(),
                                      xmap->GetBasis(1)->GetBasisKey(),
                                      boost::dynamic_pointer_cast<
                                          SpatialDomains::TriGeom>(face));
            }

            const int nPts = faceExp->GetTotPoints();
            Array<Nektar::OneD, NekDouble> x(nPts), y(nPts), z(nPts);
            faceExp->GetCoords(x, y, z);

            ElVisFloat minX = x[0], minY = y[0], minZ = z[0];
            ElVisFloat maxX = x[0], maxY = y[0], maxZ = z[0];

            for (j = 1; j < nPts; ++j)
            {
                minX = std::min((ElVisFloat)x[j], minX);
                minY = std::min((ElVisFloat)y[j], minY);
                minZ = std::min((ElVisFloat)z[j], minZ);
                maxX = std::max((ElVisFloat)x[j], maxX);
                maxY = std::max((ElVisFloat)y[j], maxY);
                maxZ = std::max((ElVisFloat)z[j], maxZ);
            }

            fInfo.MinExtent.x = minX;
            fInfo.MinExtent.y = minY;
            fInfo.MinExtent.z = minZ;
            fInfo.MaxExtent.x = maxX;
            fInfo.MaxExtent.y = maxY;
            fInfo.MaxExtent.z = maxZ;

            m_faceInfo.push_back(fInfo);

            extentMinX = std::min(minX, extentMinX);
            extentMinY = std::min(minY, extentMinY);
            extentMinZ = std::min(minZ, extentMinZ);
            extentMaxX = std::max(maxX, extentMaxX);
            extentMaxY = std::max(maxY, extentMaxY);
            extentMaxZ = std::max(maxZ, extentMaxZ);
        }

        PopulateElementToFacesMap();

        m_extentMin = WorldPoint(extentMinX, extentMinY, extentMinZ);
        m_extentMax = WorldPoint(extentMaxX, extentMaxY, extentMaxZ);

        cout << "Read " << m_fields[0]->GetExpSize() << " elements" << endl;
        cout << "Extent = ["
             << extentMinX << ", " << extentMinY << ", " << extentMinZ << "] <-> ["
             << extentMaxX << ", " << extentMaxY << ", " << extentMaxZ << "]" << endl;
        cout << "Mesh has " << m_planarFaces.size() << " planar face(s)" << endl;
    }

    void NektarModel::PopulateElementToFacesMap()
    {
        unsigned int numElements = m_fields[0]->GetExpSize();
        m_elementFacesMapping = new std::vector<unsigned int>* [numElements];
        for (int i = 0; i < numElements; ++i)
        	m_elementFacesMapping[i] = new std::vector<unsigned int>;

        unsigned int matchingElement = 0;
        unsigned int numFacesToIterate = m_faces.size();
        for (unsigned int i = 0; i < numFacesToIterate; ++i)
        {
            matchingElement = NektarModel::DoGetFaceDefinition(i).CommonElements[0].Id;
            if( matchingElement < numElements )
                m_elementFacesMapping[matchingElement]->push_back(i);
        }
    }

    std::vector<unsigned int> NektarModel::DoGetFacesBelongingToElement(unsigned int elementNum) const
	{
        int numElements = m_fields[0]->GetExpSize();
        if( (numElements - 1) < elementNum )
            elementNum = numElements - 1;
        return *m_elementFacesMapping[elementNum];
	}

    void NektarModel::LoadFields(const boost::filesystem::path& fieldFile)
    {
        std::vector<std::vector<NekDouble> > fieldData;
        LibUtilities::Import(fieldFile.string(), m_fieldDef, fieldData);

        // Copy FieldData into m_fields
        for(int j = 0; j < m_fields.size(); ++j)
        {
            Vmath::Zero(m_fields[j]->GetNcoeffs(),
                        m_fields[j]->UpdateCoeffs(),1);
                
            for(int i = 0; i < m_fieldDef.size(); ++i)
            {
                ASSERTL1(m_fieldDef[i]->m_fields[j] ==
                         m_session->GetVariable(j),
                         std::string("Order of ") + fieldFile.string()
                         + std::string(" data and that defined in "
                                       "field data differs"));

                m_fields[j]->ExtractDataToCoeffs(m_fieldDef[i], fieldData[i],
                                                 m_fieldDef[i]->m_fields[j],
                                                 m_fields[j]->UpdateCoeffs());
            }
        }
    }

    int NektarModel::DoGetNumFields() const
    {
        return m_fields.size();
    }

    int NektarModel::DoGetModelDimension() const
    {
        return m_graph->GetMeshDimension();
    }

    FieldInfo NektarModel::DoGetFieldInfo(unsigned int index) const
    {
        FieldInfo returnval;
        returnval.Name = m_session->GetVariable(index);
        returnval.Id = index;
        returnval.Shortcut = m_session->GetVariable(index);
        return returnval;
    }

    int NektarModel::DoGetNumberOfBoundarySurfaces() const
    {
        return 0;
    }

    void NektarModel::DoGetBoundarySurface(
        int surfaceIndex, std::string& name, std::vector<int>& faceIds)
    {
        
    }

    void NektarModel::DoCalculateExtents(WorldPoint& min, WorldPoint& max)
    {
        min = m_extentMin;
        max = m_extentMax;
    }

    const std::string& NektarModel::DoGetPTXPrefix() const
    {
        static std::string extensionName("NektarPlusPlusExtension");
        return extensionName;
    }

    unsigned int NektarModel::DoGetNumberOfElements() const
    {
        cout << "num elements = " << m_fields[0]->GetExpSize() << endl;
        return m_fields[0]->GetExpSize();
    }

    std::vector<optixu::GeometryInstance> NektarModel::DoGet2DPrimaryGeometry(
        boost::shared_ptr<Scene> scene, optixu::Context context)
    {
        return std::vector<optixu::GeometryInstance>();
    }

    optixu::Material NektarModel::DoGet2DPrimaryGeometryMaterial(SceneView* view)
    {
        return optixu::Material();
    }

    size_t NektarModel::DoGetNumberOfFaces() const
    {
        return m_faces.size();
    }

    FaceInfo NektarModel::DoGetFaceDefinition(size_t globalFaceId) const
    {
        FaceInfo f = m_faceInfo[globalFaceId];
#if 0
        cout << "Face " << globalFaceId << ": left = "
             << f.CommonElements[0].Id << " " << f.CommonElements[0].Type << " "
             << f.CommonElements[1].Id << " " << f.CommonElements[1].Type
             << "   minext = " << f.MinExtent.x << " " << f.MinExtent.y
             << " " << f.MinExtent.z
             << "   maxext = " << f.MaxExtent.x << " " << f.MaxExtent.y << " "
             << f.MaxExtent.z << endl;
#endif
        return m_faceInfo[globalFaceId];
    }

    size_t NektarModel::DoGetNumberOfPlanarFaceVertices() const
    {
        cout << "num planar verts = " << m_planarVerts.size() << endl;
        return m_planarVerts.size();
    }

    WorldPoint NektarModel::DoGetPlanarFaceVertex(size_t vertexIdx) const
    {
        cout << "planar vert " << vertexIdx << " = " << m_planarVerts[vertexIdx] << endl;
        return m_planarVerts[vertexIdx];
    }

    size_t NektarModel::DoGetNumberOfVerticesForPlanarFace(size_t localFaceIdx) const
    {
        cout << "face " << localFaceIdx << " has " << m_planarFaces[localFaceIdx]->GetNumVerts() << " verts" << endl;
        return m_planarFaces[localFaceIdx]->GetNumVerts();
    }

    size_t NektarModel::DoGetPlanarFaceVertexIndex(size_t localFaceIdx, size_t vertexId)
    {
        cout << "face " << localFaceIdx << " vId " << vertexId << " has index " << m_planarVertIdMap[m_planarFaces[localFaceIdx]->GetVid(vertexId)] << endl;
        return m_planarVertIdMap[m_planarFaces[localFaceIdx]->GetVid(vertexId)];
    }

    WorldVector NektarModel::DoGetPlanarFaceNormal(size_t localFaceIdx) const
    {
        cout << "face " << localFaceIdx << " normal " << m_planarFaceNormals[localFaceIdx] << endl;
        return m_planarFaceNormals[localFaceIdx];
    }

    void NektarModel::DoCopyExtensionSpecificDataToOptiX(optixu::Context context)
    {
        vector<int> fieldNcoeffs(m_fields.size());
        int i, j, k, cnt, nCoeffs = 0, nVerts = 0;

        // Count number of coefficients in fields
        for (i = 0; i < m_fields.size(); ++i)
        {
            fieldNcoeffs[i] = m_fields[0]->GetNcoeffs();
            nCoeffs        += m_fields[0]->GetNcoeffs();
        }

        for (i = 0; i < m_fields[0]->GetExpSize(); ++i)
        {
            nVerts += m_fields[0]->GetExp(i)->GetNverts();
        }

        // Create buffers for OptiX
        ElVis::OptiXBuffer<ElVisFloat> solutionBuffer("SolutionBuffer");
        solutionBuffer.SetContext   (context);
        solutionBuffer.SetDimensions(nCoeffs);
        BOOST_AUTO(solution, solutionBuffer.map());

        for (cnt = i = 0; i < m_fields.size(); ++i)
        {
            for (j = 0; j < m_fields[i]->GetExpSize(); ++j)
            {
                int offset = m_fields[i]->GetCoeff_Offset(j);
                for (k = 0; k < m_fields[i]->GetExp(j)->GetNcoeffs(); ++k)
                {
                    solution[cnt++] = (ElVisFloat)m_fields[i]->GetCoeff(k + offset);
                }
            }
        }

        // Buffer for coefficient offsets
        ElVis::OptiXBuffer<int> coeffOffsetBuffer("CoeffOffsetBuffer");
        coeffOffsetBuffer.SetContext   (context);
        coeffOffsetBuffer.SetDimensions(m_fields[0]->GetExpSize());
        BOOST_AUTO(coeffOffset, coeffOffsetBuffer.map());

        // Buffer for coordinates
        ElVis::OptiXBuffer<ElVisFloat3> coordBuffer("CoordBuffer");
        coordBuffer.SetContext   (context);
        coordBuffer.SetDimensions(nVerts);
        BOOST_AUTO(coord, coordBuffer.map());

        // Buffer for coordinate offsets
        ElVis::OptiXBuffer<int> coordOffsetBuffer("CoordOffsetBuffer");
        coordOffsetBuffer.SetContext   (context);
        coordOffsetBuffer.SetDimensions(m_fields[0]->GetExpSize());
        BOOST_AUTO(coordOffset, coordOffsetBuffer.map());

        // Buffer for number of modes
        ElVis::OptiXBuffer<uint3> expNumModesBuffer("ExpNumModesBuffer");
        expNumModesBuffer.SetContext   (context);
        expNumModesBuffer.SetDimensions(m_fields[0]->GetExpSize());
        BOOST_AUTO(expNumModes, expNumModesBuffer.map());

        cnt = 0;
        for (i = 0; i < m_fields[0]->GetExpSize(); ++i)
        {
            coeffOffset[i] = m_fields[0]->GetCoeff_Offset(i);
            coordOffset[i] = cnt;

            for (j = 0; j < m_fields[0]->GetExp(i)->GetNverts(); ++j)
            {
                ElVisFloat3 tmp;
                SpatialDomains::PointGeomSharedPtr vertex =
                    m_fields[0]->GetExp(i)->GetGeom()->GetVertex(j);
                tmp.x = (*vertex)(0);
                tmp.y = (*vertex)(1);
                tmp.z = (*vertex)(2);

                coord[cnt+j] = tmp;
            }

            uint3 numModes;
            numModes.x = m_fields[0]->GetExp(i)->GetBasisNumModes(0);
            numModes.y = m_fields[0]->GetExp(i)->GetBasisNumModes(1);
            numModes.z = m_fields[0]->GetExp(i)->GetBasisNumModes(2);
            expNumModes[i] = numModes;
            
            cnt += m_fields[0]->GetExp(i)->GetNverts();
        }

        // 
        ElVis::OptiXBuffer<ElVisFloat> normalFlipBuffer("NormalFlipBuffer");
        normalFlipBuffer.SetContext   (context);
        normalFlipBuffer.SetDimensions(m_faces.size());
        BOOST_AUTO(normalFlip, normalFlipBuffer.map());

        for (i = 0; i < m_faces.size(); ++i)
        {
            normalFlip[i] = m_faceNormalFlip[i] ? (ElVisFloat) (-1.0) : (ElVisFloat) 1.0;
        }

        // Count number of curved faces and record number of coefficients.
        int nCurvedFaces = 0, nFaceCoeffs = 0;
        for (i = 0; i < m_faceInfo.size(); ++i)
        {
            if (m_faceInfo[i].Type == eCurved)
            {
                nCurvedFaces++;
                nFaceCoeffs += m_faces[i]->GetXmap()->GetNcoeffs();
            }
        }

        // Buffer for curved face coefficients
        ElVis::OptiXBuffer<ElVisFloat> faceCoeffsBuffer("FaceCoeffsBuffer");
        faceCoeffsBuffer.SetContext   (context);
        faceCoeffsBuffer.SetDimensions(nFaceCoeffs*3);
        BOOST_AUTO(faceCoeffs, faceCoeffsBuffer.map());

        ElVis::OptiXBuffer<int> faceCoeffsOffsetBuffer("FaceCoeffsOffsetBuffer");
        faceCoeffsOffsetBuffer.SetContext   (context);
        faceCoeffsOffsetBuffer.SetDimensions(nCurvedFaces);
        BOOST_AUTO(faceCoeffsOffset, faceCoeffsOffsetBuffer.map());

        ElVis::OptiXBuffer<uint2> faceNumModesBuffer("FaceNumModesBuffer");
        faceNumModesBuffer.SetContext   (context);
        faceNumModesBuffer.SetDimensions(nCurvedFaces);
        BOOST_AUTO(faceNumModes, faceNumModesBuffer.map());

        int cnt2 = 0;
        for (cnt = i = 0; i < m_faceInfo.size(); ++i)
        {
            if (m_faceInfo[i].Type != eCurved)
            {
                continue;
            }

            SpatialDomains::Geometry2DSharedPtr face = m_faces[i];
            const int nFaceCoeffs = face->GetXmap()->GetNcoeffs();

            faceNumModes[cnt2].x = m_faces[i]->GetXmap()->GetBasisNumModes(0);
            faceNumModes[cnt2].y = m_faces[i]->GetXmap()->GetBasisNumModes(1);
            faceCoeffsOffset[cnt2++] = cnt;

            for (j = 0; j < 3; ++j)
            {
                for (k = 0; k < nFaceCoeffs; ++k)
                {
                    faceCoeffs[cnt++] = (ElVisFloat)face->GetCoeffs(j)[k];
                }
            }
        }
        
        // Set in OptiX buffer.
        context["nCurvedFaces"]->setInt(nCurvedFaces);

        // Count number of deformed elements
        int nCurvedElmt = 0, nCurvedGeomCoeffs = 0;
        for (i = 0; i < m_fields[0]->GetExpSize(); ++i)
        {
            if (m_fields[0]->GetExp(i)->GetMetricInfo()->GetGtype() ==
                    SpatialDomains::eDeformed)
            {
                nCurvedElmt++;
                nCurvedGeomCoeffs += m_fields[0]->GetExp(i)->GetGeom()
                                                ->GetXmap()->GetNcoeffs() * 3;
            }
        }

        ElVis::OptiXBuffer<ElVisFloat> curvedGeomBuffer("CurvedGeomBuffer");
        curvedGeomBuffer.SetContext   (context);
        curvedGeomBuffer.SetDimensions(nCurvedGeomCoeffs);
        BOOST_AUTO(curvedGeom, curvedGeomBuffer.map());

        ElVis::OptiXBuffer<int> curvedGeomOffsetBuffer("CurvedGeomOffsetBuffer");
        curvedGeomOffsetBuffer.SetContext   (context);
        curvedGeomOffsetBuffer.SetDimensions(m_fields[0]->GetExpSize());
        BOOST_AUTO(curvedGeomOffset, curvedGeomOffsetBuffer.map());

        ElVis::OptiXBuffer<uint3> curvedGeomNumModesBuffer("CurvedGeomNumModesBuffer");
        curvedGeomNumModesBuffer.SetContext   (context);
        curvedGeomNumModesBuffer.SetDimensions(m_fields[0]->GetExpSize());
        BOOST_AUTO(curvedGeomNumModes, curvedGeomNumModesBuffer.map());

        for (cnt = i = 0; i < m_fields[0]->GetExpSize(); ++i)
        {
            SpatialDomains::GeometrySharedPtr geom =
                m_fields[0]->GetExp(i)->GetGeom();
            uint3 nm;
            nm.x = nm.y = nm.z = -1;

            if (m_fields[0]->GetExp(i)->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed)
            {
                curvedGeomOffset[i] = cnt;
                nm.x = geom->GetXmap()->GetBasisNumModes(0);
                nm.y = geom->GetXmap()->GetBasisNumModes(1);
                nm.z = geom->GetXmap()->GetBasisNumModes(2);

                for (j = 0; j < 3; ++j)
                {
                    for (k = 0; k < geom->GetXmap()->GetNcoeffs(); ++k)
                    {
                        curvedGeom[cnt++] = (ElVisFloat)geom->GetCoeffs(j)[k];
                    }
                }
            }
            else
            {
                curvedGeomOffset[i] = -1;
            }

            curvedGeomNumModes[i] = nm;
        }
    }
}
}
