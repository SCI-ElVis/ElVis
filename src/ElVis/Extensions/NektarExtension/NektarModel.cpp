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


#include <ElVis/Extensions/NektarExtension/NektarModel.h>

//#include <ElVis/Core/Util.hpp>
//#include <ElVis/Core/PtxManager.h>
//#include <ElVis/Core/Buffer.h>
//#include <ElVis/Core/Float.h>

namespace ElVis
{
    namespace NektarExtension
    {
        NektarModel::NektarModel(const std::string& fileName) :
            m_elementList(0)
        {
            FILE* inFile = fopen(fileName.c_str(), "r");
            ReadParams(inFile);
            m_elementList = ReadMesh(inFile, "volume_name");
            fclose(inFile);
        }

        NektarModel::~NektarModel()
        {
        }

        unsigned int NektarModel::DoGetNumberOfPoints() const
        {
            return 0;
            //return m_graph->GetNvertices();
        }

        WorldPoint NektarModel::DoGetPoint(unsigned int id) const
        {
            return WorldPoint();
    //        BOOST_AUTO(vertex, m_graph->GetVertex(id));
    //        ElVis::WorldPoint result(vertex->x(), vertex->y(), vertex->z());
    //        return result;
        }

        const std::string& NektarModel::DoGetPTXPrefix() const
        {
            static std::string prefix("NektarExtension");
            return prefix;
        }


    //    void NektarModel::InitializeWithGeometry(const std::string& geomFileName)
    //    {

    ////        m_graph = boost::shared_ptr<Nektar::SpatialDomains::MeshGraph3D>(new Nektar::SpatialDomains::MeshGraph3D());
    ////        m_graph->ReadGeometry(geomFileName);

    ////        CalculateExtents();
    //    }

    //    void NektarModel::InitializeWithGeometryAndField(const std::string& geomFileName,
    //        const std::string& fieldFileName)
    //    {
    //        InitializeWithGeometry(geomFileName);

    //        std::vector<Nektar::SpatialDomains::FieldDefinitionsSharedPtr> fieldDefinitions;
    //        std::vector<std::vector<double> > fieldData;

    //        m_graph->Import(fieldFileName, fieldDefinitions, fieldData);

    //        // It probably is possible to figure out which coefficients belong to which element
    //        // at this point, but it will be easier to setup the expansions in the mesh graph
    //        // and then have Nektar++ copy them over.  The SetExpansions call appears to be
    //        // incomplete, so we'll just do it ourselves.
    //        m_graph->SetExpansions(fieldDefinitions);

    ////        std::string vCommModule("Serial");
    ////        int argc = 0;
    ////        char** argv = 0;
    ////        Nektar::LibUtilities::CommSharedPtr vComm = Nektar::LibUtilities::GetCommFactory().CreateInstance(vCommModule, argc, argv);

    //        m_globalExpansion = MemoryManager<Nektar::MultiRegions::ExpList3D>
    //            ::AllocateSharedPtr(*m_graph);
    //        for(unsigned int i = 0; i < fieldDefinitions.size(); ++i)
    //        {
    //            m_globalExpansion->ExtractDataToCoeffs(fieldDefinitions[i], fieldData[i], fieldDefinitions[i]->m_fields[0]);
    //        }

    //        m_globalExpansion->BwdTrans(m_globalExpansion->GetCoeffs(), m_globalExpansion->UpdatePhys());

    //        m_globalExpansion->PutCoeffsInToElmtExp();
    //        m_globalExpansion->PutPhysInToElmtExp();
    //    }

        void NektarModel::SetupOptixCoefficientBuffers(optixu::Context context)
        {
    //        // Coefficients are stored in a global array for all element types.
    //        std::cout << "Nektar++ Coefficient Buffer Size = " << m_globalExpansion->GetNcoeffs() << std::endl;
    //        FloatingPointBuffer coefficientBuffer("Coefficients", 1);
    //        coefficientBuffer.Create(context, RT_BUFFER_INPUT, m_globalExpansion->GetNcoeffs());
    //        context[coefficientBuffer.Name().c_str()]->set(*coefficientBuffer);

    //        ElVisFloat* coeffData = static_cast<ElVisFloat*>(coefficientBuffer->map());
    //        for(unsigned int i = 0; i < m_globalExpansion->GetNcoeffs(); ++i)
    //        {
    //            coeffData[i] = m_globalExpansion->GetCoeff(i);
    //        }
    //        coefficientBuffer->unmap();

    //        optixu::Buffer CoefficientIndicesBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, m_globalExpansion->GetNumElmts());
    //        context["CoefficientIndices"]->set(CoefficientIndicesBuffer);
    //        int* coefficientIndicesData = static_cast<int*>(CoefficientIndicesBuffer->map());

    //        for(unsigned int i = 0; i < m_globalExpansion->GetNumElmts(); ++i)
    //        {
    //            coefficientIndicesData[i] = m_globalExpansion->GetCoeff_Offset(i);
    //        }
    //        CoefficientIndicesBuffer->unmap();

        }

        void NektarModel::SetupOptixVertexBuffers(optixu::Context context)
        {
    //        FloatingPointBuffer vertexBuffer("Vertices", 4);
    //        vertexBuffer.Create(context, RT_BUFFER_INPUT, m_graph->GetNvertices());
    //        context[vertexBuffer.Name().c_str()]->set(*vertexBuffer);

    //        ElVisFloat* vertexData = static_cast<ElVisFloat*>(vertexBuffer->map());
    //        for(unsigned int i = 0; i < m_graph->GetNvertices(); ++i)
    //        {
    //            BOOST_AUTO(vertex, m_graph->GetVertex(i));
    //            vertexData[i*4] = vertex->x();
    //            vertexData[i*4 + 1] = vertex->y();
    //            vertexData[i*4 + 2] = vertex->z();
    //        }
    //        vertexBuffer->unmap();
        }

        std::vector<optixu::GeometryGroup> NektarModel::DoGetCellGeometry(optixu::Context context, unsigned int elementFinderRayIndex)
        {
            std::vector<optixu::GeometryGroup> result;
            return result;
    //        const std::string hexFileName("NektarHexahedron.cu.ptx");
    //        const std::string hexClosestHitProgram("NektarEvaluateHexScalarValue");

    //        try
    //        {
    //            std::vector<optixu::GeometryGroup> result;
    //            if( !m_graph ) return result;

    //            SetupOptixCoefficientBuffers(context);
    //            SetupOptixVertexBuffers(context);

    //            std::list<optixu::GeometryInstance> m_allHexes;
    //            std::list<optixu::GeometryInstance> allPrisms;

    //            optixu::Material m_hexCutSurfaceMaterial = context->createMaterial();
    //            m_hexCutSurfaceMaterial->setClosestHitProgram(elementFinderRayIndex, PtxManager::LoadProgram(context, hexFileName, hexClosestHitProgram));

    //            //optixu::Material prismCutSurfaceMaterial = context->createMaterial();
    //            //prismCutSurfaceMaterial->setClosestHitProgram(elementFinderRayIndex, PtxManager::LoadProgram(context, PrismPtxFileName, PrismClosestHitProgramName));

    //            optixu::Program hexBoundingProgram = PtxManager::LoadProgram(context, hexFileName, "NektarHexahedronBounding");
    //            optixu::Program hexIntersectionProgram = PtxManager::LoadProgram(context, hexFileName, "NektarHexahedronContainsOriginByCheckingPointMapping");

    //            //optixu::Program prismBoundingProgram = PtxManager::LoadProgram(context, PrismPtxFileName, PrismBoundingProgramName);
    //            //optixu::Program prismIntersectionProgram = PtxManager::LoadProgram(context, PrismPtxFileName, PrismIntersectionProgramName);

    //            optixu::GeometryInstance hexInstance = CreateGeometryForElementType<Nektar::SpatialDomains::HexGeom>(context);
    //            hexInstance->setMaterialCount(1);
    //            hexInstance->setMaterial(0, m_hexCutSurfaceMaterial);

    //            optixu::Geometry hexGeometry = hexInstance->getGeometry();
    //            hexGeometry->setBoundingBoxProgram( hexBoundingProgram );
    //            hexGeometry->setIntersectionProgram( hexIntersectionProgram );

    //            //optixu::GeometryInstance prismInstance = CreateGeometryForElementType<OriginalNektar::Prism>(m_volume, context);
    //            //prismInstance->setMaterialCount(1);
    //            //prismInstance->setMaterial(0, prismCutSurfaceMaterial);

    //            //optixu::Geometry prismGeometry = prismInstance->getGeometry();
    //            //prismGeometry->setBoundingBoxProgram( prismBoundingProgram );
    //            //prismGeometry->setIntersectionProgram( prismIntersectionProgram );


    //            optixu::GeometryGroup group = context->createGeometryGroup();
    //            group->setChildCount(1);
    //            group->setChild(0, hexInstance);
    //            //group->setChild(1, prismInstance);


    //            group->setAcceleration( context->createAcceleration("Sbvh","Bvh") );
    //            result.push_back(group);

    //            return result;
    //        }
    //        catch(optixu::Exception& e)
    //        {
    //            std::cerr << e.getErrorString() << std::endl;
    //            throw;
    //        }
    //        catch(std::exception& f)
    //        {
    //            std::cerr << f.what() << std::endl;
    //            throw;
    //        }
        }

        //void NektarModel::DisplayGeometry()
        //{
        //    boost::shared_ptr<OriginalNektar::SpatialDomains::MeshGraph1D> c1 =
        //        boost::dynamic_pointer_cast<OriginalNektar::SpatialDomains::MeshGraph1D >(m_graph);
        //    boost::shared_ptr<OriginalNektar::SpatialDomains::MeshGraph2D> c2 =
        //        boost::dynamic_pointer_cast<OriginalNektar::SpatialDomains::MeshGraph2D >(m_graph);
        //    boost::shared_ptr<OriginalNektar::SpatialDomains::MeshGraph3D> c3 =
        //        boost::dynamic_pointer_cast<OriginalNektar::SpatialDomains::MeshGraph3D >(m_graph);

        //}

        void NektarModel::DoSetupCudaContext(CUmodule module) const
        {
        }

        const std::string& NektarModel::DoGetCUBinPrefix() const
        {
            return "";
        }

    }
}
