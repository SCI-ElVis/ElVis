
#include <iostream>
#include <set>
#include <map>
#include <iostream>

#include <SpatialDomains/MeshGraph3D.h>
#include <MultiRegions/ContField3D.h>

#include <ElVis/ModelConversion/Interface/DynamicLib.h>
#include <ElVis/ModelConversion/Interface/IElVisModelConverter.h>
#include <ElVis/Core/Point.hpp>

#include <boost/program_options.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <boost/bind.hpp>
#include <boost/bind.hpp>

void PrintUsage()
{
    std::cerr << "ElVisModelConverter --Module <SharedLibrary> --InFile <file> --OutFile <file>" << std::endl;
    exit(1);
}

template<typename F>
void AddComposites(Nektar::SpatialDomains::MeshGraph3D& convertedMesh, const F& f)
{
    Nektar::SpatialDomains::Composite composite(Nektar::MemoryManager<Nektar::SpatialDomains::GeometryVector>::AllocateSharedPtr());
    for(unsigned int i = 0; i < f().size(); ++i)
    {
        composite->push_back(f()[i]);
    }
    if( composite->size() > 0 )
    {
        convertedMesh.AddComposite(composite);
        convertedMesh.AddDomainComposite(composite);
    }
}


// Converts the Nektar formatted data used in my Master's thesis to the new 
// Nektar++ format.
int main(int argc, char** argv)
{
    try
    {
        const char* inFileLabel = "InFile";
        const char* outFileLabel = "OutFile";
        const char* moduleLabel = "Module";

        boost::program_options::options_description desc("Options");
        desc.add_options()
	        (inFileLabel, boost::program_options::value<std::string>(), "Input Nektar file.")
	        (outFileLabel, boost::program_options::value<std::string>(), "Output Nektar++.")
            (moduleLabel, boost::program_options::value<std::string>(), "Module.")
	        ;

        boost::program_options::variables_map vm;
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
        boost::program_options::notify(vm);

        if( vm.count(inFileLabel) == 0 ) PrintUsage();
        if( vm.count(outFileLabel) == 0 ) PrintUsage();
        if( vm.count(moduleLabel) == 0 ) PrintUsage();

        std::string inFileName = vm[inFileLabel].as<std::string>();
        std::string outFileName = vm[outFileLabel].as<std::string>();
        std::string moduleName = vm[moduleLabel].as<std::string>();

        std::string meshFileName = outFileName + ".mesh";
        std::string fieldFileName = outFileName + ".fld";

        boost::filesystem::path moduleFile(moduleName);
        boost::filesystem::path inFile(inFileName);
        boost::filesystem::path outFile(outFileName);

        if( !boost::filesystem::exists(inFile) )
        {
            std::cerr << "The file " << inFile << " does not exist." << std::endl;
            return 1;
        }
        
        if( !boost::filesystem::exists(moduleFile) )
        {
            std::cerr << "The module " << moduleFile << " does not exist." << std::endl;
            return 1;
        }

        ElVis::DynamicLib lib(moduleFile);

        // The module must provide a function called "CreateConverter" that creates
        // and returns a pointer to an IModelConverter.  
        typedef ElVis::IModelConverter* (*CreateFunc)();
        CreateFunc f = (CreateFunc)lib.GetFunction("CreateConverter");
        if( f == 0 )
        {
            std::cerr << "The module " << moduleFile << " does not contain a CreateConverter function." << std::endl;
            return 1;
        }

        std::cout << "Loading volume " << inFileName << std::endl;
        ElVis::IModelConverter* converter = f();
        converter->SetInputFileName(inFileName);
        
        // Create a 3D Nektar++ mesh that lives in 3D space.
        Nektar::SpatialDomains::MeshGraph3D convertedMesh;
        
        std::cout << "Creating vertices." << std::endl;
        std::map<ElVis::WorldPoint, Nektar::SpatialDomains::VertexComponentSharedPtr> pointVertexMap;
        for(unsigned int i = 0; i < converter->GetNumberOfVertices(); ++i)
        {
            double x, y, z;
            converter->GetVertex(i, x, y, z);
            Nektar::SpatialDomains::VertexComponentSharedPtr vertex = convertedMesh.AddVertex(x, y, z);

            ElVis::WorldPoint p(x, y, z);
            pointVertexMap[p] = vertex;
            
        }
        
        std::cout << "Creating edges." << std::endl;
        for(unsigned int i = 0; i < converter->GetNumberOfEdges(); ++i)
        {
            unsigned int v0Id, v1Id;
            converter->GetEdge(i, v0Id, v1Id);
            Nektar::SpatialDomains::SegGeomSharedPtr segGeom = 
                    convertedMesh.AddEdge(convertedMesh.GetVertex(v0Id), 
                    convertedMesh.GetVertex(v1Id));
        }

        std::cout << "Creating triangular faces." << std::endl;
        for(unsigned int i = 0; i < converter->GetNumberOfTriangularFaces(); ++i)
        {
            const unsigned int numEdges = 3;
            unsigned int edgeIds[numEdges];
            converter->GetTriangleFaceEdgeIds(i, edgeIds);

            Nektar::SpatialDomains::SegGeomSharedPtr edges[numEdges];
            for(unsigned int i = 0; i < numEdges; ++i)
            {
                edges[i] = convertedMesh.GetEdge(edgeIds[i]);
            }

            Nektar::StdRegions::Orientation edgeOrientation[numEdges];
            for(unsigned int i = 0; i < numEdges - 1; ++i)
            {
                edgeOrientation[i] = Nektar::SpatialDomains::SegGeom::GetEdgeOrientation(*(edges[i]), *(edges[i+1]));
            }
            edgeOrientation[numEdges-1] = Nektar::SpatialDomains::SegGeom::GetEdgeOrientation(*(edges[numEdges-1]), *(edges[0]));

            convertedMesh.AddTriangle(edges, edgeOrientation);
        }

	    std::cout << "Creating quad faces." << std::endl;
        for(unsigned int i = 0; i < converter->GetNumberOfQuadrilateralFaces(); ++i)
        {
            const unsigned int numEdges = 4;
            unsigned int edgeIds[numEdges];
            converter->GetQuadrilateralFaceEdgeIds(i, edgeIds);

            Nektar::SpatialDomains::SegGeomSharedPtr edges[numEdges];
            for(unsigned int i = 0; i < numEdges; ++i)
            {
                edges[i] = convertedMesh.GetEdge(edgeIds[i]);
            }

            Nektar::StdRegions::Orientation edgeOrientation[numEdges];
            for(unsigned int i = 0; i < numEdges - 1; ++i)
            {
                edgeOrientation[i] = Nektar::SpatialDomains::SegGeom::GetEdgeOrientation(*(edges[i]), *(edges[i+1]));
            }
            edgeOrientation[numEdges-1] = Nektar::SpatialDomains::SegGeom::GetEdgeOrientation(*(edges[numEdges-1]), *(edges[0]));

            convertedMesh.AddQuadrilateral(edges, edgeOrientation);
        }


	    std::cout << "Creating hexes." << std::endl;
        for(unsigned int i = 0; i < converter->GetNumberOfHexahedra(); ++i)
        {
            unsigned int faceIds[Nektar::SpatialDomains::HexGeom::kNqfaces];
            converter->GetHexahedronFaceIds(i, faceIds);
            Nektar::SpatialDomains::QuadGeomSharedPtr qfaces[Nektar::SpatialDomains::HexGeom::kNqfaces];

            for(unsigned int j = 0; j < Nektar::SpatialDomains::HexGeom::kNqfaces; ++j)
            {
                qfaces[j] = (*convertedMesh.GetAllQuadGeoms().find(faceIds[j])).second;
                    //convertedMesh.GetAllQuadGeoms()[faceIds[j]];
            }

            SpatialDomains::HexGeomSharedPtr hex = convertedMesh.AddHexahedron(qfaces);

            unsigned int degree[3];

            converter->GetHexahedronDegree(i, degree);
            Nektar::LibUtilities::BasisKey key0(Nektar::LibUtilities::eModified_A, degree[0]+1,
                Nektar::LibUtilities::PointsKey(degree[0]+1, Nektar::LibUtilities::eGaussLobattoLegendre));
            Nektar::LibUtilities::BasisKey key1(Nektar::LibUtilities::eModified_A, degree[1]+1,
                Nektar::LibUtilities::PointsKey(degree[1]+1, Nektar::LibUtilities::eGaussLobattoLegendre));
            Nektar::LibUtilities::BasisKey key2(Nektar::LibUtilities::eModified_A, degree[2]+1,
                Nektar::LibUtilities::PointsKey(degree[2]+1, Nektar::LibUtilities::eGaussLobattoLegendre));

            LibUtilities::BasisKeyVector basisKeyVector;
            basisKeyVector.push_back(key0);
            basisKeyVector.push_back(key1);
            basisKeyVector.push_back(key2);
            SpatialDomains::ExpansionShPtr expansion = MemoryManager<SpatialDomains::Expansion>::AllocateSharedPtr
                (hex, basisKeyVector);
            convertedMesh.AddExpansion(expansion);
        }

	    std::cout << "Creating prisms." << std::endl;
        for(unsigned int i = 0; i < converter->GetNumberOfPrisms(); ++i)
        {
            unsigned int quadFaceIds[Nektar::SpatialDomains::PrismGeom::kNqfaces];
            unsigned int triFaceIds[Nektar::SpatialDomains::PrismGeom::kNtfaces];

            converter->GetPrismQuadFaceIds(i, quadFaceIds);
            converter->GetPrismTriangleFaceIds(i, triFaceIds);

            Nektar::SpatialDomains::QuadGeomSharedPtr qfaces[Nektar::SpatialDomains::PrismGeom::kNqfaces];
            Nektar::SpatialDomains::TriGeomSharedPtr tfaces[Nektar::SpatialDomains::PrismGeom::kNtfaces];

            for(unsigned int j = 0; j < Nektar::SpatialDomains::PrismGeom::kNqfaces; ++j)
            {
                qfaces[j] = (*convertedMesh.GetAllQuadGeoms().find(quadFaceIds[j])).second;
                    //convertedMesh.GetAllQuadGeoms()[quadFaceIds[j]];
            }

            for(unsigned int j = 0; j < Nektar::SpatialDomains::PrismGeom::kNtfaces; ++j)
            {
                tfaces[j] = (*convertedMesh.GetAllTriGeoms().find(triFaceIds[j])).second;
                    //convertedMesh.GetAllTriGeoms()[triFaceIds[j]];
            }

            SpatialDomains::PrismGeomSharedPtr prism = convertedMesh.AddPrism(tfaces, qfaces);

            unsigned int degree[3];
            converter->GetPrismDegree(i, degree);
            Nektar::LibUtilities::BasisKey key0(Nektar::LibUtilities::eModified_A, degree[0]+1,
                Nektar::LibUtilities::PointsKey(degree[0]+1, Nektar::LibUtilities::eGaussLobattoLegendre));
            Nektar::LibUtilities::BasisKey key1(Nektar::LibUtilities::eModified_A, degree[1]+1,
                Nektar::LibUtilities::PointsKey(degree[1]+1, Nektar::LibUtilities::eGaussLobattoLegendre));
            Nektar::LibUtilities::BasisKey key2(Nektar::LibUtilities::eModified_A, degree[2]+1,
                Nektar::LibUtilities::PointsKey(degree[2]+1, Nektar::LibUtilities::eGaussLobattoLegendre));

            LibUtilities::BasisKeyVector basisKeyVector;
            basisKeyVector.push_back(key0);
            basisKeyVector.push_back(key1);
            basisKeyVector.push_back(key2);
            SpatialDomains::ExpansionShPtr expansion = MemoryManager<SpatialDomains::Expansion>::AllocateSharedPtr
                (prism, basisKeyVector);
            convertedMesh.AddExpansion(expansion);

        }

        AddComposites(convertedMesh, boost::bind(&Nektar::SpatialDomains::MeshGraph::GetAllHexGeoms, convertedMesh));
        AddComposites(convertedMesh, boost::bind(&Nektar::SpatialDomains::MeshGraph::GetAllPrismGeoms, convertedMesh));
        AddComposites(convertedMesh, boost::bind(&Nektar::SpatialDomains::MeshGraph::GetAllPyrGeoms, convertedMesh));
        AddComposites(convertedMesh, boost::bind(&Nektar::SpatialDomains::MeshGraph::GetAllTetGeoms, convertedMesh));
        
        std::cout << "Creating global expansion." << std::endl;
        bool useGlobalContinuousExpansion = false;
        Nektar::MultiRegions::ExpList3DSharedPtr Exp;

        if( useGlobalContinuousExpansion )
        {
            Exp = MemoryManager<Nektar::MultiRegions::ContField3D>::AllocateSharedPtr(convertedMesh);
        }
        else
        {
            Exp = MemoryManager<Nektar::MultiRegions::ExpList3D>::AllocateSharedPtr(convertedMesh);
        }
        
        


        if( !useGlobalContinuousExpansion )
        {
            double max = -std::numeric_limits<double>::max();
            double min = std::numeric_limits<double>::max();
            
            std::cout << "Starting projection" << std::endl;
            for(unsigned int i = 0; i < Exp->GetExpSize(); ++i)
            {
                BOOST_AUTO(expansion, Exp->GetExp(i));
                BOOST_AUTO(geom, expansion->GetGeom());

                int id = expansion->GetGeom()->GetGlobalID();
                Array<OneD, NekDouble> c0 = Array<OneD, NekDouble>(expansion->GetTotPoints());
                Array<OneD, NekDouble> c1 = Array<OneD, NekDouble>(expansion->GetTotPoints());
                Array<OneD, NekDouble> c2 = Array<OneD, NekDouble>(expansion->GetTotPoints());
                expansion->GetCoords(c0, c1, c2);

                const Array<OneD, const NekDouble> d0 = expansion->GetPoints(0);
                const Array<OneD, const NekDouble> d1 = expansion->GetPoints(1);
                const Array<OneD, const NekDouble> d2 = expansion->GetPoints(2);

                Nektar::Array<OneD, double> f(expansion->GetTotPoints());

                for(int j = 0; j < expansion->GetTotPoints(); ++j)
                {
                    f[j] = converter->CalculateScalarValue(c0[j], c1[j], c2[j], id);
                }

                std::cout << "FwdTrans for element " << i << std::endl;
                expansion->FwdTrans(f, expansion->UpdateCoeffs());

                // To call PhysEvaluate, m_phys must be populated.
                expansion->BwdTrans(expansion->GetCoeffs(), expansion->UpdatePhys());

                // Double check
                //Nektar::Array<OneD, double> worldPointArray(3);
                //worldPointArray[0] = c0[0];
                //worldPointArray[1] = c1[0];
                //worldPointArray[2] = c2[0];


                //double a0 = expansion->PhysEvaluate(worldPointArray);
                //double a1 = converter->CalculateScalarValue(worldPointArray[0], worldPointArray[1], worldPointArray[2], id);


                //double h = .1;
                //unsigned int n = 2.0/h;
                //for(unsigned int i = 0; i <= n; ++i)
                //{
                //    for(unsigned int j = 0; j <= n; ++j)
                //    {
                //        for(unsigned int k = 0; k <= n; ++k)
                //        {
                //            Nektar::Array<OneD, double> localArray(3);
                //            localArray[0] = -1.0 + i*h;;
                //            localArray[1] = -1.0 + j*h;
                //            localArray[2] = -1.0 + k*h;;

                //            expansion->GetCoord(localArray, worldPointArray);

                //            double t0 = expansion->PhysEvaluate(worldPointArray);
                //            double t1 = converter->CalculateScalarValue(worldPointArray[0], worldPointArray[1], worldPointArray[2], id);
                //            
                //            double t2 = TestHexExpansion(expansion->GetCoeffs().data(),
                //                localArray[0], localArray[1], localArray[2],
                //                expansion->GetBasis(0)->GetNumModes(),
                //                expansion->GetBasis(1)->GetNumModes(),
                //                expansion->GetBasis(2)->GetNumModes());

                //            
                //            std::cout << "Converter = " << t1 << std::endl;
                //            std::cout << "Expansion = " << t0 << std::endl;
                //            std::cout << "Test Expansion = " << t2 << std::endl;
                //        }
                //    }
                //}
            }

            Exp->PutElmtExpInToCoeffs();
        }
        else
        {
            std::cout << "Sampling" << std::endl;
            Array<OneD, NekDouble> c0 = Array<OneD, NekDouble>(Exp->GetTotPoints());
            Array<OneD, NekDouble> c1 = Array<OneD, NekDouble>(Exp->GetTotPoints());
            Array<OneD, NekDouble> c2 = Array<OneD, NekDouble>(Exp->GetTotPoints());
            Exp->GetCoords(c0, c1, c2);

            Array<OneD, NekDouble> f(Exp->GetTotPoints());
            for(int i = 0; i < Exp->GetTotPoints(); ++i)
            {
                f[i] = converter->CalculateScalarValue(c0[i], c1[i], c2[i]);
            }
            std::cout << "FwdTrans" << std::endl;
            Exp->FwdTrans(f, Exp->UpdateCoeffs());

            // To call PhysEvaluate, m_phys must be populated.
            //Exp->BwdTrans(Exp->GetCoeffs(), Exp->UpdatePhys());
        }

        convertedMesh.WriteGeometry(meshFileName);

        std::vector<SpatialDomains::FieldDefinitionsSharedPtr> FieldDef = Exp->GetFieldDefinitions();
        std::vector<std::vector<NekDouble> > FieldData(FieldDef.size());
        for(unsigned int i = 0; i < FieldDef.size(); ++i)
        {
            FieldDef[i]->m_fields.push_back("p");
            Exp->AppendFieldData(FieldDef[i], FieldData[i]);
        }
        convertedMesh.Write(fieldFileName.c_str(), FieldDef, FieldData);

        //std::vector<Nektar::SpatialDomains::FieldDefinitionsSharedPtr> checkFieldDefinitions;
        //std::vector<std::vector<double> > checkFieldData;

        //convertedMesh.Import(fieldFileName.c_str(), checkFieldDefinitions, checkFieldData);

    }
    catch(std::exception& e)
    {
        std::cerr << "Exception encountered " << e.what() << std::endl;
        return 1;
    }
    catch(std::string& e)
    {
        std::cerr << "Exception encountered " << e << std::endl;
        return 1;
    }


    return 0;
}
