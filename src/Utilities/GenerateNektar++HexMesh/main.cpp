
#include <boost/program_options.hpp>
#include <ElVis/Extensions/JacobiExtension/Point.hpp>
#include <ElVis/Extensions/JacobiExtension/Edge.h>
#include <ElVis/Extensions/JacobiExtension/Face.h>
#include <ElVis/Extensions/JacobiExtension/Hexahedron.h>
#include <SpatialDomains/MeshGraph3D.h>
#include <boost/bind.hpp>

struct Hex
{
    std::vector<ElVis::WorldPoint> Points;
    std::vector<SpatialDomains::VertexComponentSharedPtr> Vertices;
    std::vector<SpatialDomains::SegGeomSharedPtr> Edges;
    std::vector<SpatialDomains::QuadGeomSharedPtr> Faces;
    SpatialDomains::HexGeomSharedPtr HexGeom;

    void SetupVertices(SpatialDomains::MeshGraph3D& mesh)
    {
        for(unsigned int i = 0; i < Points.size(); ++i)
        {
            const ElVis::WorldPoint& p = Points[i];
            SpatialDomains::VertexVector::const_iterator found = 
                std::find_if(mesh.GetAllVertices().begin(), mesh.GetAllVertices().end(),
                boost::bind(&Hex::CompareVertexAndPoint, _1, p));

            Nektar::SpatialDomains::VertexComponentSharedPtr vertex;
            if( found == mesh.GetAllVertices().end() )
            {
                vertex = mesh.AddVertex(p.x(), p.y(), p.z());
            }
            else
            {
                vertex = *found;
            }

            Vertices.push_back(vertex);
        }
    }


    void SetupEdges(SpatialDomains::MeshGraph3D& mesh)
    {
        for(unsigned int i = 0; i < 12; ++i)
        {
            const ElVis::JacobiExtension::Edge& e = ElVis::JacobiExtension::Hexahedron::Edges[i];
            Nektar::SpatialDomains::VertexComponentSharedPtr vertex0 = Vertices[e.GetVertex0()];
            Nektar::SpatialDomains::VertexComponentSharedPtr vertex1 = Vertices[e.GetVertex1()];

            SpatialDomains::SegGeomVector::const_iterator found = 
                std::find_if(mesh.GetAllSegGeoms().begin(), mesh.GetAllSegGeoms().end(),
                boost::bind(&Hex::CompareSegGeomAndVertices, _1, vertex0, vertex1));

            SpatialDomains::SegGeomSharedPtr edge;
            if( found == mesh.GetAllSegGeoms().end() )
            {
                edge = mesh.AddEdge(vertex0, vertex1);
            }
            else
            {
                edge = *found;
            }
            Edges.push_back(edge);
        }
    }

    void SetupFaces(SpatialDomains::MeshGraph3D& mesh)
    {
        for(unsigned int i = 0; i < 6; ++i)
        {
            const ElVis::JacobiExtension::Face& f = ElVis::JacobiExtension::Hexahedron::Faces[i];

            Nektar::SpatialDomains::SegGeomSharedPtr faceEdges[] = {
                Edges[f.EdgeId(0)],
                Edges[f.EdgeId(1)],
                Edges[f.EdgeId(2)],
                Edges[f.EdgeId(3)]};

            Nektar::SpatialDomains::QuadGeomVector::const_iterator found =
                std::find_if(mesh.GetAllQuadGeoms().begin(),
                mesh.GetAllQuadGeoms().end(),
                boost::bind(&Hex::CompareQuadAndEdges, _1, faceEdges[0], faceEdges[1], faceEdges[2], faceEdges[3]));

            Nektar::SpatialDomains::QuadGeomSharedPtr quad;
            if( found == mesh.GetAllQuadGeoms().end() )
            {
                Nektar::StdRegions::EdgeOrientation edgeOrientation[4];
                for(unsigned int i = 0; i < 4 - 1; ++i)
                {
                    edgeOrientation[i] = Nektar::SpatialDomains::SegGeom::GetEdgeOrientation(*(faceEdges[i]), *(faceEdges[i+1]));
                }
                edgeOrientation[4-1] = Nektar::SpatialDomains::SegGeom::GetEdgeOrientation(*(faceEdges[4-1]), *(faceEdges[0]));

                quad = mesh.AddQuadrilateral(faceEdges, edgeOrientation);
            }
            else
            {
                quad = *found;
            }

            Faces.push_back(quad);
        }
    }

    void SetupElement(SpatialDomains::MeshGraph3D& mesh)
    {
        SpatialDomains::QuadGeomSharedPtr qfaces[] = 
        {
            Faces[0], Faces[1], Faces[2], Faces[3], Faces[4], Faces[5] 
        };

        HexGeom = mesh.AddHexahedron(qfaces);
    }

    static bool CompareQuadAndEdges(Nektar::SpatialDomains::QuadGeomSharedPtr quad,
        Nektar::SpatialDomains::SegGeomSharedPtr e0,
        Nektar::SpatialDomains::SegGeomSharedPtr e1, 
        Nektar::SpatialDomains::SegGeomSharedPtr e2, 
        Nektar::SpatialDomains::SegGeomSharedPtr e3)
    {
        unsigned int quadIds[] = {quad->GetEid(0), quad->GetEid(1), quad->GetEid(2), quad->GetEid(3) };
        unsigned int edges[] = {e0->GetEid(), e1->GetEid(), e2->GetEid(), e3->GetEid() };

        std::sort(quadIds, quadIds+4);
        std::sort(edges, edges+4);

        for(unsigned int i = 0; i < 4; ++i)
        {
            if( quadIds[i] != edges[i] ) return false;
        }
        return true;
    }

    static bool CompareSegGeomAndVertices(Nektar::SpatialDomains::SegGeomSharedPtr edge,
        Nektar::SpatialDomains::VertexComponentSharedPtr v0,
        Nektar::SpatialDomains::VertexComponentSharedPtr v1)
    {
        bool test1 = edge->GetVertex(0)->GetVid() == v0->GetVid() &&
            edge->GetVertex(1)->GetVid() == v1->GetVid();
        bool test2 = edge->GetVertex(0)->GetVid() == v1->GetVid() &&
            edge->GetVertex(1)->GetVid() == v0->GetVid();
        return test1 || test2;
    }

    static bool CompareVertexAndPoint(Nektar::SpatialDomains::VertexComponentSharedPtr vertex,
        const ElVis::WorldPoint& p)
    {
        return vertex->x() == p.x() &&
            vertex->y() == p.y() &&
            vertex->z() == p.z();
    }
};

int main(int argc, char** argv)
{
    const char* outFileLabel = "Out";
    const char* nxLabel = "nx";
    const char* nyLabel = "ny";
    const char* nzLabel = "nz";

    boost::program_options::options_description desc("Options");
    desc.add_options()
        (outFileLabel, boost::program_options::value<std::string>(), "Output file name.")
        (nxLabel, boost::program_options::value<unsigned int>(), "Number of hexes in the x direction.")
        (nyLabel, boost::program_options::value<unsigned int>(), "Number of hexes in the y direction.")
        (nzLabel, boost::program_options::value<unsigned int>(), "Number of hexes in the z direction.")
        ;

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    unsigned int nx = vm[nxLabel].as<unsigned int>();
    unsigned int ny = vm[nyLabel].as<unsigned int>();
    unsigned int nz = vm[nzLabel].as<unsigned int>();
    std::string outputFileName = vm[outFileLabel].as<std::string>();
    Nektar::SpatialDomains::MeshGraph3D mesh;

    double hx = 2.0/nx;
    double hy = 2.0/ny;
    double hz = 2.0/nz;

    Nektar::SpatialDomains::Composite composite(Nektar::MemoryManager<Nektar::SpatialDomains::GeometryVector>::AllocateSharedPtr());
    std::list<Hex*> allHexes;

    double basex = -1.0;
    double basey = -1.0;
    double basez = -4.0;
    unsigned int count = 0;
    for(unsigned int i = 0; i < nx; ++i)
    {
        for(unsigned int j = 0; j < ny; ++j)
        {
            std::cout << "Completed " << (double)count/((nx-1)*(ny-1)*(nz-1)) << std::endl;
            for(unsigned int k = 0; k < nz; ++k)
            {
                Hex* h = new Hex();
                h->Points.push_back(ElVis::WorldPoint(basex+(i+0)*hx, basey+(j+0)*hy, basez+(k+0)*hz));
                h->Points.push_back(ElVis::WorldPoint(basex+(i+1)*hx, basey+(j+0)*hy, basez+(k+0)*hz));
                h->Points.push_back(ElVis::WorldPoint(basex+(i+1)*hx, basey+(j+1)*hy, basez+(k+0)*hz));
                h->Points.push_back(ElVis::WorldPoint(basex+(i+0)*hx, basey+(j+1)*hy, basez+(k+0)*hz));

                h->Points.push_back(ElVis::WorldPoint(basex+(i+0)*hx, basey+(j+0)*hy, basez+(k+1)*hz));
                h->Points.push_back(ElVis::WorldPoint(basex+(i+1)*hx, basey+(j+0)*hy, basez+(k+1)*hz));
                h->Points.push_back(ElVis::WorldPoint(basex+(i+1)*hx, basey+(j+1)*hy, basez+(k+1)*hz));
                h->Points.push_back(ElVis::WorldPoint(basex+(i+0)*hx, basey+(j+1)*hy, basez+(k+1)*hz));
                allHexes.push_back(h);
                h->SetupVertices(mesh);
                h->SetupEdges(mesh);
                h->SetupFaces(mesh);
                h->SetupElement(mesh);
                composite->push_back(h->HexGeom);
		delete h;
		++count;
            }
        }
    }

    mesh.AddComposite(composite);
    mesh.AddDomainComposite(composite);
    

    mesh.WriteGeometry(outputFileName);

    return 0;
}
