#include <ElVis/HighOrderIsosurfaceModel.h>
#include <ElVis/Camera.h>
#include <ElVis/ColorMap.h>
#include <ElVis/Ppm.h>
#include <ElVis/Camera.h>
#include <ElVis/CutSurfaceColorMap.h>
#include <ElVis/CutSurfaceCellContour.h>
#include <ElVis/CutSurfaceTesselator.h>
#include <Nektar/Hexahedron.h>
#include <Nektar/Tetrahedron.h>
#include <Nektar/Prism.h>
#include <ElVis/ElVisTypedefs.h>
#include <ElVis/Scene.h>
#include <ElVis/Light.h>
#include <ElVis/Color.h>
#include <ElVis/PointLight.h>
#include <ElVis/Triangle.h>
#include <ElVis/TriangleMesh.h>
#include <ElVis/LightingModule.h>
#include <ElVis/SampleTriangleMeshVertices.h>
#include <ElVis/GeometrySamplerObject.h>
#include <ElVis/ColorMapperModule.h>
#include <ElVis/SurfaceRenderer.h>
#include <ElVis/PrimaryRayModule.h>
#include <ElVis/SampleVolumeSamplerObject.h>
#include <ElVis/TriangleMeshModule.h>
#include <ElVis/SampleTriangleMeshVertices.h>
#include <ElVis/SurfaceObject.h>
#include <ElVis/CutSurfaceContourModule.h>
#include <ElVis/NektarModel.h>
#include <ElVis/WorldSpaceContourModule.h>
#include <string>
#include <boost/bind.hpp>
#include <boost/timer.hpp>
#include <ElVis/Cylinder.h>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <ElVis/Octree.hpp>
#include <ElVis/Timer.h>

#include <vtkFloatArray.h>
#include <vtkDelaunay2D.h>
#include <vtkDelaunay3D.h>
#include <vtkDataSet.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyDataMapper.h>
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridWriter.h>
#include <vtkDataSetTriangleFilter.h>
#include <vtkClipPolyData.h>
#include <vtkPlane.h>

#include <vtkSmartPointer.h>
#include <vtkPolyDataWriter.h>
#include <vtkCellArray.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkLinearSubdivisionFilter.h>
#include <vtkPolyDataWriter.h>
#include <vtkColorTransferFunction.h>

#include <fstream>

int main(int argc, char** argv)
{
    const char* nLabel = "n";
    
    boost::program_options::options_description desc("Options");
    desc.add_options()
            (nLabel, boost::program_options::value<unsigned int>(), "n")
            ;

    boost::program_options::variables_map vm;

    try
    {
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
        boost::program_options::notify(vm);
    }
    catch(std::exception& e)
    {
        std::cout << "Exception encountered parsing command line options." << std::endl;
        std::cout << e.what() << std::endl;
    }
    catch(...)
    {
        std::cout << "Exception encountered parsing command line options." << std::endl;
    }

    unsigned int n = vm[nLabel].as<unsigned int>();

    std::ofstream divergingOutFile("diverging.cmap");
    divergingOutFile << -1 << std::endl;
    divergingOutFile << 1 << std::endl;
    divergingOutFile << n << std::endl;

    vtkColorTransferFunction* tf = vtkColorTransferFunction::New();
    tf->SetColorSpaceToDiverging();
    //tf->AddRGBPoint(0.0, 59.0/255.0, 76.0/255.0, 192.0/255.0);
    //tf->AddRGBPoint(1.0, 180.0/255.0, 4.0/255.0, 38.0/255.0);
    //tf->AddRGBPoint(0.0, 0.0, 0.0, 1.0);
    //tf->AddRGBPoint(1.0, 1.0, 0.0, 0.0);
    tf->AddRGBPoint(0.0, 0.2298057, .298717966, 0.753683153);
    tf->AddRGBPoint(1.0, 0.705673158, 0.01555616, 0.150232812);
    
    tf->Build();

    for(unsigned int i = 0; i < n; ++i)
    {
        double scalar = (double)i/(n-1);
        double* rgb = tf->GetColor(scalar);
        divergingOutFile << rgb[0] << std::endl;
        divergingOutFile << rgb[1] << std::endl;
        divergingOutFile << rgb[2] << std::endl;
        divergingOutFile << 1.0 << std::endl;
    }
    divergingOutFile.close();
}

