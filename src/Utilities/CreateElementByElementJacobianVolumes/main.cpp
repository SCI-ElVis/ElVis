#include <ElVis/JacobiExtension/FiniteElementVolume.h>
#include <ElVis/JacobiExtension/JacobiExtensionElVisModel.h>
#include <iostream>


int main(int argc, char** argv)
{
    if( argc != 3 )
    {
        std::cerr << "<volume> <prefix>" << std::endl;
        return 1;
    }

    ElVis::JacobiExtension::JacobiExtensionModel model;
    model.LoadVolume(argv[1]);

    for(unsigned int i = 0; i < model.Volume()->numElements(); ++i)
    {
        std::cout << "Writing element " << i << " of " << model.Volume()->numElements()<< std::endl;
        model.Volume()->WriteCellVolume(i, argv[2]);
        model.Volume()->WriteCellVolumeForVTK(i, argv[2]);
    }
    return 0;
}
