////////////////////////////////////////////////////////////////////////////////
//
//  File: hoWriters.cpp
//
//
//  The MIT License
//
//  Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
//  Department of Aeronautics, Imperial College London (UK), and Scientific
//  Computing and Imaging Institute, University of Utah (USA).
//
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
//  Description:
//
////////////////////////////////////////////////////////////////////////////////
#include <ElVis/Extensions/JacobiExtension/Isosurface.h>
#include <ElVis/Extensions/JacobiExtension/Writers.h>
#include <iostream>
using namespace std;

namespace ElVis
{
    namespace JacobiExtension
    {
        void writeUnstructuredData(const std::vector<ElVis::WorldPoint>& points, const std::vector<double>& scalars, ofstream& outFile)
        {
            if( scalars.size() <= 0 ||
                points.size() <= 0 )
            {
                cerr << "Invalid sizes." << endl;
                return;
            }

            if( scalars.size() != points.size() )
            {
                cerr << "sizes don't match." << endl;
                return;
            }

            outFile << "# vtk DataFile Version 2.0" << endl;
            outFile << "Samples." << endl;
            outFile << "ASCII" << endl;
            outFile << "DATASET UNSTRUCTURED_GRID" << endl;
            outFile << "POINTS " << static_cast<unsigned int>(points.size()) << " float" << endl;

            for(unsigned int i = 0; i < points.size(); ++i)
            {
                outFile << points[i].x() << " "
                    << points[i].y() << " "
                    << points[i].z() << endl;
            }

            outFile << "CELLS " << static_cast<unsigned int>(points.size()) << " " << static_cast<unsigned int>(points.size()*2) << endl;
            for(unsigned int j = 0; j < points.size(); ++j)
            {
                outFile << "1 " << j << endl;
            }

            outFile << "CELL_TYPES " << static_cast<unsigned int>(points.size()) << endl;
            for(unsigned int k = 0; k < points.size(); ++k)
            {
                outFile << 1 << endl;
            }

            outFile << "POINT_DATA " << static_cast<unsigned int>(points.size()) << endl;
            outFile << "SCALARS scalars float 1" << endl;
            outFile << "LOOKUP_TABLE default" << endl;

            for(unsigned int l = 0; l < scalars.size(); ++l)
            {
                outFile << scalars[l] << endl;
            }
        }

        void writeStructuredData(double min_x, double min_y, double min_z, double x_h, double y_h, double z_h,
            unsigned int dim_x, unsigned int dim_y, unsigned int dim_z,
            std::vector<double> scalars, const char* fileName)
        {
            ofstream outFile(fileName, ios::out);
            if( scalars.size() != dim_x*dim_y*dim_z )
            {
                cerr << "mismatching dimensions." << endl;
                return;
            }

            outFile << "# vtk DataFile Version 2.0" << endl;
            outFile << "Samples" << endl;
            outFile << "ASCII" << endl;
            outFile << "DATASET STRUCTURED_POINTS" << endl;
            outFile << "DIMENSIONS " << dim_x << " " << dim_y << " " << dim_z << endl;
            outFile << "ORIGIN " << min_x << " " << min_y << " " << min_z << endl;
            outFile << "SPACING " << x_h << " " << y_h << " " << z_h << endl;
            outFile << "POINT_DATA " << dim_x*dim_y*dim_z << endl;
            outFile << "SCALARS values float" << endl;
            outFile << "LOOKUP_TABLE default" << endl;

            for(unsigned int i = 0; i < scalars.size(); ++i)
            {
                outFile << scalars[i] << endl;
            }
            outFile.close();
        }
    }
}
