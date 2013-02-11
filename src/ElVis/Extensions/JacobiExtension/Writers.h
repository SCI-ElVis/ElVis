////////////////////////////////////////////////////////////////////////////////
//
//  File: hoWriters.h
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

#ifndef ELVIS_JACOBI_EXTENSION__WRITERS_H_
#define ELVIS_JACOBI_EXTENSION__WRITERS_H_

#include <fstream>
#include <vector>
#include <ElVis/Core/Point.hpp>

using std::ifstream;
using std::ofstream;
using std::endl;

namespace ElVis
{
    namespace JacobiExtension
    {
        void writeUnstructuredData(const std::vector<ElVis::WorldPoint>& points, const std::vector<double>& scalars, ofstream& outFile);
        void writeStructuredData(double min_x, double min_y, double min_z, double x_h, double y_h, double z_h,
            unsigned int dim_x, unsigned int dim_y, unsigned int dim_z,
            std::vector<double> scalars, const char* fileName);
    }
}

#endif
