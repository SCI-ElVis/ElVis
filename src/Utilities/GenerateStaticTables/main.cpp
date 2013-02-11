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

#include <vector>
#include <fstream>
#include <math.h>

std::vector<std::vector<double> > GenerateMonomialCoefficients(unsigned int order)
{
    /// \todo I could build a static table upt to a certain
    /// order if this proves to be a bottleneck.
    std::vector<std::vector<double> > coeffTable;
    coeffTable.resize(order+1);
    for(unsigned int i = 0; i < order+1; ++i)
    {
        coeffTable[i].assign(order+1, 0.0);
    }

    coeffTable[0][0] = 1.0;
    coeffTable[1][0] = 0.0;
    coeffTable[1][1] = 1.0;
    for(unsigned int i = 2; i < order+1; ++i)
    {
        double n = i-1;
        coeffTable[i][0] = -n/(n+1.0) * coeffTable[i-2][0];
        for(unsigned int j = 1; j < order+1; ++j)
        {
            coeffTable[i][j] = (2.0*n+1.0)/(n+1.0)*coeffTable[i-1][j-1] -
                n/(n+1.0)*coeffTable[i-2][j];
        }
    }

    // Now that the table has been generated, apply the factor which
    // makes it orthogonal.
    for(unsigned int i = 0; i <= order; ++i)
    {
        double factor = sqrt((2.0*static_cast<double>(i)+1.0)/2.0);
        for(unsigned int j = 0; j <= order; ++j)
        {
            coeffTable[i][j] *= factor;
        }
    }

    return coeffTable;
}


int main()
{
    // Generate the monomial coefficient tables used to convert legendre polynomials to
    // monomials.
    std::ofstream outFile("MonomialConversionTables.txt");

    for(unsigned int i = 1; i < 40; ++i)
    {
        std::vector<std::vector<double> > table = GenerateMonomialCoefficients(i);

        for(unsigned int j = 0; j < table.size(); ++j)
        {
            std::vector<double>& row = table[j];
            for(unsigned int k = 0; k < row.size(); ++k)
            {
                outFile.precision(20);
                outFile << row[k] << std::endl;
            }
        }
    }

    outFile.close();
}
