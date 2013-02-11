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

#include <fstream>
#include <iostream>
#include <string>

struct G7K15NodesAndWeights
{
    static const double Nodes[];

    static const double GWeights[];

    static const double KWeights[];

    static const std::string Name;
};

const double G7K15NodesAndWeights::Nodes[] =
{
    -0.9914553711208126392068547,
    -0.9491079123427585245261897,
    -0.8648644233597690727897128,
    -0.7415311855993944398638648,
    -0.5860872354676911302941448,
    -0.4058451513773971669066064,
    -0.2077849550078984676006894,
    0.0000000000000000000000000,
    0.2077849550078984676006894,
    0.4058451513773971669066064,
    0.5860872354676911302941448,
    0.7415311855993944398638648,
    0.8648644233597690727897128,
    0.9491079123427585245261897,
    0.9914553711208126392068547,
};

const double G7K15NodesAndWeights::GWeights[] =
{
    0.0,
    0.1294849661688696932706114,
    0.0,
    0.2797053914892766679014678,
    0.0,
    0.3818300505051189449503698,
    0.0,
    0.4179591836734693877551020,
    0.0,
    0.3818300505051189449503698,
    0.0,
    0.2797053914892766679014678,
    0.0,
    0.1294849661688696932706114,
    0.0
};

const double G7K15NodesAndWeights::KWeights[] =
{
    0.0229353220105292249637320,
    0.0630920926299785532907007,
    0.1047900103222501838398763,
    0.1406532597155259187451896,
    0.1690047266392679028265834,
    0.1903505780647854099132564,
    0.2044329400752988924141620,
    0.2094821410847278280129992,
    0.2044329400752988924141620,
    0.1903505780647854099132564,
    0.1690047266392679028265834,
    0.1406532597155259187451896,
    0.1047900103222501838398763,
    0.0630920926299785532907007,
    0.0229353220105292249637320
};

const std::string G7K15NodesAndWeights::Name("G7K15");




void SampleField(double node, double gweight, double kweight, std::ostream& outFile)
{
    outFile << "        {" << std::endl;
    outFile << "            ElVisFloat t = (b-a)*MAKE_FLOAT(.5)*MAKE_FLOAT(" << node << ") + (b+a)*MAKE_FLOAT(0.5);" << std::endl;
    outFile << "            ElVisFloat s = f(t);" << std::endl;
    outFile << "            ResultType transferSample = integrand(t, s);" << std::endl;
    if( gweight != 0.0 )
    {
        outFile << "            gresult += MAKE_FLOAT(" << gweight << ") * transferSample; " << std::endl;
    }

    outFile << "            kresult += MAKE_FLOAT(" << kweight << ") * transferSample; " << std::endl;
    outFile << "        }" << std::endl;
}

//        ElVisFloat Nodes[15];
//        ElVisFloat KWeights[15];
//        ElVisFloat GWeights[15];

//        #pragma unroll
//        for(int i = 0; i < 15; ++i)
//        {
//            ElVisFloat t = (b-a)*MAKE_FLOAT(.5)*Nodes[i] + (b+a)*MAKE_FLOAT(0.5);
//            ElVisFloat s = f(t);
//            ResultType transferSample = integrand(t, s);
//            kresult += KWeights[i] * transferSample;
//            gresult += GWeights[i] * transferSample;

void PrintArray(const double* values, int size, std::ofstream& outFile)
{
    outFile << "{" << std::endl;
    for(unsigned int i = 0; i < size; ++i)
    {
        if( i > 0 )
        {
            outFile << "," << std::endl;
        }
        outFile << "            " << values[i];
    }
    outFile << std::endl;
    outFile << "        };" << std::endl;
}

template<typename Method>
void GenerateSpecificSingleThreadedGaussKronrod(std::ofstream& outFile)
{
    outFile << "__device__ void PrintResultType(ElVisFloat f)" << std::endl;
    outFile << "{" << std::endl;
    outFile << "    printf(\"%2.15f\", f);" << std::endl;
    outFile << "}" << std::endl;
    outFile << std::endl;
    outFile << "__device__ void PrintResultType(ElVisFloat3 f)" << std::endl;
    outFile << "{" << std::endl;
    outFile << "    printf(\"(%2.15f, %2.15f, %2.15f)\", f.x, f.y, f.z);" << std::endl;
    outFile << "}" << std::endl;
    outFile << std::endl;

    outFile << "template<>" << std::endl;
    outFile << "struct SingleThreadGaussKronrod<" << Method::Name << ">" << std::endl;
    outFile << "{" << std::endl;
    outFile << "    template<typename ResultType, typename IntegrandType>" << std::endl;
    outFile << "    ELVIS_DEVICE static ResultType Integrate(const IntegrandType& integrand, const ElVisFloat& a, const ElVisFloat& b, const FieldEvaluator& f, ResultType& errorEstimate, bool traceEnabled=false)" << std::endl;
    outFile << "    {" << std::endl;
    outFile << "        if( traceEnabled )" << std::endl;
    outFile << "        {" << std::endl;
    outFile << "            printf(\"Starting GK Quadrature on interval [%f, %f]\\n\", a, b);" << std::endl;
    outFile << "        }" << std::endl;
    outFile << "        ResultType gresult = DefaultFloat<ResultType>::GetValue();" << std::endl;
    outFile << "        ResultType kresult = DefaultFloat<ResultType>::GetValue();" << std::endl;
    outFile << "        ElVisFloat nodes[" << sizeof(Method::Nodes)/sizeof(double) << "] = "; PrintArray(Method::Nodes, sizeof(Method::Nodes)/sizeof(double), outFile);
    outFile << "        ElVisFloat gweights[" << sizeof(Method::GWeights)/sizeof(double) << "] = "; PrintArray(Method::GWeights, sizeof(Method::GWeights)/sizeof(double), outFile);
    outFile << "        ElVisFloat kweights[" << sizeof(Method::KWeights)/sizeof(double) << "] = "; PrintArray(Method::KWeights, sizeof(Method::KWeights)/sizeof(double), outFile);
    outFile << std::endl;

    outFile << "        for(int i = 0; i < " << sizeof(Method::Nodes)/sizeof(double) << "; ++i)" << std::endl;
    outFile << "        {" << std::endl;
    outFile << "            ElVisFloat t = (b-a)*MAKE_FLOAT(.5)*nodes[i] + (b+a)*MAKE_FLOAT(0.5);" << std::endl;
    outFile << "            ElVisFloat s = f(t);" << std::endl;
    outFile << "            ResultType transferSample = integrand(t, s, traceEnabled);" << std::endl;
    outFile << "            if( traceEnabled )" << std::endl;
    outFile << "            {" << std::endl;
    outFile << "                printf(\"GK sample at t = %2.15f, s = %2.15f, transferSample = \", t, s);" << std::endl;
    outFile << "                PrintResultType(transferSample);" << std::endl;
    outFile << "                printf(\"\\n\");" << std::endl;
    outFile << "            }" << std::endl;
    outFile << "            kresult += kweights[i] * transferSample;" << std::endl;
    outFile << "            gresult += gweights[i] * transferSample;" << std::endl;
    outFile << "            if( traceEnabled )" << std::endl;
    outFile << "            {" << std::endl;
    outFile << "                printf(\"KResult = \");" << std::endl;
    outFile << "                PrintResultType(kresult);" << std::endl;
    outFile << "                printf(\"\\n\");" << std::endl;
    outFile << "            }" << std::endl;
    outFile << "        }" << std::endl;


    outFile << "        gresult *= (b-a)*MAKE_FLOAT(.5);" << std::endl;
    outFile << "        kresult *= (b-a)*MAKE_FLOAT(.5);" << std::endl;
    outFile << "        errorEstimate = Fabsf(gresult-kresult);" << std::endl;
    outFile << "        if( traceEnabled )" << std::endl;
    outFile << "        {" << std::endl;
    outFile << "            printf(\"Final KResult = \");" << std::endl;
    outFile << "            PrintResultType(kresult);" << std::endl;
    outFile << "            printf(\"\\n\");" << std::endl;
    outFile << "        }" << std::endl;
    outFile << "        return kresult;" << std::endl;
    outFile << "    }" << std::endl;
    outFile << "};" << std::endl;
}

void GenerateSingleThreadedGaussKronrod(std::ofstream& outFile)
{
    outFile << "enum GaussKronrodType" << std::endl;
    outFile << "{" << std::endl;
    outFile << "    G7K15" << std::endl;
    outFile << "};" << std::endl << std::endl;

    outFile << "template<GaussKronrodType T>" << std::endl;
    outFile << "struct SingleThreadGaussKronrod;" << std::endl;
    outFile << std::endl << std::endl;
    GenerateSpecificSingleThreadedGaussKronrod<G7K15NodesAndWeights>(outFile);
}

int main()
{
    std::ofstream outFile("GaussKronrod.cu");

    outFile << "///////////////////////////////////////////////////////////////////////////////" << std::endl;
    outFile << "//" << std::endl;
    outFile << "// The MIT License" << std::endl;
    outFile << "//" << std::endl;
    outFile << "// Copyright (c) 2006 Scientific Computing and Imaging Institute," << std::endl;
    outFile << "// University of Utah (USA)" << std::endl;
    outFile << "//" << std::endl;
    outFile << "// License for the specific language governing rights and limitations under" << std::endl;
    outFile << "// Permission is hereby granted, free of charge, to any person obtaining a" << std::endl;
    outFile << "// copy of this software and associated documentation files (the \"Software\")," << std::endl;
    outFile << "// to deal in the Software without restriction, including without limitation" << std::endl;
    outFile << "// the rights to use, copy, modify, merge, publish, distribute, sublicense," << std::endl;
    outFile << "// and/or sell copies of the Software, and to permit persons to whom the" << std::endl;
    outFile << "// Software is furnished to do so, subject to the following conditions:" << std::endl;
    outFile << "//" << std::endl;
    outFile << "// The above copyright notice and this permission notice shall be included" << std::endl;
    outFile << "// in all copies or substantial portions of the Software." << std::endl;
    outFile << "//" << std::endl;
    outFile << "// THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS" << std::endl;
    outFile << "// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY," << std::endl;
    outFile << "// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL" << std::endl;
    outFile << "// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER" << std::endl;
    outFile << "// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING" << std::endl;
    outFile << "// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER" << std::endl;
    outFile << "// DEALINGS IN THE SOFTWARE." << std::endl;
    outFile << "//" << std::endl;
    outFile << "///////////////////////////////////////////////////////////////////////////////" << std::endl;
    outFile << std::endl << std::endl;
    outFile << "// Do not modify this file, it has been autogenerated." << std::endl;
    outFile << std::endl << std::endl;
    outFile << "#include <ElVis/Core/Float.cu>" << std::endl;
    outFile << "#include <ElVis/Core/FieldEvaluator.cu>" << std::endl;
    outFile << "#include <ElVis/Core/TransferFunction.h>" << std::endl;
    outFile << std::endl << std::endl;

    outFile.precision(20);
    outFile.setf(std::ios::fixed);
    outFile << "namespace ElVis {" << std::endl;
    GenerateSingleThreadedGaussKronrod(outFile);
    outFile << "}" << std::endl;
    outFile.close();
    return 0;
}
