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


#include <ElVis/Core/SampleOntoNrrdModule.h>
#include <ElVis/Core/SceneView.h>
#include <ElVis/Core/OptiXBuffer.hpp>

#include <fstream>
#include <boost/typeof/typeof.hpp>

namespace ElVis
{
    RayGeneratorProgram SampleOntoNrrd::Program;

    SampleOntoNrrd::SampleOntoNrrd()
    {
    }


    void SampleOntoNrrd::Sample(boost::shared_ptr<SceneView> view, const std::string& filePrefix, ElVisFloat targeth)
    {
        if( !Program.IsValid() )
        {
            Program = view->AddRayGenerationProgram("SampleOntoNrrd");
        }

        ElVisFloat3 h = MakeFloat3(targeth, targeth, targeth);
        uint3 n;

        view->GetScene()->GetModel()->CalculateExtents();
        WorldPoint minExtent = view->GetScene()->GetModel()->MinExtent();
        WorldPoint maxExtent = view->GetScene()->GetModel()->MaxExtent();
        ElVisFloat3 d = MakeFloat3(maxExtent.x() - minExtent.x(), maxExtent.y() - minExtent.y(), maxExtent.z() - minExtent.z());

        n.x = d.x/h.x + 1;
        n.y = d.y/h.y + 1;
        n.z = d.z/h.z + 1;

        h.x = d.x/(n.x-1);
        h.y = d.y/(n.y-1);
        h.z = d.z/(n.z-1);

        std::cout << "Samples (" << h.x << ", " << h.y << ", " << h.z << ")" << std::endl;
        std::cout << "Samples per direction (" << n.x << ", " << n.y << ", " << n.z << ")" << std::endl;

        std::string nrrdHeaderFileName = filePrefix + ".nhdr";
        std::string nrrdDataFileName = filePrefix + ".raw";

        boost::filesystem::path p(nrrdDataFileName);

        std::ofstream nrrdHeaderFile(nrrdHeaderFileName.c_str(), std::ios::out);
        FILE* nrrdDataFile = fopen(nrrdDataFileName.c_str(), "wb");

        nrrdHeaderFile << "NRRD0001" << std::endl;
        nrrdHeaderFile << "type: float" << std::endl;
        nrrdHeaderFile << "dimension: 3" << std::endl;
        nrrdHeaderFile << "sizes: " << n.x << " " << n.y << " " << n.z << std::endl;
        nrrdHeaderFile << "encoding: raw" << std::endl;
        nrrdHeaderFile << "datafile: ./" << p.filename().string() << std::endl;
        nrrdHeaderFile << "endian: little" << std::endl;
        nrrdHeaderFile << "space dimension: 3" << std::endl;
        nrrdHeaderFile << "space directions: " << "(" << h.x << ",0,0) " << "(0," << h.y << ",0) " << "(0,0," << h.z << ")" << std::endl;
        nrrdHeaderFile << "space origin: (" << minExtent.x() << ", " << minExtent.y() << ", " << minExtent.z() << ")" << std::endl;
        nrrdHeaderFile << std::endl;
        nrrdHeaderFile.close();

        optixu::Context context = view->GetContext();
        SetFloat(context["SampleOntoNrrdH"], h);

        int bufferSize = n.x*n.y;
        OptiXBuffer<ElVisFloat> sampleBuffer("SampleOntoNrrdSamples");
        sampleBuffer.SetContext(context);
        sampleBuffer.SetDimensions(n.x, n.y);

        float* convertBuffer = new float[n.x*n.y];
        SetFloat(context["SampleOntoNrrdMinExtent"], minExtent);
        SetFloat(context["SampleOntoNrrdMissValue"], std::numeric_limits<ElVisFloat>::signaling_NaN());
        std::cout << "Validating and compiling." << std::endl;
        context->validate();
        context->compile();
        std::cout << "Done validating and compiling." << std::endl;


        try
        {

            for(unsigned int i = 0; i < n.z; ++i)
            {
                context["SampleOntoNrrdPlane"]->setInt(i);
                std::cout << "Sampling " << i << " of " << n.z-1 << std::endl;
                context->launch(Program.Index, n.x, n.y);
                std::cout << "Done sampling." << std::endl;

                BOOST_AUTO(data, sampleBuffer.Map());
                for(unsigned int i = 0; i < n.x*n.y; ++i)
                {
                    convertBuffer[i] = data[i];
                }

                fwrite(convertBuffer, sizeof(float), bufferSize, nrrdDataFile);
            }
        }
        catch(optixu::Exception& e)
        {
            std::cout << "Exception encountered rendering primary rays." << std::endl;
            std::cerr << e.getErrorString() << std::endl;
            std::cout << e.getErrorString().c_str() << std::endl;
        }
        catch(std::exception& e)
        {
            std::cout << "Exception encountered rendering primary rays." << std::endl;
            std::cout << e.what() << std::endl;
        }
        catch(...)
        {
            std::cout << "Exception encountered rendering primary rays." << std::endl;
        }

        delete [] convertBuffer;
        fclose(nrrdDataFile);
    }

}

