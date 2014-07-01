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

#ifndef PTX_MANAGER_H
#define PTX_MANAGER_H


#include <optixu/optixpp.h>
#include <ElVis/Core/ElVisDeclspec.h>
#include <string>
#include <map>
#include <boost/signals2.hpp>

namespace ElVis
{
    class PtxManager
    {
        public:
            ELVIS_EXPORT static optixu::Program LoadProgram(optixu::Context context, const std::string& prefix, const std::string& programName);


            // Loads a program.
            ELVIS_EXPORT static optixu::Program LoadProgram(const std::string& prefix, const std::string& programName);

            // Gets a perviously loaded program.
            ELVIS_EXPORT static optixu::Program GetProgram(const std::string& prefix, const std::string& programName);

            ELVIS_EXPORT static void SetupContext(const std::string& prefix, optixu::Context context);

            // When a model is loaded, we find the Ptx file for the contect, create a context,
            // then ask people to add their programs.
            ELVIS_EXPORT static boost::signals2::signal<void (const std::string&, optixu::Context)>& GetOnPtxLoaded();

        private:
            typedef std::map<std::string, optixu::Program> ProgramMap;

            struct Data
            {
                public:
                    Data() :
                        Programs(),
                        Context() {}

                    Data(const Data& rhs) :
                        Programs(rhs.Programs),
                        Context(rhs.Context) {}

                    ProgramMap Programs;
                    optixu::Context Context;
                private:
                    Data& operator=(const Data&);
            };


            static std::map<std::string, Data> AllPrograms;


    };
}

#endif
