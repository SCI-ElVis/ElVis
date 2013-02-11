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

#ifndef ELVIS_RAY_GENERATOR_PROGRAM_H
#define ELVIS_RAY_GENERATOR_PROGRAM_H

#include <optixu/optixpp.h>

namespace ElVis
{
    class RayGeneratorProgram
    {
        public:
            ELVIS_EXPORT RayGeneratorProgram() :
                Program(),
                Index(-1)
            {
            }

            ELVIS_EXPORT RayGeneratorProgram(const RayGeneratorProgram& rhs) :
                Program(rhs.Program),
                Index(rhs.Index)
            {
            }

            ELVIS_EXPORT ~RayGeneratorProgram() {}

            const RayGeneratorProgram& operator=(const RayGeneratorProgram& rhs)
            {
                Program = rhs.Program;
                Index = rhs.Index;
                return *this;
            }

            bool IsValid() const { return Index != -1; }

            optixu::Program Program;
            int Index;

        private:
            
                        
            
    };
}

#endif 
