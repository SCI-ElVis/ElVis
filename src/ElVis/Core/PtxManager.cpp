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

#include <ElVis/Core/PtxManager.h>
#include <ElVis/Core/ElVisConfig.h>
#include <ElVis/Core/Timer.h>

#include <iostream>

namespace ElVis
{
    std::map<std::string, PtxManager::Data> PtxManager::AllPrograms;

    optixu::Program PtxManager::LoadProgram(optixu::Context context, const std::string& prefix, const std::string& programName)
    {
        try
        {
            std::string pathToPtx = GetPtxPath() + "/" +
                    prefix + "_generated_ElVisOptiX.cu.ptx";
            std::cout << "Loading program " << programName << " from PTX File: " << pathToPtx << std::endl;
            Timer timer;
            timer.Start();
            optixu::Program result =  context->createProgramFromPTXFile(pathToPtx.c_str(), programName.c_str());
            timer.Stop();
            std::cout << "Time = " << timer.TimePerTest(1) << std::endl;
            return result;
        }
        catch(optix::Exception& e)
        {
            std::cout << "Exception loading OptiX program." << std::endl;
            std::cerr << e.getErrorString() << std::endl;
            std::cout << e.getErrorString().c_str() << std::endl;
            throw;
        }
        catch(...)
        {
            std::cout << "Exception loading file." << std::endl;
        }
    }

    optixu::Program PtxManager::LoadProgram(const std::string& prefix, const std::string& programName)
    {
        // We can only do this if we have a context for this ptx file.
        std::map<std::string, Data>::iterator found = AllPrograms.find(prefix);
        if( found != AllPrograms.end() )
        {
            optixu::Context context = (*found).second.Context;
            optixu::Program p = LoadProgram(context, prefix, programName);
            (*found).second.Programs[programName] = p;
            return p;
        }
        else
        {
            std::string error = "Error loading program at " + prefix + " with name " + programName;
            std::cout << error << std::endl;
            throw std::runtime_error(error);
        }
    }

    optixu::Program PtxManager::GetProgram(const std::string& prefix, const std::string& programName)
    {
        std::map<std::string, Data>::const_iterator found = AllPrograms.find(prefix);
        if( found != AllPrograms.end() )
        {
            ProgramMap::const_iterator innerFound = (*found).second.Programs.find(programName);
            if( innerFound != (*found).second.Programs.end() )
            {
                return (*innerFound).second;
            }
            else
            {
                std::string error = "Error loading program at " + prefix + " with name " + programName;
                std::cout << error << std::endl;
                throw std::runtime_error(error);
            }
        }
        else
        {
            std::string error = "Error loading program at " + prefix + " with name " + programName;
            std::cout << error << std::endl;
            throw std::runtime_error(error);
        }
    }

    void PtxManager::SetupContext(const std::string& prefix, optixu::Context context)
    {
        std::map<std::string, Data>::iterator found = AllPrograms.find(prefix);
        if( found == AllPrograms.end() )
        {
            Data data;
            data.Context = context;
            std::map<std::string, Data>::value_type obj(prefix, data);
            AllPrograms.insert(obj);
            GetOnPtxLoaded()(prefix, context);
        }

    }

    boost::signals2::signal<void (const std::string&, optixu::Context)>& PtxManager::GetOnPtxLoaded()
    {
        static boost::signals2::signal<void (const std::string&, optixu::Context)> result;
        return result;
    }

}
