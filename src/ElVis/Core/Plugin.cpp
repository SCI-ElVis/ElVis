////////////////////////////////////////////////////////////////////////////////
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

#include <ElVis/Core/Plugin.h>
#include <ElVis/Core/Util.hpp>

namespace ElVis
{
    const std::string Plugin::GetNameFunctionName("GetPluginName");
    const std::string Plugin::LoadModelFunctionName("LoadModel");
    const std::string Plugin::GetVolumeFileFilterFunctionName("GetVolumeFileFilter");

    Plugin::Plugin(const boost::filesystem::path& pluginPath) :
        m_path(pluginPath),
        m_dynamicLib(pluginPath),
        m_name(),
        m_volumeFileFilter(),
        m_getNameFunction(0),
        m_loadModelFunction(0),
        m_getVolumeFileFilterFunction(0)
    {
        m_getNameFunction = m_dynamicLib.GetFunction<GetNameFunction>(GetNameFunctionName);
        m_loadModelFunction = m_dynamicLib.GetFunction<LoadModelFunction>(LoadModelFunctionName);
        m_getVolumeFileFilterFunction = m_dynamicLib.GetFunction<GetVolumeFileFilterFunction>(GetVolumeFileFilterFunctionName);

        if( !m_getNameFunction )
        {
            throw UnableToLoadDynamicLibException("Extenstion at " + pluginPath.string() + " does not have " + GetNameFunctionName);
        }

        if( !m_loadModelFunction )
        {
            throw UnableToLoadDynamicLibException("Extenstion at " + pluginPath.string() + " does not have " + LoadModelFunctionName);
        }

        if( !m_getVolumeFileFilterFunction )
        {
            throw UnableToLoadDynamicLibException("Extenstion at " + pluginPath.string() + " does not have " + GetVolumeFileFilterFunctionName);
        }

        m_name = m_getNameFunction();
        m_volumeFileFilter = m_getVolumeFileFilterFunction();
    }

    Plugin::~Plugin()
    {
    }

    boost::shared_ptr<ElVis::Model> Plugin::LoadModel(const std::string& name)
    {
        auto rawPtr = m_loadModelFunction(name.c_str());
        rawPtr->SetPlugin(shared_from_this());

        // We want to return a shared ptr to maintain consistency across ElVis,
        // but we can't assume that the model can be deleted in the context 
        // of ElVis (for example, if the plugin is written with a different 
        // C runtime library on Windows, deleting the model in ElVis rather 
        // than the plugin will cause a crash).  
        // TODO - Require plugins to implement a cleanup function that we can 
        // then attach to the shared ptr deleter.
        return boost::shared_ptr<ElVis::Model>(rawPtr, noDelete);
    }
}

