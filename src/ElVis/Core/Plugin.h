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

#ifndef ELVIS_CORE_PLUGIN_H
#define ELVIS_CORE_PLUGIN_H

#include <string>
#include <boost/filesystem/operations.hpp>
#include <ElVis/Core/Model.h>

#include <ElVis/Core/DynamicLib.h>
#include <ElVis/Core/ElVisDeclspec.h>

namespace ElVis
{
    class Plugin
    {
        public:
            ELVIS_EXPORT explicit Plugin(const boost::filesystem::path& pluginPath);
            ELVIS_EXPORT virtual ~Plugin();

            const std::string& GetName() const { return m_name; }
            const std::string& GetModelFileFilter() const { return m_volumeFileFilter; }
            ELVIS_EXPORT ElVis::Model* LoadModel(const std::string& name);

        private:
            Plugin(const Plugin& rhs);
            Plugin& operator=(const Plugin& rhs);

            const static std::string GetNameFunctionName;
            const static std::string LoadModelFunctionName;
            const static std::string GetVolumeFileFilterFunctionName;

            typedef std::string (*GetNameFunction)();
            typedef std::string (*GetVolumeFileFilterFunction)();
            typedef ElVis::Model* (*LoadModelFunction)(const char* path);

            boost::filesystem::path m_path;
            DynamicLib m_dynamicLib;
            std::string m_name;
            std::string m_volumeFileFilter;

            GetNameFunction m_getNameFunction;
            LoadModelFunction m_loadModelFunction;
            GetVolumeFileFilterFunction m_getVolumeFileFilterFunction;
    };
}

#endif

