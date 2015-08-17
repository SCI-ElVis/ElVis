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

#include <ElVis/Core/DynamicLib.h>
#include <iostream>
#include <sstream>
#include <string>
#include <stdexcept>

using std::cerr;
using std::cout;
using std::endl;

namespace ElVis
{

  UnableToLoadDynamicLibException::UnableToLoadDynamicLibException(
    const std::string& error)
    : std::runtime_error(error)
  {
  }

  DynamicLib::DynamicLib(const boost::filesystem::path& path)
    : m_libraryName(path), m_handle(0)
  {

    std::string actualDllName(path.string());
    boost::filesystem::path realPath(actualDllName);

    boost::filesystem::path currentDirectory(boost::filesystem::initial_path());
    std::string fullPath =
      boost::filesystem::system_complete(realPath).string();

#ifdef WIN32

    // First, try to get the handle without loading the library.
    // If we can get a handle then it is already loaded.
    HMODULE mod = GetModuleHandle(fullPath.c_str());
    if (mod)
    {
      m_handle = mod;
      return;
    }

    m_handle = LoadLibrary(fullPath.c_str());
    if (!m_handle)
    {
      std::stringstream str;
      str << "ERROR loading dll " << fullPath << " ";
      switch (GetLastError())
      {
        case ERROR_MOD_NOT_FOUND:
          str << "The specified DLL could not be found ";
          str << "(" << (long)ERROR_MOD_NOT_FOUND << ").";
          str << "A common reason for this is if the plugin has an implicit ";
          str << "DLL requirement that can't be found.";
          break;

        case ERROR_INVALID_MODULETYPE:
          str << "The operating system cannot run the specified module ";
          str << "(" << (long)ERROR_INVALID_MODULETYPE << ").";
          break;

        case ERROR_TOO_MANY_MODULES:
          str << "Too many dynamic-link modules are attached to this program "
                 "or dynamic-link module ";
          str << "(" << (long)ERROR_TOO_MANY_MODULES << ").";
          break;

        case ERROR_DLL_INIT_FAILED:
          str << "Dynamic link library (DLL) initialization routine failed ";
          str << "(" << (long)ERROR_DLL_INIT_FAILED << ").";
          break;

        case ERROR_DLL_NOT_FOUND:
          str << "The library file cannot be found ";
          str << "(" << (long)ERROR_DLL_NOT_FOUND << ").";
          break;

        case ERROR_INVALID_DLL:
          str << "The library file is damaged ";
          str << "(" << (long)ERROR_INVALID_DLL << ").";
          break;

        default:
          str << "\tERROR NUMBER = " << GetLastError();
          break;
      }
      throw UnableToLoadDynamicLibException(str.str());
    }
#else

    m_handle = dlopen(actualDllName.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (!m_handle)
    {
      char* error = dlerror();
      std::string strError(error);
      throw UnableToLoadDynamicLibException(strError);
    }
#endif
  }

  DynamicLib::~DynamicLib()
  {
#ifdef WIN32
    FreeLibrary((HMODULE)m_handle);
#else
    dlclose(m_handle);
#endif
  }
}
