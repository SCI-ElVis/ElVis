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


#ifndef ELVIS_CORE_DYNAMIC_LIB_H
#define ELVIS_CORE_DYNAMIC_LIB_H

#include <string>
#include <boost/filesystem/operations.hpp>
#include <exception>
#include <stdexcept>

#include <ElVis/Core/ElVisDeclspec.h>

#ifdef WIN32
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

namespace ElVis
{
    class UnableToLoadDynamicLibException : public std::runtime_error
    {
        public:
            ELVIS_EXPORT UnableToLoadDynamicLibException(const std::string& error);
    };

    /// \brief A cross-platorm interface for loading shared libraries.
    class DynamicLib
    {
        public:
            /// \brief Load the dynamic library at the given location.
            /// \throw UnableToLoadDynamicLibException If the file can't be loaded.
            ELVIS_EXPORT explicit DynamicLib(const boost::filesystem::path& path);
            ELVIS_EXPORT ~DynamicLib();

            /// \brief Gets a pointer to an exported function with the given name.
            ///
            /// \param funcName The name of the function to load.
            /// \return The function pointer cast to FuncType, NULL if the function is not found.
            ///
            /// This method finds a function with the name funcName in the shared library.
            /// The intended use is to look up functions with C linkage, so there is no
            /// parameter information available.  Therefore, the user is responsible
            /// for providing this information in the template parameter.
            ///
            /// Example:
            /// DynamicLib* library;
            /// typedef const float (*FunctionPtr)(int a, float b);
            /// FunctionPtr func = library->GetFunction<FunctionPtr>("FuncName");
            template<typename FuncType>
            FuncType GetFunction(const std::string& funcName)
            {
                void* aPtr = NULL;
                 #ifdef WIN32
                        aPtr = GetProcAddress((HMODULE)m_handle, funcName.c_str());
                 #else
                        aPtr = dlsym(m_handle, funcName.c_str());
                #endif
                return (FuncType)aPtr;
            }

        private:
            // The name of the library.
            boost::filesystem::path m_libraryName;

            // A pointer to the library.  We use this
            // to gain access to the functions in the dll.
            void* m_handle;
    };
}

#endif


