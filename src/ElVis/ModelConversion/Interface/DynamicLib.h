////////////////////////////////////////////////////////////////////////////////
//
//  File: evDynamicLib.h
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


#ifndef ELVIS_DYNAMIC_LIB_H
#define ELVIS_DYNAMIC_LIB_H

#include <string>
#include <boost/filesystem/operations.hpp>
#include <exception>
#include <stdexcept>

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
            UnableToLoadDynamicLibException(const std::string& error);
    };

    class DynamicLib
    {
        public:
            /// \brief Load the dynamic library at the given location.
            /// \throw std::runtime_error if something goes wrong trying to get the function.
            explicit DynamicLib(const boost::filesystem::path& path);
            ~DynamicLib();

            /// \brief Gets a pointer to an exported function with the given name.
            /// returns NULL if the function can't be found.
            ///
            /// Obtains a function pointer for the given function.  Note that there
            /// is no type information (number and type of parameters), so the return
            /// value must be cast to a function pointer of the appropriate type.
            void* GetFunction(const std::string& funcName);

        private:
            boost::filesystem::path m_name;
            void* m_handle;
    };
}

#endif

