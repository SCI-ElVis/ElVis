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

#ifndef ELVIS_PRIMARY_RAY_MODULE_H
#define ELVIS_PRIMARY_RAY_MODULE_H

#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/RenderModule.h>
#include <ElVis/Core/PrimaryRayObject.h>
#include <ElVis/Core/RayGeneratorProgram.h>

#include <vector>

#include <boost/signals2.hpp>

namespace ElVis
{
    class PrimaryRayModule : public RenderModule
    {
        public:
            ELVIS_EXPORT PrimaryRayModule();
            ELVIS_EXPORT virtual ~PrimaryRayModule() {}

            ELVIS_EXPORT void AddObject(boost::shared_ptr<PrimaryRayObject> obj);
            ELVIS_EXPORT boost::shared_ptr<PrimaryRayObject> GetObject(int i) { return m_objects[i]; }
            ELVIS_EXPORT size_t NumberOfObjects() { return m_objects.size(); }

            boost::signals2::signal< void (boost::shared_ptr<PrimaryRayObject>) > OnObjectAdded;
            boost::signals2::signal< void (const PrimaryRayObject&)> OnObjectChanged;

        protected:
            ELVIS_EXPORT virtual void DoSetup(SceneView* view);
            ELVIS_EXPORT virtual void DoSynchronize(SceneView* view);
            ELVIS_EXPORT virtual void DoRender(SceneView* view); 


            virtual int DoGetNumberOfRequiredEntryPoints() { return 1; }
            virtual void DoResize(unsigned int newWidth, unsigned int newHeight) {}
            virtual std::string DoGetName() const { return "Surface Rendering"; }

        private:
            PrimaryRayModule& operator=(const PrimaryRayModule& rhs);
            PrimaryRayModule(const PrimaryRayModule& rhs);

            void HandleObjectChanged(const PrimaryRayObject&);

            std::vector<boost::shared_ptr<PrimaryRayObject> > m_objects;
            RayGeneratorProgram m_program;
            optixu::Group m_group;
    };
}


#endif 
