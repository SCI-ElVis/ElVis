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

#ifndef ELVIS_RENDER_MODULE_H
#define ELVIS_RENDER_MODULE_H

#include <ElVis/Core/ElVisDeclspec.h>
#include <boost/signals2.hpp>
#include <bitset>

namespace ElVis
{
    class SceneView;

    enum RenderModuleFlags
    {
        eNoChangesNecessary,

        // The module should be rendered, all setup and
        // syncing with the GPU has alrady been done.
        eRenderRequired,

        // The module needs to be initialized.
        eSetupRequired,

        // some local data has been updated but not synchronized
        // with the GPU.
        eSyncRequired,

        eRenderSetupAndSyncRequired,

        eNumRenderModuleFlags
    };

    class RenderModule
    {
        public:
            ELVIS_EXPORT explicit RenderModule();
            ELVIS_EXPORT virtual ~RenderModule() {}

            /// \brief Prepares the module for rendering.  This method is only 
            /// called once and is always called before Render is called.
            ELVIS_EXPORT void Setup(SceneView* view);
            ELVIS_EXPORT void Synchronize(SceneView* view);
            ELVIS_EXPORT void Render(SceneView* view);


            ELVIS_EXPORT void Resize(unsigned int newWidth, unsigned int newHeight);
            
            ELVIS_EXPORT int GetNumberOfRequiredEntryPoints()
            {
                return DoGetNumberOfRequiredEntryPoints();
            }

            ELVIS_EXPORT bool GetEnabled() const { return m_enabled; }
            ELVIS_EXPORT void SetEnabled(bool value);
            ELVIS_EXPORT std::string GetName() const { return DoGetName(); }

            // Handlers
            ELVIS_EXPORT void SetSyncAndRenderRequired();
            ELVIS_EXPORT void SetRenderRequired();
            ELVIS_EXPORT bool GetRenderRequired() const;
            // Signals
            boost::signals2::signal<void (const RenderModule&)> OnModuleChanged;
            boost::signals2::signal<void (const RenderModule&, bool)> OnEnabledChanged;
            boost::signals2::signal<void (const RenderModule&, const std::bitset<eNumRenderModuleFlags>& )> OnRenderFlagsChanged;

        protected:
            virtual void DoSetup(SceneView* view) = 0;
            virtual void DoSynchronize(SceneView* view);
            virtual void DoRender(SceneView* view) = 0;

            virtual void DoUpdateBeforeRender(SceneView* view) {}

            virtual int DoGetNumberOfRequiredEntryPoints() = 0;
            virtual std::string DoGetName() const = 0;

            virtual void DoResize(unsigned int newWidth, unsigned int newHeight);



        private:
            RenderModule& operator=(const RenderModule& rhs);
            RenderModule(const RenderModule& rhs);

            std::bitset<eNumRenderModuleFlags> m_flags;
            bool m_enabled;
    };
}


#endif //ELVIS_RENDER_MODULE_H
