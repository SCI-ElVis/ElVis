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

#include <ElVis/Core/RenderModule.h>
#include <string>

namespace ElVis
{
    RenderModule::RenderModule() :
        m_flags(),
        m_enabled(true)
    {
        m_flags.set(eSetupRequired);
        m_flags.set(eRenderRequired);
        m_flags.set(eSyncRequired);
    }

    /// \brief Prepares the module for rendering.  This method is only
    /// called once and is always called before Render is called.
    void RenderModule::Setup(SceneView* view)
    {
        if( m_flags.test(eSetupRequired) )
        {
            DoSetup(view);
            m_flags.reset(eSetupRequired);
        }
    }

    void RenderModule::Render(SceneView* view)
    {
        Setup(view);
        Synchronize(view);

        if( m_flags.test(eRenderRequired) )
        {
            DoRender(view);
            m_flags.reset(eRenderRequired);
        }
    }

    void RenderModule::Synchronize(SceneView* view)
    {
        if( m_flags.test(eSyncRequired) )
        {
            DoSynchronize(view);
            m_flags.reset(eSyncRequired);
        }
    }

    void RenderModule::SetEnabled(bool value)
    {
        if( value != m_enabled )
        {
            m_enabled = value;

            if( m_enabled )
            {
                // We expect all modules, even if they are disabled, to keep track
                // of if they need to be setup or synchronized.
                m_flags.set(eRenderRequired);
            }
            else
            {
                // Disabling the module turns off the rendering, but we still need
                // to keep track of outstanding synchronization.
                m_flags.reset(eRenderRequired);
            }

            OnRenderFlagsChanged(*this, m_flags);
            OnModuleChanged(*this);
            OnEnabledChanged(*this,value);
        }
    }


    void RenderModule::SetSyncAndRenderRequired()
    {
        if( !m_flags.test(eRenderRequired)  ||
            !m_flags.test(eSyncRequired) )
        {
            m_flags.set(eRenderRequired);
            m_flags.set(eSyncRequired);
            OnRenderFlagsChanged(*this, m_flags);
        }
    }

    void RenderModule::SetRenderRequired()
    {
        // Don't change the flag if it is setup and render required.
        if( !m_flags.test(eRenderRequired) )
        {
            m_flags.set(eRenderRequired);
            OnRenderFlagsChanged(*this, m_flags);
        }
    }

    bool RenderModule::GetRenderRequired() const
    {
        return m_flags.test(eRenderRequired);
    }
    void RenderModule::DoSynchronize(SceneView* view)
    {
    }

    void RenderModule::DoResize(unsigned int newWidth, unsigned int newHeight)
    {
    }

    void RenderModule::Resize(unsigned int newWidth, unsigned int newHeight)
    {
        DoResize(newWidth, newHeight);
    }

}
