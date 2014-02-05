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


#include <ElVis/Core/PrimaryRayModule.h>
#include <ElVis/Core/SceneView.h>
#include <ElVis/Core/PtxManager.h>
#include <boost/timer.hpp>
#include <iostream>
#include <ElVis/Core/Util.hpp>
#include <stdio.h>

#include <boost/bind.hpp>

namespace ElVis
{
    PrimaryRayModule::PrimaryRayModule() :
        OnObjectAdded(),
        m_objects(),
        m_program(),
        m_group()
    {
        m_program.Index = -1;
    }
    
    void PrimaryRayModule::AddObject(boost::shared_ptr<PrimaryRayObject> obj) 
    { 
        m_objects.push_back(obj); 
        obj->OnObjectChanged.connect(boost::bind(&PrimaryRayModule::HandleObjectChanged, this, _1));
        SetSyncAndRenderRequired();
        OnObjectAdded(obj); 
        OnModuleChanged(*this);
    }

    void PrimaryRayModule::DoSetup(SceneView* view)
    {
        try
        {
            optixu::Context context = view->GetContext();

            m_group = context->createGroup();
            m_group->setAcceleration( context->createAcceleration("NoAccel","NoAccel") );
            context["SurfaceGeometryGroup"]->set( m_group );

            m_program = view->AddRayGenerationProgram("GeneratePrimaryRays");
            context->setMissProgram(0, PtxManager::LoadProgram(context, view->GetPTXPrefix(), "PrimaryRayMissed"));
        }
        catch(optixu::Exception& e)
        {
            std::cout << "Exception encountered rendering primary rays." << std::endl;
            std::cerr << e.getErrorString() << std::endl;
            std::cout << e.getErrorString().c_str() << std::endl;
        }
        catch(std::exception& e)
        {
            std::cout << "Exception encountered rendering primary rays." << std::endl;
            std::cout << e.what() << std::endl;
        }
        catch(...)
        {
            std::cout << "Exception encountered rendering primary rays." << std::endl;
        }
    }

    void PrimaryRayModule::DoSynchronize(SceneView* view)
    {
        try
        {
            optixu::Context context = view->GetContext();
            assert(context);

            int numSurfaces = m_objects.size();

            assert(m_group);
            m_group->setChildCount(numSurfaces);

            int curChild = 0;
            for(std::vector<boost::shared_ptr<PrimaryRayObject> >::const_iterator iter = m_objects.begin(); iter != m_objects.end(); ++iter)
            {
                optixu::Transform transform;
                optixu::GeometryGroup geometryGroup;

                (*iter)->CreateNode(view, transform, geometryGroup);

                bool test = true;
                if( transform.get() && test )
                {
                    m_group->setChild(curChild, transform);
                    ++curChild;
                }
                else if ( geometryGroup.get() && test )
                {
                    m_group->setChild(curChild, geometryGroup);
                    ++curChild;
                }
                else
                {
                    --numSurfaces;
                    m_group->setChildCount(numSurfaces);
                }


                
            }

            m_group->getAcceleration()->markDirty();
        }
        catch(optixu::Exception& e)
        {
            std::cout << "Exception encountered rendering primary rays." << std::endl;
            std::cerr << e.getErrorString() << std::endl;
            std::cout << e.getErrorString().c_str() << std::endl;
        }
        catch(std::exception& e)
        {
            std::cout << "Exception encountered rendering primary rays." << std::endl;
            std::cout << e.what() << std::endl;
        }
        catch(...)
        {
            std::cout << "Exception encountered rendering primary rays." << std::endl;
        }
    }

    void PrimaryRayModule::DoRender(SceneView* view)
    {
        try
        {      
            optixu::Context context = view->GetContext();
            context->launch(m_program.Index, view->GetWidth(), view->GetHeight());
        }
        catch(optixu::Exception& e)
        {
            std::cout << "Exception encountered rendering primary rays." << std::endl;
            std::cerr << e.getErrorString() << std::endl;
            std::cout << e.getErrorString().c_str() << std::endl;
        }
        catch(std::exception& e)
        {
            std::cout << "Exception encountered rendering primary rays." << std::endl;
            std::cout << e.what() << std::endl;
        }
        catch(...)
        {
            std::cout << "Exception encountered rendering primary rays." << std::endl;
        }
    }



    void PrimaryRayModule::HandleObjectChanged(const PrimaryRayObject& obj)
    {
        OnObjectChanged(obj);
        OnModuleChanged(*this);
        SetSyncAndRenderRequired();
    }
            
}

