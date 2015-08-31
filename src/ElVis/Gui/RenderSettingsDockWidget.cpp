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

#include <ElVis/Gui/RenderSettingsDockWidget.h>
#include <ElVis/Gui/ApplicationState.h>

#include <ElVis/Core/RenderModule.h>

#include <ElVis/QtPropertyBrowser/qtgroupboxpropertybrowser.h>
#include <ElVis/QtPropertyBrowser/qttreepropertybrowser.h>

#include <boost/shared_ptr.hpp>
#include <boost/foreach.hpp>
#include <boost/thread.hpp>

#include <QDockWidget>
#include <QGridLayout>

namespace ElVis
{
    namespace Gui
    {
        RenderSettingsDockWidget::RenderSettingsDockWidget(boost::shared_ptr<ApplicationState> appData) :
            QDockWidget("Rendering Settings"),
            m_appData(appData),
            m_browser(new QtTreePropertyBrowser()),
            m_groupManager(new QtGroupPropertyManager()),
            m_boolPropertyManager(new QtBoolPropertyManager()),
            m_renderModules(),
            m_checkBoxFactory(new QtCheckBoxFactory()),
            m_intPropertyManager(new QtIntPropertyManager()),
            m_stackSizeProperty(0),
            m_spinBoxFactory(new QtSpinBoxFactory()),
            m_showLookAtPointProperty(0),
            m_showBoundingBoxProperty(0)
        {
            this->setObjectName("RenderSettings");
            QScrollArea* m_scrollArea = new QScrollArea();

            QtProperty* topLevelProperty = m_groupManager->addProperty("Render Modules");
            m_browser->addProperty(topLevelProperty);

            // For some reason, it appears that I need to set the factory before setting the
            // actual values, even though this doesn't appear to be true of the other widgets.
            m_browser->setFactoryForManager(m_boolPropertyManager, m_checkBoxFactory);
            m_browser->setFactoryForManager(m_intPropertyManager, m_spinBoxFactory);

            BOOST_FOREACH( boost::shared_ptr<RenderModule> module, m_appData->GetSurfaceSceneView()->GetRenderModules() )
            {
                QtProperty* property = m_boolPropertyManager->addProperty(module->GetName().c_str());
                m_boolPropertyManager->setValue(property, module->GetEnabled());
                topLevelProperty->addSubProperty(property);
                m_renderModules[property] = module;

                module->OnEnabledChanged.connect(boost::bind(&RenderSettingsDockWidget::HandleEnabledChanged, this, property, _1, _2));
            }

            // Propogate changes to the render modules.
            connect(m_boolPropertyManager, SIGNAL(valueChanged(QtProperty*,bool)), this, SLOT(HandleEnabledChangedByGui(QtProperty*,bool)));

            m_stackSizeProperty = m_intPropertyManager->addProperty("Stack Size");
            m_intPropertyManager->setValue(m_stackSizeProperty, appData->GetSurfaceSceneView()->GetOptixStackSize());
            topLevelProperty->addSubProperty(m_stackSizeProperty);
            connect(m_intPropertyManager, SIGNAL(valueChanged(QtProperty*,int)), this, SLOT(HandleStackSizeChanged(QtProperty*,int)));

            m_showLookAtPointProperty = m_boolPropertyManager->addProperty("Show LookAt Point");
            m_boolPropertyManager->setValue(m_showLookAtPointProperty, m_appData->GetShowLookAtPoint());
            topLevelProperty->addSubProperty(m_showLookAtPointProperty);

            m_showBoundingBoxProperty = m_boolPropertyManager->addProperty("Show Bounding Box");
            m_boolPropertyManager->setValue(m_showBoundingBoxProperty, m_appData->GetShowBoundingBox());
            topLevelProperty->addSubProperty(m_showBoundingBoxProperty);

            connect(m_boolPropertyManager, SIGNAL(valueChanged(QtProperty*,bool)), this, SLOT(HandleShowLookAtChanged(QtProperty*,bool)));

            m_scrollArea->setWidget(m_browser);
            m_layout = new QGridLayout(m_scrollArea);

            m_layout->addWidget(m_browser,0,0);
            this->setWidget(m_scrollArea);

            m_browser->show();
        }

        void RenderSettingsDockWidget::HandleEnabledChangedByGui(QtProperty* property, bool newValue)
        {
            boost::shared_ptr<RenderModule> module = m_renderModules[property];

            if( module && module->GetEnabled() != newValue )
            {
                module->SetEnabled(newValue);
            }
        }

        void RenderSettingsDockWidget::HandleStackSizeChanged(QtProperty* property, int newValue)
        {
            m_appData->GetSurfaceSceneView()->SetOptixStackSize(newValue);
        }

        void RenderSettingsDockWidget::HandleShowLookAtChanged(QtProperty* property, bool newValue)
        {
            if( property == m_showLookAtPointProperty )
            {
                if( m_appData->GetShowLookAtPoint() != newValue )
                {
                    m_appData->SetShowLookAtPoint(newValue);
                }
            }
            else if( property == m_showBoundingBoxProperty )
            {
                if( m_appData->GetShowBoundingBox() != newValue )
                {
                    m_appData->SetShowBoundingBox(newValue);
                }
            }

        }

        void RenderSettingsDockWidget::HandleEnabledChanged(QtProperty* property, const RenderModule& module, bool newValue)
        {
            if( m_boolPropertyManager->value(property) != newValue)
            {
                m_boolPropertyManager->setValue(property, newValue);
            }
        }
    }
}
