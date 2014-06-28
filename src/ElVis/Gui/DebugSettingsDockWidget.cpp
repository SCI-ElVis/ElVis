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

#include <ElVis/Gui/DebugSettingsDockWidget.h>
#include <ElVis/Gui/Property.hpp>

namespace ElVis
{
    namespace Gui
    {
        DebugSettingDockWidget::DebugSettingDockWidget(boost::shared_ptr<ApplicationState> appData, QWidget* parent, Qt::WindowFlags f) :
            QDockWidget("Debug Settings", parent, f),
            m_appData(appData),
            m_dockWidgetContents(new QScrollArea()),
            m_browser(new QtTreePropertyBrowser()),
            m_layout(new QGridLayout()),
            m_pointManager(new PointPropertyManager()),
            m_intPropertyManager(new QtIntPropertyManager()),
            m_colorPropertyManager(new ColorPropertyManager()),
            m_doublePropertyManager(new QtDoublePropertyManager()),
            m_intersectionPointProperty(0),
            m_normalManager(new PointPropertyManager()),
            m_normalProperty(0),
            m_pointInfo(),
            m_groupPropertyManager(new QtGroupPropertyManager()),
            m_optixProperties(0),
            m_optixStackSizeProperty(0),
            m_enableTraceProperty(0),
            m_pixelXProperty(0),
            m_pixelYProperty(0),
            m_pixelProperty(0),
            m_selectedPixelProperty(0),
            m_sampleProperty(0)
        {
            this->setObjectName("DebugSettings");
            m_dockWidgetContents->setLayout(m_layout);

            /////////////////////////////
            // Pixel
            //////////////////////////////
            m_pixelProperty = m_groupPropertyManager->addProperty("Selected Pixel");
            m_selectedPixelProperty = m_groupPropertyManager->addProperty("Point");

            m_pixelXProperty = m_intPropertyManager->addProperty("X");
            m_intPropertyManager->setValue(m_pixelXProperty, m_pointInfo.Pixel.x());
            m_selectedPixelProperty->addSubProperty(m_pixelXProperty);

            m_pixelYProperty = m_intPropertyManager->addProperty("Y");
            m_intPropertyManager->setValue(m_pixelYProperty, m_pointInfo.Pixel.y());
            m_selectedPixelProperty->addSubProperty(m_pixelYProperty);

            m_pixelProperty->addSubProperty(m_selectedPixelProperty);
            m_browser->addProperty(m_pixelProperty);

            /////////////////////////////
            // Intersection Point
            /////////////////////////////
            m_intersectionPointProperty = m_pointManager->addProperty("Intersection Point", m_pointInfo.IntersectionPoint);
            m_pixelProperty->addSubProperty(m_intersectionPointProperty);

            m_normalProperty = m_normalManager->addProperty("Normal", m_pointInfo.Normal);
            m_pixelProperty->addSubProperty(m_normalProperty);

            m_optixProperties = m_groupPropertyManager->addProperty("Tracing");
            m_browser->addProperty(m_optixProperties);

            boost::shared_ptr<Scene> scene = m_appData->GetScene();

            /////////////////////
            // Enable optix trace
            //////////////////////
            m_enableTraceProperty = new Property<bool>("Enable Trace",
                                                       boost::bind(&Scene::SetEnableOptixTrace, scene.get(), _1),
                                                       boost::bind(&Scene::GetEnableOptixTrace, scene.get()),
                                                       scene->OnEnableTraceChanged);
            m_enableTraceProperty->SetupPropertyManagers(m_browser);
            m_optixProperties->addSubProperty(m_enableTraceProperty->GetProperty());


            //////////////////////
            // Optix stack size.
            //////////////////////
            m_optixStackSizeProperty = new Property<int>("Print Buffer Size", boost::bind(&Scene::SetOptixTraceBufferSize, scene.get(), _1),
                                                         boost::bind(&Scene::GetOptixTraceBufferSize, scene.get()),
                                                         scene->OnOptixPrintBufferSizeChanged);
            m_optixStackSizeProperty->SetupPropertyManagers(m_browser);
            m_optixProperties->addSubProperty(m_optixStackSizeProperty->GetProperty());

            /////////////////////////////
            // Element Id
            /////////////////////////////
            m_elementIdProperty = m_intPropertyManager->addProperty("Element ID");
            m_intPropertyManager->setValue(m_elementIdProperty, m_pointInfo.ElementId);
            m_pixelProperty->addSubProperty(m_elementIdProperty);

            /////////////////////////////
            // Element Type
            /////////////////////////////
            m_elementTypeProperty = m_intPropertyManager->addProperty("Element Type");
            m_intPropertyManager->setValue(m_elementTypeProperty, m_pointInfo.ElementTypeId);
            m_pixelProperty->addSubProperty(m_elementTypeProperty);

            /////////////////////////////
            // Sample
            /////////////////////////////
            m_sampleProperty = m_doublePropertyManager->addProperty("Sample");
            m_doublePropertyManager->setValue(m_sampleProperty, 0.0);
            m_pixelProperty->addSubProperty(m_sampleProperty);
            m_doublePropertyManager->setDecimals(m_sampleProperty, 8);


            m_layout->addWidget(m_browser, 0, 0);
            this->setWidget(m_dockWidgetContents);
            m_appData->OnSelectedPointChanged.connect(boost::bind(&DebugSettingDockWidget::HandleSelectedPointChanged, this, _1));
        }


        void DebugSettingDockWidget::HandleSelectedPointChanged(const PointInfo& info)
        {
            if( info.Valid )
            {
                m_pointInfo = info;

                if( info.Pixel.x() != m_intPropertyManager->value(m_pixelXProperty) )
                {
                    m_intPropertyManager->setValue(m_pixelXProperty, info.Pixel.x());
                }
                if( info.Pixel.y() != m_intPropertyManager->value(m_pixelYProperty) )
                {
                    m_intPropertyManager->setValue(m_pixelYProperty, info.Pixel.y());
                }

                if( info.ElementId != m_intPropertyManager->value(m_elementIdProperty) )
                {
                    m_intPropertyManager->setValue(m_elementIdProperty, info.ElementId);
                }

                if( info.ElementTypeId != m_intPropertyManager->value(m_elementTypeProperty) )
                {
                    m_intPropertyManager->setValue(m_elementTypeProperty, info.ElementTypeId);
                }

                if( info.Scalar != m_doublePropertyManager->value(m_sampleProperty) )
                {
                    m_doublePropertyManager->setValue(m_sampleProperty, info.Scalar);
                }
            }
            else
            {
            }
        }

    }
}


