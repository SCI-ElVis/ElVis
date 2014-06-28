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

#include <ElVis/Gui/VolumeRenderingSettingsWidget.h>
#include <ElVis/Gui/ApplicationState.h>

#include <ElVis/QtPropertyBrowser/qttreepropertybrowser.h>

#include <ElVis/Core/VolumeRenderingModule.h>

#include <QtWidgets/QScrollArea>

#include <boost/bind.hpp>

namespace ElVis
{
    namespace Gui
    {
        VolumeRenderingSettingsWidget::VolumeRenderingSettingsWidget(boost::shared_ptr<ApplicationState> appData,
            QWidget* parent, Qt::WindowFlags f) :
            QDockWidget("Volume Rendering Settings", parent, f),
            m_appData(appData),
            m_boolPropertyManager(new QtBoolPropertyManager()),
            m_checkBoxFactory(new QtCheckBoxFactory()),
            m_doublePropertyManager(new QtDoublePropertyManager()),
            m_doubleSpinBoxFactory(new QtDoubleSpinBoxFactory()),
            m_enumPropertyManager(new QtEnumPropertyManager()),
            m_enumEditorFactory(new QtEnumEditorFactory()),
            m_layout(0),
            m_browser(new QtTreePropertyBrowser()),
            m_enabledProperty(0),
            m_integrationTypeProperty(0),
            m_sampleSpacingProperty(0),
            m_adaptiveErrorProperty(0),
            m_trackSamplesProperty(0)
        {
            setObjectName("VolumeRenderingDockWidget");
            QScrollArea* m_scrollArea = new QScrollArea();

            if (m_appData->GetVolumeRenderingModule() )
            {
                m_enabledProperty = m_boolPropertyManager->addProperty("Enabled");
                m_boolPropertyManager->setValue(m_enabledProperty, m_appData->GetVolumeRenderingModule()->GetEnabled());
                m_browser->addProperty(m_enabledProperty);

                connect(m_boolPropertyManager, SIGNAL(valueChanged(QtProperty*,bool)), this, SLOT(HandleEnabledChangedInGui(QtProperty*,bool)));
                m_appData->GetVolumeRenderingModule()->OnEnabledChanged.connect(boost::bind(&VolumeRenderingSettingsWidget::HandleEnabledChanged, this, _1, _2));

                m_trackSamplesProperty = m_boolPropertyManager->addProperty("Track Samples");
                m_boolPropertyManager->setValue(m_trackSamplesProperty, m_appData->GetVolumeRenderingModule()->GetTrackNumberOfSamples());
                m_browser->addProperty(m_trackSamplesProperty);


                //////////////////////////////////
                // Integration Type
                //////////////////////////////////
                m_integrationTypeProperty = m_enumPropertyManager->addProperty("Integration Type");
                QStringList names;
                names.push_back("Riemann");
                names.push_back("Trapezoidal");
                names.push_back("Rieman-Block");
                names.push_back("WPR-Cat Up Front");
                names.push_back("Categorization Only.");
                names.push_back("Block Trap");
                names.push_back("Two Warps/Segment");
                names.push_back("Full");
                m_enumPropertyManager->setEnumNames(m_integrationTypeProperty, names);
                m_browser->addProperty(m_integrationTypeProperty);
                connect(m_enumPropertyManager, SIGNAL(valueChanged(QtProperty*,int)), this, SLOT(HandleIntegrationTypeChangedInGui(QtProperty*,int)));


                //////////////////////////////////////
                // Sample Spacing
                //////////////////////////////////////
                m_sampleSpacingProperty = m_doublePropertyManager->addProperty("Sample Spacing (h)");
                m_doublePropertyManager->setMinimum(m_sampleSpacingProperty, 1e-10);
                m_doublePropertyManager->setDecimals(m_sampleSpacingProperty, 8);
                m_doublePropertyManager->setValue(m_sampleSpacingProperty, m_appData->GetVolumeRenderingModule()->GetCompositingStepSize());
                m_browser->addProperty(m_sampleSpacingProperty);
                connect(m_doublePropertyManager, SIGNAL(valueChanged(QtProperty*,double)), this, SLOT(HandleDoublePropertyChangedInGui(QtProperty*,double)));

                //////////////////////////////////////
                // Adaptive Error
                //////////////////////////////////////
                m_adaptiveErrorProperty = m_doublePropertyManager->addProperty("Target Error");
                m_doublePropertyManager->setMinimum(m_adaptiveErrorProperty, 1e-4);
                m_doublePropertyManager->setDecimals(m_adaptiveErrorProperty, 8);
                m_doublePropertyManager->setValue(m_adaptiveErrorProperty, m_appData->GetVolumeRenderingModule()->GetEpsilon());
                m_browser->addProperty(m_adaptiveErrorProperty);


                m_browser->setFactoryForManager(m_doublePropertyManager, m_doubleSpinBoxFactory);
                m_browser->setFactoryForManager(m_boolPropertyManager, m_checkBoxFactory);
                m_browser->setFactoryForManager(m_enumPropertyManager, m_enumEditorFactory);
                m_scrollArea->setWidget(m_browser);

                m_layout = new QGridLayout(m_scrollArea);
                m_layout->addWidget(m_browser, 0, 0);

                this->setWidget(m_scrollArea);
            }
        }

        void VolumeRenderingSettingsWidget::HandleEnabledChangedInGui(QtProperty* property, bool newValue)
        {
            if( property == m_enabledProperty && m_appData->GetVolumeRenderingModule() && m_appData->GetVolumeRenderingModule()->GetEnabled() != newValue )
            {
                m_appData->GetVolumeRenderingModule()->SetEnabled(newValue);
            }
            else if( property == m_trackSamplesProperty && m_appData->GetVolumeRenderingModule() && m_appData->GetVolumeRenderingModule()->GetTrackNumberOfSamples() != newValue)
            {
                m_appData->GetVolumeRenderingModule()->SetTrackNumberOfSamples(newValue);
            }
        }

        void VolumeRenderingSettingsWidget::HandleEnabledChanged(const RenderModule&, bool newValue)
        {
            if( m_boolPropertyManager->value(m_enabledProperty) != newValue )
            {
                m_boolPropertyManager->setValue(m_enabledProperty, newValue);
            }
        }

        void VolumeRenderingSettingsWidget::HandleIntegrationTypeChangedInGui(QtProperty* property, int newValue)
        {
            if(m_appData->GetVolumeRenderingModule() && m_appData->GetVolumeRenderingModule()->GetIntegrationType() != newValue )
            {
                m_appData->GetVolumeRenderingModule()->SetIntegrationType(static_cast<VolumeRenderingIntegrationType>(newValue));
            }
        }

        void VolumeRenderingSettingsWidget::HandleDoublePropertyChangedInGui(QtProperty* property, double newValue)
        {
            if( !m_appData->GetVolumeRenderingModule() ) return;

            if( property == m_sampleSpacingProperty )
            {
                if( newValue != m_appData->GetVolumeRenderingModule()->GetCompositingStepSize() )
                {
                    m_appData->GetVolumeRenderingModule()->SetCompositingStepSize(newValue);
                }
            }
            else if( property == m_adaptiveErrorProperty )
            {
                if( newValue != m_appData->GetVolumeRenderingModule()->GetEpsilon() )
                {
                    m_appData->GetVolumeRenderingModule()->SetEpsilon(newValue);
                }
            }
        }

    }
}
