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

#include <ElVis/Gui/LightingDockWidget.h>
#include <ElVis/Gui/ApplicationState.h>

#include <ElVis/QtPropertyBrowser/qttreepropertybrowser.h>

#include <ElVis/Core/VolumeRenderingModule.h>

#include <QtWidgets/QScrollArea>
#include <QtWidgets/QColorDialog>

#include <boost/bind.hpp>

namespace ElVis
{
    namespace Gui
    {
        LightingDockWidget::LightingDockWidget(boost::shared_ptr<ApplicationState> appData,
            QWidget* parent, Qt::WindowFlags f) :
            QDockWidget("Light Settings", parent, f),
            m_appData(appData),
            m_layout(0),
            m_changeHeadlightColor(new QPushButton("Change Headlight")),
            m_headlightColorLabel(new QLabel("Headlight")),
            m_changeBackgroundColor(new QPushButton("Change Background")),
            m_backgroundColorLabel(new QLabel("Background")),
            m_changeAmbientColor(new QPushButton("Change Ambient Light")),
            m_ambientColorLabel(new QLabel("Ambient"))
        {
            setObjectName("LightingDockWidget");
            QScrollArea* m_scrollArea = new QScrollArea();

            QWidget* widget = new QWidget();

            m_layout = new QGridLayout(widget);
            m_layout->addWidget(m_headlightColorLabel, 0, 0);
            m_layout->addWidget(m_changeHeadlightColor, 0, 1);
            m_layout->addWidget(m_backgroundColorLabel, 1, 0);
            m_layout->addWidget(m_changeBackgroundColor, 1, 1);

            m_layout->addWidget(m_ambientColorLabel, 2, 0);
            m_layout->addWidget(m_changeAmbientColor, 2, 1);

            ///////////////////////////////////
            // Subscriptions
            ///////////////////////////////////
            connect(m_changeHeadlightColor, SIGNAL(clicked()), this, SLOT(HandleChangeHeadlightColorButtonPushed()));
            connect(m_changeAmbientColor, SIGNAL(clicked()), this, SLOT(HandleChangeAmbientButtonPushed()));

            this->setWidget(widget);
        }

        void LightingDockWidget::HandleChangeAmbientButtonPushed()
        {
            const Color& ambientColor = m_appData->GetSurfaceSceneView()->GetScene()->AmbientLightColor();
            QColor initial = Convert(ambientColor);

            QColor result = QColorDialog::getColor(initial);
            if( result.isValid() )
            {
                Color newColor = Convert(result);

                if( newColor != ambientColor )
                {
                    m_appData->GetSurfaceSceneView()->GetScene()->SetAmbientLightColor(newColor);
                }
            }
        }

        void LightingDockWidget::HandleChangeHeadlightColorButtonPushed()
        {
            const Color& headlightColor = m_appData->GetSurfaceSceneView()->GetHeadlightColor();
            QColor initial = Convert(headlightColor);

            QColor result = QColorDialog::getColor(initial);
            if( result.isValid() )
            {
                Color newColor = Convert(result);

                if( newColor != headlightColor )
                {
                    m_appData->GetSurfaceSceneView()->SetHeadlightColor(newColor);
                }
            }
        }

        void LightingDockWidget::HandleBackgroundButtonPushed()
        {
            const Color& headlightColor = m_appData->GetSurfaceSceneView()->GetBackgroundColor();
            QColor initial = Convert(headlightColor);

            QColor result = QColorDialog::getColor(initial);
            if( result.isValid() )
            {
                Color newColor = Convert(result);

                if( newColor != headlightColor )
                {
                    m_appData->GetSurfaceSceneView()->SetBackgroundColor(newColor);
                }
            }
        }

        Color LightingDockWidget::Convert(const QColor& result)
        {
            Color newColor;
            int r, g, b, a;
            result.getRgb(&r, &g, &b, &a);
            newColor.SetRed(r);
            newColor.SetGreen(g);
            newColor.SetBlue(b);
            newColor.SetAlpha(a);
            return newColor;
        }

        QColor LightingDockWidget::Convert(const Color& headlightColor)
        {
            QColor initial;
            initial.setRed(headlightColor.RedAsInt());
            initial.setGreen(headlightColor.GreenAsInt());
            initial.setBlue(headlightColor.BlueAsInt());
            initial.setAlpha(headlightColor.AlphaAsInt());
            return initial;
        }

    }
}
