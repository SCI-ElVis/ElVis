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

#include <ElVis/Gui/SampleOntoNrrdDockWidget.h>
#include <boost/bind.hpp>

namespace ElVis
{
    namespace Gui
    {
        SampleOntoNrrdDockWidget::SampleOntoNrrdDockWidget(boost::shared_ptr<ApplicationState> appData, QWidget* parent, Qt::WindowFlags f) :
            QDockWidget("Sample Onto Nrrd", parent, f),
            m_appData(appData),
            m_fileName(new QLineEdit()),
            m_fileNameLabel(new QLabel("File Name")),
            m_generateButton(new QPushButton("Generate File")),
            m_spacing(new QDoubleSpinBox()),
            m_layout(0),
            m_hLabel(new QLabel("H")),
            m_module(new SampleOntoNrrd())
        {
            setObjectName("SampleOntoNrrdModule");

            QWidget* widget = new QWidget();

            m_layout = new QGridLayout(widget);
            m_layout->addWidget(m_fileNameLabel, 0, 0);
            m_layout->addWidget(m_fileName, 0, 1);
            m_layout->addWidget(m_hLabel, 1,0);
            m_layout->addWidget(m_spacing, 1,1);
            m_layout->addWidget(m_generateButton, 2, 0);

            m_spacing->setMinimum(.00000000001);
            m_spacing->setMaximum(std::numeric_limits<ElVisFloat>::max());
            m_spacing->setDecimals(10);

            this->setWidget(widget);

            connect(m_generateButton, SIGNAL(clicked()), this, SLOT(HandleGeneratePressed()));

        }

        void SampleOntoNrrdDockWidget::HandleGeneratePressed()
        {
            std::string fileName = m_fileName->text().toStdString();
            ElVisFloat h = m_spacing->value();

            m_module->Sample(m_appData->GetSurfaceSceneView(), fileName, h);
        }

    }
}

