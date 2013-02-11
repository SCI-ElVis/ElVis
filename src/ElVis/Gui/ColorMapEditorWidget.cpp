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

#include <ElVis/Core/HostTransferFunction.h>


#include <ElVis/Gui/ColorMapEditorWidget.h>
#include <ElVis/Gui/ColorMapRect.h>
#include <boost/bind.hpp>


namespace ElVis
{
    namespace Gui
    {
        ColorMapEditorWidget::ColorMapEditorWidget(boost::shared_ptr<ApplicationState> appData) :
            QWidget(),
            m_appData(appData),
            m_rect(new ColorMapRect()),
            m_layout(new QVBoxLayout())
        {
            Clear();
            appData->OnSelectedTransferFunctionChanged.connect(boost::bind(&ColorMapEditorWidget::HandleColorMapSelectionChanged, this, _1));

            connect(m_rect, SIGNAL(OnColorMapChanged()), this, SLOT(HandleColorMapChanged()));

            m_layout->addWidget(m_rect);
            this->setLayout(m_layout);
        }

        void ColorMapEditorWidget::Clear()
        {
        }

        void ColorMapEditorWidget::HandleColorMapSelectionChanged(boost::shared_ptr<PiecewiseLinearColorMap> value)
        {
            m_rect->SetTransferFunction(value);
            HandleColorMapChanged();   
        }

        void ColorMapEditorWidget::HandleColorMapChanged()
        {
            boost::shared_ptr<HostTransferFunction> hostFunc = m_appData->GetHostTransferFunction();
            hostFunc->Clear();

            const std::map<ElVisFloat, ColorMapBreakpoint>& breakpoints = m_rect->GetTransferFunction()->GetBreakpoints();
            for(std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator iter = breakpoints.begin(); iter != breakpoints.end(); ++iter)
            {
                ElVisFloat percent = (*iter).first;
                ElVisFloat scalar = percent*(m_rect->GetTransferFunction()->GetMax()-m_rect->GetTransferFunction()->GetMin()) + m_rect->GetTransferFunction()->GetMin();
                Color color = (*iter).second.Col;
                hostFunc->SetBreakpoint(scalar, color);
            }

            std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator selectedPoint = m_rect->GetSelectedBreakpoint();
            if( m_rect->GetTransferFunction() && 
                selectedPoint != m_rect->GetTransferFunction()->GetBreakpoints().end() )
            {
            }
            m_rect->update();
        }

    }
}
