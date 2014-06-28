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
#include <ElVis/Core/ColorMap.h>

#include <ElVis/Gui/ColorMapRect.h>

#include <QtWidgets/QPushButton>

namespace ElVis
{
    namespace Gui
    {
        ColorMapRect::ColorMapRect() :
            m_shade(),
            m_colorMap(),
            m_hoverPoints(0)
        {
        }

        void ColorMapRect::paintEvent(QPaintEvent* event)
        {
            GenerateShade();

            QPainter p(this);
            p.drawImage(0, 0, m_shade);
            p.setPen(QColor(145, 145, 145));
            p.drawRect(0,0, width()-1, height()-1);
        }

        QSize ColorMapRect::sizeHint () const
        {
            QSize result;
            result.setWidth(100);
            result.setHeight(50);
            return result;
        }

        void ColorMapRect::SetTransferFunction(boost::shared_ptr<PiecewiseLinearColorMap> value)
        {
            if( m_colorMap != value )
            {
                m_shade = QImage(0, 0, QImage::Format_RGB32);

                // Set this before creating the breakpoint guis.
                m_colorMap = value;

                //disconnect(m_hoverPoints);
                delete m_hoverPoints;
                m_hoverPoints = new Breakpoints(this);
                connect(m_hoverPoints, SIGNAL(OnSelectedPointChanged(const std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator&)),
                    this, SLOT(HandleSelectedPointChanged(const std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator&)));

                connect(m_hoverPoints, SIGNAL(OnBreakpointsChanged()),
                    this, SLOT(HandleBreakpointsChanged()));

                update();
            }
        }

        boost::shared_ptr<PiecewiseLinearColorMap> ColorMapRect::GetTransferFunction() const
        {
            return m_colorMap;
        }

        void ColorMapRect::HandleSelectedPointChanged(const std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator& value)
        {
            Q_EMIT OnSelectedPointChanged(value);
            HandleBreakpointsChanged();
        }

        void ColorMapRect::HandleBreakpointsChanged()
        {
            m_shade = QImage();
            OnColorMapChanged();
            update();
        }

        void ColorMapRect::ForceUpdate()
        {
            m_shade = QImage();
            update();
        }

        std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator ColorMapRect::GetSelectedBreakpoint() const
        {
            return m_hoverPoints->GetSelectedBreakpoint();
        }

        void ColorMapRect::GenerateShade()
        {
            if( m_shade.isNull() || m_shade.size() != size() )
            {
                if( m_colorMap )
                {
                    m_shade = QImage(size(), QImage::Format_RGB32);
                    QPainter p(&m_shade);

                    QLinearGradient shade(0, 0, width(), 0);

                    const std::map<ElVisFloat, ColorMapBreakpoint>& breakpoints = m_colorMap->GetBreakpoints();
                    double start = (*breakpoints.begin()).first;
                    double end = (*breakpoints.rbegin()).first;
                    double h = end-start;

                    for(std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator iter = breakpoints.begin(); iter != breakpoints.end(); ++iter)
                    {
                        double pos = ((*iter).first - start)/h;
                        Color c = (*iter).second.Col;
                        shade.setColorAt(pos,QColor(c.RedAsInt(), c.GreenAsInt(), c.BlueAsInt()));
                    }

                    p.fillRect(rect(), shade);
                }
                else
                {
                    m_shade = QImage(size(), QImage::Format_RGB32);
                    QPainter p(&m_shade);

                    p.fillRect(rect(), QColor(255, 255, 255, 255));
                }
            }
        }
    }
}
