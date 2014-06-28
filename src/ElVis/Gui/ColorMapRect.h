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

#ifndef ELVIS_GUI_COLOR_MAP_RECT_H
#define ELVIS_GUI_COLOR_MAP_RECT_H

#include <QtWidgets/QWidget>
#include <boost/shared_ptr.hpp>

#include <ElVis/Gui/Breakpoints.h>
#include <ElVis/Gui/ApplicationState.h>
#include <ElVis/Core/HostTransferFunction.h>

#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QGraphicsScene>
#include <QtWidgets/QGraphicsRectItem>
#include <QtGui/QImage>

namespace ElVis
{
    namespace Gui
    {
        class ColorMapRect : public QWidget
        {
            Q_OBJECT

            public:
                ColorMapRect();
                virtual ~ColorMapRect() {}

                void SetTransferFunction(boost::shared_ptr<PiecewiseLinearColorMap> value);
                boost::shared_ptr<PiecewiseLinearColorMap> GetTransferFunction() const;

                std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator GetSelectedBreakpoint() const;

                void ForceUpdate();

            Q_SIGNALS:
                void OnColorMapChanged();
                void OnSelectedPointChanged(const std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator&);

            protected Q_SLOTS:
                void HandleSelectedPointChanged(const std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator& value);
                void HandleBreakpointsChanged();

            protected:
                virtual void paintEvent(QPaintEvent *);
                virtual QSize sizeHint () const;

            private:
                ColorMapRect(const ColorMapRect&);
                ColorMapRect& operator=(const ColorMapRect&);

                void GenerateShade();
                QImage m_shade;

                boost::shared_ptr<PiecewiseLinearColorMap> m_colorMap;
                Breakpoints* m_hoverPoints;
        };

    }
}

#endif
