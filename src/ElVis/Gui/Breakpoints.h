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

#ifndef ELVIS_GUI_BREAKPOINTS_H
#define ELVIS_GUI_BREAKPOINTS_H

#include <QtGui>

#include <ElVis/Core/ColorMap.h>
#include <ElVis/Core/HostTransferFunction.h>

#include <boost/shared_ptr.hpp>

namespace ElVis
{
    namespace Gui
    {
        class ColorMapRect;
        class Breakpoints : public QObject
        {
            Q_OBJECT

            public:
                enum ConnectionType 
                {
                    NoConnection,
                    LineConnection
                };

                explicit Breakpoints(ColorMapRect* widget);

                bool eventFilter(QObject *object, QEvent *event);

                void paintPoints();

                QRectF boundingRect() const;

                QSizeF pointSize() const { return m_pointSize; }
                void setPointSize(const QSizeF &size) { m_pointSize = size; }

                ConnectionType connectionType() const { return m_connectionType; }
                void setConnectionType(ConnectionType connectionType) { m_connectionType = connectionType; }

                void setConnectionPen(const QPen &pen) { m_connectionPen = pen; }
                void setShapePen(const QPen &pen) { m_pointPen = pen; }
                void setShapeBrush(const QBrush &brush) { m_pointBrush = brush; }

                std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator GetSelectedBreakpoint() const;

            public Q_SLOTS:

            Q_SIGNALS:
                void OnSelectedPointChanged(const std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator& value);
                void OnBreakpointsChanged();

            private:
                QRectF pointBoundingRect(std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator iter) const;
                void CalculatePosition(double percentX, double percentY, double& x, double& y) const;
                void CalculatePercent(double x, double y, double& percentX, double& percentY) const;

                ColorMapRect* m_widget;
                ConnectionType m_connectionType;

                QSizeF m_pointSize;
                std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator m_selectedBreakpoint;

                QPen m_pointPen;
                QPen m_selectedPointPen;
                QBrush m_pointBrush;
                QPen m_connectionPen;
                boost::shared_ptr<ColorMap> m_model;
                ElVis::HostTransferFunction* m_hostTransferFunction;
        };

    }

}

#endif // ELVIS_GUI_BREAKPOINTS_H
