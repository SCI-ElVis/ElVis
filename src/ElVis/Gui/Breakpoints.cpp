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


#ifdef QT_OPENGL_SUPPORT
#include <QGLWidget>
#endif

#include <ElVis/Gui/Breakpoints.h>
#include <ElVis/Gui/ColorMapRect.h>

#include <boost/bind.hpp>

namespace ElVis
{
    namespace Gui
    {
        Breakpoints::Breakpoints(ColorMapRect* widget) : 
            QObject(widget),
            m_widget(widget),
            m_connectionType(LineConnection),
            m_pointPen(QColor(255, 255, 255, 191), 1),
            m_selectedPointPen(QColor(0, 0, 0, 191), 1),
            m_connectionPen(QColor(255, 255, 255, 127), 2),
            m_pointBrush(QColor(191, 191, 191, 127)),
            m_pointSize(11, 11),
            m_selectedBreakpoint(widget->GetTransferFunction()->GetBreakpoints().end()),
            m_model(widget->GetTransferFunction()),
            m_hostTransferFunction(0)
        {
            widget->installEventFilter(this);
            m_model->OnColorMapChanged.connect(boost::bind(&QWidget::update, m_widget));
        }

        bool Breakpoints::eventFilter(QObject *object, QEvent *event)
        {
            if( object != m_widget )
            {
                return false;
            }

            switch (event->type()) 
            {
                case QEvent::MouseButtonPress:
                {
                    QMouseEvent *me = (QMouseEvent *) event;
                    QPointF clickPos = me->pos();

                    int i = 0;
                    for(m_selectedBreakpoint = m_model->GetBreakpoints().begin(); m_selectedBreakpoint != m_model->GetBreakpoints().end(); ++m_selectedBreakpoint)
                    {
                        QPainterPath path;
                        path.addEllipse(pointBoundingRect(m_selectedBreakpoint));
                        if (path.contains(clickPos)) 
                        {
                            break;
                        }
                    }

                    if (me->button() == Qt::LeftButton) 
                    {
                        if (m_selectedBreakpoint == m_model->GetBreakpoints().end() ) 
                        {
                            double px, py;
                            CalculatePercent(clickPos.x(), clickPos.y(), px, py);
                            ElVis::Color c;
                            c.SetAlpha(py);
                            c.SetGreen(1.0);
                            c.SetRed(1.0);
                            c.SetBlue(1.0);
                            m_selectedBreakpoint = m_model->InsertBreakpoint(px, c);
                        } 
                    } 
                    else if (me->button() == Qt::RightButton) 
                    {
                        if( m_selectedBreakpoint != m_model->GetBreakpoints().end() )
                        {
                            m_model->RemoveBreakpoint(m_selectedBreakpoint);
                            m_selectedBreakpoint = m_model->GetBreakpoints().end();
                        }
                    }
                    OnSelectedPointChanged(m_selectedBreakpoint);
                    OnBreakpointsChanged();
                    return true;
                }
                break;

                case QEvent::MouseButtonRelease:
                    break;

                case QEvent::MouseMove:
                    if (m_selectedBreakpoint != m_model->GetBreakpoints().end() )
                    {
                        const QPointF& point = ((QMouseEvent *)event)->pos();
                        double px, py;
                        CalculatePercent(point.x(), point.y(), px, py);

                        ElVisFloat key = px;
                        while( m_model->GetBreakpoints().find(key) != m_model->GetBreakpoints().end())
                        {
                            key += .00001;
                        }

                        ElVis::Color c = (*m_selectedBreakpoint).second.Col;
                        c.SetAlpha(py);
                        m_model->RemoveBreakpoint(m_selectedBreakpoint);
                        m_selectedBreakpoint = m_model->InsertBreakpoint(key, c);
                        OnBreakpointsChanged();
                    }
                    break;

                case QEvent::Resize:
                {
                    //QResizeEvent *e = (QResizeEvent *) event;
                    //if (e->oldSize().width() == 0 || e->oldSize().height() == 0)
                    //    break;
                    //qreal stretch_x = e->size().width() / qreal(e->oldSize().width());
                    //qreal stretch_y = e->size().height() / qreal(e->oldSize().height());
                    //for (int i=0; i<m_points.size(); ++i) 
                    //{
                    //    QPointF p = m_points[i];
                    //    movePoint(i, QPointF(p.x() * stretch_x, p.y() * stretch_y), false);
                    //}

                    //update();
                    break;
                }

                case QEvent::Paint:
                {
                    ColorMapRect* that_widget = m_widget;
                    m_widget = 0;
                    QApplication::sendEvent(object, event);
                    m_widget = that_widget;
                    paintPoints();
                    return true;
                }
                default:
                    break;
            }

            return false;
        }


        void Breakpoints::paintPoints()
        {
            const std::map<ElVisFloat, ColorMapBreakpoint>& breakpoints = m_model->GetBreakpoints();
            if( breakpoints.size() == 0) return;

            QPainter p;
            p.begin(m_widget);
            p.setRenderHint(QPainter::Antialiasing);

            p.setPen(m_connectionPen);
            QPainterPath path;

            typedef std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator IterType;
            IterType iter = breakpoints.begin();
            double px = (*iter).first;
            double py = (*iter).second.Col.Alpha();

            double x,y;
            CalculatePosition(px, py, x, y);
            path.moveTo( x, y );

            for( ; iter != breakpoints.end(); ++iter)
            {
                px = (*iter).first;
                py = (*iter).second.Col.Alpha();
                CalculatePosition(px, py, x, y);

                path.lineTo(x, y);
            }
            p.drawPath(path);


            p.setPen(m_pointPen);
            p.setBrush(m_pointBrush);

            for(iter = breakpoints.begin() ; iter != breakpoints.end(); ++iter)
            {
                if( iter == m_selectedBreakpoint )
                {
                    p.setPen(m_selectedPointPen);
                }
                else
                {
                    p.setPen(m_pointPen);
                }

                QRectF bounds = pointBoundingRect(iter);
                p.drawEllipse(bounds);
            }

            //if (m_connectionPen.style() != Qt::NoPen && m_connectionType != NoConnection) 
            //{
            //    p.setPen(m_connectionPen);

            //    QPainterPath path;
            //    path.moveTo(m_points.at(0));
            //    for (int i=1; i<m_points.size(); ++i) {
            //        path.lineTo(m_points.at(i));
            //    }
            //    p.drawPath(path);
            //}

            //p.setPen(m_pointPen);
            //p.setBrush(m_pointBrush);

            //for (int i=0; i<m_points.size(); ++i) 
            //{
            //    if( i == m_currentIndex ) continue;
            //    QRectF bounds = pointBoundingRect(i);
            //    p.drawEllipse(bounds);
            //}

            //if( m_currentIndex >= 0 && m_currentIndex < m_points.size() )
            //{
            //    p.setPen(m_selectedPointPen);
            //    QRectF bounds = pointBoundingRect(m_currentIndex);
            //    p.drawEllipse(bounds);
            //}
        }

        static QPointF bound_point(const QPointF &point, const QRectF &bounds, int lock)
        {
            QPointF p = point;

            qreal left = bounds.left();
            qreal right = bounds.right();
            qreal top = bounds.top();
            qreal bottom = bounds.bottom();

            //if (p.x() < left || (lock & Breakpoints::LockToLeft)) p.setX(left);
            //else if (p.x() > right || (lock & Breakpoints::LockToRight)) p.setX(right);

            //if (p.y() < top || (lock & Breakpoints::LockToTop)) p.setY(top);
            //else if (p.y() > bottom || (lock & Breakpoints::LockToBottom)) p.setY(bottom);

            return p;
        }


        //inline static bool x_less_than(const QPointF &p1, const QPointF &p2)
        //{
        //    return p1.x() < p2.x();
        //}


        //inline static bool y_less_than(const QPointF &p1, const QPointF &p2)
        //{
        //    return p1.y() < p2.y();
        //}



        QRectF Breakpoints::pointBoundingRect(std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator iter) const
        {
            qreal w = m_pointSize.width();
            qreal h = m_pointSize.height();

            double x, y;
            CalculatePosition((*iter).first, (*iter).second.Col.Alpha(), x, y);
            x -= w/2.0;
            y -= h/2.0;
            return QRectF(x, y, w, h);
        }

        QRectF Breakpoints::boundingRect() const
        {
            return m_widget->rect();
        }

        void Breakpoints::CalculatePosition(double percentX, double percentY, double& x, double& y) const
        {
            const QRectF& bounds = boundingRect();

            x = percentX * (bounds.right()-bounds.left()) + bounds.left();
            y = (1.0-percentY) * (bounds.bottom()-bounds.top()) + bounds.top();

            x = std::max(x, bounds.left());
            x = std::min(x, bounds.right());
            y = std::max(y, bounds.top());
            y = std::min(y, bounds.bottom());
        }

        void Breakpoints::CalculatePercent(double x, double y, double& percentX, double& percentY) const
        {
            const QRectF& bounds = boundingRect();

            percentX = (double)(x-bounds.left())/(double)bounds.width();
            percentY = 1.0 - (double)(y-bounds.top())/(double)bounds.height();

            percentX = std::max(percentX, 0.0);
            percentX = std::min(percentX, 1.0);
            percentY = std::max(percentY, 0.0);
            percentY = std::min(percentY, 1.0);

        }

        std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator Breakpoints::GetSelectedBreakpoint() const
        {
            return m_selectedBreakpoint;
        }
    }
}
