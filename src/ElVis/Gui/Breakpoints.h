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

//#include <QtGui/QtGui>
#include <QtGui/QtGuiDepends>
#include <QtGui/qaccessible.h>
#include <QtGui/qaccessiblebridge.h>
#include <QtGui/qaccessibleobject.h>
#include <QtGui/qaccessibleplugin.h>
#include <QtGui/qbitmap.h>
#include <QtGui/qicon.h>
#include <QtGui/qiconengine.h>
#include <QtGui/qiconengineplugin.h>
#include <QtGui/qimage.h>
#include <QtGui/qimageiohandler.h>
#include <QtGui/qimagereader.h>
#include <QtGui/qimagewriter.h>
#include <QtGui/qmovie.h>
#include <QtGui/qpicture.h>
#include <QtGui/qpictureformatplugin.h>
#include <QtGui/qpixmap.h>
#include <QtGui/qpixmapcache.h>
#include <QtGui/qstandarditemmodel.h>
#include <QtGui/qclipboard.h>
#include <QtGui/qcursor.h>
#include <QtGui/qdrag.h>
#include "qevent.h"
#include <QtGui/qgenericplugin.h>
#include <QtGui/qgenericpluginfactory.h>
#include <QtGui/qguiapplication.h>
#include <QtGui/qinputmethod.h>
#include <QtGui/qkeysequence.h>
#include <QtGui/qoffscreensurface.h>
//#include "qopenglcontext.h"
#include <QtGui/qpalette.h>
#include <QtGui/qscreen.h>
#include <QtGui/qsessionmanager.h>
#include <QtGui/qstylehints.h>
#include <QtGui/qsurface.h>
#include <QtGui/qsurfaceformat.h>
#include <QtGui/qtouchdevice.h>
#include <QtGui/qwindow.h>
#include <QtGui/qwindowdefs.h>
#include <QtGui/qgenericmatrix.h>
#include <QtGui/qmatrix4x4.h>
#include <QtGui/qquaternion.h>
#include <QtGui/qvector2d.h>
#include <QtGui/qvector3d.h>
#include <QtGui/qvector4d.h>
//#include "qopengl.h"
//#include "qopenglbuffer.h"
//#include "qopengldebug.h"
//#include "qopenglframebufferobject.h"
//#include "qopenglfunctions.h"
//#include "qopenglpaintdevice.h"
//#include "qopenglpixeltransferoptions.h"
//#include "qopenglshaderprogram.h"
//#include "qopengltexture.h"
//#include "qopengltimerquery.h"
//#include "qopenglversionfunctions.h"
//#include "qopenglvertexarrayobject.h"
#include <QtGui/qbackingstore.h>
#include <QtGui/qbrush.h>
#include <QtGui/qcolor.h>
#include <QtGui/qmatrix.h>
#include <QtGui/qpagedpaintdevice.h>
#include <QtGui/qpagelayout.h>
#include <QtGui/qpagesize.h>
#include <QtGui/qpaintdevice.h>
#include <QtGui/qpaintengine.h>
#include <QtGui/qpainter.h>
#include <QtGui/qpainterpath.h>
#include <QtGui/qpdfwriter.h>
#include <QtGui/qpen.h>
#include <QtGui/qpolygon.h>
#include <QtGui/qregion.h>
#include <QtGui/qrgb.h>
#include <QtGui/qtransform.h>
#include <QtGui/qabstracttextdocumentlayout.h>
#include <QtGui/qfont.h>
#include <QtGui/qfontdatabase.h>
#include <QtGui/qfontinfo.h>
#include <QtGui/qfontmetrics.h>
//#include "qglyphrun.h"
#include <QtGui/qrawfont.h>
#include <QtGui/qstatictext.h>
#include <QtGui/qsyntaxhighlighter.h>
#include <QtGui/qtextcursor.h>
#include <QtGui/qtextdocument.h>
#include <QtGui/qtextdocumentfragment.h>
#include <QtGui/qtextdocumentwriter.h>
#include <QtGui/qtextformat.h>
#include <QtGui/qtextlayout.h>
#include <QtGui/qtextlist.h>
#include <QtGui/qtextobject.h>
#include <QtGui/qtextoption.h>
#include <QtGui/qtexttable.h>
#include <QtGui/qdesktopservices.h>
#include <QtGui/qvalidator.h>
#include <QtGui/qtguiversion.h>
#include <QtWidgets/QApplication>

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
                boost::shared_ptr<PiecewiseLinearColorMap> m_model;
                ElVis::HostTransferFunction* m_hostTransferFunction;
        };

    }

}

#endif // ELVIS_GUI_BREAKPOINTS_H
