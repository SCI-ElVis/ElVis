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

#ifndef ELVIS_GUI_SCENE_VIEW_WIDGET_H
#define ELVIS_GUI_SCENE_VIEW_WIDGET_H

#include <ElVis/Core/OpenGL.h>
#include <ElVis/Core/SceneView.h>

#include <ElVis/Gui/ApplicationState.h>

#include <boost/shared_ptr.hpp>
#include <boost/array.hpp>

#include <QtOpenGL/QGLWidget>
#include <QWidget>
#include <QPoint>


namespace ElVis
{
    namespace Gui
    {



        /// \brief Represents a specific view of a scene.
        ///
        /// The ElVis scene is represented by a Scene object.  This object
        /// represents a particular view of that scene.  All of the heavy lifting 
        /// comes from the associated SceneViewWidget object, this widget is essentially 
        /// responsible for providing the OpenGL context and nothing more.
        class SceneViewWidget : public QGLWidget
        {
            Q_OBJECT

            public:
                /// \brief Create a view of the given scene.
                ///
                /// \param parent QGLWidget parameter.
                /// \param name QGLWidget parameter.
                /// \param shareWidget QGLWidget parameter.
                /// \param f QGLWidget parameter.
                ///
                /// Create a view object for the given scene.  The parameters
                /// parent, name, shareWidget, and name are all passed directly
                /// to the appropriate QGLWidget constructor.
                SceneViewWidget(boost::shared_ptr<ApplicationState> appData,
                                QWidget* parent = NULL, const char* name = NULL,
                                const QGLWidget* shareWidget = NULL, Qt::WindowFlags f = 0);

                virtual ~SceneViewWidget();

                //rt::WorldPoint screenPointToWorldPoint(unsigned int row, unsigned int col) const;

                boost::array<GLint, 4> GetOpenGLViewport() const;
                boost::array<GLdouble, 16> GetOpenGLModelViewMatrix() const;
                boost::array<GLdouble, 16> GetOpenGLProjectionMatrix() const;
                GLfloat GetDepthClearValue() const;

                boost::shared_ptr<ElVis::SceneView> GetSceneView() const;
                void GetModelViewMatrixForVisual3(double* out);


            public Q_SLOTS:
                void HandleNeedsRedraw(const SceneView& view);

            Q_SIGNALS:

            protected:
                //////////////////////////////
                // QGLWidget Interface
                //////////////////////////////
                // Called before resizeGL or paintGL.  Use for initialization.
                virtual void initializeGL();

                // Called when the window is resized.
                virtual void resizeGL(int width, int height);

                // Draws the scene.
                virtual void paintGL();

                ///////////////////////////////
                // QWidget Interface
                //////////////////////////////
                virtual void mousePressEvent(QMouseEvent* event);
                virtual void mouseReleaseEvent(QMouseEvent* event);
                virtual void mouseDoubleClickEvent(QMouseEvent* event);
                virtual void mouseMoveEvent(QMouseEvent* event);
                virtual void wheelEvent(QWheelEvent* event);

            private Q_SLOTS:
                // Sets up default viewing parameters for the initial view of a loaded model.
                void SetupDefaultModelView();
            private:
                void SetupPerspective();
                SceneViewWidget(const SceneViewWidget& rhs);
                SceneViewWidget& operator=(const SceneViewWidget& rhs);

                void HandleSceneViewChanged(const SceneView&);
                void HandleSceneChanged(const Scene&);

                static const GLenum LIGHT_ENUM_VALUES[];

                GLfloat clearDepth;

                boost::array<GLint, 4> viewport;
                boost::array<GLdouble, 16> modelview;
                boost::array<GLdouble, 16> projection;

                QPoint m_moveStartPixel;
                boost::shared_ptr<ApplicationState> m_appData;
        };
    }
}

#endif

