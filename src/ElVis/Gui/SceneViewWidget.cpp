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

#include <boost/multi_array.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/call_traits.hpp>
#include <boost/type_traits.hpp>
#include <boost/bind.hpp>

#include <ElVis/Gui/SceneViewWidget.h>
#include <ElVis/Core/SceneViewProjection.h>

#include <ElVis/Core/OpenGL.h>
#include <ElVis/Core/PrimaryRayModule.h>
#include <ElVis/Core/Timer.h>

#include <string>
#include <fstream>
#include <iostream>

#include <QMouseEvent>

using std::cout;
using std::endl;

namespace ElVis
{
    namespace Gui
    {
        const GLenum SceneViewWidget::LIGHT_ENUM_VALUES[] = { GL_LIGHT0,
            GL_LIGHT1, GL_LIGHT2, GL_LIGHT3, GL_LIGHT4, GL_LIGHT5, GL_LIGHT6,
            GL_LIGHT7 };

        SceneViewWidget::SceneViewWidget(boost::shared_ptr<ApplicationState> appData, QWidget* parent, const char* name, const QGLWidget* shareWidget, Qt::WFlags f) :
            QGLWidget(parent, shareWidget, f),
            clearDepth(1.0),
            viewport(),
            modelview(),
            projection(),
            m_appData(appData)
        {
            m_appData->GetScene()->OnModelChanged.connect(boost::bind(&SceneViewWidget::SetupDefaultModelView, this));
            m_appData->GetScene()->OnModelChanged.connect(boost::bind(&SceneViewWidget::update, this));
            m_appData->GetSurfaceSceneView()->GetViewSettings()->OnCameraChanged.connect(boost::bind(&SceneViewWidget::update, this));

            m_appData->GetSurfaceSceneView()->GetPrimaryRayModule()->OnObjectAdded.connect(boost::bind(&SceneViewWidget::update, this));
            m_appData->GetSurfaceSceneView()->Resize(width(), height());
            m_appData->GetSurfaceSceneView()->OnSceneViewChanged.connect(boost::bind(&SceneViewWidget::HandleSceneViewChanged, this, _1));
            m_appData->GetSurfaceSceneView()->OnNeedsRedraw.connect(boost::bind(&SceneViewWidget::HandleNeedsRedraw, this, _1));
            m_appData->GetScene()->OnSceneChanged.connect(boost::bind(&SceneViewWidget::HandleSceneChanged, this, _1));
            m_appData->OnApplicationStateChanged.connect(boost::bind(&SceneViewWidget::update, this));


            this->resize(100,100);
        }


        SceneViewWidget::~SceneViewWidget()
        {
        }



        void SceneViewWidget::HandleNeedsRedraw(const SceneView& view)
        {
            update();
        }

        void SceneViewWidget::SetupDefaultModelView()
        {
            m_appData->GetScene()->GetModel()->CalculateExtents();
            const WorldPoint& maxExtent = m_appData->GetScene()->GetModel()->MaxExtent();
            const WorldPoint& minExtent = m_appData->GetScene()->GetModel()->MinExtent();

            const WorldPoint& center = m_appData->GetScene()->GetModel()->GetMidpoint();
            WorldPoint eye = center;
            eye.SetZ(center.z() + (maxExtent.z()-minExtent.z()));
            if( eye.z() == center.z() )
            {
                eye.SetX(center.x() + (maxExtent.x() - minExtent.x()));
            }
            std::cout << "Center = " << center << std::endl;
            WorldVector up(0, 1, 0);

            m_appData->GetSurfaceSceneView()->GetViewSettings()->SetParameters(eye, center, up);
        }

        boost::shared_ptr<ElVis::SceneView> SceneViewWidget::GetSceneView() const
        {
            return m_appData->GetSurfaceSceneView();
        }

        void SceneViewWidget::initializeGL()
        {
            glClearColor(1.0, 1.0, 1.0, 0.0);
            glEnable(GL_DEPTH_TEST);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
            glShadeModel(GL_FLAT);

            int depthBits;
            glGetIntegerv(GL_DEPTH_BITS, &depthBits);
            m_appData->GetSurfaceSceneView()->SetDepthBufferBits(depthBits);

        }

        // Called when the window is resized.
        void SceneViewWidget::resizeGL(int w, int h)
        {
            glViewport(0, 0, w, h);

            m_appData->GetSurfaceSceneView()->Resize(width(), height());
            m_appData->GetSurfaceSceneView()->GetViewSettings()->SetAspectRatio(w, h);
            SetupProjection();

            std::cout << "New window size (" << w << ", " << h << ")" << std::endl;
            //emit windowResized(w, h);
        }

        void SceneViewWidget::SetupProjection()
        {
            SceneViewProjection projType = m_appData->GetSurfaceSceneView()->GetProjectionType();
            if( projType == ePerspective )
                m_appData->GetSurfaceSceneView()->GetViewSettings()->SetupOpenGLPerspective();
            else if( projType == eOrthographic )
                m_appData->GetSurfaceSceneView()->GetViewSettings()->SetupOpenGLOrtho();
        }


        // Draws the scene.
        void SceneViewWidget::paintGL()
        {
            Timer timer;
            timer.Start();
            //// Is doing the perspective each time we paint too expensive?
            SetupProjection();
            glColor3f(0.0, 1.0, 0.0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
            glMatrixMode(GL_MODELVIEW);
            glEnable(GL_DEPTH_TEST);
            glLoadIdentity();

            const WorldPoint& eye = m_appData->GetSurfaceSceneView()->GetEye();
            const WorldPoint& lookAt = m_appData->GetSurfaceSceneView()->GetLookAt();
            const WorldVector& up = m_appData->GetSurfaceSceneView()->GetUp();

            gluLookAt(eye.x(), eye.y(), eye.z(),
                lookAt.x(), lookAt.y(), lookAt.z(),
                up.x(), up.y(), up.z());


            // Draw OptiX/Cuda first so they don't have to worry about previously
            // rendered depth buffers.
            m_appData->GetSurfaceSceneView()->Draw();
            m_appData->GetSurfaceSceneView()->DisplayBuffersToScreen();


            //glGetIntegerv(GL_VIEWPORT, m_impl->viewport.begin());
            //glGetDoublev(GL_MODELVIEW_MATRIX, m_impl->modelview.begin());
            //glGetDoublev(GL_PROJECTION_MATRIX, m_impl->projection.begin());

            //emit sceneDrawn();

//            GLfloat mat_specular[] = {1.0, 1.0, 1.0, 1.0 };
//            GLfloat mat_shininess[] = { 50.0 };

//            GLfloat mat_red[] = {1.0, 0.0, 0.0, 1.0 };
//            GLfloat mat_green[] = {0.0, 1.0, 0.0, 1.0 };
//            GLfloat mat_blue[] = {0.0, 0.0, 1.0, 1.0 };
//            //GLfloat mat_yellow[] = {1.0, 1.0, 0.0, 1.0 };
//            GLfloat mat_black[] = {0.0, 0.0, 0.0, 0.0 };
//            //setupOpenGLLighting();

//            glMaterialfv(GL_FRONT, GL_AMBIENT, mat_black);
//            glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_red);
//            glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
//            glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);



//            glTranslatef(4.0, 0.0, 0.0);
//            glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_green);
//            glutSolidSphere(1.0, 1000, 1000);

//            glTranslatef(-2.0, 2.0, 0.0);
//            glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_blue);
//            glutSolidSphere(1.0, 1000, 1000);

//            glTranslatef(0.0, -4.0, 0.0);
//            glutSolidSphere(.99, 1000, 1000);

            //glTranslatef(0.0, -4.0, 0.0);
            //glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_yellow);
            //glutSolidSphere(1.0, 1000, 1000);

            if( m_appData->GetShowBoundingBox() )
            {
                boost::shared_ptr<Model> model = m_appData->GetScene()->GetModel();
                if( model )
                {
                    WorldPoint minExtent = model->MinExtent();
                    WorldPoint maxExtent = model->MaxExtent();

                    glColor3f(0.0, 0.0, 0.0);
                    glBegin(GL_LINES);
                        glVertex3f(minExtent.x(), minExtent.y(), minExtent.z());
                        glVertex3f(minExtent.x(), maxExtent.y(), minExtent.z());

                        glVertex3f(minExtent.x(), minExtent.y(), minExtent.z());
                        glVertex3f(maxExtent.x(), minExtent.y(), minExtent.z());

                        glVertex3f(maxExtent.x(), minExtent.y(), minExtent.z());
                        glVertex3f(maxExtent.x(), maxExtent.y(), minExtent.z());

                        glVertex3f(minExtent.x(), maxExtent.y(), minExtent.z());
                        glVertex3f(maxExtent.x(), maxExtent.y(), minExtent.z());

                        glVertex3f(minExtent.x(), minExtent.y(), maxExtent.z());
                        glVertex3f(minExtent.x(), maxExtent.y(), maxExtent.z());

                        glVertex3f(minExtent.x(), minExtent.y(), maxExtent.z());
                        glVertex3f(maxExtent.x(), minExtent.y(), maxExtent.z());

                        glVertex3f(maxExtent.x(), minExtent.y(), maxExtent.z());
                        glVertex3f(maxExtent.x(), maxExtent.y(), maxExtent.z());

                        glVertex3f(minExtent.x(), maxExtent.y(), maxExtent.z());
                        glVertex3f(maxExtent.x(), maxExtent.y(), maxExtent.z());


                        glVertex3f(minExtent.x(), minExtent.y(), minExtent.z());
                        glVertex3f(minExtent.x(), minExtent.y(), maxExtent.z());

                        glVertex3f(minExtent.x(), maxExtent.y(), minExtent.z());
                        glVertex3f(minExtent.x(), maxExtent.y(), maxExtent.z());

                        glVertex3f(maxExtent.x(), minExtent.y(), minExtent.z());
                        glVertex3f(maxExtent.x(), minExtent.y(), maxExtent.z());

                        glVertex3f(maxExtent.x(), maxExtent.y(), minExtent.z());
                        glVertex3f(maxExtent.x(), maxExtent.y(), maxExtent.z());
                    glEnd();

                }
            }

            if( m_appData->GetShowLookAtPoint() )
            {
                glColor3f(1.0, 1.0, 1.0);
                glBegin(GL_LINES);
                    glColor3f(1.0, 0.0, 0.0);
                    glVertex3f(lookAt.x(), lookAt.y(), lookAt.z());
                    glVertex3f(lookAt.x()+1, lookAt.y(), lookAt.z());

                    glColor3f(1.0, 1.0, 0.0);
                    glVertex3f(lookAt.x(), lookAt.y(), lookAt.z());
                    glVertex3f(lookAt.x(), lookAt.y()+1, lookAt.z());

                    glColor3f(0.0, 1.0, 0.0);
                    glVertex3f(lookAt.x(), lookAt.y(), lookAt.z());
                    glVertex3f(lookAt.x(), lookAt.y(), lookAt.z()+1);

                glEnd();
            }

            //GLdouble modelview[16];
            //GLdouble projection[16];


//            glGetDoublev(GL_MODELVIEW_MATRIX, &modelview[0]);
//            glGetDoublev(GL_PROJECTION_MATRIX, &projection[0]);

//            std::cout << "Modelview matrix" << std::endl;
//            std::cout << modelview[0] << ", " << modelview[4] << ", " << modelview[8] << ", " << modelview[12] << std::endl;
//            std::cout << modelview[1] << ", " << modelview[5] << ", " << modelview[9] << ", " << modelview[13] << std::endl;
//            std::cout << modelview[2] << ", " << modelview[6] << ", " << modelview[10] << ", " << modelview[14] << std::endl;
//            std::cout << modelview[3] << ", " << modelview[7] << ", " << modelview[11] << ", " << modelview[15] << std::endl;

//            std::cout << "Projection matrix" << std::endl;
//            std::cout << projection[0] << ", " << projection[4] << ", " << projection[8] << ", " << projection[12] << std::endl;
//            std::cout << projection[1] << ", " << projection[5] << ", " << projection[9] << ", " << projection[13] << std::endl;
//            std::cout << projection[2] << ", " << projection[6] << ", " << projection[10] << ", " << projection[14] << std::endl;
//            std::cout << projection[3] << ", " << projection[7] << ", " << projection[11] << ", " << projection[15] << std::endl;

            glFlush();
            timer.Stop();
            //double singleFrameTime = timer.TimePerTest(1);
            //std::cout << "Time to draw frame is " << singleFrameTime << " seconds, or " << 1.0/singleFrameTime << " fps." << std::endl;
        }


        void SceneViewWidget::GetModelViewMatrixForVisual3(double* out)
        {


            // x and y in visual 3 is (0,0), so adjust by the lookat point.
            boost::shared_ptr<Camera> camera = m_appData->GetSurfaceSceneView()->GetViewSettings();
            boost::shared_ptr<Model> model = m_appData->GetSurfaceSceneView()->GetScene()->GetModel();
            const WorldPoint& center = model->GetMidpoint();


//            // Scale.
//            double scale = 1.0/(distanceBetween(center, camera->GetEye()));
//            out[0] *= scale;
//            out[5] *= scale;
//            out[10] *= scale;

            glPushMatrix();
            glLoadIdentity();

            const WorldPoint& eye = m_appData->GetSurfaceSceneView()->GetEye();
            const WorldPoint& lookAt = m_appData->GetSurfaceSceneView()->GetLookAt();
            const WorldVector& up = m_appData->GetSurfaceSceneView()->GetUp();

            gluLookAt(-eye.x(), -eye.y(), -eye.z(),
                lookAt.x()-center.x(), lookAt.y()-center.y(), lookAt.z()-center.z(),
                -up.x(), -up.y(), -up.z());

            glGetDoublev(GL_MODELVIEW_MATRIX, out);
//            out[12] += center.x();
//            out[13] += center.y();
            glPopMatrix();

            double mv_buf[16];
            double persp_buf[16];

            glGetDoublev(GL_MODELVIEW_MATRIX, mv_buf);
            glGetDoublev(GL_PROJECTION_MATRIX, persp_buf);

            ElVis::Matrix4x4 mv;
            mv[0] = mv_buf[0];
            mv[1] = mv_buf[4];
            mv[2] = mv_buf[8];
            mv[3] = mv_buf[12];

            mv[4] = mv_buf[1];
            mv[5] = mv_buf[5];
            mv[6] = mv_buf[9];
            mv[7] = mv_buf[13];

            mv[8] = mv_buf[2];
            mv[9] = mv_buf[6];
            mv[10] = mv_buf[10];
            mv[11] = mv_buf[14];

            mv[12] = mv_buf[3];
            mv[13] = mv_buf[7];
            mv[14] = mv_buf[11];
            mv[15] = mv_buf[15];



            ElVis::Matrix4x4 p;
            p[0] = persp_buf[0];
            p[1] = persp_buf[4];
            p[2] = persp_buf[8];
            p[3] = persp_buf[12];

            p[4] = persp_buf[1];
            p[5] = persp_buf[5];
            p[6] = persp_buf[9];
            p[7] = persp_buf[13];

            p[8] = persp_buf[2];
            p[9] = persp_buf[6];
            p[10] = persp_buf[10];
            p[11] = persp_buf[14];

            p[12] = persp_buf[3];
            p[13] = persp_buf[7];
            p[14] = persp_buf[11];
            p[15] = persp_buf[15];

            ElVis::Matrix4x4 trial1 = p*mv;
            ElVis::Matrix4x4 trial2 = mv*p;

            std::cout << "Matrix 1" << std::endl;
            std::cout << trial1[0] << " " << trial1[1] << " " << trial1[2] << " " << trial1[3] << std::endl;
            std::cout << trial1[4] << " " << trial1[5] << " " << trial1[6] << " " << trial1[7] << std::endl;
            std::cout << trial1[8] << " " << trial1[9] << " " << trial1[10] << " " << trial1[11] << std::endl;
            std::cout << trial1[12] << " " << trial1[13] << " " << trial1[14] << " " << trial1[15] << std::endl;

            std::cout << "Matrix 2" << std::endl;
            std::cout << trial2[0] << " " << trial2[1] << " " << trial2[2] << " " << trial2[3] << std::endl;
            std::cout << trial2[4] << " " << trial2[5] << " " << trial2[6] << " " << trial2[7] << std::endl;
            std::cout << trial2[8] << " " << trial2[9] << " " << trial2[10] << " " << trial2[11] << std::endl;
            std::cout << trial2[12] << " " << trial2[13] << " " << trial2[14] << " " << trial2[15] << std::endl;
        }

        void SceneViewWidget::mousePressEvent(QMouseEvent* event)
        {
            if( event->modifiers() & Qt::ShiftModifier )
            {
                m_appData->GeneratePointInformation(event->pos().x(), event->pos().y());
            }
            else if( event->modifiers() & Qt::ControlModifier )
            {
                m_appData->SetLookAtPointToIntersectionPoint(event->pos().x(), event->pos().y());
            }
            else
            {
                m_moveStartPixel = event->pos();
            }
            event->accept();
        }

        void SceneViewWidget::mouseReleaseEvent(QMouseEvent* event)
        {
        }

        void SceneViewWidget::mouseDoubleClickEvent(QMouseEvent* event)
        {
        }


        void SceneViewWidget::wheelEvent(QWheelEvent* event)
        {
            m_appData->GetSurfaceSceneView()->GetViewSettings()->MoveEyeAlongGaze(0, 0,
                                                event->delta(), event->delta(),
                                                this->width(), this->height());

            event->accept();
        }

        void SceneViewWidget::HandleSceneViewChanged(const SceneView&)
        {
            update();
        }

        void SceneViewWidget::HandleSceneChanged(const Scene&)
        {
            update();
        }

        void SceneViewWidget::mouseMoveEvent(QMouseEvent* event)
        {
            bool rotateMode = false;
            bool zoomMode = false;
            bool panMode = false;

            if( event->buttons() & Qt::RightButton )
            {
                zoomMode = true;
            }
            else if( event->buttons() & Qt::LeftButton )
            {
                if( event->modifiers() & Qt::ShiftModifier )
                {
                    panMode = true;
                }
                else
                {
                    rotateMode = true;
                }
            }
            else if( event->buttons() & Qt::MidButton )
            {
                panMode = true;
            }

            if( rotateMode )
            {
                m_appData->GetSurfaceSceneView()->GetViewSettings()->Rotate(m_moveStartPixel.x(), m_moveStartPixel.y(),
                                                       event->pos().x(), event->pos().y(),
                                                       this->width(), this->height());
                m_moveStartPixel = event->pos();
            }
            else if( zoomMode )
            {
                m_appData->GetSurfaceSceneView()->GetViewSettings()->MoveEyeAlongGaze(m_moveStartPixel.x(), m_moveStartPixel.y(),
                                                                 event->pos().x(), event->pos().y(),
                                                                 this->width(), this->height());
                m_moveStartPixel = event->pos();
            }
            else if( panMode )
            {

                m_appData->GetSurfaceSceneView()->GetViewSettings()->Pan(m_moveStartPixel.x(), m_moveStartPixel.y(),
                                                    event->pos().x(), event->pos().y(),
                                                    this->width(), this->height());
                m_moveStartPixel = event->pos();
            }

        }


        //void SceneViewWidget::setupOpenGLLighting()
        //{
        //    // Push the modelview matrix and reset it so that the lights are
        //    // not influenced by whatever state we happen to have.
        //    //glPushMatrix();
        //    //glLoadIdentity();

        //    // First, disable everything.
        //    for(unsigned int i = 0; i < sizeof(SceneViewImpl::LIGHT_ENUM_VALUES)/sizeof(GLenum); ++i)
        //    {
        //        glDisable(SceneViewImpl::LIGHT_ENUM_VALUES[i]);
        //    }

        //    glEnable(GL_LIGHTING);
        //    const rt::RGBColor& globalAmbient = m_impl->shadingModel->getAmbientLightColor();
        //    GLfloat global_ambient[] = {globalAmbient.red(), globalAmbient.green(), globalAmbient.blue(), globalAmbient.alpha() };
        //    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, global_ambient);

        //    glShadeModel(m_impl->shadingModel->getShadeMode());

        //    const rt::ShadingModel::LightVector& lights = m_impl->shadingModel->getLights();
        //    for(unsigned int i = 0; i < lights.size(); ++i)
        //    {
        //        GLenum lightEnum = SceneViewImpl::LIGHT_ENUM_VALUES[i];
        //        glEnable(lightEnum);
        //        boost::shared_ptr<rt::Material> lightColor = lights[i]->getLightColor();

        //        const rt::RGBColor& ambient = lightColor->ambient();
        //        const rt::RGBColor& diffuse = lightColor->diffuse();
        //        const rt::RGBColor& specular = lightColor->specular();
        //        const rt::WorldPoint& position = lights[i]->getPosition();

        //        GLfloat light_ambient[] = { ambient.red(), ambient.green(), ambient.blue(), ambient.alpha() };
        //        GLfloat light_diffuse[] = { diffuse.red(), diffuse.green(), diffuse.blue(), diffuse.alpha() };
        //        GLfloat light_specular[] = { specular.red(), specular.green(), specular.blue(), specular.alpha() };
        //        GLfloat light_position[] = { position.x(), position.y(), position.z(), 1.0 };

        //        if( lights[i]->isDirectional() )
        //        {
        //            light_position[3] = 0.0;
        //        }

        //        glLightfv(lightEnum, GL_AMBIENT, light_ambient);
        //        glLightfv(lightEnum, GL_DIFFUSE, light_diffuse);
        //        glLightfv(lightEnum, GL_SPECULAR, light_specular);
        //        glLightfv(lightEnum, GL_POSITION, light_position);
        //    }

        //    const rt::RGBColor& backColor = m_impl->shadingModel->getBackgroundColor();
        //    glClearColor(backColor.red(), backColor.green(), backColor.blue(), 1.0);

        //    if( m_impl->shadingModel->twoSidedLightingEnabled() )
        //    {
        //        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
        //    }
        //    else
        //    {
        //        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
        //    }

        //    if( m_impl->shadingModel->viewerIsLocal() )
        //    {
        //        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);
        //    }
        //    else
        //    {
        //        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_FALSE);
        //    }
        //    //glPopMatrix();
        //}


        //rt::WorldPoint SceneViewWidget::screenPointToWorldPoint(unsigned int row, unsigned int col) const
        //{
        //    // We need the middle of the pixel.

        //    rt::WorldVector leftToRight = leftToRightDirection();
        //    rt::WorldVector bottomToTop = bottomToTopDirection();

        //    double worldWidth = leftToRight.Magnitude();
        //    double worldHeight = bottomToTop.Magnitude();

        //    leftToRight.Normalize();
        //    bottomToTop.Normalize();

        //    double pixelWidth = worldWidth/width();
        //    double pixelHeight = worldHeight/height();

        //    rt::WorldPoint targetPoint = lowerLeftPoint() + 
        //        findPointAlongVector(leftToRight, pixelWidth*col + pixelWidth/2.0);
        //    targetPoint = targetPoint + 
        //        findPointAlongVector(bottomToTop, pixelHeight*row + pixelHeight/2.0);
        //    return targetPoint;
        //    ////return targetPoint;

        //    //rt::WorldPoint targetPoint = lowerLeftPoint() + rt::findPointAlongVector(leftToRightDirection(), (double)col/width());
        //    //targetPoint = targetPoint + rt::findPointAlongVector(bottomToTopDirection(), (double)row/height());
        //    //return targetPoint;
        //}

        //rt::WorldVector SceneViewWidget::leftToRightDirection() const
        //{
        //    return createVectorFromPoints(lowerLeftPoint(), lowerRightPoint());
        //}

        //rt::WorldVector SceneViewWidget::topToBottomDirection() const
        //{
        //    return createVectorFromPoints(upperRightPoint(), lowerRightPoint());
        //}

        //rt::WorldVector SceneViewWidget::bottomToTopDirection() const
        //{
        //    return createVectorFromPoints(lowerRightPoint(), upperRightPoint());
        //}

        //boost::array<GLint, 4> SceneViewWidget::getOpenGLViewport() const
        //{
        //    return m_impl->viewport;
        //    //boost::array<GLint, 4> result;
        //    //glGetIntegerv(GL_VIEWPORT, result.begin());
        //    //return result;
        //}

        //boost::array<GLdouble, 16> SceneViewWidget::getOpenGLModelViewMatrix() const
        //{
        //    return m_impl->modelview;
        //    //boost::array<GLdouble, 16> result;
        //    //glGetDoublev(GL_MODELVIEW_MATRIX, result.begin());
        //    //return result;
        //}

        //boost::array<GLdouble, 16> SceneViewWidget::getOpenGLProjectionMatrix() const
        //{
        //    return m_impl->projection;
        //    //boost::array<GLdouble, 16> result;
        //    //glGetDoublev(GL_PROJECTION_MATRIX, result.begin());
        //    //return result;
        //}
        //  
        //GLfloat SceneViewWidget::getDepthClearValue() const
        //{
        //    return m_impl->clearDepth;
        //}
    }
}
