/////////////////////////////////////////////////////////////////////////////////
////
//// The MIT License
////
//// Copyright (c) 2006 Scientific Computing and Imaging Institute,
//// University of Utah (USA)
////
//// License for the specific language governing rights and limitations under
//// Permission is hereby granted, free of charge, to any person obtaining a
//// copy of this software and associated documentation files (the "Software"),
//// to deal in the Software without restriction, including without limitation
//// the rights to use, copy, modify, merge, publish, distribute, sublicense,
//// and/or sell copies of the Software, and to permit persons to whom the
//// Software is furnished to do so, subject to the following conditions:
////
//// The above copyright notice and this permission notice shall be included
//// in all copies or substantial portions of the Software.
////
//// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//// DEALINGS IN THE SOFTWARE.
////
/////////////////////////////////////////////////////////////////////////////////

//#ifndef ELVIS_GUI_BREAKPOINT_PROPERTY_MANAGER_H
//#define ELVIS_GUI_BREAKPOINT_PROPERTY_MANAGER_H

//#include <ElVis/QtPropertyBrowser/qtpropertymanager.h>
//#include <ElVis/QtPropertyBrowser/qteditorfactory.h>
//#include <ElVis/QtPropertyBrowser/MemberProperty.hpp>
//#include <ElVis/QtPropertyBrowser/MemberPropertyManager.hpp>
//#include <ElVis/Gui/PointPropertyManager.h>
//#include <ElVis/Gui/VectorPropertyManager.h>
//#include <ElVis/Core/Camera.h>

//#include <boost/bind.hpp>
//#include <boost/foreach.hpp>

//namespace ElVis
//{
//    namespace Gui
//    {
//        class BreakpointPropertyManager : public QtAbstractPropertyManager
//        {
//            Q_OBJECT

//            public:
//                BreakpointPropertyManager() :
//                    QtAbstractPropertyManager(),
//                    m_lookAtPropertyManager(new PointPropertyManager()),
//                    m_eyePropertyManager(new PointPropertyManager()),
//                    m_upPropertyManager(new VectorPropertyManager()),
//                    m_groupPropertyManager(new QtGroupPropertyManager()),
//                    m_doublePropertyManager(new QtDoublePropertyManager()),
//                    m_fieldOfViewProperty(0),
//                    m_doubleSpinBoxFactory(new QtDoubleSpinBoxFactory()),
//                    m_camera()
//                {
//                }

//                virtual ~BreakpointPropertyManager() {}

//                // Hide AddProperty on purpose.
//                QtProperty* addProperty(const QString& name, boost::shared_ptr<Camera> obj)
//                {
//                    QtProperty* topLevelItem = m_groupPropertyManager->addProperty(name);

//                    QtProperty* lookAtProperty = m_lookAtPropertyManager->addProperty("Look At", obj->GetLookAt());
//                    QtProperty* eyeProperty = m_eyePropertyManager->addProperty("Eye", obj->GetEye());
//                    QtProperty* upProperty = m_upPropertyManager->addProperty("Up", obj->GetUp());

//                    m_fieldOfViewProperty = m_doublePropertyManager->addProperty("FOV");
//                    m_doublePropertyManager->setValue(m_fieldOfViewProperty, obj->GetFieldOfView());

//                    connect(m_doublePropertyManager, SIGNAL(valueChanged(QtProperty*,double)), this, SLOT(HandleFOVChanged(QtProperty*,double)));

//                    topLevelItem->addSubProperty(eyeProperty);
//                    topLevelItem->addSubProperty(lookAtProperty);
//                    topLevelItem->addSubProperty(upProperty);
//                    topLevelItem->addSubProperty(m_fieldOfViewProperty);

//                    m_camera = obj;
//                    return topLevelItem;
//                }

//                void SetupPropertyManagers(QtAbstractPropertyBrowser* browser, QObject* parent)
//                {
//                    browser->setFactoryForManager(m_doublePropertyManager, m_doubleSpinBoxFactory);
//                    m_lookAtPropertyManager->SetupPropertyManagers(browser, parent);
//                    m_eyePropertyManager->SetupPropertyManagers(browser, parent);
//                    m_upPropertyManager->SetupPropertyManagers(browser, parent);
//                }

//            protected Q_SLOTS:
//                void HandleFOVChanged(QtProperty* property, double value)
//                {
//                    if( !m_camera ) return;

//                    m_camera->SetFieldOfView(value);
//                }

//            protected:

//                virtual void initializeProperty(QtProperty *property)
//                {

//                }

//                virtual QtProperty* createProperty()
//                {
//                    return 0;
//                }

//            private:
//                CameraPropertyManager(const CameraPropertyManager& rhs);
//                CameraPropertyManager& operator=(const CameraPropertyManager& rhs);

//                PointPropertyManager* m_lookAtPropertyManager;
//                PointPropertyManager* m_eyePropertyManager;
//                VectorPropertyManager* m_upPropertyManager;
//                QtGroupPropertyManager* m_groupPropertyManager;
//                QtDoublePropertyManager* m_doublePropertyManager;
//                QtProperty* m_fieldOfViewProperty;
//                QtDoubleSpinBoxFactory* m_doubleSpinBoxFactory;
//                boost::shared_ptr<Camera> m_camera;
//        };
//    }
//}

//#endif
