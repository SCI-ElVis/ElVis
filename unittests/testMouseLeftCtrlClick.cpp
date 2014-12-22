#include "../src/ElVis/Gui/SceneViewWidget.h"
#include "../src/ElVis/Gui/ApplicationState.h"
#include <QDockWidget>
#include <QFileDialog>
#include <QStringList>
#include <QMenuBar>
#include <QEvent>
#include <QMouseEvent>
#include <QtTest/QtTest>

#include <iostream>
#include <string>
using namespace std;
using namespace Qt;

//Fix this path later
//#include <boost/serialization/shared_ptr.hpp>
#include "../src/Externals/boost/boost/serialization/shared_ptr.hpp"

QTEST_MAIN(TestGui);
#include "testMouseLeftCtrlClick.moc"

QPoint ApplicationStateMock_storedPoint;

namespace ElVis
{
  namespace Gui
  {

    ApplicationState::ApplicationState(void)
    {
      ApplicationStateMock_storedPoint.setX(0);
      ApplicationStateMock_storedPoint.setY(0);
    }
    
    void ApplicationState::SetLookAtPointToIntersectionPoint(unsigned int x,
                                                             unsigned int y)
    {
      ApplicationStateMock_storedPoint.setX(x);
      ApplicationStateMock_storedPoint.setY(y);
    }

    class TestGui : public QObject
    {
      Q_OBJECT

      private slots:
        void testGui();
    }

    void TestGui::testGui()
    {
      int x_coord = 50;
      int y_coord = 50;
      QPoint qp = QPoint(x_coord, y_coord);

      boost::shared_ptr<ApplicationState> appData = boost::shared_ptr<ApplicationState>(new ApplicationState);
      SceneViewWidget svw = SceneViewWidget(appData);
 
      QTest::mousePress(&svw, Qt::LeftButton, Qt::ControlModifier, qp, -1);

      QCOMPARE(qp.x(), ApplicationStateMock_storedPoint.x());
      QCOMPARE(qp.y(), ApplicationStateMock_storedPoint.y());
    }



  }
}
