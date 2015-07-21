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

#include <ElVis/Gui/ElVisUI.h>

#include <QDockWidget>
#include <QFileDialog>
#include <QStringList>
#include <QMenuBar>

#include <iostream>

#include <boost/filesystem/path.hpp>
#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>

#include <ElVis/Gui/SceneViewWidget.h>
#include <ElVis/Gui/DebugSettingsDockWidget.h>

#include <ElVis/Extensions/JacobiExtension/JacobiExtensionElVisModel.h>
#include <ElVis/Core/Camera.h>
#include <ElVis/Core/ColorMap.h>
#include <ElVis/Core/Camera.h>
#include <ElVis/Core/Point.hpp>
#include <ElVis/Core/Scene.h>
#include <ElVis/Core/Light.h>
#include <ElVis/Core/Color.h>
#include <ElVis/Core/Light.h>
#include <ElVis/Core/Triangle.h>
#include <ElVis/Core/SurfaceObject.h>
#include <ElVis/Core/Plane.h>
#include <ElVis/Core/LightingModule.h>
#include <ElVis/Core/PrimaryRayModule.h>
#include <ElVis/Core/CutSurfaceContourModule.h>
#include <ElVis/Core/SurfaceObject.h>
#include <ElVis/Core/Cylinder.h>
#include <ElVis/Core/SampleVolumeSamplerObject.h>

#include <string>
#include <boost/bind.hpp>
#include <boost/timer.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include <ElVis/Core/Cylinder.h>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <ElVis/Core/ColorMapperModule.h>
#include <ElVis/Core/SampleVolumeSamplerObject.h>
#include <ElVis/Core/PtxManager.h>

#include <tinyxml.h>

#include <boost/serialization/shared_ptr.hpp>

namespace ElVis
{
  namespace Gui
  {
    const char* ElVisUI::DEFAULT_MODEL_DIR_SETTING_NAME = "DefaultModelDir";
    const char* ElVisUI::DEFAULT_SCREENSHOT_DIR_SETTING_NAME = "DefaultScreenshotDir";
    const char* ElVisUI::DEFAULT_STATE_DIR_SETTING_NAME = "DefaultStateDir";

    const char* ElVisUI::RECENT_FILE_LIST = "RecentFileList";
    const char* ElVisUI::RECENT_FILE_FILTER = "RecentFileFilter";

    const char* ElVisUI::STATE_SUFFIX = ".xml";

    const QString ElVisUI::WindowsSettings("WindowsSettings");
    const QString ElVisUI::OptixStackSize("OptixStackSize");

    namespace
    {
      // Serialization constants
      const std::string SCENE_VIEW_ELEMENT_NAME("SceneView");
    }

    ElVisUI::ElVisUI()
      : QMainWindow(),
        m_mdiArea(0),
        m_appData(new ApplicationState()),
        m_sceneViewWidget(new SceneViewWidget(m_appData)),
        m_viewSettings(new ViewSettingsUI(m_appData)),
        m_objectInspector(new ObjectInspectorUI(m_appData)),
        m_modelInspectorWidget(0),
        m_colorMapWidget(new ColorMapWidget(m_appData)),
        m_renderSettingsWidget(0),
        m_debugDockWidget(new DebugSettingDockWidget(m_appData)),
        m_volumeRenderingSettings(new VolumeRenderingSettingsWidget(m_appData)),
        m_elementRenderingDockWidget(new ElementFaceRenderingDockWidget(m_appData)),
        m_contourDockWidget(new ContourDockWidget(m_appData)),
        m_isosurfaceDockWidget(new IsosurfaceDockWidget(m_appData)),
        m_lightingDockWidget(new LightingDockWidget(m_appData)),
        m_sampleDockWidget(new SampleOntoNrrdDockWidget(m_appData)),
        m_plugins(),
        m_pluginLocationData(),
        m_errorNotification(new QErrorMessage(0)),
        m_mainToolbar(0),
        m_sceneItems(0),
        m_settings(new QSettings("ElVis.ini", QSettings::IniFormat)),
        m_actionExit(0),
        m_actionLoadPlugin(0),
        m_actionLoadModel(0),
        m_actionCreateCutPlane(0),
        m_actionSaveScreenshot(0),
        m_actionExportVisual3View(0),
        m_loadViewAction(0),
        m_saveViewAction(0),
        m_calculateScalarRangeAction(0),
        m_menuBar(0),
        m_fileMenu(0),
        m_pluginMenu(0),
        m_sourcesMenu(0),
        m_windowMenu(0),
        m_utilsMenu(0),
        m_separatorAct(0),
        m_recentFileActs(),
        m_fieldLabel(new QLabel("Field: ")),
        m_fieldComboBox(new QComboBox()),
        m_initializing(true)
    {
      //            m_mdiArea = new QMdiArea();
      //            m_mdiArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
      //            m_mdiArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

      this->setCentralWidget(m_sceneViewWidget.get());

      CreateActions();
      CreateMenus();

      // The setting populate menus, so must appear after they are created.
      ReadSettings();

      CreateToolBars();
      SetupSceneViews();
      SetupDocking();

      m_appData->GetScene()->OnModelChanged.connect(boost::bind(&ElVisUI::HandleModelChanged, this, _1));

      this->setUnifiedTitleAndToolBarOnMac(true);
      // m_mdiArea->addSubWindow(m_sceneViewWidget.get());

      // m_sceneViewWidget->showMaximized();

      LoadDefaultColorMaps();
      this->setWindowTitle("ElVis");

      //            // Styling
      //            std::string fileData;
      //            std::ifstream inFile("ElVis.qss");
      //            while( inFile )
      //            {
      //                std::string data;
      //                std::getline(inFile, data);

      //                fileData = fileData + data;
      //            }
      //            inFile.close();
      //            this->setStyleSheet(fileData.c_str());

      this->show();
      m_initializing = false;
    }

    ElVisUI::~ElVisUI() {}

    void ElVisUI::SetupSceneViews()
    {
      // Initialization of surface scene view
      {
        boost::shared_ptr<PrimaryRayModule> primaryRayModule = m_appData->GetSurfaceSceneView()->GetPrimaryRayModule();
        primaryRayModule->OnObjectAdded.connect(boost::bind(&ElVisUI::HandlePrimaryRayObjectAdded, this, _1));
      }
    }

    void ElVisUI::HandlePrimaryRayObjectAdded(boost::shared_ptr<PrimaryRayObject> obj) {}

    void ElVisUI::closeEvent(QCloseEvent* event)
    {
      WriteSettings();
      delete m_settings;
      m_settings = 0;
    }

    void ElVisUI::Exit() {}

    void ElVisUI::CreateMenus()
    {
      m_menuBar = new QMenuBar(0);
      this->setMenuBar(m_menuBar);

      m_fileMenu = menuBar()->addMenu(tr("&File"));
      m_fileMenu->addAction(m_actionLoadModel);
      m_fileMenu->addAction(m_actionSaveScreenshot);
      m_fileMenu->addAction(m_loadViewAction);
      m_fileMenu->addAction(m_saveViewAction);

      m_separatorAct = m_fileMenu->addSeparator();
      for (int i = 0; i < MaxRecentFiles; ++i)
      {
        m_fileMenu->addAction(m_recentFileActs[i]);
      }
      m_fileMenu->addSeparator();
      m_fileMenu->addAction(m_actionExit);

      m_pluginMenu = menuBar()->addMenu(tr("&Plugins"));
      m_pluginMenu->addAction(m_actionLoadPlugin);

      m_sourcesMenu = menuBar()->addMenu(tr("&Sources"));
      m_sourcesMenu->addAction(m_actionCreateCutPlane);

      m_utilsMenu = menuBar()->addMenu(tr("&Utilities"));
      m_utilsMenu->addAction(m_actionExportVisual3View);
      m_utilsMenu->addAction(m_calculateScalarRangeAction);

      m_windowMenu = menuBar()->addMenu(tr("&Windows"));
    }

    void ElVisUI::CreateToolBars()
    {
      m_mainToolbar = addToolBar("Toolbar");
      m_mainToolbar->setObjectName("Toolbar");
      m_mainToolbar->addWidget(m_fieldLabel);
      m_mainToolbar->addWidget(m_fieldComboBox);

      connect(m_fieldComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(HandleFieldSelectionChanged(int)));
    }

    void ElVisUI::CreateActions()
    {
      m_actionExit = new QAction(tr("E&xit"), this);
      m_actionExit->setShortcuts(QKeySequence::Quit);
      m_actionExit->setStatusTip(tr("Exit the application"));
      connect(m_actionExit, SIGNAL(triggered()), this, SLOT(close()));

      m_actionLoadPlugin = new QAction(tr("&Load Plugin"), this);
      m_actionLoadPlugin->setStatusTip(tr("Load a plugin."));
      connect(m_actionLoadPlugin, SIGNAL(triggered()), this, SLOT(LoadPlugin()));

      m_actionLoadModel = new QAction(tr("&Load Model"), this);
      m_actionLoadModel->setStatusTip(tr("Load a model."));
      connect(m_actionLoadModel, SIGNAL(triggered()), this, SLOT(LoadVolume()));

      m_actionCreateCutPlane = new QAction(tr("Create Cut &Plane"), this);
      m_actionCreateCutPlane->setStatusTip(tr("Create a cut-plane"));
      connect(m_actionCreateCutPlane, SIGNAL(triggered()), this, SLOT(CreateCutPlane()));

      for (int i = 0; i < MaxRecentFiles; ++i)
      {
        m_recentFileActs[i] = new QAction(this);
        m_recentFileActs[i]->setVisible(false);
        connect(m_recentFileActs[i], SIGNAL(triggered()), this, SLOT(HandleOpenRecentFile()));
      }

      m_actionSaveScreenshot = new QAction(tr("Save Screenshot"), this);
      m_actionSaveScreenshot->setStatusTip("Save screenshot in png format.");
      connect(m_actionSaveScreenshot, SIGNAL(triggered()), this, SLOT(SaveScreenshot()));

      m_actionExportVisual3View = new QAction(tr("Export View Settings"), this);
      m_actionExportVisual3View->setStatusTip("Export view settings for Visual 3");
      connect(m_actionExportVisual3View, SIGNAL(triggered()), this, SLOT(ExportViewSettings()));

      m_loadViewAction = new QAction(tr("Load View Settings"), this);
      m_loadViewAction->setStatusTip("Load camera settings from a file.");
      connect(m_loadViewAction, SIGNAL(triggered()), this, SLOT(LoadState()));

      m_saveViewAction = new QAction(tr("Save View Settings"), this);
      m_saveViewAction->setStatusTip("Save camera settings to a file.");
      connect(m_saveViewAction, SIGNAL(triggered()), this, SLOT(SaveState()));

      m_calculateScalarRangeAction = new QAction(tr("Calculate Scalar Range"), this);
      m_calculateScalarRangeAction->setStatusTip("Calcualte scalar range of samples.");
      connect(m_calculateScalarRangeAction, SIGNAL(triggered()), this, SLOT(CalculateScalarRange()));
    }

    void ElVisUI::CalculateScalarRange()
    {
      Stat result = m_appData->GetSurfaceSceneView()->CalculateScalarSampleStats();
      printf("Min %2.15f, Max %2.15f\n", result.Min, result.Max);
    }

    void ElVisUI::HandleOpenRecentFile()
    {
      QAction* action = qobject_cast<QAction*>(sender());
      if (action)
      {
        QStringList data = action->data().toStringList();
        LoadFile(data[0], data[1]);
      }
    }

    void ElVisUI::CreateCutPlane()
    {
      if (!m_appData->GetScene()->GetModel())
      {
        return;
      }

      boost::shared_ptr<ElVis::Model> model = m_appData->GetScene()->GetModel();

      const WorldPoint& minExtent = model->MinExtent();
      const WorldPoint& maxExtent = model->MaxExtent();
      double x_mid = (maxExtent.x() + minExtent.x()) / 2.0;
      double y_mid = (maxExtent.y() + minExtent.y()) / 2.0;
      double z_mid = (maxExtent.z() + minExtent.z()) / 2.0;

      // Default to a cut plane that spans the volume with a normal (0,0,1).
      WorldPoint normal(0.0, 0.0, 1.0);
      WorldPoint p(x_mid, y_mid, z_mid);
      boost::shared_ptr<ElVis::Plane> cutPlane(new ElVis::Plane(normal, p));

      boost::shared_ptr<ElVis::PrimaryRayModule> primaryRayModule =
        m_appData->GetSurfaceSceneView()->GetPrimaryRayModule();

      boost::shared_ptr<ElVis::SampleVolumeSamplerObject> sampler(new ElVis::SampleVolumeSamplerObject(cutPlane));
      primaryRayModule->AddObject(sampler);
    }

    void ElVisUI::WriteSettings()
    {
      if (m_initializing) return;

      m_settings->beginGroup("MainWindow");
      m_settings->setValue("size", size());
      m_settings->setValue("pos", pos());
      m_settings->endGroup();

      if (m_pluginLocationData.size() > 0)
      {
        m_settings->beginWriteArray("Plugins");

        int i = 0;
        for (std::map<std::string, PluginLocationData>::const_iterator iter = m_pluginLocationData.begin();
             iter != m_pluginLocationData.end(); ++iter)
        {
          m_settings->setArrayIndex(i);
          const PluginLocationData& data = (*iter).second;
          const std::string& pluginName = (*iter).first;
          m_settings->setValue("PluginName", pluginName.c_str());
          m_settings->setValue("Path", data.Path.string().c_str());
          m_settings->setValue("Loaded", data.Loaded);
          ++i;
        }
        m_settings->endArray();
      }

      m_settings->beginGroup("OptiX");
      m_settings->setValue(OptixStackSize, m_appData->GetScene()->GetOptixStackSize());
      m_settings->endGroup();

      m_settings->setValue(WindowsSettings, saveState());
    }

    void ElVisUI::ReadSettings()
    {
      m_settings->beginGroup("MainWindow");
      resize(m_settings->value("size", QSize(400, 400)).toSize());
      move(m_settings->value("pos", QPoint(200, 200)).toPoint());
      m_settings->endGroup();

      int size = m_settings->beginReadArray("Plugins");
      for (int i = 0; i < size; ++i)
      {
        m_settings->setArrayIndex(i);
        std::string pluginName = std::string(m_settings->value("PluginName").toString().toAscii().constData());
        boost::filesystem::path path(m_settings->value("Path").toString().toAscii().constData());
        bool loaded = m_settings->value("Loaded").toBool();

        AddPlugin(path, loaded, pluginName);
      }
      m_settings->endArray();
      UpdateRecentFileActions();

      m_settings->beginGroup("OptiX");
      m_appData->GetScene()->SetOptixStackSize(
        m_settings->value(OptixStackSize, m_appData->GetScene()->GetOptixStackSize()).toInt());
      m_settings->endGroup();
    }

    void ElVisUI::UpdateRecentFileActions()
    {
      QStringList files = m_settings->value(RECENT_FILE_LIST).toStringList();
      QStringList filters = m_settings->value(RECENT_FILE_FILTER).toStringList();

      int numRecentFiles = qMin(files.size(), (int)MaxRecentFiles);

      for (int i = 0; i < numRecentFiles; ++i)
      {
        QString text = tr("&%1 %2").arg(i + 1).arg(StrippedName(files[i]));
        m_recentFileActs[i]->setText(text);

        QStringList data;
        data.append(files[i]);
        data.append(filters[i]);
        m_recentFileActs[i]->setData(data);
        m_recentFileActs[i]->setVisible(true);
      }
      for (int j = numRecentFiles; j < MaxRecentFiles; ++j)
      {
        m_recentFileActs[j]->setVisible(false);
      }

      m_separatorAct->setVisible(numRecentFiles > 0);
    }

    void ElVisUI::SetCurrentFile(const QString& fileName, const QString& filterName)
    {
      QString curFile = fileName;
      setWindowFilePath(curFile);

      QStringList files = m_settings->value(RECENT_FILE_LIST).toStringList();
      QStringList filters = m_settings->value(RECENT_FILE_FILTER).toStringList();

      if (files.contains(fileName))
      {
        return;
      }

      //            files.removeAll(fileName);
      //            filters.removeAll(filterName);

      files.prepend(fileName);
      filters.prepend(filterName);

      while (files.size() > MaxRecentFiles)
      {
        files.removeLast();
      }

      while (filters.size() > MaxRecentFiles)
      {
        filters.removeLast();
      }

      m_settings->setValue(RECENT_FILE_LIST, files);
      m_settings->setValue(RECENT_FILE_FILTER, filters);

      UpdateRecentFileActions();
    }

    QString ElVisUI::StrippedName(const QString& fullFileName) { return QFileInfo(fullFileName).fileName(); }

    void ElVisUI::SetupDocking()
    {

      this->setDockOptions(QMainWindow::AnimatedDocks);
      this->setDockOptions(QMainWindow::AllowTabbedDocks);

      // QDockWidget::DockWidgetFeatures features =
      //    QDockWidget::DockWidgetMovable|
      //    QDockWidget::DockWidgetFloatable;

      m_sceneItems = new SceneItemsDockWidget(m_appData, this, 0);
      this->addDockWidget(Qt::LeftDockWidgetArea, m_sceneItems);

      this->addDockWidget(Qt::RightDockWidgetArea, m_viewSettings);

      m_renderSettingsWidget = new RenderSettingsDockWidget(m_appData);
      this->addDockWidget(Qt::RightDockWidgetArea, m_renderSettingsWidget);
      this->tabifyDockWidget(m_renderSettingsWidget, m_viewSettings);

      addDockWidget(Qt::RightDockWidgetArea, m_volumeRenderingSettings);
      tabifyDockWidget(m_renderSettingsWidget, m_volumeRenderingSettings);

      m_elementRenderingDockWidget = new ElementFaceRenderingDockWidget(m_appData);
      addDockWidget(Qt::RightDockWidgetArea, m_elementRenderingDockWidget);
      tabifyDockWidget(m_renderSettingsWidget, m_elementRenderingDockWidget);

      addDockWidget(Qt::RightDockWidgetArea, m_contourDockWidget);
      tabifyDockWidget(m_renderSettingsWidget, m_contourDockWidget);

      addDockWidget(Qt::RightDockWidgetArea, m_isosurfaceDockWidget);
      tabifyDockWidget(m_renderSettingsWidget, m_isosurfaceDockWidget);

      addDockWidget(Qt::RightDockWidgetArea, m_lightingDockWidget);
      tabifyDockWidget(m_renderSettingsWidget, m_lightingDockWidget);

      addDockWidget(Qt::RightDockWidgetArea, m_sampleDockWidget);
      tabifyDockWidget(m_renderSettingsWidget, m_sampleDockWidget);

      this->addDockWidget(Qt::LeftDockWidgetArea, m_objectInspector);
      this->addDockWidget(Qt::BottomDockWidgetArea, m_colorMapWidget);

      tabifyDockWidget(m_objectInspector, m_debugDockWidget);

      m_modelInspectorWidget = new ModelInspectorWidget(m_appData->GetScene());
      this->addDockWidget(Qt::RightDockWidgetArea, m_modelInspectorWidget);

      m_windowMenu->addAction(m_viewSettings->toggleViewAction());
      m_windowMenu->addAction(m_objectInspector->toggleViewAction());
      m_windowMenu->addAction(m_modelInspectorWidget->toggleViewAction());
      m_windowMenu->addAction(m_colorMapWidget->toggleViewAction());
      m_windowMenu->addAction(m_renderSettingsWidget->toggleViewAction());
      m_windowMenu->addAction(m_debugDockWidget->toggleViewAction());
      m_windowMenu->addAction(m_sceneItems->toggleViewAction());
      m_windowMenu->addAction(m_volumeRenderingSettings->toggleViewAction());
      m_windowMenu->addAction(m_isosurfaceDockWidget->toggleViewAction());
      m_windowMenu->addAction(m_sampleDockWidget->toggleViewAction());
      m_windowMenu->addAction(m_contourDockWidget->toggleViewAction());
      m_windowMenu->addAction(m_elementRenderingDockWidget->toggleViewAction());
      m_windowMenu->addAction(m_lightingDockWidget->toggleViewAction());

      // Finally, restore positions.
      restoreState(m_settings->value(WindowsSettings).toByteArray());
    }

    void ElVisUI::HandleModelChanged(boost::shared_ptr<Model> model)
    {
      int maxCharacters = 0;
      m_fieldComboBox->clear();
      for (int i = 0; i < model->GetNumFields(); ++i)
      {
        FieldInfo info = model->GetFieldInfo(i);
        m_fieldComboBox->addItem(QString(info.Name.c_str()), QVariant(info.Id));
        maxCharacters = std::max(maxCharacters, static_cast<int>(info.Name.size()));
      }
      m_fieldComboBox->setCurrentIndex(0);
      m_fieldComboBox->setMinimumContentsLength(maxCharacters);
      m_fieldComboBox->setSizeAdjustPolicy(QComboBox::AdjustToContents);
      QSizePolicy p(QSizePolicy::Minimum, QSizePolicy::Fixed);
      m_fieldComboBox->setSizePolicy(p);

      static bool first = true;

      if (!first) return;

      first = false;
    }

    void ElVisUI::HandleFieldSelectionChanged(int index)
    {
      int fieldId = m_fieldComboBox->itemData(index).toInt();
      m_appData->GetSurfaceSceneView()->SetScalarFieldIndex(fieldId);
    }

    void ElVisUI::LoadPlugin()
    {

      boost::shared_ptr<QFileDialog> chooseFile(new QFileDialog(NULL, "Load plugin", "plugins"));
      chooseFile->setFileMode(QFileDialog::ExistingFile);

#if defined _MSC_VER
      chooseFile->setFilter("Plugins (*.dll)");
#elif defined __APPLE__
      chooseFile->setFilter("Plugins (*.dylib)");
#else
      chooseFile->setFilter("Plugins (*.so)");
#endif

      chooseFile->setViewMode(QFileDialog::Detail);
      chooseFile->setAcceptMode(QFileDialog::AcceptOpen);

      if (chooseFile->exec() != QDialog::Accepted)
      {
        return;
      }

      QStringList list = chooseFile->selectedFiles();
      QStringList::Iterator it = list.begin();
      QString fileName = *it;
      QByteArray bytes = fileName.toAscii();
      boost::filesystem::path path(bytes.constData());
      AddPlugin(path, true);
    }

    void ElVisUI::AddPlugin(const boost::filesystem::path& path, bool load, const std::string& pluginName)
    {
      try
      {
        std::string nameToUse = pluginName;
        if (load)
        {
          boost::shared_ptr<Plugin> plugin(new Plugin(path));
          m_plugins[plugin->GetName()] = plugin;
          nameToUse = plugin->GetName();
        }

        PluginLocationData data;
        data.Path = path;
        data.Loaded = load;
        m_pluginLocationData[nameToUse] = data;
        QAction* pluginAction = new QAction(nameToUse.c_str(), this);
        pluginAction->setCheckable(true);
        pluginAction->setChecked(load);
        m_pluginMenu->addAction(pluginAction);
        connect(pluginAction, SIGNAL(triggered()), this, SLOT(HandlePluginClick()));
        WriteSettings();
      }
      catch (UnableToLoadDynamicLibException& e)
      {
        m_errorNotification->showMessage(e.what());
      }
      catch (boost::filesystem::filesystem_error& e)
      {
        m_errorNotification->showMessage(e.what());
      }
      catch (std::runtime_error& e)
      {
        m_errorNotification->showMessage(e.what());
      }
      catch (std::exception& e)
      {
        m_errorNotification->showMessage(e.what());
      }
      catch (...)
      {
        m_errorNotification->showMessage("Caught an unknown exception trying to load plugin.");
      }
    }

    void ElVisUI::HandlePluginClick()
    {
      // How do I get teh data about which menu item was clicked?
      QAction* action = dynamic_cast<QAction*>(sender());
      if (!action) return;

      std::string name = action->text().toStdString();
      PluginLocationData& data = m_pluginLocationData[name];

      if (action->isChecked())
      {
        boost::shared_ptr<Plugin> plugin(new Plugin(data.Path));
        m_plugins[name] = plugin;
        data.Loaded = true;
      }
      else
      {
        data.Loaded = false;
        std::map<std::string, boost::shared_ptr<Plugin>>::iterator found = m_plugins.find(name);
        if (found != m_plugins.end())
        {
          m_plugins.erase(found);
        }
        std::cout << "Is not checked." << std::endl;
      }
    }

    void ElVisUI::LoadVolume()
    {
      if (m_plugins.size() == 0)
      {
        // No plugins means no models.
        return;
      }

      boost::shared_ptr<QFileDialog> chooseFile(
        new QFileDialog(NULL, "Load plugin", m_settings->value(DEFAULT_MODEL_DIR_SETTING_NAME).toString()));
      chooseFile->setFileMode(QFileDialog::ExistingFile);

      QStringList filters;
      typedef std::map<std::string, boost::shared_ptr<Plugin>>::value_type IterType;
      BOOST_FOREACH (IterType iter, m_plugins)
      {
        boost::shared_ptr<Plugin> plugin = iter.second;
        filters.push_back(plugin->GetModelFileFilter().c_str());
      }

      chooseFile->setNameFilters(filters);
      chooseFile->setViewMode(QFileDialog::Detail);
      chooseFile->setAcceptMode(QFileDialog::AcceptOpen);

      if (chooseFile->exec() != QDialog::Accepted)
      {
        return;
      }

      QStringList list = chooseFile->selectedFiles();
      QStringList::Iterator it = list.begin();
      QString fileName = *it;
      QDir CurrentDir;
      m_settings->setValue(DEFAULT_MODEL_DIR_SETTING_NAME, CurrentDir.absoluteFilePath(fileName));
      LoadFile(fileName, chooseFile->selectedNameFilter());
    }

    void ElVisUI::LoadFile(const QString& fileName, const QString& filter)
    {
      QByteArray bytes = fileName.toAscii();

      typedef std::map<std::string, boost::shared_ptr<Plugin>>::value_type IterType;
      BOOST_FOREACH (IterType iter, m_plugins)
      {
        boost::shared_ptr<Plugin> plugin = iter.second;

        if (plugin->GetModelFileFilter() == filter.toStdString())
        {
          boost::shared_ptr<ElVis::Model> model = plugin->LoadModel(bytes.constData());
          m_appData->GetScene()->SetModel(model);
          SetCurrentFile(fileName, filter);
          break;
        }
      }
    }

    void ElVisUI::LoadDefaultColorMaps()
    {
      // Get a file listing for every file in ElVis/ColorMaps.
      boost::filesystem::path colorMapPath = QApplication::applicationDirPath().toStdString() + "/../ColorMaps";

      if (!exists(colorMapPath) || !is_directory(colorMapPath))
      {
        return;
      }

      for (boost::filesystem::directory_iterator iter = boost::filesystem::directory_iterator(colorMapPath);
           iter != boost::filesystem::directory_iterator(); ++iter)
      {
        boost::filesystem::directory_entry& entry = *iter;
        m_appData->GetScene()->LoadColorMap(entry.path());
      }
    }

    void ElVisUI::SaveScreenshot()
    {
      boost::shared_ptr<QFileDialog> chooseFile(
        new QFileDialog(NULL, "Save Screenshot", m_settings->value(DEFAULT_SCREENSHOT_DIR_SETTING_NAME).toString()));

      QStringList filters;
      filters.push_back("PNG files (*.png)");
      chooseFile->setNameFilters(filters);
      chooseFile->setViewMode(QFileDialog::Detail);
      chooseFile->setAcceptMode(QFileDialog::AcceptSave);
      chooseFile->setConfirmOverwrite(true);

      if (chooseFile->exec() != QDialog::Accepted)
      {
        return;
      }

      QStringList list = chooseFile->selectedFiles();
      QStringList::Iterator it = list.begin();
      QString fileName = *it;
      QDir CurrentDir;
      m_settings->setValue(DEFAULT_SCREENSHOT_DIR_SETTING_NAME, CurrentDir.absoluteFilePath(fileName));

      QImage result = m_sceneViewWidget->grabFrameBuffer();
      result.save(fileName);
    }

    std::string ElVisUI::GetElVisStateFilter()
    {
      std::string filterValue = std::string("ElVis State files (*") + STATE_SUFFIX + std::string(")");
      return filterValue;
    }

    template <typename T>
    void addElement(const std::string& elementName, tinyxml::TiXmlNode* parentNode, const T& value)
    {
      auto childElement = new tinyxml::TiXmlElement(elementName.c_str());
      parentNode->LinkEndChild(childElement);

      std::string asStr = boost::lexical_cast<std::string>(value);
      auto text = new tinyxml::TiXmlText(asStr.c_str());
      childElement->LinkEndChild(text);
    }

    template <typename T>
    T getElement(const std::string& elementName, tinyxml::TiXmlNode* parentNode)
    {
      auto childElement = parentNode->FirstChildElement(elementName.c_str());
      std::string text = childElement->GetText();
      return boost::lexical_cast<T>(text);
    }

    void ElVisUI::SaveState()
    {
      boost::shared_ptr<QFileDialog> chooseFile(
        new QFileDialog(NULL, "Save State", m_settings->value(DEFAULT_STATE_DIR_SETTING_NAME).toString()));

      QStringList filters;
      filters.push_back(GetElVisStateFilter().c_str());
      chooseFile->setNameFilters(filters);
      chooseFile->setViewMode(QFileDialog::Detail);
      chooseFile->setAcceptMode(QFileDialog::AcceptSave);
      chooseFile->setConfirmOverwrite(true);

      if (chooseFile->exec() != QDialog::Accepted)
      {
        return;
      }

      QStringList list = chooseFile->selectedFiles();
      QStringList::Iterator it = list.begin();
      QString fileName = *it;

      auto pScene = m_appData->GetSurfaceSceneView()->GetScene();
      std::ofstream outFile(fileName.toStdString().c_str());
      boost::archive::xml_oarchive oa(outFile);
      // ElVis::Scene& scene = *pScene;
      auto pSceneView = m_appData->GetSurfaceSceneView();
      oa << boost::serialization::make_nvp(SCENE_VIEW_ELEMENT_NAME.c_str(), *pSceneView);
      outFile.close();

      // tinyxml::TiXmlDocument doc;
      // BOOST_AUTO(decl, new tinyxml::TiXmlDeclaration("1.0", "", ""));
      // doc.LinkEndChild(decl);

      // BOOST_AUTO(settings, new tinyxml::TiXmlElement("ElVisSettings"));
      // doc.LinkEndChild(settings);

      // BOOST_AUTO(scene, m_appData->GetSurfaceSceneView()->GetScene());

      //// Camera
      // boost::shared_ptr<Camera> camera = m_appData->GetSurfaceSceneView()->GetViewSettings();
      // BOOST_AUTO(cameraElement, new tinyxml::TiXmlElement("Camera"));
      // settings->LinkEndChild(cameraElement);
      // addElement("EyeX", cameraElement, camera->GetEye().x());
      // addElement("EyeY", cameraElement, camera->GetEye().y());
      // addElement("EyeZ", cameraElement, camera->GetEye().z());
      // addElement("LookAtX", cameraElement, camera->GetLookAt().x());
      // addElement("LookAtY", cameraElement, camera->GetLookAt().y());
      // addElement("LookAtZ", cameraElement, camera->GetLookAt().z());
      // addElement("UpX", cameraElement, camera->GetUp().x());
      // addElement("UpY", cameraElement, camera->GetUp().y());
      // addElement("UpZ", cameraElement, camera->GetUp().z());
      // addElement("FOV", cameraElement, camera->GetFieldOfView());
      // addElement("Near", cameraElement, camera->GetNear());
      // addElement("Far", cameraElement, camera->GetFar());

      // doc.SaveFile(fileName.toStdString().c_str());

      QDir CurrentDir;
      m_settings->setValue(DEFAULT_STATE_DIR_SETTING_NAME, CurrentDir.absoluteFilePath(fileName));
    }

    void ElVisUI::LoadState()
    {
      boost::shared_ptr<QFileDialog> chooseFile(
        new QFileDialog(NULL, "Load State", m_settings->value(DEFAULT_STATE_DIR_SETTING_NAME).toString()));
      chooseFile->setFileMode(QFileDialog::ExistingFile);

      chooseFile->setFilter(GetElVisStateFilter().c_str());

      chooseFile->setViewMode(QFileDialog::Detail);
      chooseFile->setAcceptMode(QFileDialog::AcceptOpen);

      if (chooseFile->exec() != QDialog::Accepted)
      {
        return;
      }

      QStringList list = chooseFile->selectedFiles();
      QStringList::Iterator it = list.begin();
      QString fileName = *it;

      std::ifstream inFile(fileName.toStdString());
      boost::archive::xml_iarchive ia(inFile);
      auto pSceneView = m_appData->GetSurfaceSceneView();
      ia >> boost::serialization::make_nvp(SCENE_VIEW_ELEMENT_NAME.c_str(), *pSceneView);
      inFile.close();

//      tinyxml::TiXmlDocument doc(fileName.toStdString().c_str());
//      bool loadOkay = doc.LoadFile();

//      if (!loadOkay)
//      {
//        throw std::runtime_error("Unable to load file " + fileName.toStdString());
//      }

//      tinyxml::TiXmlHandle docHandle(&doc);
//      // tinyxml::TiXmlNode* node = 0;
//      tinyxml::TiXmlElement* rootElement = doc.FirstChildElement("ElVisSettings");

//      // Camera
//      auto cameraElement = rootElement->FirstChildElement("Camera");
//      boost::shared_ptr<Camera> camera = m_appData->GetSurfaceSceneView()->GetViewSettings();
//      ElVis::WorldPoint eye;
//      ElVis::WorldPoint lookAt;
//      ElVis::WorldVector up;

//      eye.SetX(getElement<double>("EyeX", cameraElement));
//      eye.SetY(getElement<double>("EyeY", cameraElement));
//      eye.SetZ(getElement<double>("EyeZ", cameraElement));
//      lookAt.SetX(getElement<double>("LookAtX", cameraElement));
//      lookAt.SetY(getElement<double>("LookAtY", cameraElement));
//      lookAt.SetZ(getElement<double>("LookAtZ", cameraElement));
//      up.SetX(getElement<double>("UpX", cameraElement));
//      up.SetY(getElement<double>("UpY", cameraElement));
//      up.SetZ(getElement<double>("UpZ", cameraElement));
//      double fov = getElement<double>("FOV", cameraElement);
//      double nearVal = getElement<double>("Near", cameraElement);
//      double farVal = getElement<double>("Far", cameraElement);

//      camera->SetParameters(eye, lookAt, up, fov, nearVal, farVal);

      QDir CurrentDir;
      m_settings->setValue(DEFAULT_STATE_DIR_SETTING_NAME, CurrentDir.absoluteFilePath(fileName));
    }

    void ElVisUI::ExportViewSettings()
    {
      std::cout << "Visual 3 Settings: " << std::endl;
      double modelviewMatrix[16];
      m_sceneViewWidget->GetModelViewMatrixForVisual3(modelviewMatrix);

      std::cout << modelviewMatrix[0] << " " << modelviewMatrix[4] << " " << modelviewMatrix[8] << " "
                << modelviewMatrix[12] << std::endl;
      std::cout << modelviewMatrix[1] << " " << modelviewMatrix[5] << " " << modelviewMatrix[9] << " "
                << modelviewMatrix[13] << std::endl;
      std::cout << modelviewMatrix[2] << " " << modelviewMatrix[6] << " " << modelviewMatrix[10] << " "
                << modelviewMatrix[14] << std::endl;
    }
  }
}
