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

#ifndef ELVIS_GUI_ELVISUI_H
#define ELVIS_GUI_ELVISUI_H

#include <ElVis/Core/Scene.h>
#include <ElVis/Core/TransferFunction.h>
#include <ElVis/Core/Plugin.h>

#include <ElVis/Gui/ViewSettingsUI.h>
#include <ElVis/Gui/ObjectInspectorUI.h>
#include <ElVis/Gui/SceneViewWidget.h>
#include <ElVis/Gui/SceneItemsDockWidget.h>
#include <ElVis/Gui/ModelInspectorWidget.h>
#include <ElVis/Gui/RenderSettingsDockWidget.h>
#include <ElVis/Gui/ApplicationState.h>
#include <ElVis/Gui/ColorMapWidget.h>
#include <ElVis/Gui/DebugSettingsDockWidget.h>
#include <ElVis/Gui/VolumeRenderingSettingsWidget.h>
#include <ElVis/Gui/ElementFaceRenderingDockWidget.h>
#include <ElVis/Gui/ContourDockWidget.h>
#include <ElVis/Gui/IsosurfaceDockWidget.h>
#include <ElVis/Gui/LightingDockWidget.h>
#include <ElVis/Gui/SampleOntoNrrdDockWidget.h>

#include <QErrorMessage>
#include <QMdiArea>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QToolBar>
#include <QSettings>
#include <QComboBox>
#include <QMainWindow>

#include <boost/shared_ptr.hpp>

namespace ElVis
{
    namespace Gui
    {
        class ElVisUI : public QMainWindow
        {
            Q_OBJECT;

            public:
                ElVisUI();
                ~ElVisUI();

            public Q_SLOTS:
                void LoadPlugin();
                void LoadVolume();
                void HandleModelChanged(boost::shared_ptr<Model> model);
                void HandlePrimaryRayObjectAdded(boost::shared_ptr<PrimaryRayObject> obj);
                void Exit();
                void CreateCutPlane();
                void HandlePluginClick();
                void HandleOpenRecentFile();
                void HandleFieldSelectionChanged(int index);
                void SaveScreenshot();
                void SaveState();
                void LoadState();
                void ExportViewSettings();
                void CalculateScalarRange();
            Q_SIGNALS:

            protected:
                void closeEvent(QCloseEvent *event);

            private:
                static const QString WindowsSettings;
                static const QString OptixStackSize;

                struct PluginLocationData
                {
                    boost::filesystem::path Path;
                    bool Loaded;
                };

                static const char* DEFAULT_MODEL_DIR_SETTING_NAME;
                static const char* DEFAULT_SCREENSHOT_DIR_SETTING_NAME;
                static const char* DEFAULT_STATE_DIR_SETTING_NAME;
                static const char* STATE_SUFFIX;
                static const char* RECENT_FILE_LIST;
                static const char* RECENT_FILE_FILTER;

                ElVisUI(const ElVisUI& rhs);
                ElVisUI& operator=(const ElVisUI& rhs);

                static std::string GetElVisStateFilter();

                void UpdateRecentFileActions();
                QString StrippedName(const QString& fullFileName);
                void SetCurrentFile(const QString &fileName, const QString& filterName);

                /// \brief Loads a volume from the given location.
                /// \param fileName The path to the volume.
                /// \param filter The filter used to select the file.  This is used to indicate which plugin should be used to load the volume.
                void LoadFile(const QString &fileName, const QString& filter);
                void SetupDocking();
                void ReadSettings();
                void WriteSettings();
                void CreateActions();
                void CreateMenus();
                void CreateToolBars();
                void AddPlugin(const boost::filesystem::path& p, bool load, const std::string& pluginName = std::string());
                void SetupSceneViews();
                void LoadDefaultColorMaps();


                QMdiArea* m_mdiArea;

                boost::shared_ptr<ApplicationState> m_appData;

                boost::shared_ptr<SceneViewWidget> m_sceneViewWidget;

                ViewSettingsUI* m_viewSettings;
                ObjectInspectorUI* m_objectInspector;
                ModelInspectorWidget* m_modelInspectorWidget;
                ColorMapWidget* m_colorMapWidget;
                RenderSettingsDockWidget* m_renderSettingsWidget;
                DebugSettingDockWidget* m_debugDockWidget;
                VolumeRenderingSettingsWidget* m_volumeRenderingSettings;
                ElementFaceRenderingDockWidget* m_elementRenderingDockWidget;
                ContourDockWidget* m_contourDockWidget;
                IsosurfaceDockWidget* m_isosurfaceDockWidget;
                LightingDockWidget* m_lightingDockWidget;
                SampleOntoNrrdDockWidget* m_sampleDockWidget;

                std::map<std::string, boost::shared_ptr<Plugin> > m_plugins;
                std::map<std::string, PluginLocationData> m_pluginLocationData;

                QErrorMessage* m_errorNotification;

                QToolBar* m_mainToolbar;
                SceneItemsDockWidget* m_sceneItems;

                QSettings* m_settings;

                QAction* m_actionExit;
                QAction* m_actionLoadPlugin;
                QAction* m_actionLoadModel;
                QAction* m_actionCreateCutPlane;
                QAction* m_actionSaveScreenshot;
                QAction* m_actionExportVisual3View;
                QAction* m_loadViewAction;
                QAction* m_saveViewAction;
                QAction* m_calculateScalarRangeAction;

                QMenuBar* m_menuBar;
                QMenu* m_fileMenu;
                QMenu* m_pluginMenu;
                QMenu* m_sourcesMenu;
                QMenu* m_windowMenu;
                QMenu* m_utilsMenu;

                QAction* m_separatorAct;

                enum { MaxRecentFiles = 10 };
                QAction* m_recentFileActs[MaxRecentFiles];
                QLabel* m_fieldLabel;
                QComboBox* m_fieldComboBox;
                bool m_initializing;
        };
    }
}

#endif //ELVIS_GUI_ELVISUI_H
