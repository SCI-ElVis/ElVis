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


#ifndef ELVIS_GUI_APPLICATION_STATE_H
#define ELVIS_GUI_APPLICATION_STATE_H

#include <ElVis/Core/Scene.h>
#include <ElVis/Core/SceneView.h>
#include <ElVis/Core/TransferFunction.h>
#include <ElVis/Core/HostTransferFunction.h>
#include <ElVis/Core/ColorMapperModule.h>
#include <ElVis/Core/CutSurfaceContourModule.h>
#include <ElVis/Core/IsosurfaceModule.h>
#include <ElVis/Core/CutSurfaceMeshModule.h>
#include <boost/signal.hpp>

namespace ElVis
{
    class VolumeRenderingModule;
    namespace Gui
    {
        struct PointInfo
        {
            PointInfo() :
                IntersectionPoint(),
                Normal(),
                Valid(false),
                Pixel(),
                ElementId(-1),
                ElementTypeId(-1),
                Scalar(0.0)
            {
            }

            PointInfo(const PointInfo& rhs) :
                IntersectionPoint(rhs.IntersectionPoint),
                Normal(rhs.Normal),
                Valid(rhs.Valid),
                Pixel(rhs.Pixel),
                ElementId(rhs.ElementId),
                ElementTypeId(rhs.ElementTypeId),
                Scalar(rhs.Scalar)
            {
            }

            PointInfo& operator=(const PointInfo& rhs)
            {
                IntersectionPoint = rhs.IntersectionPoint;
                Normal = rhs.Normal;
                Valid = rhs.Valid;
                Pixel = rhs.Pixel;
                ElementId = rhs.ElementId;
                ElementTypeId = rhs.ElementTypeId;
                Scalar = rhs.Scalar;
                return *this;
            }

            WorldPoint IntersectionPoint;
            WorldPoint Normal;
            bool Valid;
            Point<int, TwoD> Pixel;
            int ElementId;
            int ElementTypeId;
            ElVisFloat Scalar;
        };

        class ApplicationState
        {
            public:
                ApplicationState();
                ~ApplicationState();

                boost::shared_ptr<Scene> GetScene() const { return m_scene; }
                boost::shared_ptr<SceneView> GetSurfaceSceneView() const { return m_surfaceSceneView; }

                boost::shared_ptr<PrimaryRayObject> GetSelectedObject() const { return m_selectedObject; }
                void SetSelectedObject(boost::shared_ptr<PrimaryRayObject> obj)
                {
                    m_selectedObject = obj;
                    OnSelectedObjectChanged(obj);
                    OnApplicationStateChanged();
                }

                void SetSelectedColorMap(boost::shared_ptr<PiecewiseLinearColorMap> f);

                boost::shared_ptr<PiecewiseLinearColorMap> GetSelectedColorMap() { return m_selectedTransferFunction; }


                void ClearSelectedTransferFunction()
                {
                    m_selectedTransferFunction.reset();
                    OnSelectedTransferFunctionChanged(m_selectedTransferFunction);
                    OnApplicationStateChanged();
                }

                void SetLookAtPointToIntersectionPoint(unsigned int pixel_x, unsigned int pixel_y);
                void GeneratePointInformation(unsigned int pixel_x, unsigned int pixel_y);

                boost::signal<void (boost::shared_ptr<PrimaryRayObject>)> OnSelectedObjectChanged;
                boost::signal<void (boost::shared_ptr<PiecewiseLinearColorMap>)> OnSelectedTransferFunctionChanged;
                boost::signal<void ()> OnApplicationStateChanged;
                boost::signal<void (const PointInfo&)> OnSelectedPointChanged;

                bool GetShowLookAtPoint() const { return m_showLookAtPoint; }
                void SetShowLookAtPoint(bool newValue);

                bool GetShowBoundingBox() const { return m_showBoundingBox; }
                void SetShowBoundingBox(bool newValue);

                boost::shared_ptr<VolumeRenderingModule> GetVolumeRenderingModule() { return m_volumeRenderingModule; }
                boost::shared_ptr<CutSurfaceContourModule> GetContourModule() { return m_cutSurfaceContourModule; }

                boost::shared_ptr<SampleFaceObject> GetFaceSampler() { return m_faceSampler; }


                boost::shared_ptr<IsosurfaceModule> GetIsosurfaceModule() { return m_isosurfaceModule; }


                boost::shared_ptr<HostTransferFunction> GetHostTransferFunction();

            private:
                ApplicationState(const ApplicationState&);
                ApplicationState& operator=(const ApplicationState& rhs);

                void HandleSelectedColorMapMinOrMaxChanged(float value);

                boost::shared_ptr<Scene> m_scene;

                // A view of the scene defined by surfaces (cut surfaces/isosurfaces/geometry)
                boost::shared_ptr<SceneView> m_surfaceSceneView;

                boost::shared_ptr<PrimaryRayObject> m_selectedObject;

                boost::shared_ptr<PiecewiseLinearColorMap> m_selectedTransferFunction;

                boost::shared_ptr<ElVis::ColorMapperModule> m_colorMapperModule;

                boost::shared_ptr<ElVis::CutSurfaceContourModule> m_cutSurfaceContourModule;
                boost::shared_ptr<ElVis::CutSurfaceMeshModule> m_cutSurfaceMeshModule;

                PointInfo m_selectedPointInfo;

                // Data for debuggins a single point.
                boost::signals::connection m_minConnection;
                boost::signals::connection m_maxConnection;

                bool m_showLookAtPoint;
                bool m_showBoundingBox;

                boost::shared_ptr<VolumeRenderingModule> m_volumeRenderingModule;
                boost::shared_ptr<PrimaryRayModule> m_primaryRayModule;

                ///////////////////
                // Face Rendering
                ///////////////////
                boost::shared_ptr<SampleFaceObject> m_faceSampler;
                boost::shared_ptr<IsosurfaceModule> m_isosurfaceModule;

        };
    }
}

#endif
