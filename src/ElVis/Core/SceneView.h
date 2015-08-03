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

#ifndef ELVIS_SCENE_VIEW_H
#define ELVIS_SCENE_VIEW_H

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/string.hpp>

#include <ElVis/Core/Model.h>
#include <ElVis/Core/Point.hpp>
#include <ElVis/Core/Camera.h>
#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/Float.h>
#include <ElVis/Core/SampleFaceObject.h>
#include <ElVis/Core/Stat.h>
#include <ElVis/Core/SceneViewProjection.h>
#include <ElVis/Core/SynchedObject.hpp>
#include <ElVis/Core/Timer.h>

#include <optixu/optixpp.h>
#include <ElVis/Core/Scene.h>
#include <map>

#include <boost/signals2.hpp>
#include <boost/foreach.hpp>



namespace ElVis
{
  class RenderModule;
  class PrimaryRayModule;


  class SceneView
  {
  public:
    friend class boost::serialization::access;
    ELVIS_EXPORT SceneView();
    ELVIS_EXPORT virtual ~SceneView();

    ELVIS_EXPORT Timer Draw();
    ELVIS_EXPORT void Resize(unsigned int width, unsigned int height);

    ELVIS_EXPORT unsigned int GetWidth() const { return m_width; }
    ELVIS_EXPORT unsigned int GetHeight() const { return m_height; }

    ELVIS_EXPORT void SetScene(boost::shared_ptr<Scene> value)
    {
      m_scene = value;
    }
    ELVIS_EXPORT boost::shared_ptr<Scene> GetScene() const { return m_scene; }

    ELVIS_EXPORT const WorldPoint& GetEye() const
    {
      return m_viewSettings->GetEye();
    }
    ELVIS_EXPORT const WorldPoint& GetLookAt() const
    {
      return m_viewSettings->GetLookAt();
    }
    ELVIS_EXPORT const WorldVector& GetUp() const
    {
      return m_viewSettings->GetUp();
    }
    ELVIS_EXPORT float GetFieldOfView() const
    {
      return static_cast<float>(m_viewSettings->GetFieldOfView());
    }
    ELVIS_EXPORT boost::shared_ptr<const Camera> GetViewSettings() const
    {
      return m_viewSettings;
    }
    ELVIS_EXPORT boost::shared_ptr<Camera> GetViewSettings()
    {
      return m_viewSettings;
    }

    ELVIS_EXPORT void SetCamera(const Camera& view);

    ELVIS_EXPORT void WriteColorBufferToFile(const std::string& fileName);

    ELVIS_EXPORT RayGeneratorProgram
    AddRayGenerationProgram(const std::string& programName);

    ELVIS_EXPORT optixu::Context GetContext()
    {
      return GetScene()->GetContext();
    }

    ELVIS_EXPORT void AddRenderModule(boost::shared_ptr<RenderModule> module);

    ELVIS_EXPORT OptiXBuffer<uchar4>& GetColorBuffer() { return m_colorBuffer; }
    ELVIS_EXPORT OptiXBuffer<ElVisFloat3>& GetRawColorBuffer()
    {
      return m_rawColorBuffer;
    }

    ELVIS_EXPORT OptiXBuffer<ElVisFloat3>& GetIntersectionPointBuffer()
    {
      return m_intersectionBuffer;
    }
    ELVIS_EXPORT OptiXBuffer<ElVisFloat3>& GetNormalBuffer()
    {
      return m_normalBuffer;
    }
    ELVIS_EXPORT OptiXBuffer<ElVisFloat>& GetSampleBuffer()
    {
      return m_sampleBuffer;
    }
    ELVIS_EXPORT double GetTimings(boost::shared_ptr<RenderModule> m) const
    {
      std::map<boost::shared_ptr<RenderModule>, double>::const_iterator found =
        m_timings.find(m);
      if (found != m_timings.end())
      {
        return (*found).second;
      }
      else
      {
        return -1;
      }
    }

    ELVIS_EXPORT const std::map<boost::shared_ptr<RenderModule>, double>&
    GetTimings() const
    {
      return m_timings;
    }

    std::string GetPTXPrefix()
    {
      if (m_scene && m_scene->GetModel())
      {
        return m_scene->GetModel()->GetPTXPrefix();
      }
      return "";
    }

    ELVIS_EXPORT boost::shared_ptr<PrimaryRayModule> GetPrimaryRayModule()
      const;

    ELVIS_EXPORT void DisplayBuffersToScreen();

    ELVIS_EXPORT void SetDepthBufferBits(int newValue);

    ELVIS_EXPORT void WriteDepthBuffer(const std::string& filePrefix);

    ELVIS_EXPORT std::list<boost::shared_ptr<RenderModule>> GetRenderModules()
    {
      return m_allRenderModules;
    }

    ELVIS_EXPORT WorldPoint
    GetIntersectionPoint(unsigned int pixel_x, unsigned int pixel_y);
    ELVIS_EXPORT WorldPoint
    GetNormal(unsigned int pixel_x, unsigned int pixel_y);
    ELVIS_EXPORT int GetElementId(unsigned int pixel_x, unsigned int pixel_y);
    ELVIS_EXPORT int GetElementTypeId(unsigned int pixel_x,
                                      unsigned int pixel_y);
    ELVIS_EXPORT ElVisFloat
    GetScalarSample(unsigned int pixel_x, unsigned int pixel_y);

    ELVIS_EXPORT void SetScalarFieldIndex(int index);

    ELVIS_EXPORT int GetScalarFieldIndex() const { return m_scalarFieldIndex; }

    ELVIS_EXPORT ElVisFloat GetFaceIntersectionTolerance() const;
    ELVIS_EXPORT void SetFaceIntersectionTolerance(ElVisFloat value);

    ELVIS_EXPORT const Color& GetHeadlightColor() const;
    ELVIS_EXPORT void SetHeadlightColor(const Color& newValue);

    ELVIS_EXPORT const Color& GetBackgroundColor() const
    {
      return m_backgroundColor;
    }
    ELVIS_EXPORT void SetBackgroundColor(const Color& newValue);

    ELVIS_EXPORT Stat CalculateScalarSampleStats();

    ELVIS_EXPORT void SetProjectionType(SceneViewProjection type);
    ELVIS_EXPORT SceneViewProjection GetProjectionType() const;

    boost::signals2::signal<void(const SceneView&)> OnSceneViewChanged;
    boost::signals2::signal<void(int w, int h)> OnWindowSizeChanged;
    boost::signals2::signal<void(const SceneView&)> OnNeedsRedraw;

    ELVIS_EXPORT void SetOptixStackSize(int size);
    ELVIS_EXPORT int GetOptixStackSize() const { return m_optixStackSize; }

  protected:
    virtual void DoWindowSizeHasChanged() {}
    virtual void DoPrepareForDisplay() {}

  private:
    SceneView(const SceneView&);
    SceneView& operator=(const SceneView& rhs);
    void SynchronizeWithGPUIfNeeded(optixu::Context context);

    template <typename T>
    void HandleSynchedObjectChanged(const SynchedObject<T>& obj)
    {
      OnSceneViewChanged(*this);
    }

    static const std::string ColorBufferName;
    static const std::string DepthBufferName;
    static const std::string StencilBufferName;
    static const std::string NormalBufferName;
    static const std::string RawColorBufferName;
    static const std::string IntersectionBufferName;

    void PrepareForDisplay();
    void WindowSizeHasChanged();
    void ViewingParametersHaveChanged();
    void SetupCamera();
    void ClearDepthBuffer();
    void ClearColorBuffer();
    void HandleRenderModuleChanged(const RenderModule&);

    /// \brief Serializes this to an archive.
    /// \param ar The serialization destination.
    template <typename Archive>
    void save(Archive& ar, const unsigned int version) const;

    /// \brief Deserializes this from an archive.
    /// \param ar The serialization source.
    template <typename Archive>
    void load(Archive& ar, const unsigned int version);

    /// This macro is required to support the save/load interface above.
    /// Without it, serialization and deserialization are performed by
    /// the same function.
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    boost::shared_ptr<Scene> m_scene;
    unsigned int m_width;
    unsigned int m_height;
    boost::shared_ptr<Camera> m_viewSettings;

    OptiXBuffer<uchar4> m_colorBuffer;
    OptiXBuffer<ElVisFloat3> m_rawColorBuffer;

    // Must be float for OpenGL.
    OptiXBuffer<float> m_depthBuffer;
    OptiXBuffer<ElVisFloat3> m_normalBuffer;
    OptiXBuffer<ElVisFloat3> m_intersectionBuffer;
    OptiXBuffer<ElVisFloat> m_sampleBuffer;
    OptiXBuffer<int> m_elementIdBuffer;
    OptiXBuffer<int> m_elementTypeBuffer;
    int m_depthBits;

    std::map<std::string, RayGeneratorProgram> m_rayGenerationPrograms;
    std::list<boost::shared_ptr<RenderModule>> m_allRenderModules;
    std::map<boost::shared_ptr<RenderModule>, double> m_timings;
    int m_scalarFieldIndex;
    bool m_passedInitialOptixSetup;

    bool m_faceIntersectionToleranceDirty;
    ElVisFloat m_faceIntersectionTolerance;
    Color m_headlightColor;
    bool m_headlightColorIsDirty;
    bool m_enableOptiXExceptions;
    optixu::Program m_exceptionProgram;
    Color m_backgroundColor;
    bool m_backgroundColorIsDirty;
    SynchedObject<SceneViewProjection> m_projectionType;
    int m_optixStackSize;
  };
}

#endif // ELVIS_SCENE_VIEW_H
