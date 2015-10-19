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

#include <ElVis/Core/ColorMapperModule.h>
#include <ElVis/Core/Float.h>
#include <ElVis/Core/SceneView.h>

#include <iostream>
#include <stdio.h>

#include <boost/timer.hpp>
#include <boost/bind.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/make_shared.hpp>

// Serialization keys
namespace
{
  const std::string MIN_KEY_NAME("Min");
  const std::string MAX_KEY_NAME("Max");
  const std::string SIZE_KEY_NAME("Size");
  const std::string COLOR_MAP_KEY_NAME("ColorMap");
}
namespace ElVis
{
  ColorMapperModule::ColorMapperModule()
    : m_colorMap(),
      m_program(),
      m_data(),
      m_textureSampler(),
      m_size(1024),
      m_dirty(true),
      m_minConnection(),
      m_maxConnection(),
      m_changedConnection()
  {
  }

  void ColorMapperModule::DoSetup(SceneView* view)
  {
    optixu::Context context = view->GetContext();
    assert(context);

    std::string programName("TextureColorMap");
    m_program = view->AddRayGenerationProgram(programName);

    m_data = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, m_size);

    context["ColorMapTexture"]->set(m_data);

    if( m_colorMap )
    {
      context["TextureMaxScalar"]->setFloat(m_colorMap->GetMax());
      context["TextureMinScalar"]->setFloat(m_colorMap->GetMin());
    }
  }

  void ColorMapperModule::DoSynchronize(SceneView* view)
  {
    if (!m_dirty) return;

    optixu::Context context = view->GetContext();

    if (m_colorMap)
    {
      m_data->setSize(m_size);
      m_colorMap->PopulateTexture(m_data);
      context["TextureMaxScalar"]->setFloat(m_colorMap->GetMax());
      context["TextureMinScalar"]->setFloat(m_colorMap->GetMin());
    }

    m_dirty = false;
  }

  void ColorMapperModule::DoRender(SceneView* view)
  {
    optixu::Context context = view->GetContext();
    context->launch(m_program.Index, view->GetWidth(), view->GetHeight());
  }

  void ColorMapperModule::HandleMinOrMaxChanged(float value)
  {
    m_dirty = true;
    SetSyncAndRenderRequired();
  }

  void ColorMapperModule::HandleColorMapChanged(const ColorMap& rhs)
  {
    m_dirty = true;
    SetSyncAndRenderRequired();
  }

  void ColorMapperModule::SetColorMap(boost::shared_ptr<ColorMap> map)
  {
    if (m_colorMap != map)
    {
      m_minConnection.disconnect();
      m_maxConnection.disconnect();
      m_changedConnection.disconnect();
      m_colorMap = map;
      m_dirty = true;

      if (m_colorMap)
      {
        m_minConnection = m_colorMap->OnMinChanged.connect(
          boost::bind(&ColorMapperModule::HandleMinOrMaxChanged, this, _1));
        m_maxConnection = m_colorMap->OnMaxChanged.connect(
          boost::bind(&ColorMapperModule::HandleMinOrMaxChanged, this, _1));
        m_changedConnection = m_colorMap->OnColorMapChanged.connect(
          boost::bind(&ColorMapperModule::HandleColorMapChanged, this, _1));
      }

      SetSyncAndRenderRequired();
    }
  }

  void ColorMapperModule::DoSerialize(std::unique_ptr<ElVis::Serialization::RenderModule>& pResult) const
  {
    auto pSerializedModule = Serialize();
    pResult->mutable_concrete_module()->PackFrom(*pSerializedModule);
  }

  std::unique_ptr<ElVis::Serialization::ColorMapperModule> ColorMapperModule::Serialize() const
  {
    auto pResult = std::unique_ptr<ElVis::Serialization::ColorMapperModule>(new ElVis::Serialization::ColorMapperModule());
    pResult->set_allocated_color_map(m_colorMap->Serialize().release());
    pResult->set_size(m_size);
    return pResult;
  }

  void ColorMapperModule::Deserialize(const ElVis::Serialization::ColorMapperModule& input)
  {
    m_size = input.size();

    // TODO - Why don't we just create a default color map up front?
    if( m_colorMap )
    {
      m_colorMap->Deserialize(input.color_map());
    }
    else
    {
      auto pMap = boost::make_shared<ColorMap>();
      pMap->Deserialize(input.color_map());
      SetColorMap(pMap);
    }
    m_dirty = true;
    SetSyncAndRenderRequired();
  }

}
