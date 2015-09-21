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

#include <ElVis/Core/RenderModuleFactory.h>
#include <ElVis/Core/RenderModule.h>
#include <ElVis/Core/IsosurfaceModule.h>
#include <ElVis/Core/IsosurfaceModule.pb.h>
#include <ElVis/Core/PrimaryRayModule.h>
#include <ElVis/Core/PrimaryRayModule.pb.h>
#include <ElVis/Core/LightingModule.h>
#include <ElVis/Core/LightingModule.pb.h>
#include <ElVis/Core/ColorMapperModule.h>
#include <ElVis/Core/ColorMapperModule.pb.h>
#include <ElVis/Core/VolumeRenderingModule.h>
#include <ElVis/Core/VolumeRenderingModule.pb.h>
#include <boost/make_shared.hpp>

namespace ElVis
{
  namespace
  {
    template<typename T, typename SerializationType>
    boost::shared_ptr<RenderModule> createModule(const google::protobuf::Any& data)
    {
      SerializationType moduleData;
      data.UnpackTo(&moduleData);
      auto pModule = boost::make_shared<T>();
      pModule->Deserialize(moduleData);
      return pModule;
    }
  }

  boost::shared_ptr<RenderModule> createRenderModule(const ElVis::Serialization::RenderModule& data)
  {
    boost::shared_ptr<RenderModule> pResult;
    const auto& concreteModule = data.concrete_module();

    if( concreteModule.Is<ElVis::Serialization::IsosurfaceModule>() )
    {
      pResult = createModule<IsosurfaceModule, ElVis::Serialization::IsosurfaceModule>(concreteModule);
    }

    if( concreteModule.Is<ElVis::Serialization::PrimaryRayModule>() )
    {
      pResult = createModule<PrimaryRayModule, ElVis::Serialization::PrimaryRayModule>(concreteModule);
    }

    if( concreteModule.Is<ElVis::Serialization::ColorMapperModule>() )
    {
      pResult = createModule<ColorMapperModule, ElVis::Serialization::ColorMapperModule>(concreteModule);
    }

    if( concreteModule.Is<ElVis::Serialization::LightingModule>() )
    {
      pResult = createModule<LightingModule, ElVis::Serialization::LightingModule>(concreteModule);
    }

    if( concreteModule.Is<ElVis::Serialization::VolumeRenderingModule>() )
    {
      pResult = createModule<VolumeRenderingModule, ElVis::Serialization::VolumeRenderingModule>(concreteModule);
    }

    if( pResult )
    {
      pResult->Deserialize(data);
      return pResult;
    }
    else
    {
      throw std::runtime_error("Invalid serialized module.");
    }
  }
}

