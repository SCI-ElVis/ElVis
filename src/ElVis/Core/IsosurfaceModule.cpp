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

#include <ElVis/Core/IsosurfaceModule.h>
#include <ElVis/Core/SceneView.h>
#include <ElVis/Core/PtxManager.h>
#include <ElVis/Core/Float.h>
#include <ElVis/Core/Util.hpp>

#include <fstream>

#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/set.hpp>

namespace ElVis
{
  // Serialization Keys
  namespace
  {
    const std::string ISOVALUE_KEY_NAME("Isovalues");
  }

  RayGeneratorProgram IsosurfaceModule::m_FindIsosurface;

  IsosurfaceModule::IsosurfaceModule()
    : RenderModule(),
      OnIsovaluesChanged(),
      m_isovalues(),
      m_dirty(true),
      m_isovalueBuffer("SurfaceIsovalues"),
      m_gaussLegendreNodesBuffer("Nodes"),
      m_gaussLegendreWeightsBuffer("Weights"),
      m_requiredOrderBuffer("RequiredOrder"),
      m_epsilonBuffer("Epsilon"),
      m_monomialConversionTableBuffer("MonomialConversionTable")
  {
  }

  void IsosurfaceModule::AddIsovalue(const ElVisFloat& value)
  {
    if (m_isovalues.find(value) == m_isovalues.end())
    {
      m_isovalues.insert(value);
      m_dirty = true;
      SetSyncAndRenderRequired();
      OnIsovaluesChanged();
      OnModuleChanged(*this);
    }
  }

  void IsosurfaceModule::RemoveIsovalue(const ElVisFloat& value)
  {
    std::set<ElVisFloat>::iterator found = m_isovalues.find(value);
    if (found != m_isovalues.end())
    {
      m_isovalues.erase(found);
      m_dirty = true;
      SetSyncAndRenderRequired();
      OnIsovaluesChanged();
      OnModuleChanged(*this);
    }
  }

  void IsosurfaceModule::SetRequiredOrder(int newValue)
  {
	  bool change = false;

	  if( m_requiredOrder.size() != 0 )
	  {
		  if( newValue != m_requiredOrder[0] )
		      change = true;
	  }
	  else
	  {
		  change = true;
	  }

	  if( change )
	  {
		  m_requiredOrder.clear();
          m_requiredOrder.push_back(newValue);
		  SetSyncAndRenderRequired();
          OnModuleChanged(*this);
	  }
  }

  void IsosurfaceModule::SetEpsilon(int newValue)
  {
	  bool change = false;
	  double newValueD = pow(10, newValue);

	  if( m_epsilon.size() != 0 )
	  {
		  if( newValueD != m_epsilon[0] )
		      change = true;
	  }
	  else
	  {
		  change = true;
	  }

	  if( change )
	  {
		  m_epsilon.clear();
		  m_epsilon.push_back(newValueD);
		  SetSyncAndRenderRequired();
          OnModuleChanged(*this);
	  }
  }

  void IsosurfaceModule::DoRender(SceneView* view)
  {
    if (m_isovalues.empty()) return;

    try
    {
      optixu::Context context = view->GetContext();

      context->launch(
        m_FindIsosurface.Index, view->GetWidth(), view->GetHeight());
    }
    catch (optixu::Exception& e)
    {
      std::cout << "Exception encountered rendering isosurface." << std::endl;
      std::cerr << e.getErrorString() << std::endl;
      std::cout << e.getErrorString().c_str() << std::endl;
    }
    catch (std::exception& e)
    {
      std::cout << "Exception encountered rendering isosurface." << std::endl;
      std::cout << e.what() << std::endl;
    }
    catch (...)
    {
      std::cout << "Exception encountered rendering isosurface." << std::endl;
    }
  }

  void IsosurfaceModule::DoSetup(SceneView* view)
  {
    try
    {
      optixu::Context context = view->GetContext();

      if (!m_FindIsosurface.IsValid())
      {
        m_FindIsosurface = view->AddRayGenerationProgram("FindIsosurface");
      }

      if (!m_gaussLegendreNodesBuffer.Initialized())
      {
        std::vector<ElVisFloat> nodes;
        ReadFloatVector("Nodes.txt", nodes);
        m_gaussLegendreNodesBuffer.SetContext(context);
        m_gaussLegendreNodesBuffer.SetDimensions(nodes.size());
        auto nodeData = m_gaussLegendreNodesBuffer.Map();
        std::copy(nodes.begin(), nodes.end(), nodeData.get());
      }

      if (!m_gaussLegendreWeightsBuffer.Initialized())
      {
        std::vector<ElVisFloat> weights;
        ReadFloatVector("Weights.txt", weights);
        m_gaussLegendreWeightsBuffer.SetContext(context);
        m_gaussLegendreWeightsBuffer.SetDimensions(weights.size());
        auto data = m_gaussLegendreWeightsBuffer.Map();
        std::copy(weights.begin(), weights.end(), data.get());
      }

      if (!m_monomialConversionTableBuffer.Initialized())
      {
        std::vector<ElVisFloat> monomialCoversionData;
        ReadFloatVector("MonomialConversionTables.txt", monomialCoversionData);
        m_monomialConversionTableBuffer.SetContext(context);
        m_monomialConversionTableBuffer.SetDimensions(
          monomialCoversionData.size());
        auto data = m_monomialConversionTableBuffer.Map();
        std::copy(monomialCoversionData.begin(), monomialCoversionData.end(),
                  data.get());
      }
      m_isovalueBuffer.SetContext(context);
      m_isovalueBuffer.SetDimensions(0);
      m_requiredOrderBuffer.SetContext(context);
      m_requiredOrderBuffer.SetDimensions(0);
      m_epsilonBuffer.SetContext(context);
      m_epsilonBuffer.SetDimensions(0);
    }
    catch (optixu::Exception& e)
    {
      std::cout << "Exception encountered setting up isosurface." << std::endl;
      std::cerr << e.getErrorString() << std::endl;
      std::cout << e.getErrorString().c_str() << std::endl;
    }
    catch (std::exception& e)
    {
      std::cout << "Exception encountered setting up isosurface." << std::endl;
      std::cout << e.what() << std::endl;
    }
    catch (...)
    {
      std::cout << "Exception encountered setting up isosurface." << std::endl;
    }
  }

  void IsosurfaceModule::DoSynchronize(SceneView* view)
  {
    if( m_dirty )
    {
      m_isovalueBuffer.SetDimensions(m_isovalues.size());

      if (!m_isovalues.empty())
      {
        auto isovalueData = m_isovalueBuffer.Map();
        std::copy(m_isovalues.begin(), m_isovalues.end(), isovalueData.get());
      }

      if( ! m_requiredOrder.empty() )
      {
          m_requiredOrderBuffer.SetDimensions(m_requiredOrder.size());
          auto requiredOrderData = m_requiredOrderBuffer.Map();
          std::copy(m_requiredOrder.begin(), m_requiredOrder.end(), requiredOrderData.get());
      }

      if( ! m_epsilon.empty() )
      {
          m_epsilonBuffer.SetDimensions(m_epsilon.size());
          auto epsilonData = m_epsilonBuffer.Map();
          std::copy(m_epsilon.begin(), m_epsilon.end(), epsilonData.get());
      }
      m_dirty = true;
    }
  }

  void IsosurfaceModule::ReadFloatVector(const std::string& fileName,
                                         std::vector<ElVisFloat>& values)
  {
    std::ifstream inFile(fileName.c_str());

    while (!inFile.eof())
    {
      std::string line;
      std::getline(inFile, line);
      if (line.empty())
      {
        continue;
      }

      try
      {
        ElVisFloat value = boost::lexical_cast<ElVisFloat>(line);
        values.push_back(value);
      }
      catch (boost::bad_lexical_cast&)
      {
        std::cout << "Unable to parse " << line << std::endl;
      }
    }
    inFile.close();
  }

  void IsosurfaceModule::serialize(boost::archive::xml_oarchive& ar, unsigned int version) const
  {
    ar & boost::serialization::make_nvp(ISOVALUE_KEY_NAME.c_str(), m_isovalues);
  }

  void IsosurfaceModule::deserialize(boost::archive::xml_iarchive& ar, unsigned int version)
  {
    ar & boost::serialization::make_nvp(ISOVALUE_KEY_NAME.c_str(), m_isovalues);
    m_dirty = true;
    OnIsovaluesChanged();
    SetSyncAndRenderRequired();
  }
}
