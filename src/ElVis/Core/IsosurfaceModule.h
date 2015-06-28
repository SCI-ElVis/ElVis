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

#ifndef ELVIS_ISOSURFACE_MODULE_H
#define ELVIS_ISOSURFACE_MODULE_H

#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/RenderModule.h>
#include <ElVis/Core/RenderModule.h>
#include <ElVis/Core/RayGeneratorProgram.h>
#include <ElVis/Core/Float.h>
#include <ElVis/Core/ElementId.h>
#include <ElVis/Core/OptiXBuffer.hpp>

#include <set>

#include <boost/signals2.hpp>

namespace ElVis
{
  class IsosurfaceModule : public RenderModule
  {
  public:
    ELVIS_EXPORT IsosurfaceModule();
    ELVIS_EXPORT virtual ~IsosurfaceModule() {}
    IsosurfaceModule& operator=(const IsosurfaceModule& rhs) = delete;
    IsosurfaceModule(const IsosurfaceModule& rhs) = delete;

    ELVIS_EXPORT virtual void DoRender(SceneView* view);

    ELVIS_EXPORT void AddIsovalue(const ElVisFloat& value);
    ELVIS_EXPORT void RemoveIsovalue(const ElVisFloat& value);

    ELVIS_EXPORT const std::set<ElVisFloat> GetIsovalues() const
    {
      return m_isovalues;
    }

    boost::signals2::signal<void()> OnIsovaluesChanged;

  protected:
    ELVIS_EXPORT virtual void DoSynchronize(SceneView* view);
    ELVIS_EXPORT virtual void DoSetup(SceneView* view);

    virtual int DoGetNumberOfRequiredEntryPoints() { return 1; }
    virtual std::string DoGetName() const { return "Isosurface Rendering"; }

    virtual void serialize(boost::archive::xml_oarchive&, unsigned int version) const override;
    virtual void deserialize(boost::archive::xml_iarchive&, unsigned int version) override;

  private:
    /// \brief Reads a vector of floating point value from a file.
    /// Projecting an arbitrary smooth function onto a polynomial requires
    /// the nodes and weights for numerical ingegration.  These values
    /// are stored in a file in the ElVis installation directory.
    static void ReadFloatVector(const std::string& fileName,
                                std::vector<ElVisFloat>& values);

    std::set<ElVisFloat> m_isovalues;
    bool m_dirty;

    OptiXBuffer<ElVisFloat, RT_BUFFER_INPUT> m_isovalueBuffer;
    OptiXBuffer<ElVisFloat, RT_BUFFER_INPUT> m_gaussLegendreNodesBuffer;
    OptiXBuffer<ElVisFloat, RT_BUFFER_INPUT> m_gaussLegendreWeightsBuffer;
    OptiXBuffer<ElVisFloat, RT_BUFFER_INPUT> m_monomialConversionTableBuffer;

    static RayGeneratorProgram m_FindIsosurface;
  };
}

#endif
