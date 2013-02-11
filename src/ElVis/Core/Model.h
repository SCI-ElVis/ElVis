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

#ifndef ELVIS_MODEL_H
#define ELVIS_MODEL_H

#include <ElVis/Core/Point.hpp>
#include <ElVis/Core/FieldInfo.h>

#include <optixu/optixpp.h>
#include <vector>
#include <ElVis/Core/ElVisDeclspec.h>

#include <cuda.h>

namespace ElVis
{
    class Scene;

    class Model
    {
        public:
            ELVIS_EXPORT Model();
            ELVIS_EXPORT Model(const Model& rhs);
            ELVIS_EXPORT ~Model();

            const WorldPoint& MinExtent() const { return m_minExtent; }
            const WorldPoint& MaxExtent() const { return m_maxExtent; }

            ELVIS_EXPORT unsigned int GetNumberOfPoints() const { return DoGetNumberOfPoints(); }
            ELVIS_EXPORT WorldPoint GetPoint(unsigned int id) const { return DoGetPoint(id); }

            ELVIS_EXPORT unsigned int GetNumberOfElements() const { return DoGetNumberOfElements(); }

            ELVIS_EXPORT void SetupCudaContext(CUmodule module) const { return DoSetupCudaContext(module); }

            /// \brief Generates teh OptiX geometry nodes necessary to perform ray/element intersections.
            ELVIS_EXPORT std::vector<optixu::GeometryGroup> GetCellGeometry(Scene* scene, optixu::Context context, CUmodule module);
            ELVIS_EXPORT void GetFaceGeometry(Scene* scene, optixu::Context context, CUmodule module, optixu::Geometry& faces);

            // This doesn't really belong in the model class, but is a good, quick place to put it for now.
            // It really should be a part of the SceneView class, but the individual extensions don't currently 
            // created customized SceneViews.
            ELVIS_EXPORT const std::string& GetPTXPrefix() const { return DoGetPTXPrefix(); }

            ELVIS_EXPORT const std::string& GetCUBinPrefix() const { return DoGetCUBinPrefix(); }

            ELVIS_EXPORT void MapInteropBuffersForCuda() { return DoMapInteropBufferForCuda(); }
            ELVIS_EXPORT void UnMapInteropBuffersForCuda() { return DoUnMapInteropBufferForCuda(); }
            ELVIS_EXPORT void CalculateExtents();
            ELVIS_EXPORT const WorldPoint& GetMidpoint();

            ELVIS_EXPORT int GetNumFields() const { return DoGetNumFields(); }
            ELVIS_EXPORT FieldInfo GetFieldInfo(unsigned int index) const { return DoGetFieldInfo(index); }

            ELVIS_EXPORT int GetNumberOfBoundarySurfaces() const;
            ELVIS_EXPORT void GetBoundarySurface(int surfaceIndex, std::string& name, std::vector<int>& faceIds);

        protected:
            ELVIS_EXPORT virtual std::vector<optixu::GeometryGroup> DoGetCellGeometry(Scene* scene, optixu::Context context, CUmodule module) = 0;
            ELVIS_EXPORT virtual void DoGetFaceGeometry(Scene* scene, optixu::Context context, CUmodule module, optixu::Geometry& faces) = 0;

            /// \brief Maps Opix/Cuda interop buffers to be used by Cuda.
            ///
            /// If the model has created any OptiX/Cuda interop buffers, they must be mapped here.
            /// This method is called immediately before starting a cuda kernel.
            ELVIS_EXPORT virtual void DoMapInteropBufferForCuda() = 0;

            /// \brief Unmaps Opix/Cuda interop buffers to be used by OptiX.
            ///
            /// If the model has created any OptiX/Cuda interop buffers, they must be unmapped here.
            /// This method is called immediately after a cuda kernel completes.
            ELVIS_EXPORT virtual void DoUnMapInteropBufferForCuda() = 0;

            ELVIS_EXPORT virtual int DoGetNumberOfBoundarySurfaces() const = 0;
            ELVIS_EXPORT virtual void DoGetBoundarySurface(int surfaceIndex, std::string& name, std::vector<int>& faceIds) = 0;

            void SetMinExtent(const WorldPoint& min) { m_minExtent = min; }
            void SetMaxExtent(const WorldPoint& max) { m_maxExtent = max; }
            

            ELVIS_EXPORT virtual void DoCalculateExtents(WorldPoint& min, WorldPoint& max) = 0;
            virtual unsigned int DoGetNumberOfPoints() const = 0;
            virtual WorldPoint DoGetPoint(unsigned int id) const = 0;

            virtual void DoSetupCudaContext(CUmodule module) const = 0;
            virtual const std::string& DoGetCUBinPrefix() const = 0;
            virtual const std::string& DoGetPTXPrefix() const = 0;
            virtual unsigned int DoGetNumberOfElements() const = 0;
            virtual int DoGetNumFields() const = 0;
            virtual FieldInfo DoGetFieldInfo(unsigned int index) const = 0;

        private:
            Model& operator=(const Model& rhs);

            WorldPoint m_minExtent;
            WorldPoint m_maxExtent;
            WorldPoint m_center;
    };

}

#endif //ELVIS_MODEL_H
