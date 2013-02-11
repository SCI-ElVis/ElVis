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

#ifndef ELVIS_NEKTAR_MODEL_H
#define ELVIS_NEKTAR_MODEL_H

#include <ElVis/Core/Model.h>
#include <ElVis/Extensions/Nektar++Extension/Declspec.h>

#include <ElVis/Extensions/NektarExtension/Declspec.h>

#include <boost/filesystem/path.hpp>

#include <ElVis/Extensions/NektarExtension/include/nektar.h>

namespace ElVis
{
    namespace NektarExtension
    {
        class NektarModel : public Model
        {
            public:
                NEKTAR_EXTENSION_EXPORT explicit NektarModel(const std::string& fileName);
    //            NEKTAR_EXTENSION_EXPORT explicit NektarModel(const std::string& geometryFile);
    //            NEKTAR_EXTENSION_EXPORT NektarModel(const std::string& geometryFile, const std::string& fieldFile);

                NEKTAR_EXTENSION_EXPORT virtual ~NektarModel();

    //            NEKTAR_EXTENSION_EXPORT void InitializeWithGeometry(const std::string& geomFileName);
    //            NEKTAR_EXTENSION_EXPORT void InitializeWithGeometryAndField(const std::string& geomFileName,
    //                const std::string& fieldFileName);

                /// \brief Displays the model geometry to the current OpenGL context.
                //void DisplayGeometry();

            protected:
                virtual std::vector<optixu::GeometryGroup> DoGetCellGeometry(optixu::Context context, unsigned int elementFinderRayIndex);

                NEKTAR_EXTENSION_EXPORT unsigned int DoGetNumberOfPoints() const;
                NEKTAR_EXTENSION_EXPORT WorldPoint DoGetPoint(unsigned int id) const;
                NEKTAR_EXTENSION_EXPORT const std::string& DoGetPTXPrefix() const;

                NEKTAR_EXTENSION_EXPORT virtual void DoSetupCudaContext(CUmodule module) const;
                NEKTAR_EXTENSION_EXPORT virtual const std::string& DoGetCUBinPrefix() const;

            private:
                NektarModel(const NektarModel& rhs);
                NektarModel& operator=(NektarModel& rhs);

                void SetupOptixCoefficientBuffers(optixu::Context context);
                void SetupOptixVertexBuffers(optixu::Context context);
                template<typename T>
                optixu::GeometryInstance CreateGeometryForElementType(optixu::Context context)
                {
    //                unsigned int numElements = m_graph->GetAllElementsOfType<T>().size();

    //                optixu::Geometry geometry = context->createGeometry();
    //                geometry->setPrimitiveCount(numElements);

                    optixu::GeometryInstance instance = context->createGeometryInstance();
    //                instance->setGeometry(geometry);

    //                optixu::Buffer vertexIndexBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, numElements*8);
    //                instance["HexVertexIndices"]->set(vertexIndexBuffer);
    //                unsigned int* coefficientIndicesData = static_cast<unsigned int*>(vertexIndexBuffer->map());
    //                for(unsigned int i = 0; i < numElements; ++i)
    //                {
    //                    // TODO - Check the correspondence between vertex id and global id.
    //                    BOOST_AUTO(hex, m_graph->GetAllElementsOfType<T>()[i]);
    //                    for(unsigned int j = 0; j < 8; ++j)
    //                    {
    //                        unsigned int vid = hex->GetVid(j);
    //                        coefficientIndicesData[i*8 + j] = vid;
    //                    }
    //                }
    //                vertexIndexBuffer->unmap();

    //                optixu::Buffer numberOfModesBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, numElements*3);
    //                instance["NumberOfModes"]->set(numberOfModesBuffer);
    //                unsigned int* modesData = static_cast<unsigned int*>(numberOfModesBuffer->map());

    //                for(unsigned int i = 0; i < numElements; ++i)
    //                {
    //                    // TODO - Check the correspondence between vertex id and global id.
    //                    BOOST_AUTO(hex, m_graph->GetAllElementsOfType<T>()[i]);
    //                    BOOST_AUTO(localExpansion, m_globalExpansion->GetExp(hex->GetGlobalID()));

    //                    modesData[i*3] = localExpansion->GetBasis(0)->GetNumModes();
    //                    modesData[i*3+1] = localExpansion->GetBasis(1)->GetNumModes();
    //                    modesData[i*3+2] = localExpansion->GetBasis(2)->GetNumModes();

    //                }

    //                numberOfModesBuffer->unmap();


                    return instance;

                }

                Element_List* m_elementList;
        };
    }
}


#endif //ELVIS_NEKTAR_MODEL_H
