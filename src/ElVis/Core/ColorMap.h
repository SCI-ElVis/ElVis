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

#ifndef COLOR_MAP_H
#define COLOR_MAP_H

#include <optixu/optixpp.h>
#include <optix_math.h>

#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/Color.h>
#include <ElVis/Core/RayGeneratorProgram.h>
#include <ElVis/Core/PtxManager.h>
#include <ElVis/Core/Float.h>

#include <iostream>

#include <boost/signals2.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/map.hpp>

namespace ElVis
{
    class SceneView;

    // an interface to color maps on [0,1]
    class ColorMap
    {
        public:
            friend class boost::serialization::access;
            ELVIS_EXPORT ColorMap();
            ELVIS_EXPORT virtual ~ColorMap() {}

            ELVIS_EXPORT void PopulateTexture(optixu::Buffer& buffer) { return DoPopulateTexture(buffer); }

            ELVIS_EXPORT void SetMin(float value);
            ELVIS_EXPORT void SetMax(float value);
            ELVIS_EXPORT float GetMin() const { return m_min; }
            ELVIS_EXPORT float GetMax() const { return m_max; }

            boost::signals2::signal<void (float)> OnMinChanged;
            boost::signals2::signal<void (float)> OnMaxChanged;
            boost::signals2::signal<void (const ColorMap&)> OnColorMapChanged;

        protected:
            virtual void DoPopulateTexture(optixu::Buffer& buffer) = 0;

        private:
            template<typename Archive>
            void serialize(Archive& ar, const unsigned int version)
            {
                ar & BOOST_SERIALIZATION_NVP(m_min);
                ar & BOOST_SERIALIZATION_NVP(m_max);
            }
            float m_min;
            float m_max;
    };

  
    /// \brief A color map formed by directly reading a block of color values.
    class TextureColorMap : public ColorMap
    {
        public:
            ELVIS_EXPORT TextureColorMap() {}
            ELVIS_EXPORT TextureColorMap(const std::string& fileName);
            
//            template<typename FuncType>
//            void GenerateEqualizedColorMapFromSamples(const std::vector<float>& samples, FuncType& f, unsigned int newSize)
//            {
//                m_size = newSize;
//                m_localData.resize(4*newSize);
                
//                std::cout << "Sample range is " << samples.front() << " - " << samples.back() << std::endl;
//                float range = samples.back() - samples.front();
//                float min = samples.front();
//                for(int i = 0; i < m_size; ++i)
//                {
//                    float percentOfColorRange = (float)i/(float)m_size;
                    
//                    // Say percetn of ColorRange is 10%.  Then 10% of all samples should map to this
//                    // color, even if the 10% value is 50% of the total scalar range.
//                    float sample = percentOfColorRange*range + min;
                    
//                    std::cout << "Sample # " << i << " = " << sample << std::endl;
                    
//                    std::vector<float>::const_iterator found = std::lower_bound(samples.begin(), samples.end(), sample);
//                    int diff = abs(std::distance(samples.begin(), found));
//                    float p = static_cast<float>(diff)/static_cast<float>(samples.size() );
//                    std::cout << "Percent of Color Range = " << percentOfColorRange << std::endl;
//                    std::cout << "p = " << p << std::endl;
                    
//                    float3 color = f(p);
//                    std::cout << "Color = (" << color.x << ", " << color.y << ", " << color.z << std::endl;
                      
//                    m_localData[4*i] = color.x;
//                    m_localData[4*i+1] = color.y;
//                    m_localData[4*i+2] = color.z;
//                    m_localData[4*i+3] = 1.0;
//                }
//                SetMin(samples.front());
//                SetMax(samples.back());
//            }
                    
            ELVIS_EXPORT void Read(const std::string& fileName);
//            ELVIS_EXPORT void Write(const std::string& fileName);
//            ELVIS_EXPORT void WriteVTK(const std::string& fileName);
        protected:
            virtual void DoPopulateTexture(optixu::Buffer& buffer);

        private:
            TextureColorMap(const TextureColorMap&);
            std::vector<float> m_localData;
    };
    
    struct ColorMapBreakpoint
    {
        friend class boost::serialization::access;
        ElVis::Color Col;
        ElVisFloat Scalar;
    private:
        template<typename Archive>
        void serialize(Archive& ar, const unsigned int version)
        {
            ar & BOOST_SERIALIZATION_NVP(Col);
            ar & BOOST_SERIALIZATION_NVP(Scalar);
        }
    };

    /// \brief Provides a color map with linear interpolation between points.
    ///
    /// This produces a color map that is slower than texture based color maps,
    /// but is potentially more accurate since the calculated color is not based
    /// on a discretization into a texture.
    class PiecewiseLinearColorMap : public ColorMap
    {
        public:
            friend class boost::serialization::access;
            ELVIS_EXPORT PiecewiseLinearColorMap();
            ELVIS_EXPORT virtual ~PiecewiseLinearColorMap() {}

            ELVIS_EXPORT void SetBreakpoint(ElVisFloat value, const Color& c);
            ELVIS_EXPORT void SetBreakpoint(const std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator& iter, const Color& c);

            ELVIS_EXPORT const std::map<ElVisFloat, ColorMapBreakpoint>& GetBreakpoints() const { return m_breakpoints; }
            ELVIS_EXPORT void RemoveBreakpoint(const std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator iter);
            ELVIS_EXPORT Color Sample(const ElVisFloat& value) const;

            ELVIS_EXPORT std::map<ElVisFloat, ColorMapBreakpoint>::iterator InsertBreakpoint(ElVisFloat value, const Color& c);

            ELVIS_EXPORT bool IsValid() const { return m_breakpoints.size() >= 2; }


        protected:
            virtual void DoPopulateTexture(optixu::Buffer& buffer);

        private:
            template<typename Archive>
            void serialize(Archive& ar, const unsigned int version)
            {
                ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ColorMap);
                ar & BOOST_SERIALIZATION_NVP(m_breakpoints);
            }

            PiecewiseLinearColorMap(const PiecewiseLinearColorMap& rhs);
            PiecewiseLinearColorMap& operator=(const PiecewiseLinearColorMap& rhs);

            std::map<ElVisFloat, ColorMapBreakpoint> m_breakpoints;
    };

}

#endif //COLOR_MAP_H

