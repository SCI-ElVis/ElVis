///////////////////////////////////////////////////////////////////////////////
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
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

#ifndef ELVIS_CORE_STAT_H
#define ELVIS_CORE_STAT_H

#include <math.h>
#include <boost/math/distributions/students_t.hpp>
#include <ElVis/Core/Float.h>
#include <limits>

namespace ElVis
{
    class Stat
    {
        public:

            Stat() :
                Mean(0.0),
                StdDev(0.0),
                Min(std::numeric_limits<double>::max()),
                Max(-std::numeric_limits<double>::max())
            {
            };

            Stat(const Stat& rhs) :
                Mean(rhs.Mean),
                StdDev(rhs.StdDev),
                Min(rhs.Min),
                Max(rhs.Max) {}

            Stat& operator=(const Stat& rhs)
            {
                Mean = rhs.Mean;
                StdDev = rhs.StdDev;
                Min = rhs.Min;
                Max = rhs.Max;
                return *this;
            }


            Stat(const double* samples, double cutoff, int sampleSize, double confidence) :
                Mean(0.0),
                HalfWidth(0.0),
                Confidence(confidence),
                StdDev(0.0)
            {
                Calculate(samples, cutoff, sampleSize);
            }

            Stat(const double* samples, double cutoff, int sampleSize) :
                Mean(0.0),
                HalfWidth(0.0),
                Confidence(.95),
                StdDev(0.0)
            {
                Calculate(samples, cutoff, sampleSize);
            }

            bool Overlaps(const Stat& other)
            {
                return (Low() >= other.Low() && Low() <= other.High()) ||
                    (High() >= other.Low() && High() <= other.High());
            }

            double Low() const { return Mean - HalfWidth; }
            double High() const { return Mean + HalfWidth; }
            double Mean;
            double HalfWidth;
            double Confidence;
            double StdDev;
            double Min;
            double Max;

        private:
            void Calculate(const double* samples, double cutoff, int sampleSize)
            {
                double sum = 0.0;
                int numValidSamples = 0;
                for(int i = 0; i < sampleSize; ++i)
                {
                    if( samples[i] < cutoff )
                    {
                        sum += samples[i];
                        ++numValidSamples;

                        Min = std::min(Min, samples[i]);
                        Max = std::max(Max, samples[i]);
                    }
                }
                Mean = sum/(double)numValidSamples;

                StdDev = 0.0;
                for(int i = 0; i < sampleSize; ++i)
                {
                    if( samples[i] < cutoff )
                    {
                        StdDev += (samples[i] - Mean)*(samples[i]-Mean);
                    }
                }

                if( numValidSamples > 1 )
                {
                    StdDev = sqrt(StdDev/(numValidSamples-1));
                }

                boost::math::students_t dist(sampleSize);
                double T = boost::math::quantile(boost::math::complement(dist, (1.0-Confidence)/2.0));
                HalfWidth = T*StdDev/sqrt(static_cast<double>(sampleSize));
            }
    };
}

#endif
