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

#ifndef ELVIS_CORE_VIEW_SETTINGS_H
#define ELVIS_CORE_VIEW_SETTINGS_H

#include <ElVis/Core/Point.hpp>
#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/Vector.hpp>
#include <ElVis/Core/matrix.cu>

#include <boost/signals2.hpp>

#include <iostream>

#include <boost/serialization/split_member.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

namespace ElVis
{
    class Camera
    {
        public:
            boost::signals2::signal< void() > OnCameraChanged;

        public:
            ELVIS_EXPORT Camera();
            ELVIS_EXPORT Camera(const Camera& rhs);
            ELVIS_EXPORT ~Camera();
            ELVIS_EXPORT Camera& operator=(const Camera& rhs);

            /// \brief Returns the camera's current eye point.
            ELVIS_EXPORT const WorldPoint& GetEye() const { return m_eye; }

            /// \brief Returns the camera's current look at point.
            ELVIS_EXPORT const WorldPoint& GetLookAt() const { return m_lookAt; }

            /// \brief Returns the camera's current up vector.
            ELVIS_EXPORT const WorldVector& GetUp() const { return m_up; }

            /// \brief Returns the camera's current eye point.
            ///
            /// It is safe to change the eye point through the returned reference.
            ELVIS_EXPORT WorldPoint& GetEye() { return m_eye; }

            /// \brief Returns the camera's current look at point.
            ///
            /// It is safe to change the look at point through the reference.
            ELVIS_EXPORT WorldPoint& GetLookAt() { return m_lookAt; }

            /// \brief Returns the camera's current up vector.
            ///
            /// It is safe to change the up vector through the reference.
            ELVIS_EXPORT WorldVector& GetUp() { return m_up; }


            /// \brief Get the un-normalized vector from the eye point to the look at point.
            ELVIS_EXPORT const WorldVector& GetGaze() const { return m_w; }

            /// \brief Returns the camera's current vertical field of view
            ELVIS_EXPORT const double GetFieldOfView() const { return m_fieldOfView; }

            /// \brief Returns the aspect ratio of the view port.
            ELVIS_EXPORT const double GetAspectRatio() const { return m_aspectRatio; }

            /// \brief Sets the vertical field of view.
            ELVIS_EXPORT void SetFieldOfView(double value);

            /// \brief Sets the aspect ratio width/height.
            ELVIS_EXPORT void SetAspectRatio(double value);

            /// \brief Sets the aspect ratio width/height.
            ELVIS_EXPORT void SetAspectRatio(unsigned int width, unsigned int height);

            ELVIS_EXPORT float GetNear() const;
            ELVIS_EXPORT float GetFar() const;
            ELVIS_EXPORT void SetNear(float value);
            ELVIS_EXPORT void SetFar(float value);

            ELVIS_EXPORT void SetLookAt(const WorldPoint& value);
            ELVIS_EXPORT void SetParameters(const WorldPoint& eye, const WorldPoint& at, const WorldVector& up);
            ELVIS_EXPORT void SetParameters(const WorldPoint& eye, const WorldPoint& at, const WorldVector& up, ElVisFloat fov, ElVisFloat near, ElVisFloat far);

            ELVIS_EXPORT void MoveEyeAlongGaze(int from_x, int from_y, int to_x, int to_y, int imageWidth, int imageHeight);

            /// Rotate based on mouse movement.
            ELVIS_EXPORT void Rotate(int from_x, int from_y, int to_x, int to_y, int width, int height);

            ELVIS_EXPORT void Pan(int from_x, int from_y, int to_x, int to_y, int imageWidth, int imageHeight);

            ELVIS_EXPORT const WorldVector& GetU() const { return m_u; }
            ELVIS_EXPORT const WorldVector& GetV() const { return m_v; }
            ELVIS_EXPORT const WorldVector& GetW() const { return m_w; }

            ELVIS_EXPORT WorldVector GetNormalizedU() const;
            ELVIS_EXPORT WorldVector GetNormalizedV() const;
            ELVIS_EXPORT WorldVector GetNormalizedW() const;

            ELVIS_EXPORT void SetupOpenGLPerspective();

            template<typename Archive>
            void NotifyLoad(Archive& ar, const unsigned int version, 
                typename boost::enable_if<typename Archive::is_saving>::type* p = 0)
            {
            }

            template<typename Archive>
            void NotifyLoad(Archive& ar, const unsigned int version, 
                typename boost::enable_if<typename Archive::is_loading>::type* p = 0)
            {
                UpdateBasisVectors();
                OnCameraChanged();
            }

            template<typename Archive>
            void serialize(Archive& ar, const unsigned int version)
            {
                ar & BOOST_SERIALIZATION_NVP(m_fieldOfView);    
                ar & BOOST_SERIALIZATION_NVP(m_aspectRatio);
                ar & BOOST_SERIALIZATION_NVP(m_near);
                ar & BOOST_SERIALIZATION_NVP(m_far);

                ar & BOOST_SERIALIZATION_NVP(m_eye);
                ar & BOOST_SERIALIZATION_NVP(m_lookAt);
                ar & BOOST_SERIALIZATION_NVP(m_up);

                ar & BOOST_SERIALIZATION_NVP(m_u);
                ar & BOOST_SERIALIZATION_NVP(m_v);
                ar & BOOST_SERIALIZATION_NVP(m_w);
            }

        private:
            static ElVisFloat3 ProjectToSphere( ElVisFloat x, ElVisFloat y, ElVisFloat radius );
            static Matrix4x4 RotationMatrix( const ElVisFloat3& _to, const ElVisFloat3& _from );

            void SetupSignals();
            void HandleEyeAtChanged(const WorldPoint& p);
            void HandleUpChanged(const WorldVector& v);

            ELVIS_EXPORT void UpdateBasisVectors();

            Matrix4x4 InitWithBasis();
            void Transform( const Matrix4x4& trans, bool maintainDistance);

            WorldPoint m_eye;
            WorldPoint m_lookAt;
            WorldVector m_up;

            WorldVector m_u;
            WorldVector m_v;
            // The vector from eye to lookat is -m_w.
            WorldVector m_w;

            double m_fieldOfView;
            double m_aspectRatio;
            float m_near;
            float m_far;
    };

    ELVIS_EXPORT bool operator==(const Camera& lhs, const Camera& rhs);
    ELVIS_EXPORT std::ostream& operator<<(std::ostream& os, const Camera& c);
}


#endif //ELVIS_CORE_VIEW_SETTINGS_H
