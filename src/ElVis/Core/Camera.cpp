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

#include <ElVis/Core/Camera.h>
#include <ElVis/Core/Util.hpp>
#include <ElVis/Core/Float.cu>
#include <ElVis/Core/OpenGL.h>
#include <ElVis/Core/Vector.hpp>
#include <ElVis/Core/Point.hpp>

#include <boost/bind.hpp>

namespace ElVis
{
    Camera::Camera() :
        m_eye(0, 0, 5),
        m_lookAt(0, 0, 0),
        m_up(0, 1, 0),
        m_u(-1, 0, 0),
        m_v(0, 1, 0),
        m_w(0, 0, -1),
        m_fieldOfView(60.0),
        m_aspectRatio(1.0),
        m_near(.1),
        m_far(10000.0)
    {
        SetupSignals();
    }
    
    Camera::Camera(const Camera& rhs) :
        m_eye(rhs.m_eye),
        m_lookAt(rhs.m_lookAt),
        m_up(rhs.m_up),
        m_u(rhs.m_u),
        m_v(rhs.m_v),
        m_w(rhs.m_w),
        m_fieldOfView(rhs.m_fieldOfView),
        m_aspectRatio(rhs.m_aspectRatio),
        m_near(rhs.m_near),
        m_far(rhs.m_far)
    {
        SetupSignals();
    }

    Camera::~Camera() { }
    
    Camera& Camera::operator=(const Camera& rhs)
    {
        m_eye = rhs.m_eye;
        m_lookAt = rhs.m_lookAt;
        m_up = rhs.m_up;
        m_u = rhs.m_u;
        m_v = rhs.m_v;
        m_w = rhs.m_w;
        m_fieldOfView = rhs.m_fieldOfView;
        m_aspectRatio = rhs.m_aspectRatio;
        OnCameraChanged();
        return *this;
    }


    void Camera::SetupSignals()
    {
        m_eye.OnPointChanged.connect(boost::bind(&Camera::HandleEyeAtChanged, this, _1));
        m_lookAt.OnPointChanged.connect(boost::bind(&Camera::HandleEyeAtChanged, this, _1));
        m_up.OnVectorChanged.connect(boost::bind(&Camera::HandleUpChanged, this, _1));
    }

    void Camera::HandleEyeAtChanged(const WorldPoint& p)
    {
        UpdateBasisVectors();
        OnCameraChanged();
    }

    void Camera::HandleUpChanged(const WorldVector& v)
    {
        UpdateBasisVectors();
        OnCameraChanged();
    }

    void Camera::SetParameters(const WorldPoint& eye, const WorldPoint& at, const WorldVector& up)
    {
        SetParameters(eye, at, up, m_fieldOfView, m_near, m_far);
    }

    void Camera::SetParameters(const WorldPoint& eye, const WorldPoint& at, const WorldVector& up, ElVisFloat fov, ElVisFloat nearParam, ElVisFloat farParam)
    {
        m_eye = eye;
        m_lookAt = at;
        m_up = up;
        m_fieldOfView = fov;
        m_near = nearParam;
        m_far = farParam;
        UpdateBasisVectors();
        OnCameraChanged();
    }

    ELVIS_EXPORT void Camera::SetLookAt(const WorldPoint& value)
    {
        m_lookAt = value;
        UpdateBasisVectors();
        OnCameraChanged();
    }


    void Camera::SetFieldOfView(double value)
    {
        if( value != m_fieldOfView &&
            value > 0.0 &&
            value < 90.0 )
        {
            m_fieldOfView = value;
            UpdateBasisVectors();
            OnCameraChanged();
        }
    }
    void Camera::SetAspectRatio(double value)
    {
        if( value != m_aspectRatio )
        {
            m_aspectRatio = value;
            UpdateBasisVectors();
            OnCameraChanged();
        }
    }

    void Camera::SetAspectRatio(unsigned int width, unsigned int height)
    {
        double ratio = static_cast<double>(width)/static_cast<double>(height);
        SetAspectRatio(ratio);
    }


    float Camera::GetNear() const { return m_near; }
    float Camera::GetFar() const { return m_far; }
    void Camera::SetNear(float value)
    {
        if( m_near != value )
        {
            m_near = value;
            OnCameraChanged();
        }
    }

    void Camera::SetFar(float value)
    {
        if( m_far != value )
        {
            m_far = value;
            OnCameraChanged();
        }
    }

    //    Calculate appropriate U,V,W for pinhole_camera shader.
    //      eye          : camera eye position
    //      lookat       : point in scene camera looks at
    //      up           : up direction
    //      hfov         : vertical field of fiew
    //      aspect_ratio : image aspect ratio (width/height)
    //      U            : [out] U coord for camera shader
    //      V            : [out] V coord for camera shader
    //      W            : [out] W coord for camera shader
    // Note that U,V,W are not normalized
    void Camera::UpdateBasisVectors()
    {
        m_w[0] = m_lookAt[0] - m_eye[0];
        m_w[1] = m_lookAt[1] - m_eye[1];  // Do not normalize W -- it implies focal length
        m_w[2] = m_lookAt[2] - m_eye[2];

        double wlen = sqrt( Dot( m_w, m_w ) );
        m_u = Cross( m_w, m_up);
        Normalize( m_u );
        m_v = Cross( m_u, m_w);
        Normalize( m_v );
        double vlen = wlen * tan( m_fieldOfView / 2.0 * 3.14159265358979323846 / 180.0 );
        m_v[0] *= vlen;
        m_v[1] *= vlen;
        m_v[2] *= vlen;
        double ulen =  vlen*m_aspectRatio;
        m_u[0] *= ulen;
        m_u[1] *= ulen;
        m_u[2] *= ulen;
    }

    void Camera::MoveEyeAlongGaze(int from_x, int from_y, int to_x, int to_y, int imageWidth, int imageHeight)
    {
        ElVisFloat percent = static_cast<ElVisFloat>(from_y-to_y)/static_cast<ElVisFloat>(imageHeight);
        ElVisFloat startDistance = distanceBetween(m_eye, m_lookAt);
        ElVisFloat endDistance = startDistance + percent*5;

        if( endDistance < m_near)
        {
            endDistance = m_near;
        }
        WorldVector reverseVector = -GetGaze();
        reverseVector.Normalize();
        m_eye = m_lookAt + findPointAlongVector(reverseVector, endDistance);

    }

    void Camera::Pan(int from_x, int from_y, int to_x, int to_y, int imageWidth, int imageHeight)
    {
        double percentVertical = static_cast<double>(to_y - from_y)/static_cast<double>(imageHeight) ;
        double percentHorizontal =  static_cast<double>(-to_x + from_x)/static_cast<double>(imageWidth);

        WorldVector dir = percentVertical*m_v + percentHorizontal*m_u;

        m_eye = m_eye + findPointAlongVector(dir, 2.0);
        m_lookAt = m_lookAt + findPointAlongVector(dir, 2.0);
    }
    
    WorldVector Camera::GetNormalizedU() const
    {
        WorldVector result = m_u;
        result.Normalize();
        return result;
    }

    WorldVector Camera::GetNormalizedV() const
    {
        WorldVector result = m_v;
        result.Normalize();
        return result;
    }

    WorldVector Camera::GetNormalizedW() const
    {
        WorldVector result = m_w;
        result.Normalize();
        return result;
    }


    void Camera::Rotate(int from_x, int from_y, int to_x, int to_y, int width, int height)
    {
        ElVisFloat from_x_pos = 2.0*static_cast<ElVisFloat>(from_x)/static_cast<ElVisFloat>(width) - 1.0;
        ElVisFloat from_y_pos = 1.0 - 2.0*static_cast<ElVisFloat>(from_y)/static_cast<ElVisFloat>(height);
        
        ElVisFloat to_x_pos = 2.0*static_cast<ElVisFloat>(to_x)/static_cast<ElVisFloat>(width) - 1.0;
        ElVisFloat to_y_pos = 1.0 - 2.0*static_cast<ElVisFloat>(to_y)/static_cast<ElVisFloat>(height);

        
        ElVisFloat3 to = ProjectToSphere( from_x_pos, from_y_pos, 0.8 );
        ElVisFloat3 from = ProjectToSphere(to_x_pos, to_y_pos, .8);

        Matrix4x4 m = RotationMatrix( to, from);
        Transform( m, true);
        OnCameraChanged();
    }

    void Camera::SetupOpenGLPerspective()
    {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(m_fieldOfView, m_aspectRatio, m_near, m_far);
        glMatrixMode(GL_MODELVIEW);
    }

    Matrix4x4 Camera::InitWithBasis()
    {
        ElVisFloat m[16];
        m[0] = static_cast<ElVisFloat>(m_u.x());
        m[1] = static_cast<ElVisFloat>(m_v.x());
        m[2] = static_cast<ElVisFloat>(m_w.x());
        m[3] = static_cast<ElVisFloat>(m_lookAt.x());

        m[4] = static_cast<ElVisFloat>(m_u.y());
        m[5] = static_cast<ElVisFloat>(m_v.y());
        m[6] = static_cast<ElVisFloat>(m_w.y());
        m[7] = static_cast<ElVisFloat>(m_lookAt.y());

        m[8] = static_cast<ElVisFloat>(m_u.z());
        m[9] = static_cast<ElVisFloat>(m_v.z());
        m[10] = static_cast<ElVisFloat>(m_w.z());
        m[11] = static_cast<ElVisFloat>(m_lookAt.z());

        m[12] = 0.0f;
        m[13] = 0.0f;
        m[14] = 0.0f;
        m[15] = 1.0f;

        return Matrix4x4( m );
    }

    inline ElVisFloat det3 (ElVisFloat a, ElVisFloat b, ElVisFloat c,
                     ElVisFloat d, ElVisFloat e, ElVisFloat f,
                     ElVisFloat g, ElVisFloat h, ElVisFloat i)
    { return a*e*i + d*h*c + g*b*f - g*e*c - d*b*i - a*h*f; }

    #define mm(i,j) m[i*4+j]
    ElVisFloat det4( const Matrix4x4& m )
    {
        ElVisFloat det;
        det  = mm(0,0) * det3(mm(1,1), mm(1,2), mm(1,3),
                          mm(2,1), mm(2,2), mm(2,3),
                          mm(3,1), mm(3,2), mm(3,3));
        det -= mm(0,1) * det3(mm(1,0), mm(1,2), mm(1,3),
                          mm(2,0), mm(2,2), mm(2,3),
                          mm(3,0), mm(3,2), mm(3,3));
        det += mm(0,2) * det3(mm(1,0), mm(1,1), mm(1,3),
                          mm(2,0), mm(2,1), mm(2,3),
                          mm(3,0), mm(3,1), mm(3,3));
        det -= mm(0,3) * det3(mm(1,0), mm(1,1), mm(1,2),
                          mm(2,0), mm(2,1), mm(2,2),
                          mm(3,0), mm(3,1), mm(3,2));
        return det;
    }
    

    Matrix4x4 inverse( const Matrix4x4& m )
    {
        Matrix4x4 inverse;
        ElVisFloat det = det4( m );

        inverse[0]  =  det3(mm(1,1), mm(1,2), mm(1,3),
                        mm(2,1), mm(2,2), mm(2,3),
                        mm(3,1), mm(3,2), mm(3,3)) / det;
        inverse[1]  = -det3(mm(0,1), mm(0,2), mm(0,3),
                        mm(2,1), mm(2,2), mm(2,3),
                        mm(3,1), mm(3,2), mm(3,3)) / det;
        inverse[2]  =  det3(mm(0,1), mm(0,2), mm(0,3),
                        mm(1,1), mm(1,2), mm(1,3),
                        mm(3,1), mm(3,2), mm(3,3)) / det;
        inverse[3]  = -det3(mm(0,1), mm(0,2), mm(0,3),
                        mm(1,1), mm(1,2), mm(1,3),
                        mm(2,1), mm(2,2), mm(2,3)) / det;

        inverse[4]  = -det3(mm(1,0), mm(1,2), mm(1,3),
                        mm(2,0), mm(2,2), mm(2,3),
                        mm(3,0), mm(3,2), mm(3,3)) / det;
        inverse[5]  =  det3(mm(0,0), mm(0,2), mm(0,3),
                        mm(2,0), mm(2,2), mm(2,3),
                        mm(3,0), mm(3,2), mm(3,3)) / det;
        inverse[6]  = -det3(mm(0,0), mm(0,2), mm(0,3),
                        mm(1,0), mm(1,2), mm(1,3),
                        mm(3,0), mm(3,2), mm(3,3)) / det;
        inverse[7]  =  det3(mm(0,0), mm(0,2), mm(0,3),
                        mm(1,0), mm(1,2), mm(1,3),
                        mm(2,0), mm(2,2), mm(2,3)) / det;

        inverse[8]  =  det3(mm(1,0), mm(1,1), mm(1,3),
                        mm(2,0), mm(2,1), mm(2,3),
                        mm(3,0), mm(3,1), mm(3,3)) / det;
        inverse[9]  = -det3(mm(0,0), mm(0,1), mm(0,3),
                        mm(2,0), mm(2,1), mm(2,3),
                        mm(3,0), mm(3,1), mm(3,3)) / det;
        inverse[10] =  det3(mm(0,0), mm(0,1), mm(0,3),
                        mm(1,0), mm(1,1), mm(1,3),
                        mm(3,0), mm(3,1), mm(3,3)) / det;
        inverse[11] = -det3(mm(0,0), mm(0,1), mm(0,3),
                        mm(1,0), mm(1,1), mm(1,3),
                        mm(2,0), mm(2,1), mm(2,3)) / det;

        inverse[12] = -det3(mm(1,0), mm(1,1), mm(1,2),
                        mm(2,0), mm(2,1), mm(2,2),
                        mm(3,0), mm(3,1), mm(3,2)) / det;
        inverse[13] =  det3(mm(0,0), mm(0,1), mm(0,2),
                        mm(2,0), mm(2,1), mm(2,2),
                        mm(3,0), mm(3,1), mm(3,2)) / det;
        inverse[14] = -det3(mm(0,0), mm(0,1), mm(0,2),
                        mm(1,0), mm(1,1), mm(1,2),
                        mm(3,0), mm(3,1), mm(3,2)) / det;
        inverse[15] =  det3(mm(0,0), mm(0,1), mm(0,2),
                        mm(1,0), mm(1,1), mm(1,2),
                        mm(2,0), mm(2,1), mm(2,2)) / det;

        return inverse;
    }
    #undef mm

    void Camera::Transform( const Matrix4x4& trans, bool maintainDistance)
    {      
        Matrix4x4 frame = InitWithBasis();
        Matrix4x4 frame_inv = inverse( frame );

        Matrix4x4 final_trans = frame * trans * frame_inv;
        ElVisFloat4 up4     = MakeFloat4( m_v );
        ElVisFloat4 eye4    = MakeFloat4( m_eye );
        eye4.w         = 1.0f;
        ElVisFloat4 lookat4 = MakeFloat4( m_lookAt );
        lookat4.w      = 1.0f;

        // Floating point errors tend to cause the camera to move away or towards the 
        // look at point.  Store the distance so we can restore it after the 
        // transform.
        ElVisFloat distance = distanceBetween(m_lookAt, m_eye);
        WorldVector temp_up = WorldVector( final_trans*up4 );
        WorldPoint temp_eye = WorldPoint( final_trans*eye4 );
        WorldPoint temp_lookAt = WorldPoint( final_trans*lookat4 );

        if( maintainDistance )
        {
            WorldVector restoreVector = createVectorFromPoints(temp_lookAt, temp_eye);
            restoreVector.Normalize();
            temp_eye = findPointAlongVector(restoreVector, distance) + temp_lookAt;
        }

        WorldVector reducedUp(temp_up.x(), temp_up.y(), temp_up.z());
        SetParameters(temp_eye, temp_lookAt, reducedUp);

        
    }

    Matrix4x4 Camera::RotationMatrix( const ElVisFloat3& _to, const ElVisFloat3& _from )
    {
        ElVisFloat3 from = normalize( _from );
        ElVisFloat3 to   = normalize( _to );

        ElVisFloat3 v = cross(from, to);
        ElVisFloat  e = dot(from, to);
        if ( e > 1.0f-1.e-9f ) 
        {
            return Matrix4x4::identity();
        } 
        else 
        {
            ElVisFloat h = 1.0f/(1.0f + e);
            ElVisFloat mtx[16];
            mtx[0] = e + h * v.x * v.x;
            mtx[1] = h * v.x * v.y + v.z;
            mtx[2] = h * v.x * v.z - v.y;
            mtx[3] = 0.0f;

            mtx[4] = h * v.x * v.y - v.z;
            mtx[5] = e + h * v.y * v.y;
            mtx[6] = h * v.y * v.z + v.x;
            mtx[7] = 0.0f; 

            mtx[8] = h * v.x * v.z + v.y;
            mtx[9] = h * v.y * v.z - v.x;
            mtx[10] = e + h * v.z * v.z;
            mtx[11] = 0.0f; 

            mtx[12] = 0.0f; 
            mtx[13] = 0.0f; 
            mtx[14] = 0.0f; 
            mtx[15] = 1.0f; 

            return Matrix4x4( mtx );
        }
    }

    ElVisFloat3 Camera::ProjectToSphere( ElVisFloat x, ElVisFloat y, ElVisFloat radius )
    {
        x /= radius;
        y /= radius;
        ElVisFloat r2 = x*x+y*y;
        if(r2 > 1.0f) 
        {
            ElVisFloat rad = sqrt(r2);
            x /= rad;
            y /= rad;
            return ::MakeFloat3( x, y, static_cast<ElVisFloat>(0.0) );
        } 
        else 
        {
            ElVisFloat z = sqrt(1.0f-r2);
            return ::MakeFloat3( x, y, z );
        }
    }

    bool operator==(const Camera& lhs, const Camera& rhs)
    {
        return lhs.GetEye() == rhs.GetEye() &&
            lhs.GetUp() == rhs.GetUp() &&
            lhs.GetLookAt() == rhs.GetLookAt() &&
            lhs.GetFieldOfView() == rhs.GetFieldOfView() &&
            lhs.GetAspectRatio() == rhs.GetAspectRatio() &&
            lhs.GetNear() == rhs.GetNear() &&
            lhs.GetFar() == rhs.GetFar();
    }

    std::ostream& operator<<(std::ostream& os, const Camera& c)
    {
        std::cout << "Eye: " << c.GetEye() << std::endl;
        std::cout << "Look At: " << c.GetLookAt() << std::endl;
        std::cout << "Up: " << c.GetUp() << std::endl;
        std::cout << "Gaze: " << c.GetGaze() << std::endl;
        return os;
    }
}

