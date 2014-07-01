#ifndef HIGH_ORDER_ISOSURFACE_POINT_HPP
#define HIGH_ORDER_ISOSURFACE_POINT_HPP

#include <ElVis/Core/Spaces.h>
#include <ElVis/Core/Float.h>
#include <boost/static_assert.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#include <boost/signals2.hpp>
#include <boost/serialization/split_member.hpp>

#include <string>
#include <string>

namespace ElVis
{
    template<typename data_type, typename dim, typename space = DefaultSpace>
    class Point
    {
        public:
            typedef data_type DataType;
            typedef Point<data_type, dim, space> ThisType;

            boost::signals2::signal< void (const ThisType&) > OnPointChanged;
            boost::signals2::signal< void (const DataType&) > OnXChanged;
            boost::signals2::signal< void (const DataType&) > OnYChanged;
            boost::signals2::signal< void (const DataType&) > OnZChanged;

        public:
            Point()
            {
                // This may be suboptimal if DataType isn't numeric.
                // If we use them then maybe we could look at an enable_if
                // template to choose a better constructor.
                for(unsigned int i = 0; i < dim::Value; ++i)
                {
                    // If you get a compile error pointing you here then
                    // the DataType being stored in the point doesn't have an
                    // accessible operator= or default copy constructor.
                    m_data[i] = DataType();
                }
            }

            Point(const std::string& pointValues)
            {
                bool result = fromString(pointValues, *this);
            }


            Point(const DataType& x,
                  const DataType& y)
            {
                BOOST_STATIC_ASSERT(dim::Value == 2);
                m_data[0] = x;
                m_data[1] = y;
            }

            Point(const DataType& x,
                  const DataType& y,
                  const DataType& z)
            {
                BOOST_STATIC_ASSERT(dim::Value == 3);
                m_data[0] = x;
                m_data[1] = y;
                m_data[2] = z;
            }

            explicit Point(const ElVisFloat3& values)
            {
                BOOST_STATIC_ASSERT(dim::Value == 3);
                m_data[0] = values.x;
                m_data[1] = values.y;
                m_data[2] = values.z;
            }


            explicit Point(const ElVisFloat4& values)
            {
                BOOST_STATIC_ASSERT(dim::Value == 3);
                m_data[0] = values.x;
                m_data[1] = values.y;
                m_data[2] = values.z;
            }

            explicit Point(const DataType& a)
            {
                for(unsigned int i = 0; i < dim::Value; ++i)
                {
                    m_data[i] = a;
                }
            }

            Point(const Point<DataType, dim, space>& rhs) :
              OnPointChanged(),
              OnXChanged(),
              OnYChanged(),
              OnZChanged()
            {
                for(unsigned int i = 0; i < dim::Value; ++i)
                {
                    m_data[i] = rhs.m_data[i];
                }
            }

            ~Point()
            {
            }

            Point<DataType, dim, space>& operator=(const Point<DataType, dim, space>& rhs)
            {
                for(unsigned int i = 0; i < dim::Value; ++i)
                {
                    m_data[i] = rhs.m_data[i];
                }
                if( dim::Value <= 1 )
                {
                    OnXChanged(m_data[0]);
                }
                if( dim::Value <= 2 )
                {
                    OnYChanged(m_data[1]);
                }
                if( dim::Value <= 3)
                {
                    OnZChanged(m_data[2]);
                }
                OnPointChanged(*this);
                return *this;
            }

            /// \brief Returns the number of dimensions for the point.
            static unsigned int dimension() { return dim::Value; }

            /// \brief Returns i^{th} element.
            /// \param i The element to return.
            /// \pre i < dim
            /// \return A reference to the i^{th} element.
            ///
            /// Retrieves the i^{th} element.  Since it returns a reference you may
            /// assign a new value (i.e., p(2) = 3.2;)
            ///
            /// This operator performs range checking.
//            DataType& operator()(unsigned int i)
//            {
//                return m_data[i];
//            }

            const DataType& operator()(unsigned int i) const
            {
                return m_data[i];
            }

//            DataType& operator[](unsigned int i)
//            {
//                return m_data[i];
//            }

            void SetValue(unsigned int i, const DataType& value)
            {
                m_data[i] = value;
            }

            const DataType& operator[](unsigned int i) const
            {
                return m_data[i];
            }

            const DataType& x() const
            {
                BOOST_STATIC_ASSERT(dim::Value >= 1);
                return m_data[0];
            }

            const DataType& y() const
            {
                BOOST_STATIC_ASSERT(dim::Value >= 2);
                return (*this)[1];
            }

            const DataType& z() const
            {
                BOOST_STATIC_ASSERT(dim::Value >= 3);
                return (*this)[2];
            }

            const DataType& a() const
            {
                BOOST_STATIC_ASSERT(dim::Value >= 1);
                return m_data[0];
            }

            const DataType& b() const
            {
                BOOST_STATIC_ASSERT(dim::Value >= 2);
                return (*this)[1];
            }

            const DataType& c() const
            {
                BOOST_STATIC_ASSERT(dim::Value >= 3);
                return (*this)[2];
            }

            const DataType& r() const
            {
                BOOST_STATIC_ASSERT(dim::Value >= 1);
                return m_data[0];
            }

            const DataType& s() const
            {
                BOOST_STATIC_ASSERT(dim::Value >= 2);
                return (*this)[1];
            }

            const DataType& t() const
            {
                BOOST_STATIC_ASSERT(dim::Value >= 3);
                return (*this)[2];
            }

            void SetX(const DataType& val)
            {
                BOOST_STATIC_ASSERT(dim::Value >= 1);
                if( m_data[0] == val ) return;
                m_data[0] = val;
                OnPointChanged(*this);
                OnXChanged(val);
            }

            void SetY(const DataType& val)
            {
                BOOST_STATIC_ASSERT(dim::Value >= 2);
                if( m_data[1] == val ) return;
                m_data[1] = val;
                OnPointChanged(*this);
                OnYChanged(val);
            }

            void SetZ(const DataType& val)
            {
                BOOST_STATIC_ASSERT(dim::Value >= 2);
                if( m_data[2] == val ) return;
                m_data[2] = val;
                OnPointChanged(*this);
                OnZChanged(val);
            }



//            DataType& x()
//            {
//                BOOST_STATIC_ASSERT(dim::Value >= 1);
//                return (*this)(0);
//            }

//            DataType& y()
//            {
//                BOOST_STATIC_ASSERT(dim::Value >= 2);
//                return (*this)(1);
//            }

//            DataType& z()
//            {
//                BOOST_STATIC_ASSERT(dim::Value >= 3);
//                return (*this)(2);
//            }

//            DataType* GetPtr()
//            {
//                return &m_data[0];
//            }

            const DataType* GetPtr() const
            {
                return &m_data[0];
            }

            bool operator==(const Point<DataType, dim, space>& rhs) const
            {
                for(unsigned int i = 0; i < dim::Value; ++i)
                {
                    // If you get a compile error here then you have to
                    // add a != operator to the DataType class.
                    if( m_data[i] != rhs.m_data[i] )
                    {
                        return false;
                    }
                }
                return true;
            }

            bool operator!=(const Point<DataType, dim, space>& rhs) const
            {
                return !(*this == rhs);
            }

            void PublishElementChanges()
            {
                if( dim::Value <= 1 )
                {
                    OnXChanged(m_data[0]);
                }
                if( dim::Value <= 2 )
                {
                    OnYChanged(m_data[1]);
                }
                if( dim::Value <= 3)
                {
                    OnZChanged(m_data[2]);
                }
            }

            /// Arithmetic Routines

            // Unitary operators
            void negate()
            {
                for(int i=0; i < dim::Value; ++i)
                {
                    m_data[i] = -m_data[i];
                }
                PublishElementChanges();
                OnPointChanged(*this);
            }

            Point<DataType, dim, space> operator-() const
            {
                Point<DataType, dim, space> result(*this);
                result.negate();
                return result;
            }


            Point<DataType, dim, space>& operator+=(const Point<DataType, dim, space>& rhs)
            {
                for(unsigned int i=0; i < dim::Value; ++i)
                {
                    m_data[i] += rhs.m_data[i];
                }
                PublishElementChanges();
                OnPointChanged(*this);
                return *this;
            }

            Point<DataType, dim, space>& operator+=(const DataType& rhs)
            {
                for(unsigned int i = 0; i < dim::Value; ++i)
                {
                    m_data[i] += rhs;
                }
                PublishElementChanges();
                OnPointChanged(*this);
                return *this;
            }

            Point<DataType, dim, space>& operator-=(const Point<DataType, dim, space>& rhs)
            {
                for(unsigned int i=0; i < dim::Value; ++i)
                {
                    m_data[i] -= rhs.m_data[i];
                }
                PublishElementChanges();
                OnPointChanged(*this);
                return *this;
            }


            Point<DataType, dim, space>& operator-=(const DataType& rhs)
            {
                for(unsigned int i = 0; i < dim::Value; ++i)
                {
                    m_data[i] -= rhs;
                }
                PublishElementChanges();
                OnPointChanged(*this);
                return *this;
            }

            Point<DataType, dim, space>& operator*=(const DataType& rhs)
            {
                for(unsigned int i = 0; i < dim::Value; ++i)
                {
                    m_data[i] *= rhs;
                }
                PublishElementChanges();
                OnPointChanged(*this);
                return *this;
            }

            Point<DataType, dim, space>& operator/=(const DataType& rhs)
            {
                for(unsigned int i = 0; i < dim::Value; ++i)
                {
                    m_data[i] /= rhs;
                }
                PublishElementChanges();
                OnPointChanged(*this);
                return *this;
            }

            std::string AsString() const
            {
                std::string result = "(";
                for(unsigned int i = 0; i < dim::Value; ++i)
                {
                    result += boost::lexical_cast<std::string>(m_data[i]);
                    if( i < dim::Value-1 )
                    {
                        result += ", ";
                    }
                }
                result += ")";
                return result;
            }

            template<typename Archive>
            void NotifyLoad(Archive& ar, const unsigned int version, 
                typename boost::enable_if<typename Archive::is_saving>::type* p = 0)
            {
            }

            template<typename Archive>
            void NotifyLoad(Archive& ar, const unsigned int version, 
                typename boost::enable_if<typename Archive::is_loading>::type* p = 0)
            {
                this->UpdateBasisVectors();
                this->OnCameraChanged();
            }

            template<typename Archive>
            void serialize(Archive& ar, const unsigned int version)
            {
                ar & BOOST_SERIALIZATION_NVP(m_data);    
                NotifyLoad(ar, version);
            }

        private:
            DataType m_data[dim::Value];
    };
    
    template<typename DataType, typename dim, typename space>
    Point<DataType, dim, space>
    operator+(const Point<DataType, dim, space>& lhs, const Point<DataType, dim, space>& rhs)
    {
        Point<DataType, dim, space> result(lhs);
        result += rhs;
        return result;
    }

    template<typename DataType, typename dim, typename space>
    Point<DataType, dim, space>
    operator+(const DataType& lhs, const Point<DataType, dim, space>& rhs)
    {
        Point<DataType, dim, space> result(rhs);
        result += lhs;
        return result;
    }

    template<typename DataType, typename dim, typename space>
    Point<DataType, dim, space>
    operator+(const Point<DataType, dim, space>& lhs, const DataType& rhs)
    {
        Point<DataType, dim, space> result(lhs);
        result += rhs;
        return result;
    }

    template<typename DataType, typename dim, typename space>
    Point<DataType, dim, space>
    operator-(const Point<DataType, dim, space>& lhs, const Point<DataType, dim, space>& rhs)
    {
        Point<DataType, dim, space> result(lhs);
        result -= rhs;
        return result;
    }

    template<typename DataType, typename dim, typename space>
    Point<DataType, dim, space>
    operator-(const DataType& lhs, const Point<DataType, dim, space>& rhs)
    {
    Point<DataType, dim, space> result(-rhs);
    result += lhs;
    return result;
    }

    template<typename DataType, typename dim, typename space>
    Point<DataType, dim, space>
    operator-(const Point<DataType, dim, space>& lhs, const DataType& rhs)
    {
        Point<DataType, dim, space> result(lhs);
        result -= rhs;
        return result;
    }
    
    template<typename DataType, typename dim, typename space, typename ScalarType>
    Point<DataType, dim, space>
    operator*(const ScalarType& lhs, const Point<DataType, dim, space>& rhs)
    {
        Point<DataType, dim, space> result(rhs);
        result *= lhs;
        return result;
    }

    template<typename DataType, typename dim, typename space, typename ScalarType>
    Point<DataType, dim, space>
    operator*(const Point<DataType, dim, space>& lhs, const ScalarType& rhs)
    {
        Point<DataType, dim, space> result(lhs);
        result *= rhs;
        return result;
    }

    template<typename DataType, typename dim, typename space>
    Point<DataType, dim, space>
    operator/(const Point<DataType, dim, space>& lhs, const DataType& rhs)
    {
        Point<DataType, dim, space> result(lhs);
        result /= rhs;
        return result;
    }

    template<typename DataType, typename dim, typename space>
    DataType distanceBetween(const Point<DataType, dim, space>& lhs,
    const Point<DataType, dim, space>& rhs)
    {
        DataType result = 0.0;
        for(unsigned int i = 0; i < dim::Value; ++i)
        {
            DataType temp = lhs[i] - rhs[i];
            result += temp*temp;
        }
        return sqrt(result);
    }

    template<typename DataType, typename dim, typename space>
    bool fromString(const std::string& str, Point<DataType, dim, space>& result)
    {
        try
        {
            typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
            boost::char_separator<char> sep("(<,>) ");
            tokenizer tokens(str, sep);
            unsigned int i = 0;
            for(tokenizer::iterator iter = tokens.begin(); iter != tokens.end(); ++iter)
            {
                result[i] = boost::lexical_cast<DataType>(*iter);
                ++i;
            }

        return i == dim::Value;
        }
        catch(boost::bad_lexical_cast&)
        {
            return false;
        }
    }

    template<typename DataType, typename dim, typename space>
    std::ostream& operator<<(std::ostream& os, const Point<DataType, dim, space>& p)
    {
        os << p.AsString();
        return os;
    }

    template<typename DataType, typename dim, typename space>
    std::istream& operator>>(std::istream& is, Point<DataType, dim, space>& obj)
    {
        for(unsigned int i = 0; i < dim::Value; ++i)
        {
            DataType temp;
            is >> temp;
            obj[i] = temp;
        }
        return is;
    }

    template<typename DataType, typename dim, typename space>
    bool operator<(const Point<DataType, dim, space>& lhs, const Point<DataType, dim, space>& rhs)
    {
        for(unsigned int i = 0; i < dim::Value; ++i)
        {
            if( lhs[i] < rhs[i] ) return true;
            if( lhs[i] > rhs[i] ) return false;
        }
        return false;
    }
    
    template<typename DataType, typename dim, typename space>
    Point<DataType, dim, space> CalcMin(const Point<DataType, dim, space>& lhs, const Point<DataType, dim, space>& rhs)
    {
        Point<DataType, dim, space> result;
        for(unsigned int i = 0; i < dim::Value; ++i)
        {
            result.SetValue(i, std::min(lhs[i], rhs[i]));
        }
        return result;
    }

    template<typename DataType, typename dim, typename space>
    Point<DataType, dim, space> CalcMax(const Point<DataType, dim, space>& lhs, const Point<DataType, dim, space>& rhs)
    {
        Point<DataType, dim, space> result;
        for(unsigned int i = 0; i < dim::Value; ++i)
        {
            result.SetValue(i, std::max(lhs[i], rhs[i]));
        }
        return result;
    }

    /// \brief A point in world space.
    typedef Point<ElVisFloat, ThreeD, WorldSpace> WorldPoint;

    /// \brief A point in reference space.
    typedef Point<ElVisFloat, ThreeD, ReferenceSpace> ReferencePoint;
}

#endif //HIGH_ORDER_ISOSURFACE_POINT_HPP
