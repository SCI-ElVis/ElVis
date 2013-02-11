//#ifndef ELVIS_JACOBI_EXTENSION_HIGH_ORDER_ISOSURFACE_POINT_HPP
//#define ELVIS_JACOBI_EXTENSION_HIGH_ORDER_ISOSURFACE_POINT_HPP
//
//#include <ElVis/Core/Point.hpp>
//
//
//namespace JacobiExtension
//{
//    typedef ElVis::ElVis::WorldPoint ElVis::WorldPoint;
//    typedef OriginalNektar::ElVis::TensorPoint ElVis::TensorPoint;
//    typedef OriginalNektar::ElVis::ReferencePoint ElVis::ReferencePoint;
//}
//
////#include <ElVis/Extensions/JacobiExtension/Spaces.h>
////#include <boost/static_assert.hpp>
////#include <boost/lexical_cast.hpp>
////#include <boost/tokenizer.hpp>
////#include <string>
////#include <string>
////
////namespace JacobiExtension
////{
////    template<typename data_type, typename dim, typename space = DefaultSpace>
////    class Point
////    {
////        public:
////            typedef data_type DataType;
////
////        public:
////            Point()
////            {
////                // This may be suboptimal if DataType isn't numeric.
////                // If we use them then maybe we could look at an enable_if
////                // template to choose a better constructor.
////                for(unsigned int i = 0; i < dim::Value; ++i)
////                {
////                    // If you get a compile error pointing you here then
////                    // the DataType being stored in the point doesn't have an
////                    // accessible operator= or default copy constructor.
////                    m_data[i] = DataType();
////                }
////            }
////
////            Point(const std::string& pointValues)
////            {
////                bool result = fromString(pointValues, *this);
////            }
////
////
////            Point(const DataType& x,
////                  const DataType& y)
////            {
////                BOOST_STATIC_ASSERT(dim::Value == 2);
////                m_data[0] = x;
////                m_data[1] = y;
////            }
////
////            Point(const DataType& x,
////                  const DataType& y,
////                  const DataType& z)
////            {
////                BOOST_STATIC_ASSERT(dim::Value == 3);
////                m_data[0] = x;
////                m_data[1] = y;
////                m_data[2] = z;
////            }
////
////            explicit Point(const DataType& a)
////            {
////                for(unsigned int i = 0; i < dim::Value; ++i)
////                {
////                    m_data[i] = a;
////                }
////            }
////
////            Point(const Point<DataType, dim, space>& rhs)
////            {
////                for(unsigned int i = 0; i < dim::Value; ++i)
////                {
////                    m_data[i] = rhs.m_data[i];
////                }
////            }
////
////            ~Point()
////            {
////            }
////
////            Point<DataType, dim, space>& operator=(const Point<DataType, dim, space>& rhs)
////            {
////                for(unsigned int i = 0; i < dim::Value; ++i)
////                {
////                    m_data[i] = rhs.m_data[i];
////                }
////                return *this;
////            }
////
////            /// \brief Returns the number of dimensions for the point.
////            static unsigned int dimension() { return dim::Value; }
////
////            /// \brief Returns i^{th} element.
////            /// \param i The element to return.
////            /// \pre i < dim
////            /// \return A reference to the i^{th} element.
////            ///
////            /// Retrieves the i^{th} element.  Since it returns a reference you may
////            /// assign a new value (i.e., p(2) = 3.2;)
////            ///
////            /// This operator performs range checking.
////            DataType& operator()(unsigned int i)
////            {
////                return m_data[i];
////            }
////
////            const DataType& operator()(unsigned int i) const
////            {
////                return m_data[i];
////            }
////
////            DataType& operator[](unsigned int i)
////            {
////                return m_data[i];
////            }
////
////            const DataType& operator[](unsigned int i) const
////            {
////                return m_data[i];
////            }
////
////            const DataType& x() const
////            {
////                BOOST_STATIC_ASSERT(dim::Value >= 1);
////                return m_data[0];
////            }
////
////            const DataType& y() const
////            {
////                BOOST_STATIC_ASSERT(dim::Value >= 2);
////                return (*this)[1];
////            }
////
////            const DataType& z() const
////            {
////                BOOST_STATIC_ASSERT(dim::Value >= 3);
////                return (*this)[2];
////            }
////
////            const DataType& a() const
////            {
////                BOOST_STATIC_ASSERT(dim::Value >= 1);
////                return m_data[0];
////            }
////
////            const DataType& b() const
////            {
////                BOOST_STATIC_ASSERT(dim::Value >= 2);
////                return (*this)[1];
////            }
////
////            const DataType& c() const
////            {
////                BOOST_STATIC_ASSERT(dim::Value >= 3);
////                return (*this)[2];
////            }
////
////            const DataType& r() const
////            {
////                BOOST_STATIC_ASSERT(dim::Value >= 1);
////                return m_data[0];
////            }
////
////            const DataType& s() const
////            {
////                BOOST_STATIC_ASSERT(dim::Value >= 2);
////                return (*this)[1];
////            }
////
////            const DataType& t() const
////            {
////                BOOST_STATIC_ASSERT(dim::Value >= 3);
////                return (*this)[2];
////            }
////
////            void SetX(const DataType& val)
////            {
////                BOOST_STATIC_ASSERT(dim::Value >= 1);
////                m_data[0] = val;
////            }
////
////            void SetY(const DataType& val)
////            {
////                BOOST_STATIC_ASSERT(dim::Value >= 2);
////                m_data[1] = val;
////            }
////
////            void SetZ(const DataType& val)
////            {
////                BOOST_STATIC_ASSERT(dim::Value >= 2);
////                m_data[2] = val;
////            }
////
////            DataType& x()
////            {
////                BOOST_STATIC_ASSERT(dim::Value >= 1);
////                return (*this)(0);
////            }
////
////            DataType& y()
////            {
////                BOOST_STATIC_ASSERT(dim::Value >= 2);
////                return (*this)(1);
////            }
////
////            DataType& z()
////            {
////                BOOST_STATIC_ASSERT(dim::Value >= 3);
////                return (*this)(2);
////            }
////
////            const DataType* GetPtr() const
////            {
////                return &m_data[0];
////            }
////
////            bool operator==(const Point<DataType, dim, space>& rhs) const
////            {
////                for(unsigned int i = 0; i < dim::Value; ++i)
////                {
////            // If you get a compile error here then you have to
////            // add a != operator to the DataType class.
////                    if( m_data[i] != rhs.m_data[i] )
////                    {
////                        return false;
////                    }
////                }
////                return true;
////            }
////
////            bool operator!=(const Point<DataType, dim, space>& rhs) const
////            {
////                return !(*this == rhs);
////            }
////
////            /// Arithmetic Routines
////
////            // Unitary operators
////            void negate()
////            {
////                for(int i=0; i < dim::Value; ++i)
////                {
////                    (*this)[i] = -(*this)[i];
////                }
////            }
////
////            Point<DataType, dim, space> operator-() const
////            {
////                Point<DataType, dim, space> result(*this);
////                result.negate();
////                return result;
////            }
////
////
////            Point<DataType, dim, space>& operator+=(const Point<DataType, dim, space>& rhs)
////            {
////                for(unsigned int i=0; i < dim::Value; ++i)
////                {
////                    m_data[i] += rhs.m_data[i];
////                }
////                return *this;
////            }
////
////            Point<DataType, dim, space>& operator+=(const DataType& rhs)
////            {
////                for(unsigned int i = 0; i < dim::Value; ++i)
////                {
////                    m_data[i] += rhs;
////                }
////                return *this;
////            }
////
////            Point<DataType, dim, space>& operator-=(const Point<DataType, dim, space>& rhs)
////            {
////                for(unsigned int i=0; i < dim::Value; ++i)
////                {
////                    m_data[i] -= rhs.m_data[i];
////                }
////                return *this;
////            }
////
////
////            Point<DataType, dim, space>& operator-=(const DataType& rhs)
////            {
////                for(unsigned int i = 0; i < dim::Value; ++i)
////                {
////                    m_data[i] -= rhs;
////                }
////                return *this;
////            }
////
////            Point<DataType, dim, space>& operator*=(const DataType& rhs)
////            {
////                for(unsigned int i = 0; i < dim::Value; ++i)
////                {
////                    m_data[i] *= rhs;
////                }
////                return *this;
////            }
////
////            Point<DataType, dim, space>& operator/=(const DataType& rhs)
////            {
////                for(unsigned int i = 0; i < dim::Value; ++i)
////                {
////                    m_data[i] /= rhs;
////                }
////                return *this;
////            }
////
////            std::string AsString() const
////            {
////                std::string result = "(";
////                for(unsigned int i = 0; i < dim::Value; ++i)
////                {
////                    result += boost::lexical_cast<std::string>(m_data[i]);
////                    if( i < dim::Value-1 )
////                    {
////                        result += ", ";
////                    }
////                }
////                result += ")";
////                return result;
////            }
////
////        private:
////            DataType m_data[dim::Value];
////    };
////    
////    template<typename DataType, typename dim, typename space>
////    Point<DataType, dim, space>
////    operator+(const Point<DataType, dim, space>& lhs, const Point<DataType, dim, space>& rhs)
////    {
////        Point<DataType, dim, space> result(lhs);
////        result += rhs;
////        return result;
////    }
////
////    template<typename DataType, typename dim, typename space>
////    Point<DataType, dim, space>
////    operator+(const DataType& lhs, const Point<DataType, dim, space>& rhs)
////    {
////        Point<DataType, dim, space> result(rhs);
////        result += lhs;
////        return result;
////    }
////
////    template<typename DataType, typename dim, typename space>
////    Point<DataType, dim, space>
////    operator+(const Point<DataType, dim, space>& lhs, const DataType& rhs)
////    {
////        Point<DataType, dim, space> result(lhs);
////        result += rhs;
////        return result;
////    }
////
////    template<typename DataType, typename dim, typename space>
////    Point<DataType, dim, space>
////    operator-(const Point<DataType, dim, space>& lhs, const Point<DataType, dim, space>& rhs)
////    {
////        Point<DataType, dim, space> result(lhs);
////        result -= rhs;
////        return result;
////    }
////
////    template<typename DataType, typename dim, typename space>
////    Point<DataType, dim, space>
////    operator-(const DataType& lhs, const Point<DataType, dim, space>& rhs)
////    {
////    Point<DataType, dim, space> result(-rhs);
////    result += lhs;
////    return result;
////    }
////
////    template<typename DataType, typename dim, typename space>
////    Point<DataType, dim, space>
////    operator-(const Point<DataType, dim, space>& lhs, const DataType& rhs)
////    {
////        Point<DataType, dim, space> result(lhs);
////        result -= rhs;
////        return result;
////    }
////    
////    template<typename DataType, typename dim, typename space, typename ScalarType>
////    Point<DataType, dim, space>
////    operator*(const ScalarType& lhs, const Point<DataType, dim, space>& rhs)
////    {
////        Point<DataType, dim, space> result(rhs);
////        result *= lhs;
////        return result;
////    }
////
////    template<typename DataType, typename dim, typename space, typename ScalarType>
////    Point<DataType, dim, space>
////    operator*(const Point<DataType, dim, space>& lhs, const ScalarType& rhs)
////    {
////        Point<DataType, dim, space> result(lhs);
////        result *= rhs;
////        return result;
////    }
////
////    template<typename DataType, typename dim, typename space>
////    Point<DataType, dim, space>
////    operator/(const Point<DataType, dim, space>& lhs, const DataType& rhs)
////    {
////        Point<DataType, dim, space> result(lhs);
////        result /= rhs;
////        return result;
////    }
////
////    template<typename DataType, typename dim, typename space>
////    DataType distanceBetween(const Point<DataType, dim, space>& lhs,
////    const Point<DataType, dim, space>& rhs)
////    {
////        DataType result = 0.0;
////        for(unsigned int i = 0; i < dim::Value; ++i)
////        {
////            DataType temp = lhs[i] - rhs[i];
////            result += temp*temp;
////        }
////        return sqrt(result);
////    }
////
////    template<typename DataType, typename dim, typename space>
////    bool fromString(const std::string& str, Point<DataType, dim, space>& result)
////    {
////        try
////        {
////            typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
////            boost::char_separator<char> sep("(<,>) ");
////            tokenizer tokens(str, sep);
////            unsigned int i = 0;
////            for(tokenizer::iterator iter = tokens.begin(); iter != tokens.end(); ++iter)
////            {
////                result[i] = boost::lexical_cast<DataType>(*iter);
////                ++i;
////            }
////
////        return i == dim::Value;
////        }
////        catch(boost::bad_lexical_cast&)
////        {
////            return false;
////        }
////    }
////
////    template<typename DataType, typename dim, typename space>
////    std::ostream& operator<<(std::ostream& os, const Point<DataType, dim, space>& p)
////    {
////        os << p.AsString();
////        return os;
////    }
////
////    template<typename DataType, typename dim, typename space>
////    std::istream& operator>>(std::istream& is, Point<DataType, dim, space>& obj)
////    {
////        for(unsigned int i = 0; i < dim::Value; ++i)
////		{
////            DataType temp;
////            is >> temp;
////            obj[i] = temp;
////        }
////        return is;
////    }
////
////    template<typename DataType, typename dim, typename space>
////    bool operator<(const Point<DataType, dim, space>& lhs, const Point<DataType, dim, space>& rhs)
////    {
////        for(unsigned int i = 0; i < dim::Value; ++i)
////        {
////            if( lhs[i] < rhs[i] ) return true;
////            if( lhs[i] > rhs[i] ) return false;
////        }
////        return false;
////    }
////    
////    template<typename DataType, typename dim, typename space>
////    Point<DataType, dim, space> Min(const Point<DataType, dim, space>& lhs, const Point<DataType, dim, space>& rhs)
////    {
////        Point<DataType, dim, space> result(lhs);
////        for(unsigned int i = 0; i < dim::Value; ++i)
////        {
////            result[i] = std::min(lhs[i], rhs[i]);
////        }
////        return result;
////    }
////
////    template<typename DataType, typename dim, typename space>
////    Point<DataType, dim, space> Max(const Point<DataType, dim, space>& lhs, const Point<DataType, dim, space>& rhs)
////    {
////        Point<DataType, dim, space> result(lhs);
////        for(unsigned int i = 0; i < dim::Value; ++i)
////        {
////            result[i] = std::max(lhs[i], rhs[i]);
////        }
////        return result;
////    }
////
////    typedef Point<double, ThreeD, WorldSpace> ElVis::WorldPoint;
////    typedef Point<double, ThreeD, ReferenceSpace> ElVis::ReferencePoint;
////    typedef Point<double, ThreeD, TensorSpace> ElVis::TensorPoint;
////}
//
//#endif //HIGH_ORDER_ISOSURFACE_POINT_HPP