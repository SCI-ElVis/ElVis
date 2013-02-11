//#ifndef ELVIS_JACOBI_EXTENSION_HIGH_ORDER_ISOSURFACE_VECTOR_HPP
//#define ELVIS_JACOBI_EXTENSION_HIGH_ORDER_ISOSURFACE_VECTOR_HPP
//
//#include <ElVis/Core/Point.hpp>
//
//namespace JacobiExtension
//{
//    typedef ElVis::ElVis::WorldVector ElVis::WorldVector;
//    typedef OriginalNektar::ElVis::TensorVector ElVis::TensorVector;
//    typedef OriginalNektar::ElVis::ReferenceVector ElVis::ReferenceVector;
//}
//
////#include <ElVis/Extensions/JacobiExtension/Spaces.h>
////#include <ElVis/Core/Point.hpp>
////#include <cassert>
////#include <vector>
////#include <iostream>
////
////#include <boost/type_traits.hpp>
////#include <boost/call_traits.hpp>
////#include <boost/utility/enable_if.hpp>
////
////namespace JacobiExtension
////{
////    template<typename DataType, typename enabled = void>
////    class AbsTypeTraits;
////
////    template<typename DataType>
////    class AbsTypeTraits<DataType, typename boost::enable_if<boost::is_floating_point<DataType> >::type >
////    {
////        public:
////            static 
////            typename boost::call_traits<DataType>::value_type 
////            abs(typename boost::call_traits<DataType>::const_reference v)
////            {
////                return fabs(v);
////            }
////    };
////    
////    template<typename DataType>
////    class AbsTypeTraits<DataType, typename boost::enable_if<boost::is_integral<DataType> >::type >
////    {
////        public:
////            static 
////            typename boost::call_traits<DataType>::value_type 
////            abs(typename boost::call_traits<DataType>::const_reference v)
////            {
////                return abs(v);
////            }
////    };
////    
////    template<typename DataType>
////    std::vector<DataType> FromString(const std::string& str)
////    {
////        std::vector<DataType> result;
////
////        try
////        {
////            typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
////            boost::char_separator<char> sep("(<,>) ");
////            tokenizer tokens(str, sep);
////            for( tokenizer::iterator strIter = tokens.begin(); strIter != tokens.end(); ++strIter)
////            {
////                result.push_back(boost::lexical_cast<DataType>(*strIter));
////            }
////        }
////        catch(boost::bad_lexical_cast&)
////        {
////        }
////
////        return result;
////    }
////   
////    template<typename DataType, typename space>
////    class Vector;
////    
////    template<typename DataType, typename space>
////    DataType Magnitude(const Vector<DataType, space>& v);
////    
////    
////    template<typename DataType, typename dim, typename space>
////    Vector<DataType, space> createVectorFromPoints(const Point<DataType, dim, space>& source, const Point<DataType, dim, space>& dest);
////                
////    template<typename DataType, typename space>
////    Point<DataType, ThreeD, space> findPointAlongVector(const Vector<DataType, space>& lhs, const DataType& t);
////                                
////    template<typename DataType, typename space>
////    DataType L1Norm(const Vector<DataType, space>& v);
////                                        
////    template<typename DataType, typename space>
////    DataType L2Norm(const Vector<DataType, space>& v);
////                                
////    template<typename DataType, typename space>
////    DataType InfinityNorm(const Vector<DataType, space>& v);
////        
////    template<typename DataType, typename space>
////    Vector<DataType, space> Negate(const Vector<DataType, space>& v);
////
////    template<typename DataType, typename space>
////    DataType Magnitude(const Vector<DataType, space>& v);
////                                                         
////    template<typename DataType, typename space>
////    DataType Dot(const Vector<DataType, space>& lhs, const Vector<DataType, space>& rhs);
////                                                                            
////
////    template<typename DataType, typename space>
////    void Normalize(Vector<DataType, space>& v);
////                                               
////    template<typename DataType, typename space>
////    Vector<DataType, space> Cross(const Vector<DataType, space>& lhs, const Vector<DataType, space>& rhs);
////                                                                            
////    template<typename DataType, typename space>
////    std::string AsString(const Vector<DataType, space>& v);
////                                                                            
////                                                                            
////    template<typename DataType, typename space>
////    class Vector
////    {
////        public:
////            /// \brief Creates a vector of size 3.
////            Vector() :
////                m_data(3)
////            {
////            }
////            
////            /// \brief Creates a vector of given size.  The elements are not initialized.
////            explicit Vector(unsigned int size) :
////                m_data(size)
////            {
////            }
////
////            /// \brief Creates a vector with given size and initial value.
////            Vector(unsigned int size, const DataType& a) :
////                m_data(size, a)
////            {
////            }
////
////
////            explicit Vector(const std::string& vectorValues) :
////                m_data()
////            {
////                m_data = FromString<DataType>(vectorValues);
////            }
////
////            Vector(const DataType& x,
////                   const DataType& y,
////                   const DataType& z) :
////                m_data(3)
////            {
////                m_data[0] = x;
////                m_data[1] = y;
////                m_data[2] = z;
////            }
////            
////            Vector(const Vector<DataType, space>& rhs) :
////                m_data(rhs.m_data)
////            {
////            }
////            
////            
////            Vector(unsigned int size, const DataType* const ptr) :
////                m_data(size, ptr)
////            {
////            }
////
////    
////            Vector<DataType, space>& operator=(const Vector<DataType, space>& rhs)
////            {
////                m_data = rhs.m_data;
////                return *this;
////
////            }
////            
////            
////            /// \brief Returns the number of dimensions for the point.
////            inline unsigned int GetDimension() const
////            {
////                return m_data.size();
////            }
////
////            inline unsigned int GetRows() const
////            {
////                return m_data.size();
////            }
////
////
////            const DataType* GetRawPtr() const
////            {
////                return &m_data[0];
////            }
////            
////            DataType* GetRawPtr()
////            {
////                return &m_data[0];
////            }
////
////                                  
////            typedef const DataType* const_iterator;
////            const_iterator begin() const { return GetRawPtr(); }
////            const_iterator end() const { return GetRawPtr() + GetDimension(); }
////
////            typedef DataType* iterator;
////            
////            iterator begin() { return GetRawPtr(); }
////            iterator end() { return GetRawPtr() + this->GetDimension(); }
////            
////            const DataType& operator()(unsigned int i) const
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
////                return (*this)(0);
////            }
////
////            const DataType& y() const
////            {
////                return (*this)(1);
////            }
////
////            const DataType& z() const
////            {
////                return (*this)(2);
////            }
////
////            DataType& operator()(unsigned int i)
////            {
////                return m_data[i];
////            }
////
////            DataType& operator[](unsigned int i)
////            {
////                return m_data[i];
////            }
////
////            DataType& x()
////            {
////                return (*this)(0);
////            }
////
////            DataType& y()
////            {
////                return (*this)(1);
////            }
////
////            DataType& z()
////            {
////                return (*this)(2);
////            }
////
////            void SetX(const DataType& val)
////            {
////                m_data[0] = val;
////            }
////
////            void SetY(const DataType& val)
////            {
////                m_data[1] = val;
////            }
////
////            void SetZ(const DataType& val)
////            {
////                m_data[2] = val;
////            }
////            
////            Vector<DataType, space> operator-() const { return Negate(*this); }
////        
////            DataType Magnitude() const { return JacobiExtension::Magnitude(*this); }
////        
////            DataType Dot(const Vector<DataType, space>& rhs) const { return JacobiExtension::Dot(*this, rhs); }
////        
////            Vector<DataType, space> Cross(const Vector<DataType, space>& rhs) const
////            {
////                return JacobiExtension::Cross(*this, rhs);
////            }
////        
////            std::string AsString() const { return JacobiExtension::AsString(*this); }
////
////            // Norms
////            DataType L1Norm() const { return JacobiExtension::L1Norm(*this); }
////            DataType L2Norm() const { return JacobiExtension::L2Norm(*this); }
////            DataType InfinityNorm() const { return JacobiExtension::InfinityNorm(*this); }
////    
////  
////            Vector<DataType, space>& operator+=(const Vector<DataType, space>& rhs)
////            {
////                PlusEqual(*this, rhs);
////                return *this;
////            }
////
////            Vector<DataType, space>& operator-=(const Vector<DataType, space>& rhs)
////            {
////                MinusEqual(*this, rhs);
////                return *this;
////            }
////
////            Vector<DataType, space>& operator*=(const DataType& rhs)
////            {
////                TimesEqual(*this, rhs);
////                return *this;
////            }
////            
////            Vector<DataType, space>& operator/=(const DataType& rhs)
////            {
////                DivideEqual(*this, rhs);
////                return *this;
////            }
////
////            
////            void Normalize() { return JacobiExtension::Normalize(*this); }
////            
////        protected:
////                        
////        private:
////            std::vector<DataType> m_data;
////    };
////    
////	template<typename DataType, typename space>
////	std::ostream& operator<<(std::ostream& os, const Vector<DataType, space>& rhs)
////	{
////		for(unsigned int i = 0; i < rhs.GetDimension(); ++i)
////		{
////			if( i > 0 )
////			{
////				os << ", ";
////			}
////			os << rhs[i];
////		}
////		return os;
////	}
////
////    template<typename DataType, typename space>
////    std::istream& operator>>(std::istream& is, Vector<DataType, space>& obj)
////    {
////        for(unsigned int i = 0; i < obj.GetDimension(); ++i)
////        {
////            DataType temp;
////            is >> temp;
////            obj[i] = temp;
////        }
////        return is;
////    }
////
////	template<typename DataType, typename space>
////	bool operator==(const Vector<DataType, space>& lhs, const Vector<DataType, space>& rhs)
////	{
////		if( lhs.GetDimension() != rhs.GetDimension() ) return false;
////
////		for(unsigned int i = 0; i < lhs.GetDimension(); ++i)
////		{
////			if( lhs[i] != rhs[i] ) return false;
////		}
////
////		return true;
////	}
////    
////    template<typename DataType, typename dim, typename space>
////    Vector<DataType, space> createVectorFromPoints(const Point<DataType, dim, space>& source,
////                                                   const Point<DataType, dim, space>& dest)
////    {
////		Vector<DataType, space> result(dim::Value);
////        for(unsigned int i = 0; i < dim::Value; ++i)
////        {
////            result[i] = dest[i]-source[i];
////        }
////        return result;
////    }
////
////    template<typename DataType, typename space>
////    Point<DataType, ThreeD, space> findPointAlongVector(const Vector<DataType, space>& lhs,
////                                                        const DataType& t)
////    {
////        Point<DataType, ThreeD, space> result;
////        for(unsigned int i = 0; i < ThreeD::Value; ++i)
////        {
////            result[i] = lhs[i]*t;
////        }
////
////        return result;
////    }
////
////
////    template<typename DataType, typename space>
////    DataType L1Norm(const Vector<DataType, space>& v)
////    {
////        typedef Vector<DataType, space> VectorType;
////
////        DataType result(0);
////        for(typename VectorType::const_iterator iter = v.begin(); iter != v.end(); ++iter)
////        {
////            result += AbsTypeTraits<DataType>::abs(*iter);
////        }
////
////        return result;
////    }
////    
////    template<typename DataType, typename space>
////    DataType L2Norm(const Vector<DataType, space>& v)
////    {
////        typedef Vector<DataType, space> VectorType;
////
////        DataType result(0);
////        for(typename VectorType::const_iterator iter = v.begin(); iter != v.end(); ++iter)
////        {
////            DataType v = AbsTypeTraits<DataType>::abs(*iter);
////            result += v*v;
////        }
////        return sqrt(result);
////    }
////    
////    template<typename DataType, typename space>
////    DataType InfinityNorm(const Vector<DataType, space>& v) 
////    {
////        DataType result = AbsTypeTraits<DataType>::abs(v[0]);
////        for(unsigned int i = 1; i < v.GetDimension(); ++i)
////        {
////            result = std::max(AbsTypeTraits<DataType>::abs(v[i]), result);
////        }
////        return result;
////    }
////
////    template<typename DataType, typename space>
////    Vector<DataType, space> Negate(const Vector<DataType, space>& v) 
////    {
////        Vector<DataType, space> temp(v);
////        for(unsigned int i=0; i < temp.GetDimension(); ++i)
////        {
////            temp(i) = -temp(i);
////        }
////        return temp;
////    }
////
////
////    template<typename DataType, typename space>
////    DataType Magnitude(const Vector<DataType, space>& v) 
////    {
////        DataType result = DataType(0);
////
////        for(unsigned int i = 0; i < v.GetDimension(); ++i)
////        {
////            result += v[i]*v[i];
////        }
////        return sqrt(result);
////    }
////
////    template<typename DataType, typename space>
////    DataType Dot(const Vector<DataType, space>& lhs, 
////                 const Vector<DataType, space>& rhs) 
////    {
////        DataType result = DataType(0);
////        for(unsigned int i = 0; i < lhs.GetDimension(); ++i)
////        {
////            result += lhs[i]*rhs[i];
////        }
////
////        return result;
////    }
////
////    template<typename DataType, typename space>
////    void Normalize(Vector<DataType, space>& v)
////    {
////        DataType m = v.Magnitude();
////        if( m > DataType(0) )
////        {
////            v /= m;
////        }
////    }
////
////    template<typename DataType, typename space>
////    Vector<DataType, space> Cross(const Vector<DataType, space>& lhs,
////                                  const Vector<DataType, space>& rhs)
////    {
////        assert(lhs.GetDimension() == 3 && rhs.GetDimension() == 3);
////
////        DataType first = lhs.y()*rhs.z() - lhs.z()*rhs.y();
////        DataType second = lhs.z()*rhs.x() - lhs.x()*rhs.z();
////        DataType third = lhs.x()*rhs.y() - lhs.y()*rhs.x();
////
////        Vector<DataType, space> result(first, second, third);
////        return result;
////    }
////
////    template<typename DataType, typename space>
////    std::string AsString(const Vector<DataType, space>& v)
////    {
////        unsigned int d = v.GetRows();
////        std::string result = "(";
////        for(unsigned int i = 0; i < d; ++i)
////        {
////            result += boost::lexical_cast<std::string>(v[i]);
////            if( i < v.GetDimension()-1 )
////            {
////                result += ", ";
////            }
////        }
////        result += ")";
////        return result;
////    }
////
////
////	template<typename DataType, typename space>
////    void Add(Vector<DataType, space>& result,
////           const Vector<DataType, space>& lhs,
////           const Vector<DataType, space>& rhs)
////    {
////        DataType* r_buf = result.GetRawPtr();
////        const DataType* lhs_buf = lhs.GetRawPtr();
////        const DataType* rhs_buf = rhs.GetRawPtr();
////        for(unsigned int i = 0; i < lhs.GetDimension(); ++i)
////        {
////            r_buf[i] = lhs_buf[i] + rhs_buf[i];
////        }
////    }
////    
////    template<typename DataType, typename space>
////    void AddEqual(Vector<DataType, space>& result,
////                  const Vector<DataType, space>& rhs)
////    {
////        DataType* r_buf = result.GetRawPtr();
////        const DataType* rhs_buf = rhs.GetRawPtr();
////        for(int i = 0; i < rhs.GetDimension(); ++i)
////        {
////            //result[i] += rhs[i];
////            r_buf[i] += rhs_buf[i];
////        }
////    }
////    
////    template<typename DataType, typename space>
////    Vector<DataType, space> Add(const Vector<DataType, space>& lhs, 
////                                const Vector<DataType, space>& rhs)
////    {
////        Vector<DataType, space> result(lhs.GetDimension());
////        Add(result, lhs, rhs);
////        return result;
////    }
////    
////    template<typename ResultDataType, typename InputDataType, typename space>
////    void Subtract(Vector<ResultDataType, space>& result,
////           const Vector<InputDataType, space>& lhs,
////           const Vector<InputDataType, space>& rhs)
////    {
////        ResultDataType* r_buf = result.GetRawPtr();
////        typename boost::add_const<InputDataType>::type* lhs_buf = lhs.GetRawPtr();
////        typename boost::add_const<InputDataType>::type* rhs_buf = rhs.GetRawPtr();
////        for(int i = 0; i < lhs.GetDimension(); ++i)
////        {
////            r_buf[i] = lhs_buf[i] - rhs_buf[i];
////        }
////    }
////    
////    template<typename ResultDataType, typename InputDataType, typename space>
////    void SubtractEqual(Vector<ResultDataType, space>& result,
////           const Vector<InputDataType, space>& rhs)
////    {
////        ResultDataType* r_buf = result.GetRawPtr();
////        typename boost::add_const<InputDataType>::type* rhs_buf = rhs.GetRawPtr();
////        for(int i = 0; i < rhs.GetDimension(); ++i)
////        {
////            r_buf[i] -= rhs_buf[i];
////        }
////    }
////    
////    template<typename DataType, typename space>
////    Vector<DataType, space> Subtract(const Vector<DataType, space>& lhs,
////                                     const Vector<DataType, space>& rhs)
////    {
////        Vector<DataType, space> result(lhs.GetDimension());
////        Subtract(result, lhs, rhs);
////        return result;
////    }
////
////
////
////
////
////	template<typename ResultDataType, typename InputDataType, typename space>
////    void Divide(Vector<ResultDataType, space>& result,
////           const Vector<InputDataType, space>& lhs,
////           const double& rhs)
////    {
////        ResultDataType* r_buf = result.GetRawPtr();
////        typename boost::add_const<InputDataType>::type* lhs_buf = lhs.GetRawPtr();
////        
////        for(int i = 0; i < lhs.GetDimension(); ++i)
////        {
////            r_buf[i] = lhs_buf[i] / rhs;
////        }
////    }
////    
////    template<typename ResultDataType, typename space>
////    void DivideEqual(Vector<ResultDataType, space>& result,
////					 const double& rhs)
////    {
////        ResultDataType* r_buf = result.GetRawPtr();
////        for(unsigned int i = 0; i < result.GetDimension(); ++i)
////        {
////            r_buf[i] /= rhs;
////        }
////    }
////    
////    template<typename DataType, typename dim, typename space>
////    Vector<DataType, space>
////    Divide(const Vector<DataType, space>& lhs,
////           const double& rhs)
////    {
////        Vector<DataType, space> result(lhs.GetDimension());
////        Divide(result, lhs, rhs);
////        return result;
////    }
////
////
////	template<typename ResultDataType, typename InputDataType, typename space>
////    void Multiply(Vector<ResultDataType, space>& result,
////                  const Vector<InputDataType, space>& lhs,
////                  const double& rhs)
////    {
////        ResultDataType* r_buf = result.GetRawPtr();
////        typename boost::add_const<InputDataType>::type* lhs_buf = lhs.GetRawPtr();
////        
////        for(unsigned int i = 0; i < lhs.GetDimension(); ++i)
////        {
////            r_buf[i] = lhs_buf[i] * rhs;
////        }
////    }
////    
////    template<typename ResultDataType, typename space>
////    void MultiplyEqual(Vector<ResultDataType, space>& result,
////					   const double& rhs)
////    {
////        ResultDataType* r_buf = result.GetRawPtr();
////        for(int i = 0; i < result.GetDimension(); ++i)
////        {
////            r_buf[i] *= rhs;
////        }
////    }
////    
////    template<typename DataType, typename space>
////    Vector<DataType, space>
////    Multiply(const Vector<DataType, space>& lhs,
////             const double& rhs)
////    {
////        Vector<DataType, space> result(lhs.GetDimension());
////        Multiply(result, lhs, rhs);
////        return result;
////    }
////
////	template<typename ResultDataType, typename InputDataType, typename space>
////    void Multiply(Vector<ResultDataType, space>& result,
////				  const double& lhs,   
////			      const Vector<InputDataType, space>& rhs)
////    {
////		Multiply(result, rhs, lhs);
////    }
////        
////    template<typename DataType, typename space>
////    Vector<DataType, space>
////	Multiply(const double& lhs,
////			 const Vector<DataType, space>& rhs)
////    {
////		return Multiply(rhs, lhs);
////    }
////
////	template<typename DataType, typename space>
////	Vector<DataType, space> operator+(const Vector<DataType, space>& lhs, const Vector<DataType, space>& rhs)
////	{
////		return Add(lhs, rhs);
////	}
////
////	template<typename DataType, typename space>
////	Vector<DataType, space> operator-(const Vector<DataType, space>& lhs, const Vector<DataType, space>& rhs)
////	{
////		return Subtract(lhs, rhs);
////	}
////
////	template<typename DataType, typename space>
////	Vector<DataType, space> operator*(double lhs, const Vector<DataType, space>& rhs)
////	{
////		return Multiply(lhs, rhs);
////	}
////
////	template<typename DataType, typename space>
////	Vector<DataType, space> operator*(const Vector<DataType, space>& lhs, double rhs)
////	{
////		return Multiply(lhs, rhs);
////	}
////
////    typedef Vector<double, WorldSpace> ElVis::WorldVector;
////    typedef Vector<double, TensorSpace> ElVis::TensorVector;
////    typedef Vector<double, ReferenceSpace> ElVis::ReferenceVector;
////}
//
//#endif //HIGH_ORDER_ISOSURFACE_VECTOR_HPP
//
