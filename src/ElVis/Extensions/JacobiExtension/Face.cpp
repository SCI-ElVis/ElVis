
#include <ElVis/Extensions/JacobiExtension/Face.h>
#include <algorithm>

namespace ElVis
{
    namespace JacobiExtension
    {

        bool operator<(const Face& lhs, const Face& rhs)
        {
            if( lhs.NumberOfEdges() != rhs.NumberOfEdges() )
            {
                return lhs.NumberOfEdges() < rhs.NumberOfEdges();
            }

            std::vector<unsigned int> lhsValues = lhs.m_edges;
            std::vector<unsigned int> rhsValues = rhs.m_edges;

            std::sort(lhsValues.begin(), lhsValues.end());
            std::sort(rhsValues.begin(), rhsValues.end());

            for(unsigned int i = 0; i < lhsValues.size(); ++i)
            {
                if( lhsValues[i] != rhsValues[i] )
                {
                    return lhsValues[i] < rhsValues[i];
                }
            }
            return false;
        }
    }
}
