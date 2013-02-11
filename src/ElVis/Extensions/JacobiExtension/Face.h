#ifndef ELVIS_JACOBI_EXTENSION_ORIGINAL_NEKTAR_FACE_H
#define ELVIS_JACOBI_EXTENSION_ORIGINAL_NEKTAR_FACE_H

#include <vector>

namespace ElVis
{
    namespace JacobiExtension
    {
        // Represents an element face.
        // Faces returned by the individual elements use local edge indexing.
        // It is the responsibility of the converter to convert to a global 
        // face using global indexes.
        class Face
        {
        public:
            Face(unsigned int v0, unsigned int v1, unsigned int v2)
            {
                m_edges.push_back(v0);
                m_edges.push_back(v1);
                m_edges.push_back(v2);
            }

            Face(unsigned int v0, unsigned int v1, unsigned int v2, unsigned int v3)
            {
                m_edges.push_back(v0);
                m_edges.push_back(v1);
                m_edges.push_back(v2);
                m_edges.push_back(v3);
            }

            unsigned int NumberOfEdges() const { return m_edges.size(); }
            unsigned int EdgeId(unsigned int i) const { return m_edges[i]; }

            friend bool operator<(const Face& lhs, const Face& rhs);

        private:
            std::vector<unsigned int> m_edges;
        };

        bool operator<(const Face& lhs, const Face& rhs);
    }
}
#endif
