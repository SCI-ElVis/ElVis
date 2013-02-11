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

//#ifndef ELVIS_SETUP_OPTIX_H
//#define ELVIS_SETUP_OPTIX_H
//
//#include <vector>
//#include "ModelView.h"
//#include "OptixView.h"
//#include <optixu/optixpp.h>
//
//namespace ElVis
//{
//    /// Creates a single Optix context that can be shared among multiple model views.
//    /// It also forces all Optix related computation to finish before moving on to display
//    /// (without this, VisTrails would completely execute some paths before others).
//    class SetupOptix
//    {
//        public:
//            SetupOptix();
//            SetupOptix(const SetupOptix& rhs);
//            virtual ~SetupOptix();
//            
//            void SetModel(Model* model) { m_model = model; }
//            void AddModelView(OptixView* view) { m_modelViews.push_back(view); }
//            int NumberOfModelViews() const { return m_modelViews.size(); }
//            OptixView* GetModelView(int i) const { return m_modelViews[i]; }
//            
//            // Sets up model geometry and view specific optix information.
//            void SetupOptixContext();
//            
//        private:
//            SetupOptix& operator==(const SetupOptix& rhs);
//            
//            std::vector<OptixView*> m_modelViews;
//            Model* m_model;    
//            optixu::Context m_context;        
//    };
//}
//
//
//#endif //ELVIS_SETUP_OPTIX_H
