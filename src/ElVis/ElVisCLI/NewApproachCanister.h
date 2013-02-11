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

#ifndef ELVIS_CLI_NEW_APPROACH_CANISTER
#define ELVIS_CLI_NEW_APPROACH_CANISTER

// Designed for the synthetic datasets on (0, 0, 0) - (1, 1, 1)
void NewApproachCanisterMeshRendering(int argc, char** argv);
int ColorMapBulletNewApproachVolumeSampling(int argc, char** argv, boost::shared_ptr<ElVis::Model> model, unsigned int width, unsigned int height, const std::string& outFilePath);
int GenericCLIInterface(int argc, char** argv, boost::shared_ptr<ElVis::Scene> scene, boost::shared_ptr<ElVis::Model> model,
                        boost::shared_ptr<ElVis::ColorMap> colorMap,
                        unsigned int width, unsigned int height, const std::string& outFilePath, const ElVis::Camera& c);


#endif
