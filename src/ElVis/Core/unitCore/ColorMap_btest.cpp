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

#include <boost/test/unit_test.hpp>
#include <ElVis/Core/ColorMap.h>
#include <ElVis/Core/Color.h>

using namespace ElVis;

BOOST_AUTO_TEST_CASE(TestMinMax)
{
  ColorMap map;
  BOOST_CHECK_EQUAL(map.GetMin(), 0.0f);
  BOOST_CHECK_EQUAL(map.GetMax(), 1.0f);

  float newMin = -6.7f;
  map.SetMin(newMin);
  BOOST_CHECK_EQUAL(map.GetMin(), newMin);

  float invalidNewMin = 8.9f;
  map.SetMin(invalidNewMin);
  BOOST_CHECK_EQUAL(map.GetMin(), newMin);

  float newMax = 9.8f;
  map.SetMax(newMax);
  BOOST_CHECK_EQUAL(map.GetMax(), newMax);

  float invalidNewMax = -99.1;
  map.SetMax(invalidNewMax);
  BOOST_CHECK_EQUAL(map.GetMax(), newMax);
}

BOOST_AUTO_TEST_CASE(TestSetBreakpoint)
{
  ColorMap map;
  Color red(255, 0, 0);
  Color green(0, 255, 0);
  Color blue(0, 0, 255);

  map.SetBreakpoint(-.1, red);
  map.SetBreakpoint(1.1, blue);
  BOOST_CHECK_EQUAL(map.GetBreakpoints().size(), 0);

  float redIdx = .2f;
  map.SetBreakpoint(redIdx, red);
  BOOST_CHECK_EQUAL(map.GetBreakpoints().size(), 1);
  BOOST_CHECK_EQUAL(map.GetBreakpoints().find(redIdx)->second.Col, red);

  float greenIdx = .3f;
  map.SetBreakpoint(greenIdx, green);
  BOOST_CHECK_EQUAL(map.GetBreakpoints().size(), 2);
  BOOST_CHECK_EQUAL(map.GetBreakpoints().find(greenIdx)->second.Col, green);

  map.SetBreakpoint(redIdx, blue);
  BOOST_CHECK_EQUAL(map.GetBreakpoints().size(), 2);
  BOOST_CHECK_EQUAL(map.GetBreakpoints().find(redIdx)->second.Col, blue);
}

BOOST_AUTO_TEST_CASE(TestClamping)
{
  ColorMap map;
  Color red(255, 0, 0);
  Color green(0, 255, 0);
  Color blue(0, 0, 255);

  map.SetBreakpoint(.1, red);
  map.SetBreakpoint(.2, blue);
  BOOST_CHECK_EQUAL(map.Sample(.05), red);
  BOOST_CHECK_EQUAL(map.Sample(.25), blue);
}

BOOST_AUTO_TEST_CASE(TestRemoveBreakpoint)
{
  ColorMap map;
  Color red(255, 0, 0);
  Color green(0, 255, 0);
  Color blue(0, 0, 255);
  float redIdx = .2f;

  auto redIter = map.SetBreakpoint(redIdx, red);
  BOOST_CHECK_EQUAL(map.GetBreakpoints().size(), 1);

  float greenIdx = .3f;
  map.SetBreakpoint(greenIdx, green);
  BOOST_CHECK_EQUAL(map.GetBreakpoints().size(), 2);

  map.RemoveBreakpoint(redIter);
  BOOST_CHECK_EQUAL(map.GetBreakpoints().size(), 1);
  BOOST_CHECK_EQUAL(map.GetBreakpoints().find(greenIdx)->second.Col, green);
  BOOST_CHECK(map.GetBreakpoints().find(redIdx) == map.GetBreakpoints().end());
}
