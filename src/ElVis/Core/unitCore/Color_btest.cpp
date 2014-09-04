#include <boost/test/unit_test.hpp>

#include <ElVis/Core/Color.h>

using namespace ElVis;

//############################################################################//
BOOST_AUTO_TEST_SUITE( ElVisCore )

//----------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( Color_default_ctor )
{
  //Test the default constructor
  Color c;

  BOOST_CHECK_EQUAL( c.Red()  , 0 );
  BOOST_CHECK_EQUAL( c.Green(), 0 );
  BOOST_CHECK_EQUAL( c.Blue() , 0 );
  BOOST_CHECK_EQUAL( c.Alpha(), 0 );
}

//----------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( Color_set )
{
  //Test the set functions
  Color c;

  c.SetRed(0.8);
  c.SetGreen(0.1);
  c.SetBlue(0.2);
  c.SetAlpha(0.3);

  BOOST_CHECK_CLOSE( c.Red()  , 0.8, 5e-6 );
  BOOST_CHECK_CLOSE( c.Green(), 0.1, 5e-6 );
  BOOST_CHECK_CLOSE( c.Blue() , 0.2, 5e-6 );
  BOOST_CHECK_CLOSE( c.Alpha(), 0.3, 5e-6 );

  c.SetRed(25);
  c.SetGreen(126);
  c.SetBlue(245);
  c.SetAlpha(60);

  BOOST_CHECK_CLOSE( c.Red()  , (float) 25/255.0, 5e-6 );
  BOOST_CHECK_CLOSE( c.Green(), (float)126/255.0, 5e-6 );
  BOOST_CHECK_CLOSE( c.Blue() , (float)245/255.0, 5e-6 );
  BOOST_CHECK_CLOSE( c.Alpha(), (float) 60/255.0, 5e-6 );

  BOOST_CHECK_EQUAL( c.RedAsInt()  ,  25 );
  BOOST_CHECK_EQUAL( c.GreenAsInt(), 126 );
  BOOST_CHECK_EQUAL( c.BlueAsInt() , 245 );
  BOOST_CHECK_EQUAL( c.AlphaAsInt(),  60 );
}

//----------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( Color_ctors )
{
  //Test the set functions
  Color c1;

  c1.SetRed(0.8);
  c1.SetGreen(0.1);
  c1.SetBlue(0.2);
  c1.SetAlpha(0.3);

  Color c2(c1);

  BOOST_CHECK_CLOSE( c2.Red()  , c1.Red()  , 5e-6 );
  BOOST_CHECK_CLOSE( c2.Green(), c1.Green(), 5e-6 );
  BOOST_CHECK_CLOSE( c2.Blue() , c1.Blue() , 5e-6 );
  BOOST_CHECK_CLOSE( c2.Alpha(), c1.Alpha(), 5e-6 );

  Color c3(c1.Red(), c1.Green(), c1.Blue());

  BOOST_CHECK_CLOSE( c3.Red()  , c1.Red()  , 5e-6 );
  BOOST_CHECK_CLOSE( c3.Green(), c1.Green(), 5e-6 );
  BOOST_CHECK_CLOSE( c3.Blue() , c1.Blue() , 5e-6 );
  BOOST_CHECK_EQUAL( c3.Alpha(), 1.0f );

  Color c4((double)c1.Red(), (double)c1.Green(), (double)c1.Blue());

  BOOST_CHECK_CLOSE( c4.Red()  , c1.Red()  , 5e-6 );
  BOOST_CHECK_CLOSE( c4.Green(), c1.Green(), 5e-6 );
  BOOST_CHECK_CLOSE( c4.Blue() , c1.Blue() , 5e-6 );
  BOOST_CHECK_EQUAL( c4.Alpha(), 1.0f );

  Color c5(c1.Red(), c1.Green(), c1.Blue(), c1.Alpha());

  BOOST_CHECK_CLOSE( c5.Red()  , c1.Red()  , 5e-6 );
  BOOST_CHECK_CLOSE( c5.Green(), c1.Green(), 5e-6 );
  BOOST_CHECK_CLOSE( c5.Blue() , c1.Blue() , 5e-6 );
  BOOST_CHECK_CLOSE( c5.Alpha(), c1.Alpha(), 5e-6 );

  Color c6((double)c1.Red(), (double)c1.Green(), (double)c1.Blue(), (double)c1.Alpha());

  BOOST_CHECK_CLOSE( c6.Red()  , c1.Red()  , 5e-6 );
  BOOST_CHECK_CLOSE( c6.Green(), c1.Green(), 5e-6 );
  BOOST_CHECK_CLOSE( c6.Blue() , c1.Blue() , 5e-6 );
  BOOST_CHECK_CLOSE( c6.Alpha(), c1.Alpha(), 5e-6 );

}

//----------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( Color_operators )
{
  //Test the set functions
  Color c1, c3;

  c1.SetRed(0.8);
  c1.SetGreen(0.1);
  c1.SetBlue(0.2);
  c1.SetAlpha(0.3);

  Color c2 = c1;

  BOOST_CHECK_CLOSE( c2.Red()  , c1.Red()  , 5e-6 );
  BOOST_CHECK_CLOSE( c2.Green(), c1.Green(), 5e-6 );
  BOOST_CHECK_CLOSE( c2.Blue() , c1.Blue() , 5e-6 );
  BOOST_CHECK_CLOSE( c2.Alpha(), c1.Alpha(), 5e-6 );

  BOOST_CHECK( c2 == c1 );

  c3 = c2 + c1;

  BOOST_CHECK_CLOSE( c3.Red()  , 2*c1.Red()  , 5e-6 );
  BOOST_CHECK_CLOSE( c3.Green(), 2*c1.Green(), 5e-6 );
  BOOST_CHECK_CLOSE( c3.Blue() , 2*c1.Blue() , 5e-6 );
  BOOST_CHECK_CLOSE( c3.Alpha(), 2*c1.Alpha(), 5e-6 );

  BOOST_CHECK( c3 != c1 );

  c3 = c2 - c1;

  BOOST_CHECK_EQUAL( c3.Red()  , 0 );
  BOOST_CHECK_EQUAL( c3.Green(), 0 );
  BOOST_CHECK_EQUAL( c3.Blue() , 0 );
  BOOST_CHECK_EQUAL( c3.Alpha(), 0 );

  c3 = 2*c1;

  BOOST_CHECK_CLOSE( c3.Red()  , 2*c1.Red()  , 5e-6 );
  BOOST_CHECK_CLOSE( c3.Green(), 2*c1.Green(), 5e-6 );
  BOOST_CHECK_CLOSE( c3.Blue() , 2*c1.Blue() , 5e-6 );
  BOOST_CHECK_CLOSE( c3.Alpha(), 2*c1.Alpha(), 5e-6 );

  c3 = c1*2;

  BOOST_CHECK_CLOSE( c3.Red()  , 2*c1.Red()  , 5e-6 );
  BOOST_CHECK_CLOSE( c3.Green(), 2*c1.Green(), 5e-6 );
  BOOST_CHECK_CLOSE( c3.Blue() , 2*c1.Blue() , 5e-6 );
  BOOST_CHECK_CLOSE( c3.Alpha(), 2*c1.Alpha(), 5e-6 );

}

//############################################################################//
BOOST_AUTO_TEST_SUITE_END()
