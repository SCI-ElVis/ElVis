#include <boost/test/unit_test.hpp>

//############################################################################//
BOOST_AUTO_TEST_SUITE( Example )

//----------------------------------------------------------------------------//
BOOST_AUTO_TEST_CASE( Example_test_1 )
{
  bool check = true;
  BOOST_CHECK( check == true );
}

//############################################################################//
BOOST_AUTO_TEST_SUITE_END()
