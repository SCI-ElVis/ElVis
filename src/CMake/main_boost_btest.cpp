#define BOOST_TEST_MODULE ElVis_unit_test
#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>
#include <boost/test/debug.hpp>


struct GlobalElVisFixture
{
  GlobalElVisFixture()
  {
    boost::debug::detect_memory_leaks(false);
  }

  ~GlobalElVisFixture()
  {
  }
};

BOOST_GLOBAL_FIXTURE( GlobalElVisFixture );
