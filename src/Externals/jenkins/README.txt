**************************************
*-=-=-=- Testing with Jenkins -=-=-=-*
**************************************

This folder contains the scripts used by Jenkins to test SANS.

The script ElVis_Commit.sh is executed when new changes on a branch following the 
pattern SCI-ElVis/*/develop is committed. If the build and all tests pass then the 
modifications are merged into the develop branch. All developers merge changes
from the develop branch into their respective SCI-ElVis/*/develop branches.

The ElVis_Nightly.sh script executes on a nightly bases and tests the develop
branch for memory errors. If all tests pass the develop branch is merged into the
apprentice branch.

Hence, while the develop branch may not have passed all memory and regression test,
the master branch will only contain code that has passed all tests.

A ElVis git repository can be cloned and configured as a developer repository using the script

ElVis_developer.sh



**************************************
*-=-=-=- Testing on Cleopatra -=-=-=-*
**************************************

If the Jenkins tests fail on reynolds or macys, the build configurations used by Jenkins 
can be created automatically by cloning the ElVis git repository on cleopatra
and running the script:

src/Externals/jenkins/configure_jenkins.sh

This script will create a 'build' directory and all build configurations inside of it
that are used by Jenkins testing. A makefile is also placed in the build directory that
can be used to run tests with all the compiler configurations.



**********************************************
*-=-=-=- Updating the testing scripts -=-=-=-*
**********************************************

The script jenkins_env.sh contains the necessary environment variables
to run the tests on reynolds or macys. This file should be modified if additional 
compiler versions other than the standard gnu, clang and intel compilers 
are included in the testing. The list of compiler names then need to be
updated in configure_jenkins.sh and the Jenkins configurations through 
the Jenkins website.
