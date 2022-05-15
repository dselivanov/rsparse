AC_PREREQ([2.71])
AC_INIT
AC_CONFIG_SRCDIR([DESCRIPTION])

## Set R_HOME, respecting an environment variable if one is set
: ${R_HOME=$(R RHOME)}
if test -z "${R_HOME}"; then
    AC_MSG_ERROR([Could not determine R_HOME.])
fi

## Use R to set CXX and CXXFLAGS
CXX=$(${R_HOME}/bin/R CMD config CXX)
CXXFLAGS=$("${R_HOME}/bin/R" CMD config CXXFLAGS)

## configure float libs
FLOAT_LIBS=`${R_HOME}/bin/Rscript -e "float:::ldflags(static = (Sys.info()[['sysname']] == 'Darwin'))"`

#------------------------------------------------------

AC_SUBST(FLOAT_LIBS)
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
