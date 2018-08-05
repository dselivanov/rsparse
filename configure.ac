AC_PREREQ([2.69])
AC_INIT(DESCRIPTION)


dnl Determine R_HOME.
: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  AC_MSG_ERROR([could not determine R_HOME])
fi

# Obtain from R.
R_SCMD="${R_HOME}/bin/R CMD config"
BLAS_LIBS=`${R_SCMD} BLAS_LIBS`
LAPACK_LIBS=`${R_SCMD} LAPACK_LIBS`

# Check tools.
AC_CHECK_PROG([echo_ok], [echo], [yes], [no], ,)
AC_CHECK_PROG([grep_ok], [grep], [yes], [no], ,)


#------------------------------------------------------
# borrowed from https://github.com/wrathematics/float
#------------------------------------------------------
### Check for single precision BLAS/LAPACK symbols. These can be present/absent
### in almost any imaginable combination. For example, OpenBLAS (and others?)
### ship LAPACK symbols in the blas.[so|dylib|dll] file.

# Check Rblas.
if test "X${echo_ok}" = "Xyes" -a "X${grep_ok}" = "Xyes"; then
  useRblas=`echo ${BLAS_LIBS} | grep "\-lRblas"`
  if test "X${useRblas}" = "X"; then
    AC_CHECK_LIB(blas, [sgemm_], [HAVESBLAS=yes], [HAVESBLAS=no])
    AC_CHECK_LIB(blas, [sgetrf_], [HAVESLAPACK=yes], [HAVESLAPACK=no])
  else
    LDFLAGS_OLD="${LDFLAGS}"
    LDFLAGS="${LDFLAGS} ${BLAS_LIBS}"
    AC_CHECK_LIB(Rblas, [sgemm_], [HAVESBLAS=yes], [HAVESBLAS=no])
    AC_CHECK_LIB(Rblas, [sgetrf_], [HAVESLAPACK=yes], [HAVESLAPACK=no])
    LDFLAGS="${LDFLAGS_OLD}"
  fi
else
  HAVESBLAS="no"
fi


# Check for single precision LAPACK symbols.
if test "X${HAVESLAPACK}" = "X" -o "X${HAVESLAPACK}" = "Xno"; then
  if test "X${echo_ok}" = "Xyes" -a "X${grep_ok}" = "Xyes"; then
    useRlapack=`echo ${LAPACK_LIBS} | grep "\-lRlapack"`
    if test "X${useRlapack}" = "X"; then
      AC_CHECK_LIB(lapack, [sgetrf_], [HAVESLAPACK=yes], [HAVESLAPACK=no])
    else
      LDFLAGS_OLD="${LDFLAGS}"
      LDFLAGS="${LDFLAGS} ${LAPACK_LIBS}"
      AC_CHECK_LIB(Rlapack, [sgetrf_], [HAVESLAPACK=yes], [HAVESLAPACK=no])
      LDFLAGS="${LDFLAGS_OLD}"
    fi
  else
    HAVESLAPACK="no"
  fi
fi


# Set additional object files and messages
ARMA_BLAS=""
SINGLE_PRECISION_BLAS_AVAILABLE="FALSE"

if test "X${HAVESBLAS}" = "Xno"; then
  echo "* NOTE: defining ARMA_DONT_USE_BLAS"
  echo "* WARNING: using Armadillo reference BLAS implementation"
  ARMA_BLAS="${ARMA_BLAS} -DARMA_DONT_USE_BLAS"
else
  echo "* NOTE: using system-wide BLAS"
  SINGLE_PRECISION_BLAS_AVAILABLE="TRUE"
fi

# Set additional object files and messages
SINGLE_PRECISION_LAPACK_AVAILABLE="FALSE"
DSINGLE_PRECISION_LAPACK_AVAILABLE=""

if test "X${HAVESLAPACK}" = "Xyes"; then
  echo "* NOTE: using system-wide LAPACK"
  SINGLE_PRECISION_LAPACK_AVAILABLE="TRUE"
  DSINGLE_PRECISION_LAPACK_AVAILABLE="-DSINGLE_PRECISION_LAPACK_AVAILABLE"
else
  echo "* WARNING: single precision LAPACK is not available"
fi

#------------------------------------------------------
# borrowed from https://github.com/zhangh12/ARTP2/blob/88597ea2c81119821feaed218142372178aaf430/configure.ac
#------------------------------------------------------
CXX1X=`"${R_HOME}/bin/R" CMD config CXX1X`
CXX1XSTD=`"${R_HOME}/bin/R" CMD config CXX1XSTD`
CXX="${CXX1X} ${CXX1XSTD}"
CXXFLAGS=`"${R_HOME}/bin/R" CMD config CXX1XFLAGS`

AC_LANG(C++)
dnl this the meat of R's m4/openmp.m4
  OPENMP_[]_AC_LANG_PREFIX[]FLAGS=
  AC_ARG_ENABLE([openmp],
    [AS_HELP_STRING([--disable-openmp], [do not use OpenMP])])
  if test "$enable_openmp" != no; then
    AC_CACHE_CHECK([for $[]_AC_CC[] option to support OpenMP],
      [ac_cv_prog_[]_AC_LANG_ABBREV[]_openmp],
      [AC_LINK_IFELSE([_AC_LANG_OPENMP],
	 [ac_cv_prog_[]_AC_LANG_ABBREV[]_openmp='none needed'],
	 [ac_cv_prog_[]_AC_LANG_ABBREV[]_openmp='unsupported'
	  for ac_option in -fopenmp -xopenmp -qopenmp \
                           -openmp -mp -omp -qsmp=omp -homp \
			   -fopenmp=libomp \
                           -Popenmp --openmp; do
	    ac_save_[]_AC_LANG_PREFIX[]FLAGS=$[]_AC_LANG_PREFIX[]FLAGS
	    _AC_LANG_PREFIX[]FLAGS="$[]_AC_LANG_PREFIX[]FLAGS $ac_option"
	    AC_LINK_IFELSE([_AC_LANG_OPENMP],
	      [ac_cv_prog_[]_AC_LANG_ABBREV[]_openmp=$ac_option])
	    _AC_LANG_PREFIX[]FLAGS=$ac_save_[]_AC_LANG_PREFIX[]FLAGS
	    if test "$ac_cv_prog_[]_AC_LANG_ABBREV[]_openmp" != unsupported; then
	      break
	    fi
	  done])])
    case $ac_cv_prog_[]_AC_LANG_ABBREV[]_openmp in #(
      "none needed" | unsupported)
	;; #(
      *)
	OPENMP_[]_AC_LANG_PREFIX[]FLAGS=$ac_cv_prog_[]_AC_LANG_ABBREV[]_openmp ;;
    esac
  fi
#------------------------------------------------------

AC_SUBST(OPENMP_CXXFLAGS)
AC_SUBST(ARMA_BLAS)
AC_SUBST(DSINGLE_PRECISION_LAPACK_AVAILABLE)
AC_OUTPUT(src/Makevars)

AC_SUBST(SINGLE_PRECISION_LAPACK_AVAILABLE)
AC_SUBST(SINGLE_PRECISION_BLAS_AVAILABLE)
AC_OUTPUT([R/configure.R])

