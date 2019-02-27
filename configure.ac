AC_PREREQ([2.69])
AC_INIT(DESCRIPTION)


dnl Determine R_HOME.
: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  AC_MSG_ERROR([could not determine R_HOME])
fi

FLOAT_LIBS=`${R_HOME}/bin/Rscript -e "float:::ldflags()"`

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
AC_SUBST(FLOAT_LIBS)
AC_OUTPUT(src/Makevars)
