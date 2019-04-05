AC_PREREQ([2.69])
AC_INIT(DESCRIPTION)


dnl Determine R_HOME.
: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  AC_MSG_ERROR([could not determine R_HOME])
fi

FLOAT_LIBS=`${R_HOME}/bin/Rscript -e "float:::ldflags()"`

#------------------------------------------------------
# borrowed from https://github.com/RcppCore/RcppArmadillo/blob/c2fde536fe9b70238a95c2e2e413bca9681d23d2/configure.ac#L1
#------------------------------------------------------
## default to not even thinking about OpenMP as Armadillo wants a pragma
## variant available if and only if C++11 is used with g++ 5.4 or newer
can_use_openmp=""

## Check the C++ compiler using the CXX value set
AC_PROG_CXX
## If it is g++, we have GXX set so let's examine it
if test "${GXX}" = yes; then
    AC_MSG_CHECKING([whether g++ version is sufficient])
    gxx_version=$(${CXX} -v 2>&1 | awk '/^.*g.. version/ {print $3}')
    case ${gxx_version} in
        1.*|2.*|3.*|4.0.*|4.1.*|4.2.*|4.3.*|4.4.*|4.5.*|4.6.*|4.7.0|4.7.1)
             AC_MSG_RESULT([no])
             AC_MSG_WARN([Only g++ version 4.7.2 or greater can be used with RcppArmadillo.])
             AC_MSG_ERROR([Please use a different compiler.])
        ;;
        4.7.*|4.8.*|4.9.*|5.0*|5.1*|5.2*|5.3*)
             AC_MSG_RESULT([yes, but without OpenMP as version ${gxx_version} (Armadillo constraint)])
             ## we know this one is bad
             can_use_openmp="no"
        ;;
        5.4*|5.5*|5.6*|5.7*|5.8*|5.9*|6.*|7.*|8.*|9.*|10.*)
             AC_MSG_RESULT([yes, with OpenMP as version ${gxx_version}])
             ## we know this one is good, yay
             can_use_openmp="yes"
        ;;
        *)
             AC_MSG_RESULT([almost])
             AC_MSG_WARN([Compiler self-identifies as being compliant with GNUC extensions but is not g++.])
             ## we know nothing, so no
             can_use_openmp="no"
        ;;
    esac
fi

## Check for Apple LLVM

AC_MSG_CHECKING([for macOS])
RSysinfoName=$("${R_HOME}/bin/Rscript" --vanilla -e 'cat(Sys.info()[["sysname"]])')

if test x"${RSysinfoName}" == x"Darwin"; then
    AC_MSG_RESULT([found])
    AC_MSG_CHECKING([for macOS Apple compiler])

    apple_compiler=$($CXX --version 2>&1 | grep -i -c -e 'apple llvm')

    if test x"${apple_compiler}" == x"1"; then
        AC_MSG_RESULT([found])
        AC_MSG_WARN([OpenMP unavailable and turned off.])
        can_use_openmp="no"
    else
        AC_MSG_RESULT([not found])
        AC_MSG_CHECKING([for clang compiler])
        clang_compiler=$($CXX --version 2>&1 | grep -i -c -e 'clang ')

        if test x"${clang_compiler}" == x"1"; then
            AC_MSG_RESULT([found])
            AC_MSG_CHECKING([for OpenMP compatible version of clang])
            clang_version=$(${CXX} -v 2>&1 | awk '/^.*clang version/ {print $3}')

            case ${clang_version} in
                4.*|5.*|6.*|7.*|8.*)
                    AC_MSG_RESULT([found and suitable])
                    can_use_openmp="yes"
                ;;
                *)
                    AC_MSG_RESULT([not found])
                    AC_MSG_WARN([OpenMP unavailable and turned off.])
                    can_use_openmp="no"
                ;;
            esac
        else
            AC_MSG_RESULT([not found])
            AC_MSG_WARN([unsupported macOS build detected; if anything breaks, you keep the pieces.])
        fi
    fi
fi

## Default the OpenMP flag to the empty string.
## If and only if OpenMP is found, expand to $(SHLIB_OPENMP_CXXFLAGS)
openmp_flag=""

if test x"${can_use_openmp}" == x"yes"; then
    AC_MSG_CHECKING([for OpenMP])
    ## if R has -fopenmp we should be good
    allldflags=$(${R_HOME}/bin/R CMD config --ldflags)
    hasOpenMP=$(echo ${allldflags} | grep -- -fopenmp)
    if test x"${hasOpenMP}" == x""; then
        AC_MSG_RESULT([missing])
    else
        AC_MSG_RESULT([found and suitable])
        openmp_flag='$(SHLIB_OPENMP_CXXFLAGS)'
    fi
fi

#------------------------------------------------------

AC_SUBST(OPENMP_CXXFLAGS)
AC_SUBST([OPENMP_FLAG], ["${openmp_flag}"])
AC_SUBST(FLOAT_LIBS)
AC_OUTPUT(src/Makevars)
