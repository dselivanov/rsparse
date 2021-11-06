# New release

- fixed autoconf 2.71 warnings

# Test environments

- laptop OS X, R 4.0.5
- win-builder (devel)

# R CMD check results

0 errors | 0 warning | 1 notes

> File ‘rsparse/libs/rsparse.so’:
  Found ‘__ZNSt3__14cerrE’, possibly from ‘std::cerr’ (C++)
    Objects: ‘wrmf_explicit.o’, ‘wrmf_implicit.o’
    
this is spurious NOTE, we don't touch stderr.
