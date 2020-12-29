clang_format=`which clang-format`

format: $(shell find . -name *.hpp) $(shell find . -type f \( -iname "*.cpp" ! -iname "RcppExports.cpp" \))
	@${clang_format} -i $?
