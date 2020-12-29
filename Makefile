clang_format=`which clang-format`
format: $(shell find . -name *.hpp) $(shell find . -name *.cpp)
	@${clang_format} -i $?
