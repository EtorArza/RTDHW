include Makefile.conf




ifeq (${DEVMODE}, true)
	OPT=-O0 
	OPENMP=-fopenmp
	MISC_FLAGS=  
	NVCC_FLAGS=-G -g 
	UBSAN=-fsanitize=undefined
else
	OPT=-O3
	OPENMP=-fopenmp
	MISC_FLAGS=-DNDEBUG
	UBSAN=

endif

OTHER_SOURCES=$(shell find src/ -name "*.cpp")
INCLUDES=$(patsubst %,-I%,$(shell find src/ -type d))
NEW_OBJECTS=${OTHER_SOURCES:src/%.cpp=obj/cpp/%.o}
LIBS=-lgomp
DEPENDS=${OBJECTS:%.o=%.d}


CC_FLAGS=-Wall ${DEFINES} ${MISC_FLAGS} ${PROFILE} ${INCLUDES} ${OPENMP} ${OPT} ${UBSAN} -c -g -gdwarf-3  -Wextra


.PHONY: clean cleanall


main.out: ${NEW_OBJECTS}
	g++ ${OBJECTS} ${NEW_OBJECTS} ${LIBS} ${UBSAN} -o $@


clean:
	rm obj/ -r -f
	rm main.out -f



obj/cpp/%.o: src/%.cpp
	@mkdir -p $(shell dirname $@)
	g++ ${CC_FLAGS} -std=c++11 -MMD $< -o $@





-include ${DEPENDS}
