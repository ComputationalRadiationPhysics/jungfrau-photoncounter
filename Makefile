# Jungfrau Photoncounter Makefile
# -- using GNU Make

# APP: Application name
APP := photoncounter
# CXX: Compiler. Choose between g++, clang++, nvcc
CXX := g++
# CXXFLAGS: Compiler flags. Required: c++11, MMD
CXXFLAGS = -std=c++11 -MMD -Wall -Wextra -pedantic -g3
NVCCFLAGS := -std=c++11 --compiler-options -Wall,-Wextra,-pedantic,-MMD,-g3
ifeq ($(CXX),nvcc)
	CXXFLAGS = $(NVCCFLAGS)
endif
# LD: Linker. Usually the same as the compiler
LD := $(CXX)
# LDFLAGS: Linker flags.
LDFLAGS := -lm
# Directories for source and built files
SRC_DIR := src
OBJ_DIR := obj
# Commands for convenience
COMPILE := $(CXX) $(CXXFLAGS) -c
LINK := $(LD)

SRC := $(wildcard $(SRC_DIR)/*.cpp)
OBJ := $(addprefix $(OBJ_DIR)/, $(notdir $(SRC:.cpp=.o)))
DEP := $(addprefix $(OBJ_DIR)/, $(notdir $(SRC:.cpp=.d)))

all: $(APP)

clean:
	rm -f $(APP) $(OBJ) $(DEP)

.PHONY: all clean

$(APP): $(OBJ)
	$(LINK) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp Makefile
	$(COMPILE) $< -o $@

-include $(DEP)
