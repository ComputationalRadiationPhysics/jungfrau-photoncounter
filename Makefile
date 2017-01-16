# Jungfrau Photoncounter Makefile
# -- using GNU Make

# APP: Application name
APP := photoncounter
# CXX: Compiler. Choose between g++, clang++, nvcc
CXX := nvcc
# CXXFLAGS: Compiler flags. Required: c++11, MMD
CXXFLAGS := -std=c++11 -MMD -Wall -Wextra -pedantic -g3
NVCCFLAGS := -std=c++11 --compiler-options -Wall,-Wextra,-MMD,-g3
# LD: Linker. Usually the same as the compiler
LD := $(CXX)
# LDFLAGS: Linker flags.
LDFLAGS := -lm
# Directories for source and built files
SRC_DIR := src
OBJ_DIR := obj

SRC := $(wildcard $(SRC_DIR)/*.cpp)
OBJ := $(addprefix $(OBJ_DIR)/, $(notdir $(SRC:.cpp=.o)))
DEP := $(addprefix $(OBJ_DIR)/, $(notdir $(SRC:.cpp=.d)))

CUDA_SRC := $(wildcard $(SRC_DIR)/*.cu)
CUDA_OBJ := $(addprefix $(OBJ_DIR)/, $(notdir $(CUDA_SRC:.cu=.cu.o)))
CUDA_DEP := $(addprefix $(OBJ_DIR)/, $(notdir $(CUDA_SRC:.cu=.cu.d)))

ifeq ("$(strip $(CXX))","$(strip nvcc)")
	CXXFLAGS_ALL := $(NVCCFLAGS)
	SRC := $(SRC) $(CUDA_SRC)
	OBJ := $(OBJ) $(CUDA_OBJ)
	DEP := $(DEP) $(CUDA_DEP)
else
	CXXFLAGS_ALL := $(CXXFLAGS)
endif

# Commands for convenience
COMPILE = $(CXX) $(CXXFLAGS_ALL) -c
LINK := $(LD)

all: $(APP)

debug:
	@echo $(CXXFLAGS_ALL)
	@echo $(CUDA_SRC)
	@echo $(CUDA_OBJ)
	@echo $(CUDA_DEP)

clean:
	rm -f $(APP) $(OBJ) $(DEP)
	rm -f *.d
	rm -f obj/*.d

.PHONY: all clean debug

$(APP): $(OBJ)
	$(LINK) $^ -o $@

$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu Makefile
	$(COMPILE) $< -o $@
	mv *.d obj/

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp Makefile
	$(COMPILE) $< -o $@

-include $(DEP)
