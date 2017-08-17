
# Jungfrau Photoncounter Makefile
# -- using GNU Make

# APP: Application name
APP := photoncounter

# CXX: Compiler. Choose between g++, clang++, nvcc
CXX := nvcc
ARCH := -arch=sm_35
CXXFLAGS := -std=c++11 --compiler-options -Wall,-Wextra -O3 $(ARCH)
LDFLAGS := $(ARCH)



# Directories for source and built files
SRC_DIR := src
OBJ_DIR := obj

SRC := $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(SRC_DIR)/cuda/*.cu)
OBJ := $(addsuffix .o, $(basename $(addprefix $(OBJ_DIR)/, $(notdir $(SRC)))))
DEP := $(OBJ:.o=.d)

all: $(APP)

debug:
	@echo $(OBJ)
	@echo $(DEP)

clean:
	rm -f $(APP) $(OBJ) $(DEP)
	rm -f obj/*.d

.PHONY: all clean debug


$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp Makefile
	$(shell $(CXX) $(CXXFLAGS) $< -M -odir $(OBJ_DIR) > $(patsubst %.o,%.d, $@))
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu Makefile
	$(shell $(CXX) $(CXXFLAGS) $< -M -odir $(OBJ_DIR) > $(patsubst %.o,%.d, $@))
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/cuda/%.cu Makefile
	$(shell $(CXX) $(CXXFLAGS) $< -M -odir $(OBJ_DIR) > $(patsubst %.o,%.d, $@))
	$(CXX) $(CXXFLAGS) -c $< -o $@


$(APP): $(OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

.PHONY: all clean

-include $(DEP)
