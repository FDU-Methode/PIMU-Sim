# Compiler configuration
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -g
CPPFLAGS = -I./include
LDLIBS = -lpthread

# Directories
SRC_DIR = src
BUILD_DIR = build

# Sources
ALL_SRCS := $(sort $(shell find $(SRC_DIR) -name '*.cpp'))

CLI_SRC := $(SRC_DIR)/main.cpp

CORE_SRCS := $(filter-out \
	$(CLI_SRC) \
	$(SRC_DIR)/func_sim.cpp, \
	$(ALL_SRCS))

# Separate object directories
CORE_OBJS := $(patsubst \
	$(SRC_DIR)/%.cpp, \
	$(BUILD_DIR)/core/%.o, \
	$(CORE_SRCS))

CLI_OBJ := $(BUILD_DIR)/cli/main.o

# Automatically generated header dependencies
DEPS := $(CORE_OBJS:.o=.d) $(CLI_OBJ:.o=.d)

# Output targets
CORE_LIB := $(BUILD_DIR)/libpimu_sim.so
TARGET := $(BUILD_DIR)/npu_sim

# CLI run configuration
TOP ?= smolvlm2_decode
TOP_CFG := ./config/$(TOP)_cfg.txt

.PHONY: all build run clean

# Default: build only
all: build

build: $(CORE_LIB) $(TARGET)

# Build the shared simulator library
$(CORE_LIB): $(CORE_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) \
		-shared \
		-Wl,-z,defs \
		-Wl,-soname,libpimu_sim.so \
		-o $@ $(CORE_OBJS) $(LDLIBS)

# Build the CLI and link it against the simulator library
$(TARGET): $(CLI_OBJ) $(CORE_LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) \
		-o $@ $(CLI_OBJ) \
		-L$(BUILD_DIR) -lpimu_sim \
		-Wl,-rpath,'$$ORIGIN' \
		$(LDLIBS)

# Compile simulator sources as position-independent code
$(BUILD_DIR)/core/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
		-fPIC -MMD -MP \
		-c $< -o $@

# Compile the CLI separately
$(CLI_OBJ): $(CLI_SRC)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
		-MMD -MP \
		-c $< -o $@

run: $(TARGET)
	./$(TARGET) $(TOP_CFG)

clean:
	rm -rf $(BUILD_DIR)

-include $(DEPS)