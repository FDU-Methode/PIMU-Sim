########################################################
# NPU Simulator Makefile
########################################################

# Compiler Config
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -g

# Include Directories
INCLUDES = -I./include -I./external/DRAMsim3/src

# Library Paths & Linked Libraries
# LIBS = -L./external/DRAMsim3 -ldramsim3 -lpthread
LIBS = -L./external/DRAMsim3 -Wl,-rpath,'$$ORIGIN/../external/DRAMsim3' -ldramsim3 -lpthread

# Source & Build Directories
SRC_DIR = src
BUILD_DIR = build

# Top Selection
TOP = smolvlm2_decode
TOP_CFG := ./config/$(TOP)_cfg.txt
LOG_FILE := ./log/tb_$(TOP)_log.txt

# Source Files & Object Files
SRCS = $(shell find $(SRC_DIR) -name "*.cpp")
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

# Final Excutable
TARGET = $(BUILD_DIR)/npu_sim

########################################################
# Build Rules
########################################################

# Default Target: Build & Run
all: build run

# Build Only
build: $(TARGET)
	@echo "Build complete! Generated npu simulator: $(TARGET)"
	@echo "Run example programs with 'make run'"
	@echo "Or run with './$(TARGET) <config_file>'"

# Compile Rules
$(TARGET): $(OBJS) $(DRAMSIM3_LIB)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Run Only (Requires Build First)
run: $(TARGET)
	@echo "Running NPU Simulator... (TOP=$(TOP), CFG=$(TOP_CFG))"
	./$(TARGET) $(TOP_CFG)

# Clean Build
clean:
	rm -rf $(BUILD_DIR)
	@if [ -f "$(LOG_FILE)" ]; then \
		: > "$(LOG_FILE)"; \
	fi

.PHONY: all build run clean
