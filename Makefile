########################################################
# NPU Simulator Makefile
########################################################

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -g
INCLUDES = -I./include -I./external/DRAMsim3/src

# 库路径和链接库
# LIBS = -L./external/DRAMsim3 -ldramsim3 -lpthread
LIBS = -L./external/DRAMsim3 -Wl,-rpath,'$$ORIGIN/../external/DRAMsim3' -ldramsim3 -lpthread

SRC_DIR = src
BUILD_DIR = build

SRCS = $(shell find $(SRC_DIR) -name "*.cpp")
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

TARGET = $(BUILD_DIR)/npu_sim

########################################################
# 构建规则
########################################################

# 默认目标：编译并运行
all: build run

# 只编译
build: $(TARGET)
	@echo "Build complete! Generated npu simulator: $(TARGET)"
	@echo "Run example programs with 'make run'"
	@echo "Or run with './$(TARGET) <config_file>'"

# 编译规则
$(TARGET): $(OBJS) $(DRAMSIM3_LIB)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# 只运行（需要先编译）
run: $(TARGET)
	@echo "Running NPU Simulator..."
	./$(TARGET) ./config/my_test.txt

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all build run clean
