# NPU Simulator

NPU Simulator是一个轻量级的NPU模拟器，支持 ISA 级执行、功能仿真与周期精确仿真的通用仿真平台。
本项目支持单指令测试、模型级评估(TODO)和面向各类神经网络的 NPU 性能探索。

---

## ✨ 特点

- **指令级执行**：支持基于 ISA 的指令级执行，可以更真实模拟 NPU 的详细行为。
- **通用仿真框架**：支持各组件（core, DRAM）的功能仿真与周期精准仿真的混合仿真。
- **内存模拟**：集成 DRAMSim3 的周期精确 DRAM 模型，能够模拟外存访问延迟、带宽限制；同时包含 DDR 分析模型，用于在不关心详细访存行为时加速仿真。

## 📁 项目结构

```text
perf_model/
├── src/                      # Main C++ sources
├── include/                  # Header files
├── config/                   # Configuration files
├── benchmarks/               # Instruction test binaries
│   └── single_tests/
├── log/                      # Simulator logs
│
├── external/
│   └── DRAMsim3/             # Memory simulator dependency
│
├── tests/
│   └── test_single_inst.py   # Single instruction test script
│
├── build/                    # Build output
├── Makefile
├── README.md
└── Env.cfg                   # Environment configuration file
```

## 🛠 环境依赖

本项目使用 GNU Makefile 构建，依赖项极少：

- g++ (C++17)
- DRAMSim3 (已包含在 `external/DRAMsim3` 中，无需额外安装)

## 构建方法

新启动terminal后需设置环境变量：

```bash
source Env.cfg
```

### 编译

```bash
make build
```

生成的模拟器位于:
build/npu_sim

## 运行默认测试

```bash
make run
```

这会执行：

```bash
./build/npu_sim ./config/dma_cfg.txt
```

输出log文件位于:

```bash
./log/example_log.txt
```

### 🧹 清理构建文件

```bash
make clean
```

⚠️ 在运行测试前，请先确保模拟器已成功编译。

```
当前实现了performance model的基本框架，仅建模了dma模块进行测试，且暂未联通DRAM。下一步将要对其余模块进行建模，测试跑通完整指令行为，并联通dma和DRAM。
```
