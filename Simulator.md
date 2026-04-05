# Simulator

> **作者**：王希晨
>
> **日期**：2026-02-11

## 修改记录

### 2026-01-16

创建Github远程仓库，同步本地仓库。前期修改如下：

- `perf_func_unit.hpp`添加组件：SPM、SystolicArray、Accumulator、SFU、Quant、Im2col、GEMV。PipelineController添加以上成员。

- `perf_func_unit.hpp`修改PipelineController::execute()，添加其他启动位对应的任务队列task_queue

- 修改了所有文件的头文件路径

- 连接DMA和DDR(AnalyticalDDR)，设计如下：DDR和PerfNpuCore通过sendMsg()和recvMsg()连接，DMA通过PerfNpuCore和DDR传输数据。考虑到DDR请求队列深度有限，PerfNpuCore设置输入输出FIFO outbox_和inbox_，DMA调用sendMsg()时，将msg入队outbox_，DDR调用recvMsg()时，将msg入队inbox_。PerfNpuCore每次执行tick()时，若outbox_有未发送的msg，尝试向DDR发送msg直到DDR不能接受。修改DMA->tick()：调用core->tryConsumeResp()，根据expectedMsgType在inbox_寻找需要的resp。修改DMA->execute()：根据MVIN/MVOUT和寄存器生成对应msg，调用core->sendMsg()。

> 待修改：
- 需要考虑PerfFuncUnit中是否需要SPM组件，并修改相应execute()任务队列
- 测试DMA是否正常工作(不同大小的数据传输，不同请求)

### 2026-02-06

- 实现DMA建模，后续需要进一步优化和验证(后续优化burst cycle等参数)

- 实现SA的GEMM建模，vcs测试周期数一致，后续加入CONV建模

- 优化task_queue，删除dequant模块

- 统一文件命名，`./config`文件命名`xxx_cfg.txt`，`./benchmark/tb_test`文件命名`tb_xxx.txt`，`./log`文件命名`tb_xxx_log.txt`。修改Makefile，可直接改变TOP调整仿真参数。

### 2026-02-09

- 实现MATADD，vcs测试周期数一致

### 2026-02-11

- 实现SA的CONV建模，测试一组卷积，周期数大致相近(258 / 263)

### 2026-02-15

- 实现SFU的GELU建模，分fast/slow情况测试，周期数一致

### 2026-02-16

- 实现SFU的SOFTMAX建模，vcs测试周期数一致

### 2026-03-05

- 修改各模块systolic_array_size为可配置(根据cfg)

### 2026-03-06

- 实现SFU的LAYERNORM、POOLMAX、TRANSPOSE建模，vcs测试周期数一致

- 修改CONV和vcs仿真周期数一致，优化DMA

- `perf_func_unit.cpp`拆分为各组件单独文件，位于`src/perf/func_units`路径下

### 2026-03-13

- 删除`Quant`模块，更改其他所有模块，测试结果一致

### 2026-03-16

- `Im2col`模块并入`SystolicArray`模块

- 修改`PipelineController`单独放在`PerfNpuCore`和`PerfFuncUnit`之间，修改`task_queue`，每次`tick()`推进所有处于`Busy`模块，实现模块并行

- 重新建模`SystolicArray`和`Accumulator`模块，测试结果和串行一致

- 添加硬件利用率模块`HardWareUtilizationTracker`，对各算子和`test`进行测试

### 2026-03-18

- 修改中断产生和消除方式，由模块执行完成产生中断

### 2026-03-19

- 添加`SPM`建模，添加`SystolicArray`、`Accumulator`、`DMA`、`SFU`对SPM访问(`ReadMemory`、`WriteMemory`)

- 硬件利用率模块添加`DRAM`和`SPM`带宽统计
> 目前`DRAM`带宽建模不准确，需要在ddr_freq下进行建模

### 2026-03-20

- `Accumulator`添加SRAM建模，添加`ACC_SRAM`的硬件利用率和带宽分析

### 2026-03-27

- 修改`DRAM`模型和`DMA`建模，重新运行仿真

### 2026-03-29

- 添加`InstCache`模块，删除原有`inst.hpp`，修改仿真读取指令逻辑为每周期从inst_file读取一条指令，各配置文件添加配置`inst_cache_size`(待添加)

- 添加`InstQueue`模块，实现三种指令队列`DMAInstQueue`、`SAInstQueue`、`SFUInstQueue`。各配置文件添加配置`inst_queue_depth`(待添加)

- 添加`Decoder`模块，实现`fetch`、`decode`、`dispatch`三个阶段

### 2026-03-31

- 添加`EventSyncUnit`模块，各配置文件添加配置`sram_bank_num`、`acc_bank_num`、`bias_bank_num`(待添加)

- `EventSyncUnit`中需要确定同步控制方式

- 下一步需要确定DMA分发机制，在发射和退休时调用`EventSyncUnit`，并在Core中从后向前实现整个流水线

### 2026-04-02

- 完善`EventSyncUnit`模块，修改`issue`

### 2026-04-05

- core中完成各阶段流水线连接，修复之前代码一些问题，修改`func_unit`不再依赖启动寄存器