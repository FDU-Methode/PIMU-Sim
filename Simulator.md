# Simulator

> **作者**：王希晨
>
> **日期**：2026-02-06

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