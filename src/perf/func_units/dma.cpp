#include <algorithm>
#include <cassert>
#include "../../../include/perf/perf_func_unit.hpp"
#include "../../../include/perf/perf_sim.hpp"
#include "../../../include/perf/pipeline_controller.hpp"

namespace {

int build_dma_spm_plan(const Config& cfg,
                       const Message& req,
                       std::vector<int>& row_beat_plan,
                       std::vector<int>& burst_beat_plan) {
    row_beat_plan.clear();
    burst_beat_plan.clear();

    const int row_num = req.row_num + 1;
    const int col_num = req.col_num + 1;
    const int axi_beat_bytes = cfg.axi_data_width / 8;
    const uint32_t dram_base_addr = (req.type == MsgType::ST_REQ) ? req.dst_addr : req.src_addr;

    int transfer_cycles = 0;
    int total_burst_count = 0;

    for (int i = 0; i < row_num; ++i) {
        uint32_t addr = dram_base_addr + i * req.dram_stride * req.data_size;
        int offset = addr % axi_beat_bytes;
        int beat_count = (offset + (axi_beat_bytes - 1) + col_num * req.data_size) / axi_beat_bytes;

        row_beat_plan.push_back(beat_count);

        int remain_beat_count = beat_count;
        uint32_t addr_aligned = addr & ~(axi_beat_bytes - 1);

        while (remain_beat_count > 0) {
            int beats_to_4k = (4096 - (addr_aligned & 0xFFF)) / axi_beat_bytes;
            int burst_beat = std::min(std::min(remain_beat_count, beats_to_4k), 256);
            burst_beat_plan.push_back(burst_beat);
            remain_beat_count -= burst_beat;
            addr_aligned += burst_beat * axi_beat_bytes;
            total_burst_count += 1;
        }

        transfer_cycles += beat_count + total_burst_count * 0; // placeholder, keep local logic simple
    }

    const int base_latency = cfg.tRCD + cfg.tCAS + cfg.tRP;
    const int setup_cycles = (req.type == MsgType::LD_REQ) ? 1 : 2;
    const int write_resp_drain_cycles = (req.type == MsgType::ST_REQ) ? total_burst_count : 0;
    const int core_cycles = base_latency + transfer_cycles + total_burst_count * cfg.tBURST + setup_cycles + write_resp_drain_cycles;
    const int ddr_total_cycles = static_cast<int>((core_cycles + req.quant) * cfg.npu_freq / cfg.ddr_freq);

    return std::max(ddr_total_cycles, 1);
}

}  // namespace

DMA::DMA(PerfSim* sim, PipelineController* controller, Logger& logger, RegFile& regfile, const Config& cfg) :
    PerfFunctionalUnit(sim, controller, logger, regfile, cfg) {
    UnitName = "DMA";
}

void DMA::tick(){
    if(!isBusy()) return;

    Scratchpad* spm = controller->getSpm();
    assert(spm && "DMA requires a valid Scratchpad");

    if(current_burst_idx < burst_beat_plan.size()) {
        uint32_t spm_addr = spm_base_addr
                          + static_cast<uint32_t>(current_row_idx) * static_cast<uint32_t>(spm_row_stride_bytes)
                          + static_cast<uint32_t>(current_row_beat_idx) * static_cast<uint32_t>(axi_beat_bytes);

        bool access_ok = false;
        if(mvin_start) {
            access_ok = spm->WriteMemory(spm_addr);
        } else if(mvout_start) {
            access_ok = spm->ReadMemory(spm_addr);
        }

        // Port conflict/backpressure: stall this beat and retry next cycle.
        if(access_ok) {
            // Compute Bandwidth
            HardWareUtilizationTracker* util_tracker = controller->getUtilTracker();
            if(util_tracker) {
                if(mvin_start) {
                    util_tracker->addDdrReadBytes(axi_beat_bytes);
                    util_tracker->addSpmWriteBytes(axi_beat_bytes);
                } else if(mvout_start) {
                    util_tracker->addSpmReadBytes(axi_beat_bytes);
                    util_tracker->addDdrWriteBytes(axi_beat_bytes);
                }
            }

            --current_burst_beats_remaining;
            ++current_row_beat_idx;

            if(current_row_idx < row_beat_plan.size() &&
               current_row_beat_idx == row_beat_plan[current_row_idx]) {
                ++current_row_idx;
                current_row_beat_idx = 0;
            }

            if(current_burst_beats_remaining == 0) {
                ++current_burst_idx;
                if(current_burst_idx < burst_beat_plan.size()) {
                    current_burst_beats_remaining = burst_beat_plan[current_burst_idx];
                }
            }
        }
    }

    if(remaining_cycles > 0) {
        remaining_cycles--;
    }

    auto core = simulator->getCore();
    Message getMsg{};
    if(!resp_done && core->tryConsumeResp(expectedMsgType, getMsg)){
        logger.log("[DMA] get expected resp type = %s", MsgTypeToStr(getMsg.type).c_str());
        resp_done = true;
    }

    const bool spm_transfer_done = (current_burst_idx >= burst_beat_plan.size());

    if(resp_done && spm_transfer_done && remaining_cycles == 0){
        logger.log("[DMA] DDR response and SPM transfer both completed, DMA complete");
        completeInstruction();
    }
}

void DMA::execute(){
    assert(!isBusy() && !isFinish());
    assert(state == State::IDLE);

    state = State::BUSY;

    resp_done = false;
    row_beat_plan.clear();
    burst_beat_plan.clear();
    current_row_idx = 0;
    current_row_beat_idx = 0;
    current_burst_idx = 0;
    current_burst_beats_remaining = 0;

    uint64_t start_reg = regfile.read(regfile.start_reg_addr);
    mvin_start  = (start_reg & (1ULL << 0)) != 0;
    mvout_start = (start_reg & (1ULL << 1)) != 0;

    Message req{};
    uint8_t data_precision = 0;

    if(mvin_start && !mvout_start){
        req.type = MsgType::LD_REQ;
        req.row_num = (int)((regfile.read(regfile.mvin_rs1_reg_addr) >> 32) & 0xFFFF);
        req.col_num = (int)((regfile.read(regfile.mvin_rs2_reg_addr) >> 32) & 0xFFFF);
        req.dram_stride = (int)((regfile.read(regfile.config_mvin_rs1_reg_addr) >> 48) & 0xFFFF);
        req.sram_stride = (int)((regfile.read(regfile.config_mvin_rs1_reg_addr) >> 32) & 0xFFFF);

        data_precision |= ((regfile.read(regfile.config_mvin_rs1_reg_addr) >> 1) & 0x1) << 0;
        data_precision |= ((regfile.read(regfile.config_mvin_rs1_reg_addr) >> 2) & 0x3) << 1;
        data_precision |= ((regfile.read(regfile.config_mvin_rs1_reg_addr) >> 4) & 0x1) << 3;
        switch(data_precision){
            case 2:  req.data_size = 1; break;
            case 4:  req.data_size = 2; break;
            case 10: req.data_size = 4; break;
            case 3:  req.data_size = 1; break;
            default: req.data_size = 0; break;
        }

        req.quant = (regfile.read(regfile.config_mvin_rs1_reg_addr) >> 4) & 0x1;
        req.src_addr = (uint32_t)(regfile.read(regfile.mvin_rs1_reg_addr) & 0xFFFFFFFF);
        req.dst_addr = (uint32_t)(regfile.read(regfile.mvin_rs2_reg_addr) & 0xFFFFFFFF);
        expectedMsgType = MsgType::LD_RESP;

        logger.log("[DMA] MVIN: send LD_REQ, src_addr = 0x%x, dst_addr = 0x%x, row_num = %d, col_num = %d, dram_stride = %d, sram_stride = %d, data_size = %d",
            req.src_addr, req.dst_addr, req.row_num, req.col_num, req.dram_stride, req.sram_stride, req.data_size);

    } else if(!mvin_start && mvout_start){
        req.type = MsgType::ST_REQ;
        req.row_num = (int)((regfile.read(regfile.mvout_rs1_reg_addr) >> 32) & 0xFFFF);
        req.col_num = (int)((regfile.read(regfile.mvout_rs2_reg_addr) >> 32) & 0xFFFF);
        req.dram_stride = (int)((regfile.read(regfile.config_mvout_rs1_reg_addr) >> 48) & 0xFFFF);
        req.sram_stride = (int)((regfile.read(regfile.config_mvout_rs1_reg_addr) >> 32) & 0xFFFF);

        data_precision |= ((regfile.read(regfile.config_mvout_rs1_reg_addr) >> 1) & 0x1) << 0;
        data_precision |= ((regfile.read(regfile.config_mvout_rs1_reg_addr) >> 2) & 0x3) << 1;
        data_precision |= ((regfile.read(regfile.config_mvout_rs1_reg_addr) >> 4) & 0x1) << 3;
        switch(data_precision){
            case 2:  req.data_size = 1; break;
            case 4:  req.data_size = 2; break;
            case 10: req.data_size = 4; break;
            case 3:  req.data_size = 1; break;
            default: req.data_size = 0; break;
        }

        req.quant = (regfile.read(regfile.config_mvout_rs1_reg_addr) >> 4) & 0x1;
        req.src_addr = (uint32_t)(regfile.read(regfile.mvout_rs1_reg_addr) & 0xFFFFFFFF);
        req.dst_addr = (uint32_t)(regfile.read(regfile.mvout_rs2_reg_addr) & 0xFFFFFFFF);
        expectedMsgType = MsgType::ST_RESP;

        logger.log("[DMA] MVOUT: send ST_REQ, src_addr = 0x%x, dst_addr = 0x%x, row_num = %d, col_num = %d, dram_stride = %d, sram_stride = %d, data_size = %d",
            req.src_addr, req.dst_addr, req.row_num, req.col_num, req.dram_stride, req.sram_stride, req.data_size);

    } else{
        remaining_cycles = 10;
        logger.log("[DMA] No clear MVIN/MVOUT, fallback %d cycles", remaining_cycles);
        return;
    }

    axi_beat_bytes = cfg.axi_data_width / 8;
    spm_base_addr = mvin_start ? req.dst_addr : req.src_addr;
    spm_row_stride_bytes = req.sram_stride * req.data_size;

    remaining_cycles = build_dma_spm_plan(cfg, req, row_beat_plan, burst_beat_plan);

    if(!burst_beat_plan.empty()) {
        current_burst_beats_remaining = burst_beat_plan[0];
    }

    auto core = simulator->getCore();
    core->sendMsg(req);
    logger.log("[DMA] Send request to DDR, DMA set to busy");
}

void DMA::completeInstruction(){
    PerfFunctionalUnit::completeInstruction();

    resp_done = false;
    row_beat_plan.clear();
    burst_beat_plan.clear();
    current_row_idx = 0;
    current_row_beat_idx = 0;
    current_burst_idx = 0;
    current_burst_beats_remaining = 0;

    // Update ISR Reg
    assert((mvin_start || mvout_start) && "DMA No ISR Reg Write");
    uint64_t isr_val = regfile.read(regfile.isr_reg_addr);

    if(mvin_start) {
        isr_val |= (1ULL << 0);
        regfile.write(regfile.isr_reg_addr, isr_val);
        logger.log("[DMA] ISR Reg[0] Updated");
        mvin_start = false;
    }

    if(mvout_start) {
        isr_val |= (1ULL << 1);
        regfile.write(regfile.isr_reg_addr, isr_val);
        logger.log("[DMA] ISR Reg[1] Updated");
        mvout_start = false;
    }
}
