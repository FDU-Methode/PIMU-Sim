#include <cassert>
#include "../../../include/perf/perf_func_unit.hpp"
#include "../../../include/perf/perf_sim.hpp"

DMA::DMA(PerfSim* sim, PipelineController* controller, Logger& logger, RegFile& regfile, const Config& cfg) :
    PerfFunctionalUnit(sim, controller, logger, regfile, cfg) {
    UnitName = "DMA";
}

void DMA::tick(){
    if(!isBusy()) return;

    if(remaining_cycles > 0){
        remaining_cycles--;
        if(remaining_cycles == 0) completeInstruction();
        return;
    }

    auto core = simulator->getCore();
    Message getMsg{};
    if(core->tryConsumeResp(expectedMsgType, getMsg)){
        logger.log("[DMA] get expected resp type = %s, DMA complete", MsgTypeToStr(getMsg.type).c_str());
        completeInstruction();
    }
}

void DMA::execute(){
    assert(!isBusy() && !isFinish());
    assert(state == State::IDLE);

    state = State::BUSY;

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

    auto core = simulator->getCore();
    core->sendMsg(req);
    logger.log("[DMA] Send request to DDR, DMA set to busy");
}

void DMA::completeInstruction(){
    PerfFunctionalUnit::completeInstruction();

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