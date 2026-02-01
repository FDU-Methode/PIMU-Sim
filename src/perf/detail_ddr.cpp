#include "../../include/perf/detail_ddr.hpp"

DetailDDR::DetailDDR(PerfSim* sim, const Config& cfg, Logger& logger) :
    BaseModule(sim, cfg, logger) {
        
        std::string config_file = cfg.dramsim_cfg_path; // DRAMSim3配置文件路径
        std::string output_dir = "./log/dramsim3_output"; // DRAMSim3输出文件路径
      
        m_dramsim = std::make_unique<MemorySystem>(
            config_file,
            output_dir,
            std::bind(&DetailDDR::readComplete, this, std::placeholders::_1),
            std::bind(&DetailDDR::writeComplete, this, std::placeholders::_1)
        );
        request_queue_depth = cfg.ddr_request_queue_depth;
    
        logger.log("----------------------------------------------------");
        logger.log("Initializing Detailed DDR Model");
        logger.log("  DDR Frequency: %2f GHz", cfg.ddr_freq);
        logger.log("----------------------------------------------------");
}

void DetailDDR::tick() {
    //logger.log("DDR is ticked");
    _cycle++;
    m_dramsim->ClockTick();

    // 如果有活跃请求，尝试发送剩余的事务
    if (current_request.active) {
        // 检查是否还有未发送的事务
        if (current_request.trans_index < current_request.total_trans) {
            // 计算当前事务的地址和类型
            uint64_t addr = current_request.msg.dst_addr + 
                          current_request.trans_index * dram_unit;
            bool is_write = (current_request.msg.type == MsgType::ST_REQ);

            // 检查是否可以发送事务
            if (m_dramsim->WillAcceptTransaction(addr, is_write)) {
                // 发送事务
                m_dramsim->AddTransaction(addr, is_write);
                current_request.trans_index++;
                logger.log("[DDR] Sent transaction %d/%d", current_request.trans_index, current_request.total_trans);
            }
        }
        //logger.log("[DDR] Active request");
        return;  // 有活跃请求时，不处理新请求
    }

    // 如果没有活跃请求，尝试处理新请求
    if (!request_queue.empty()) {
        Message& msg = request_queue.front();

        logger.log("[DDR] Processing new message (type=%s, size=%d)", 
                  (msg.type == MsgType::ST_REQ ? "STORE_REQ" : "LOAD_REQ"), msg.data_size);
        
        // 计算需要的事务数量
        // int access_data_size;
        // if (msg.type == MsgType::ST_REQ) {
        //     access_data_size = msg.data_size; // ST请求时，需写入data_size大小的数据
        // } else {
        //     access_data_size = msg.request_data_size; // LD请求时，需从DRAM读取request_data_size大小的数据
        // }
        // int num_trans = (access_data_size + dram_unit - 1) / dram_unit;
        int num_trans = 1;  // update this later........
        logger.log("[DDR] Message requires %d transactions", num_trans);
        
        // 初始化新请求
        current_request.active = true;
        current_request.msg = msg;
        current_request.total_trans = num_trans;
        current_request.trans_index = 0;
        current_request.completed_count = 0;
        current_request.completed_trans.resize(num_trans, false);
        
        // 从队列中移除消息
        request_queue.pop();
        
        // 尝试发送第一个事务
        uint64_t addr = msg.dst_addr;
        bool is_write = (msg.type == MsgType::ST_REQ);

        logger.log("[DDR] Attempting to send first transaction to Dramsim3");
        
        if (m_dramsim->WillAcceptTransaction(addr, is_write)) {
            m_dramsim->AddTransaction(addr, is_write);
            current_request.trans_index++;
            logger.log("[DDR] First transaction sent successfully");
        }
    }
}

void DetailDDR::sendMsg(Message& msg) {
    auto core = sim->getCore();
    core->recvMsg(msg);
}

bool DetailDDR::recvMsg(Message& msg) {
    if (request_queue.size() >= request_queue_depth) {
        logger.log("[DDR] request queue is full, backpressure asserted");
        return false; // DDR不能接收新的请求
    } else {
        request_queue.push(msg);
        logger.log("[DDR] received request from core");
        return true;
    }
}

bool DetailDDR::tryConsumeResp(MsgType expectedMsg, Message& getMsg) {};

void DetailDDR::readComplete(uint64_t addr) {
    logger.log("[DDR] Read request completed");

    if (!current_request.active) {
        std::cout << "[DDR] Warning: Read callback with no active request" << std::endl;
        throw std::runtime_error("Read callback with no active request");
    }

    // 获取当前请求的信息
    Message& msg = current_request.msg;
    uint64_t base_addr = msg.dst_addr;
    int trans_index = (addr - base_addr) / dram_unit;

    logger.log("[DDR] Read completion for transaction %d/%d", trans_index, current_request.total_trans);
    
    // 检查地址是否在当前请求的范围内且未完成
    logger.log("[DDR] Checking transaction %d ...", trans_index);
    logger.log("[DDR] total_trans: %d", current_request.total_trans);
    //logger.log("[DDR] completed_trans: %d", current_request.completed_trans[trans_index]);

    if (trans_index >= 0 && trans_index < current_request.total_trans && 
        !current_request.completed_trans[trans_index]) {
    
        // 标记事务为完成状态
        current_request.completed_trans[trans_index] = true;
        current_request.completed_count++;
        logger.log("[DDR] Transaction marked as completed (%d/%d)", current_request.completed_count, current_request.total_trans);
        
        // 检查是否所有事务都已完成且都已发送
        if (current_request.completed_count == current_request.total_trans &&
            current_request.trans_index == current_request.total_trans) {
            //std::cout << "[DDR] All transactions completed, sending response op_id=" << 
            //current_request.msg.payload.reserved << std::endl;

            // 创建响应消息
            Message resp_msg = msg;
            //resp_msg.src = coord;
            //resp_msg.dst = msg.src;
            resp_msg.type = MsgType::LD_RESP;
            // 发送响应
            sendMsg(resp_msg);
            // 清除当前请求
            clearCurrentRequest();
        }
    }
}

void DetailDDR::writeComplete(uint64_t addr) {
    if (!current_request.active) {
        std::cout << "[DDR] Warning: Write callback with no active request" << std::endl;
        throw std::runtime_error("Write callback with no active request");
    }

    // 获取当前请求的信息
    Message& msg = current_request.msg;
    uint64_t base_addr = msg.dst_addr;
    int trans_index = (addr - base_addr) / dram_unit;

    logger.log("[DDR] Write completion for transaction %d/%d", trans_index, current_request.total_trans);
    
    // 检查地址是否在当前请求的范围内且未完成
    if (trans_index >= 0 && 
        trans_index < current_request.total_trans && 
        !current_request.completed_trans[trans_index]) {
        
        // 标记事务为完成状态
        current_request.completed_trans[trans_index] = true;
        current_request.completed_count++;
        logger.log("[DDR] Transaction marked as completed (%d/%d)", current_request.completed_count, current_request.total_trans);

        // 检查是否所有事务都已完成且都已发送
        if (current_request.completed_count == current_request.total_trans &&
            current_request.trans_index == current_request.total_trans) {
            logger.log("[DDR] All transactions completed, sending response");

            // 创建响应消息
            Message resp_msg = msg;
            //resp_msg.src = coord;
            //resp_msg.dst = msg.src;
            resp_msg.type = MsgType::ST_RESP;
            
            // 发送响应
            sendMsg(resp_msg);
            
            // 清除当前请求
            clearCurrentRequest();
        }
    }
}

void DetailDDR::clearCurrentRequest() {
    logger.log("[DDR] Clearing current request state");
    current_request.active = false;
    current_request.msg = Message();  // 重置消息对象
    current_request.completed_count = 0;
    current_request.total_trans = 0;
    current_request.trans_index = 0;
    current_request.completed_trans.clear();
    logger.log("[DDR] Request state cleared");
}