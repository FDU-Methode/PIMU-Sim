#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/config.hpp"
#include "../include/func_sim.hpp"
#include "../include/perf/perf_sim.hpp"
#include "../npux-mlir/model_test/runtime/npu_runtime.h"
#include "../npux-mlir/src/Runtime/ExecutionSession.hpp"
#include "../npux-mlir/include/OnnxMlirRuntime.h"

namespace {

namespace fs = std::filesystem;

struct InputSpec {
    std::string name;
    OM_DATA_TYPE dtype;
    size_t elemBytes;
    std::vector<int64_t> dims;
};

std::string trim(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
        s.erase(s.begin());
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
        s.pop_back();
    return s;
}

std::string shellQuote(const std::string &s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'')
            out += "'\\''";
        else
            out += c;
    }
    out += "'";
    return out;
}

std::string inferModelPrefix(const std::string &inputDir, const std::string &modelLibPath) {
    fs::path inputPath(inputDir);
    if (!inputPath.filename().empty())
        return inputPath.filename().string();

    fs::path libPath(modelLibPath);
    return libPath.stem().string();
}

std::string dtypeToString(OM_DATA_TYPE dtype) {
    return std::string(OM_DATA_TYPE_NAME[static_cast<int>(dtype)]);
}

size_t computeNumElems(const std::vector<int64_t> &dims) {
    size_t numElems = 1;
    for (int64_t d : dims)
        numElems *= static_cast<size_t>(d);
    return numElems;
}

std::string extractJsonStringField(const std::string &obj, const std::string &field) {
    std::regex re("\"" + field + R"__REGEX__(\s*:\s*"([^"]+)")__REGEX__");
    std::smatch m;
    if (std::regex_search(obj, m, re))
        return m[1].str();
    return "";
}

std::string extractJsonArrayField(const std::string &obj, const std::string &field) {
    std::regex re("\"" + field + R"__REGEX__(\s*:\s*\[([^\]]*)\])__REGEX__");
    std::smatch m;
    if (std::regex_search(obj, m, re))
        return m[1].str();
    return "";
}

InputSpec parseInputObject(const std::string &obj) {
    InputSpec spec{};
    spec.name = extractJsonStringField(obj, "name");

    std::string typeStr = extractJsonStringField(obj, "type");
    std::string dimsStr = extractJsonArrayField(obj, "dims");
    if (typeStr.empty())
        throw std::runtime_error("Missing input type in signature object: " + obj);

    std::string t = typeStr;
    std::transform(t.begin(), t.end(), t.begin(),
        [](unsigned char c) { return std::tolower(c); });

    if (t == "f32") {
        spec.dtype = ONNX_TYPE_FLOAT;
    } else if (t == "f64") {
        spec.dtype = ONNX_TYPE_DOUBLE;
    } else if (t == "f16") {
        spec.dtype = ONNX_TYPE_FLOAT16;
    } else if (t == "bf16") {
        spec.dtype = ONNX_TYPE_BFLOAT16;
    } else if (t == "i64") {
        spec.dtype = ONNX_TYPE_INT64;
    } else if (t == "i32") {
        spec.dtype = ONNX_TYPE_INT32;
    } else if (t == "i16") {
        spec.dtype = ONNX_TYPE_INT16;
    } else if (t == "i8") {
        spec.dtype = ONNX_TYPE_INT8;
    } else if (t == "ui64") {
        spec.dtype = ONNX_TYPE_UINT64;
    } else if (t == "ui32") {
        spec.dtype = ONNX_TYPE_UINT32;
    } else if (t == "ui16") {
        spec.dtype = ONNX_TYPE_UINT16;
    } else if (t == "ui8") {
        spec.dtype = ONNX_TYPE_UINT8;
    } else if (t == "bool") {
        spec.dtype = ONNX_TYPE_BOOL;
    } else {
        throw std::runtime_error("Unsupported input element type: " + typeStr);
    }

    spec.elemBytes = static_cast<size_t>(getDataTypeSize(spec.dtype));
    if (spec.elemBytes == 0)
        throw std::runtime_error("Unsupported element byte size for type: " + typeStr);

    std::string dims = trim(dimsStr);
    if (!dims.empty()) {
        std::stringstream ss(dims);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            tok = trim(tok);
            if (tok.empty())
                continue;
            int64_t dim = std::stoll(tok);
            if (dim < 0)
                throw std::runtime_error("Dynamic dimension is not supported: " + tok);
            spec.dims.push_back(dim);
        }
    }

    return spec;
}

std::vector<InputSpec> parseInputSignature(const std::string &signature) {
    std::vector<InputSpec> specs;
    const std::regex objRe(R"__REGEX__(\{[^{}]*\})__REGEX__");

    auto begin = std::sregex_iterator(signature.begin(), signature.end(), objRe);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        specs.push_back(parseInputObject(it->str()));
    }

    if (specs.empty())
        throw std::runtime_error("Failed to parse model input signature: " + signature);

    return specs;
}

std::vector<uint8_t> loadBinaryFileExact(const fs::path &path, size_t expectedBytes) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open())
        throw std::runtime_error("Cannot open input file: " + path.string());

    ifs.seekg(0, std::ios::end);
    std::streamsize actualSize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    if (actualSize != static_cast<std::streamsize>(expectedBytes)) {
        std::ostringstream oss;
        oss << "Input file size mismatch for " << path.string()
            << ", expected " << expectedBytes << " bytes, got " << actualSize;
        throw std::runtime_error(oss.str());
    }

    std::vector<uint8_t> data(expectedBytes == 0 ? 1 : expectedBytes);
    if (!ifs.read(reinterpret_cast<char *>(data.data()), actualSize))
        throw std::runtime_error("Failed to read input file: " + path.string());

    return data;
}

fs::path resolveInputFile(
    const std::string &inputDir,
    const std::string &modelPrefix,
    const InputSpec &spec,
    size_t index,
    size_t totalInputs) {

    fs::path dir(inputDir);
    std::vector<fs::path> candidates;

    candidates.push_back(dir / ("input_" + std::to_string(index) + ".bin"));

    if (totalInputs == 1)
        candidates.push_back(dir / "input.bin");

    if (!spec.name.empty()) {
        candidates.push_back(dir / (spec.name + ".bin"));
        if (!modelPrefix.empty())
            candidates.push_back(dir / (modelPrefix + "_" + spec.name + ".bin"));
    }

    if (totalInputs == 1 && !modelPrefix.empty())
        candidates.push_back(dir / (modelPrefix + "_input.bin"));

    for (const auto &p : candidates) {
        if (fs::exists(p))
            return p;
    }

    std::ostringstream oss;
    oss << "Cannot resolve input file for input #" << index << " under " << inputDir;
    throw std::runtime_error(oss.str());
}

OMTensorList *loadInputListFromDir(
    const std::vector<InputSpec> &specs,
    const std::string &inputDir,
    const std::string &modelPrefix) {

    std::vector<OMTensor *> tensors;
    tensors.reserve(specs.size());

    for (size_t i = 0; i < specs.size(); ++i) {
        const InputSpec &spec = specs[i];
        const size_t numElems = computeNumElems(spec.dims);
        const size_t expectedBytes = numElems * spec.elemBytes;
        fs::path inputPath = resolveInputFile(inputDir, modelPrefix, spec, i, specs.size());

        std::vector<uint8_t> data = loadBinaryFileExact(inputPath, expectedBytes);

        void *buffer = std::malloc(expectedBytes == 0 ? 1 : expectedBytes);
        if (!buffer) {
            for (OMTensor *t : tensors)
                omTensorDestroy(t);
            throw std::runtime_error("Failed to allocate tensor buffer for input");
        }

        if (expectedBytes > 0)
            std::memcpy(buffer, data.data(), expectedBytes);

        OMTensor *tensor = omTensorCreateWithOwnership(
            buffer,
            spec.dims.empty() ? nullptr : spec.dims.data(),
            static_cast<int64_t>(spec.dims.size()),
            spec.dtype,
            1);

        if (!tensor) {
            std::free(buffer);
            for (OMTensor *t : tensors)
                omTensorDestroy(t);
            throw std::runtime_error("Failed to create input tensor");
        }

        tensors.push_back(tensor);
    }

    OMTensorList *inputList =
        omTensorListCreate(tensors.data(), static_cast<int64_t>(tensors.size()));
    if (!inputList) {
        for (OMTensor *t : tensors)
            omTensorDestroy(t);
        throw std::runtime_error("Failed to create input tensor list");
    }

    return inputList;
}

void dumpOutputListToDir(OMTensorList *outputList, const std::string &outputDir) {
    fs::create_directories(outputDir);

    const int64_t outputCount = omTensorListGetSize(outputList);
    for (int64_t i = 0; i < outputCount; ++i) {
        OMTensor *tensor = omTensorListGetOmtByIndex(outputList, i);
        if (!tensor)
            throw std::runtime_error("Null output tensor at index " + std::to_string(i));

        const void *dataPtr = omTensorGetDataPtr(tensor);
        const int64_t rank = omTensorGetRank(tensor);
        const int64_t *shape = omTensorGetShape(tensor);
        const OM_DATA_TYPE dtype = omTensorGetDataType(tensor);
        const int64_t bufferSize = omTensorGetBufferSize(tensor);

        fs::path binPath = fs::path(outputDir) / ("output_" + std::to_string(i) + ".bin");
        fs::path metaPath = fs::path(outputDir) / ("output_" + std::to_string(i) + ".meta.txt");

        {
            std::ofstream ofs(binPath, std::ios::binary);
            if (!ofs.is_open())
                throw std::runtime_error("Cannot open output file for write: " + binPath.string());
            if (bufferSize > 0)
                ofs.write(reinterpret_cast<const char *>(dataPtr), bufferSize);
        }

        {
            std::ofstream meta(metaPath);
            if (!meta.is_open())
                throw std::runtime_error("Cannot open output meta file for write: " + metaPath.string());
            meta << "dtype=" << dtypeToString(dtype) << "\n";
            meta << "rank=" << rank << "\n";
            meta << "shape=";
            for (int64_t d = 0; d < rank; ++d) {
                if (d) meta << ",";
                meta << shape[d];
            }
            meta << "\n";
            meta << "buffer_size=" << bufferSize << "\n";
        }
    }
}

std::string compileOnnxModelToSharedLib(const Config &cfg) {
    if (cfg.onnx_model_file.empty())
        throw std::runtime_error("onnx_model_file is empty");

    fs::path onnxPath(cfg.onnx_model_file);
    if (!fs::exists(onnxPath))
        throw std::runtime_error("onnx_model_file does not exist: " + onnxPath.string());

    fs::path workDir(cfg.compile_work_dir.empty() ? "./build/compiler_run" : cfg.compile_work_dir);
    fs::create_directories(workDir);

    fs::path outputBase = workDir / onnxPath.stem();
    std::string compiler = cfg.onnx_mlir_bin.empty() ? "onnx-mlir" : cfg.onnx_mlir_bin;

    std::ostringstream cmd;
    cmd << shellQuote(compiler) << " "
        << shellQuote(onnxPath.string()) << " "
        << "-o " << shellQuote(outputBase.string()) << " ";
    if (!cfg.onnx_mlir_flags.empty())
        cmd << cfg.onnx_mlir_flags;

    int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        std::ostringstream oss;
        oss << "onnx-mlir compile failed with exit code " << rc;
        throw std::runtime_error(oss.str());
    }

    fs::path sharedLib = outputBase;
    sharedLib += ".so";
    if (!fs::exists(sharedLib))
        throw std::runtime_error("Compiled shared library not found: " + sharedLib.string());

    return sharedLib.string();
}

void runCompiledModelLibrary(
    const std::string &modelLibPath,
    const std::string &inputDir,
    const std::string &outputDir) {

    const std::string modelPrefix = inferModelPrefix(inputDir, modelLibPath);

    onnx_mlir::ExecutionSession session(modelLibPath, "NONE", true);
    const std::string inputSignature = session.inputSignature();
    std::vector<InputSpec> inputSpecs = parseInputSignature(inputSignature);

    OMTensorList *inputList = loadInputListFromDir(inputSpecs, inputDir, modelPrefix);
    OMTensorList *outputList = nullptr;

    try {
        outputList = session.run(inputList);
        dumpOutputListToDir(outputList, outputDir);
    } catch (...) {
        if (outputList) omTensorListDestroy(outputList);
        omTensorListDestroy(inputList);
        throw;
    }

    omTensorListDestroy(inputList);
    omTensorListDestroy(outputList);
}

} // namespace

int main(int argc, char* argv[]){
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }

    std::string cfg_file = argv[1];
    Config cfg = Config::from_file(cfg_file);

    std::cout << "Starting to create simulator... Max sim cycle = "
              << cfg.max_sim_cycle << std::endl;

    std::unique_ptr<SimBase> simulator;

    std::cout << "Initializing simulator in mode: "
              << (cfg.simulator_type == SimMode::FUNCTIONAL ? "FUNCTIONAL" : "PERFORMANCE")
              << std::endl;

    if (cfg.simulator_type == SimMode::FUNCTIONAL){
        simulator = std::make_unique<FuncSim>(cfg);
        simulator->run();
        return 0;
    }

    cfg.inst_file.clear();
    simulator = std::make_unique<PerfSim>(cfg);

    auto* perf_sim = static_cast<PerfSim*>(simulator.get());
    npu_bind_simulator_backend(perf_sim);
    if (npu_init() != 0) {
        std::cerr << "Failed to initialize NPU runtime." << std::endl;
        return 1;
    }

    perf_sim->getPerfCore()->enableExternalInstFeed();

    std::string modelLibPath = cfg.compiled_model_lib;
    if (modelLibPath.empty()) {
        if (cfg.onnx_model_file.empty()) {
            std::cerr << "Both compiled_model_lib and onnx_model_file are empty." << std::endl;
            npu_destroy();
            return 1;
        }
        try {
            modelLibPath = compileOnnxModelToSharedLib(cfg);
        } catch (const std::exception& e) {
            std::cerr << "Failed to compile ONNX model: " << e.what() << std::endl;
            npu_destroy();
            return 1;
        }
    }

    if (cfg.model_input_dir.empty()) {
        std::cerr << "model_input_dir is empty in PERFORMANCE mode." << std::endl;
        npu_destroy();
        return 1;
    }

    try {
        perf_sim->beginExternalExecution();
        runCompiledModelLibrary(modelLibPath, cfg.model_input_dir, cfg.model_output_dir);
        perf_sim->endExternalExecution();
        perf_sim->print_stats();
    } catch (const std::exception& e) {
        std::cerr << "Failed to execute compiled model: " << e.what() << std::endl;
        npu_destroy();
        return 1;
    }

    npu_destroy();
    return 0;
}