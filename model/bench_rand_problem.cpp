#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include <nlohmann/json.hpp>

#include "cogent.hpp"
#include "tcont.hpp"
#include "ttgt_cutt.hpp"

namespace
{

std::string trim(const std::string& text)
{
    size_t start = 0;
    while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start]))) {
        ++start;
    }

    size_t end = text.size();
    while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return text.substr(start, end - start);
}

std::vector<std::string> split_tokens(const std::string& text, char delimiter)
{
    std::vector<std::string> tokens;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, delimiter)) {
        tokens.push_back(trim(item));
    }
    return tokens;
}

struct EquationProblem {
    std::string name;
    std::string raw_equation;
    std::vector<char> modeC;
    std::vector<char> modeA;
    std::vector<char> modeB;
    std::unordered_map<char, int64_t> extent;
};

struct ProblemMetadata {
    std::string input_path;
    std::string source_collection;
    std::string equation_group;
    std::string problem_file;
    int problem_id = -1;
};

std::vector<char> parse_mode_list(const std::string& spec, std::unordered_map<char, int64_t>& extent_map)
{
    std::vector<char> modes;
    std::vector<std::string> tokens = split_tokens(spec, ',');
    if (tokens.empty()) {
        return modes;
    }

    bool has_extents = false;
    if (tokens.size() % 2 == 0) {
        has_extents = true;
        for (size_t i = 0; i + 1 < tokens.size(); i += 2) {
            const std::string& mode_token = tokens[i];
            const std::string& extent_token = tokens[i + 1];
            if (mode_token.empty() || extent_token.empty()) {
                has_extents = false;
                break;
            }

            const unsigned char first = static_cast<unsigned char>(mode_token[0]);
            if (!std::isalpha(first)) {
                has_extents = false;
                break;
            }

            const size_t numeric_start = extent_token[0] == '-' ? 1U : 0U;
            if (numeric_start >= extent_token.size()) {
                has_extents = false;
                break;
            }

            if (!std::all_of(
                    extent_token.begin() + static_cast<std::ptrdiff_t>(numeric_start),
                    extent_token.end(),
                    [](unsigned char ch) { return std::isdigit(ch) != 0; })) {
                has_extents = false;
                break;
            }
        }
    }

    if (has_extents) {
        for (size_t i = 0; i + 1 < tokens.size(); i += 2) {
            const std::string& mode_token = tokens[i];
            const std::string& extent_token = tokens[i + 1];
            if (mode_token.empty()) {
                throw std::invalid_argument("Empty mode token in: " + spec);
            }
            const char mode = mode_token[0];
            const int64_t value = std::stoll(extent_token);
            modes.push_back(mode);
            extent_map[mode] = value;
        }
    } else {
        for (const std::string& token : tokens) {
            if (!token.empty()) {
                modes.push_back(token[0]);
            }
        }
    }

    return modes;
}

std::unordered_map<char, int64_t> parse_sum_extents(const std::string& sum_text)
{
    std::unordered_map<char, int64_t> extents;
    std::vector<std::string> tokens = split_tokens(sum_text, ',');
    if (tokens.size() % 2 != 0) {
        throw std::invalid_argument("Malformed sum extents: " + sum_text);
    }

    for (size_t i = 0; i + 1 < tokens.size(); i += 2) {
        if (tokens[i].empty()) {
            throw std::invalid_argument("Malformed sum extents: " + sum_text);
        }
        extents[tokens[i][0]] = std::stoll(tokens[i + 1]);
    }
    return extents;
}

EquationProblem parse_equation_file(const std::filesystem::path& input_path)
{
    std::ifstream file(input_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open equation file: " + input_path.string());
    }

    std::string line;
    std::string primary_line;
    std::string comment_line;
    while (std::getline(file, line)) {
        const std::string trimmed = trim(line);
        if (trimmed.empty()) {
            continue;
        }
        if (trimmed.rfind('#', 0) == 0) {
            comment_line = trim(trimmed.substr(1));
            continue;
        }
        primary_line = trimmed;
        break;
    }

    if (primary_line.empty()) {
        throw std::invalid_argument("No equation line found in " + input_path.string());
    }

    EquationProblem result;
    result.name = comment_line;
    result.raw_equation = primary_line;

    std::smatch sum_match;
    const std::regex sum_regex(R"((?:sum|Sum)\s*\(([^)]*)\))");
    if (std::regex_search(primary_line, sum_match, sum_regex)) {
        const auto extents = parse_sum_extents(sum_match[1].str());
        result.extent.insert(extents.begin(), extents.end());
    }

    const std::regex tensor_regex(R"(([A-Za-z0-9_]+)\s*\[([^\]]+)\])");
    std::vector<std::pair<std::string, std::string>> tensor_specs;
    for (std::sregex_iterator it(primary_line.begin(), primary_line.end(), tensor_regex), end; it != end; ++it) {
        tensor_specs.emplace_back((*it)[1].str(), (*it)[2].str());
    }

    if (tensor_specs.size() < 3) {
        throw std::invalid_argument("Expected at least 3 tensor specifications in equation: " + primary_line);
    }

    result.modeC = parse_mode_list(tensor_specs[0].second, result.extent);
    result.modeA = parse_mode_list(tensor_specs[1].second, result.extent);
    result.modeB = parse_mode_list(tensor_specs[2].second, result.extent);

    for (char mode : result.modeC) {
        if (result.extent.find(mode) == result.extent.end()) {
            throw std::invalid_argument("Missing extent for mode: " + std::string(1, mode));
        }
    }
    for (char mode : result.modeA) {
        if (result.extent.find(mode) == result.extent.end()) {
            throw std::invalid_argument("Missing extent for mode: " + std::string(1, mode));
        }
    }
    for (char mode : result.modeB) {
        if (result.extent.find(mode) == result.extent.end()) {
            throw std::invalid_argument("Missing extent for mode: " + std::string(1, mode));
        }
    }

    return result;
}

ProblemMetadata extract_problem_metadata(const std::filesystem::path& input_path)
{
    ProblemMetadata metadata;
    metadata.input_path = input_path.string();
    metadata.problem_file = input_path.filename().string();

    const auto problem_parent = input_path.parent_path();
    if (!problem_parent.empty()) {
        metadata.equation_group = problem_parent.filename().string();
        const auto collection_parent = problem_parent.parent_path();
        if (!collection_parent.empty()) {
            metadata.source_collection = collection_parent.filename().string();
        }
    }

    const std::regex problem_regex(R"(problem_(\d+)\.in)");
    std::smatch match;
    if (std::regex_match(metadata.problem_file, match, problem_regex)) {
        metadata.problem_id = std::stoi(match[1].str());
    }

    return metadata;
}

TconT::ScalarType parse_scalar_type(const std::string& precision)
{
    if (precision == "fp64" || precision == "float64") {
        return TconT::ScalarType::Float64;
    }
    if (precision == "tf32" || precision == "fp32" || precision == "float32") {
        return TconT::ScalarType::Float32;
    }
    throw std::invalid_argument("Unknown precision: " + precision);
}

TconT::BackendSelection parse_backend_selection(const std::string& backend)
{
    if (backend == "direct" || backend == "cogent") {
        return TconT::BackendSelection::COGENT;
    }
    if (backend == "ttgt" || backend == "ttgt_cutt") {
        return TconT::BackendSelection::TTGT_CUTT;
    }
    throw std::invalid_argument("Unsupported backend: " + backend);
}

std::string to_lower(std::string value)
{
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

bool vector_contains(const std::vector<char>& values, char target)
{
    return std::find(values.begin(), values.end(), target) != values.end();
}

int64_t product_of_modes(const std::vector<char>& modes, const std::unordered_map<char, int64_t>& extent)
{
    int64_t total = 1;
    for (char mode : modes) {
        total *= extent.at(mode);
    }
    return total;
}

nlohmann::json build_equation_features(const EquationProblem& problem)
{
    nlohmann::json features;

    const int64_t size_a = product_of_modes(problem.modeA, problem.extent);
    const int64_t size_b = product_of_modes(problem.modeB, problem.extent);
    const int64_t size_c = product_of_modes(problem.modeC, problem.extent);

    std::set<char> unique_modes;
    std::set<char> internal_modes;
    std::set<char> output_only_left;
    std::set<char> output_only_right;
    std::set<char> output_shared;

    for (const auto& entry : problem.extent) {
        unique_modes.insert(entry.first);
    }

    for (char mode : problem.modeC) {
        const bool in_a = vector_contains(problem.modeA, mode);
        const bool in_b = vector_contains(problem.modeB, mode);
        if (in_a && in_b) {
            output_shared.insert(mode);
        } else if (in_a) {
            output_only_left.insert(mode);
        } else if (in_b) {
            output_only_right.insert(mode);
        }
    }

    for (char mode : problem.modeA) {
        if (vector_contains(problem.modeB, mode) && !vector_contains(problem.modeC, mode)) {
            internal_modes.insert(mode);
        }
    }

    int64_t internal_volume = 1;
    for (char mode : internal_modes) {
        internal_volume *= problem.extent.at(mode);
    }

    long double operation_count = 2.0L;
    for (char mode : unique_modes) {
        operation_count *= static_cast<long double>(problem.extent.at(mode));
    }

    std::vector<int64_t> extents;
    extents.reserve(problem.extent.size());
    for (const auto& entry : problem.extent) {
        extents.push_back(entry.second);
    }
    std::sort(extents.begin(), extents.end());

    const int64_t min_extent = extents.empty() ? 0 : extents.front();
    const int64_t max_extent = extents.empty() ? 0 : extents.back();
    const double mean_extent =
        extents.empty() ? 0.0 : std::accumulate(extents.begin(), extents.end(), 0.0) / static_cast<double>(extents.size());
    double variance = 0.0;
    for (int64_t value : extents) {
        const double diff = static_cast<double>(value) - mean_extent;
        variance += diff * diff;
    }
    if (!extents.empty()) {
        variance /= static_cast<double>(extents.size());
    }

    features["equation_name"] = problem.name;
    features["equation_string"] = problem.raw_equation;
    features["modeC"] = std::string(problem.modeC.begin(), problem.modeC.end());
    features["modeA"] = std::string(problem.modeA.begin(), problem.modeA.end());
    features["modeB"] = std::string(problem.modeB.begin(), problem.modeB.end());
    features["num_modes_C"] = static_cast<int>(problem.modeC.size());
    features["num_modes_A"] = static_cast<int>(problem.modeA.size());
    features["num_modes_B"] = static_cast<int>(problem.modeB.size());
    features["num_unique_modes"] = static_cast<int>(unique_modes.size());
    features["num_internal_modes"] = static_cast<int>(internal_modes.size());
    features["num_output_modes"] = static_cast<int>(problem.modeC.size());
    features["num_output_only_left_modes"] = static_cast<int>(output_only_left.size());
    features["num_output_only_right_modes"] = static_cast<int>(output_only_right.size());
    features["num_output_shared_modes"] = static_cast<int>(output_shared.size());
    features["size_A"] = size_a;
    features["size_B"] = size_b;
    features["size_C"] = size_c;
    features["output_volume"] = size_c;
    features["internal_volume"] = internal_volume;
    features["operation_count"] = static_cast<double>(operation_count);
    features["min_extent"] = min_extent;
    features["max_extent"] = max_extent;
    features["mean_extent"] = mean_extent;
    features["std_extent"] = std::sqrt(variance);
    features["extent_ratio_max_min"] = min_extent > 0 ? static_cast<double>(max_extent) / static_cast<double>(min_extent) : 0.0;
    features["log2_size_A"] = size_a > 0 ? std::log2(static_cast<double>(size_a)) : 0.0;
    features["log2_size_B"] = size_b > 0 ? std::log2(static_cast<double>(size_b)) : 0.0;
    features["log2_size_C"] = size_c > 0 ? std::log2(static_cast<double>(size_c)) : 0.0;
    features["log2_internal_volume"] = internal_volume > 0 ? std::log2(static_cast<double>(internal_volume)) : 0.0;

    features["mode_extents"] = nlohmann::json::object();
    for (const auto& entry : problem.extent) {
        features["mode_extents"][std::string(1, entry.first)] = entry.second;
    }

    features["sorted_extents"] = extents;
    features["internal_modes"] = nlohmann::json::array();
    for (char mode : internal_modes) {
        features["internal_modes"].push_back(std::string(1, mode));
    }

    return features;
}

nlohmann::json run_cogent_planner(const TconT::TCEquation& equation)
{
    nlohmann::json input;
    input["modeC"] = std::vector<char>(equation.modeC.begin(), equation.modeC.end());
    input["extentC"] = std::vector<int64_t>();
    for (char mode : equation.modeC) {
        input["extentC"].push_back(equation.extent.at(mode));
    }
    input["modeA"] = std::vector<char>(equation.modeA.begin(), equation.modeA.end());
    input["extentA"] = std::vector<int64_t>();
    for (char mode : equation.modeA) {
        input["extentA"].push_back(equation.extent.at(mode));
    }
    input["modeB"] = std::vector<char>(equation.modeB.begin(), equation.modeB.end());
    input["extentB"] = std::vector<int64_t>();
    for (char mode : equation.modeB) {
        input["extentB"].push_back(equation.extent.at(mode));
    }
    input["op"] = equation.op;
    input["type"] = TconT::scalar_type_name(equation.scalar_type);

    std::filesystem::path candidate = std::filesystem::current_path();
    std::filesystem::path script_path;
    while (true) {
        const std::filesystem::path maybe = candidate / "backends" / "cogent" / "plan" / "tc_plan.py";
        if (std::filesystem::exists(maybe)) {
            script_path = maybe;
            break;
        }
        if (!candidate.has_parent_path()) {
            break;
        }
        const std::filesystem::path parent = candidate.parent_path();
        if (parent == candidate) {
            break;
        }
        candidate = parent;
    }

    if (script_path.empty()) {
        throw std::runtime_error("Cogent planner script not found from current path: " + std::filesystem::current_path().string());
    }

    const std::string command = std::string("python3 ") + script_path.string() + " '" + input.dump() + "'";
    std::array<char, 4096> buffer{};
    std::string output;

    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        throw std::runtime_error("Failed to launch cogent planner");
    }

    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }

    const int status = pclose(pipe);
    if (status != 0) {
        throw std::runtime_error("Cogent planner failed with exit code " + std::to_string(status));
    }

    return nlohmann::json::parse(output);
}

nlohmann::json extract_ttgt_plan_features(const TconT::TCEquation& equation)
{
    ttgt_cuTT_handle plan;
    ttgt_cuTT_plan(plan, equation.modeC, equation.modeA, equation.modeB, equation.extent, 0);

    nlohmann::json config;
    config["swap_AB"] = plan.swap_AB;
    config["transpose_input_left"] = plan.transpose_input_left;
    config["transpose_input_right"] = plan.transpose_input_right;
    config["transpose_output"] = plan.transpose_output;
    config["gemm_m"] = plan.gemm_m;
    config["gemm_n"] = plan.gemm_n;
    config["gemm_k"] = plan.gemm_k;
    config["gemm_trans_A"] = plan.gemm_trans_A;
    config["gemm_trans_B"] = plan.gemm_trans_B;
    config["info_trans_A_dim"] = plan.info_trans_A_dim;
    config["info_trans_B_dim"] = plan.info_trans_B_dim;
    config["info_trans_C_dim"] = plan.info_trans_C_dim;
    config["info_trans_A_perm"] = plan.info_trans_A_perm;
    config["info_trans_B_perm"] = plan.info_trans_B_perm;
    config["info_trans_C_perm"] = plan.info_trans_C_perm;
    return config;
}

nlohmann::json extract_cogent_plan_features(const TconT::TCEquation& equation)
{
    const nlohmann::json output = run_cogent_planner(equation);
    nlohmann::json config;
    config["kernel_name"] = output.value("kernel_name", "");
    config["block_size"] = output.value("block_size", 0);
    if (output.contains("stage")) {
        if (output["stage"].is_array() && !output["stage"].empty()) {
            config["stage"] = output["stage"].front().get<int>();
        } else {
            config["stage"] = output["stage"].get<int>();
        }
    } else {
        config["stage"] = 0;
    }
    config["smem_x"] = output.value("smem_x", 0);
    config["smem_y"] = output.value("smem_y", 0);
    config["swap_flag"] = output.value("swap_flag", false);
    config["tile_sizes"] = output.value("tile_sizes", nlohmann::json::array());
    config["external_index"] = output.value("external_index", nlohmann::json::array());
    config["internal_index"] = output.value("internal_index", nlohmann::json::array());
    config["warp_shape"] = output.value("warp_shape", nlohmann::json::array());
    return config;
}

TconT::ContractionResult benchmark_run(
    const TconT::ExecutionRun& run,
    const TconT::TCEquation& equation,
    const TconT::RunOptions& options)
{
    TconT::warmup(run, options.warmup);
    TconT::zero_output(run);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    TconT::check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    TconT::check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    float total_ms = 0.0f;
    try {
        TconT::check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
        for (int i = 0; i < options.repeats; ++i) {
            TconT::launch(run);
        }
        TconT::check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
        TconT::check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
        TconT::check_cuda(cudaEventElapsedTime(&total_ms, start, stop), "cudaEventElapsedTime");
        TconT::check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
        TconT::check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    } catch (...) {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        throw;
    }

    const float avg_ms = options.repeats > 0 ? total_ms / static_cast<float>(options.repeats) : 0.0f;
    return TconT::ContractionResult{
        avg_ms,
        TconT::compute_gflops(equation, avg_ms),
        true,
    };
}

}  // namespace

int main(int argc, char* argv[])
{
    std::string input_path;
    std::string backend = "cogent";
    std::string precision = "fp64";
    int warmup = 3;
    int repeats = 10;
    bool print_help = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--backend" && i + 1 < argc) {
            backend = argv[++i];
        } else if (arg == "--precision" && i + 1 < argc) {
            precision = argv[++i];
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::stoi(argv[++i]);
        } else if (arg == "--repeats" && i + 1 < argc) {
            repeats = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_help = true;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    if (print_help || input_path.empty()) {
        std::cout
            << "Usage: " << argv[0]
            << " --input <problem_file> [--backend cogent|ttgt] [--precision fp64|tf32]"
            << " [--warmup N] [--repeats N]"
            << std::endl;
        return 0;
    }

    nlohmann::json output;
    output["input_path"] = input_path;
    output["requested_backend"] = backend;
    output["precision"] = precision;
    output["warmup"] = warmup;
    output["repeats"] = repeats;

    try {
        const std::string normalized_backend = to_lower(backend);
        const std::string normalized_precision = to_lower(precision);
        const EquationProblem problem = parse_equation_file(input_path);
        const ProblemMetadata metadata = extract_problem_metadata(std::filesystem::path(input_path));

        TconT::TCEquation equation;
        equation.modeC = problem.modeC;
        equation.modeA = problem.modeA;
        equation.modeB = problem.modeB;
        equation.extent = problem.extent;
        equation.op = '+';
        equation.scalar_type = parse_scalar_type(normalized_precision);
        equation.backend_selection = parse_backend_selection(normalized_backend);

        if (equation.backend_selection == TconT::BackendSelection::TTGT_CUTT &&
            equation.scalar_type != TconT::ScalarType::Float64) {
            throw std::invalid_argument("TTGT currently supports fp64 only");
        }

        output["problem_metadata"] = {
            {"input_path", metadata.input_path},
            {"source_collection", metadata.source_collection},
            {"equation_group", metadata.equation_group},
            {"problem_file", metadata.problem_file},
            {"problem_id", metadata.problem_id},
        };
        output["equation_features"] = build_equation_features(problem);

        if (normalized_backend == "direct" || normalized_backend == "cogent") {
            output["plan_features"] = extract_cogent_plan_features(equation);
        } else if (normalized_backend == "ttgt" || normalized_backend == "ttgt_cutt") {
            output["plan_features"] = extract_ttgt_plan_features(equation);
        }

        const TconT::ExecutionPlan plan = TconT::plan(equation);
        const TconT::ExecutionRun run = TconT::prepare(plan);
        const TconT::RunOptions options{warmup, repeats};
        const TconT::ContractionResult result = benchmark_run(run, equation, options);
        const TconT::ExecutionStageTimes stage_times = TconT::stage_times(run);

        output["status"] = "OK";
        output["avg_ms"] = result.avg_ms;
        output["gflops"] = result.gflops;
        output["backend_name"] = TconT::backend_name(plan.backend);
        output["resolved_backend"] = TconT::backend_name(plan.backend);
        output["operation_count"] = output["equation_features"]["operation_count"];
        output["stage_times"] = {
            {"available", stage_times.available},
            {"total_ms", stage_times.total_ms},
            {"transpose_a_ms", stage_times.transpose_a_ms},
            {"transpose_b_ms", stage_times.transpose_b_ms},
            {"gemm_ms", stage_times.gemm_ms},
            {"transpose_c_ms", stage_times.transpose_c_ms},
        };
    } catch (const std::exception& ex) {
        output["status"] = "ERROR";
        output["error_message"] = ex.what();
    } catch (...) {
        output["status"] = "ERROR";
        output["error_message"] = "unknown exception";
    }

    std::cout << output.dump() << std::endl;
    return 0;
}
