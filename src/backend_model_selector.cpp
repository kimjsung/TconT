#include "backend_model_selector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace TconT
{
    namespace
    {
        struct TreeNode {
            double value = 0.0;
            int feature_idx = -1;
            double num_threshold = 0.0;
            bool missing_go_to_left = false;
            int left = -1;
            int right = -1;
            bool is_leaf = false;
        };

        struct ExportedClassifierModel {
            std::vector<std::string> feature_names;
            std::vector<std::vector<TreeNode>> trees;
            double baseline_prediction = 0.0;
            std::string negative_backend = "cogent";
            std::string nonnegative_backend = "ttgt";
        };

        struct CachedExportedModel {
            ExportedClassifierModel model;
            double load_ms = 0.0;
        };

        ExportedClassifierModel load_exported_model_from_json(const std::filesystem::path& path)
        {
            std::ifstream input(path);
            if (!input) {
                throw std::runtime_error("Failed to open exported model JSON: " + path.string());
            }

            const nlohmann::json payload = nlohmann::json::parse(input);
            ExportedClassifierModel model;
            model.feature_names = payload.at("feature_names").get<std::vector<std::string>>();
            model.baseline_prediction = payload.at("baseline_prediction").get<double>();
            model.negative_backend = payload.value("selected_backend_if_score_negative", std::string("cogent"));
            model.nonnegative_backend = payload.value("selected_backend_if_score_nonnegative", std::string("ttgt"));

            for (const auto& tree_json : payload.at("trees")) {
                std::vector<TreeNode> tree;
                tree.reserve(tree_json.size());
                for (const auto& node_json : tree_json) {
                    tree.push_back(
                        TreeNode{
                            .value = node_json.at("value").get<double>(),
                            .feature_idx = node_json.at("feature_idx").get<int>(),
                            .num_threshold = node_json.at("num_threshold").get<double>(),
                            .missing_go_to_left = node_json.at("missing_go_to_left").get<bool>(),
                            .left = node_json.at("left").get<int>(),
                            .right = node_json.at("right").get<int>(),
                            .is_leaf = node_json.at("is_leaf").get<bool>(),
                        });
                }
                model.trees.push_back(std::move(tree));
            }

            return model;
        }

        std::filesystem::path resolve_model_json_path()
        {
            if (const char* env = std::getenv("TCONT_MODEL_JSON"); env != nullptr && env[0] != '\0') {
                return std::filesystem::path(env);
            }

#ifdef TCONT_DEFAULT_MODEL_JSON
            return std::filesystem::path(TCONT_DEFAULT_MODEL_JSON);
#else
            return std::filesystem::path("model/results/backend_selector_paper13/classification_model_export.json");
#endif
        }

        const CachedExportedModel& get_exported_model()
        {
            static const CachedExportedModel cached = [] {
                const auto load_start = std::chrono::steady_clock::now();
                CachedExportedModel result;
                result.model = load_exported_model_from_json(resolve_model_json_path());
                const auto load_end = std::chrono::steady_clock::now();
                result.load_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
                return result;
            }();
            return cached;
        }

        std::unordered_map<char, int64_t> validate_and_copy_extents(const TCEquation& desc)
        {
            std::unordered_map<char, int64_t> extents = desc.extent;
            for (char mode : desc.modeC) {
                if (!extents.contains(mode)) {
                    throw std::invalid_argument(std::string("Missing extent for mode: ") + mode);
                }
            }
            for (char mode : desc.modeA) {
                if (!extents.contains(mode)) {
                    throw std::invalid_argument(std::string("Missing extent for mode: ") + mode);
                }
            }
            for (char mode : desc.modeB) {
                if (!extents.contains(mode)) {
                    throw std::invalid_argument(std::string("Missing extent for mode: ") + mode);
                }
            }
            return extents;
        }

        double product_for_modes(const std::vector<char>& modes, const std::unordered_map<char, int64_t>& extents)
        {
            double total = 1.0;
            for (char mode : modes) {
                total *= static_cast<double>(extents.at(mode));
            }
            return total;
        }

        std::vector<double> canonical_mode_ids(
            const std::vector<char>& mode_c,
            const std::vector<char>& mode_a,
            const std::vector<char>& mode_b,
            const std::vector<char>& target)
        {
            std::unordered_map<char, double> mapping;
            double next_id = 1.0;
            auto assign = [&](char mode) mutable {
                if (!mapping.contains(mode)) {
                    mapping.emplace(mode, next_id);
                    next_id += 1.0;
                }
            };

            for (char mode : mode_c) {
                assign(mode);
            }
            for (char mode : mode_a) {
                assign(mode);
            }
            for (char mode : mode_b) {
                assign(mode);
            }

            std::vector<double> ids;
            ids.reserve(target.size());
            for (char mode : target) {
                ids.push_back(mapping.at(mode));
            }
            return ids;
        }

        double inversion_count(const std::vector<double>& values)
        {
            double total = 0.0;
            for (size_t left = 0; left < values.size(); ++left) {
                for (size_t right = left + 1; right < values.size(); ++right) {
                    if (values[left] > values[right]) {
                        total += 1.0;
                    }
                }
            }
            return total;
        }

        std::vector<double> output_mode_ids(
            const std::vector<char>& target,
            const std::vector<char>& output_modes,
            const std::vector<double>& ids)
        {
            std::vector<double> output_ids;
            output_ids.reserve(ids.size());
            for (size_t index = 0; index < target.size(); ++index) {
                if (std::find(output_modes.begin(), output_modes.end(), target[index]) != output_modes.end()) {
                    output_ids.push_back(ids[index]);
                }
            }
            return output_ids;
        }

        std::unordered_map<std::string, double> build_model_features(const TCEquation& desc)
        {
            const auto extents = validate_and_copy_extents(desc);
            const double size_a = product_for_modes(desc.modeA, extents);
            const double size_b = product_for_modes(desc.modeB, extents);
            const double size_c = product_for_modes(desc.modeC, extents);

            std::vector<char> internal_modes;
            for (char mode : desc.modeA) {
                const bool in_b = std::find(desc.modeB.begin(), desc.modeB.end(), mode) != desc.modeB.end();
                const bool in_c = std::find(desc.modeC.begin(), desc.modeC.end(), mode) != desc.modeC.end();
                if (in_b && !in_c &&
                    std::find(internal_modes.begin(), internal_modes.end(), mode) == internal_modes.end()) {
                    internal_modes.push_back(mode);
                }
            }

            std::vector<char> output_only_left_modes;
            std::vector<char> output_only_right_modes;
            std::vector<char> output_shared_modes;
            for (char mode : desc.modeC) {
                const bool in_a = std::find(desc.modeA.begin(), desc.modeA.end(), mode) != desc.modeA.end();
                const bool in_b = std::find(desc.modeB.begin(), desc.modeB.end(), mode) != desc.modeB.end();
                if (in_a && !in_b) {
                    output_only_left_modes.push_back(mode);
                }
                if (in_b && !in_a) {
                    output_only_right_modes.push_back(mode);
                }
                if (in_a && in_b) {
                    output_shared_modes.push_back(mode);
                }
            }

            double internal_volume = 1.0;
            for (char mode : internal_modes) {
                internal_volume *= static_cast<double>(extents.at(mode));
            }

            double operation_count = 2.0;
            std::vector<double> all_extents;
            all_extents.reserve(extents.size());
            for (const auto& [_, extent] : extents) {
                operation_count *= static_cast<double>(extent);
                all_extents.push_back(static_cast<double>(extent));
            }

            const auto [min_extent_it, max_extent_it] = std::minmax_element(all_extents.begin(), all_extents.end());
            const double min_extent = *min_extent_it;
            const double max_extent = *max_extent_it;
            const double mean_extent =
                std::accumulate(all_extents.begin(), all_extents.end(), 0.0) / static_cast<double>(all_extents.size());
            double variance = 0.0;
            for (double extent : all_extents) {
                const double diff = extent - mean_extent;
                variance += diff * diff;
            }
            const double std_extent = std::sqrt(variance / static_cast<double>(all_extents.size()));

            const std::vector<double> mode_a_ids = canonical_mode_ids(desc.modeC, desc.modeA, desc.modeB, desc.modeA);
            const std::vector<double> mode_b_ids = canonical_mode_ids(desc.modeC, desc.modeA, desc.modeB, desc.modeB);
            const std::vector<double> mode_a_output_ids = output_mode_ids(desc.modeA, desc.modeC, mode_a_ids);
            const std::vector<double> mode_b_output_ids = output_mode_ids(desc.modeB, desc.modeC, mode_b_ids);

            const auto seq_sum = [](const std::vector<double>& values) {
                return std::accumulate(values.begin(), values.end(), 0.0);
            };
            const auto seq_prod = [](const std::vector<double>& values) {
                double product = 1.0;
                for (double value : values) {
                    product *= value;
                }
                return values.empty() ? 0.0 : product;
            };
            const auto seq_first = [](const std::vector<double>& values) {
                return values.empty() ? 0.0 : values.front();
            };
            const auto seq_last = [](const std::vector<double>& values) {
                return values.empty() ? 0.0 : values.back();
            };
            const auto seq_spread = [](const std::vector<double>& values) {
                if (values.empty()) {
                    return 0.0;
                }
                const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
                return *max_it - *min_it;
            };

            const double data_volume = std::max(size_a + size_b + size_c, 1.0);

            std::unordered_map<std::string, double> features;
            features["num_modes_C"] = static_cast<double>(desc.modeC.size());
            features["num_modes_A"] = static_cast<double>(desc.modeA.size());
            features["num_modes_B"] = static_cast<double>(desc.modeB.size());
            features["num_unique_modes"] = static_cast<double>(extents.size());
            features["num_internal_modes"] = static_cast<double>(internal_modes.size());
            features["num_output_modes"] = static_cast<double>(desc.modeC.size());
            features["num_output_only_left_modes"] = static_cast<double>(output_only_left_modes.size());
            features["num_output_only_right_modes"] = static_cast<double>(output_only_right_modes.size());
            features["num_output_shared_modes"] = static_cast<double>(output_shared_modes.size());
            features["size_A"] = size_a;
            features["size_B"] = size_b;
            features["size_C"] = size_c;
            features["output_volume"] = size_c;
            features["internal_volume"] = internal_volume;
            features["operation_count"] = operation_count;
            features["min_extent"] = min_extent;
            features["max_extent"] = max_extent;
            features["mean_extent"] = mean_extent;
            features["std_extent"] = std_extent;
            features["extent_ratio_max_min"] = max_extent / min_extent;
            features["log2_size_A"] = std::log2(size_a);
            features["log2_size_B"] = std::log2(size_b);
            features["log2_size_C"] = std::log2(size_c);
            features["log2_internal_volume"] = std::log2(internal_volume);
            features["log2_size_A_over_B"] = std::log2(size_a / size_b);
            features["log2_size_A_over_C"] = std::log2(size_a / size_c);
            features["log2_size_B_over_C"] = std::log2(size_b / size_c);
            features["log2_output_over_internal"] = std::log2(size_c / internal_volume);
            features["log2_work_over_data"] = std::log2(operation_count / data_volume);
            features["arithmetic_intensity_proxy"] = operation_count / data_volume;
            features["modeA_pattern_id_sum"] = seq_sum(mode_a_ids);
            features["modeA_pattern_id_prod"] = seq_prod(mode_a_ids);
            features["modeA_pattern_id_first"] = seq_first(mode_a_ids);
            features["modeA_pattern_id_last"] = seq_last(mode_a_ids);
            features["modeA_pattern_id_spread"] = seq_spread(mode_a_ids);
            features["modeA_pattern_inversions"] = inversion_count(mode_a_output_ids);
            features["modeB_pattern_id_sum"] = seq_sum(mode_b_ids);
            features["modeB_pattern_id_prod"] = seq_prod(mode_b_ids);
            features["modeB_pattern_id_first"] = seq_first(mode_b_ids);
            features["modeB_pattern_id_last"] = seq_last(mode_b_ids);
            features["modeB_pattern_id_spread"] = seq_spread(mode_b_ids);
            features["modeB_pattern_inversions"] = inversion_count(mode_b_output_ids);

            // Compatibility with older exported paper13 models.
            features["modeA_pattern_pos_1"] = mode_a_ids.size() > 1 ? mode_a_ids[1] : 0.0;
            features["modeA_pattern_pos_2"] = mode_a_ids.size() > 2 ? mode_a_ids[2] : 0.0;
            features["modeB_pattern_pos_2"] = mode_b_ids.size() > 2 ? mode_b_ids[2] : 0.0;
            return features;
        }

        double evaluate_tree(const std::vector<TreeNode>& tree, const std::vector<double>& feature_vector)
        {
            int index = 0;
            while (true) {
                const TreeNode& node = tree.at(static_cast<size_t>(index));
                if (node.is_leaf) {
                    return node.value;
                }

                const double value = feature_vector.at(static_cast<size_t>(node.feature_idx));
                const bool go_left = std::isnan(value) ? node.missing_go_to_left : (value <= node.num_threshold);
                index = go_left ? node.left : node.right;
            }
        }

        Backend backend_from_name(const std::string& backend_name)
        {
            if (backend_name == "cogent") {
                return Backend::COGENT;
            }
            if (backend_name == "ttgt" || backend_name == "ttgt_cutt") {
                return Backend::TTGT_CUTT;
            }
            throw std::runtime_error("Unsupported backend label in exported model: " + backend_name);
        }
    }

    Backend select_backend_by_exported_model(const TCEquation& desc)
    {
        if (desc.scalar_type != ScalarType::Float64) {
            const auto inference_start = std::chrono::steady_clock::now();
            const auto inference_end = std::chrono::steady_clock::now();
            const auto inference_ms =
                std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
            std::cout << "[TconT] ML selector chose: cogent (FP64-only fallback, model_load_ms=0.000, inference_ms="
                      << inference_ms << ")" << std::endl;
            return Backend::COGENT;
        }

        const CachedExportedModel& cached_model = get_exported_model();
        const ExportedClassifierModel& model = cached_model.model;
        const auto feature_map = build_model_features(desc);
        
        const auto inference_start = std::chrono::steady_clock::now();
        std::vector<double> feature_vector;
        feature_vector.reserve(model.feature_names.size());
        for (const std::string& feature_name : model.feature_names) {
            const auto it = feature_map.find(feature_name);
            if (it == feature_map.end()) {
                throw std::runtime_error("Missing feature required by exported model: " + feature_name);
            }
            feature_vector.push_back(it->second);
        }

        double score = model.baseline_prediction;
        for (const auto& tree : model.trees) {
            score += evaluate_tree(tree, feature_vector);
        }

        const std::string selected_name = score < 0.0 ? model.negative_backend : model.nonnegative_backend;
        const auto inference_end = std::chrono::steady_clock::now();
        const auto inference_ms = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
        std::cout << "[TconT] ML selector chose: " << selected_name << " (model_load_ms=" << cached_model.load_ms
                  << ", inference_ms=" << inference_ms << ")" << std::endl;
        return backend_from_name(selected_name);
    }
}
