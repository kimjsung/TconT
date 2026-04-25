#pragma once
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "tcont.hpp"

static inline void print_indices(const std::vector<char>& modes)
{
    for (auto it = modes.begin(); it != modes.end(); ++it) {
        std::cout << *it;
        if (std::next(it) != modes.end()) {
            std::cout << ",";
        }
    }
}

inline void print_equation(const TconT::TCEquation& eq)
{
    std::cout << "[TconT] Equation: C[";
    print_indices(eq.modeC);
    std::cout << "] ";

    if (eq.op == '-') {
        std::cout << "-= ";
    } else if (eq.op == '+') {
        std::cout << "+= ";
    } else {
        std::cout << eq.op << "= ";
    }

    std::cout << "A[";
    print_indices(eq.modeA);
    std::cout << "] * B[";
    print_indices(eq.modeB);
    std::cout << "]\n";

    std::vector<char> keys;
    for (const auto& kv : eq.extent) {
        keys.push_back(kv.first);
    }
    std::sort(keys.begin(), keys.end());

    std::cout << "[TconT] Extents: ";
    for (size_t i = 0; i < keys.size(); ++i) {
        const char idx = keys[i];
        std::cout << idx << "=" << eq.extent.at(idx);
        if (i + 1 != keys.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "\n";
}

inline int64_t compute_tensor_size(
    const std::vector<char>& modes,
    const std::unordered_map<char, int64_t>& extent)
{
    int64_t size = 1;
    for (char mode : modes) {
        const auto it = extent.find(mode);
        if (it == extent.end()) {
            throw std::runtime_error(std::string("Extent for index '") + mode + "' not found.");
        }
        size *= it->second;
    }
    return size;
}

static inline int extent_of(const TconT::TCEquation& eq, char idx)
{
    const auto it = eq.extent.find(idx);
    if (it == eq.extent.end()) {
        throw std::runtime_error(std::string("Extent for index '") + idx + "' not found.");
    }
    return static_cast<int>(it->second);
}
