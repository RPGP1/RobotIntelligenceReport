#pragma once

#include "./declaration.hpp"

#include <Eigen/Core>

#include <array>
#include <filesystem>
#include <vector>


namespace RobotIntelligence
{


struct CNN {
    using Scalar = double;

    CNN() = default;

    void set_random();

    Scalar train(
        EigenSTLVector<std::array<Eigen::Array<Scalar, 28, 28>, 1>> const& input,
        EigenSTLVector<std::array<Eigen::Array<Scalar, 10, 1>, 1>> const& output,
        Scalar const& learning_rate, Scalar const& momentum_rate);
    size_t test(
        EigenSTLVector<std::array<Eigen::Array<Scalar, 28, 28>, 1>> const& input,
        std::vector<uint8_t> const& answer) const;

    void load(std::filesystem::path const& path);
    void save(std::filesystem::path const& path) const;


protected:
    std::array<std::array<Eigen::Array<Scalar, 5, 5>, 1>, 5> conv1_weight;
    Eigen::Array<Scalar, 5, 1> conv1_bias;

    std::array<std::array<Eigen::Array<Scalar, 5, 5>, 5>, 10> conv2_weight;
    Eigen::Array<Scalar, 10, 1> conv2_bias;

    Eigen::Array<Scalar, 10, 490> linear_weight;
    inline static const Eigen::Array<Scalar, 10, 1> linear_bias{Eigen::Array<Scalar, 10, 1>::Zero()};


    std::array<std::array<Eigen::Array<Scalar, 5, 5>, 1>, 5> conv1_weight_momentum;
    Eigen::Array<Scalar, 5, 1> conv1_bias_momentum;

    std::array<std::array<Eigen::Array<Scalar, 5, 5>, 5>, 10> conv2_weight_momentum;
    Eigen::Array<Scalar, 10, 1> conv2_bias_momentum;

    Eigen::Array<Scalar, 10, 490> linear_weight_momentum;
};

}  // namespace RobotIntelligence
