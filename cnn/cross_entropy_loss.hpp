#include "./declaration.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iterator>
#include <numeric>

#include <iostream>


namespace RobotIntelligence
{

template <class Scalar, size_t channels, int rows, int cols>
Scalar cross_entropy_loss(EigenSTLVector<std::array<Eigen::Array<Scalar, rows, cols>, channels>> const& input,
    EigenSTLVector<std::array<Eigen::Array<Scalar, rows, cols>, channels>> const& answer)
{
    assert(input.size() == answer.size());

    return std::inner_product(
        input.begin(), input.end(),
        answer.begin(),
        Scalar{0},
        std::plus<>{},
        [](auto const& in_batch, auto const& ans_batch) {
            return std::inner_product(
                in_batch.begin(), in_batch.end(),
                ans_batch.begin(),
                Scalar{0},
                std::plus<>{},
                [](auto const& in_channel, auto const& ans_channel) {
                    auto tmp = (in_channel - in_channel.maxCoeff()).eval();
                    return -((tmp - log(tmp.exp().sum())) * ans_channel).sum();
                });
        });
}

template <class Scalar, size_t channels, int rows, int cols>
EigenSTLVector<std::array<Eigen::Array<Scalar, rows, cols>, channels>>
cross_entropy_loss_backpropagation(
    EigenSTLVector<std::array<Eigen::Array<Scalar, rows, cols>, channels>> const& input,
    EigenSTLVector<std::array<Eigen::Array<Scalar, rows, cols>, channels>> const& answer,
    Scalar& loss)
{
    auto const batches = input.size();
    assert(batches == answer.size());

    EigenSTLVector<std::array<Eigen::Array<Scalar, rows, cols>, channels>> grad;
    grad.reserve(batches);
    loss = Scalar{0};

    std::transform(
        input.begin(), input.end(),
        answer.begin(),
        std::back_inserter(grad),
        [&](auto const& in_batch, auto const& ans_batch) {
            std::array<Eigen::Array<Scalar, rows, cols>, channels> grad;

            std::transform(
                in_batch.begin(), in_batch.end(),
                ans_batch.begin(),
                grad.begin(),
                [&](auto const& in_channel, auto const& ans_channel) {
                    auto tmp = (in_channel - in_channel.maxCoeff()).exp().eval();
                    tmp /= tmp.sum();

                    loss -= (tmp.log() * ans_channel).sum();

                    return (tmp - ans_channel).eval();
                });

            return grad;
        });

    return grad;
}

}  // namespace RobotIntelligence
