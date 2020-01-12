#include "./declaration.hpp"

#include <Eigen/Core>

#include <array>
#include <functional>
#include <numeric>


namespace RobotIntelligence
{

template <class Scalar, size_t channels, int in_rows, int in_cols, int out_rows>
EigenSTLVector<std::array<Eigen::Array<Scalar, out_rows, in_cols>, channels>>
linear(
    EigenSTLVector<std::array<Eigen::Array<Scalar, in_rows, in_cols>, channels>> const& input,
    Eigen::Array<Scalar, out_rows, in_rows> const& weight,
    Eigen::Array<Scalar, out_rows, in_cols> const& bias)
{
    auto const batches = input.size();

    EigenSTLVector<std::array<Eigen::Array<Scalar, out_rows, in_cols>, channels>> output(batches);

    for (size_t batch_idx = 0; batch_idx < batches; batch_idx++) {
        auto& out_batch = output.at(batch_idx);
        auto const& in_batch = input.at(batch_idx);

        for (size_t channel_idx = 0; channel_idx < channels; channel_idx++) {
            auto& out_channel = out_batch.at(channel_idx);
            auto const& in_channel = in_batch.at(channel_idx);

            out_channel = (weight.matrix() * in_channel.matrix()).array() + bias;
        }
    }

    return output;
}

template <class Scalar, size_t channels, int out_rows, int in_cols, int in_rows>
EigenSTLVector<std::array<Eigen::Array<Scalar, in_rows, in_cols>, channels>>
linear_backpropagation(
    EigenSTLVector<std::array<Eigen::Array<Scalar, out_rows, in_cols>, channels>> const& output,
    Eigen::Array<Scalar, out_rows, in_rows> const& weight)
{
    auto const batches = output.size();

    EigenSTLVector<std::array<Eigen::Array<Scalar, in_rows, in_cols>, channels>> input(batches);

    for (size_t batch_idx = 0; batch_idx < batches; batch_idx++) {
        auto& in_batch = input.at(batch_idx);
        auto const& out_batch = output.at(batch_idx);

        for (size_t channel_idx = 0; channel_idx < channels; channel_idx++) {
            auto& in_channel = in_batch.at(channel_idx);
            auto const& out_channel = out_batch.at(channel_idx);

            in_channel = (weight.matrix().transpose() * out_channel.matrix()).array();
        }
    }

    return input;
}

template <class Scalar, size_t channels, int in_rows, int in_cols, int out_rows>
Eigen::Array<Scalar, out_rows, in_rows>
linear_weight_grad(
    EigenSTLVector<std::array<Eigen::Array<Scalar, in_rows, in_cols>, channels>> const& input,
    EigenSTLVector<std::array<Eigen::Array<Scalar, out_rows, in_cols>, channels>> const& out_grad)
{
    auto const batches = input.size();
    assert(batches == out_grad.size());

    return std::inner_product(
               input.begin(), input.end(),
               out_grad.begin(),
               Eigen::Array<Scalar, out_rows, in_rows>::Zero().eval(),
               std::plus<>{},
               [](auto const& in_batch, auto const& out_batch) {
                   return std::inner_product(
                       in_batch.begin(), in_batch.end(),
                       out_batch.begin(),
                       Eigen::Array<Scalar, out_rows, in_rows>::Zero().eval(),
                       std::plus<>{},
                       [](auto const& in_channel, auto const& out_channel) {
                           return (out_channel.matrix() * in_channel.transpose().matrix()).array().eval();
                       });
               })
           / static_cast<Scalar>(batches);
}
template <class Scalar, size_t channels, int rows, int cols>
Eigen::Array<Scalar, rows, cols>
linear_bias_grad(EigenSTLVector<std::array<Eigen::Array<Scalar, rows, cols>, channels>> const& out_grad)
{
    return std::accumulate(
               out_grad.begin(), out_grad.end(),
               Eigen::Array<Scalar, rows, cols>::Zero().eval(),
               [](auto const& tmp, auto const& batch) {
                   return (tmp
                           + std::accumulate(
                                 batch.begin(), batch.end(),
                                 Eigen::Array<Scalar, rows, cols>::Zero().eval()))
                       .eval();
               })
           / static_cast<Scalar>(out_grad.size());
}

}  // namespace RobotIntelligence
