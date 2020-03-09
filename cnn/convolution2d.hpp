#include "./declaration.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>


namespace RobotIntelligence
{


constexpr int convolution_output_size(int input_size, int kernel_size, int padding, int stride)
{
    return (input_size + padding * 2 - kernel_size) / stride + 1;
}

template <int padding_rows = 0, int stride_rows = 1, int padding_cols = padding_rows, int stride_cols = stride_rows,
    class Scalar = void,
    size_t in_channels = 0, size_t out_channels = 0,
    int in_rows = 0, int in_cols = 0,
    int kernel_rows = 0, int kernel_cols = 0>
EigenSTLVector<
    std::array<
        Eigen::Array<
            Scalar,
            convolution_output_size(in_rows, kernel_rows, padding_rows, stride_rows),
            convolution_output_size(in_cols, kernel_cols, padding_cols, stride_cols)>,
        out_channels>>
convolution2D(
    EigenSTLVector<std::array<Eigen::Array<Scalar, in_rows, in_cols>, in_channels>> const& input,
    std::array<std::array<Eigen::Array<Scalar, kernel_rows, kernel_cols>, in_channels>, out_channels> const& weight,
    Eigen::Array<Scalar, static_cast<int>(out_channels), 1> const& bias)
{
    static_assert(padding_rows >= 0);
    static_assert(padding_cols >= 0);
    static_assert(stride_rows > 0);
    static_assert(stride_cols > 0);
    static_assert(in_rows > 0);
    static_assert(in_cols > 0);
    static_assert(kernel_rows > 0);
    static_assert(kernel_cols > 0);
    static_assert(kernel_rows > padding_rows);
    static_assert(kernel_cols > padding_cols);


    constexpr auto out_rows = convolution_output_size(in_rows, kernel_rows, padding_rows, stride_rows);
    constexpr auto out_cols = convolution_output_size(in_cols, kernel_cols, padding_cols, stride_cols);
    static_assert(out_rows > 0);
    static_assert(out_cols > 0);

    const auto batches = input.size();

    EigenSTLVector<
        std::array<
            Eigen::Array<Scalar, out_rows, out_cols>,
            out_channels>>
        output(batches);


    for (size_t batch_idx = 0; batch_idx < batches; batch_idx++) {
        auto const& in_batch = input.at(batch_idx);
        auto& out_batch = output.at(batch_idx);

        if constexpr (padding_rows == 0 && padding_cols == 0) {
            for (size_t out_channel_idx = 0; out_channel_idx < out_channels; out_channel_idx++) {
                auto& out_channel = out_batch.at(out_channel_idx);
                auto const& weight_channel = weight.at(out_channel_idx);
                auto const& bias_channel = bias(out_channel_idx);

                for (auto col = 0; col < out_cols; col++) {
                    for (auto row = 0; row < out_rows; row++) {
                        out_channel(row, col) = std::inner_product(
                            in_batch.begin(), in_batch.end(),
                            weight_channel.begin(),
                            Scalar{0},
                            std::plus<>{},
                            [&](auto const& left, auto const& right) {
                                return (left.template block<kernel_rows, kernel_cols>(row * stride_rows, col * stride_cols) * right).sum();
                            });
                    }
                }

                out_channel += bias_channel;
            }


        } else {
            std::array<Eigen::Array<Scalar, in_rows + padding_rows * 2, in_cols + padding_cols * 2>, in_channels> in_batch_with_padding;

            for (size_t in_channel_idx = 0; in_channel_idx < in_channels; in_channel_idx++) {
                auto& in_channel_with_padding = in_batch_with_padding.at(in_channel_idx);
                auto& in_channel = in_batch.at(in_channel_idx);

                in_channel_with_padding.template topRows<padding_rows>().setZero();
                in_channel_with_padding.template bottomRows<padding_rows>().setZero();
                in_channel_with_padding.template leftCols<padding_cols>().setZero();
                in_channel_with_padding.template rightCols<padding_cols>().setZero();

                in_channel_with_padding.template block<in_rows, in_cols>(padding_rows, padding_cols) = in_channel;
            }


            for (size_t out_channel_idx = 0; out_channel_idx < out_channels; out_channel_idx++) {
                auto& out_channel = out_batch.at(out_channel_idx);
                auto const& weight_channel = weight.at(out_channel_idx);
                auto const& bias_channel = bias(out_channel_idx);

                for (auto col = 0; col < out_cols; col++) {
                    for (auto row = 0; row < out_rows; row++) {
                        out_channel(row, col) = std::inner_product(
                            in_batch_with_padding.begin(), in_batch_with_padding.end(),
                            weight_channel.begin(),
                            Scalar{0},
                            std::plus<>{},
                            [&](auto const& input, auto const& weight) {
                                return (input.template block<kernel_rows, kernel_cols>(row * stride_rows, col * stride_cols) * weight).sum();
                            });
                    }
                }

                out_channel += bias_channel;
            }
        }
    }


    return output;
}


constexpr int convolution2D_backpropagation_input_size(int output_size, int kernel_size, int padding, int input_padding, int stride)
{
    return (output_size - 1) * stride - padding * 2 + kernel_size + input_padding;
}

template <int padding_rows = 0, int stride_rows = 1, int in_padding_rows = 0,
    int padding_cols = padding_rows, int stride_cols = stride_rows, int in_padding_cols = in_padding_rows,
    class Scalar = void,
    size_t out_channels = 0, size_t in_channels = 0,
    int out_rows = 0, int out_cols = 0,
    int kernel_rows = 0, int kernel_cols = 0>
EigenSTLVector<
    std::array<
        Eigen::Array<
            Scalar,
            convolution2D_backpropagation_input_size(out_rows, kernel_rows, padding_rows, in_padding_rows, stride_rows),
            convolution2D_backpropagation_input_size(out_cols, kernel_cols, padding_cols, in_padding_cols, stride_cols)>,
        in_channels>>
convolution2D_backpropagation(
    EigenSTLVector<std::array<Eigen::Array<Scalar, out_rows, out_cols>, out_channels>> const& output,
    std::array<std::array<Eigen::Array<Scalar, kernel_rows, kernel_cols>, in_channels>, out_channels> const& weight)
{
    static_assert(padding_rows >= 0);
    static_assert(padding_cols >= 0);
    static_assert(in_padding_rows >= 0);
    static_assert(in_padding_cols >= 0);
    static_assert(stride_rows > 0);
    static_assert(stride_cols > 0);
    static_assert(out_rows > 0);
    static_assert(out_cols > 0);
    static_assert(kernel_rows > 0);
    static_assert(kernel_cols > 0);
    static_assert(kernel_rows > padding_rows);
    static_assert(kernel_cols > padding_cols);


    constexpr auto in_rows = convolution2D_backpropagation_input_size(out_rows, kernel_rows, padding_rows, in_padding_rows, stride_rows);
    constexpr auto in_cols = convolution2D_backpropagation_input_size(out_cols, kernel_cols, padding_cols, in_padding_cols, stride_cols);
    static_assert(in_rows > 0);
    static_assert(in_cols > 0);

    const auto batches = output.size();

    EigenSTLVector<
        std::array<
            Eigen::Array<Scalar, in_rows, in_cols>,
            in_channels>>
        input(batches);


    for (size_t batch_idx = 0; batch_idx < batches; batch_idx++) {
        auto const& out_batch = output.at(batch_idx);
        auto& in_batch = input.at(batch_idx);

        if constexpr (padding_rows + 1 == kernel_rows && padding_cols + 1 == kernel_cols
                      && in_padding_rows == 0 && in_padding_cols == 0
                      && stride_rows == 1 && stride_cols == 1) {

            for (size_t in_channel_idx = 0; in_channel_idx < in_channels; in_channel_idx++) {
                auto& in_channel = in_batch.at(in_channel_idx);

                for (auto col = 0; col < in_cols; col++) {
                    for (auto row = 0; row < in_rows; row++) {
                        in_channel(row, col) = std::inner_product(
                            out_batch.begin(), out_batch.end(),
                            weight.begin(),
                            Scalar{0},
                            std::plus<>{},
                            [&](auto const& output, auto const& weight_channel) {
                                return (output.template block<kernel_rows, kernel_cols>(row, col) * weight_channel.at(in_channel_idx).reverse()).sum();
                            });
                    }
                }
            }


        } else {
            constexpr auto out_pad_rows = in_rows + kernel_rows - 1;
            constexpr auto out_pad_cols = in_cols + kernel_cols - 1;

            std::array<Eigen::Array<Scalar, out_pad_rows, out_pad_cols>, out_channels> out_batch_with_padding;

            for (size_t out_channel_idx = 0; out_channel_idx < out_channels; out_channel_idx++) {
                auto& out_channel_with_padding = out_batch_with_padding.at(out_channel_idx);
                auto& out_channel = out_batch.at(out_channel_idx);

                out_channel_with_padding.setZero();

                Eigen::Map<Eigen::Array<Scalar, out_rows, out_cols>, Eigen::Unaligned, Eigen::Stride<out_pad_rows * stride_cols, stride_rows>>{
                    &out_channel_with_padding(kernel_rows - padding_rows - 1, kernel_cols - padding_cols - 1)}
                = out_channel;
            }


            for (size_t in_channel_idx = 0; in_channel_idx < in_channels; in_channel_idx++) {
                auto& in_channel = in_batch.at(in_channel_idx);

                for (auto col = 0; col < in_cols; col++) {
                    for (auto row = 0; row < in_rows; row++) {
                        in_channel(row, col) = std::inner_product(
                            out_batch_with_padding.begin(), out_batch_with_padding.end(),
                            weight.begin(),
                            Scalar{0},
                            std::plus<>{},
                            [&](auto const& output, auto const& weight_channel) {
                                return (output.template block<kernel_rows, kernel_cols>(row, col) * weight_channel.at(in_channel_idx).reverse()).sum();
                            });
                    }
                }
            }
        }
    }


    return input;
}

template <int padding_rows = 0, int stride_rows = 1, int padding_cols = padding_rows, int stride_cols = stride_rows,
    class Scalar = void,
    size_t in_channels = 0, size_t out_channels = 0,
    int in_rows = 0, int in_cols = 0,
    int kernel_rows = 0, int kernel_cols = 0>
std::array<std::array<Eigen::Array<Scalar, kernel_rows, kernel_cols>, in_channels>, out_channels>
convolution2D_weight_grad(
    EigenSTLVector<std::array<Eigen::Array<Scalar, in_rows, in_cols>, in_channels>> const& input,
    std::array<std::array<Eigen::Array<Scalar, kernel_rows, kernel_cols>, in_channels>, out_channels> const&,
    EigenSTLVector<std::array<Eigen::Array<
                                  Scalar,
                                  convolution_output_size(in_rows, kernel_rows, padding_rows, stride_rows),
                                  convolution_output_size(in_cols, kernel_cols, padding_cols, stride_cols)>,
        out_channels>> const& out_grad)
{
    static_assert(padding_rows >= 0);
    static_assert(padding_cols >= 0);
    static_assert(stride_rows > 0);
    static_assert(stride_cols > 0);
    static_assert(in_rows > 0);
    static_assert(in_cols > 0);
    static_assert(kernel_rows > 0);
    static_assert(kernel_cols > 0);
    static_assert(kernel_rows > padding_rows);
    static_assert(kernel_cols > padding_cols);


    constexpr auto out_rows = convolution_output_size(in_rows, kernel_rows, padding_rows, stride_rows);
    constexpr auto out_cols = convolution_output_size(in_cols, kernel_cols, padding_cols, stride_cols);
    static_assert(out_rows > 0);
    static_assert(out_cols > 0);

    const auto batches = input.size();
    assert(batches == out_grad.size());


    std::array<std::array<Eigen::Array<Scalar, kernel_rows, kernel_cols>, in_channels>, out_channels> result;
    for (auto& out_channel : result) {
        for (auto& channel : out_channel) {
            channel.setZero();
        }
    }


    for (size_t batch_idx = 0; batch_idx < batches; batch_idx++) {
        auto const& in_batch = input.at(batch_idx);
        auto const& out_batch = out_grad.at(batch_idx);

        for (size_t out_channel_idx = 0; out_channel_idx < out_channels; out_channel_idx++) {
            auto& result_out_channel = result.at(out_channel_idx);
            auto const& out_channel = out_batch.at(out_channel_idx);

            for (size_t in_channel_idx = 0; in_channel_idx < in_channels; in_channel_idx++) {
                auto& result_channel = result_out_channel.at(in_channel_idx);
                auto const& in_channel = in_batch.at(in_channel_idx);

                if constexpr (padding_rows == 0 && padding_cols == 0) {
                    for (auto col = 0; col < kernel_cols; col++) {
                        for (auto row = 0; row < kernel_rows; row++) {
                            result_channel(row, col) += (Eigen::Map<
                                                             Eigen::Array<Scalar, out_rows, out_cols>,
                                                             Eigen::Unaligned,
                                                             Eigen::Stride<in_rows * stride_cols, stride_rows>>{&in_channel(row, col)}
                                                         * out_channel)
                                                            .sum();
                        }
                    }

                } else {
                    constexpr auto in_pad_rows = in_rows + padding_rows * 2;
                    constexpr auto in_pad_cols = in_cols + padding_cols * 2;

                    Eigen::Array<Scalar, in_pad_rows, in_pad_cols> in_channel_with_padding;

                    in_channel_with_padding.template topRows<padding_rows>().setZero();
                    in_channel_with_padding.template bottomRows<padding_rows>().setZero();
                    in_channel_with_padding.template leftCols<padding_cols>().setZero();
                    in_channel_with_padding.template rightCols<padding_cols>().setZero();

                    in_channel_with_padding.template block<in_rows, in_cols>(padding_rows, padding_cols) = in_channel;

                    for (auto col = 0; col < kernel_cols; col++) {
                        for (auto row = 0; row < kernel_rows; row++) {
                            result_channel(row, col) += (Eigen::Map<
                                                             Eigen::Array<Scalar, out_rows, out_cols>,
                                                             Eigen::Unaligned,
                                                             Eigen::Stride<in_pad_rows * stride_cols, stride_rows>>{&in_channel_with_padding(row, col)}
                                                         * out_channel)
                                                            .sum();
                        }
                    }
                }
            }
        }
    }


    for (auto& out_channel : result) {
        for (auto& channel : out_channel) {
            channel /= static_cast<Scalar>(batches);
        }
    }

    return result;
}
template <class Scalar, size_t channels, int rows, int cols>
Eigen::Array<Scalar, static_cast<int>(channels), 1>
convolution2D_bias_grad(EigenSTLVector<std::array<Eigen::Array<Scalar, rows, cols>, channels>> const& out_grad)
{
    Eigen::Array<Scalar, static_cast<int>(channels), 1> result;
    result.setZero();

    for (auto const& batch : out_grad) {
        for (auto channel_idx = 0; channel_idx < static_cast<int>(channels); channel_idx++) {
            result(channel_idx) += batch.at(channel_idx).sum();
        }
    }

    return result / static_cast<Scalar>(out_grad.size());
}

template <class Scalar,
    size_t in_channels, size_t out_channels,
    int kernel_rows, int kernel_cols>
void convolution2D_update(
    std::array<std::array<Eigen::Array<Scalar, kernel_rows, kernel_cols>, in_channels>, out_channels>& weight,
    Eigen::Array<Scalar, static_cast<int>(out_channels), 1>& bias,
    std::array<std::array<Eigen::Array<Scalar, kernel_rows, kernel_cols>, in_channels>, out_channels> const& weight_grad,
    Eigen::Array<Scalar, static_cast<int>(out_channels), 1> const& bias_grad,
    Scalar const& learning_rate)
{
    for (size_t out_channel_idx = 0; out_channel_idx < out_channels; out_channel_idx++) {
        auto& weight_out_channel = weight.at(out_channel_idx);
        auto const& weight_grad_out_channel = weight_grad.at(out_channel_idx);

        for (size_t in_channel_idx = 0; in_channel_idx < in_channels; in_channel_idx++) {
            weight_out_channel.at(in_channel_idx) -= learning_rate * weight_grad_out_channel.at(in_channel_idx);
        }
    }

    bias -= learning_rate * bias_grad;
}
template <class Scalar,
    size_t in_channels, size_t out_channels,
    int kernel_rows, int kernel_cols>
void convolution2D_update(
    std::array<std::array<Eigen::Array<Scalar, kernel_rows, kernel_cols>, in_channels>, out_channels>& weight,
    Eigen::Array<Scalar, static_cast<int>(out_channels), 1>& bias,
    std::array<std::array<Eigen::Array<Scalar, kernel_rows, kernel_cols>, in_channels>, out_channels>& weight_momentum,
    Eigen::Array<Scalar, static_cast<int>(out_channels), 1>& bias_momentum,
    std::array<std::array<Eigen::Array<Scalar, kernel_rows, kernel_cols>, in_channels>, out_channels> const& weight_grad,
    Eigen::Array<Scalar, static_cast<int>(out_channels), 1> const& bias_grad,
    Scalar const& learning_rate, Scalar const& momentum_rate)
{
    for (size_t out_channel_idx = 0; out_channel_idx < out_channels; out_channel_idx++) {
        auto& weight_out_channel = weight.at(out_channel_idx);
        auto& weight_momentum_out_channel = weight_momentum.at(out_channel_idx);
        auto const& weight_grad_out_channel = weight_grad.at(out_channel_idx);

        for (size_t in_channel_idx = 0; in_channel_idx < in_channels; in_channel_idx++) {
            auto& weight_momentum_channel = weight_momentum_out_channel.at(in_channel_idx);

            weight_momentum_channel *= momentum_rate;
            weight_momentum_channel += weight_grad_out_channel.at(in_channel_idx);
            weight_out_channel.at(in_channel_idx) -= learning_rate * weight_momentum_channel;
        }
    }

    bias_momentum *= momentum_rate;
    bias_momentum += bias_grad;
    bias -= learning_rate * bias_momentum;
}

}  // namespace RobotIntelligence
