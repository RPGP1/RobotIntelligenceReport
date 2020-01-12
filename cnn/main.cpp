#include "./convolution2d.hpp"
#include "./cross_entropy_loss.hpp"
#include "./declaration.hpp"
#include "./linear.hpp"
#include "./relu.hpp"
#include "./reshape.hpp"

#include <Eigen/Core>

#include <array>
#include <limits>

#include <iostream>


struct CNN {
    using Scalar = double;

    CNN()
    {
        constexpr auto conv1_scale = sqrt(1 * 5 * 5);
        for (auto& out_channel : conv1_weight) {
            for (auto& channel : out_channel) {
                channel.setRandom();
                channel /= conv1_scale;
            }
        }
        conv1_bias.setRandom();
        conv1_bias /= conv1_scale;

        constexpr auto conv2_scale = sqrt(5 * 5 * 5);
        for (auto& out_channel : conv2_weight) {
            for (auto& channel : out_channel) {
                channel.setRandom();
                channel /= conv2_scale;
            }
        }
        conv2_bias.setRandom();
        conv2_bias /= conv2_scale;

        constexpr auto linear_scale = sqrt(490);
        linear_weight.setRandom();
        linear_weight /= linear_scale;
    }

    Scalar train(
        RobotIntelligence::EigenSTLVector<std::array<Eigen::Array<Scalar, 28, 28>, 1>> const& input,
        RobotIntelligence::EigenSTLVector<std::array<Eigen::Array<Scalar, 10, 1>, 1>> const& output,
        Scalar const& learning_rate)
    {
        using namespace RobotIntelligence;

        auto conv2_input = convolution2D<2, 2>(input, conv1_weight, conv1_bias);
        ReLu_inplace(conv2_input);
        auto linear_input = asVector(convolution2D<2, 2>(conv2_input, conv2_weight, conv2_bias));
        ReLu_inplace(linear_input);
        auto loss_input = linear(linear_input, linear_weight, linear_bias);

        Scalar loss;
        auto linear_output_grad = cross_entropy_loss_backpropagation(loss_input, output, loss);

        auto conv2_output_grad = asRect<7, 7>(positiveMask(linear_backpropagation(linear_output_grad, linear_weight), linear_input));
        auto conv1_output_grad = positiveMask(convolution2D_backpropagation<2, 2, 1>(conv2_output_grad, conv2_weight), conv2_input);

        convolution2D_update(conv1_weight, conv1_bias, convolution2D_weight_grad<2, 2>(input, conv1_weight, conv1_output_grad), convolution2D_bias_grad(conv1_output_grad), learning_rate);
        convolution2D_update(conv2_weight, conv2_bias, convolution2D_weight_grad<2, 2>(conv2_input, conv2_weight, conv2_output_grad), convolution2D_bias_grad(conv2_output_grad), learning_rate);
        linear_weight -= learning_rate * linear_weight_grad(linear_input, linear_output_grad);

        return loss;
    }

private:
    std::array<std::array<Eigen::Array<Scalar, 5, 5>, 1>, 5> conv1_weight;
    Eigen::Array<Scalar, 5, 1> conv1_bias;

    std::array<std::array<Eigen::Array<Scalar, 5, 5>, 5>, 10> conv2_weight;
    Eigen::Array<Scalar, 10, 1> conv2_bias;

    Eigen::Array<Scalar, 10, 490> linear_weight;
    const Eigen::Array<Scalar, 10, 1> linear_bias{Eigen::Array<Scalar, 10, 1>::Zero()};
};


int main()
{
    using namespace RobotIntelligence;

    CNN cnn;

    constexpr auto data_size = 60000;

    EigenSTLVector<std::array<
        Eigen::Array<double, 28, 28>,
        1>>
        input(data_size);
    {
        for (auto& in_batch : input) {
            in_batch.at(0).setRandom();
        }
    }

    EigenSTLVector<std::array<
        Eigen::Array<double, 10, 1>,
        1>>
        output(data_size);
    {
        for (auto batch_idx = 0; batch_idx < data_size; batch_idx++) {
            output.at(batch_idx).at(0).setZero();
            output.at(batch_idx).at(0)(batch_idx % 10) = 1;
        }
    }


    constexpr auto batches = 100;

    EigenSTLVector<std::array<
        Eigen::Array<double, 28, 28>,
        1>>
        in_batch(batches);
    EigenSTLVector<std::array<
        Eigen::Array<double, 10, 1>,
        1>>
        out_batch(batches);

    for (auto epoch = 0; epoch < 100; epoch++) {
        double loss = 0;

        for (auto i = 0; i < 600; i++) {
            std::copy(
                input.begin() + batches * i,
                input.begin() + batches * (i + 1),
                in_batch.begin());
            std::copy(
                output.begin() + batches * i,
                output.begin() + batches * (i + 1),
                out_batch.begin());

            loss = cnn.train(in_batch, out_batch, 0.01);
        }

        std::cout << loss << std::endl;
    }


    return 0;
}
