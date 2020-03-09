#include "./cnn.hpp"

#include "./convolution2d.hpp"
#include "./cross_entropy_loss.hpp"
#include "./linear.hpp"
#include "./relu.hpp"
#include "./reshape.hpp"

#include <fstream>
#include <type_traits>


namespace RobotIntelligence
{


void CNN::set_random()
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


    for (auto& out_channel : conv1_weight_momentum) {
        for (auto& channel : out_channel) {
            channel.setZero();
        }
    }
    conv1_bias_momentum.setZero();

    for (auto& out_channel : conv2_weight_momentum) {
        for (auto& channel : out_channel) {
            channel.setZero();
        }
    }
    conv2_bias_momentum.setZero();

    linear_weight_momentum.setZero();
}


CNN::Scalar CNN::train(
    EigenSTLVector<std::array<Eigen::Array<Scalar, 28, 28>, 1>> const& input,
    EigenSTLVector<std::array<Eigen::Array<Scalar, 10, 1>, 1>> const& output,
    Scalar const& learning_rate, Scalar const& momentum_rate)
{
    auto conv2_input = convolution2D<2, 2>(input, conv1_weight, conv1_bias);
    ReLu_inplace(conv2_input);
    auto linear_input = asVector(convolution2D<2, 2>(conv2_input, conv2_weight, conv2_bias));
    ReLu_inplace(linear_input);
    auto loss_input = linear(linear_input, linear_weight, linear_bias);

    Scalar loss;
    auto linear_output_grad = cross_entropy_loss_backpropagation(loss_input, output, loss);

    auto conv2_output_grad = asRect<7, 7>(positiveMask(linear_backpropagation(linear_output_grad, linear_weight), linear_input));
    auto conv1_output_grad = positiveMask(convolution2D_backpropagation<2, 2, 1>(conv2_output_grad, conv2_weight), conv2_input);

    convolution2D_update(conv1_weight, conv1_bias,
        conv1_weight_momentum, conv1_bias_momentum,
        convolution2D_weight_grad<2, 2>(input, conv1_weight, conv1_output_grad), convolution2D_bias_grad(conv1_output_grad),
        learning_rate, momentum_rate);
    convolution2D_update(
        conv2_weight, conv2_bias,
        conv2_weight_momentum, conv2_bias_momentum,
        convolution2D_weight_grad<2, 2>(conv2_input, conv2_weight, conv2_output_grad), convolution2D_bias_grad(conv2_output_grad),
        learning_rate, momentum_rate);
    linear_weight_momentum *= momentum_rate;
    linear_weight_momentum += linear_weight_grad(linear_input, linear_output_grad);
    linear_weight -= learning_rate * linear_weight_momentum;

    return loss;
}

size_t CNN::test(
    EigenSTLVector<std::array<Eigen::Array<Scalar, 28, 28>, 1>> const& input,
    std::vector<uint8_t> const& answer) const
{
    auto const batches = input.size();
    assert(batches == answer.size());

    auto loss_input = linear(
        ReLu(
            asVector(convolution2D<2, 2>(
                ReLu(
                    convolution2D<2, 2>(input, conv1_weight, conv1_bias)),
                conv2_weight, conv2_bias))),
        linear_weight, linear_bias);

    size_t correct_count = 0;
    for (size_t batch_idx = 0; batch_idx < batches; batch_idx++) {
        Eigen::Index max_idx;

        loss_input.at(batch_idx).at(0).maxCoeff(&max_idx);
        correct_count += (max_idx == answer.at(batch_idx));
    }

    return correct_count;
}


void CNN::load(std::filesystem::path const& path)
{
    std::ifstream ifs{path, std::ios_base::binary};

    for (auto& out_channel : conv1_weight) {
        for (auto& channel : out_channel) {
            ifs.read(reinterpret_cast<char*>(channel.data()), sizeof(Scalar) * std::decay_t<decltype(channel)>::SizeAtCompileTime);
        }
    }
    ifs.read(reinterpret_cast<char*>(conv1_bias.data()), sizeof(Scalar) * std::decay_t<decltype(conv1_bias)>::SizeAtCompileTime);

    for (auto& out_channel : conv2_weight) {
        for (auto& channel : out_channel) {
            ifs.read(reinterpret_cast<char*>(channel.data()), sizeof(Scalar) * std::decay_t<decltype(channel)>::SizeAtCompileTime);
        }
    }
    ifs.read(reinterpret_cast<char*>(conv2_bias.data()), sizeof(Scalar) * std::decay_t<decltype(conv2_bias)>::SizeAtCompileTime);

    ifs.read(reinterpret_cast<char*>(linear_weight.data()), sizeof(Scalar) * std::decay_t<decltype(linear_weight)>::SizeAtCompileTime);


    for (auto& out_channel : conv1_weight_momentum) {
        for (auto& channel : out_channel) {
            ifs.read(reinterpret_cast<char*>(channel.data()), sizeof(Scalar) * std::decay_t<decltype(channel)>::SizeAtCompileTime);
        }
    }
    ifs.read(reinterpret_cast<char*>(conv1_bias_momentum.data()), sizeof(Scalar) * std::decay_t<decltype(conv1_bias_momentum)>::SizeAtCompileTime);

    for (auto& out_channel : conv2_weight_momentum) {
        for (auto& channel : out_channel) {
            ifs.read(reinterpret_cast<char*>(channel.data()), sizeof(Scalar) * std::decay_t<decltype(channel)>::SizeAtCompileTime);
        }
    }
    ifs.read(reinterpret_cast<char*>(conv2_bias_momentum.data()), sizeof(Scalar) * std::decay_t<decltype(conv2_bias_momentum)>::SizeAtCompileTime);

    ifs.read(reinterpret_cast<char*>(linear_weight_momentum.data()), sizeof(Scalar) * std::decay_t<decltype(linear_weight_momentum)>::SizeAtCompileTime);
}

void CNN::save(std::filesystem::path const& path) const
{
    std::ofstream ofs{path, std::ios_base::binary};

    for (auto const& out_channel : conv1_weight) {
        for (auto const& channel : out_channel) {
            ofs.write(reinterpret_cast<const char*>(channel.data()), sizeof(Scalar) * std::decay_t<decltype(channel)>::SizeAtCompileTime);
        }
    }
    ofs.write(reinterpret_cast<const char*>(conv1_bias.data()), sizeof(Scalar) * std::decay_t<decltype(conv1_bias)>::SizeAtCompileTime);

    for (auto const& out_channel : conv2_weight) {
        for (auto const& channel : out_channel) {
            ofs.write(reinterpret_cast<const char*>(channel.data()), sizeof(Scalar) * std::decay_t<decltype(channel)>::SizeAtCompileTime);
        }
    }
    ofs.write(reinterpret_cast<const char*>(conv2_bias.data()), sizeof(Scalar) * std::decay_t<decltype(conv2_bias)>::SizeAtCompileTime);

    ofs.write(reinterpret_cast<const char*>(linear_weight.data()), sizeof(Scalar) * std::decay_t<decltype(linear_weight)>::SizeAtCompileTime);


    for (auto const& out_channel : conv1_weight_momentum) {
        for (auto const& channel : out_channel) {
            ofs.write(reinterpret_cast<const char*>(channel.data()), sizeof(Scalar) * std::decay_t<decltype(channel)>::SizeAtCompileTime);
        }
    }
    ofs.write(reinterpret_cast<const char*>(conv1_bias_momentum.data()), sizeof(Scalar) * std::decay_t<decltype(conv1_bias_momentum)>::SizeAtCompileTime);

    for (auto const& out_channel : conv2_weight_momentum) {
        for (auto const& channel : out_channel) {
            ofs.write(reinterpret_cast<const char*>(channel.data()), sizeof(Scalar) * std::decay_t<decltype(channel)>::SizeAtCompileTime);
        }
    }
    ofs.write(reinterpret_cast<const char*>(conv2_bias_momentum.data()), sizeof(Scalar) * std::decay_t<decltype(conv2_bias_momentum)>::SizeAtCompileTime);

    ofs.write(reinterpret_cast<const char*>(linear_weight_momentum.data()), sizeof(Scalar) * std::decay_t<decltype(linear_weight_momentum)>::SizeAtCompileTime);
}

}  // namespace RobotIntelligence
