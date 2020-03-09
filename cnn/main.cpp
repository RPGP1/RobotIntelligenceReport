#include "./cnn.hpp"
#include "./declaration.hpp"

#include <cmdline.h>

#include <Eigen/Core>

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>


void swap_endian(uint32_t& val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    val = (val << 16) | (val >> 16);
};

uint32_t image_file_check(std::istream& ifs)
{
    uint32_t magic, image_number, rows, cols;

    ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    ifs.read(reinterpret_cast<char*>(&image_number), sizeof(image_number));
    ifs.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    ifs.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    swap_endian(magic);
    swap_endian(image_number);
    swap_endian(rows);
    swap_endian(cols);

    assert(magic == 2051);
    assert(rows == 28);
    assert(cols == 28);

    return image_number;
}

uint32_t label_file_check(std::istream& ifs)
{
    uint32_t magic, image_number;

    ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    ifs.read(reinterpret_cast<char*>(&image_number), sizeof(image_number));
    swap_endian(magic);
    swap_endian(image_number);

    assert(magic == 2049);

    return image_number;
}


int main(int argc, char* argv[])
{
    // commmand line options --------------------
    cmdline::parser parser;
    parser.add<std::string>("data", 'd', "directory containing mnist data");
    parser.add<std::string>("output", 'o', "directory to output model parameters and log", false, "result");
    parser.add<unsigned int>("batch", 'b', "batch size", false, 100);
    parser.add<unsigned int>("epoch", 'e', "the number of epochs", false, 100);
    parser.add<unsigned int>("save_interval", 'i', "interval of saving model parameters (specified by epoch number)", false, 1);
    parser.add<unsigned int>("model", 'm', "model parameters to load (specified by epoch number)", false, 0);
    parser.add<double>("learning_rate", 'l', "learning rate", false, 0.001, cmdline::range(0.0, 1.0));
    parser.add<double>("momentum_rate", 0, "momentum rate", false, 0.9, cmdline::range(0.0, 1.0));
    parser.add<double>("noise_rate", 'n', "noise rate", false, 0.0, cmdline::range(0.0, 1.0));
    parser.add("help", 'h', "show this description");

    if (!parser.parse(argc, argv) || parser.exist("help")) {
        std::cout << parser.error_full() << parser.usage();
        return EXIT_FAILURE;
    }


    // mnist data --------------------
    const std::filesystem::path data_dir{parser.get<std::string>("data")};

    std::ifstream stream_train_images{data_dir / "train-images-idx3-ubyte", std::ios_base::binary};
    std::ifstream stream_train_labels{data_dir / "train-labels-idx1-ubyte", std::ios_base::binary};
    std::ifstream stream_test_images{data_dir / "t10k-images-idx3-ubyte", std::ios_base::binary};
    std::ifstream stream_test_labels{data_dir / "t10k-labels-idx1-ubyte", std::ios_base::binary};

    const uint32_t train_image_number{image_file_check(stream_train_images)};
    const uint32_t test_image_number{image_file_check(stream_test_images)};
    {
        [[maybe_unused]] const uint32_t train_image_number_check { label_file_check(stream_train_labels) };
        [[maybe_unused]] const uint32_t test_image_number_check { label_file_check(stream_test_labels) };
        assert(train_image_number == train_image_number_check);
        assert(test_image_number == test_image_number_check);
    }

    const auto header_train_images = static_cast<unsigned int>(stream_train_images.tellg());
    const auto header_train_labels = static_cast<unsigned int>(stream_train_labels.tellg());
    const auto header_test_images = static_cast<unsigned int>(stream_test_images.tellg());
    const auto header_test_labels = static_cast<unsigned int>(stream_test_labels.tellg());

    constexpr auto image_seek = 28 * 28;
    constexpr auto label_seek = 1;


    // output --------------------
    const std::filesystem::path output_dir{parser.get<std::string>("output")};
    const auto output_model_dir = output_dir / "model";
    std::filesystem::create_directories(output_model_dir);
    std::ofstream output_log{output_dir / "test.log", std::ios_base::binary | std::ios_base::app};


    // train --------------------
    using namespace RobotIntelligence;

    CNN cnn;

    auto const epoch_begin = parser.get<unsigned int>("model");
    auto const epoch_end = epoch_begin + parser.get<unsigned int>("epoch");
    if (epoch_begin == 0) {
        cnn.set_random();
    } else {
        cnn.load(output_model_dir / (std::to_string(epoch_begin) + ".dat"));
    }

    auto const save_interval = parser.get<unsigned int>("save_interval");
    auto const batches = parser.get<unsigned int>("batch");
    auto const learning_rate = parser.get<double>("learning_rate");
    auto const momentum_rate = parser.get<double>("momentum_rate");
    auto const noise_rate = parser.get<double>("noise_rate");

    EigenSTLVector<std::array<Eigen::Array<double, 28, 28>, 1>> input;
    EigenSTLVector<std::array<Eigen::Array<double, 10, 1>, 1>> answer;
    std::vector<uint8_t> test_answer;

    std::default_random_engine engine;
    std::uniform_real_distribution noise_flag{-noise_rate, 1 - noise_rate};
    std::uniform_int_distribution noise_value{0, 255};
    std::vector<unsigned int> train_image_indices(train_image_number);
    std::vector<unsigned int> test_image_indices(test_image_number);
    std::iota(train_image_indices.begin(), train_image_indices.end(), (unsigned int){0});
    std::iota(test_image_indices.begin(), test_image_indices.end(), (unsigned int){0});

    auto epoch = epoch_begin;
    for (; epoch < epoch_end; epoch++) {
        if (epoch % save_interval == 0 && epoch != epoch_begin) {
            cnn.save(output_model_dir / (std::to_string(epoch) + ".dat"));
        }


        {
            std::shuffle(train_image_indices.begin(), train_image_indices.end(), engine);

            unsigned int image_count = 0;
            for (; image_count < train_image_number;) {
                auto this_batches = std::min({batches, train_image_number - image_count});
                input.resize(this_batches);
                answer.resize(this_batches);

                for (unsigned int batch_idx = 0; batch_idx < this_batches; batch_idx++) {
                    auto& in_channel = input.at(batch_idx).at(0);
                    auto& ans_channel = answer.at(batch_idx).at(0);

                    stream_train_images.seekg(header_train_images + image_seek * train_image_indices.at(image_count + batch_idx));
                    stream_train_labels.seekg(header_train_labels + label_seek * train_image_indices.at(image_count + batch_idx));

                    for (auto pixel_idx = 0; pixel_idx < image_seek; pixel_idx++) {
                        in_channel.data()[pixel_idx] = static_cast<double>(stream_train_images.get());

                        if (noise_flag(engine) < 0) {
                            in_channel.data()[pixel_idx] = noise_value(engine);
                        }
                    }
                    in_channel /= 255;

                    auto label = stream_train_labels.get();
                    ans_channel.setZero();
                    ans_channel(label) = 1;
                }

                auto loss = cnn.train(input, answer, learning_rate, momentum_rate);

                image_count += this_batches;

                std::cout << "epoch " << epoch + 1 << ": [" << image_count << '/' << train_image_number << ']'
                          << " loss = " << loss << std::endl;
            }
        }


        {
            std::shuffle(test_image_indices.begin(), test_image_indices.end(), engine);

            unsigned int image_count = 0;
            size_t correct_count = 0;
            for (; image_count < test_image_number;) {
                auto this_batches = std::min({batches, test_image_number - image_count});
                input.resize(this_batches);
                test_answer.resize(this_batches);

                for (unsigned int batch_idx = 0; batch_idx < this_batches; batch_idx++) {
                    auto& in_channel = input.at(batch_idx).at(0);
                    auto& ans_batch = test_answer.at(batch_idx);

                    stream_test_images.seekg(header_test_images + image_seek * test_image_indices.at(image_count + batch_idx));
                    stream_test_labels.seekg(header_test_labels + label_seek * test_image_indices.at(image_count + batch_idx));

                    for (auto pixel_idx = 0; pixel_idx < image_seek; pixel_idx++) {
                        in_channel.data()[pixel_idx] = static_cast<double>(stream_test_images.get());

                        if (noise_flag(engine) < 0) {
                            in_channel.data()[pixel_idx] = noise_value(engine);
                        }
                    }
                    in_channel /= 255;

                    ans_batch = static_cast<uint8_t>(stream_test_labels.get());
                }

                correct_count += cnn.test(input, test_answer);

                image_count += this_batches;
            }

            auto rate = static_cast<double>(correct_count) / static_cast<double>(test_image_number);
            std::cout << "epoch " << epoch + 1 << ": TEST " << rate << std::endl;
            output_log << epoch + 1 << '\t' << rate << std::endl;
        }
    }
    cnn.save(output_model_dir / (std::to_string(epoch) + ".dat"));


    return 0;
}
