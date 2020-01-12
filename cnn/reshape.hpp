#include "./declaration.hpp"

#include <Eigen/Core>

#include <array>


namespace RobotIntelligence
{

template <class Scalar, size_t channels, int rows, int cols>
EigenSTLVector<std::array<Eigen::Array<Scalar, static_cast<int>(channels* rows* cols), 1>, 1>>
asVector(EigenSTLVector<std::array<Eigen::Array<Scalar, rows, cols>, channels>> const& input)
{
    constexpr auto channel_size = rows * cols;
    constexpr auto size = static_cast<int>(channels * rows * cols);

    auto const batches = input.size();

    EigenSTLVector<std::array<Eigen::Array<Scalar, size, 1>, 1>> output(batches);

    for (size_t batch_idx = 0; batch_idx < batches; batch_idx++) {
        auto& out_batch = output.at(batch_idx);
        auto const& in_batch = input.at(batch_idx);

        for (size_t channel_idx = 0; channel_idx < channels; channel_idx++) {
            out_batch.front().template segment<channel_size>(channel_idx * channel_size)
                = Eigen::Map<const Eigen::Array<Scalar, channel_size, 1>>{in_batch.at(channel_idx).data()};
        }
    }

    return output;
}

template <int rows, int cols, class Scalar, int size>
EigenSTLVector<std::array<Eigen::Array<Scalar, rows, cols>, static_cast<size_t>(size / rows / cols)>>
    asRect(EigenSTLVector<std::array<Eigen::Array<Scalar, size, 1>, 1>> const& input)
{
    static_assert(size % (rows * cols) == 0);

    constexpr auto channel_size = rows * cols;
    constexpr auto channels = static_cast<size_t>(size / rows / cols);

    auto const batches = input.size();

    EigenSTLVector<std::array<Eigen::Array<Scalar, rows, cols>, channels>> output(batches);

    for (size_t batch_idx = 0; batch_idx < batches; batch_idx++) {
        auto& out_batch = output.at(batch_idx);
        auto const& in_batch = input.at(batch_idx);

        for (size_t channel_idx = 0; channel_idx < channels; channel_idx++) {
            Eigen::Map<Eigen::Array<Scalar, channel_size, 1>>{out_batch.at(channel_idx).data()}
            = in_batch.front().template segment<channel_size>(channel_idx * channel_size);
        }
    }

    return output;
}

}  // namespace RobotIntelligence
