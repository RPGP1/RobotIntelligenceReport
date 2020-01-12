#include "./declaration.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <iterator>


namespace RobotIntelligence
{

template <class Scalar, size_t channels, int rows, int cols>
void ReLu_inplace(EigenSTLVector<std::array<Eigen::Array<Scalar, rows, cols>, channels>>& input)
{
    for (auto& batch : input) {
        for (auto& channel : batch) {
            channel = channel.max(0).eval();
        }
    }
}
template <class Scalar, size_t channels, int rows, int cols>
EigenSTLVector<std::array<Eigen::Array<Scalar, rows, cols>, channels>>
ReLu(EigenSTLVector<std::array<Eigen::Array<Scalar, rows, cols>, channels>> const& input)
{
    auto output = input;
    ReLu_inplace(output);
    return output;
}

template <class Scalar, size_t channels, int rows, int cols>
EigenSTLVector<std::array<Eigen::Array<Scalar, rows, cols>, channels>>
positiveMask(EigenSTLVector<std::array<Eigen::Array<Scalar, rows, cols>, channels>> const& input,
    EigenSTLVector<std::array<Eigen::Array<Scalar, rows, cols>, channels>> const& mask)
{
    assert(input.size() == mask.size());

    EigenSTLVector<std::array<Eigen::Array<Scalar, rows, cols>, channels>> output;
    output.reserve(input.size());

    std::transform(
        input.begin(), input.end(),
        mask.begin(),
        std::back_inserter(output),
        [](auto const& input, auto const& mask) {
            std::array<Eigen::Array<Scalar, rows, cols>, channels> output;

            std::transform(
                input.begin(), input.end(),
                mask.begin(),
                output.begin(),
                [](auto const& input, auto const& mask) {
                    return (input * (mask > 0).template cast<Scalar>()).eval();
                });

            return output;
        });

    return output;
}

}  // namespace RobotIntelligence
