#include <Eigen/StdVector>


namespace RobotIntelligence
{

template <class EigenClass>
using EigenSTLVector = std::vector<EigenClass, Eigen::aligned_allocator<EigenClass>>;

}  // namespace RobotIntelligence
