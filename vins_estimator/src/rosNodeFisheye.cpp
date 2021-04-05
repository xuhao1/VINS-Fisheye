#include "fisheyeNode.hpp"

#ifdef USE_BACKWARD
#define BACKWARD_HAS_DW 1
#include <backward.hpp>
namespace backward
{
    backward::SignalHandling sh;
}
#endif


class VinsNodeFisheye :  public VinsNodeBaseClass
{
    public:
        VinsNodeFisheye(ros::NodeHandle & nh)
        {
            Init(nh);
        }
};

int main(int argc, char **argv)
{
    cv::setNumThreads(1);
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);


    VinsNodeFisheye fisheye(n);

    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    return 0;
}

