#include "fisheyeNode.hpp"

#ifdef USE_BACKWARD
#define BACKWARD_HAS_DW 1
#include <backward.hpp>
namespace backward
{
    backward::SignalHandling sh;
}
#endif

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

namespace vins_nodelet_pkg
{
    class VinsNodeletClass : public nodelet::Nodelet, public VinsNodeBaseClass
    {
        public:
            VinsNodeletClass() {}
        private:
            message_filters::Subscriber<sensor_msgs::Image> * image_sub_l;
            message_filters::Subscriber<sensor_msgs::Image> * image_sub_r;
    
            virtual void onInit() override
            {
                ros::NodeHandle & n = getMTPrivateNodeHandle();
                Init(n);
            }
    };
    PLUGINLIB_EXPORT_CLASS(vins_nodelet_pkg::VinsNodeletClass, nodelet::Nodelet);

}
