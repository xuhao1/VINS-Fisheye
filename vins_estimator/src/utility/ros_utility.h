#include <eigen3/Eigen/Dense>
#include <geometry_msgs/Pose.h>
#include <cv_bridge/cv_bridge.h>

geometry_msgs::Pose pose_from_PQ(Eigen::Vector3d P, 
    const Eigen::Quaterniond & Q);
cv_bridge::CvImageConstPtr getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);
cv_bridge::CvImageConstPtr getImageFromMsg(const sensor_msgs::Image &img_msg);