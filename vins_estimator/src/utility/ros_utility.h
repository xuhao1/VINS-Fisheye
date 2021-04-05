#include <eigen3/Eigen/Dense>
#include <geometry_msgs/Pose.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

geometry_msgs::Pose pose_from_PQ(Eigen::Vector3d P, 
    const Eigen::Quaterniond & Q);
cv_bridge::CvImageConstPtr getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);
cv::Mat getImageFromMsg(const sensor_msgs::CompressedImageConstPtr &img_msg, int flag = cv::IMREAD_COLOR);
cv::Mat getImageFromMsg(const sensor_msgs::CompressedImage &img_msg, int flag = cv::IMREAD_COLOR);
cv_bridge::CvImageConstPtr getImageFromMsg(const sensor_msgs::Image &img_msg);