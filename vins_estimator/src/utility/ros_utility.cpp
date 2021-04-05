#include "ros_utility.h"

cv_bridge::CvImageConstPtr getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    // std::cout << img_msg->encoding << std::endl;
    if (img_msg->encoding == "8UC1" || img_msg->encoding == "mono8")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::MONO8);
    } else
    {
        ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::BGR8);        
    }
    return ptr;
}

cv_bridge::CvImageConstPtr getImageFromMsg(const sensor_msgs::Image &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg.width > 0) {
        if (img_msg.encoding == "8UC1")
        {
            ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
        } else {
            ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
        }
        return ptr;
    }
    return nullptr;
}

cv::Mat getImageFromMsg(const sensor_msgs::CompressedImageConstPtr &img_msg, int flag) {
    return cv::imdecode(img_msg->data, flag);
}

cv::Mat getImageFromMsg(const sensor_msgs::CompressedImage &img_msg, int flag) {
    return cv::imdecode(img_msg.data, flag);
}

geometry_msgs::Pose pose_from_PQ(Eigen::Vector3d P, 
    const Eigen::Quaterniond & Q) {
    geometry_msgs::Pose pose;
    pose.position.x = P.x();
    pose.position.y = P.y();
    pose.position.z = P.z();
    pose.orientation.x = Q.x();
    pose.orientation.y = Q.y();
    pose.orientation.z = Q.z();
    pose.orientation.w = Q.w();
    return pose;
}


