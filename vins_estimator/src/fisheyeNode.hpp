#include <stdio.h>
#include <queue>
#include <map>
#include <mutex>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "utility/visualization.h"
#include "utility/tic_toc.h"

#include <boost/thread.hpp>
#include "vins/FlattenImages.h"

#include "utility/opencv_cuda.h"
#include "utility/ros_utility.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/CompressedImage.h>

class Estimator;
class FisheyeUndist;
class DepthCamManager;

class FisheyeFlattenHandler
{
    vector<FisheyeUndist> fisheys_undists;

    ros::Publisher flatten_gray_pub;
    ros::Publisher flatten_pub;
    std::vector<bool> mask_up, mask_down;
    ros::Time stamp;

    bool is_color = false;

    std::mutex buf_lock;

    public:


        int raw_height();
        int raw_width();


        std::queue<CvCudaImages> fisheye_cuda_buf_up, fisheye_cuda_buf_down;
        std::queue<CvCudaImages> fisheye_cuda_buf_up_color, fisheye_cuda_buf_down_color;

        std::queue<CvImages> fisheye_buf_up, fisheye_buf_down;
        std::queue<CvImages> fisheye_buf_up_color, fisheye_buf_down_color;

        std::queue<double> fisheye_buf_t;
        //Only gray image will be saved in buf now

        CvCudaImages fisheye_up_imgs_cuda, fisheye_down_imgs_cuda;
        CvCudaImages fisheye_up_imgs_cuda_gray, fisheye_down_imgs_cuda_gray;
        
        CvImages fisheye_up_imgs, fisheye_down_imgs;
        CvImages fisheye_up_imgs_gray, fisheye_down_imgs_gray;

        
        FisheyeFlattenHandler(ros::NodeHandle & n, bool _is_color = true);


        void imgs_callback(const sensor_msgs::ImageConstPtr &img1_msg, const sensor_msgs::ImageConstPtr &img2_msg);

        void imgs_callback(double t, const cv::Mat & img1, const cv::Mat img2, bool is_blank_init = false);

        bool has_image_in_buffer();

        double pop_from_buffer(CvCudaImages & up_gray, CvCudaImages & down_gray,
            CvCudaImages & up_color_gray, CvCudaImages & down_color_gray
        );
        
        double pop_from_buffer(CvImages & up_gray, CvImages & down_gray,
            CvImages & up_color_gray, CvImages & down_color_gray
        );

        void setup_extrinsic(vins::FlattenImages & images, const Estimator & estimator);

        void pack_and_send(ros::Time stamp, 
            cv::InputArray fisheye_up_imgs, cv::InputArray fisheye_down_imgs, 
            cv::InputArray fisheye_up_imgs_gray, cv::InputArray fisheye_down_imgs_gray, 
            const Estimator & estimator);


        void readIntrinsicParameter(const vector<string> &calib_file);

};


class VinsNodeBaseClass {
        message_filters::Subscriber<sensor_msgs::Image> * image_sub_l;
        message_filters::Subscriber<sensor_msgs::Image> * image_sub_r;
        message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> * sync;

        message_filters::Subscriber<sensor_msgs::CompressedImage> * comp_image_sub_l;
        message_filters::Subscriber<sensor_msgs::CompressedImage> * comp_image_sub_r;
        message_filters::TimeSynchronizer<sensor_msgs::CompressedImage, sensor_msgs::CompressedImage> * comp_sync;

        FisheyeFlattenHandler * fisheye_handler;
        ros::Timer timer1, timer2;

        DepthCamManager * cam_manager = nullptr;


        double t_last = 0;

        double last_time;
        
        bool is_color = true;

        double t_last_send = 0;
        std::mutex pack_and_send_mtx;
        bool need_to_pack_and_send = false;

        CvCudaImages cur_up_color_cuda, cur_down_color_cuda;
        CvCudaImages cur_up_gray_cuda, cur_down_gray_cuda;

        CvImages cur_up_color, cur_down_color;
        CvImages cur_up_gray, cur_down_gray;

        double cur_frame_t;

        Estimator estimator;
        ros::Subscriber sub_imu;
        ros::Subscriber sub_feature;
        ros::Subscriber sub_restart;
        ros::Subscriber flatten_sub;

    protected:

        void pack_and_send_thread(const ros::TimerEvent & e);

        void processFlattened(const ros::TimerEvent & e);

        void fisheye_imgs_callback(const sensor_msgs::ImageConstPtr &img1_msg, const sensor_msgs::ImageConstPtr &img2_msg);
        
        void fisheye_comp_imgs_callback(const sensor_msgs::CompressedImageConstPtr &img1_msg, const sensor_msgs::CompressedImageConstPtr &img2_msg);

        void imgs_callback(const sensor_msgs::ImageConstPtr &img1_msg, const sensor_msgs::ImageConstPtr &img2_msg);
        
        void comp_imgs_callback(const sensor_msgs::CompressedImageConstPtr &img1_msg, const sensor_msgs::CompressedImageConstPtr &img2_msg);

        void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg);

        void restart_callback(const std_msgs::BoolConstPtr &restart_msg);

        virtual void Init(ros::NodeHandle & n);
};