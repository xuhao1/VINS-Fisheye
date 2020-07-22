#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "utility/visualization.h"
#include "utility/tic_toc.h"

#include <boost/thread.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include "depth_generation/depth_camera_manager.h"
#include "vins/FlattenImages.h"
#include "featureTracker/fisheye_undist.hpp"

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
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include "estimator/parameters.h"
#include <boost/thread.hpp>
#include "depth_generation/depth_camera_manager.h"
#include "featureTracker/fisheye_undist.hpp"
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "utility/tic_toc.h"
#include "vins/FlattenImages.h"
#include "utility/opencv_cuda.h"

class FisheyeFlattenHandler
{
    vector<FisheyeUndist> fisheys_undists;

    ros::Publisher flatten_pub;
    std::vector<bool> mask_up, mask_down;
    ros::Time stamp;

    bool is_color = false;

    public:


        int raw_height() {
            return fisheys_undists[0].raw_height;
        }

        int raw_width() {
            return fisheys_undists[0].raw_width;
        }


        std::queue<CvCudaImages> fisheye_cuda_buf_up, fisheye_cuda_buf_down;
        std::queue<double> fisheye_cuda_buf_t;
        //Only gray image will be saved in buf now

        CvCudaImages fisheye_up_imgs_cuda, fisheye_down_imgs_cuda;
        CvCudaImages fisheye_up_imgs_cuda_gray, fisheye_down_imgs_cuda_gray;
        
        CvImages fisheye_up_imgs, fisheye_down_imgs;
        
        FisheyeFlattenHandler(ros::NodeHandle & n): mask_up(5, 0), mask_down(5, 0) 
        {

            readIntrinsicParameter(CAM_NAMES);

            flatten_pub = n.advertise<vins::FlattenImages>("/vins_estimator/flattened_raw", 1);

            if (enable_up_top) {
                mask_up[0] = true;        
            }

            if (enable_down_top) {
                mask_down[0] = true;
            }

            if (enable_up_side) {
                mask_up[1] = true;
                mask_up[2] = true;
                mask_up[3] = true;
            }

            if(enable_rear_side) {
                mask_up[4] = true;
                mask_down[4] = true;
            }

            if (enable_down_side) {
                mask_down[1] = true;
                mask_down[2] = true;
                mask_down[3] = true;
            }
        }

        void img_callback(const sensor_msgs::ImageConstPtr &img1_msg, const sensor_msgs::ImageConstPtr &img2_msg)
        {
            auto img1 = getImageFromMsg(img1_msg);
            auto img2 = getImageFromMsg(img2_msg);
            stamp = img1_msg->header.stamp;
            img_callback(img1_msg->header.stamp.toSec(), img1->image, img2->image);
        }

        void img_callback(double t, const cv::Mat & img1, const cv::Mat img2, bool is_blank_init = false) {

            static double flatten_time_sum = 0;
            static double count = 0;

            count += 1;

            TicToc t_f;


            if (USE_GPU) {
                is_color = true;
                fisheye_up_imgs_cuda = fisheys_undists[0].undist_all_cuda(img1, is_color, mask_up); 
                fisheye_up_imgs_cuda_gray.clear();
                fisheye_down_imgs_cuda_gray.clear();


                fisheye_down_imgs_cuda = fisheys_undists[1].undist_all_cuda(img2, is_color, mask_down);

                TicToc t_c;
                for (auto & img: fisheye_up_imgs_cuda) {
                    cv::cuda::GpuMat gray;
                    if(!img.empty()) {
                        cv::cuda::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
                    }
                    fisheye_up_imgs_cuda_gray.push_back(gray);
                }

                for (auto & img: fisheye_down_imgs_cuda) {
                    cv::cuda::GpuMat gray;
                    if(!img.empty()) {
                        cv::cuda::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
                    }

                    fisheye_down_imgs_cuda_gray.push_back(gray);
                }

                if (!is_blank_init) {
                    fisheye_cuda_buf_t.push(t);
                    fisheye_cuda_buf_up.push(fisheye_up_imgs_cuda_gray);
                    fisheye_cuda_buf_down.push(fisheye_down_imgs_cuda_gray);
                } else {
                    return;
                }

                ROS_INFO("CvtColor %fms", t_c.toc());

            } else {
                fisheys_undists[0].stereo_flatten(img1, img2, &fisheys_undists[1], 
                    fisheye_up_imgs, fisheye_down_imgs, false, 
                    enable_up_top, enable_rear_side, enable_down_top, enable_rear_side);
            }

            pack_and_send_cuda();

            ROS_INFO("Flatten cost %fms Flatten AVG %fms", t_f.toc(), flatten_time_sum/count);

            flatten_time_sum += t_f.toc();
        }

        bool has_image_in_buffer() {
            return fisheye_cuda_buf_down.size() > 0;
        }

        std::tuple<double, CvCudaImages, CvCudaImages> pop_from_buffer() {
            if (fisheye_cuda_buf_t.size() > 0) {
                auto t = fisheye_cuda_buf_t.front();
                auto u = fisheye_cuda_buf_up.front();
                auto d = fisheye_cuda_buf_down.front();

                fisheye_cuda_buf_t.pop();
                fisheye_cuda_buf_up.pop();
                fisheye_cuda_buf_down.pop();
                
                return std::make_tuple(t, u, d);
            } else {
                return std::make_tuple(0.0, CvCudaImages(0), CvCudaImages(0));
            }
        }

        void pack_and_send_cuda() {
            TicToc t_p;
            vins::FlattenImages images;
            static double pack_send_time = 0;

            images.header.stamp = stamp;
            static int count = 0;
            count ++;

            for (unsigned int i = 0; i < fisheye_up_imgs_cuda.size(); i++) {
                cv_bridge::CvImage outImg;
                if (is_color) {
                    outImg.encoding = "bgr8";
                } else {
                    outImg.encoding = "mono8";
                }

                fisheye_up_imgs_cuda[i].download(outImg.image);
                images.up_cams.push_back(*outImg.toImageMsg());
            }

            for (unsigned int i = 0; i < fisheye_down_imgs_cuda.size(); i++) {
                cv_bridge::CvImage outImg;
                if (is_color) {
                    outImg.encoding = "bgr8";
                } else {
                    outImg.encoding = "mono8";
                }

                fisheye_down_imgs_cuda[i].download(outImg.image);
                images.down_cams.push_back(*outImg.toImageMsg());
            }

            flatten_pub.publish(images);
            pack_send_time += t_p.toc();

            ROS_INFO("Pack and send AVG %fms", pack_send_time/count);
        }


        void readIntrinsicParameter(const vector<string> &calib_file)
        {
            for (size_t i = 0; i < calib_file.size(); i++)
            {
                if (FISHEYE) {
                    ROS_INFO("Flatten read fisheye %s, id %ld", calib_file[i].c_str(), i);
                    FisheyeUndist un(calib_file[i].c_str(), i, FISHEYE_FOV, true, COL);
                    fisheys_undists.push_back(un);
                }
            }
        }


        cv_bridge::CvImageConstPtr getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
        {
            cv_bridge::CvImageConstPtr ptr;
            if (img_msg->encoding == "8UC1")
            {
                ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::MONO8);
            }
            else
            {
                if (FISHEYE) {
                    ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::BGR8);
                } else {
                    ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::MONO8);        
                }
            }
            return ptr;
        }
};

namespace vins_nodelet_pkg
{
    class VinsNodeletClass : public nodelet::Nodelet
    {
        public:
            VinsNodeletClass() {}
        private:
            message_filters::Subscriber<sensor_msgs::Image> * image_sub_l;
            message_filters::Subscriber<sensor_msgs::Image> * image_sub_r;
            FisheyeFlattenHandler * fisheye_handler;

            ros::Timer timer;

            DepthCamManager * cam_manager = nullptr;
            virtual void onInit()
            {
                auto n = getMTNodeHandle();
                auto private_n = getMTPrivateNodeHandle();
                std::string config_file;
                private_n.getParam("config_file", config_file);
                bool fisheye_external_flatten;
                private_n.getParam("fisheye_external_flatten", fisheye_external_flatten);
                
                std::cout << "config file is " << config_file << '\n';
                readParameters(config_file);
                estimator.setParameter();

                ROS_INFO("Will %d GPU", USE_GPU);
                
                if (ENABLE_DEPTH) {
                    cam_manager = new DepthCamManager(n, &(estimator.featureTracker.fisheys_undists[0]));
                    cam_manager -> init_with_extrinsic(estimator.ric[0], estimator.tic[0], estimator.ric[1], estimator.tic[1]);
                    estimator.depth_cam_manager = cam_manager;
                }
            #ifdef EIGEN_DONT_PARALLELIZE
                ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
            #endif

                ROS_WARN("waiting for image and imu...");

                registerPub(n);

                if (FISHEYE) {
                    fisheye_handler = new FisheyeFlattenHandler(n);
                }
                //We use blank images to initialize cuda before every thing
                TicToc blank;
                cv::Mat mat(fisheye_handler->raw_width(), fisheye_handler->raw_height(), CV_8UC3);
                fisheye_handler->img_callback(0, mat, mat, true);
                estimator.inputFisheyeImage(0, 
                        fisheye_handler->fisheye_up_imgs_cuda_gray, fisheye_handler->fisheye_down_imgs_cuda_gray, true);
                std::cout<< "Initialize with blank cost" << blank.toc() << std::endl;

                sub_imu = n.subscribe(IMU_TOPIC, 2000, &VinsNodeletClass::imu_callback, this);
                sub_restart = n.subscribe("/vins_restart", 100, &VinsNodeletClass::restart_callback, this);

                ROS_INFO("Will directly receive raw images");
                image_sub_l = new message_filters::Subscriber<sensor_msgs::Image> (n, IMAGE0_TOPIC, 1000);
                image_sub_r = new message_filters::Subscriber<sensor_msgs::Image> (n, IMAGE1_TOPIC, 1000);
                sync = new message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> (*image_sub_l, *image_sub_r, 1000);

                timer = n.createTimer(ros::Duration(0.001), boost::bind(&VinsNodeletClass::processFlattened, this, _1 ));
                
                if (FISHEYE) {
                    sync->registerCallback(boost::bind(&VinsNodeletClass::fisheye_imgs_callback, this, _1, _2));
                } else {    
                    sync->registerCallback(boost::bind(&VinsNodeletClass::img_callback, this, _1, _2));
                }
            }


            double t_last = 0;

            void processFlattened(const ros::TimerEvent & e) {
                TicToc t0;
                if (fisheye_handler->has_image_in_buffer()) {
                    auto ret = fisheye_handler->pop_from_buffer();
                    estimator.inputFisheyeImage(std::get<0>(ret), std::get<1>(ret), std::get<2>(ret));
                    ROS_INFO("Input Image: %fms", t0.toc());
                }
            }

            void fisheye_imgs_callback(const sensor_msgs::ImageConstPtr &img1_msg, const sensor_msgs::ImageConstPtr &img2_msg) {
                TicToc tic_input;
                fisheye_handler->img_callback(img1_msg, img2_msg);

                if (img1_msg->header.stamp.toSec() - t_last > 0.11) {
                    ROS_WARN("Duration between two images is %fms", img1_msg->header.stamp.toSec() - t_last);
                }
                t_last = img1_msg->header.stamp.toSec();

                if (USE_GPU) {
                } else {
                    estimator.inputImage(img1_msg->header.stamp.toSec(), cv::Mat(), cv::Mat(), 
                        fisheye_handler->fisheye_up_imgs, fisheye_handler->fisheye_down_imgs);
                }
            }

            void img_callback(const sensor_msgs::ImageConstPtr &img1_msg, const sensor_msgs::ImageConstPtr &img2_msg)
            {
                auto img1 = getImageFromMsg(img1_msg);
                auto img2 = getImageFromMsg(img2_msg);
                estimator.inputImage(img1_msg->header.stamp.toSec(), img1->image, img2->image);
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

            void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
            {
                double t = imu_msg->header.stamp.toSec();
                double dx = imu_msg->linear_acceleration.x;
                double dy = imu_msg->linear_acceleration.y;
                double dz = imu_msg->linear_acceleration.z;
                double rx = imu_msg->angular_velocity.x;
                double ry = imu_msg->angular_velocity.y;
                double rz = imu_msg->angular_velocity.z;
                Vector3d acc(dx, dy, dz);
                Vector3d gyr(rx, ry, rz);
                estimator.inputIMU(t, acc, gyr);

                // test, should be deleted
                if (! last_time_initialized)
                {
                    last_time = ros::Time::now().toSec();
                    last_time_initialized = true;
                }
                else
                {
                    double now_time = ros::Time::now().toSec();
                    if (now_time - last_time > 3)
                        ros::shutdown();
                    last_time = now_time;
                }
                // test end
            }

            void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
            {
                if (restart_msg->data == true)
                {
                    ROS_WARN("restart the estimator!");
                    estimator.clearState();
                    estimator.setParameter();
                }
                return;
            }

            Estimator estimator;

            ros::Subscriber sub_imu;
            ros::Subscriber sub_feature;
            ros::Subscriber sub_restart;
            ros::Subscriber flatten_sub;

            message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> * sync;
            double last_time;
            bool last_time_initialized;
    };
    PLUGINLIB_EXPORT_CLASS(vins_nodelet_pkg::VinsNodeletClass, nodelet::Nodelet);

}
