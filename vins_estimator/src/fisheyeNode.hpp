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
#include "depth_generation/depth_camera_manager.h"
#include "vins/FlattenImages.h"
#include "featureTracker/fisheye_undist.hpp"


#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
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
#include "utility/ros_utility.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>


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


        int raw_height() {
            return fisheys_undists[0].raw_height;
        }

        int raw_width() {
            return fisheys_undists[0].raw_width;
        }


        std::queue<CvCudaImages> fisheye_cuda_buf_up, fisheye_cuda_buf_down;
        std::queue<CvCudaImages> fisheye_cuda_buf_up_color, fisheye_cuda_buf_down_color;
        std::queue<double> fisheye_cuda_buf_t;
        //Only gray image will be saved in buf now

        CvCudaImages fisheye_up_imgs_cuda, fisheye_down_imgs_cuda;
        CvCudaImages fisheye_up_imgs_cuda_gray, fisheye_down_imgs_cuda_gray;
        
        CvImages fisheye_up_imgs, fisheye_down_imgs;
        
        FisheyeFlattenHandler(ros::NodeHandle & n): mask_up(5, 0), mask_down(5, 0) 
        {

            readIntrinsicParameter(CAM_NAMES);

            flatten_pub = n.advertise<vins::FlattenImages>("/vins_estimator/flattened_raw", 1);
            flatten_gray_pub = n.advertise<vins::FlattenImages>("/vins_estimator/flattened_gray", 1);

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
                    buf_lock.lock();
                    fisheye_cuda_buf_t.push(t);
                    fisheye_cuda_buf_up.push(fisheye_up_imgs_cuda_gray);
                    fisheye_cuda_buf_down.push(fisheye_down_imgs_cuda_gray);

                    fisheye_cuda_buf_up_color.push(fisheye_up_imgs_cuda);
                    fisheye_cuda_buf_down_color.push(fisheye_down_imgs_cuda);

                    buf_lock.unlock();
                } else {
                    return;
                }
                
                if (ENABLE_PERF_OUTPUT) {
                    ROS_INFO("CvtColor %fms", t_c.toc());
                }

            } else {
                fisheys_undists[0].stereo_flatten(img1, img2, &fisheys_undists[1], 
                    fisheye_up_imgs, fisheye_down_imgs, false, 
                    enable_up_top, enable_rear_side, enable_down_top, enable_rear_side);
            }

            double tf = t_f.toc();
            if (ENABLE_PERF_OUTPUT) {
                ROS_INFO("img_callback cost %fms flatten %fms Flatten AVG %fms", t_f.toc(), tf, flatten_time_sum/count);
            }

            flatten_time_sum += t_f.toc();
        }

        bool has_image_in_buffer() {
            return fisheye_cuda_buf_down.size() > 0;
        }

        std::pair<std::tuple<double, CvCudaImages, CvCudaImages>, std::tuple<double, CvCudaImages, CvCudaImages>> pop_from_buffer() {
            buf_lock.lock();
            if (fisheye_cuda_buf_t.size() > 0) {
                auto t = fisheye_cuda_buf_t.front();
                auto u = fisheye_cuda_buf_up.front();
                auto d = fisheye_cuda_buf_down.front();

                auto uc = fisheye_cuda_buf_up_color.front();
                auto dc = fisheye_cuda_buf_down_color.front();


                fisheye_cuda_buf_t.pop();
                fisheye_cuda_buf_up.pop();
                fisheye_cuda_buf_up_color.pop();
                fisheye_cuda_buf_down.pop();
                fisheye_cuda_buf_down_color.pop();
                buf_lock.unlock();

                
                return std::make_pair(std::make_tuple(t, u, d), std::make_tuple(t, uc, dc));
            } else {
                buf_lock.unlock();
                return std::make_pair(std::make_tuple(0.0, CvCudaImages(0), CvCudaImages(0)), std::make_tuple(0.0, CvCudaImages(0), CvCudaImages(0)));
            }
        }

        void setup_extrinsic(vins::FlattenImages & images, const Estimator & estimator) {
            static Eigen::Quaterniond t_left = Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d(1, 0, 0)));
            static Eigen::Quaterniond t_down = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d(1, 0, 0)));

            std::vector<Eigen::Quaterniond> t_dirs;
            t_dirs.push_back(Eigen::Quaterniond::Identity());
            t_dirs.push_back(t_left);
            for (unsigned int i = 0; i < 3; i ++) {
                t_dirs.push_back(t_dirs.back() * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0)));
            }

            for (unsigned int i = 0; i < 4; i ++) {
                images.extrinsic_up_cams.push_back(
                    pose_from_PQ(estimator.tic[0], Eigen::Quaterniond(estimator.ric[0])*t_dirs[i])
                );
                images.extrinsic_down_cams.push_back(
                    pose_from_PQ(estimator.tic[1], Eigen::Quaterniond(estimator.ric[1])*t_down*t_dirs[i])
                );
            }
        }

        void pack_and_send_cuda(ros::Time stamp, 
            const CvCudaImages & fisheye_up_imgs_cuda, const CvCudaImages & fisheye_down_imgs_cuda, 
            const CvCudaImages & fisheye_up_imgs_cuda_gray, const CvCudaImages & fisheye_down_imgs_cuda_gray, 
            const Estimator & estimator) {
            TicToc t_p;
            vins::FlattenImages images;
            vins::FlattenImages images_gray;
            static double pack_send_time = 0;

            setup_extrinsic(images, estimator);
            setup_extrinsic(images_gray, estimator);

            images.header.stamp = stamp;
            images_gray.header.stamp = stamp;
            static int count = 0;
            count ++;

            for (unsigned int i = 0; i < fisheye_up_imgs_cuda.size(); i++) {
                cv_bridge::CvImage outImg;
                cv_bridge::CvImage outImg_gray;
                if (is_color) {
                    outImg.encoding = "8UC3";
                } else {
                    outImg.encoding = "mono8";
                }

                outImg_gray.encoding = "mono8";
                TicToc to;
                fisheye_up_imgs_cuda[i].download(outImg.image);
                images.up_cams.push_back(*outImg.toImageMsg());

                if (i == 2) {
                    fisheye_up_imgs_cuda_gray[i].download(outImg_gray.image);
                }
                images_gray.up_cams.push_back(*outImg_gray.toImageMsg());
            }

            for (unsigned int i = 0; i < fisheye_down_imgs_cuda.size(); i++) {
                cv_bridge::CvImage outImg;
                cv_bridge::CvImage outImg_gray;

                if (is_color) {
                    outImg.encoding = "8UC3";
                } else {
                    outImg.encoding = "mono8";
                }
                
                outImg_gray.encoding = "mono8";
                fisheye_down_imgs_cuda[i].download(outImg.image);
                images.down_cams.push_back(*outImg.toImageMsg());
                
                if (i == 2) {
                    fisheye_down_imgs_cuda_gray[i].download(outImg_gray.image);
                }
                images_gray.down_cams.push_back(*outImg_gray.toImageMsg());
            }

            // ROS_INFO("Pack cost %fms", t_p.toc());
            flatten_pub.publish(images);
            flatten_gray_pub.publish(images_gray);
            pack_send_time += t_p.toc();

            if (ENABLE_PERF_OUTPUT) {
                ROS_INFO("Pack and send AVG %fms this %fms", pack_send_time/count, t_p.toc());
            }
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


class VinsNodeBaseClass {
        message_filters::Subscriber<sensor_msgs::Image> * image_sub_l;
        message_filters::Subscriber<sensor_msgs::Image> * image_sub_r;
    
        FisheyeFlattenHandler * fisheye_handler;
        ros::Timer timer1, timer2;

        DepthCamManager * cam_manager = nullptr;


        double t_last = 0;

        double last_time;
        bool last_time_initialized;


        double t_last_send = 0;
        std::mutex pack_and_send_mtx;
        bool need_to_pack_and_send = false;
        std::tuple<double, CvCudaImages, CvCudaImages> cur_frame;
        std::tuple<double, CvCudaImages, CvCudaImages> cur_frame_gray;

        Estimator estimator;
        ros::Subscriber sub_imu;
        ros::Subscriber sub_feature;
        ros::Subscriber sub_restart;
        ros::Subscriber flatten_sub;

        message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> * sync;
    protected:

        void pack_and_send_thread(const ros::TimerEvent & e) {               
            if (need_to_pack_and_send && std::get<0>(cur_frame) > t_last_send) {
                //need to pack and send
                pack_and_send_mtx.lock();
                t_last_send = std::get<0>(cur_frame);
                need_to_pack_and_send = false;
                fisheye_handler->pack_and_send_cuda(ros::Time(t_last_send), 
                    std::get<1>(cur_frame), std::get<2>(cur_frame), 
                    std::get<1>(cur_frame_gray), std::get<2>(cur_frame_gray), 
                    estimator);
                pack_and_send_mtx.unlock();
            }
        }

        void processFlattened(const ros::TimerEvent & e) {
            TicToc t0;
            if (fisheye_handler->has_image_in_buffer()) {
                auto ret = fisheye_handler->pop_from_buffer();
                cur_frame_gray = ret.first;
                cur_frame = ret.second;
                bool is_odometry_frame = estimator.is_next_odometry_frame();

                if (is_odometry_frame) {
                    need_to_pack_and_send = true;
                    // fisheye_handler->pack_and_send_cuda(ros::Time(t_last_send), std::get<1>(cur_frame), std::get<2>(cur_frame));
                }
                estimator.inputFisheyeImage(std::get<0>(cur_frame_gray), std::get<1>(cur_frame_gray), std::get<2>(cur_frame_gray));

                double t_0 = t0.toc();
                //Need to wait for pack and send to end
                pack_and_send_mtx.lock();
                pack_and_send_mtx.unlock();

                if(ENABLE_PERF_OUTPUT) {
                    ROS_INFO("[processFlattened]Input Image: %fms, whole %fms", t_0, t0.toc());
                }

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


        virtual void Init(ros::NodeHandle & n)
        {
            std::string config_file;
            n.getParam("config_file", config_file);
            bool fisheye_external_flatten;
            n.getParam("fisheye_external_flatten", fisheye_external_flatten);
            
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

            sub_imu = n.subscribe(IMU_TOPIC, 2000, &VinsNodeBaseClass::imu_callback, (VinsNodeBaseClass*)this);
            sub_restart = n.subscribe("/vins_restart", 100, &VinsNodeBaseClass::restart_callback, (VinsNodeBaseClass*)this);

            ROS_INFO("Will directly receive raw images");
            image_sub_l = new message_filters::Subscriber<sensor_msgs::Image> (n, IMAGE0_TOPIC, 1000);
            image_sub_r = new message_filters::Subscriber<sensor_msgs::Image> (n, IMAGE1_TOPIC, 1000);
            sync = new message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> (*image_sub_l, *image_sub_r, 1000);

            timer1 = n.createTimer(ros::Duration(0.004), boost::bind(&VinsNodeBaseClass::processFlattened, (VinsNodeBaseClass*)this, _1 ));
            timer2 = n.createTimer(ros::Duration(0.004), boost::bind(&VinsNodeBaseClass::pack_and_send_thread, (VinsNodeBaseClass*)this, _1 ));
            
            if (FISHEYE) {
                sync->registerCallback(boost::bind(&VinsNodeBaseClass::fisheye_imgs_callback, (VinsNodeBaseClass*)this, _1, _2));
            } else {    
                sync->registerCallback(boost::bind(&VinsNodeBaseClass::img_callback, (VinsNodeBaseClass*)this, _1, _2));
            }
        }
};