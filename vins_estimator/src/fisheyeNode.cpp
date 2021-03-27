#include "fisheyeNode.hpp"
#include "featureTracker/feature_tracker_fisheye.hpp"
#include "featureTracker/fisheye_undist.hpp"
#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "depth_generation/depth_camera_manager.h"

using namespace FeatureTracker;     
FisheyeFlattenHandler::FisheyeFlattenHandler(ros::NodeHandle & n, bool _is_color): mask_up(5, 0), mask_down(5, 0), is_color(_is_color)
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



int FisheyeFlattenHandler::raw_height() {
    return fisheys_undists[0].raw_height;
}

int FisheyeFlattenHandler::raw_width() {
    return fisheys_undists[0].raw_width;
}


void FisheyeFlattenHandler::imgs_callback(const sensor_msgs::ImageConstPtr &img1_msg, const sensor_msgs::ImageConstPtr &img2_msg)
{
    auto img1 = getImageFromMsg(img1_msg);
    auto img2 = getImageFromMsg(img2_msg);
    stamp = img1_msg->header.stamp;
    imgs_callback(img1_msg->header.stamp.toSec(), img1->image, img2->image);
}

void FisheyeFlattenHandler::imgs_callback(double t, const cv::Mat & img1, const cv::Mat img2, bool is_blank_init) {

    static double flatten_time_sum = 0;
    static double count = 0;

    count += 1;

    TicToc t_f;

    if (USE_GPU) {
        // is_color = true;
        if (is_color) {
            fisheye_up_imgs_cuda = fisheys_undists[0].undist_all_cuda(img1, true, mask_up); 
            fisheye_down_imgs_cuda = fisheys_undists[1].undist_all_cuda(img2, true, mask_down);
            fisheye_up_imgs_cuda_gray.clear();
            fisheye_down_imgs_cuda_gray.clear();


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
        } else {
            fisheye_up_imgs_cuda_gray = fisheys_undists[0].undist_all_cuda(img1, false, mask_up); 
            fisheye_down_imgs_cuda_gray = fisheys_undists[1].undist_all_cuda(img2, false, mask_down);
        }

        if (!is_blank_init) {
            buf_lock.lock();
            fisheye_buf_t.push(t);
            fisheye_cuda_buf_up.push(fisheye_up_imgs_cuda_gray);
            fisheye_cuda_buf_down.push(fisheye_down_imgs_cuda_gray);
            if(is_color) {
                fisheye_cuda_buf_up_color.push(fisheye_up_imgs_cuda);
                fisheye_cuda_buf_down_color.push(fisheye_down_imgs_cuda);
            }
            buf_lock.unlock();
        }
    } else {
        if (is_color) {
            fisheys_undists[0].stereo_flatten(img1, img2, &fisheys_undists[1], 
                fisheye_up_imgs, fisheye_down_imgs, true, 
                enable_up_top, enable_rear_side, enable_down_top, enable_rear_side);
            fisheye_up_imgs_gray.clear();
            fisheye_down_imgs_gray.clear();
            for (auto & img: fisheye_up_imgs) {
                cv::Mat gray;
                if(!img.empty()) {
                    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
                }
                fisheye_up_imgs_gray.push_back(gray);
            }

            for (auto & img: fisheye_down_imgs) {
                cv::Mat gray;
                if(!img.empty()) {
                    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
                }
                fisheye_down_imgs_gray.push_back(gray);
            }
        } else {
            fisheys_undists[0].stereo_flatten(img1, img2, &fisheys_undists[1], 
                fisheye_up_imgs_gray, fisheye_down_imgs_gray, false, 
                enable_up_top, enable_rear_side, enable_down_top, enable_rear_side);
        }

        buf_lock.lock();
        fisheye_buf_t.push(t);
        fisheye_buf_up.push(fisheye_up_imgs_gray);
        fisheye_buf_down.push(fisheye_down_imgs_gray);

        if (is_color) {
            fisheye_buf_up_color.push(fisheye_up_imgs);
            fisheye_buf_down_color.push(fisheye_down_imgs);
        }

        buf_lock.unlock();
    }

    double tf = t_f.toc();
    if (ENABLE_PERF_OUTPUT) {
        ROS_INFO("img_callback cost %fms flatten %fms Flatten AVG %fms", t_f.toc(), tf, flatten_time_sum/count);
    }

    flatten_time_sum += t_f.toc();
}

bool FisheyeFlattenHandler::has_image_in_buffer() {
    return fisheye_buf_t.size() > 0;
}

double FisheyeFlattenHandler::pop_from_buffer(
            CvCudaImages & up_gray, CvCudaImages & down_gray,
            CvCudaImages & up_color, CvCudaImages & down_color) {
    if (USE_GPU) {
        buf_lock.lock();
        if (fisheye_buf_t.size() > 0) {
            auto t = fisheye_buf_t.front();
            up_gray = fisheye_cuda_buf_up.front();
            down_gray = fisheye_cuda_buf_down.front();

            if(is_color) {
                up_color = fisheye_cuda_buf_up_color.front();
                down_color = fisheye_cuda_buf_down_color.front();
            }


            fisheye_buf_t.pop();
            fisheye_cuda_buf_up.pop();
            fisheye_cuda_buf_down.pop();

            if(is_color) {
                fisheye_cuda_buf_up_color.pop();
                fisheye_cuda_buf_down_color.pop();
            }

            buf_lock.unlock();
            return t;
        }
    }
    return -1;
}

double FisheyeFlattenHandler::pop_from_buffer(
            CvImages & up_gray, CvImages & down_gray,
            CvImages & up_color, CvImages & down_color) {
    if(!USE_GPU) {
        buf_lock.lock();
        if (fisheye_buf_t.size() > 0) {
            auto t = fisheye_buf_t.front();

            up_gray = fisheye_buf_up.front();
            down_gray = fisheye_buf_down.front();

            if(is_color) {
                up_color = fisheye_buf_up_color.front();
                down_color = fisheye_buf_down_color.front();
            }

            fisheye_buf_t.pop();
            fisheye_buf_up.pop();
            fisheye_buf_down.pop();

            if(is_color) {
                fisheye_buf_up_color.pop();
                fisheye_buf_down_color.pop();
            }
            
            buf_lock.unlock();
            return t;
        }
    }
    return -1;
}

void FisheyeFlattenHandler::setup_extrinsic(vins::FlattenImages & images, const Estimator & estimator) {
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

void FisheyeFlattenHandler::pack_and_send(ros::Time stamp, 
        cv::InputArray fisheye_up_imgs, cv::InputArray fisheye_down_imgs, 
        cv::InputArray fisheye_up_imgs_gray, cv::InputArray fisheye_down_imgs_gray, 
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

    CvCudaImages fisheye_up_imgs_cuda, fisheye_down_imgs_cuda;
    CvCudaImages fisheye_up_imgs_cuda_gray, fisheye_down_imgs_cuda_gray;
    if (USE_GPU) {
        fisheye_up_imgs.getGpuMatVector(fisheye_up_imgs_cuda);
        fisheye_down_imgs.getGpuMatVector(fisheye_down_imgs_cuda);
        fisheye_up_imgs_gray.getGpuMatVector(fisheye_up_imgs_cuda_gray);
        fisheye_down_imgs_gray.getGpuMatVector(fisheye_down_imgs_cuda_gray);
        // std::cout << "fisheye_up_imgs_cuda_gray size: " << fisheye_up_imgs_cuda_gray.size() << std::endl;

    }

    size_t _size = fisheye_up_imgs_cuda.size();
    
    if (!is_color) {
        _size = fisheye_up_imgs_cuda_gray.size();
    }

    for (unsigned int i = 0; i < _size; i++) {
        cv_bridge::CvImage outImg;
        cv_bridge::CvImage outImg_gray;

        outImg.encoding = "8UC3";
        outImg_gray.encoding = "mono8";
        outImg.header = images.header;
        outImg_gray.header = images.header;
        TicToc to;
        
        if (USE_GPU) {
            if (is_color)
                fisheye_up_imgs_cuda[i].download(outImg.image);
            // std::cout << "Sending image " << i << " size " << fisheye_up_imgs_cuda_gray[i].size() << std::endl;
            fisheye_up_imgs_cuda_gray[i].download(outImg_gray.image);
        } else {
            if (is_color)
                outImg.image = fisheye_up_imgs.getMat(i);
            outImg_gray.image = fisheye_up_imgs_gray.getMat(i);
        }
        
        if (is_color) {
            images.up_cams.push_back(*outImg.toImageMsg());
        }

        images_gray.up_cams.push_back(*outImg_gray.toImageMsg());
    }

    for (unsigned int i = 0; i < _size; i++) {
        cv_bridge::CvImage outImg;
        cv_bridge::CvImage outImg_gray;

        outImg.encoding = "8UC3";
        outImg_gray.encoding = "mono8";

        if (USE_GPU) {
            if (is_color) {
                fisheye_down_imgs_cuda[i].download(outImg.image);
            }
            fisheye_down_imgs_cuda_gray[i].download(outImg_gray.image);
        } else {
            if (is_color) {
                outImg.image = fisheye_down_imgs.getMat(i);
            }
            outImg_gray.image = fisheye_down_imgs_gray.getMat(i);
        }
        
        if (is_color) {
            images.down_cams.push_back(*outImg.toImageMsg());
        }

        images_gray.down_cams.push_back(*outImg_gray.toImageMsg());
    }

    if (is_color) {
        flatten_pub.publish(images);
    }

    flatten_gray_pub.publish(images_gray);
    pack_send_time += t_p.toc();

    if (ENABLE_PERF_OUTPUT) {
        ROS_INFO("Pack and send AVG %fms this %fms", pack_send_time/count, t_p.toc());
    }
}


void FisheyeFlattenHandler::readIntrinsicParameter(const vector<string> &calib_file)
{
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        if (FISHEYE) {
            ROS_INFO("Flatten read fisheye %s, id %ld", calib_file[i].c_str(), i);
            FisheyeUndist un(calib_file[i].c_str(), i, FISHEYE_FOV, true, WIDTH);
            fisheys_undists.push_back(un);
        }
    }
}


void VinsNodeBaseClass::pack_and_send_thread(const ros::TimerEvent & e) {               
    if (need_to_pack_and_send && cur_frame_t > t_last_send) {
        //need to pack and send
        pack_and_send_mtx.lock();
        t_last_send = cur_frame_t;
        need_to_pack_and_send = false;
        if (USE_GPU) {
            fisheye_handler->pack_and_send(ros::Time(t_last_send), 
                cur_up_color_cuda, cur_down_color_cuda,
                cur_up_gray_cuda, cur_down_gray_cuda,
                estimator);
        } else {
                fisheye_handler->pack_and_send(ros::Time(t_last_send), 
                cur_up_color, cur_down_color,
                cur_up_gray, cur_down_gray,
                estimator);
        }
        pack_and_send_mtx.unlock();
    }
}

void VinsNodeBaseClass::processFlattened(const ros::TimerEvent & e) {
    TicToc t0;
    if (fisheye_handler->has_image_in_buffer()) {
        pack_and_send_mtx.lock();

        if (USE_GPU) {
            cur_frame_t = fisheye_handler->pop_from_buffer(
                cur_up_gray_cuda,
                cur_down_gray_cuda,
                cur_up_color_cuda,
                cur_down_color_cuda
            );

            bool is_odometry_frame = estimator.is_next_odometry_frame();

            if (is_odometry_frame) {
                need_to_pack_and_send = true;
            }
            estimator.inputFisheyeImage(cur_frame_t, cur_up_gray_cuda, cur_down_gray_cuda);
        } else {
            cur_frame_t = fisheye_handler->pop_from_buffer(
                cur_up_gray,
                cur_down_gray,
                cur_up_color,
                cur_down_color
            );

            bool is_odometry_frame = estimator.is_next_odometry_frame();

            if (is_odometry_frame) {
                need_to_pack_and_send = true;
            }
            estimator.inputFisheyeImage(cur_frame_t, cur_up_gray, cur_down_gray);
        }
        double t_0 = t0.toc();
        //Need to wait for pack and send to endft
        pack_and_send_mtx.unlock();

        if(ENABLE_PERF_OUTPUT) {
            ROS_INFO("[processFlattened]Input Image: %fms, whole %fms", t_0, t0.toc());
        }

    }
}

void VinsNodeBaseClass::fisheye_imgs_callback(const sensor_msgs::ImageConstPtr &img1_msg, const sensor_msgs::ImageConstPtr &img2_msg) {
    TicToc tic_input;
    fisheye_handler->imgs_callback(img1_msg, img2_msg);

    if (img1_msg->header.stamp.toSec() - t_last > 0.11) {
        ROS_WARN("Duration between two images is %fms", img1_msg->header.stamp.toSec() - t_last);
    }
    t_last = img1_msg->header.stamp.toSec();
}

void VinsNodeBaseClass::fisheye_comp_imgs_callback(const sensor_msgs::CompressedImageConstPtr &img1_msg, const sensor_msgs::CompressedImageConstPtr &img2_msg) {
    TicToc tic_input;
    auto img1 = getImageFromMsg(img1_msg);
    auto img2 = getImageFromMsg(img2_msg);

    fisheye_handler->imgs_callback(img1_msg->header.stamp.toSec(), img1, img2);

    if (img1_msg->header.stamp.toSec() - t_last > 0.11) {
        ROS_WARN("Duration between two images is %fms", img1_msg->header.stamp.toSec() - t_last);
    }
    t_last = img1_msg->header.stamp.toSec();
}

void VinsNodeBaseClass::imgs_callback(const sensor_msgs::ImageConstPtr &img1_msg, const sensor_msgs::ImageConstPtr &img2_msg)
{
    auto img1 = getImageFromMsg(img1_msg);
    auto img2 = getImageFromMsg(img2_msg);
    estimator.inputImage(img1_msg->header.stamp.toSec(), img1->image, img2->image);
}


void VinsNodeBaseClass::comp_imgs_callback(const sensor_msgs::CompressedImageConstPtr &img1_msg, const sensor_msgs::CompressedImageConstPtr &img2_msg)
{
    auto img1 = getImageFromMsg(img1_msg);
    auto img2 = getImageFromMsg(img2_msg);
    estimator.inputImage(img1_msg->header.stamp.toSec(), img1, img2);
}

void VinsNodeBaseClass::imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
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
}

void VinsNodeBaseClass::restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        estimator.clearState();
        estimator.setParameter();
    }
    return;
}


void VinsNodeBaseClass::Init(ros::NodeHandle & n)
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
        FisheyeUndist *fun = nullptr;
        if (USE_GPU) {
            auto ft = (BaseFisheyeFeatureTracker<cv::cuda::GpuMat> *)
                estimator.featureTracker;
            fun = ft->get_fisheye_undist(0);
        } else {
            auto ft = (BaseFisheyeFeatureTracker<cv::Mat> *)
                estimator.featureTracker;
            fun = ft->get_fisheye_undist(0);
        }

        cam_manager = new DepthCamManager(n, fun);
        cam_manager -> init_with_extrinsic(estimator.ric[0], estimator.tic[0], estimator.ric[1], estimator.tic[1]);
        estimator.depth_cam_manager = cam_manager;
    }
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    if (FISHEYE) {
        fisheye_handler = new FisheyeFlattenHandler(n, FLATTEN_COLOR);
    }

    //We use blank images to initialize cuda before every thing
    if (USE_GPU) {
        TicToc blank;
        cv::Mat mat(fisheye_handler->raw_width(), fisheye_handler->raw_height(), CV_8UC3);
        fisheye_handler->imgs_callback(0, mat, mat, true);
            estimator.inputFisheyeImage(0, 
            fisheye_handler->fisheye_up_imgs_cuda_gray, fisheye_handler->fisheye_down_imgs_cuda_gray, true);
        std::cout<< "Initialize with blank cost" << blank.toc() << std::endl;
    }

    sub_imu = n.subscribe(IMU_TOPIC, 2000, &VinsNodeBaseClass::imu_callback, (VinsNodeBaseClass*)this, ros::TransportHints().tcpNoDelay(true));
    sub_restart = n.subscribe("/vins_restart", 100, &VinsNodeBaseClass::restart_callback, (VinsNodeBaseClass*)this, ros::TransportHints().tcpNoDelay(true));

    if (IS_COMP_IMAGES) {
        ROS_INFO("Will directly receive compressed images %s and %s", COMP_IMAGE0_TOPIC.c_str(), COMP_IMAGE1_TOPIC.c_str());
        comp_image_sub_l = new message_filters::Subscriber<sensor_msgs::CompressedImage> (n, COMP_IMAGE0_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
        comp_image_sub_r = new message_filters::Subscriber<sensor_msgs::CompressedImage> (n, COMP_IMAGE1_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
        comp_sync = new message_filters::TimeSynchronizer<sensor_msgs::CompressedImage, sensor_msgs::CompressedImage> (*comp_image_sub_l, *comp_image_sub_r, 1000);
        if (FISHEYE) {
            comp_sync->registerCallback(boost::bind(&VinsNodeBaseClass::fisheye_comp_imgs_callback, (VinsNodeBaseClass*)this, _1, _2));
        } else {    
            comp_sync->registerCallback(boost::bind(&VinsNodeBaseClass::comp_imgs_callback, (VinsNodeBaseClass*)this, _1, _2));
        }
    } else {
        ROS_INFO("Will directly receive raw images %s and %s", IMAGE0_TOPIC.c_str(), IMAGE1_TOPIC.c_str());
        image_sub_l = new message_filters::Subscriber<sensor_msgs::Image> (n, IMAGE0_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
        image_sub_r = new message_filters::Subscriber<sensor_msgs::Image> (n, IMAGE1_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
        sync = new message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> (*image_sub_l, *image_sub_r, 1000);
        if (FISHEYE) {
            sync->registerCallback(boost::bind(&VinsNodeBaseClass::fisheye_imgs_callback, (VinsNodeBaseClass*)this, _1, _2));
        } else {    
            sync->registerCallback(boost::bind(&VinsNodeBaseClass::imgs_callback, (VinsNodeBaseClass*)this, _1, _2));
        }
    }


    timer1 = n.createTimer(ros::Duration(0.004), boost::bind(&VinsNodeBaseClass::processFlattened, (VinsNodeBaseClass*)this, _1 ));
    if (PUB_FLATTEN) {
        timer2 = n.createTimer(ros::Duration(1/PUB_FLATTEN_FREQ), boost::bind(&VinsNodeBaseClass::pack_and_send_thread, (VinsNodeBaseClass*)this, _1 ));
    }
}