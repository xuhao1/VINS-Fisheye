
#include "feature_tracker.h"
#include "../estimator/estimator.h"
#include "fisheye_undist.hpp"
#include "feature_tracker_fisheye.hpp"

namespace FeatureTracker {
Eigen::Quaterniond t1(Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d(1, 0, 0)));
Eigen::Quaterniond t2 = t1 * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
Eigen::Quaterniond t3 = t2 * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
Eigen::Quaterniond t4 = t3 * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
Eigen::Quaterniond t_down(Eigen::AngleAxisd(M_PI, Eigen::Vector3d(1, 0, 0)));

template<class CvMat>
void BaseFisheyeFeatureTracker<CvMat>::addPointsFisheye()
{
    // ROS_INFO("Up top new pts %d", n_pts_up_top.size());
    for (auto p : n_pts_up_top)
    {
        cur_up_top_pts.push_back(p);
        ids_up_top.push_back(n_id++);
        track_up_top_cnt.push_back(1);
    }

    for (auto p : n_pts_down_top)
    {
        cur_down_top_pts.push_back(p);
        ids_down_top.push_back(n_id++);
        track_down_top_cnt.push_back(1);
    }

    for (auto p : n_pts_up_side)
    {
        cur_up_side_pts.push_back(p);
        ids_up_side.push_back(n_id++);
        track_up_side_cnt.push_back(1);
    }
}

cv::Mat concat_side(const std::vector<cv::Mat> & arr) {
    int cols = arr[1].cols;
    int rows = arr[1].rows;
    if (enable_rear_side) {
        cv::Mat NewImg(rows, cols*4, arr[1].type()); 
        for (int i = 1; i < 5; i ++) {
            arr[i].copyTo(NewImg(cv::Rect(cols * (i-1), 0, cols, rows)));
        }
        return NewImg;
    } else {
        cv::Mat NewImg(rows, cols*3, arr[1].type()); 
        for (int i = 1; i < 4; i ++) {
            arr[i].copyTo(NewImg(cv::Rect(cols * (i-1), 0, cols, rows)));
        }
        return NewImg;
    }
}

template<class CvMat>
void BaseFisheyeFeatureTracker<CvMat>::drawTrackFisheye(const cv::Mat & img_up,
    const cv::Mat & img_down,
    cv::Mat imUpTop,
    cv::Mat imDownTop,
    cv::Mat imUpSide, 
    cv::Mat imDownSide)
{
    // ROS_INFO("Up image %d, down %d", imUp.size(), imDown.size());
    cv::Mat imTrack;
    cv::Mat fisheye_up;
    cv::Mat fisheye_down;
    
    int side_height = imUpSide.size().height;

    int cnt = 0;

    if (imUpTop.size().width == 0) {
        imUpTop = cv::Mat(cv::Size(WIDTH, WIDTH), CV_8UC3, cv::Scalar(0, 0, 0));
        cnt ++; 
    }

    if (imDownTop.size().width == 0) {
        imDownTop = cv::Mat(cv::Size(WIDTH, WIDTH), CV_8UC3, cv::Scalar(0, 0, 0)); 
        cnt ++; 
    }

    //128
    if (img_up.size().width == 1024) {
        fisheye_up = img_up(cv::Rect(190, 62, 900, 900));
        fisheye_down = img_down(cv::Rect(190, 62, 900, 900));
    } else {
        fisheye_up = cv::Mat(cv::Size(900, 900), CV_8UC3, cv::Scalar(0, 0, 0)); 
        fisheye_down = cv::Mat(cv::Size(900, 900), CV_8UC3, cv::Scalar(0, 0, 0)); 
        cnt ++; 
    }

    cv::resize(fisheye_up, fisheye_up, cv::Size(WIDTH, WIDTH));
    cv::resize(fisheye_down, fisheye_down, cv::Size(WIDTH, WIDTH));
    if (fisheye_up.channels() != 3) {
        cv::cvtColor(fisheye_up,   fisheye_up,   cv::COLOR_GRAY2BGR);
        cv::cvtColor(fisheye_down, fisheye_down, cv::COLOR_GRAY2BGR);
    }

    if (imUpTop.channels() != 3) {
        if (!imUpTop.empty()) {
            cv::cvtColor(imUpTop, imUpTop, cv::COLOR_GRAY2BGR);
        }
    }
    
    if (imDownTop.channels() != 3) {
        if(!imDownTop.empty()) {
            cv::cvtColor(imDownTop, imDownTop, cv::COLOR_GRAY2BGR);
        }
    }
    
    if(imUpSide.channels() != 3) {
        if(!imUpSide.empty()) {
            cv::cvtColor(imUpSide, imUpSide, cv::COLOR_GRAY2BGR);
        }
    }

    if(imDownSide.channels() != 3) {
        if(!imDownSide.empty()) {
            cv::cvtColor(imDownSide, imDownSide, cv::COLOR_GRAY2BGR);
        }
    }

    if(enable_up_top) {
        drawTrackImage(imUpTop, cur_up_top_pts, ids_up_top, up_top_prevLeftPtsMap);
    }

    if(enable_down_top) {
        drawTrackImage(imDownTop, cur_down_top_pts, ids_down_top, down_top_prevLeftPtsMap);
    }

    if(enable_up_side) {
        drawTrackImage(imUpSide, cur_up_side_pts, ids_up_side, up_side_prevLeftPtsMap);
    }

    if(enable_down_side) {
        drawTrackImage(imDownSide, cur_down_side_pts, ids_down_side, pts_map(ids_up_side, cur_up_side_pts));
    }

    //Show images
    int side_count = 3;
    if (enable_rear_side) {
        side_count = 4;
    }

    for (int i = 1; i < side_count + 1; i ++) {
        cv::line(imUpSide, cv::Point2d(i*WIDTH, 0), cv::Point2d(i*WIDTH, side_height), cv::Scalar(255, 0, 0), 1);
        cv::line(imDownSide, cv::Point2d(i*WIDTH, 0), cv::Point2d(i*WIDTH, side_height), cv::Scalar(255, 0, 0), 1);
    }
vector<cv::Point3f> BaseFeatureTracker::undistortedPtsTop(vector<cv::Point2f> &pts, FisheyeUndist & fisheye) {
    auto & cam = fisheye.cam_top;
    vector<cv::Point3f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);
        b.normalize();
#ifdef UNIT_SPHERE_ERROR
        un_pts.push_back(cv::Point3f(b.x(), b.y(), b.z()));
#else
        un_pts.push_back(cv::Point3f(b.x() / b.z(), b.y() / b.z(), 1));
#endif
    }
    return un_pts;
}

    cv::Mat top_cam;


    cv::hconcat(imUpTop, imDownTop, top_cam);
    cv::hconcat(fisheye_up, top_cam, top_cam);
    cv::hconcat(top_cam, fisheye_down, top_cam); 
    // ROS_INFO("Imtrack width %d", imUpSide.size().width);
    cv::resize(top_cam, top_cam, cv::Size(imUpSide.size().width, imUpSide.size().width/4));
    
    if (cnt < 3) {
        cv::vconcat(top_cam, imTrack, imTrack);
    }
    
    double fx = ((double)SHOW_WIDTH) / ((double) imUpSide.size().width);
    cv::resize(imTrack, imTrack, cv::Size(), fx, fx);
    cv::imshow("tracking", imTrack);
    cv::waitKey(2);
}


template<class CvMat>
vector<cv::Point3f> BaseFisheyeFeatureTracker<CvMat>::undistortedPtsTop(vector<cv::Point2f> &pts, FisheyeUndist & fisheye) {
    auto & cam = fisheye.cam_top;
    vector<cv::Point3f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);
        b.normalize();
#ifdef UNIT_SPHERE_ERROR
        un_pts.push_back(cv::Point3f(b.x(), b.y(), b.z()));
#else
        un_pts.push_back(cv::Point3f(b.x() / b.z(), b.y() / b.z(), 1));
#endif
    }
    return un_pts;
}

template<class CvMat>
FeatureFrame BaseFisheyeFeatureTracker<CvMat>::setup_feature_frame() override {
    FeatureFrame ff;
    setup_feature_frame(ff, ids_up_top, cur_up_top_pts, cur_up_top_un_pts, up_top_vel, 0);   
    setup_feature_frame(ff, ids_up_side, cur_up_side_pts, cur_up_side_un_pts, up_side_vel, 0);
    setup_feature_frame(ff, ids_down_top, cur_down_top_pts, cur_down_top_un_pts, down_top_vel, 1);
    setup_feature_frame(ff, ids_down_side, cur_down_side_pts, cur_down_side_un_pts, down_side_vel, 1);

    return ff;
}


template<class CvMat>
void BaseFisheyeFeatureTracker<CvMat>::readIntrinsicParameter(const vector<string> &calib_file)
{
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);

        ROS_INFO("Use as fisheye %s", calib_file[i].c_str());
        FisheyeUndist un(calib_file[i].c_str(), i, FISHEYE_FOV, true, WIDTH);
        fisheys_undists.push_back(un);

    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
}

template<class CvMat>
vector<cv::Point3f> BaseFisheyeFeatureTracker<CvMat>::ptsVelocity3D(vector<int> &ids, vector<cv::Point3f> &cur_pts, 
                                            map<int, cv::Point3f> &cur_id_pts, map<int, cv::Point3f> &prev_id_pts)
{
    // ROS_INFO("Pts %ld Prev pts %ld IDS %ld", cur_pts.size(), prev_id_pts.size(), ids.size());
    vector<cv::Point3f> pts_velocity;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], cur_pts[i]));
    }

    // caculate points velocity
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time;
        
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            std::map<int, cv::Point3f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end())
            {
                double v_x = (cur_pts[i].x - it->second.x) / dt;
                double v_y = (cur_pts[i].y - it->second.y) / dt;
                double v_z = (cur_pts[i].z - it->second.z) / dt;
                pts_velocity.push_back(cv::Point3f(v_x, v_y, v_z));
                // ROS_INFO("Dt %f, vel %f %f %f", v_x, v_y, v_z);

            }
            else
                pts_velocity.push_back(cv::Point3f(0, 0, 0));

        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point3f(0, 0, 0));
        }
    }
    return pts_velocity;
}

FeatureFrame FisheyeFeatureTrackerOMP::trackImage(double _cur_time, InputArray fisheye_imgs_up, InputArray fisheye_imgs_down) {
    // ROS_INFO("tracking fisheye cpu %ld:%ld", fisheye_imgs_up.size(), fisheye_imgs_down.size());
    cur_time = _cur_time;
    static double count = 0;
    count += 1;

    TicToc t_r;

    cv::Mat up_side_img = concat_side(fisheye_imgs_up);
    cv::Mat down_side_img = concat_side(fisheye_imgs_down);
    cv::Mat up_top_img = fisheye_imgs_up[0];
    cv::Mat down_top_img = fisheye_imgs_down[0];

    std::vector<cv::Mat> * up_top_pyr = nullptr, * down_top_pyr = nullptr, * up_side_pyr = nullptr, * down_side_pyr = nullptr;
    double concat_cost = t_r.toc();

    top_size = up_top_img.size();
    side_size = up_side_img.size();

    //Clear All current pts
    cur_up_top_pts.clear();
    cur_up_side_pts.clear();
    cur_down_top_pts.clear();
    cur_down_side_pts.clear();

    cur_up_top_un_pts.clear();
    cur_up_side_un_pts.clear();
    cur_down_top_un_pts.clear();
    cur_down_side_un_pts.clear();


    TicToc t_pyr;
    #pragma omp parallel sections 
    {
        #pragma omp section 
        {
            if(enable_up_top) {
                // printf("Building up top pyr\n");
                up_top_pyr = new std::vector<cv::Mat>();
                cv::buildOpticalFlowPyramid(up_top_img, *up_top_pyr, WIN_SIZE, PYR_LEVEL, true);//, cv::BORDER_REFLECT101, cv::BORDER_CONSTANT, false);
            }
        }
        
        #pragma omp section 
        {
            if(enable_down_top) {
                // printf("Building down top pyr\n");
                down_top_pyr = new std::vector<cv::Mat>();
                cv::buildOpticalFlowPyramid(down_top_img, *down_top_pyr, WIN_SIZE, PYR_LEVEL, true);
            }
        }
        
        #pragma omp section 
        {
            if(enable_up_side) {
                // printf("Building up side pyr\n");
                up_side_pyr = new std::vector<cv::Mat>();
                cv::buildOpticalFlowPyramid(up_side_img, *up_side_pyr, WIN_SIZE, PYR_LEVEL, true);
            }
        }
        
        #pragma omp section 
        {
            if(enable_down_side) {
                // printf("Building downn side pyr\n");
                down_side_pyr = new std::vector<cv::Mat>();
                cv::buildOpticalFlowPyramid(down_side_img, *down_side_pyr, WIN_SIZE, PYR_LEVEL, true);
            }
        }
    }

    static double pyr_sum = 0;
    pyr_sum += t_pyr.toc();

    TicToc t_t;
    #pragma omp parallel sections
    {
        #pragma omp section 
        {
            //If has predict;
            if (enable_up_top) {
                // printf("Start track up top\n");
                cur_up_top_pts = opticalflow_track(up_top_img, up_top_pyr, prev_up_top_img_cpu, prev_up_top_pyr, prev_up_top_pts, ids_up_top, track_up_top_cnt);
                // printf("End track up top\n");
            }
        }

        #pragma omp section 
        {
            if (enable_up_side) {
                // printf("Start track up side\n");
                cur_up_side_pts = opticalflow_track(up_side_img, up_side_pyr, prev_up_side_img_cpu, prev_up_side_pyr, prev_up_side_pts, ids_up_side, track_up_side_cnt);
                // printf("End track up side\n");
            }
        }

        #pragma omp section 
        {
            if (enable_down_top) {
                // printf("Start track down top\n");
                cur_down_top_pts = opticalflow_track(down_top_img, down_top_pyr, prev_down_top_img_cpu, prev_down_top_pyr, prev_down_top_pts, ids_down_top, track_down_top_cnt);
                // printf("End track down top\n");
            }
        }

        
       
    }
    

    static double lk_sum = 0;
    lk_sum += t_t.toc();

    TicToc t_d;

    setMaskFisheye();

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (enable_up_top) {
                detectPoints(up_top_img, mask_up_top, n_pts_up_top, cur_up_top_pts, TOP_PTS_CNT);
            }
        }

        #pragma omp section
        {
            if (enable_down_top) {
                detectPoints(down_top_img, mask_down_top, n_pts_down_top, cur_down_top_pts, TOP_PTS_CNT);
            }
        }

        #pragma omp section
        {
            if (enable_up_side) {
                detectPoints(up_side_img, mask_up_side, n_pts_up_side, cur_up_side_pts, SIDE_PTS_CNT);
            }
        }
    }

    // ROS_INFO("Detect cost %fms", t_d.toc());

    static double detect_sum = 0;

    detect_sum = detect_sum + t_d.toc();

    addPointsFisheye();
    
    TicToc t_tk;
    {
        if (enable_down_side) {
            ids_down_side = ids_up_side;
            std::vector<cv::Point2f> down_side_init_pts = cur_up_side_pts;
            if (down_side_init_pts.size() > 0) {
                cur_down_side_pts = opticalflow_track(down_side_img, down_side_pyr, up_side_img, up_side_pyr, down_side_init_pts, ids_down_side, track_down_side_cnt);
            }
        }
    }

    // ROS_INFO("Tracker 2 cost %fms", t_tk.toc());

    //Undist points
    cur_up_top_un_pts = undistortedPtsTop(cur_up_top_pts, fisheys_undists[0]);
    cur_down_top_un_pts = undistortedPtsTop(cur_down_top_pts, fisheys_undists[1]);

    cur_up_side_un_pts = undistortedPtsSide(cur_up_side_pts, fisheys_undists[0], false);
    cur_down_side_un_pts = undistortedPtsSide(cur_down_side_pts, fisheys_undists[1], true);

    //Calculate Velocitys
    up_top_vel = ptsVelocity3D(ids_up_top, cur_up_top_un_pts, cur_up_top_un_pts_map, prev_up_top_un_pts_map);
    down_top_vel = ptsVelocity3D(ids_down_top, cur_down_top_un_pts, cur_down_top_un_pts_map, prev_down_top_un_pts_map);

    up_side_vel = ptsVelocity3D(ids_up_side, cur_up_side_un_pts, cur_up_side_un_pts_map, prev_up_side_un_pts_map);
    down_side_vel = ptsVelocity3D(ids_down_side, cur_down_side_un_pts, cur_down_side_un_pts_map, prev_down_side_un_pts_map);

    // ROS_INFO("Up top VEL %ld", up_top_vel.size());
    double tcost_all = t_r.toc();
    if (SHOW_TRACK) {
        drawTrackFisheye(cv::Mat(), cv::Mat(), up_top_img, down_top_img, up_side_img, down_side_img);
    }

        
    prev_up_top_img_cpu = up_top_img;
    prev_down_top_img_cpu = down_top_img;
    prev_up_side_img_cpu = up_side_img;

    if(prev_down_top_pyr != nullptr) {
        delete prev_down_top_pyr;
    }

    if(prev_up_top_pyr != nullptr) {
        delete prev_up_top_pyr;
    }

    if (prev_up_side_pyr!=nullptr) {
        delete prev_up_side_pyr;
    }

    if (down_side_pyr!=nullptr) {
        delete down_side_pyr;
    }

    prev_down_top_pyr = down_top_pyr;
    prev_up_top_pyr = up_top_pyr;
    prev_up_side_pyr = up_side_pyr;

    prev_up_top_pts = cur_up_top_pts;
    prev_down_top_pts = cur_down_top_pts;
    prev_up_side_pts = cur_up_side_pts;
    prev_down_side_pts = cur_down_side_pts;

    prev_up_top_un_pts = cur_up_top_un_pts;
    prev_down_top_un_pts = cur_down_top_un_pts;
    prev_up_side_un_pts = cur_up_side_un_pts;
    prev_down_side_un_pts = cur_down_side_un_pts;

    prev_up_top_un_pts_map = cur_up_top_un_pts_map;
    prev_down_top_un_pts_map = cur_down_top_un_pts_map;
    prev_up_side_un_pts_map = cur_up_side_un_pts_map;
    prev_down_side_un_pts_map = cur_up_side_un_pts_map;
    prev_time = cur_time;

    up_top_prevLeftPtsMap = pts_map(ids_up_top, cur_up_top_pts);
    down_top_prevLeftPtsMap = pts_map(ids_down_top, cur_down_top_pts);
    up_side_prevLeftPtsMap = pts_map(ids_up_side, cur_up_side_pts);
    down_side_prevLeftPtsMap = pts_map(ids_down_side, cur_down_side_pts);

    // hasPrediction = false;
    auto ff = setup_feature_frame();
    
    static double whole_sum = 0.0;

    whole_sum += t_r.toc();

    printf("FT Whole %fms; AVG %fms\n DetectAVG %fms PYRAvg %fms LKAvg %fms Concat %fms PTS %ld T\n", 
        t_r.toc(), whole_sum/count, detect_sum/count, pyr_sum/count, lk_sum/count, concat_cost, ff.size());
    return ff;
}


template<class CvMat>
vector<cv::Point3f> BaseFisheyeFeatureTracker<class CvMat>::undistortedPtsSide(vector<cv::Point2f> &pts, FisheyeUndist & fisheye, bool is_downward) {
    auto & cam = fisheye.cam_side;
    std::vector<cv::Point3f> un_pts;
    //Need to rotate pts
    //Side pos 1,2,3,4 is left front right
    //For downward camera, additational rotate 180 deg on x is required


    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        
        int side_pos_id = floor(a.x() / WIDTH) + 1;

        a.x() = a.x() - floor(a.x() / WIDTH)*WIDTH;

        cam->liftProjective(a, b);

        if (side_pos_id == 1) {
            b = t1 * b;
        } else if(side_pos_id == 2) {
            b = t2 * b;
        } else if (side_pos_id == 3) {
            b = t3 * b;
        } else if (side_pos_id == 4) {
            b = t4 * b;
        } else {
            ROS_ERROR("Err pts img position; i %d side_pos_id %d!! x %f width %d", i, side_pos_id, a.x(), top_size.width);
            assert(false &&"ERROR Pts img position");
        }

        if (is_downward) {
            b = t_down * b;
        }

        b.normalize();
#ifdef UNIT_SPHERE_ERROR
        un_pts.push_back(cv::Point3f(b.x(), b.y(), b.z()));
#else
        if (fabs(b.z()) < 1e-3) {
            b.z() = 1e-3;
        }
        
        if (b.z() < - 1e-2) {
            //Is under plane, z is -1
            un_pts.push_back(cv::Point3f(b.x() / b.z(), b.y() / b.z(), -1));
        } else if (b.z() > 1e-2) {
            //Is up plane, z is 1
            un_pts.push_back(cv::Point3f(b.x() / b.z(), b.y() / b.z(), 1));
        }
#endif
    }
    return un_pts;
}
};