
#include "feature_tracker.h"
#include "../estimator/estimator.h"
#include "fisheye_undist.hpp"
#include "feature_tracker_fisheye.hpp"

namespace FeatureTracker {



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





FeatureFrame FisheyeFeatureTrackerOpenMP::trackImage(double _cur_time, cv::InputArray img0, cv::InputArray img1) {
    // ROS_INFO("tracking fisheye cpu %ld:%ld", fisheye_imgs_up.size(), fisheye_imgs_down.size());
    cur_time = _cur_time;
    static double count = 0;
    count += 1;

    CvImages fisheye_imgs_up;
    CvImages fisheye_imgs_down;

    img0.getMatVector(fisheye_imgs_up);
    img1.getMatVector(fisheye_imgs_down);
    TicToc t_r;

    cv::Mat up_side_img = concat_side(fisheye_imgs_up);
    cv::Mat down_side_img = concat_side(fisheye_imgs_down);
    cv::Mat & up_top_img = fisheye_imgs_up[0];
    cv::Mat & down_top_img = fisheye_imgs_down[0];

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
    set_predict_lock.lock();

    #pragma omp parallel sections
    {
        #pragma omp section 
        {
            //If has predict;
            if (enable_up_top) {
                // printf("Start track up top\n");
                cur_up_top_pts = opticalflow_track(up_top_img, up_top_pyr, prev_up_top_img, prev_up_top_pyr, 
                    prev_up_top_pts, ids_up_top, track_up_top_cnt, removed_pts, predict_up_top);
                // printf("End track up top\n");
            }
        }

        #pragma omp section 
        {
            if (enable_up_side) {
                // printf("Start track up side\n");
                cur_up_side_pts = opticalflow_track(up_side_img, up_side_pyr, prev_up_side_img, prev_up_side_pyr, 
                    prev_up_side_pts, ids_up_side, track_up_side_cnt, removed_pts, predict_up_side);
                // printf("End track up side\n");
            }
        }

        #pragma omp section 
        {
            if (enable_down_top) {
                // printf("Start track down top\n");
                cur_down_top_pts = opticalflow_track(down_top_img, down_top_pyr, prev_down_top_img, prev_down_top_pyr, 
                    prev_down_top_pts, ids_down_top, track_down_top_cnt, removed_pts, predict_down_top);
                // printf("End track down top\n");
            }
        }
    }
       
    set_predict_lock.unlock();
    

    static double lk_sum = 0;
    lk_sum += t_t.toc();

    TicToc t_d;

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (enable_up_top) {
                detectPoints(up_top_img, cv::Mat(), n_pts_up_top, cur_up_top_pts, TOP_PTS_CNT);
            }
        }

        #pragma omp section
        {
            if (enable_down_top) {
                detectPoints(down_top_img, cv::Mat(), n_pts_down_top, cur_down_top_pts, TOP_PTS_CNT);
            }
        }

        #pragma omp section
        {
            if (enable_up_side) {
                detectPoints(up_side_img, cv::Mat(), n_pts_up_side, cur_up_side_pts, SIDE_PTS_CNT);
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
                cur_down_side_pts = opticalflow_track(down_side_img, down_side_pyr, up_side_img, 
                    up_side_pyr, down_side_init_pts, ids_down_side, track_down_side_cnt, removed_pts, predict_down_side);
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

        
    prev_up_top_img = up_top_img;
    prev_down_top_img = down_top_img;
    prev_up_side_img = up_side_img;

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


};