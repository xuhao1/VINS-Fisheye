#include "feature_tracker.h"
#include "../estimator/estimator.h"
#include "fisheye_undist.hpp"

#ifdef USE_CUDA
void FeatureTracker::drawTrackFisheye(const cv::Mat & img_up,
    const cv::Mat & img_down,
    cv::cuda::GpuMat imUpTop,
    cv::cuda::GpuMat imDownTop,
    cv::cuda::GpuMat imUpSide_cuda, 
    cv::cuda::GpuMat imDownSide_cuda) {
    cv::Mat a, b, c, d;
    imUpTop.download(a);
    imDownTop.download(b);
    imUpSide_cuda.download(c);
    imDownSide_cuda.download(d);
    drawTrackFisheye(img_up, img_down, a, b, c, d);
}

cv::cuda::GpuMat concat_side(const std::vector<cv::cuda::GpuMat> & arr) {
    int cols = arr[1].cols;
    int rows = arr[1].rows;
    if (enable_rear_side) {
        cv::cuda::GpuMat NewImg(rows, cols*4, arr[1].type()); 
        for (int i = 1; i < 5; i ++) {
            arr[i].copyTo(NewImg(cv::Rect(cols * (i-1), 0, cols, rows)));
        }
        return NewImg;
    } else {
        cv::cuda::GpuMat NewImg(rows, cols*3, arr[1].type()); 
        for (int i = 1; i < 4; i ++) {
            arr[i].copyTo(NewImg(cv::Rect(cols * (i-1), 0, cols, rows)));
        }
        return NewImg;
    }
}



std::vector<cv::Mat> convertCPUMat(const std::vector<cv::cuda::GpuMat> & arr) {
    std::vector<cv::Mat> ret;
    for (const auto & mat:arr) {
        cv::Mat matcpu;
        mat.download(matcpu);
        cv::cvtColor(matcpu, matcpu, cv::COLOR_GRAY2BGR);
        ret.push_back(matcpu);
    }

    return ret;
}


void FeatureTracker::detectPoints(const cv::cuda::GpuMat & img, vector<cv::Point2f> & n_pts, 
        vector<cv::Point2f> & cur_pts, int require_pts) {
    int lack_up_top_pts = require_pts - static_cast<int>(cur_pts.size());

    TicToc tic;
    
    if (lack_up_top_pts > require_pts/4) {
        ROS_INFO("Lack %d pts; Require %d will detect %d", lack_up_top_pts, require_pts, lack_up_top_pts > require_pts/4);
        //Detect top img
        cv::Ptr<cv::cuda::CornersDetector> detector = cv::cuda::createGoodFeaturesToTrackDetector(
            img.type(), lack_up_top_pts, 0.01, MIN_DIST);
        cv::cuda::GpuMat d_prevPts;
        detector->detect(img, d_prevPts);
        // std::cout << "d_prevPts size: "<< d_prevPts.size()<<std::endl;
        if(!d_prevPts.empty()) {
            n_pts = cv::Mat_<cv::Point2f>(cv::Mat(d_prevPts));
        }
        else {
            n_pts.clear();
        }
    }
    else {
        n_pts.clear();
    }
#ifdef PERF_OUTPUT
    ROS_INFO("Detected %ld npts %fms", n_pts.size(), tic.toc());
#endif

 }

vector<cv::Point2f> FeatureTracker::opticalflow_track(cv::cuda::GpuMat & cur_img, 
                        cv::cuda::GpuMat & prev_img, vector<cv::Point2f> & prev_pts, 
                        vector<int> & ids, vector<int> & track_cnt,
                        bool is_lr_track, vector<cv::Point2f> prediction_points){
    if (prev_pts.size() == 0) {
        return vector<cv::Point2f>();
    }
    TicToc tic;
    vector<uchar> status;

    for (size_t i = 0; i < ids.size(); i ++) {
        int _id = ids[i];
        if (removed_pts.find(_id) == removed_pts.end()) {
            status.push_back(1);
        } else {
            status.push_back(0);
        }
    }

    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(ids, status);
    
    if (prev_pts.size() == 0) {
        return vector<cv::Point2f>();
    }

    vector<cv::Point2f> cur_pts;
    TicToc t_og;
    cv::cuda::GpuMat prev_gpu_pts(prev_pts);
    cv::cuda::GpuMat cur_gpu_pts(cur_pts);
    cv::cuda::GpuMat gpu_status;
    status.clear();

    //Assume No Prediction Need to add later
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
        cv::Size(21, 21), 3, 30, false);
    d_pyrLK_sparse->calc(prev_img, cur_img, prev_gpu_pts, cur_gpu_pts, gpu_status);
    
    // std::cout << "Prev gpu pts" << prev_gpu_pts.size() << std::endl;    
    // std::cout << "Cur gpu pts" << cur_gpu_pts.size() << std::endl;
    cur_gpu_pts.download(cur_pts);

    gpu_status.download(status);
    if(FLOW_BACK)
    {
        // ROS_INFO("Is flow back");
        cv::cuda::GpuMat reverse_gpu_status;
        cv::cuda::GpuMat reverse_gpu_pts = prev_gpu_pts;
        cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
        cv::Size(21, 21), 1, 30, true);
        d_pyrLK_sparse->calc(cur_img, prev_img, cur_gpu_pts, reverse_gpu_pts, reverse_gpu_status);

        vector<cv::Point2f> reverse_pts(reverse_gpu_pts.cols);
        reverse_gpu_pts.download(reverse_pts);

        vector<uchar> reverse_status(reverse_gpu_status.cols);
        reverse_gpu_status.download(reverse_status);

        for(size_t i = 0; i < status.size(); i++)
        {
            if(status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
            {
                status[i] = 1;
            }
            else
                status[i] = 0;
        }
    }
    // printf("gpu temporal optical flow costs: %f ms\n",t_og.toc());

    for (int i = 0; i < int(cur_pts.size()); i++){
        if (status[i] && !inBorder(cur_pts[i], cur_img.size())) {
            status[i] = 0;
        }
    }            

    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(ids, status);
    if(track_cnt.size() > 0) {
        reduceVector(track_cnt, status);
    }

#ifdef PERF_OUTPUT
    ROS_INFO("Optical flow costs: %fms Pts %ld", t_og.toc(), ids.size());
#endif

    //printf("track cnt %d\n", (int)ids.size());

    for (auto &n : track_cnt)
        n++;

    return cur_pts;
}

FeatureFrame FeatureTracker::trackImage_fisheye(double _cur_time,   
        const std::vector<cv::cuda::GpuMat> & fisheye_imgs_up,
        const std::vector<cv::cuda::GpuMat> & fisheye_imgs_down,
        bool is_blank_init) {
    cur_time = _cur_time;
    static double detected_time_sum = 0;
    static double count = 0;
    
    if (!is_blank_init) {
        count += 1;
    }

    TicToc t_r;
    cv::cuda::GpuMat up_side_img = concat_side(fisheye_imgs_up);
    cv::cuda::GpuMat down_side_img = concat_side(fisheye_imgs_down);
    cv::cuda::GpuMat up_top_img = fisheye_imgs_up[0];
    cv::cuda::GpuMat down_top_img = fisheye_imgs_down[0];
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

    if (!up_top_img.empty() && up_top_img.channels() == 3) {
        std::cout << "CVT uptop" << std::endl;
        cv::cuda::cvtColor(up_top_img, up_top_img, cv::COLOR_BGR2GRAY);
    }

    if (!down_top_img.empty() && down_top_img.channels() == 3) {
        std::cout << "CVT downtop" << std::endl;
        cv::cuda::cvtColor(down_top_img, down_top_img, cv::COLOR_BGR2GRAY);
    }

    if (!up_side_img.empty() && up_side_img.channels() == 3) {
        std::cout << "CVT upside" << std::endl;
        cv::cuda::cvtColor(up_side_img, up_side_img, cv::COLOR_BGR2GRAY);
    }

    if (!down_side_img.empty() && down_side_img.channels() == 3) {
        std::cout << "CVT downside" << std::endl;
        cv::cuda::cvtColor(down_side_img, down_side_img, cv::COLOR_BGR2GRAY);
    }

    if (enable_up_top) {
        // ROS_INFO("Tracking top");
        cur_up_top_pts = opticalflow_track(up_top_img, prev_up_top_img, prev_up_top_pts, ids_up_top, track_up_top_cnt, false);
    }
    if (enable_up_side) {
        cur_up_side_pts = opticalflow_track(up_side_img, prev_up_side_img, prev_up_side_pts, ids_up_side, track_up_side_cnt, false);
    }

    if (enable_down_top) {
        cur_down_top_pts = opticalflow_track(down_top_img, prev_down_top_img, prev_down_top_pts, ids_down_top, track_down_top_cnt, false);
    }
    
    ROS_INFO("FT %fms", t_r.toc());

    // setMaskFisheye();
    // ROS_INFO("SetMaskFisheye %fms", t_r.toc());
    
    TicToc t_d;
    if (enable_up_top) {
        // ROS_INFO("Detecting top");
        detectPoints(up_top_img, n_pts_up_top, cur_up_top_pts, TOP_PTS_CNT);
    }
    if (enable_down_top) {
        detectPoints(down_top_img, n_pts_down_top, cur_down_top_pts, TOP_PTS_CNT);
    }

    if (enable_up_side) {
        detectPoints(up_side_img, n_pts_up_side, cur_up_side_pts, SIDE_PTS_CNT);
    }


    if (!is_blank_init) {
        ROS_INFO("DetectPoints %fms", t_d.toc());
        detected_time_sum = detected_time_sum + t_d.toc();
    }

    addPointsFisheye();

    if (enable_down_side) {
        ids_down_side = ids_up_side;
        std::vector<cv::Point2f> down_side_init_pts = cur_up_side_pts;
        cur_down_side_pts = opticalflow_track(down_side_img, up_side_img, down_side_init_pts, ids_down_side, track_down_side_cnt, true);
        // ROS_INFO("Down side try to track %ld pts; gives %ld:%ld", cur_up_side_pts.size(), cur_down_side_pts.size(), ids_down_side.size());
    }

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

    printf("FT Whole %fms; MainProcess %fms Detect AVG %fms concat %fms PTS %ld T\n", t_r.toc(), detected_time_sum/count, tcost_all, concat_cost, ff.size());
    return ff;
}
#endif