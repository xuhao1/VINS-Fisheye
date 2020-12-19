/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com), Xu Hao (xuhao3e8@gmail.com)
 *******************************************************/

#include "feature_tracker.h"
#include "../estimator/estimator.h"
#include "fisheye_undist.hpp"


namespace FeatureTracker {


void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}





double distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}



std::vector<cv::Point2f> detect_orb_by_region(cv::InputArray _img, cv::InputArray _mask, int features, int cols, int rows) {
    int small_width = _img.cols() / cols;
    int small_height = _img.rows() / rows;
    
    auto _orb = cv::ORB::create(10);
    std::vector<cv::Point2f> ret;
    for (int i = 0; i < cols; i ++) {
        for (int j = 0; j < rows; j ++) {
            std::vector<cv::KeyPoint> kpts;
            cv::Rect roi(small_width*i, small_height*j, small_width, small_height);
            std::cout << "ROI " << roi << "Img " << _img.size() << std::endl;
            _orb->detect(_img.getMat()(roi), kpts, _mask.getMat()(roi));
            printf("Detected %ld features in reigion (%d, %d)\n", kpts.size(), i, j);

            for (auto kp : kpts) {
                kp.pt.x = kp.pt.x + small_width*i;
                kp.pt.y = kp.pt.y + small_width*j;
                ret.push_back(kp.pt);
            }
        }
    }

    return ret;
}


bool inBorder(const cv::Point2f &pt, cv::Size shape)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < shape.width - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < shape.height - BORDER_SIZE;
}

vector<cv::Point2f> get_predict_pts(vector<int> ids, const vector<cv::Point2f> & cur_pt, const std::map<int, cv::Point2f> & predict) {
    assert(ids.size() == cur_pt.size() && "[get_predict_pts] IDS must same size as cur pt");
    std::vector<cv::Point2f> ret(cur_pt.size());
    for (size_t i = 0; i < ids.size(); i++) {
        if(predict.find(ids[i]) != predict.end()) {
            ret[i] = predict.at(ids[i]);
        } else {
            ret[i] = cur_pt.at(i);
        }
    }
    
    return ret;
}


void detectPoints(cv::InputArray img, cv::InputArray mask, vector<cv::Point2f> & n_pts, vector<cv::Point2f> & cur_pts, int require_pts) {
    int lack_up_top_pts = require_pts - static_cast<int>(cur_pts.size());

    //Add Points Top
    TicToc tic;
    ROS_INFO("Lost %d pts; Require %d will detect %d", lack_up_top_pts, require_pts, lack_up_top_pts > require_pts/4);
    if (lack_up_top_pts > require_pts/4) {
        if(mask.empty())
            cout << "mask is empty " << endl;
        if (mask.type() != CV_8UC1)
            cout << "mask type wrong " << endl;
        
        if (!USE_ORB) {
            cv::Mat d_prevPts;
            cv::goodFeaturesToTrack(img, d_prevPts, lack_up_top_pts, 0.01, MIN_DIST, mask);
            if(!d_prevPts.empty()) {
                n_pts = cv::Mat_<cv::Point2f>(cv::Mat(d_prevPts));
            }
            else {
                n_pts.clear();
            }
        } else {
            if (img.cols() == img.rows()) {
                n_pts = detect_orb_by_region(img, mask, lack_up_top_pts, 4, 4);
            } else {
                n_pts = detect_orb_by_region(img, mask, lack_up_top_pts, 3, 1);
            }
        }

    }
    else {
        n_pts.clear();
    }
#ifdef PERF_OUTPUT
    ROS_INFO("Detected %ld npts %fms", n_pts.size(), tic.toc());
#endif

 }

void BaseFeatureTracker::setup_feature_frame(FeatureFrame & ff, vector<int> ids, vector<cv::Point2f> cur_pts, vector<cv::Point3f> cur_un_pts, vector<cv::Point3f> cur_pts_vel, int camera_id) {
    // ROS_INFO("Setup feature frame pts %ld un pts %ld vel %ld on Camera %d", cur_pts.size(), cur_un_pts.size(), cur_pts_vel.size(), camera_id);
    for (size_t i = 0; i < ids.size(); i++)
    {
        int feature_id = ids[i];
        double x, y ,z;
        x = cur_un_pts[i].x;
        y = cur_un_pts[i].y;
        z = cur_un_pts[i].z;
        double p_u, p_v;
        p_u = cur_pts[i].x;
        p_v = cur_pts[i].y;
        double velocity_x, velocity_y, velocity_z;
        velocity_x = cur_pts_vel[i].x;
        velocity_y = cur_pts_vel[i].y;
        velocity_z = cur_pts_vel[i].z;

        TrackFeatureNoId xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y, velocity_z;

        // ROS_INFO("FeaturePts Id %d; Cam %d; pos %f, %f, %f uv %f, %f, vel %f, %f, %f", feature_id, camera_id,
            // x, y, z, p_u, p_v, velocity_x, velocity_y, velocity_z);
        ff[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }
 }



vector<cv::Point2f> opticalflow_track(vector<cv::Mat> * cur_pyr, 
                        vector<cv::Mat> * prev_pyr, vector<cv::Point2f> & prev_pts, 
                        vector<int> & ids, vector<int> & track_cnt, std::set<int> removed_pts, std::map<int, cv::Point2f> prediction_points) {
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
    reduceVector(ids, status);
    reduceVector(track_cnt, status);

    if (prev_pts.size() == 0) {
        return vector<cv::Point2f>();
    }

    vector<cv::Point2f> cur_pts = get_predict_pts(ids, prev_pts, prediction_points);
    TicToc t_og;
    status.clear();
    vector<float> err;
    cv::calcOpticalFlowPyrLK(*prev_pyr, *cur_pyr, prev_pts, cur_pts, status, err, WIN_SIZE, PYR_LEVEL);
    std::cout << "Prev pts" << prev_pts.size() << std::endl;    
    if(FLOW_BACK)
    {
        vector<cv::Point2f> reverse_pts;
        vector<uchar> reverse_status;
        cv::calcOpticalFlowPyrLK(*cur_pyr, *prev_pyr, cur_pts, reverse_pts, reverse_status, err, WIN_SIZE, PYR_LEVEL);

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

    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);

#ifdef PERF_OUTPUT
    ROS_INFO("Optical flow costs: %fms Pts %ld", t_og.toc(), ids.size());
#endif

    //printf("track cnt %d\n", (int)ids.size());

    for (auto &n : track_cnt)
        n++;

    return cur_pts;
}

vector<cv::Point2f> opticalflow_track(cv::Mat & cur_img, vector<cv::Mat> * cur_pyr, 
                        cv::Mat & prev_img, vector<cv::Mat> * prev_pyr, vector<cv::Point2f> & prev_pts, 
                        vector<int> & ids, vector<int> & track_cnt, std::set<int> removed_pts, std::map<int, cv::Point2f> prediction_points) {
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
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
    
    if (prev_pts.size() == 0) {
        return vector<cv::Point2f>();
    }

    vector<cv::Point2f> cur_pts = get_predict_pts(ids, prev_pts, prediction_points);

    TicToc t_og;
    status.clear();
    vector<float> err;
    
    TicToc t_build;

    TicToc t_calc;
    cv::calcOpticalFlowPyrLK(*prev_pyr, *cur_pyr, prev_pts, cur_pts, status, err, WIN_SIZE, PYR_LEVEL, 
        cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
    // cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, WIN_SIZE, PYR_LEVEL);
    // std::cout << "Track img Prev pts" << prev_pts.size() << " TS " << t_calc.toc() << std::endl;    
    if(FLOW_BACK)
    {
        vector<cv::Point2f> reverse_pts;
        vector<uchar> reverse_status;
        cv::calcOpticalFlowPyrLK(*cur_pyr, *prev_pyr, cur_pts, reverse_pts, reverse_status, err, WIN_SIZE, PYR_LEVEL,
            cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
        // cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, WIN_SIZE, PYR_LEVEL);

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

    // std::cout << "Cur pts" << cur_pts.size() << std::endl;


#ifdef PERF_OUTPUT
    ROS_INFO("Optical flow costs: %fms Pts %ld", t_og.toc(), ids.size());
#endif

    //printf("track cnt %d\n", (int)ids.size());

    for (auto &n : track_cnt)
        n++;

    return cur_pts;
} 

map<int, cv::Point2f> pts_map(vector<int> ids, vector<cv::Point2f> cur_pts) {
    map<int, cv::Point2f> prevMap;
    for (unsigned int i = 0; i < ids.size(); i ++) {
        prevMap[ids[i]] = cur_pts[i];
    }
    return prevMap;
}

void BaseFeatureTracker::drawTrackImage(cv::Mat & img, vector<cv::Point2f> pts, vector<int> ids, map<int, cv::Point2f> prev_pts, map<int, cv::Point2f> predictions) {
    char idtext[10] = {0};
    for (size_t j = 0; j < pts.size(); j++) {
        //Not tri
        //Not solving
        //Just New point yellow
        cv::Scalar color = cv::Scalar(0, 255, 255);
        if (pts_status.find(ids[j]) != pts_status.end()) {
            int status = pts_status[ids[j]];
            if (status < 0) {
                //Removed points
                color = cv::Scalar(0, 0, 0);
            }

            if (status == 1) {
                //Good pt; But not used for solving; Blue 
                color = cv::Scalar(255, 0, 0);
            }

            if (status == 2) {
                //Bad pt; Red
                color = cv::Scalar(0, 0, 255);
            }

            if (status == 3) {
                //Good pt for solving; Green
                color = cv::Scalar(0, 255, 0);
            }

        }

        cv::circle(img, pts[j], 1, color, 2);
        sprintf(idtext, "%d", ids[j]);
	    cv::putText(img, idtext, pts[j] - cv::Point2f(5, 0), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }

    // for (auto it : predictions) {
    //     auto _id = it.first;
    //     auto pt = it.second;
    //     cv::Scalar color(255, 255, 0);
    //     cv::circle(img, pt, 1, color, 2);

    //     sprintf(idtext, "P%d", _id);
	//     cv::putText(img, idtext, pt + cv::Point2f(5, 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    // }

    for (size_t i = 0; i < ids.size(); i++)
    {
        int id = ids[i];
        auto mapIt = prev_pts.find(id);
        if(mapIt != prev_pts.end()) {
            cv::arrowedLine(img, mapIt->second, pts[i], cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }
}

#ifdef USE_CUDA
vector<cv::Point2f> opticalflow_track(cv::cuda::GpuMat & cur_img, 
                        std::vector<cv::cuda::GpuMat> & prev_pyr, vector<cv::Point2f> & prev_pts, 
                        vector<int> & ids, vector<int> & track_cnt, std::set<int> removed_pts,
                        bool is_lr_track, std::map<int, cv::Point2f> prediction_points) {


    TicToc tic1;
    auto cur_pyr = buildImagePyramid(cur_img);
    
    if (prev_pts.size() == 0) {
        if (!is_lr_track)
            prev_pyr = cur_pyr;
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
    reduceVector(ids, status);
    if(track_cnt.size() > 0) {
        reduceVector(track_cnt, status);
    }

    if (prev_pts.size() == 0) {
        if (!is_lr_track)
            prev_pyr = cur_pyr;
        return vector<cv::Point2f>();
    }

    vector<cv::Point2f> cur_pts = get_predict_pts(ids, prev_pts, prediction_points);

    TicToc t_og;
    cv::cuda::GpuMat prev_gpu_pts(prev_pts);
    cv::cuda::GpuMat cur_gpu_pts(cur_pts);
    cv::cuda::GpuMat gpu_status;
    cv::cuda::GpuMat gpu_err;
    vector<float> err;
    status.clear();

    //Assume No Prediction Need to add later
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
        cv::Size(21, 21), 3, 30, true);

    d_pyrLK_sparse->calc(prev_pyr, cur_pyr, prev_gpu_pts, cur_gpu_pts, gpu_status, gpu_err);
    
    cur_gpu_pts.download(cur_pts);
    gpu_err.download(err);

    gpu_status.download(status);

    if(FLOW_BACK)
    {
        // ROS_INFO("Is flow back");
        cv::cuda::GpuMat reverse_gpu_status;
        cv::cuda::GpuMat reverse_gpu_pts = prev_gpu_pts;
        cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
            cv::Size(21, 21), 3, 30, true);
        d_pyrLK_sparse->calc(cur_pyr, prev_pyr, cur_gpu_pts, reverse_gpu_pts, reverse_gpu_status);

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

    if (ENABLE_PERF_OUTPUT) {
        ROS_INFO("Optical flow costs: %fms Pts %ld", t_og.toc(), ids.size());
    }

    //printf("track cnt %d\n", (int)ids.size());
    if (!is_lr_track)
        prev_pyr = cur_pyr;

    for (auto &n : track_cnt)
        n++;

    return cur_pts;
}

std::vector<cv::cuda::GpuMat> buildImagePyramid(const cv::cuda::GpuMat& prevImg, int maxLevel_) {
    std::vector<cv::cuda::GpuMat> prevPyr;
    prevPyr.resize(maxLevel_ + 1);

    int cn = prevImg.channels();

    CV_Assert(cn == 1 || cn == 3 || cn == 4);

    prevPyr[0] = prevImg;
    for (int level = 1; level <= maxLevel_; ++level) {
        cv::cuda::pyrDown(prevPyr[level - 1], prevPyr[level]);
    }

    return prevPyr;
}

void detectPoints(const cv::cuda::GpuMat & img, vector<cv::Point2f> & n_pts, 
        vector<cv::Point2f> & cur_pts, int require_pts) {
    int lack_up_top_pts = require_pts - static_cast<int>(cur_pts.size());

    TicToc tic;
    

    if (lack_up_top_pts > require_pts/4) {

        // ROS_INFO("Lack %d pts; Require %d will detect %d", lack_up_top_pts, require_pts, lack_up_top_pts > require_pts/4);
        cv::Ptr<cv::cuda::CornersDetector> detector = cv::cuda::createGoodFeaturesToTrackDetector(
            img.type(), lack_up_top_pts, 0.01, MIN_DIST);
        cv::cuda::GpuMat d_prevPts;
        detector->detect(img, d_prevPts);


        std::vector<cv::Point2f> n_pts_tmp;

        // std::cout << "d_prevPts size: "<< d_prevPts.size()<<std::endl;
        if(!d_prevPts.empty()) {
            n_pts_tmp = cv::Mat_<cv::Point2f>(cv::Mat(d_prevPts));
        }
        else {
            n_pts_tmp.clear();
        }

        n_pts.clear();

        if (cur_pts.size() > 0) {
            cv::flann::KDTreeIndexParams indexParams;
            // std::cout << cv::Mat(cur_pts).reshape(1) << std::endl;
            cv::flann::Index kdtree(cv::Mat(cur_pts).reshape(1), indexParams);

            for (auto & pt : n_pts_tmp) {
                std::vector<float> query;
                query.push_back(pt.x); //Insert the 2D point we need to find neighbours to the query
                query.push_back(pt.y); //Insert the 2D point we need to find neighbours to the query

                vector<int> indices;
                vector<float> dists;
                auto ret = kdtree.radiusSearch(query, indices, dists, MIN_DIST, 1);

                if (ret && indices.size() > 0) {
                    // printf("Ret %ld Found pt %d dis %f ", ret, indices[0], dists[0]);
                    // printf("New PT %f %f foundpt %f %f Skipping...\n", pt.x, pt.y, cur_pts[indices[0]].x, cur_pts[indices[0]].y);
                } else {
                    // printf("No nearest neighbors found\n");
                    n_pts.push_back(pt);
                }
            }
        } else {
            n_pts = n_pts_tmp;
            }
        }
    else {
        n_pts.clear();
    }
#ifdef PERF_OUTPUT
    ROS_INFO("Detected %ld npts %fms", n_pts.size(), tic.toc());
#endif

}

#endif


};
