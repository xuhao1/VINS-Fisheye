#include "feature_tracker.h"
#include "../estimator/estimator.h"
#include "fisheye_undist.hpp"
#include "feature_tracker_fisheye.hpp"
namespace FeatureTracker {

#ifdef WITH_VWORKS

#include "vworks_feature_tracker.hpp"
#ifdef OVX
ovxio::ContextGuard context;
#else 
vx_context context;
#endif

pair<vector<cv::Point2f>, vector<int>> vxarray2cv_pts(vx_array fVx, bool output=false) {
    std::vector<cv::Point2f> fPts;
    vector<int> status;
    vx_size numItems = 0;
    vxQueryArray(fVx, VX_ARRAY_ATTRIBUTE_NUMITEMS, &numItems, sizeof(numItems));
    vx_size stride = sizeof(vx_size);
    void *base = NULL;
    vxAccessArrayRange(fVx, 0, numItems, &stride, &base, VX_READ_ONLY);

    //For tracker status
    // Holds tracking status. Zero indicates a lost point. Initialized to 1 by corner detectors.

    //The primitive uses tracking_status information for input points (VX_TYPE_KEYPOINT, NVX_TYPE_KEYPOINTF) and updates only points with non-zero tracking_status. 
    // The points with tracking_status == 0 gets copied to the output array as is.
    // The VisionWorks corner detectors (FastCorners, HarrisCorners, FAST Track, Harris Track) 
    // initialize the tracking_status field of detected points to 1.
    for (vx_size i = 0; i < numItems; i++)
    {
        nvx_keypointf_t* points = (nvx_keypointf_t*)base;
        vx_float32 error = points[i].error;
        vx_float32 orientation = points[i].orientation;
        vx_float32 scale = points[i].scale;
        vx_float32 strength = points[i].strength;
        vx_int32 trackingStatus = points[i].tracking_status;
        vx_float32 x = points[i].x;
        vx_float32 y = points[i].y;
        if (output) {
            std::cout << "index: " << i
                    // << ":: error:          " << error << std::endl
                    // << ":: orientation:    " << orientation << std::endl
                    // << ":: scale:          " << scale << std::endl
                    // << ":: strength:       " << strength << std::endl
                    << ":: status: " << trackingStatus
                    << ":: x:   " << x
                    << ":: y:   " << y << std::endl;
        }
        fPts.push_back(cv::Point2f(x, y));
        status.push_back((int)trackingStatus);
    }
    return pair<vector<cv::Point2f>, vector<int>>(fPts, status);
}


// tracker_up_top->printPerfs();


// //In cur pts 255 is keep tracking point
// //0 is the new pts
// ROS_INFO("PREV PTS");
// auto cv_prev_pts = vxarray2cv_pts(prev_pts);
// ROS_INFO("CUR PTS");
// auto cv_cur_pts = vxarray2cv_pts(cur_pts);
// ROS_INFO("VWorks track cost %fms cv pts %ld", tic.toc(), cv_cur_pts.first.size());
// cv::cuda::GpuMat up_top_img_Debug;
// cv::Mat uptop_debug;
// up_side_img.copyTo(up_top_img_Debug);
// up_top_img_Debug.download(uptop_debug);

int to_pt_pos_id(const cv::Point2f & pt) {
    return floor(pt.x * 100000) + floor(pt.y*100);
}

void FisheyeFeatureTrackerVWorks::process_vworks_tracking(nvx::FeatureTracker* _tracker, vector<int> & _ids, vector<cv::Point2f> & prev_pts, vector<cv::Point2f> & cur_pts, 
        vector<int> &track, vector<cv::Point2f> & n_pts, map<int, int> & _id_by_index, bool debug_output) {
    auto prev_ids = _ids;
    map<int, int> new_id_by_index;
    map<int, int> _track;
    for (unsigned int i = 0; i < track.size(); i ++) {
        _track[_ids[i]] = track[i];
    }

    _ids.clear();
    prev_ids.clear();
    prev_pts.clear();
    cur_pts.clear();

    auto vx_prev_pts_ = _tracker->getPrevFeatures();
    auto vx_cur_pts_ = _tracker->getCurrFeatures();

    auto cv_cur_pts_flag = vxarray2cv_pts(vx_cur_pts_, false);
    auto cv_prev_pts_flag = vxarray2cv_pts(vx_prev_pts_, false);
    auto cv_cur_pts = cv_cur_pts_flag.first;
    auto cv_cur_flags = cv_cur_pts_flag.second;
    bool first_frame = _id_by_index.empty();


    //For new point; prev is 1 cur is 255
    //For old point; prev and cur is 255
    //1 is create by FAST
    //255 is track by opticalflow
    //Now we always use tracked 2 frame point instead of full
    //This is because the vworks tracker
    if (!first_frame) {
        for (unsigned int i = 0; i < cv_cur_pts.size(); i ++) {
            if (cv_cur_flags[i] == 0) {
                //This is failed point
                continue;
            }
            int prev_pos_id = to_pt_pos_id(cv_prev_pts_flag.first[i]);
            int cur_pos_id = to_pt_pos_id(cv_cur_pts[i]);
            if (_id_by_index.find(prev_pos_id) != _id_by_index.end()) {
                //This is keep tracking point
                int _id = _id_by_index[prev_pos_id];
                new_id_by_index[cur_pos_id] = _id;

                _ids.push_back(_id);
                prev_pts.push_back(cv_prev_pts_flag.first[i]);
                cur_pts.push_back(cv_cur_pts[i]);
                if (debug_output) {
                    ROS_INFO("Index %d ID %d POSID %d PrevID %d PT %f %f ->  %f %f FLAG %d from %d",
                        i,
                        _ids.back(),
                        cur_pos_id,
                        prev_pos_id,
                        prev_pts.back().x, prev_pts.back().y,
                        cur_pts.back().x, cur_pts.back().y,
                        cv_cur_flags[i], cv_prev_pts_flag.second[i]
                    );
                }
                _track[_id] ++;
            }
        }
    }

    for (unsigned int i = 0; i < cv_cur_pts.size(); i ++) {
        if (cv_cur_flags[i] == 0) {
            //This is failed point
            continue;
        }
        int prev_pos_id = to_pt_pos_id(cv_prev_pts_flag.first[i]);
        //This create new points
        if (_id_by_index.find(prev_pos_id) == _id_by_index.end()) {
            //This is new create points
            int cur_pos_id = to_pt_pos_id(cv_cur_pts[i]);
            int prev_pos_id = to_pt_pos_id(cv_prev_pts_flag.first[i]);
            cur_pts.push_back(cv_cur_pts[i]);
            _ids.push_back(n_id++);
            new_id_by_index[cur_pos_id] = _ids.back();
            _track[_ids.back()] = 1;
            if (debug_output) {
                ROS_INFO("New ID %d pos_id %d  prev_id %d PT %f %f CUR %d PREV %d", _ids.back(), 
                    cur_pos_id, prev_pos_id, cv_cur_pts[i].x, cv_cur_pts[i].y, cv_cur_flags[i], cv_prev_pts_flag.second[i]);
            }
        }
    }

    track.clear();
    for (unsigned int i = 0; i < _ids.size(); i ++) {
        int cur_pos_id = to_pt_pos_id(cv_cur_pts[i]);
        int _id = new_id_by_index[cur_pos_id];
        track.push_back(_track[_id]);

        if (debug_output) {
            ROS_INFO("ID %d POSID %d Pos %f %f",
                _id, cur_pos_id, cv_cur_pts[i].x, cv_cur_pts[i].y);
        }
    }
    
    _id_by_index = new_id_by_index;
}

void FeatureTracker::init_vworks_tracker(cv::cuda::GpuMat & up_top_img, cv::cuda::GpuMat & down_top_img, cv::cuda::GpuMat & up_side_img, cv::cuda::GpuMat & down_side_img) {
    context = VX_API_CALL(vxCreateContext());

    if (enable_up_top) {
        vx_up_top_image = nvx_cv::createVXImageFromCVGpuMat(context, up_top_img_fix);
        vx_up_top_mask = nvx_cv::createVXImageFromCVGpuMat(context, mask_up_top_fix); 
    }

    if(enable_down_top) {
        vx_down_top_image = nvx_cv::createVXImageFromCVGpuMat(context, down_top_img_fix);
        vx_down_top_mask = nvx_cv::createVXImageFromCVGpuMat(context, mask_down_top_fix); 
    }

    if(enable_up_side) {
        vx_up_side_image = nvx_cv::createVXImageFromCVGpuMat(context, up_side_img_fix);
        vx_up_side_mask = nvx_cv::createVXImageFromCVGpuMat(context, mask_up_side_fix); 
    }

    if(enable_down_side) {
        vx_down_side_image = nvx_cv::createVXImageFromCVGpuMat(context, down_side_img_fix);
    }

    
    nvx::FeatureTracker::Params params;
    params.use_rgb = RGB_DEPTH_CLOUD;
    params.use_harris_detector = false;
    // params.use_harris_detector = true;
    // params.harris_k = 0.04;
    // params.harris_thresh = 10;
    params.array_capacity = TOP_PTS_CNT;
    params.fast_thresh = 10;

    params.lk_win_size = 21;
    params.detector_cell_size = MIN_DIST;
    if (enable_up_top) {
        tracker_up_top = nvx::FeatureTracker::create(context, params);
        tracker_up_top->init(vx_up_top_image, vx_up_top_mask);
    }
    if(enable_down_top) {
        tracker_down_top = nvx::FeatureTracker::create(context, params);
        tracker_down_top->init(vx_down_top_image, vx_down_top_mask);
    }
    params.detector_cell_size = MIN_DIST;
    params.array_capacity = SIDE_PTS_CNT;

    if(enable_up_side) {
        tracker_up_side = nvx::FeatureTracker::create(context, params);
        tracker_up_side->init(vx_up_side_image, vx_up_side_mask);
    }
}



FeatureFrame  FisheyeFeatureTrackerVWorks::trackImage(double _cur_time, cv::InputArray fisheye_imgs_up, cv::InputArray fisheye_imgs_down) 
{
                cur_time = _cur_time;

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
    TicToc tic;
    //TODO: simpified this to make no copy
    if (enable_up_top) {
        up_top_img.copyTo(up_top_img_fix);
    }

    if(enable_down_top) {
        down_top_img.copyTo(down_top_img_fix);
    }

    if(enable_up_side) {
        up_side_img.copyTo(up_side_img_fix);
    }

    if(enable_down_side) {
        down_side_img.copyTo(down_side_img_fix);
    }

    ROS_INFO("Copy Image cost %fms", tic.toc());
    if(first_frame) {
        setMaskFisheye();
        if (enable_up_top) {
            mask_up_top_fix.upload(mask_up_top);
        }

        if(enable_down_top) {
            mask_down_top_fix.upload(mask_down_top);
        }

        if(enable_up_side) {
            mask_up_side_fix.upload(mask_up_side);
        }
        ROS_INFO("setFisheyeMask Image cost %fms", tic.toc());

        init_vworks_tracker(up_top_img_fix, down_top_img_fix, up_side_img_fix, down_side_img_fix);
        first_frame = false;
    } else {
        if (enable_up_top) {
            tracker_up_top->track(vx_up_top_image, vx_up_top_mask);
        }

        if(enable_down_top) {
            tracker_down_top->track(vx_down_top_image, vx_down_top_mask);
        }

        if(enable_up_side) {
            tracker_up_side->track(vx_up_side_image, vx_up_side_mask);
        }
    }
    
    ROS_INFO("Track only cost %fms", tic.toc());

    if (enable_up_top) {
        process_vworks_tracking(tracker_up_top,  ids_up_top, prev_up_top_pts, cur_up_top_pts, 
            track_up_top_cnt, n_pts_up_top, up_top_id_by_index);
    }

    if(enable_down_top) {
        process_vworks_tracking(tracker_down_top,  ids_down_top, prev_down_top_pts, cur_down_top_pts, 
            track_down_top_cnt, n_pts_down_top, down_top_id_by_index);
    }

    if(enable_up_side) {
        process_vworks_tracking(tracker_up_side,  ids_up_side, prev_up_side_pts, cur_up_side_pts, 
            track_up_side_cnt, n_pts_up_side, up_side_id_by_index);
    }
    
    ROS_INFO("Visionworks cost %fms", tic.toc());

    if (enable_down_side) {
        ids_down_side = ids_up_side;
        vector<cv::Point2f> down_side_init_pts = cur_up_side_pts;
        cur_down_side_pts = opticalflow_track(down_side_img, up_side_img, down_side_init_pts, ids_down_side, track_down_side_cnt, FLOW_BACK);
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

    printf("FT Whole %fms; MainProcess %fms concat %fms PTS %ld T\n", t_r.toc(), tcost_all, concat_cost, ff.size());
    return ff;
}
#else 

FeatureFrame FisheyeFeatureTrackerVWorks::trackImage(double _cur_time, cv::InputArray fisheye_imgs_up, cv::InputArray fisheye_imgs_down) {
    ROS_ERROR("VisionWorks must enable in CMakeLists.txt first!!!");
    exit(-1);
}

#endif
};