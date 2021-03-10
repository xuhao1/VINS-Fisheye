#include "feature_tracker.h"
#include "fisheye_undist.hpp"

using namespace std;

namespace FeatureTracker {

template<class CvMat>
class BaseFisheyeFeatureTracker : public BaseFeatureTracker{
public:
    virtual FeatureFrame trackImage(double _cur_time, cv::InputArray fisheye_imgs_up, cv::InputArray fisheye_imgs_down) = 0;
    virtual void readIntrinsicParameter(const vector<string> &calib_file) override;
    FisheyeUndist * get_fisheye_undist(unsigned int index = 0) {
        assert(index<fisheys_undists.size() && "Index Must smaller than camera number");
        return &fisheys_undists[index];
    }
    BaseFisheyeFeatureTracker(Estimator * _estimator): 
            BaseFeatureTracker(_estimator),
            t1(Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d(1, 0, 0))),
            t_down(Eigen::AngleAxisd(M_PI, Eigen::Vector3d(1, 0, 0)))
    {
        t2 = t1 * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
        t3 = t2 * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
        t4 = t3 * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
    }

    virtual void setPrediction(const map<int, Eigen::Vector3d> &predictPts_cam0, const map<int, Eigen::Vector3d> &predictPt_cam1 =  map<int, Eigen::Vector3d>()) override;

protected:
    virtual FeatureFrame setup_feature_frame() override;
    
    std::mutex set_predict_lock;

    void addPointsFisheye();

    vector<cv::Point3f> undistortedPtsTop(vector<cv::Point2f> &pts, FisheyeUndist & fisheye);
    vector<cv::Point3f> undistortedPtsSide(vector<cv::Point2f> &pts, FisheyeUndist & fisheye, bool is_downward);
    vector<cv::Point3f> ptsVelocity3D(vector<int> &ids, vector<cv::Point3f> &pts, 
                                    map<int, cv::Point3f> &cur_id_pts, map<int, cv::Point3f> &prev_id_pts);

        
    virtual void drawTrackFisheye(const cv::Mat & img_up, const cv::Mat & img_down, 
                            cv::Mat imUpTop,
                            cv::Mat imDownTop,
                            cv::Mat imUpSide, 
                            cv::Mat imDownSide);

    vector<FisheyeUndist> fisheys_undists;

    cv::Size top_size;
    cv::Size side_size;

    vector<cv::Point2f> n_pts_up_top, n_pts_down_top, n_pts_up_side;
    std::map<int, cv::Point2f> predict_up_side, predict_up_top, predict_down_top, predict_down_side;
    vector<cv::Point2f> prev_up_top_pts, cur_up_top_pts, prev_up_side_pts, cur_up_side_pts, prev_down_top_pts, prev_down_side_pts;
    
    vector<cv::Point3f> prev_up_top_un_pts,  prev_up_side_un_pts, prev_down_top_un_pts, prev_down_side_un_pts;
    vector<cv::Point2f> cur_down_top_pts, cur_down_side_pts;

    vector<cv::Point3f> up_top_vel, up_side_vel, down_top_vel, down_side_vel;
    vector<cv::Point3f> cur_up_top_un_pts, cur_up_side_un_pts, cur_down_top_un_pts, cur_down_side_un_pts;

    vector<int> ids_up_top, ids_up_side, ids_down_top, ids_down_side;
    map<int, cv::Point2f> up_top_prevLeftPtsMap;
    map<int, cv::Point2f> down_top_prevLeftPtsMap;
    map<int, cv::Point2f> up_side_prevLeftPtsMap;
    map<int, cv::Point2f> down_side_prevLeftPtsMap;


    vector<int> track_up_top_cnt;
    vector<int> track_down_top_cnt;
    vector<int> track_up_side_cnt;
    vector<int> track_down_side_cnt;

    map<int, cv::Point3f> cur_up_top_un_pts_map, prev_up_top_un_pts_map;
    map<int, cv::Point3f> cur_down_top_un_pts_map, prev_down_top_un_pts_map;
    map<int, cv::Point3f> cur_up_side_un_pts_map, prev_up_side_un_pts_map;
    map<int, cv::Point3f> cur_down_side_un_pts_map, prev_down_side_un_pts_map;
    
    CvMat prev_up_top_img, prev_up_side_img, prev_down_top_img;

    Eigen::Quaterniond t1;
    Eigen::Quaterniond t2;
    Eigen::Quaterniond t3;
    Eigen::Quaterniond t4;
    Eigen::Quaterniond t_down;

};

class FisheyeFeatureTrackerCuda: public BaseFisheyeFeatureTracker<cv::cuda::GpuMat> {
public:
    virtual FeatureFrame trackImage(double _cur_time, cv::InputArray fisheye_imgs_up, cv::InputArray fisheye_imgs_down) override;

    inline FeatureFrame trackImage_blank_init(double _cur_time, cv::InputArray fisheye_imgs_up, cv::InputArray fisheye_imgs_down) {
        is_blank_init = true;
        auto ff = trackImage(_cur_time, fisheye_imgs_up, fisheye_imgs_down);
        is_blank_init = false;
        return ff;
    }

    FisheyeFeatureTrackerCuda(Estimator * _estimator): BaseFisheyeFeatureTracker<cv::cuda::GpuMat>(_estimator) {

    }

protected:
    bool is_blank_init = false;
    void drawTrackFisheye(const cv::Mat & img_up, const cv::Mat & img_down, 
                            cv::cuda::GpuMat imUpTop,
                            cv::cuda::GpuMat imDownTop,
                            cv::cuda::GpuMat imUpSide, 
                            cv::cuda::GpuMat imDownSide);
    std::vector<cv::cuda::GpuMat> prev_up_top_pyr, prev_down_top_pyr, prev_up_side_pyr;

};



class FisheyeFeatureTrackerOpenMP: public BaseFisheyeFeatureTracker<cv::Mat> {
    public:
        FisheyeFeatureTrackerOpenMP(Estimator * _estimator): BaseFisheyeFeatureTracker<cv::Mat>(_estimator) {
        }

        virtual FeatureFrame trackImage(double _cur_time, cv::InputArray fisheye_imgs_up, cv::InputArray fisheye_imgs_down) override;
    protected:
        std::vector<cv::Mat> * prev_up_top_pyr = nullptr, * prev_down_top_pyr = nullptr, * prev_up_side_pyr = nullptr;

};

class FisheyeFeatureTrackerVWorks: public FisheyeFeatureTrackerCuda {
public:
    virtual FeatureFrame trackImage(double _cur_time, cv::InputArray fisheye_imgs_up, cv::InputArray fisheye_imgs_down) override;
protected:
#ifdef WITH_VWORKS
    cv::cuda::GpuMat up_side_img_fix;
    cv::cuda::GpuMat down_side_img_fix;
    cv::cuda::GpuMat up_top_img_fix;
    cv::cuda::GpuMat down_top_img_fix;

    cv::cuda::GpuMat mask_up_top_fix, mask_down_top_fix, mask_up_side_fix;

    vx_image vx_up_top_image;
    vx_image vx_down_top_image;
    vx_image vx_up_side_image;
    vx_image vx_down_side_image;

    vx_image vx_up_top_mask;
    vx_image vx_down_top_mask;
    vx_image vx_up_side_mask;

    nvx::FeatureTracker* tracker_up_top = nullptr;InputArraytop_img, cv::cuda::GpuMat & down_top_img, cv::cuda::GpuMat & up_side_img, cv::cuda::GpuMat & down_side_img);

    void process_vworks_tracking(nvx::FeatureTracker* _tracker, vector<int> & _ids, vector<cv::Point2f> & prev_pts, vector<cv::Point2f> & cur_pts, 
        vector<int> & _track, vector<cv::Point2f> & n_pts, map<int, int> &_id_by_index, bool debug_output=false);
    bool first_frame = true;

    map<int, int> up_top_id_by_index;
    map<int, int> down_top_id_by_index;
    map<int, int> up_side_id_by_index;
#endif
};

cv::cuda::GpuMat concat_side(const std::vector<cv::cuda::GpuMat> & arr);
cv::Mat concat_side(const std::vector<cv::Mat> & arr);
std::vector<cv::Mat> convertCPUMat(const std::vector<cv::cuda::GpuMat> & arr);


template<class CvMat>
void BaseFisheyeFeatureTracker<CvMat>::setPrediction(const map<int, Eigen::Vector3d> &predictPts_cam0, const map<int, Eigen::Vector3d> &predictPts_cam1) {
    // std::cout << 
    set_predict_lock.lock();
    predict_up_top.clear();
    predict_up_side.clear();
    predict_down_top.clear();
    predict_down_side.clear();

    for (auto it : predictPts_cam0) {
        int _id = it.first;
        auto pt = it.second;
        auto ret = fisheys_undists[0].project_point_to_vcam_id(pt);
        if (ret.first >= 0 ) {
            if (ret.first == 0){
                predict_up_top[_id] = ret.second;
            } else if (ret.first > 1) {
                cv::Point2f pt(ret.second.x + (ret.first - 1)*WIDTH, ret.second.y);
                predict_up_side[_id] = pt;
            } 
        }
    }

    for (auto it : predictPts_cam1) {
        int _id = it.first;
        auto pt = it.second;
        auto ret = fisheys_undists[1].project_point_to_vcam_id(pt);
        if (ret.first >= 0 ) {
            if (ret.first == 0){
                predict_down_top[_id] = ret.second;
            } else if (ret.first > 1) {
                cv::Point2f pt(ret.second.x + (ret.first - 1)*WIDTH, ret.second.y);
                predict_down_side[_id] = pt;
            } 
        }
    }
    set_predict_lock.unlock();
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
        drawTrackImage(imUpTop, cur_up_top_pts, ids_up_top, up_top_prevLeftPtsMap, predict_up_top);
    }

    if(enable_down_top) {
        drawTrackImage(imDownTop, cur_down_top_pts, ids_down_top, down_top_prevLeftPtsMap, predict_down_top);
    }

    if(enable_up_side) {
        drawTrackImage(imUpSide, cur_up_side_pts, ids_up_side, up_side_prevLeftPtsMap, predict_up_side);
    }

    if(enable_down_side) {
        drawTrackImage(imDownSide, cur_down_side_pts, ids_down_side, pts_map(ids_up_side, cur_up_side_pts), predict_down_side);
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

    if (enable_down_side) {
        cv::vconcat(imUpSide, imDownSide, imTrack);
    } else {
        imTrack = imUpSide;
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
FeatureFrame BaseFisheyeFeatureTracker<CvMat>::setup_feature_frame() {
    FeatureFrame ff;
    BaseFeatureTracker::setup_feature_frame(ff, ids_up_top, cur_up_top_pts, cur_up_top_un_pts, up_top_vel, 0);   
    BaseFeatureTracker::setup_feature_frame(ff, ids_up_side, cur_up_side_pts, cur_up_side_un_pts, up_side_vel, 0);
    BaseFeatureTracker::setup_feature_frame(ff, ids_down_top, cur_down_top_pts, cur_down_top_un_pts, down_top_vel, 1);
    BaseFeatureTracker::setup_feature_frame(ff, ids_down_side, cur_down_side_pts, cur_down_side_un_pts, down_side_vel, 1);

    return ff;
}


template<class CvMat>
vector<cv::Point3f> BaseFisheyeFeatureTracker<CvMat>::undistortedPtsSide(vector<cv::Point2f> &pts, FisheyeUndist & fisheye, bool is_downward) {
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
