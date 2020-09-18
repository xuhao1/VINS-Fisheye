#include "feature_tracker.h"

using namespace std;
namespace FeatureTracker {

template<class CvMat>
class BaseFisheyeFeatureTracker : public BaseFeatureTracker{
public:
    virtual FeatureFrame trackImage(double _cur_time, cv::InputArray fisheye_imgs_up, cv::InputArray fisheye_imgs_down) override;

protected:
    virtual FeatureFrame BaseFeatureTracker::setup_feature_frame() override;
    

    void addPointsFisheye();

    vector<cv::Point3f> undistortedPtsTop(vector<cv::Point2f> &pts, FisheyeUndist & fisheye);
    vector<cv::Point3f> undistortedPtsSide(vector<cv::Point2f> &pts, FisheyeUndist & fisheye, bool is_downward);
    vector<cv::Point3f> ptsVelocity3D(vector<int> &ids, vector<cv::Point3f> &pts, 
                                    map<int, cv::Point3f> &cur_id_pts, map<int, cv::Point3f> &prev_id_pts);

        
    void drawTrackFisheye(const cv::Mat & img_up, const cv::Mat & img_down, 
                            cv::Mat imUpTop,
                            cv::Mat imDownTop,
                            cv::Mat imUpSide, 
                            cv::Mat imDownSide);

    vector<FisheyeUndist> fisheys_undists;

    cv::Size top_size;
    cv::Size side_size;

    vector<cv::Point2f> n_pts_up_top, n_pts_down_top, n_pts_up_side;
    vector<cv::Point2f> predict_up_side, predict_pts_left_top, predict_pts_right_top, predict_pts_down_side;
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
    
   
};

class FisheyeFeatureTrackerCuda: public BaseFisheyeFeatureTracker<cv::cuda::GpuMat> {
public:
    virtual FeatureFrame trackImage(double _cur_time, cv::InputArray fisheye_imgs_up, cv::InputArray fisheye_imgs_down) override;

    inline FeatureFrame trackImage(double _cur_time, cv::InputArray fisheye_imgs_up, cv::InputArray fisheye_imgs_down, bool _is_blank_init = false) {
        is_blank_init = true;
        trackImage(_cur_time, fisheye_imgs_up, fisheye_imgs_down);
        is_blank_init = false;
    }

protected:
    bool is_blank_init = false;
    void drawTrackFisheye(const cv::Mat & img_up, const cv::Mat & img_down, 
                            cv::cuda::GpuMat imUpTop,
                            cv::cuda::GpuMat imDownTop,
                            cv::cuda::GpuMat imUpSide, 
                            cv::cuda::GpuMat imDownSide);
    std::vector<cv::cuda::GpuMat> prev_up_top_pyr, prev_down_top_pyr, prev_up_side_pyr;
    void detectPoints(const cv::cuda::GpuMat & img, vector<cv::Point2f> & n_pts, 
        vector<cv::Point2f> & cur_pts, int require_pts);
};



class FisheyeFeatureTrackerOMP: public BaseFisheyeFeatureTracker<cv::Mat> {
    public:
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
};

