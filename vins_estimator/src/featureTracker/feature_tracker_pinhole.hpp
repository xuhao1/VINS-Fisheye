#include "feature_tracker.h"

namespace FeatureTracker {

template<class CvMat>
class PinholeFeatureTracker: public BaseFeatureTracker {
public:
    virtual FeatureFrame trackImage(double _cur_time, cv::InputArray _img, 
        cv::InputArray _img1 = cv::noArray()) override {};

    virtual void readIntrinsicParameter(const vector<string> &calib_file) override;
    PinholeFeatureTracker(Estimator * _estimator): 
            BaseFeatureTracker(_estimator) {}
protected:
    cv::Mat mask;
    cv::Mat imTrack;
    int width, height;
    bool inBorder(const cv::Point2f &pt) const;
    void setMask();
    void addPoints();
    vector<cv::Point3f> ptsVelocity(vector<int> &ids, vector<cv::Point3f> &pts, 
                                    map<int, cv::Point3f> &cur_id_pts, map<int, cv::Point3f> &prev_id_pts);
    vector<cv::Point3f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);

    void showTwoImage(const cv::Mat &img1, const cv::Mat &img2, 
                      vector<cv::Point2f> pts1, vector<cv::Point2f> pts2);

    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                    vector<int> &curLeftIds,
                    vector<cv::Point2f> &curLeftPts, 
                    vector<cv::Point2f> &curRightPts,
                    map<int, cv::Point2f> &prevLeftPtsMap);

    virtual void setPrediction(const map<int, Eigen::Vector3d> &predictPts_cam0, const map<int, Eigen::Vector3d> &predictPt_cam1 =  map<int, Eigen::Vector3d>()) override;

    vector<cv::Point2f> n_pts;
    CvMat prev_img, cur_img;
    vector<CvMat> prev_pyr;
    vector<cv::Point2f> predict_pts;
    vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;
    vector<cv::Point3f> prev_un_pts, cur_un_pts, cur_un_right_pts;
    vector<cv::Point3f> pts_velocity, right_pts_velocity;
    vector<int> ids, ids_right;
    vector<int> pts_img_id, pts_img_id_right;
    vector<int> track_cnt, track_right_cnt;
    map<int, cv::Point3f> cur_un_pts_map, prev_un_pts_map;
    map<int, cv::Point3f> cur_un_right_pts_map, prev_un_right_pts_map;
    map<int, cv::Point2f> prevLeftPtsMap;
    virtual FeatureFrame setup_feature_frame() override {};

};


class PinholeFeatureTrackerCuda: public PinholeFeatureTracker<cv::cuda::GpuMat> {
    protected:
    cv::cuda::GpuMat prev_gpu_img;
    cv::Mat cur_img, rightImg;

public:
    PinholeFeatureTrackerCuda(Estimator * _estimator): 
            PinholeFeatureTracker<cv::cuda::GpuMat>(_estimator) {}
    virtual FeatureFrame trackImage(double _cur_time, cv::InputArray _img, 
        cv::InputArray _img1 = cv::noArray()) override;
};

// class PinholeFeatureTrackerCPU: public PinholeFeatureTracker<cv::Mat> {
// public:
//     virtual FeatureFrame trackImage(double _cur_time, cv::InputArray _img, 
//         cv::InputArray _img1 = cv::noArray()) override;
// };

template<class CvMat>
bool PinholeFeatureTracker<CvMat>::inBorder(const cv::Point2f &pt) const
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < width - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < height - BORDER_SIZE;
}

FeatureFrame PinholeFeatureTrackerCuda::trackImage(double _cur_time, cv::InputArray _img, 
        cv::InputArray _img1)
{
    static double detected_time_sum = 0;
    static double ft_time_sum = 0;
    static int count = 0;
    count += 1;

    TicToc t_r;
    cur_time = _cur_time;
    cv::Mat rightImg;
    cv::cuda::GpuMat cur_gpu_img = cv::cuda::GpuMat(_img);
    cv::cuda::GpuMat right_gpu_img = cv::cuda::GpuMat(_img1);

    height = cur_gpu_img.rows;
    width = cur_gpu_img.cols;

    cur_pts.clear();
    TicToc t_ft;
    cur_pts = opticalflow_track(cur_gpu_img, prev_pyr, prev_pts, ids, track_cnt, removed_pts, false);
    ft_time_sum += t_ft.toc();

    TicToc t_d;
    detectPoints(cur_gpu_img, n_pts, cur_pts, MAX_CNT);
    detected_time_sum = detected_time_sum + t_d.toc();

    addPoints();

    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    if(!_img1.empty() && stereo_cam)
    {
        t_ft.tic();
        ids_right = ids;
        std::vector<cv::Point2f> right_side_init_pts = cur_pts;
        cur_right_pts = opticalflow_track(right_gpu_img, prev_pyr, right_side_init_pts, ids_right, track_right_cnt, removed_pts, true);
        cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
        right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
        ft_time_sum += t_ft.toc();
    }

    if(SHOW_TRACK)
    {
        cur_gpu_img.download(cur_img);
        right_gpu_img.download(rightImg);
        drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);
    }

    prev_gpu_img = cur_gpu_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_un_right_pts_map = cur_un_right_pts_map;

    prev_time = cur_time;
    hasPrediction = false;

    prevLeftPtsMap.clear();
    for(size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    FeatureFrame featureFrame;
    BaseFeatureTracker::setup_feature_frame(featureFrame, ids, cur_pts, cur_un_pts, pts_velocity, 0);   
    BaseFeatureTracker::setup_feature_frame(featureFrame, ids_right, cur_right_pts, cur_un_right_pts, right_pts_velocity, 1);   

    printf("Img: %d: trackImage: %3.1fms; PT NUM: %ld, STEREO: %ld; Avg: GFTT %3.1fms LKFlow %3.1fms\n", 
        count,
        t_r.toc(), 
        cur_pts.size(),
        cur_right_pts.size(),
        detected_time_sum/count, 
        ft_time_sum/count);
    return featureFrame;
}

}