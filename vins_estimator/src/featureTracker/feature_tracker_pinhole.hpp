#include "feature_tracker.h"

namespace FeatureTracker {

template<class CvMat>
class PinholeFeatureTracker: public BaseFeatureTracker {
public:
    virtual FeatureFrame trackImage(double _cur_time, cv::InputArray _img, 
        cv::InputArray _img1 = cv::noArray()) override;

    virtual void readIntrinsicParameter(const vector<string> &calib_file) override;

protected:
    cv::Mat mask;
    cv::Mat imTrack;
    int width, height;
    bool inBorder(const cv::Point2f &pt) const;
    void setMask();
    void addPoints();
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts);
    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);

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
    vector<cv::Point2f> predict_pts;
    vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;
    vector<cv::Point2f> pts_velocity, right_pts_velocity;
    vector<int> ids, ids_right;
    vector<int> pts_img_id, pts_img_id_right;
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
    map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;
    map<int, cv::Point2f> prevLeftPtsMap;
};
}