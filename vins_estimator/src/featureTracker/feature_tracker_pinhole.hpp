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
    TicToc t_r;
    cur_time = _cur_time;
    cv::Mat rightImg;

    cv::cuda::GpuMat cur_gpu_img = cv::cuda::GpuMat(_img);
    cv::cuda::GpuMat right_gpu_img = cv::cuda::GpuMat(_img1);

    height = cur_gpu_img.rows;
    width = cur_gpu_img.cols;

    cur_pts.clear();

    if (prev_pts.size() > 0)
    {
        vector<uchar> status;
        TicToc t_og;
        cv::cuda::GpuMat prev_gpu_pts(prev_pts);
        cv::cuda::GpuMat cur_gpu_pts(cur_pts);
        cv::cuda::GpuMat gpu_status;
        if(hasPrediction)
        {
            cur_gpu_pts = cv::cuda::GpuMat(prev_gpu_pts);
            cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
            cv::Size(21, 21), 1, 30, true);
            d_pyrLK_sparse->calc(prev_gpu_img, cur_gpu_img, prev_gpu_pts, cur_gpu_pts, gpu_status);
            
            vector<cv::Point2f> tmp_cur_pts(cur_gpu_pts.cols);
            cur_gpu_pts.download(tmp_cur_pts);
            cur_pts = tmp_cur_pts;

            vector<uchar> tmp_status(gpu_status.cols);
            gpu_status.download(tmp_status);
            status = tmp_status;

            int succ_num = 0;
            for (size_t i = 0; i < tmp_status.size(); i++)
            {
                if (tmp_status[i])
                    succ_num++;
            }
            if (succ_num < 10)
            {
                cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                cv::Size(21, 21), 3, 30, false);
                d_pyrLK_sparse->calc(prev_gpu_img, cur_gpu_img, prev_gpu_pts, cur_gpu_pts, gpu_status);

                vector<cv::Point2f> tmp1_cur_pts(cur_gpu_pts.cols);
                cur_gpu_pts.download(tmp1_cur_pts);
                cur_pts = tmp1_cur_pts;

                vector<uchar> tmp1_status(gpu_status.cols);
                gpu_status.download(tmp1_status);
                status = tmp1_status;
            }
        }
        else
        {
            cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
            cv::Size(21, 21), 3, 30, false);
            d_pyrLK_sparse->calc(prev_gpu_img, cur_gpu_img, prev_gpu_pts, cur_gpu_pts, gpu_status);

            vector<cv::Point2f> tmp1_cur_pts(cur_gpu_pts.cols);
            cur_gpu_pts.download(tmp1_cur_pts);
            cur_pts = tmp1_cur_pts;

            vector<uchar> tmp1_status(gpu_status.cols);
            gpu_status.download(tmp1_status);
            status = tmp1_status;
        }
        if(FLOW_BACK)
        {
            cv::cuda::GpuMat reverse_gpu_status;
            cv::cuda::GpuMat reverse_gpu_pts = prev_gpu_pts;
            cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
            cv::Size(21, 21), 1, 30, true);
            d_pyrLK_sparse->calc(cur_gpu_img, prev_gpu_img, cur_gpu_pts, reverse_gpu_pts, reverse_gpu_status);

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

        for (int i = 0; i < int(cur_pts.size()); i++)
            if (status[i] && !inBorder(cur_pts[i]))
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        // ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
        
        //printf("track cnt %d\n", (int)ids.size());
    }

    for (auto &n : track_cnt)
        n++;

    //rejectWithF();
    ROS_DEBUG("set mask begins");
    TicToc t_m;
    setMask();
    // ROS_DEBUG("set mask costs %fms", t_m.toc());
    // printf("set mask costs %fms\n", t_m.toc());
    ROS_DEBUG("detect feature begins");
    
    int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
    if (n_max_cnt > MAX_CNT/4)
    {
        if(mask.empty())
            cout << "mask is empty " << endl;
        if (mask.type() != CV_8UC1)
            cout << "mask type wrong " << endl;
        
        cv::Ptr<cv::cuda::CornersDetector> detector = cv::cuda::createGoodFeaturesToTrackDetector(cur_gpu_img.type(), MAX_CNT - cur_pts.size(), 0.01, MIN_DIST);
        cv::cuda::GpuMat d_prevPts;
        cv::cuda::GpuMat gpu_mask(mask);
        detector->detect(cur_gpu_img, d_prevPts, gpu_mask);
        // std::cout << "d_prevPts size: "<< d_prevPts.size()<<std::endl;
        if(!d_prevPts.empty()) {
            n_pts = cv::Mat_<cv::Point2f>(cv::Mat(d_prevPts));
        }
        else {
            n_pts.clear();
        }

        // sum_n += n_pts.size();
        // printf("total point from gpu: %d\n",sum_n);
        // printf("gpu good feature to track cost: %fms\n", t_g.toc());
    }
    else 
        n_pts.clear();
    ROS_DEBUG("add feature begins");
    TicToc t_a;
    addPoints();

    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    if(!_img1.empty() && stereo_cam)
    {
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        if(!cur_pts.empty())
        {
            //printf("stereo image; track feature on right image\n");
            
            vector<cv::Point2f> reverseLeftPts;
            vector<uchar> status, statusRightLeft;
            TicToc t_og1;
            cv::cuda::GpuMat cur_gpu_pts(cur_pts);
            cv::cuda::GpuMat cur_right_gpu_pts;
            cv::cuda::GpuMat gpu_status;
            cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
            cv::Size(21, 21), 3, 30, false);
            d_pyrLK_sparse->calc(cur_gpu_img, right_gpu_img, cur_gpu_pts, cur_right_gpu_pts, gpu_status);

            vector<cv::Point2f> tmp_cur_right_pts(cur_right_gpu_pts.cols);
            cur_right_gpu_pts.download(tmp_cur_right_pts);
            cur_right_pts = tmp_cur_right_pts;

            vector<uchar> tmp_status(gpu_status.cols);
            gpu_status.download(tmp_status);
            status = tmp_status;

            if(FLOW_BACK)
            {   
                cv::cuda::GpuMat reverseLeft_gpu_Pts;
                cv::cuda::GpuMat status_gpu_RightLeft;
                cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                cv::Size(21, 21), 3, 30, false);
                d_pyrLK_sparse->calc(right_gpu_img, cur_gpu_img, cur_right_gpu_pts, reverseLeft_gpu_Pts, status_gpu_RightLeft);

                vector<cv::Point2f> tmp_reverseLeft_Pts(reverseLeft_gpu_Pts.cols);
                reverseLeft_gpu_Pts.download(tmp_reverseLeft_Pts);
                reverseLeftPts = tmp_reverseLeft_Pts;

                vector<uchar> tmp1_status(status_gpu_RightLeft.cols);
                status_gpu_RightLeft.download(tmp1_status);
                statusRightLeft = tmp1_status;
                for(size_t i = 0; i < status.size(); i++)
                {
                    if(status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
                        status[i] = 1;
                    else
                        status[i] = 0;
                }
            }
            // printf("gpu left right optical flow cost %fms\n",t_og1.toc());
        ids_right = ids;
        reduceVector(cur_right_pts, status);
        reduceVector(ids_right, status);
        // only keep left-right pts
        // reduceVector(cur_pts, status);
        // reduceVector(ids, status);
        // reduceVector(track_cnt, status);
        // reduceVector(cur_un_pts, status);
        // reduceVector(pts_velocity, status);
        cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
        right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
            
        }
        prev_un_right_pts_map = cur_un_right_pts_map;
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
    prev_time = cur_time;
    hasPrediction = false;

    prevLeftPtsMap.clear();
    for(size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    FeatureFrame featureFrame;
    for (size_t i = 0; i < ids.size(); i++)
    {
        int feature_id = ids[i];
        double x, y ,z;
        x = cur_un_pts[i].x;
        y = cur_un_pts[i].y;
        z = 1;

#ifdef UNIT_SPHERE_ERROR
        Eigen::Vector3d un_pt(x, y, z);
        un_pt.normalize();
        x = un_pt.x();
        y = un_pt.y();
        z = un_pt.z();
#endif

        double p_u, p_v;
        p_u = cur_pts[i].x;
        p_v = cur_pts[i].y;
        int camera_id = 0;
        double velocity_x, velocity_y;
        velocity_x = pts_velocity[i].x;
        velocity_y = pts_velocity[i].y;

        TrackFeatureNoId xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y, 0;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }

    if (!_img1.empty() && stereo_cam)
    {
        for (size_t i = 0; i < ids_right.size(); i++)
        {
            int feature_id = ids_right[i];
            double x, y ,z;
            x = cur_un_right_pts[i].x;
            y = cur_un_right_pts[i].y;
            z = 1;

#ifdef UNIT_SPHERE_ERROR
            Eigen::Vector3d un_pt(x, y, z);
            un_pt.normalize();
            x = un_pt.x();
            y = un_pt.y();
            z = un_pt.z();
#endif
            double p_u, p_v;
            p_u = cur_right_pts[i].x;
            p_v = cur_right_pts[i].y;
            int camera_id = 1;
            double velocity_x, velocity_y;
            velocity_x = right_pts_velocity[i].x;
            velocity_y = right_pts_velocity[i].y;

            TrackFeatureNoId xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y, 0;
            featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
        }
    }

    if (ENABLE_PERF_OUTPUT) {
        printf("feature track whole time %f PTS %ld\n", t_r.toc(), cur_un_pts.size());
    }
    return featureFrame;
}

}