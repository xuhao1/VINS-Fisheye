#include "feature_tracker_pinhole.hpp"

namespace FeatureTracker {

template<class CvMat>
void PinholeFeatureTracker<CvMat>::setMask()
{
    mask = cv::Mat(height, width, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (removed_pts.find(it.second.second) == removed_pts.end()) {
            if (mask.at<uchar>(it.second.first) == 255)
            {
                cur_pts.push_back(it.second.first);
                ids.push_back(it.second.second);
                track_cnt.push_back(it.first);
                cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
            }
        }
    }
}


template<class CvMat>
void PinholeFeatureTracker<CvMat>::addPoints()
{
    for (auto &p : n_pts)
    {
        cur_pts.push_back(p);
        ids.push_back(n_id++);
        track_cnt.push_back(1);
    }
}

template<class CvMat>
void PinholeFeatureTracker<CvMat>::readIntrinsicParameter(const vector<string> &calib_file)
{
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
        height = camera->imageHeight();
        width = camera->imageWidth();
        if (camera->modelType() == camodocal::Camera::PINHOLE) {
            FOCAL_LENGTH = ((camodocal::PinholeCamera*)camera.get())->getParameters().fx();
        }
    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
}


template<class CvMat>
vector<cv::Point3f> PinholeFeatureTracker<CvMat>::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point3f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);
        b.normalize();
        un_pts.push_back(cv::Point3f(b.x(), b.y(), b.z()));
    }
    return un_pts;
}

template<class CvMat>
std::vector<cv::Point3f> PinholeFeatureTracker<CvMat>::ptsVelocity(vector<int> &ids, vector<cv::Point3f> &pts, 
                                            map<int, cv::Point3f> &cur_id_pts, map<int, cv::Point3f> &prev_id_pts)
{
    vector<cv::Point3f> pts_velocity;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts[ids[i]] = pts[i];
    }

    // caculate points velocity
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time;
        
        for (unsigned int i = 0; i < pts.size(); i++)
        {
            auto it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end())
            {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                double v_z = (pts[i].z - it->second.z) / dt;
                pts_velocity.push_back(cv::Point3f(v_x, v_y, v_z));
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

template <class CvMat>
void PinholeFeatureTracker<CvMat>::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts, 
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap)
{
    //int rows = imLeft.rows;
    int cols = imLeft.cols;
    if (!imRight.empty() && stereo_cam)
        cv::hconcat(imLeft, imRight, imTrack);
    else
        imTrack = imLeft.clone();

    cv::cvtColor(imTrack, imTrack, CV_GRAY2RGB);
    drawTrackImage(imTrack, curLeftPts, curLeftIds, prevLeftPtsMap);

    // for (size_t j = 0; j < curLeftPts.size(); j++)
    // {
    //     double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
    //         cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    // }
    if (!imRight.empty() && stereo_cam)
    {
        for (size_t i = 0; i < curRightPts.size(); i++)
        {
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cols;
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            //cv::Point2f leftPt = curLeftPtsTrackRight[i];
            //cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }
    
    // map<int, cv::Point2f>::iterator mapIt;
    // for (size_t i = 0; i < curLeftIds.size(); i++)
    // {
    //     int id = curLeftIds[i];
    //     mapIt = prevLeftPtsMap.find(id);
    //     if(mapIt != prevLeftPtsMap.end())
    //     {
    //             cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
    //     }
    // }



    cv::imshow("Track", imTrack);
    cv::waitKey(2);
}


template<class CvMat>
void PinholeFeatureTracker<CvMat>::setPrediction(const map<int, Eigen::Vector3d> &predictPts_cam0, const map<int, Eigen::Vector3d> &predictPt_cam1)
{
    hasPrediction = true;
    predict_pts.clear();
    for (size_t i = 0; i < ids.size(); i++)
    {
        //printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
        int id = ids[i];
        auto itPredict = predictPts_cam0.find(id);
        if (itPredict != predictPts_cam0.end())
        {
            Eigen::Vector2d tmp_uv;
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
            predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
        }
        else
            predict_pts.push_back(prev_pts[i]);
    }
}
}
