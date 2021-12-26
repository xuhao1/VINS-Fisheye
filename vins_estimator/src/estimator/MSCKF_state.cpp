#include "MSCKF_state.hpp"

MSCKFStateVector::MSCKFStateVector() {
    
}

Matrix3d MSCKFStateVector::get_imu_R() const { 
    return q_imu.toRotationMatrix();
}

MSCKFErrorStateVector::MSCKFErrorStateVector(const MSCKFStateVector & state0) {
    ang.setZero();
    pos.setZero();
    v_imu.setZero();
    bias_gyro.setZero();
    bias_acc.setZero();
    camera_extrisincs.resize(state0.camera_extrisincs.size());
    for (auto & extrinsic : camera_extrisincs) {
        extrinsic.first.setZero();
        extrinsic.second.setZero();
    }

    sld_win_poses.resize(state0.sld_win_poses.size());
    
    for (auto & pose : camera_extrisincs) {
        pose.first.setZero();
        pose.second.setZero();
    }
}

void MSCKFErrorStateVector::reset(MSCKFStateVector & nominal) {
    assert(camera_extrisincs.size() == nominal.camera_extrisincs.size());
    assert(sld_win_poses.size() == nominal.sld_win_poses.size());

    //TODO:
}

void MSCKFErrorStateVector::set_imu_vector(Eigen::Matrix<double, IMU_STATE_DIM, 1> v) {
    ang = v.block<3, 1>(0, 0);
    bias_gyro = v.block<3, 1>(3, 0);
    v_imu = v.block<3, 1>(6, 0);
    bias_acc = v.block<3, 1>(9, 0);
    pos = v.block<3, 1>(12, 0);
}

void MSCKFErrorStateVector::set_imu_P(Eigen::Matrix<double, IMU_STATE_DIM, IMU_STATE_DIM> _P) {
    P.block<IMU_STATE_DIM, IMU_STATE_DIM>(0, 0) = _P;
}


Eigen::Matrix<double, IMU_STATE_DIM, 1> MSCKFErrorStateVector::get_imu_vector() const {
    Eigen::Matrix<double, IMU_STATE_DIM, 1> v;
    v.block<3, 1>(0, 0) = ang;
    v.block<3, 1>(3, 0) = bias_gyro;
    v.block<3, 1>(6, 0) = v_imu;
    v.block<3, 1>(9, 0) = bias_acc;
    v.block<3, 1>(12, 0) = pos;
    return v;
}

Eigen::Matrix<double, IMU_STATE_DIM, IMU_STATE_DIM> MSCKFErrorStateVector::get_imu_P() const {
    return P.block<IMU_STATE_DIM, IMU_STATE_DIM>(0, 0);
}

Eigen::Matrix<double, IMU_STATE_DIM, Eigen::Dynamic> MSCKFErrorStateVector::get_imu_other_P() const {
    return P.block(0, IMU_STATE_DIM, IMU_STATE_DIM, P.cols() - IMU_STATE_DIM);
}

void MSCKFErrorStateVector::set_imu_other_P(Eigen::Matrix<double, IMU_STATE_DIM, Eigen::Dynamic> _P) {
    P.block(0, IMU_STATE_DIM, IMU_STATE_DIM, P.cols() - IMU_STATE_DIM) = _P;
    P.block(IMU_STATE_DIM, 0, P.cols() - IMU_STATE_DIM, IMU_STATE_DIM) = _P.transpose();
}

