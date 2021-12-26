#include "parameters.h"
#include "feature_manager.h"
#include "MSCKF_state.hpp"

class MSCKF {
    MSCKFStateVector nominal_state;
    MSCKFErrorStateVector error_state;
    double t_last = -1;

    Eigen::Matrix<double, IMU_NOISE_DIM, IMU_NOISE_DIM> Q_imu;

public:
    MSCKF();

    void predict(const double t, Vector3d acc, Vector3d gyro);
    void update(const FeaturePerId & feature_by_id);
};