import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, V, B, L  # pose, vel, bias, landmark
np.set_printoptions(precision=3, suppress=True)


# ------------------------------------------------------------
# Loading helpers
# ------------------------------------------------------------
def load_config(filename="vio_dataset.npz"):
    data = np.load(filename, allow_pickle=True)
    vio_data = data["vio"].tolist()     # list of dicts: t, has_cam, acc_xyz, gyro_xyz, uvu
    gt_data = data["gt"].tolist()       # list of dicts: pos_xyz, euler_zyx_deg, ...
    config = data["config"].item()      # dict: camera, imu, timing, landmarks_xyz
    return vio_data, gt_data, config


def make_preint_params_from_config(imu_config):
    params = gtsam.PreintegrationParams.MakeSharedU(0)

    acc_sigma = imu_config["acc_sigma"]
    gyro_sigma = imu_config["gyro_sigma"]

    params.setAccelerometerCovariance((acc_sigma ** 2) * np.eye(3))    
    params.setGyroscopeCovariance((gyro_sigma ** 2) * np.eye(3))
    params.setIntegrationCovariance(1e-8 * np.eye(3))

    return params


def main():
    # --------------------------------------------------------
    # 1. load
    # --------------------------------------------------------
    vio_data, gt_data, config = load_config()

    cam_config = config["camera"]
    imu_config = config["imu"]
    timing = config["timing"]
    landmarks_xyz = config["landmarks_xyz"]

    fx = cam_config["fx"]
    fy = cam_config["fy"]
    s = cam_config["s"]
    cx = cam_config["cx"]
    cy = cam_config["cy"]
    baseline = cam_config["baseline"]

    acc_bias_vec = np.array([
        imu_config["acc_bias_x"],
        imu_config["acc_bias_y"],
        imu_config["acc_bias_z"],
    ])
    gyro_bias_vec = np.array([
        imu_config["gyro_bias_x"],
        imu_config["gyro_bias_y"],
        imu_config["gyro_bias_z"],
    ])

    acc_sigma = imu_config["acc_sigma"]
    gyro_sigma = imu_config["gyro_sigma"]
    cam_sigma = imu_config["cam_sigma"]

    print(
        f"Configs:\n"
        f"  fx={fx}, fy={fy}, s={s}, cx={cx}, cy={cy}, baseline={baseline}\n"
        f"  acc_bias={acc_bias_vec.tolist()}, gyro_bias={gyro_bias_vec.tolist()}\n"
        f"  acc_sigma={acc_sigma}, gyro_sigma={gyro_sigma}, cam_sigma={cam_sigma}"
    )
    print(f"  landmarks_xyz={landmarks_xyz}")

    # --------------------------------------------------------
    # 2. make GTSAM objects
    # --------------------------------------------------------
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    isam_params = gtsam.ISAM2Params()
    isam = gtsam.ISAM2(isam_params)

    # camera calibration
    stereo_calib = gtsam.Cal3_S2Stereo(fx, fy, s, cx, cy, baseline)

    # imu params and bias
    preint_params = make_preint_params_from_config(imu_config)
    bias0 = gtsam.imuBias.ConstantBias(acc_bias_vec, gyro_bias_vec)

    # we will reuse this preintegrator and reset when we add an ImuFactor
    imu_preintegrator = gtsam.PreintegratedImuMeasurements(preint_params, bias0)

    # some noise models
    pose_prior_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3]*6))  # rot+trans
    vel_prior_model  = gtsam.noiseModel.Isotropic.Sigma(3, 1e-3)
    bias_prior_model = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
    imu_factor_model = None  # comes from preintegrator
    stereo_model = gtsam.noiseModel.Isotropic.Sigma(3, cam_sigma if cam_sigma > 0 else 1.0)

    # --------------------------------------------------------
    # 3. add landmark nodes (with priors to fix them)
    # --------------------------------------------------------
    for j, lm in enumerate(landmarks_xyz):
        key = L(j)
        pt = gtsam.Point3(*lm)
        initial.insert(key, pt)
        graph.add(gtsam.PriorFactorPoint3(key, pt, gtsam.noiseModel.Isotropic.Sigma(3, 1e-6)))

    # --------------------------------------------------------
    # 4. add initial pose/vel/bias (all zeros pose, but we can use gt[0])
    # --------------------------------------------------------
    # use GT[0] to get a nicer initial pose
    gt0 = gt_data[0]
    p0 = np.array(gt0["pos_xyz"])  # world XYZ
    eul0 = np.radians(np.array(gt0["euler_zyx_deg"]))  # [yaw,pitch,roll]
    R0 = gtsam.Rot3.Ypr(eul0[0], eul0[1], eul0[2])
    pose0 = gtsam.Pose3(R0, gtsam.Point3(*p0))
    vel0 = np.array(gt0["vel_xyz"])

    graph.add(gtsam.PriorFactorPose3(X(0), pose0, pose_prior_model))
    graph.add(gtsam.PriorFactorVector(V(0), vel0, vel_prior_model))
    graph.add(gtsam.PriorFactorConstantBias(B(0), bias0, bias_prior_model))

    initial.insert(X(0), pose0)
    initial.insert(V(0), vel0)
    initial.insert(B(0), bias0)

    # we'll advance pose index only when we get a camera frame
    current_state_idx = 0
    last_time = vio_data[0]["t"]

    # --------------------------------------------------------
    # 5. main loop over measurements (incremental optimize)
    # --------------------------------------------------------
    result = None

    current_state_idx = 0
    last_time = vio_data[0]["t"]

    # preintegrator already created above:
    # imu_preintegrator = gtsam.PreintegratedImuMeasurements(preint_params, bias0)

    for i in range(1, len(vio_data)):
        meas_i = vio_data[i]
        t = meas_i["t"]
        dt = t - last_time
        last_time = t

        acc_body =  np.array(meas_i["acc_xyz"])
        gyro_body = np.array(meas_i["gyro_xyz"])
        has_cam = meas_i["has_cam"]
        uvu = meas_i["uvu"]

        # integrate IMU no matter what
        imu_preintegrator.integrateMeasurement(acc_body, gyro_body, dt)

        if has_cam:
            new_idx = current_state_idx + 1

            # 1) IMU factor
            imu_factor = gtsam.ImuFactor(
                X(current_state_idx),
                V(current_state_idx),
                X(new_idx),
                V(new_idx),
                B(current_state_idx),
                imu_preintegrator
            )
            graph.add(imu_factor)

            # 2) bias smoothness
            graph.add(gtsam.BetweenFactorConstantBias(
                B(current_state_idx),
                B(new_idx),
                gtsam.imuBias.ConstantBias(),
                gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
            ))

            # 3) init new state from previous (NOT GT)
            if result is not None:
                prev_pose = result.atPose3(X(current_state_idx))
                prev_vel  = result.atVector(V(current_state_idx))
                prev_bias = result.atConstantBias(B(current_state_idx))
            else:
                prev_pose = initial.atPose3(X(current_state_idx))
                prev_vel  = initial.atVector(V(current_state_idx))
                prev_bias = initial.atConstantBias(B(current_state_idx))

            initial.insert(X(new_idx), prev_pose)
            initial.insert(V(new_idx), prev_vel)
            initial.insert(B(new_idx), prev_bias)

            # 4) add stereo factors
            if uvu is not None:
                for lm_id, obs in enumerate(uvu):
                    if not obs or len(obs) != 3:
                        continue
                    uL, v, uR = obs
                    if uL < 0 or v < 0 or uR < 0:
                        continue
                    z = gtsam.StereoPoint2(uL, uR, v)
                    graph.add(
                        gtsam.GenericStereoFactor3D(
                            z,
                            stereo_model,
                            X(new_idx),
                            L(lm_id),
                            stereo_calib
                        )
                    )

            # 5) optimize now
            isam.update(graph, initial)
            result = isam.calculateEstimate()

            # clear temp containers for next iteration
            graph = gtsam.NonlinearFactorGraph()
            initial = gtsam.Values()
            # 6) grab updated bias for next preintegration
            updated_bias = result.atConstantBias(B(new_idx))
            imu_preintegrator = gtsam.PreintegratedImuMeasurements(preint_params, updated_bias)
            current_state_idx = new_idx

        else:
            # no camera: just keep integrating, no optimize here
            pass


    # --------------------------------------------------------
    # 6. print estimated pose vs GT at camera frames
    # --------------------------------------------------------
    print("\n=== Estimated vs GT (camera frames) ===")
    print(f"{'Frame':>5} | {'t [s]':>6} | {'Est Pos (x,y,z) [m]':>32} | {'GT Pos (x,y,z) [m]':>32} | {'Est YPR [°]':>26} | {'GT YPR [°]':>26}")
    print("-" * 142)

    k = 0
    for i, meas_i in enumerate(vio_data):
        if meas_i["has_cam"]:
            est_pose = result.atPose3(X(k))
            gt_i = gt_data[i]

            est_t = est_pose.translation()
            est_R = est_pose.rotation()
            est_yaw, est_pitch, est_roll = est_R.ypr()
            est_eul_deg = np.degrees([est_yaw, est_pitch, est_roll])

            gt_pos = gt_i["pos_xyz"]
            gt_eul = gt_i["euler_zyx_deg"]

            est_pos_str = " ".join(f"{v:7.3f}" for v in est_t)
            gt_pos_str = " ".join(f"{v:7.3f}" for v in gt_pos)
            est_eul_str = " ".join(f"{v:7.2f}" for v in est_eul_deg)
            gt_eul_str = " ".join(f"{v:7.2f}" for v in gt_eul)

            print(f"{k:5d} | {meas_i['t']:6.3f} | {est_pos_str} | {gt_pos_str} | {est_eul_str} | {gt_eul_str}")
            k += 1



if __name__ == "__main__":
    main()
