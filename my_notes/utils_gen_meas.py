import numpy as np
import gtsam

def make_stereo_calib(fx, fy, s, cx, cy, baseline):
    return gtsam.Cal3_S2Stereo(fx, fy, s, cx, cy, baseline)



def generate_gt(first_gt, last_gt, N, dt=0.1):
    assert first_gt.shape[0] == 9 and last_gt.shape[0] == 9
    assert N >= 2

    p0, p1 = first_gt[0:3], last_gt[0:3]
    e0, e1 = first_gt[6:9], last_gt[6:9]

    # derive total duration automatically
    T = (N - 1) * dt

    D = p1 - p0
    dE = e1 - e0

    # symmetric accel/decel profile (zero vel at start/end)
    a = 4.0 * D / (T ** 2)
    ang_acc = 4.0 * dE / (T ** 2)
    v_mid = a * (T / 2.0)
    ang_vel_mid = ang_acc * (T / 2.0)

    gt_ts, gt_linpos, gt_linvel, gt_linacc, gt_eul, gt_angvel = [], [], [], [], [], []

    for i in range(N):
        t = i * dt

        if i == 0:
            pos_t, vel_t, acc_t = p0, first_gt[3:6], np.zeros(3)
            eul_t, angvel_t = e0, np.zeros(3)
        elif i == N - 1:
            pos_t, vel_t, acc_t = p1, last_gt[3:6], np.zeros(3)
            eul_t, angvel_t = e1, np.zeros(3)
        else:
            if t <= T / 2.0 + 1e-9:
                pos_t = p0 + 0.5 * a * t * t
                vel_t = a * t
                acc_t = a
                eul_t = e0 + 0.5 * ang_acc * t * t
                angvel_eul = ang_acc * t
            else:
                tau = t - T / 2.0
                p_mid = p0 + 0.5 * a * (T / 2.0) ** 2
                pos_t = p_mid + v_mid * tau - 0.5 * a * tau * tau
                vel_t = v_mid - a * tau
                acc_t = -a
                eul_mid = e0 + 0.5 * ang_acc * (T / 2.0) ** 2
                eul_t = eul_mid + ang_vel_mid * tau - 0.5 * ang_acc * tau * tau
                angvel_eul = ang_vel_mid - ang_acc * tau

            angvel_t = angvel_eul  # yaw,pitch,roll order (ZYX)

        gt_ts.append(t)
        gt_linpos.append(pos_t)
        gt_linvel.append(vel_t)
        gt_linacc.append(acc_t)
        gt_eul.append(eul_t)
        gt_angvel.append(angvel_t)

    return (
        np.array(gt_ts),
        np.vstack(gt_linpos),
        np.vstack(gt_linvel),
        np.vstack(gt_linacc),
        np.vstack(gt_eul),
        np.vstack(gt_angvel),
    )


def _pose_from_pos_eul(pos, eul_deg):
    # eul_deg = [yaw, pitch, roll] in deg
    yaw, pitch, roll = np.radians(eul_deg)
    R = gtsam.Rot3.Ypr(yaw, pitch, roll)
    return gtsam.Pose3(R, gtsam.Point3(*pos))


def generate_measurement(
    gt_ts,
    gt_linpos,
    gt_linacc,
    gt_eul,
    gt_angvel,
    cam_dt,
    landmarks,
    calib,
    acc_bias=None,
    gyro_bias=None,
    acc_sigma=0.0,
    gyro_sigma=0.0,
    cam_sigma=0.0,
    rng=None,
):
    """
    Return:
      meas: (N, 8) array
        [ts, has_cam, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]

      cam_uvus: (N, 1 + 3*L) array
        [ts,
         lm0_uL, lm0_v, lm0_uR,
         lm1_uL, lm1_v, lm1_uR,
         ...
        ]
        NaNs when no camera or failed projection.
    """
    if rng is None:
        rng = np.random.default_rng()

    if acc_bias is None:
        acc_bias = np.zeros(3)
    if gyro_bias is None:
        gyro_bias = np.zeros(3)

    N = gt_ts.shape[0]
    L = len(landmarks)

    meas_rows, cam_rows = [], []

    for i in range(N):
        ts = gt_ts[i]
        pose = _pose_from_pos_eul(gt_linpos[i], gt_eul[i])

        # --- IMU values ---
        Rwb = pose.rotation().matrix()
        acc_body = Rwb.T @ gt_linacc[i]
        ang_deg = gt_angvel[i]
        gyro_true_zyx = np.radians(ang_deg)
        gyro_true = np.array([gyro_true_zyx[2], gyro_true_zyx[1], gyro_true_zyx[0]])

        acc_meas = acc_body + acc_bias + rng.normal(0.0, acc_sigma, size=3)
        gyro_meas = gyro_true + gyro_bias + rng.normal(0.0, gyro_sigma, size=3)

        has_cam = 1 if abs((ts / cam_dt) - round(ts / cam_dt)) < 1e-6 else 0

        # --- Stereo projections for all landmarks ---
        cam_vals = [ts] + [np.nan] * (3 * L)

        if has_cam and L > 0:
            cam = gtsam.StereoCamera(pose, calib)
            for j, lm in enumerate(landmarks):
                try:
                    z = cam.project(lm)
                    uL, v, uR = z.uL(), z.v(), z.uR()
                    if cam_sigma > 0.0:
                        uL += rng.normal(0.0, cam_sigma)
                        v  += rng.normal(0.0, cam_sigma)
                        uR += rng.normal(0.0, cam_sigma)
                    base = 1 + 3 * j
                    cam_vals[base:base+3] = [uL, v, uR]
                except RuntimeError:
                    pass

        meas_rows.append([ts, float(has_cam),
                          acc_meas[0], acc_meas[1], acc_meas[2],
                          gyro_meas[0], gyro_meas[1], gyro_meas[2]])
        cam_rows.append(cam_vals)

    return np.array(meas_rows), np.array(cam_rows)


if __name__ == "__main__":

    # ---- user-definable config ----
    cam_params = {
        "fx": 1000.0,
        "fy": 1000.0,
        "s": 0.0,
        "cx": 640.0,
        "cy": 360.0,
        "baseline": 0.2,
    }

    imu_params_user = {
        "acc_bias_x": 0.05,
        "acc_bias_y": -0.02,
        "acc_bias_z": 0.01,
        "gyro_bias_x": 0.001,
        "gyro_bias_y": 0.0005,
        "gyro_bias_z": -0.0002,
        "acc_sigma": 0.02,
        "gyro_sigma": 0.001,
        "cam_sigma": 0.0,
    }

    landmarks = [ # x y z order
        gtsam.Point3(0, 0, 10), # x right, y down, z front
        gtsam.Point3(5, -2, 8),
        gtsam.Point3(-3, 1, 12),        
    ]

    # x,y,z, vx, vy, vz, yaw,pitch,roll
    first_gt = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    last_gt = np.array([3, 0, 0, 0, 0, 0, 5, 0, 0])
    N = 101
    dt = 0.01  # seconds
    cam_dt = 0.1  # camera every 0.1s

    gt_ts, gt_linpos, gt_linvel, gt_linacc, gt_eul, gt_angvel = generate_gt(
        first_gt, last_gt, N, dt=dt
    )


    # ======= Nicely formatted output =======
    print("\n=== Ground Truth Trajectory ===")
    header = (
        f"{'t [s]':>6} | "
        f"{'Pos (x,y,z) [m]':>28} | "
        f"{'Vel (x,y,z) [m/s]':>28} | "
        f"{'Acc (x,y,z) [m/s²]':>28} | "
        f"{'Euler [Yaw,Pitch,Roll] (°)':>32} | "
        f"{'AngVel [Yaw,Pitch,Roll] (°/s)':>36}"
    )
    print(header)
    print("-" * len(header))

    for i in range(N):
        t = gt_ts[i]
        pos = " ".join(f"{v:10.3f}" for v in gt_linpos[i])
        vel = " ".join(f"{v:10.3f}" for v in gt_linvel[i])
        acc = " ".join(f"{v:10.3f}" for v in gt_linacc[i])
        eul = " ".join(f"{v:10.3f}" for v in gt_eul[i])
        ang = " ".join(f"{v:10.3f}" for v in gt_angvel[i])

        print(f"{t:6.3f} | {pos} | {vel} | {acc} | {eul} | {ang}")

    # ---- build calib from user params ----
    calib = make_stereo_calib(
        cam_params["fx"],
        cam_params["fy"],
        cam_params["s"],
        cam_params["cx"],
        cam_params["cy"],
        cam_params["baseline"],
    )

    # ---- IMU bias/noise from user params ----
    acc_bias = np.array([
        imu_params_user["acc_bias_x"],
        imu_params_user["acc_bias_y"],
        imu_params_user["acc_bias_z"],
    ])
    gyro_bias = np.array([
        imu_params_user["gyro_bias_x"],
        imu_params_user["gyro_bias_y"],
        imu_params_user["gyro_bias_z"],
    ])

    meas, cam_uvus = generate_measurement(
        gt_ts,
        gt_linpos,
        gt_linacc,
        gt_eul,
        gt_angvel,
        cam_dt,
        landmarks,
        calib,
        acc_bias=acc_bias,
        gyro_bias=gyro_bias,
        acc_sigma=imu_params_user["acc_sigma"],
        gyro_sigma=imu_params_user["gyro_sigma"],
        cam_sigma=imu_params_user["cam_sigma"],
        rng=None,
    )
    # ======= Nicely formatted measurement output =======
    print("\n=== Measurement Table ===")
    header = (
        f"{'t [s]':>6} | "
        f"{'has_cam':>7} | "
        f"{'Acc (x,y,z) [m/s²]':>28} | "
        f"{'Gyro (x,y,z) [rad/s]':>28} | "        
    )
    print(header)
    print("-" * len(header))

    for row in meas:
        ts, has_cam = row[0], int(row[1])
        acc = " ".join(f"{v:9.3f}" for v in row[2:5])
        gyro = " ".join(f"{v:9.3f}" for v in row[5:8])
        stereo = " ".join(f"{v:9.3f}" for v in row[8:11])

        print(f"{ts:6.3f} | {has_cam:^7d} | {acc} | {gyro} | {stereo}")

    # =========================================================
    # Build Python dictionary list for vio (IMU + camera)
    # =========================================================
    L = len(landmarks)
    vio_data = []
    for i in range(meas.shape[0]):
        t = round(float(meas[i, 0]), 3)
        has_cam = int(meas[i, 1])
        acc_xyz = [round(x, 3) for x in meas[i, 2:5]]      # body frame XYZ
        gyro_xyz = [round(x, 3) for x in meas[i, 5:8]]     # body frame XYZ

        cam_row = cam_uvus[i]
        uvu_list = [] if has_cam else None
        for j in range(L):
            base = 1 + 3 * j
            uL, v, uR = cam_row[base:base+3]
            if np.isnan(uL) or np.isnan(v) or np.isnan(uR):
                continue
            uvu_list.append([int(round(uL)), int(round(v)), int(round(uR))])

        vio_data.append({
            "t": t,
            "has_cam": has_cam,
            "acc_xyz": acc_xyz,
            "gyro_xyz": gyro_xyz,
            "uvu": uvu_list,
        })

    # =========================================================
    # Build GT (ground truth) list-of-dicts
    # =========================================================
    gt_data = []
    for i in range(N):
        gt_data.append({
            "t": round(float(gt_ts[i]), 3),
            "pos_xyz": [round(x, 3) for x in gt_linpos[i]],
            "vel_xyz": [round(x, 3) for x in gt_linvel[i]],
            "acc_xyz": [round(x, 3) for x in gt_linacc[i]],
            "euler_zyx_deg": [round(x, 3) for x in gt_eul[i]],         # yaw,pitch,roll
            "angvel_zyx_deg_s": [round(x, 3) for x in gt_angvel[i]],   # yaw,pitch,roll rates
        })

    # =========================================================
    # Build configuration dict (safe landmark extraction)
    # =========================================================
    lm_list = []
    for lm in landmarks:
        if hasattr(lm, "x"):  # gtsam.Point3
            lm_list.append([lm.x(), lm.y(), lm.z()])
        else:  # numpy array / list / tuple
            arr = np.asarray(lm).ravel()
            lm_list.append([float(arr[0]), float(arr[1]), float(arr[2])])

    config = {
        "camera": cam_params,
        "imu": imu_params_user,
        "timing": {"N": N, "dt": dt, "cam_dt": cam_dt},
        "landmarks_xyz": lm_list,
    }

    # =========================================================
    # Save all to one NPZ
    # =========================================================
    np.savez_compressed(
        "vio_dataset.npz",
        vio=vio_data,
        gt=gt_data,
        config=config,
    )
    print("\nSaved dataset to vio_dataset.npz (keys: vio, gt, config)")


