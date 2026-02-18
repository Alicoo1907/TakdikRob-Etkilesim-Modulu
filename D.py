if __name__ == "__main__":
    from angle_converter import compute_joint_angles
    import numpy as np

    # Sahte hareket (7 joint, 3 koordinat, 5 frame)
    fake_np = np.array([
        # Center
        [[0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1],
         [2, 2, 2, 2, 2]],
        # ShoulderLeft
        [[-0.2, -0.2, -0.2, -0.2, -0.2],
         [1.4, 1.4, 1.4, 1.4, 1.4],
         [2.2, 2.2, 2.2, 2.2, 2.2]],
        # ElbowLeft
        [[-0.4, -0.4, -0.4, -0.4, -0.4],
         [1.2, 1.2, 1.2, 1.2, 1.2],
         [2.4, 2.4, 2.4, 2.4, 2.4]],
        # WristLeft
        [[-0.5, -0.5, -0.5, -0.5, -0.5],
         [1.0, 1.0, 1.0, 1.0, 1.0],
         [2.5, 2.5, 2.5, 2.5, 2.5]],
        # ShoulderRight
        [[0.2, 0.2, 0.2, 0.2, 0.2],
         [1.4, 1.4, 1.4, 1.4, 1.4],
         [2.2, 2.2, 2.2, 2.2, 2.2]],
        # ElbowRight
        [[0.4, 0.4, 0.4, 0.4, 0.4],
         [1.2, 1.2, 1.2, 1.2, 1.2],
         [2.4, 2.4, 2.4, 2.4, 2.4]],
        # WristRight
        [[0.5, 0.5, 0.5, 0.5, 0.5],
         [1.0, 1.0, 1.0, 1.0, 1.0],
         [2.5, 2.5, 2.5, 2.5, 2.5]],
    ])

    joint_list = ['Center', 'ShoulderLeft', 'ElbowLeft', 'WristLeft',
                  'ShoulderRight', 'ElbowRight', 'WristRight']

    frames_as_dicts = []
    T = fake_np.shape[2]
    for t in range(T):
        frame = {}
        for j, joint in enumerate(joint_list):
            x_k, y_k, z_k = float(fake_np[j, 0, t]), float(fake_np[j, 1, t]), float(fake_np[j, 2, t])
            nao_x = z_k        # ileri
            nao_y = -x_k       # sağ/sol ters
            nao_z = y_k        # yukarı
            frame[joint] = {"X": nao_x, "Y": nao_y, "Z": nao_z}
        frames_as_dicts.append(frame)

    # Her frame için açıları hesapla
    for idx, frame in enumerate(frames_as_dicts):
        angles = compute_joint_angles(frame)
        print(f"Frame {idx}: {angles}")
