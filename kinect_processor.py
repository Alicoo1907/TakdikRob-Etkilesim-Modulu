import numpy as np

def pre_normalization(data, zaxis=[0, 1], xaxis=[8, 4]):
    """
    Kinect'ten gelen (T, V, C) verisini NTU formatına uygun normalize eder.
    data shape: (C, T, V, M)
    """
    C, T, V, M = data.shape
    s = np.transpose(data, [3, 1, 2, 0])  # M, T, V, C

    # 1. Merkeze çekme (SpineBase - Joint 0)
    # NTU iskeletinde joint 1 merkezdir. Kinect V2'de Joint 0 (SpineBase) merkez alınabilir.
    main_body_center = s[0, :, 0:1, :].copy() # (T, 1, C)
    for i_p in range(M):
        if s[i_p].sum() == 0: continue
        mask = (s[i_p].sum(-1) != 0).reshape(T, V, 1)
        s[i_p] = (s[i_p] - main_body_center) * mask

    # 2. Z Ekseni Hizalama (0 -> 1: SpineBase -> SpineMid)
    joint_bottom = s[0, 0, zaxis[0]]
    joint_top = s[0, 0, zaxis[1]]
    diff = joint_top - joint_bottom
    if np.linalg.norm(diff) > 0:
        axis = np.cross(diff, [0, 0, 1])
        angle = angle_between(diff, [0, 0, 1])
        matrix_z = rotation_matrix(axis, angle)
        for i_p in range(M):
            for i_f in range(T):
                for i_j in range(V):
                    s[i_p, i_f, i_j] = np.dot(matrix_z, s[i_p, i_f, i_j])

    # 3. X Ekseni Hizalama (8 -> 4: RightShoulder -> LeftShoulder)
    joint_rshoulder = s[0, 0, xaxis[0]]
    joint_lshoulder = s[0, 0, xaxis[1]]
    diff_x = joint_rshoulder - joint_lshoulder
    if np.linalg.norm(diff_x) > 0:
        axis_x = np.cross(diff_x, [1, 0, 0])
        angle_x = angle_between(diff_x, [1, 0, 0])
        matrix_x = rotation_matrix(axis_x, angle_x)
        for i_p in range(M):
            for i_f in range(T):
                for i_j in range(V):
                    s[i_p, i_f, i_j] = np.dot(matrix_x, s[i_p, i_f, i_j])

    return np.transpose(s, [3, 1, 2, 0]) # (C, T, V, M)

def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rotation_matrix(axis, theta):
    if np.linalg.norm(axis) == 0: return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
