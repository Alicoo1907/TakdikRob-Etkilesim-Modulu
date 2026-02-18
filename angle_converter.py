import math

# --- Joint angle hesaplama fonksiyonları ---
def angleRShoulderPitch(x2, y2, z2, x1, y1, z1):
    try:
        if y2 < y1:
            angle = math.atan2(abs(y2 - y1), abs(z2 - z1))
            angle = -angle
            if angle < math.radians(-118):
                angle = math.radians(-117)
            return angle
        else:
            angle = math.atan2(z2 - z1, y2 - y1)
            angle = math.pi/2 - angle
            return angle
    except ZeroDivisionError:
        return 0
    except Exception as e:
        print(f"RShoulderPitch hata: {e}")
        return 0

def angleRShoulderRoll(x2, y2, z2, x1, y1, z1):
    try:
        if z2 < z1:
            z2, z1 = z1, z2
        if z2 - z1 < 0.1:
            z2 = z1 + 0.1
        angle = math.atan2(x2 - x1, z2 - z1)
        return angle
    except Exception as e:
        return 0

def angleRElbowYaw(x2, y2, z2, x1, y1, z1, shoulderpitch):
    try:
        angle = math.atan2(z2 - z1, y2 - y1)
        angle = -angle + shoulderpitch
        return -angle
    except Exception as e:
        return 0

def angleRElbowRoll(x3, y3, z3, x2, y2, z2, x1, y1, z1):
    try:
        a = (x3 - x2)**2 + (y3 - y2)**2 + (z3 - z2)**2
        b = (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2
        c = (x1 - x3)**2 + (y1 - y3)**2 + (z1 - z3)**2
        angle = math.acos((a + b - c) / (2 * math.sqrt(a) * math.sqrt(b)))
        return math.pi - angle
    except Exception as e:
        return 0

# --- Sol kollar için aynısı ---
def angleLShoulderPitch(x2, y2, z2, x1, y1, z1):
    return angleRShoulderPitch(x2, y2, z2, x1, y1, z1)

def angleLShoulderRoll(x2, y2, z2, x1, y1, z1):
    return angleRShoulderRoll(x2, y2, z2, x1, y1, z1)

def angleLElbowYaw(x2, y2, z2, x1, y1, z1, shoulderpitch):
    return -angleRElbowYaw(x2, y2, z2, x1, y1, z1, shoulderpitch)

def angleLElbowRoll(x3, y3, z3, x2, y2, z2, x1, y1, z1):
    return angleRElbowRoll(x3, y3, z3, x2, y2, z2, x1, y1, z1)


# --- Frame dictionary'den radyan açılar üretme ---
joint_list = ['Center', 'ShoulderLeft', 'ElbowLeft', 'WristLeft',
              'ShoulderRight', 'ElbowRight', 'WristRight']

def compute_joint_angles(frame):
    shoulder_right = frame["ShoulderRight"]
    elbow_right = frame["ElbowRight"]
    wrist_right = frame["WristRight"]
    
    shoulder_left = frame["ShoulderLeft"]
    elbow_left = frame["ElbowLeft"]
    wrist_left = frame["WristLeft"]

    RShoulderPitch_val = angleRShoulderPitch(
        shoulder_right["X"], shoulder_right["Y"], shoulder_right["Z"],
        elbow_right["X"], elbow_right["Y"], elbow_right["Z"]
    )
    RShoulderRoll_val = angleRShoulderRoll(
        shoulder_right["X"], shoulder_right["Y"], shoulder_right["Z"],
        elbow_right["X"], elbow_right["Y"], elbow_right["Z"]
    )
    RElbowYaw_val = angleRElbowYaw(
        elbow_right["X"], elbow_right["Y"], elbow_right["Z"],
        wrist_right["X"], wrist_right["Y"], wrist_right["Z"],
        RShoulderPitch_val
    )
    RElbowRoll_val = angleRElbowRoll(
        shoulder_right["X"], shoulder_right["Y"], shoulder_right["Z"],
        elbow_right["X"], elbow_right["Y"], elbow_right["Z"],
        wrist_right["X"], wrist_right["Y"], wrist_right["Z"]
    )
    
    LShoulderPitch_val = angleLShoulderPitch(
        shoulder_left["X"], shoulder_left["Y"], shoulder_left["Z"],
        elbow_left["X"], elbow_left["Y"], elbow_left["Z"]
    )
    LShoulderRoll_val = angleLShoulderRoll(
        shoulder_left["X"], shoulder_left["Y"], shoulder_left["Z"],
        elbow_left["X"], elbow_left["Y"], elbow_left["Z"]
    )
    LElbowYaw_val = angleLElbowYaw(
        elbow_left["X"], elbow_left["Y"], elbow_left["Z"],
        wrist_left["X"], wrist_left["Y"], wrist_left["Z"],
        LShoulderPitch_val
    )
    LElbowRoll_val = angleLElbowRoll(
        shoulder_left["X"], shoulder_left["Y"], shoulder_left["Z"],
        elbow_left["X"], elbow_left["Y"], elbow_left["Z"],
        wrist_left["X"], wrist_left["Y"], wrist_left["Z"]
    )

    return [RShoulderPitch_val, RShoulderRoll_val, RElbowRoll_val, RElbowYaw_val,
            LShoulderPitch_val, LShoulderRoll_val, LElbowRoll_val, LElbowYaw_val]