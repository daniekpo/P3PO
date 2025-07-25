joint_map = {
    "base": 0,
    "shoulder": 1,
    "elbow": 2,
    "wrist1": 3,
    "wrist2": 4,
    "wrist3": 5,
}

cameras_serial_to_name = {
    "239222302968": "front_left",
    "234322307622": "front_right",
    "239722073274": "back_left",
    "239722071825": "back_right",
    "216322070351": "robot",
}

cameras_name_to_serial = {
    "front_left": "239222302968",
    "front_right": "234322307622",
    "back_left": "239722073274",
    "back_right": "239722071825",
    "robot": "216322070351",
}

camera_serial_model = {
    "239722071825": "D435",
    "239222302968": "D455",
}

tcp_offset = 0.1734126403712062

# robot_cameras = ["234322307622", "239222302968"]
robot_camera = "239222302968"

# Robot joint positions
home_j = [
    -4.777504865323202,
    -1.7584401569762171,
    1.1920202414142054,
    -0.9933471244624634,
    4.709930896759033,
    -0.060077969227926076,
]

# End effector pose
home_l = [
    -0.011945113081169486,
    -0.46949232570875005,
    0.7260854269326472,
    0.0,
    -3.141592653589793,
    0.0,
]


dept_scale = 0.0010000000474974513  # meters