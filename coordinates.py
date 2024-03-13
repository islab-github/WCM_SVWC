# ---- Coded By DH ---- #
# - This code is for converting pixel space to 3d space.
# - To run this code, degree.npy file is needed
# - File 'degree.npy' can be generated through running calibration.py.
# --------------------- #

import numpy as np
import json
import glob
import copy
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt

class ConvertCoordinate3D:
    def __init__(self, degree_file, cam_azi, cam_tilt, cam_height, cam_distance=None):
        # ------- (u,v) to (x,y,z) ------- #
        self.uv_to_deg = np.load(degree_file)

        self.optical_v = int(self.uv_to_deg.shape[0]/2 - 1)
        self.optical_u = int(self.uv_to_deg.shape[1]/2 - 1)

        self.u_to_deg = self.uv_to_deg[self.optical_v,:]
        self.v_to_deg = self.uv_to_deg[:,self.optical_u]

        self.u_to_deg[self.optical_u:] = -self.u_to_deg[self.optical_u:]
        self.v_to_deg[self.optical_v:] = -self.v_to_deg[self.optical_v:]

        # ------- to untilted coordinate ------- #
        self.real_sense_to_our_xyz = np.array(
            [
                [+0, +0, -1],
                [-1, +0, +0],
                [+0, +1, +0],
            ]
        )

        self.cam_tilt_acc = np.matmul(self.real_sense_to_our_xyz, np.array(cam_tilt).reshape(-1,1)) # [m/s^2]

        roll  = np.pi - np.arctan2(self.cam_tilt_acc[1], self.cam_tilt_acc[2])
        pitch = np.arctan2((self.cam_tilt_acc[0]), np.sqrt(np.square(self.cam_tilt_acc[2]) + np.square(self.cam_tilt_acc[2])))

        self.roll = roll * 180/np.pi
        self.pitch = pitch * 180/np.pi

        c = np.cos(-roll[0])
        s = np.sin(-roll[0])

        self.rotate_roll_mat = np.array(
            [
                [+1, +0, +0],
                [+0, +c, -s],
                [+0, +s, +c],
            ]
        )

        c = np.cos(-pitch[0])
        s = np.sin(-pitch[0])

        self.rotate_pitch_mat = np.array(
            [
                [+c, +0, -s],
                [+0, +1, +0],
                [+s, +0, +c],
            ]
        )

        self.un_tilt_mat = np.matmul(self.rotate_roll_mat, self.rotate_pitch_mat)

        # ------- to shared coordinate ------- #
        self.cam_azi = cam_azi
        self.cam_height = cam_height
        self.cam_distance = cam_distance

        # rotate x, y axis
        c = np.cos(np.pi/180 * self.cam_azi)
        s = np.sin(np.pi/180 * self.cam_azi)

        self.rot_mat = np.array(
            [
                [c, -s, 0, 0],
                [s,  c, 0, 0],
                [0,  0, 1, 0],
                [0,  0, 0, 1]
             ]
        )
        t_z = self.cam_height
        self.trn_mat = np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, t_z],
                    [0, 0, 0, 1]
                 ]

        )

        # for Multi-view setting
        if self.cam_distance is not None:

            t_y = - 0.5 * self.cam_distance * np.sign(self.cam_azi)
            self.trn_mat = np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, t_y],
                    [0, 0, 1, t_z],
                    [0, 0, 0, 1]
                 ]
            )

    def to_angle(self, img_uv) -> np.array:
        return self.v_to_deg[img_uv[:, 0]], self.u_to_deg[img_uv[:, 1]]

    def un_tilt(self, xyz) -> np.array:
        return np.matmul(self.un_tilt_mat, xyz.transpose()).transpose()

    def translate(self, xyz):
        if self.cam_distance is not None:
            return np.matmul(self.trn_mat, xyz.transpose()).transpose()

        return xyz

    def rotate(self, xyz):
        return np.matmul(self.rot_mat, xyz.transpose()).transpose()

    def to_3D(self, img_uv, depth):
        elevation, azimuth = self.to_angle(img_uv)
        no_ele_angle_info_id = np.argwhere(np.abs(np.array(elevation)) == 1)
        no_azm_angle_info_id = np.argwhere(np.abs(np.array(azimuth)) == 1)

        x = depth[img_uv[:,0], img_uv[:,1]] # depth axis
        y = x * np.tan(np.pi*azimuth/180)  # horizontal axis (orthogonal to x)
        z = x * np.tan(np.pi*elevation/180)  # vertical axis

        x = np.reshape(x, (-1, 1))
        y = np.reshape(y, (-1, 1))
        z = np.reshape(z, (-1, 1))
        o = np.ones_like(x)

        xyz = np.concatenate((x, y, z), axis=1)
        xyz = self.un_tilt(xyz)
        xyz = np.concatenate((xyz, o), axis=1)

        xyz = self.rotate(xyz)
        xyz = self.translate(xyz)

        if no_ele_angle_info_id.shape[0] != 0:
            xyz[np.unique(no_ele_angle_info_id), :] = 0
        if no_azm_angle_info_id.shape[0] != 0:
            xyz[np.unique(no_azm_angle_info_id), :] = 0

        return xyz[:, :3]

class MultiViewCoordinate:
    def __init__(
            self,
            degree_file0: str,
            cam_height: float, # 2 [m]
            cam0_azi: float, # [deg]
            cam0_tilt: list, # [deg, deg, deg]
            cam1_azi: float, # [deg]
            cam1_tilt: list, # [deg, deg, deg]
            cam_distance: float,
            img_height: int = 480, # [px]
            img_width: int = 640, # [px]
            degree_file1: str = None,
            min_distance: float = 0.05, # [m]
    ):

        if degree_file1 is None:
            degree_file1 = degree_file0

        self.cam0 = ConvertCoordinate3D(degree_file0, cam0_azi, cam0_tilt, cam_height, cam_distance)
        self.cam1 = ConvertCoordinate3D(degree_file1, cam1_azi, cam1_tilt, cam_height, cam_distance)

        self.cam_height = cam_height # [m]
        self.cam_distance = cam_distance # [m]

        self.img_h = img_height
        self.img_w = img_width

        self.min_distance = min_distance

    def to_shared_coordinates(self, img_uv_v0, img_uv_v1, img_depth_v0, img_depth_v1):
        # ------- (x,y,z) to (x',y',z') ------- #
        uv_v0_ = np.zeros_like(img_uv_v0, dtype=int)
        uv_v1_ = np.zeros_like(img_uv_v1, dtype=int)

        uv_v0_[:,0] = np.rint(img_uv_v0[:,0] * self.img_h).astype(int) # yolo -> pixel coord.
        uv_v0_[:,1] = np.rint(img_uv_v0[:,1] * self.img_w).astype(int) # yolo -> pixel coord.

        uv_v1_[:,0] = np.rint(img_uv_v1[:,0] * self.img_h).astype(int) # yolo -> pixel coord.
        uv_v1_[:,1] = np.rint(img_uv_v1[:,1] * self.img_w).astype(int) # yolo -> pixel coord.

        xyz_0 = self.cam0.to_3D(uv_v0_, img_depth_v0)
        xyz_1 = self.cam1.to_3D(uv_v1_, img_depth_v1)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(xyz_0[:, 0], xyz_0[:, 1], xyz_0[:, 2], marker='o')
        # ax.scatter(xyz_1[:, 0], xyz_1[:, 1], xyz_1[:, 2], marker='^')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        #
        # plt.show()

        # xyz = self.to_object_level(xyz_0, xyz_1)
        # xyz = np.concatenate((xyz_0, xyz_1), axis=0)
        # plt.scatter(z, x, y)
        # plt.show()
        return xyz_0, xyz_1


if __name__ == '__main__':
    version = 4
    path_view0 = "./data/for_calibration/view0_"
    path_view1 = "./data/for_calibration/view1_"

    place = json.load(open("./adjusted_measurement_angles_v3.json"))

    degree_file = "./vw_kh/degree.npy"
    h, w = 480, 640

    place_to_save = copy.deepcopy(place)
    min_distance_to_save = {}
    for p in tqdm(place.keys(), desc = f'Calibrate Places: ', leave=False):
        files_view0 = glob.glob(path_view0 + p + '*.txt')
        files_view1 = glob.glob(path_view1 + p + '*.txt')

        files_view0.sort()
        files_view1.sort()

        # mutli-view setting
        cam0_adjust = np.arange(-2.5, 3.0, 0.25)
        cam1_adjust = np.arange(-2.5, 3.0, 0.25)

        adjusted_values = []
        point_distances = []

        for i in tqdm(cam0_adjust, desc = f'Calibrate {p} cam0 & cam1 AZ'):
            for j in cam1_adjust:
                place_adjust = copy.deepcopy(place[p])
                adjusted_values.append({
                    'cam0_azi': place_adjust['cam0_azi'] + i,
                    'cam1_azi': place_adjust['cam1_azi'] + j,
                })

                place_adjust['cam0_azi'] = adjusted_values[-1]['cam0_azi']
                place_adjust['cam1_azi'] = adjusted_values[-1]['cam1_azi']

                View = MultiViewCoordinate(**place_adjust)

                for f_v0 in files_view0:
                    d_v0 = json.load(open(f_v0[:-4] + '.json'))
                    # depth_v0 = np.array([list(d_v0.values())]).reshape(int(list(d_v0.keys())[-1][:3]) + 1, -1)
                    depth_v0 = np.array(d_v0)

                    b_v0 = open(f_v0).readlines()
                    boxes_v0 = np.array([bb.strip("\n").split(' ') for bb in b_v0],dtype=float)

                    f_v1 = path_view1 + p + '_' + f_v0[-5] + '.txt'

                    d_v1 = json.load(open(f_v1[:-4] + '.json'))
                    # depth_v1 = np.array([list(d_v1.values())]).reshape(int(list(d_v1.keys())[-1][:3]) + 1, -1)
                    depth_v1 = np.array(d_v1)

                    b_v1 = open(f_v1).readlines()
                    boxes_v1 = np.array([bb.strip("\n").split(' ') for bb in b_v1], dtype=float)

                    xyz_0, xyz_1 = View.to_shared_coordinates(
                                np.fliplr(boxes_v0[:,1:3]),
                                np.fliplr(boxes_v1[:,1:3]),
                                depth_v0,
                                depth_v1
                            )

                    point_distances.append(np.sqrt(
                        np.sum(np.square(xyz_0 - xyz_1), axis=-1)
                    ).mean())

        min_dis_idx = np.argmin(point_distances)

        min_distance_to_save[p] = point_distances[min_dis_idx]
        print(f"\n # ---------- Adjust {p} finish ---------- #")
        print(f"\n - min distance: {point_distances[min_dis_idx]}")
        print(f"\n - adjusted azi: \n{adjusted_values[min_dis_idx]}")

        place_to_save[p]['cam0_azi'] = adjusted_values[min_dis_idx]['cam0_azi']
        place_to_save[p]['cam1_azi'] = adjusted_values[min_dis_idx]['cam1_azi']

        print(f"# --------------------------------------------- #")

    with open(f"adjusted_measurement_angles_v{version}.json", "w") as json_file:
        json.dump(place_to_save, json_file, indent=4)

    with open(f"adjusted_point_distance_v{version}.json", "w") as json_file:
        json.dump(min_distance_to_save, json_file, indent=4)
