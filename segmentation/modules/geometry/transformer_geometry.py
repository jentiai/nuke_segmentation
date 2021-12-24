import numpy as np
import torch


class TransformerGeometry:
    def __init__(self):
        self.h_depth = None
        self.w_depth = None
        self.intrinsic_depth = None
        self.extrinsic_depth = None
        self.pose_depth = None
        self.xyz_normalized_depth_camera_coordinate = None

    def initialize_camera_parameter(self, w_depth, h_depth, intrinsic_depth, extrinsic_depth):
        self.w_depth = w_depth
        self.h_depth = h_depth

        x_depth_image_coordinate = np.tile(np.arange(0, w_depth)[np.newaxis, :, np.newaxis], [h_depth, 1, 1])
        y_depth_image_coordinate = np.tile(np.arange(0, h_depth)[:, np.newaxis, np.newaxis], [1, w_depth, 1])
        one_depth_image_coordinate = np.ones([h_depth, w_depth, 1])  # padding_for_homogeneous coordinate
        xy1_depth_image_coordinate = np.concatenate(
            [x_depth_image_coordinate, y_depth_image_coordinate, one_depth_image_coordinate], axis=2)
        coordinate_depth = xy1_depth_image_coordinate.reshape(-1, 3).T

        self.intrinsic_depth = torch.from_numpy(np.ascontiguousarray(intrinsic_depth)).cuda()
        self.extrinsic_depth = torch.from_numpy(np.ascontiguousarray(extrinsic_depth)).cuda()
        self.pose_depth = torch.from_numpy(np.ascontiguousarray(np.linalg.inv(extrinsic_depth))).cuda()
        self.xyz_normalized_depth_camera_coordinate = torch.from_numpy(np.linalg.inv(intrinsic_depth) @ coordinate_depth).cuda()

    def convert_depth_to_point_cloud(self, image_depth, scale_depth=1000):
        if self.xyz_normalized_depth_camera_coordinate is None:
            print("Please Initialize Normalized Depth Camera Coordinate.\n",
                  "Run initialize_normalized_depth_camera_coordinate() Before Registration.")
            exit(-1)

        xyz_depth_camera_coordinate_tensor = self.xyz_normalized_depth_camera_coordinate.clone().detach()
        image_depth_tensor = torch.from_numpy(image_depth.astype(np.float64)).cuda()

        xyz_depth_camera_coordinate_tensor *= image_depth_tensor.view(-1) / scale_depth
        one_depth_camera_coordinate_tensor = torch.ones((1, xyz_depth_camera_coordinate_tensor.shape[1])).cuda()
        xyz1_depth_camera_coordinate_tensor = torch.cat((xyz_depth_camera_coordinate_tensor, one_depth_camera_coordinate_tensor), dim=0)
        xyz1_world_coordinate_tensor = self.pose_depth @ xyz1_depth_camera_coordinate_tensor
        xyz_world_coordinate_tensor = xyz1_world_coordinate_tensor[:3, :].T
        return xyz_world_coordinate_tensor.cpu().numpy()

    def project_points_on_image(self, points, scale_depth=1000):
        xyz_world_coordinate_tensor = torch.from_numpy((points.T).astype(np.float64)).cuda()
        one_world_coordinate_tensor = torch.ones((1, xyz_world_coordinate_tensor.shape[1])).cuda()
        xyz1_world_coordinate_tensor = torch.cat((xyz_world_coordinate_tensor, one_world_coordinate_tensor), dim=0)
        xyz1_camera_coordinate_tensor = self.extrinsic_depth @ xyz1_world_coordinate_tensor
        xy1_normalized_image_coordinate_tensor = xyz1_camera_coordinate_tensor[:3] / xyz1_camera_coordinate_tensor[2]
        xy_image_coordinate_tensor = (self.intrinsic_depth @ xy1_normalized_image_coordinate_tensor)[:2]
        return xy_image_coordinate_tensor.cpu().numpy().T
