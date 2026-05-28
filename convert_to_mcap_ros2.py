#!/usr/bin/env python3
"""
Convert nuScenes mini dataset to MCAP with ROS2 CDR encoding.

Produces .mcap files with standard ROS2 message types (sensor_msgs, geometry_msgs,
nav_msgs, tf2_msgs, visualization_msgs, etc.) serialized in CDR format,
suitable for use with ROS2 tooling and Foxglove.
"""

import argparse
import json
import math
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from pypcd import pypcd
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes

from mcap.writer import Writer as McapWriter
from rosbags.typesys import get_typestore, Stores

# ---------------------------------------------------------------------------
# ROS2 Typestore (loaded once, provides all standard ROS2 message types + CDR)
# ---------------------------------------------------------------------------
_TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)


def ros2_type(name: str):
    """Look up a ROS2 message type from the typestore."""
    return _TYPESTORE.types[name]


def ros2_msgdef(name: str) -> str:
    """Generate .msg definition string for a ROS2 type."""
    return _TYPESTORE.generate_msgdef(name, ros_version=2)[0]


def ros2_serialize(msg, typename: str) -> bytes:
    """Serialize a ROS2 message instance to CDR bytes."""
    return bytes(_TYPESTORE.serialize_cdr(msg, typename))


# ---------------------------------------------------------------------------
# MCAP helper – handles schema/channel deduplication like rosbags Writer
# ---------------------------------------------------------------------------
class Ros2McapWriter:
    """Writes ROS2 CDR messages to an MCAP file.

    Automatically deduplicates schemas by type name.
    """

    def __init__(self, path: str):
        self._writer = McapWriter(path)
        self._schemas: dict[str, int] = {}   # type_name → schema_id
        self._channels: dict[str, int] = {}  # topic → channel_id

    def start(self):
        self._writer.start()

    def register_schema(self, ros_type: str) -> int:
        if ros_type not in self._schemas:
            sid = self._writer.register_schema(
                name=ros_type,
                encoding='ros2msg',
                data=ros2_msgdef(ros_type).encode(),
            )
            self._schemas[ros_type] = sid
        return self._schemas[ros_type]

    def register_channel(self, ros_type: str, topic: str) -> int:
        if topic not in self._channels:
            schema_id = self.register_schema(ros_type)
            cid = self._writer.register_channel(
                schema_id=schema_id,
                topic=topic,
                message_encoding='cdr',
            )
            self._channels[topic] = cid
        return self._channels[topic]

    def write(self, topic: str, msg, ros_type: str, timestamp_ns: int):
        """Serialize a ROS2 message as CDR and write to MCAP."""
        cid = self.register_channel(ros_type, topic)
        data = ros2_serialize(msg, ros_type)
        self._writer.add_message(
            channel_id=cid,
            log_time=timestamp_ns,
            data=data,
            publish_time=timestamp_ns,
        )

    def write_metadata(self, name: str, data: dict[str, str]):
        """Write MCAP metadata record."""
        self._writer.add_metadata(name=name, data=data)

    def finish(self):
        self._writer.finish()


# ---------------------------------------------------------------------------
# Helper – header / timestamp construction
# ---------------------------------------------------------------------------
def make_header(ts_us: int, frame_id: str):
    """Build a std_msgs/Header from a nuScenes microsecond timestamp."""
    sec = int(ts_us // 1_000_000)
    nsec = int((ts_us % 1_000_000) * 1000)
    return ros2_type('std_msgs/msg/Header')(
        stamp=ros2_type('builtin_interfaces/msg/Time')(sec=sec, nanosec=nsec),
        frame_id=frame_id,
    )


def make_header_ns(timestamp_ns: int, frame_id: str):
    """Build a header from nanosecond timestamp."""
    sec = int(timestamp_ns // 1_000_000_000)
    nsec = int(timestamp_ns % 1_000_000_000)
    return ros2_type('std_msgs/msg/Header')(
        stamp=ros2_type('builtin_interfaces/msg/Time')(sec=sec, nanosec=nsec),
        frame_id=frame_id,
    )


# ---------------------------------------------------------------------------
# Turbomap colormap (for annotations – same as original)
# ---------------------------------------------------------------------------
with open(Path(__file__).parent / "turbomap.json") as _f:
    _TURBOMAP_DATA = np.array(json.load(_f))


def turbomap(x):
    np.clip(x, 0, 1, out=x)
    x *= 255
    a = x.astype(np.uint8)
    x -= a
    b = np.minimum(254, a)
    b += 1
    color_a = _TURBOMAP_DATA[a]
    color_b = _TURBOMAP_DATA[b]
    color_b -= color_a
    color_b *= x[:, np.newaxis]
    return np.add(color_a, color_b, out=color_b)


# ---------------------------------------------------------------------------
# Map helpers (unchanged from original)
# ---------------------------------------------------------------------------
def load_bitmap(dataroot: str, map_name: str, layer_name: str) -> np.ndarray:
    if layer_name == "basemap":
        map_path = os.path.join(dataroot, "maps", "basemap", map_name + ".png")
    elif layer_name == "semantic_prior":
        map_hashes = {
            "singapore-onenorth": "53992ee3023e5494b90c316c183be829",
            "singapore-hollandvillage": "37819e65e09e5547b8a3ceaefba56bb2",
            "singapore-queenstown": "93406b464a165eaba6d9de76ca09f5da",
            "boston-seaport": "36092f0b03a857c6a3403e25b4b7aab3",
        }
        map_hash = map_hashes[map_name]
        map_path = os.path.join(dataroot, "maps", map_hash + ".png")
    else:
        raise Exception("Error: Invalid bitmap layer: %s" % layer_name)
    if os.path.exists(map_path):
        image = np.array(Image.open(map_path).convert("L"))
    else:
        raise Exception("Error: Cannot find %s!" % map_path)
    if layer_name == "semantic_prior":
        image = image.max() - image
    return image


EARTH_RADIUS_METERS = 6.378137e6
REFERENCE_COORDINATES = {
    "boston-seaport": [42.336849169438615, -71.05785369873047],
    "singapore-onenorth": [1.2882100868743724, 103.78475189208984],
    "singapore-hollandvillage": [1.2993652317780957, 103.78217697143555],
    "singapore-queenstown": [1.2782562240223188, 103.76741409301758],
}


def get_coordinate(ref_lat: float, ref_lon: float, bearing: float, dist: float) -> Tuple[float, float]:
    lat, lon = math.radians(ref_lat), math.radians(ref_lon)
    angular_distance = dist / EARTH_RADIUS_METERS
    target_lat = math.asin(
        math.sin(lat) * math.cos(angular_distance)
        + math.cos(lat) * math.sin(angular_distance) * math.cos(bearing)
    )
    target_lon = lon + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat),
        math.cos(angular_distance) - math.sin(lat) * math.sin(target_lat),
    )
    return math.degrees(target_lat), math.degrees(target_lon)


def derive_latlon(location: str, pose: Dict[str, float]):
    assert location in REFERENCE_COORDINATES, f"Reference not found: {location}"
    ref_lat, ref_lon = REFERENCE_COORDINATES[location]
    x, y = pose["translation"][:2]
    bearing = math.atan2(x, y)
    yaw = math.sqrt(x**2 + y**2)
    return get_coordinate(ref_lat, ref_lon, bearing, yaw)


def rectContains(rect, point):
    a, b, c, d = rect
    x, y = point[:2]
    return a <= x < a + c and b <= y < b + d


# ---------------------------------------------------------------------------
# Scene bounding box
# ---------------------------------------------------------------------------
def scene_bounding_box(nusc, scene, nusc_map, padding=75.0):
    box = [np.inf, np.inf, -np.inf, -np.inf]
    cur_sample = nusc.get("sample", scene["first_sample_token"])
    while cur_sample is not None:
        sample_lidar = nusc.get("sample_data", cur_sample["data"]["LIDAR_TOP"])
        ego_pose = nusc.get("ego_pose", sample_lidar["ego_pose_token"])
        x, y = ego_pose["translation"][:2]
        box[0] = min(box[0], x)
        box[1] = min(box[1], y)
        box[2] = max(box[2], x)
        box[3] = max(box[3], y)
        cur_sample = nusc.get("sample", cur_sample["next"]) if cur_sample.get("next") != "" else None
    box[0] = max(box[0] - padding, 0.0)
    box[1] = max(box[1] - padding, 0.0)
    box[2] = min(box[2] + padding, nusc_map.canvas_edge[0]) - box[0]
    box[3] = min(box[3] + padding, nusc_map.canvas_edge[1]) - box[1]
    return box


# ---------------------------------------------------------------------------
# ROS2 message factories
# ---------------------------------------------------------------------------

def make_compressed_image(data_path, sample_data, frame_id):
    """Build sensor_msgs/CompressedImage."""
    jpg_path = data_path / sample_data["filename"]
    with open(jpg_path, "rb") as f:
        jpg_data = f.read()
    return ros2_type('sensor_msgs/msg/CompressedImage')(
        header=make_header(sample_data["timestamp"], frame_id),
        format='jpeg',
        data=np.frombuffer(jpg_data, dtype=np.uint8),
    )


def make_camera_info(nusc, sample_data, frame_id):
    """Build sensor_msgs/CameraInfo."""
    calib = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    K = np.array(calib["camera_intrinsic"], dtype=np.float64).flatten()  # 9
    # Standard rectification & projection (identity R, K-based P)
    R = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64)
    P = np.array([K[0], K[1], K[2], 0,
                  K[3], K[4], K[5], 0,
                  0,    0,    1,    0], dtype=np.float64)
    return ros2_type('sensor_msgs/msg/CameraInfo')(
        header=make_header(sample_data["timestamp"], frame_id),
        height=sample_data["height"],
        width=sample_data["width"],
        distortion_model='plumb_bob',
        d=np.zeros(5, dtype=np.float64),
        k=K,
        r=R,
        p=P,
        binning_x=0,
        binning_y=0,
        roi=ros2_type('sensor_msgs/msg/RegionOfInterest')(
            x_offset=0, y_offset=0, height=0, width=0, do_rectify=False,
        ),
    )


def make_lidar_pointcloud2(data_path, sample_data, frame_id):
    """Build sensor_msgs/PointCloud2 from nuScenes LiDAR .bin file.

    nuScenes LiDAR format: x, y, z, intensity, ring — each float32.
    Returns (header, points_xyz, pointcloud2_msg).
    """
    bin_path = data_path / sample_data["filename"]
    raw = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)
    xyz = raw[:, :3].copy()  # (N, 3)
    intensity = raw[:, 3].copy()
    ring = raw[:, 4].copy()

    N = raw.shape[0]
    fields = [
        ros2_type('sensor_msgs/msg/PointField')(
            name=n, offset=i * 4, datatype=7, count=1  # FLOAT32
        )
        for i, n in enumerate(['x', 'y', 'z', 'intensity', 'ring'])
    ]

    return ros2_type('sensor_msgs/msg/PointCloud2')(
        header=make_header(sample_data["timestamp"], frame_id),
        height=1,
        width=N,
        fields=fields,
        is_bigendian=False,
        point_step=20,
        row_step=N * 20,
        data=np.frombuffer(raw.tobytes(), dtype=np.uint8),
        is_dense=True,
    )


def make_radar_pointcloud2(data_path, sample_data, frame_id):
    """Build sensor_msgs/PointCloud2 from nuScenes radar .pcd file."""
    pc_filename = data_path / sample_data["filename"]
    pc = pypcd.PointCloud.from_path(str(pc_filename))
    N = len(pc.pc_data)

    # Map pypcd types to PointField datatypes
    TYPE_MAP = {('I', 1): 2, ('U', 1): 2, ('I', 2): 3, ('U', 2): 4,
                ('I', 4): 5, ('U', 4): 6, ('F', 4): 7, ('F', 8): 8}
    fields = []
    offset = 0
    for name, size, _, ty in zip(pc.fields, pc.size, pc.count, pc.type):
        fields.append(ros2_type('sensor_msgs/msg/PointField')(
            name=name, offset=offset, datatype=TYPE_MAP[(ty, size)], count=1,
        ))
        offset += size

    return ros2_type('sensor_msgs/msg/PointCloud2')(
        header=make_header(sample_data["timestamp"], frame_id),
        height=1,
        width=N,
        fields=fields,
        is_bigendian=False,
        point_step=offset,
        row_step=N * offset,
        data=np.frombuffer(pc.pc_data.tobytes(), dtype=np.uint8),
        is_dense=True,
    )


def make_nav_sat_fix(lat: float, lon: float, alt: float, timestamp_us: int):
    """Build sensor_msgs/NavSatFix."""
    return ros2_type('sensor_msgs/msg/NavSatFix')(
        header=make_header(timestamp_us, 'gps'),
        status=ros2_type('sensor_msgs/msg/NavSatStatus')(status=0, service=1),
        latitude=lat,
        longitude=lon,
        altitude=alt,
        position_covariance=np.array([-1.0] * 9, dtype=np.float64),
        position_covariance_type=0,
    )


def make_pose_stamped(stamp_ns: int, frame_id: str, translation, rotation):
    """Build geometry_msgs/PoseStamped from nuScenes translation(3) & rotation(4)."""
    return ros2_type('geometry_msgs/msg/PoseStamped')(
        header=make_header_ns(stamp_ns, frame_id),
        pose=ros2_type('geometry_msgs/msg/Pose')(
            position=ros2_type('geometry_msgs/msg/Point')(
                x=float(translation[0]),
                y=float(translation[1]),
                z=float(translation[2]),
            ),
            orientation=ros2_type('geometry_msgs/msg/Quaternion')(
                x=float(rotation[1]),
                y=float(rotation[2]),
                z=float(rotation[3]),
                w=float(rotation[0]),
            ),
        ),
    )


def make_imu(sample_data_utime: int, linear_accel, q, rotation_rate):
    """Build sensor_msgs/Imu from CAN bus data."""
    sec, nsec = divmod(sample_data_utime, 1_000_000)
    nsec *= 1000
    return ros2_type('sensor_msgs/msg/Imu')(
        header=ros2_type('std_msgs/msg/Header')(
            stamp=ros2_type('builtin_interfaces/msg/Time')(sec=int(sec), nanosec=int(nsec)),
            frame_id='imu',
        ),
        orientation=ros2_type('geometry_msgs/msg/Quaternion')(
            x=float(q[1]), y=float(q[2]), z=float(q[3]), w=float(q[0]),
        ),
        orientation_covariance=np.array([-1.0] * 9, dtype=np.float64),
        angular_velocity=ros2_type('geometry_msgs/msg/Vector3')(
            x=float(rotation_rate[0]), y=float(rotation_rate[1]), z=float(rotation_rate[2]),
        ),
        angular_velocity_covariance=np.array([-1.0] * 9, dtype=np.float64),
        linear_acceleration=ros2_type('geometry_msgs/msg/Vector3')(
            x=float(linear_accel[0]), y=float(linear_accel[1]), z=float(linear_accel[2]),
        ),
        linear_acceleration_covariance=np.array([-1.0] * 9, dtype=np.float64),
    )


def make_odometry(sample_data_utime: int, pos, orientation, vel, rotation_rate, accel):
    """Build nav_msgs/Odometry from CAN bus data."""
    sec, nsec = divmod(sample_data_utime, 1_000_000)
    nsec *= 1000
    return ros2_type('nav_msgs/msg/Odometry')(
        header=ros2_type('std_msgs/msg/Header')(
            stamp=ros2_type('builtin_interfaces/msg/Time')(sec=int(sec), nanosec=int(nsec)),
            frame_id='map',
            #child_frame_id='base_link',
        ),
        child_frame_id='base_link',
        pose=ros2_type('geometry_msgs/msg/PoseWithCovariance')(
            pose=ros2_type('geometry_msgs/msg/Pose')(
                position=ros2_type('geometry_msgs/msg/Point')(
                    x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
                ),
                orientation=ros2_type('geometry_msgs/msg/Quaternion')(
                    x=float(orientation[1]), y=float(orientation[2]),
                    z=float(orientation[3]), w=float(orientation[0]),
                ),
            ),
            covariance=np.zeros(36, dtype=np.float64),
        ),
        twist=ros2_type('geometry_msgs/msg/TwistWithCovariance')(
            twist=ros2_type('geometry_msgs/msg/Twist')(
                linear=ros2_type('geometry_msgs/msg/Vector3')(
                    x=float(vel[0]), y=float(vel[1]), z=float(vel[2]),
                ),
                angular=ros2_type('geometry_msgs/msg/Vector3')(
                    x=float(rotation_rate[0]), y=float(rotation_rate[1]),
                    z=float(rotation_rate[2]),
                ),
            ),
            covariance=np.zeros(36, dtype=np.float64),
        ),
    )


def make_diagnostic_array(sample_data_utime: int, name: str, diag_data: dict):
    """Build diagnostic_msgs/DiagnosticArray from CAN bus key-value data."""
    sec, nsec = divmod(sample_data_utime, 1_000_000)
    nsec *= 1000
    values = []
    for k, v in diag_data.items():
        if k != "utime":
            values.append(ros2_type('diagnostic_msgs/msg/KeyValue')(
                key=k, value=str(round(float(v), 4)),
            ))
    return ros2_type('diagnostic_msgs/msg/DiagnosticArray')(
        header=ros2_type('std_msgs/msg/Header')(
            stamp=ros2_type('builtin_interfaces/msg/Time')(sec=int(sec), nanosec=int(nsec)),
            frame_id='',
        ),
        status=[
            ros2_type('diagnostic_msgs/msg/DiagnosticStatus')(
                level=0, name=name, message='OK', hardware_id='',
                values=values,
            )
        ],
    )


def make_tf_transform_stamped(stamp_ns: int, parent: str, child: str,
                               translation, rotation):
    """Build a single geometry_msgs/TransformStamped."""
    return ros2_type('geometry_msgs/msg/TransformStamped')(
        header=make_header_ns(stamp_ns, parent),
        child_frame_id=child,
        transform=ros2_type('geometry_msgs/msg/Transform')(
            translation=ros2_type('geometry_msgs/msg/Vector3')(
                x=float(translation[0]),
                y=float(translation[1]),
                z=float(translation[2]),
            ),
            rotation=ros2_type('geometry_msgs/msg/Quaternion')(
                x=float(rotation[1]),
                y=float(rotation[2]),
                z=float(rotation[3]),
                w=float(rotation[0]),
            ),
        ),
    )


def make_tf_message(transforms: list):
    """Build tf2_msgs/TFMessage from a list of TransformStamped."""
    return ros2_type('tf2_msgs/msg/TFMessage')(transforms=transforms)


def make_marker_array(markers: list):
    """Build visualization_msgs/MarkerArray."""
    return ros2_type('visualization_msgs/msg/MarkerArray')(markers=markers)


def make_occupancy_grid(header, info: Optional, data, width, height, resolution,
                        origin_translation=(0, 0, 0), origin_rotation=(0, 0, 0, 1)):
    """Build nav_msgs/OccupancyGrid."""
    if info is None:
        info = ros2_type('nav_msgs/msg/MapMetaData')(
            map_load_time=header.stamp,
            resolution=resolution,
            width=width,
            height=height,
            origin=ros2_type('geometry_msgs/msg/Pose')(
                position=ros2_type('geometry_msgs/msg/Point')(
                    x=float(origin_translation[0]),
                    y=float(origin_translation[1]),
                    z=float(origin_translation[2]),
                ),
                orientation=ros2_type('geometry_msgs/msg/Quaternion')(
                    x=float(origin_rotation[1]),
                    y=float(origin_rotation[2]),
                    z=float(origin_rotation[3]),
                    w=float(origin_rotation[0]),
                ),
            ),
        )
    # Convert to int8 signed (ROS2 OccupancyGrid uses int8, -1=unknown)
    occupancy = data.astype(np.int8)
    return ros2_type('nav_msgs/msg/OccupancyGrid')(
        header=header,
        info=info,
        data=occupancy.flatten(),
    )


# ---------------------------------------------------------------------------
# CAN bus message parsers (replacing get_imu_msg, get_odom_msg, etc.)
# Returns (timestamp_ns, topic, ros2_message)
# ---------------------------------------------------------------------------
def can_to_imu(can_msg):
    utime = can_msg["utime"]
    return (int(utime) * 1000, '/imu',
            make_imu(utime, can_msg["linear_accel"], can_msg["q"], can_msg["rotation_rate"]))


def can_to_odom(can_msg):
    utime = can_msg["utime"]
    d = can_msg
    return (int(utime) * 1000, '/odom',
            make_odometry(utime, d["pos"], d["orientation"], d["vel"],
                          d["rotation_rate"], d["accel"]))


def can_to_diagnostics(name):
    def wrap(can_msg):
        utime = can_msg["utime"]
        return (int(utime) * 1000, '/diagnostics',
                make_diagnostic_array(utime, name, can_msg))
    return wrap


# ---------------------------------------------------------------------------
# Centerline markers → MarkerArray
# ---------------------------------------------------------------------------
def make_centerline_markers(nusc, scene, nusc_map, stamp_ns):
    pose_lists = nusc_map.discretize_centerlines(1)
    bbox = scene_bounding_box(nusc, scene, nusc_map)

    contained = []
    for pose_list in pose_lists:
        filtered = [p for p in pose_list if rectContains(bbox, p)]
        if len(filtered) > 1:
            contained.append(filtered)

    markers = []
    Marker = ros2_type('visualization_msgs/msg/Marker')
    Point = ros2_type('geometry_msgs/msg/Point')
    header = make_header_ns(stamp_ns, 'map')
    Duration = ros2_type('builtin_interfaces/msg/Duration')
    ColorRGBA = ros2_type('std_msgs/msg/ColorRGBA')
    CompressedImage_empty = ros2_type('sensor_msgs/msg/CompressedImage')(
        header=ros2_type('std_msgs/msg/Header')(
            stamp=ros2_type('builtin_interfaces/msg/Time')(sec=0, nanosec=0),
            frame_id='',
        ),
        format='',
        data=np.array([], dtype=np.uint8),
    )
    MeshFile_empty = ros2_type('visualization_msgs/msg/MeshFile')(
        filename='', data=np.array([], dtype=np.uint8),
    )

    for i, pts in enumerate(contained):
        points = [Point(x=float(p[0]), y=float(p[1]), z=0.0) for p in pts]
        markers.append(Marker(
            header=header,
            ns='centerlines',
            id=i,
            type=4,  # LINE_STRIP
            action=0,  # ADD
            pose=ros2_type('geometry_msgs/msg/Pose')(
                position=Point(x=0, y=0, z=0),
                orientation=ros2_type('geometry_msgs/msg/Quaternion')(x=0, y=0, z=0, w=1),
            ),
            scale=ros2_type('geometry_msgs/msg/Vector3')(x=0.1, y=0.1, z=0.1),
            color=ColorRGBA(r=51/255, g=160/255, b=44/255, a=1.0),
            lifetime=Duration(sec=0, nanosec=0),
            frame_locked=True,
            points=points,
            colors=[],
            texture_resource='',
            texture=CompressedImage_empty,
            uv_coordinates=[],
            text='',
            mesh_resource='',
            mesh_file=MeshFile_empty,
            mesh_use_embedded_materials=False,
        ))
    return make_marker_array(markers)


# ---------------------------------------------------------------------------
# Annotations → MarkerArray (3D bounding boxes as CUBE markers)
# ---------------------------------------------------------------------------
def make_annotation_markers(nusc, ann_ids, stamp_ns):
    Marker = ros2_type('visualization_msgs/msg/Marker')
    header = make_header_ns(stamp_ns, 'map')
    Duration = ros2_type('builtin_interfaces/msg/Duration')
    ColorRGBA = ros2_type('std_msgs/msg/ColorRGBA')
    CompressedImage_empty = ros2_type('sensor_msgs/msg/CompressedImage')(
        header=ros2_type('std_msgs/msg/Header')(
            stamp=ros2_type('builtin_interfaces/msg/Time')(sec=0, nanosec=0),
            frame_id='',
        ),
        format='',
        data=np.array([], dtype=np.uint8),
    )
    MeshFile_empty = ros2_type('visualization_msgs/msg/MeshFile')(
        filename='', data=np.array([], dtype=np.uint8),
    )
    markers = []

    for i, ann_id in enumerate(ann_ids):
        ann = nusc.get("sample_annotation", ann_id)
        c = np.array(nusc.explorer.get_color(ann["category_name"])) / 255.0

        markers.append(Marker(
            header=header,
            ns='annotations',
            id=i,
            type=1,  # CUBE
            action=0,  # ADD
            pose=ros2_type('geometry_msgs/msg/Pose')(
                position=ros2_type('geometry_msgs/msg/Point')(
                    x=float(ann["translation"][0]),
                    y=float(ann["translation"][1]),
                    z=float(ann["translation"][2]),
                ),
                orientation=ros2_type('geometry_msgs/msg/Quaternion')(
                    w=float(ann["rotation"][0]),
                    x=float(ann["rotation"][1]),
                    y=float(ann["rotation"][2]),
                    z=float(ann["rotation"][3]),
                ),
            ),
            scale=ros2_type('geometry_msgs/msg/Vector3')(
                x=float(ann["size"][1]),
                y=float(ann["size"][0]),
                z=float(ann["size"][2]),
            ),
            color=ColorRGBA(r=float(c[0]), g=float(c[1]), b=float(c[2]), a=0.5),
            lifetime=Duration(sec=0, nanosec=0),
            frame_locked=True,
            points=[],
            colors=[],
            texture_resource='',
            texture=CompressedImage_empty,
            uv_coordinates=[],
            text='',
            mesh_resource='',
            mesh_file=MeshFile_empty,
            mesh_use_embedded_materials=False,
        ))
    return make_marker_array(markers)


# ---------------------------------------------------------------------------
# Number of sample data records in a scene
# ---------------------------------------------------------------------------
def get_num_sample_data(nusc, scene):
    count = 0
    sample = nusc.get("sample", scene["first_sample_token"])
    for sample_token in sample["data"].values():
        sd = nusc.get("sample_data", sample_token)
        while sd is not None:
            count += 1
            sd = nusc.get("sample_data", sd["next"]) if sd["next"] != "" else None
    return count


# ---------------------------------------------------------------------------
# Write a single scene to MCAP
# ---------------------------------------------------------------------------
def write_scene_to_mcap(nusc, nusc_can, scene, output_path):
    scene_name = scene["name"]
    log = nusc.get("log", scene["log_token"])
    location = log["location"]
    print(f'Loading map "{location}"')
    data_path = Path(nusc.dataroot)
    nusc_map = NuScenesMap(dataroot=data_path, map_name=location)
    print(f"Loading bitmap…")
    image = load_bitmap(nusc_map.dataroot, nusc_map.map_name, "basemap")
    print(f"Loaded {image.shape} bitmap")
    print(f"vehicle is {log['vehicle']}")

    mcap_path = output_path / f"NuScenes-v1.0-mini-{scene_name}.mcap"
    mcap_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Ros2McapWriter(str(mcap_path))
    writer.start()

    # Metadata
    writer.write_metadata("scene-info", {
        "name": scene["name"],
        "description": scene["description"],
        "location": location,
        "vehicle": log["vehicle"],
        "date_captured": log["date_captured"],
    })

    cur_sample = nusc.get("sample", scene["first_sample_token"])
    total = get_num_sample_data(nusc, scene)
    pbar = tqdm(total=total, unit="sample_data", desc=f"{scene_name}", leave=False)

    # CAN parsers
    can_parsers = [
        [nusc_can.get_messages(scene_name, "ms_imu"), 0, can_to_imu],
        [nusc_can.get_messages(scene_name, "pose"), 0, can_to_odom],
        [nusc_can.get_messages(scene_name, "steeranglefeedback"), 0, can_to_diagnostics("Steering Angle")],
        [nusc_can.get_messages(scene_name, "vehicle_monitor"), 0, can_to_diagnostics("Vehicle Monitor")],
        [nusc_can.get_messages(scene_name, "zoesensors"), 0, can_to_diagnostics("Zoe Sensors")],
        [nusc_can.get_messages(scene_name, "zoe_veh_info"), 0, can_to_diagnostics("Zoe Vehicle Info")],
    ]

    # -- Static map layers (written once at first frame timestamp) --
    first_lidar = nusc.get("sample_data", cur_sample["data"]["LIDAR_TOP"])
    first_ego = nusc.get("ego_pose", first_lidar["ego_pose_token"])
    first_stamp_ns = int(first_ego["timestamp"]) * 1000

    # Basemap → OccupancyGrid
    bbox = scene_bounding_box(nusc, scene, nusc_map)
    x, y, w, h = bbox
    img_x = int(x * 10)
    img_y = int(y * 10)
    img_w = int(w * 10)
    img_h = int(h * 10)
    map_img = np.flipud(image)[img_y: img_y + img_h, img_x: img_x + img_w]
    # Invert and scale for occupancy grid display (0=free, 100=occupied)
    occ = ((255 - map_img) * 100 / 255).astype(np.int8)
    map_header = make_header_ns(first_stamp_ns, "map")
    occupancy_grid = make_occupancy_grid(
        map_header, None, occ, img_w, img_h, 0.1,
        origin_translation=(x, y, 0),
    )
    writer.write('/map', occupancy_grid, 'nav_msgs/msg/OccupancyGrid', first_stamp_ns)

    # Centerlines → MarkerArray
    centerlines = make_centerline_markers(nusc, scene, nusc_map, first_stamp_ns)
    writer.write('/semantic_map', centerlines, 'visualization_msgs/msg/MarkerArray', first_stamp_ns)

    # -- Main loop over samples --
    while cur_sample is not None:
        sample_lidar = nusc.get("sample_data", cur_sample["data"]["LIDAR_TOP"])
        ego_pose = nusc.get("ego_pose", sample_lidar["ego_pose_token"])
        stamp_ns = int(ego_pose["timestamp"]) * 1000

        # CAN messages
        can_events = []
        for i in range(len(can_parsers)):
            msgs, idx, func = can_parsers[i]
            while idx < len(msgs) and (int(msgs[idx]["utime"]) * 1000) < stamp_ns:
                can_events.append(func(msgs[idx]))
                idx += 1
                can_parsers[i][1] = idx
        can_events.sort(key=lambda x: x[0])
        for ts, topic, msg in can_events:
            if topic == '/imu':
                writer.write(topic, msg, 'sensor_msgs/msg/Imu', ts)
            elif topic == '/odom':
                writer.write(topic, msg, 'nav_msgs/msg/Odometry', ts)
            elif topic == '/diagnostics':
                writer.write(topic, msg, 'diagnostic_msgs/msg/DiagnosticArray', ts)

        # TF: ego → base_link
        tf_ego = make_tf_transform_stamped(
            stamp_ns, 'map', 'base_link',
            ego_pose["translation"], ego_pose["rotation"],
        )
        writer.write('/tf', make_tf_message([tf_ego]), 'tf2_msgs/msg/TFMessage', stamp_ns)

        # Drivable area → OccupancyGrid
        translation = ego_pose["translation"]
        rotation = Quaternion(ego_pose["rotation"])
        yaw_rad = quaternion_yaw(rotation)
        yaw_deg = float(yaw_rad / np.pi * 180)
        patch_box = (translation[0], translation[1], 32, 32)
        canvas_size = (patch_box[2] * 10, patch_box[3] * 10)
        drivable = nusc_map.get_map_mask(patch_box, yaw_deg, ["drivable_area"], canvas_size)[0]
        pos_x = translation[0] - (16 * math.cos(yaw_rad)) + (16 * math.sin(yaw_rad))
        pos_y = translation[1] - (16 * math.sin(yaw_rad)) - (16 * math.cos(yaw_rad))
        qz = Quaternion(axis=(0, 0, 1), radians=yaw_rad)
        occ_grid = make_occupancy_grid(
            make_header_ns(stamp_ns, 'map'), None,
            drivable.astype(np.int8) * 100,
            drivable.shape[1], drivable.shape[0], 0.1,
            origin_translation=(pos_x, pos_y, 0.01),
            origin_rotation=(qz.w, qz.x, qz.y, qz.z),
        )
        writer.write('/drivable_area', occ_grid, 'nav_msgs/msg/OccupancyGrid', stamp_ns)

        # Sensor data
        for sensor_id, sample_token in cur_sample["data"].items():
            sd = nusc.get("sample_data", sample_token)
            pbar.update(1)
            topic = '/' + sensor_id

            # Sensor TF
            calib_sensor = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            tf_sensor = make_tf_transform_stamped(
                stamp_ns, 'base_link', sensor_id,
                calib_sensor["translation"], calib_sensor["rotation"],
            )
            writer.write('/tf', make_tf_message([tf_sensor]), 'tf2_msgs/msg/TFMessage', stamp_ns)

            if sd["sensor_modality"] == "camera":
                img = make_compressed_image(data_path, sd, sensor_id)
                writer.write(topic + '/image_rect_compressed', img,
                             'sensor_msgs/msg/CompressedImage',
                             int(sd["timestamp"]) * 1000)
                cinfo = make_camera_info(nusc, sd, sensor_id)
                writer.write(topic + '/camera_info', cinfo,
                             'sensor_msgs/msg/CameraInfo',
                             int(sd["timestamp"]) * 1000)

            elif sd["sensor_modality"] == "lidar":
                pc = make_lidar_pointcloud2(data_path, sd, sensor_id)
                writer.write(topic, pc, 'sensor_msgs/msg/PointCloud2',
                             int(sd["timestamp"]) * 1000)

            elif sd["sensor_modality"] == "radar":
                pc = make_radar_pointcloud2(data_path, sd, sensor_id)
                writer.write(topic, pc, 'sensor_msgs/msg/PointCloud2',
                             int(sd["timestamp"]) * 1000)

        # Pose (local frame)
        pose_msg = make_pose_stamped(stamp_ns, 'base_link', [0, 0, 0], [1, 0, 0, 0])
        writer.write('/pose', pose_msg, 'geometry_msgs/msg/PoseStamped', stamp_ns)

        # GPS
        lat, lon = derive_latlon(location, ego_pose)
        gps_msg = make_nav_sat_fix(lat, lon, ego_pose["translation"][2],
                                   ego_pose["timestamp"])
        writer.write('/gps', gps_msg, 'sensor_msgs/msg/NavSatFix', stamp_ns)

        # 3D annotations → MarkerArray
        if cur_sample["anns"]:
            markers = make_annotation_markers(nusc, cur_sample["anns"], stamp_ns)
            writer.write('/markers/annotations', markers,
                         'visualization_msgs/msg/MarkerArray', stamp_ns)

        # -- Non-keyframe sensor data --
        non_kf_events: list[tuple[int, str, str, object]] = []
        for sensor_id, sample_token in cur_sample["data"].items():
            sd = nusc.get("sample_data", sample_token)
            next_token = sd["next"]
            while next_token != "":
                next_sd = nusc.get("sample_data", next_token)
                if next_sd["is_key_frame"]:
                    break
                pbar.update(1)
                next_ego = nusc.get("ego_pose", next_sd["ego_pose_token"])
                nkf_stamp = int(next_ego["timestamp"]) * 1000

                if next_sd["sensor_modality"] in ("camera", "lidar", "radar"):
                    non_kf_events.append((nkf_stamp, '/tf', next_ego))

                sensor_stamp_ns = int(next_sd["timestamp"]) * 1000
                prefix = '/' + sensor_id
                if next_sd["sensor_modality"] == "camera":
                    img = make_compressed_image(data_path, next_sd, sensor_id)
                    non_kf_events.append(
                        (sensor_stamp_ns, prefix + '/image_rect_compressed', img))
                    cinfo = make_camera_info(nusc, next_sd, sensor_id)
                    non_kf_events.append(
                        (sensor_stamp_ns, prefix + '/camera_info', cinfo))
                elif next_sd["sensor_modality"] == "lidar":
                    pc = make_lidar_pointcloud2(data_path, next_sd, sensor_id)
                    non_kf_events.append(
                        (sensor_stamp_ns, prefix, pc))
                elif next_sd["sensor_modality"] == "radar":
                    pc = make_radar_pointcloud2(data_path, next_sd, sensor_id)
                    non_kf_events.append(
                        (sensor_stamp_ns, prefix, pc))

                next_token = next_sd["next"]

        # Write non-keyframe events sorted by timestamp
        non_kf_events.sort(key=lambda x: x[0])
        for ts, nkf_topic, nkf_msg in non_kf_events:
            if nkf_topic == '/tf':
                ego_data = nkf_msg
                tf_msg = make_tf_transform_stamped(
                    ts, 'map', 'base_link',
                    ego_data["translation"], ego_data["rotation"],
                )
                writer.write('/tf', make_tf_message([tf_msg]), 'tf2_msgs/msg/TFMessage', ts)
            else:
                ros_type = (
                    'sensor_msgs/msg/CompressedImage'
                    if '/image_rect_compressed' in nkf_topic
                    else 'sensor_msgs/msg/CameraInfo'
                    if '/camera_info' in nkf_topic
                    else 'sensor_msgs/msg/PointCloud2'
                )
                writer.write(nkf_topic, nkf_msg, ros_type, ts)

        # Move to next sample
        cur_sample = nusc.get("sample", cur_sample["next"]) if cur_sample.get("next") != "" else None

    pbar.close()
    writer.finish()
    print(f"Finished writing {mcap_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert nuScenes mini to MCAP with ROS2 CDR encoding",
    )
    script_dir = Path(__file__).parent
    parser.add_argument("--data-dir", "-d", default=script_dir / "data",
                        help="nuScenes data directory")
    parser.add_argument("--output-dir", "-o", type=Path,
                        default=script_dir / "output_ros2",
                        help="MCAP output directory")
    parser.add_argument("--scene", "-s", nargs="*",
                        help="specific scene(s) to convert")
    parser.add_argument("--list-only", action="store_true",
                        help="list scenes and exit")
    args = parser.parse_args()

    nusc_can = NuScenesCanBus(dataroot=str(args.data_dir))
    nusc = NuScenes(version="v1.0-mini", dataroot=str(args.data_dir), verbose=True)

    if args.list_only:
        nusc.list_scenes()
        return

    for scene in nusc.scene:
        if args.scene and scene["name"] not in args.scene:
            continue
        write_scene_to_mcap(nusc, nusc_can, scene, args.output_dir)


if __name__ == "__main__":
    main()
