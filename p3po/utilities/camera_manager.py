import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image
import threading
from typing import Optional

from cv2 import VideoWriter, VideoWriter_fourcc

from utilities.constants import cameras_serial_to_name
from utilities.general import ensure_path


class CameraDelayedFrameError(Exception):
    pass


class CameraManager:
    def __init__(self, camera_names=None, high_resolution=False, high_accuracy=False):
        self.camera_names = camera_names
        self.devices = {}
        self.color_intrinsics = {}
        self.align = {}
        self.depth_unit = 0.001  # meters. Default for 4XX Intel RealSense cameras
        self.high_resolution = high_resolution
        self.filters = {}
        self.high_accuracy = high_accuracy
        self.__configure_cameras__()

    def __configure_cameras__(self):
        if self.high_resolution:
            res_width = 1280
            res_height = 720
            fps = 15
        else:
            res_width = 640
            res_height = 480
            fps = 30

        ctx = rs.context()
        for device in ctx.query_devices():
            serial = device.get_info(rs.camera_info.serial_number)
            cam_name = cameras_serial_to_name.get(serial, "unknown")

            if self.camera_names is None or cam_name in self.camera_names:
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)

                config.enable_stream(
                    rs.stream.depth, res_width, res_height, rs.format.z16, fps
                )
                config.enable_stream(
                    rs.stream.color, res_width, res_height, rs.format.rgb8, fps
                )

                # Set up alignment to align depth and color frames
                self.align[cam_name] = rs.align(rs.stream.color)

                profile = pipeline.start(config)
                self.devices[cam_name] = pipeline

                intr = (
                    profile.get_stream(rs.stream.color)
                    .as_video_stream_profile()
                    .get_intrinsics()
                )

                self.color_intrinsics[cam_name] = intr

                if self.high_accuracy:
                    HIGH_ACCURACY_PRESET = 3
                    depth_sensor = profile.get_device().first_depth_sensor()
                    if depth_sensor.supports(rs.option.visual_preset):
                        depth_sensor.set_option(rs.option.visual_preset, HIGH_ACCURACY_PRESET)

                    self.filters[cam_name] = {
                        "decimation": rs.decimation_filter(),
                        "spatial": rs.spatial_filter(),
                        "temporal": rs.temporal_filter()
                    }

                    # self.filters[cam_name]["decimation"].set_option(rs.option.filter_magnitude, 2)
                    # self.filters[cam_name]["spatial"].set_option(rs.option.filter_magnitude, 2)
                    # self.filters[cam_name]["spatial"].set_option(rs.option.filter_smooth_alpha, 0.5)
                    # self.filters[cam_name]["spatial"].set_option(rs.option.filter_smooth_delta, 20)
                    # self.filters[cam_name]["temporal"].set_option(rs.option.filter_smooth_alpha, 0.4)
                    # self.filters[cam_name]["temporal"].set_option(rs.option.filter_smooth_delta, 20)


        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_unit = depth_sensor.get_option(rs.option.depth_units)

        self.__flush__()

    def hardware_reset(self):
        ctx = rs.context()
        for device in ctx.query_devices():
            device.hardware_reset()
        time.sleep(2)
        self.__configure_cameras__()

    def __flush__(self):
        for i in range(20):
            self.get_data()

    def get_data(self, include_timestamp=False):
        data = {}

        # use the same timestamp for all cameras
        timestamp = None

        for cam_name, pipeline in self.devices.items():
            if self.camera_names is not None and cam_name not in self.camera_names:
                continue

            try:
                frames = pipeline.wait_for_frames()
                if include_timestamp and timestamp is None:
                    timestamp = frames.get_timestamp()

                aligned_frames = self.align[cam_name].process(frames)

                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                if self.high_accuracy:
                    filters = self.filters[cam_name]
                    depth_frame = filters["decimation"].process(depth_frame)
                    depth_frame = filters["spatial"].process(depth_frame)
                    depth_frame = filters["temporal"].process(depth_frame)

                # copy to free up the buffer for reuse by the camera
                depth_image = np.asanyarray(depth_frame.get_data()).copy()
                img = np.asanyarray(color_frame.get_data()).copy()

                if include_timestamp:
                    data[cam_name] = (img, depth_image, timestamp)
                else:
                    data[cam_name] = (img, depth_image)

            except RuntimeError as e:
                print(f"Error on {cam_name}: {e}")

        return data

    def get_depth_unit(self):
        return self.depth_unit

    def save_camera_data(
        self,
        save_dir,
        include_depth=True,
        prefix=None,
        save_depth_color=False,
        save_depth_grayscale=False,
        separate_cam_dir=False,
    ):
        data = self.get_data()
        if not separate_cam_dir:
            depth_dir = ensure_path(save_dir, "depth")
            img_dir = ensure_path(save_dir, "rgb")

        for cam_name, (img, depth_image) in data.items():
            if separate_cam_dir:
                depth_dir = ensure_path(save_dir, cam_name, "depth")
                img_dir = ensure_path(save_dir, cam_name, "rgb")

            save_prefix = prefix if prefix else cam_name

            Image.fromarray(img).save(f"{img_dir}/{save_prefix}.png")

            if include_depth:
                np.save(f"{depth_dir}/{save_prefix}.npy", depth_image)

            if save_depth_color:
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                )
                Image.fromarray(depth_colormap).save(
                    f"{depth_dir}/{save_prefix}_depth_rgb.png"
                )
            if save_depth_grayscale:
                depth_grayscale = cv2.convertScaleAbs(
                    depth_image, alpha=255.0 / depth_image.max()
                )
                Image.fromarray(depth_grayscale).save(
                    f"{depth_dir}/{save_prefix}_depth.png"
                )

    def get_device_names(self):
        return self.devices.keys()

    def get_aligned_intrinsics(self):
        result = {}

        for camera_name, pipeline in self.devices.items():
            frames = pipeline.wait_for_frames()
            frames = self.align[camera_name].process(frames)
            profile = frames.get_profile()
            intrinsics = profile.as_video_stream_profile().get_intrinsics()

            result[camera_name] = {
                "width": intrinsics.width,
                "height": intrinsics.height,
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "cx": intrinsics.ppx,
                "cy": intrinsics.ppy,
                "distortion": intrinsics.coeffs,
            }

        return result

    def get_color_intrinsics(self):
        return self.color_intrinsics

    def get_depth_intrinsics(self):
        result = {}
        for key, pipeline in self.devices.items():
            try:
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()

                # Get the depth stream's intrinsic parameters
                if depth_frame:
                    intrinsics = (
                        depth_frame.profile.as_video_stream_profile().get_intrinsics()
                    )

                    result[key] = {
                        "width": intrinsics.width,
                        "height": intrinsics.height,
                        "fx": intrinsics.fx,
                        "fy": intrinsics.fy,
                        "cx": intrinsics.ppx,
                        "cy": intrinsics.ppy,
                    }

            except Exception as e:
                pipeline.stop()
                raise e

        return result

    def start_recording_video(
        self,
        camera_names: Optional[list[str]] = None,
        fps: int = 30,
        max_duration_seconds: Optional[float] = None,
        include_depth: bool = False,
    ) -> None:
        """
        Start recording video from specified cameras using a single synchronized thread.

        Args:
            camera_names: List of camera names to record from. If None, records from all initialized cameras
            fps: Frames per second for recording
            max_duration_seconds: Optional maximum duration in seconds
            include_depth: Whether to record depth data alongside video frames
        """
        # Check if already recording
        if hasattr(self, "recording_thread") and self.recording_thread is not None:
            print("Recording already in progress. Stop current recording first.")
            return

        # Use all available cameras if none specified
        if camera_names is None:
            camera_names = list(self.devices.keys())

        # Validate camera names
        invalid_cameras = [name for name in camera_names if name not in self.devices]
        if invalid_cameras:
            raise ValueError(
                f"Cameras {invalid_cameras} not initialized. Available cameras: {list(self.devices.keys())}"
            )

        # Initialize recording state
        self.recording_thread = None
        self.stop_recording = False
        self.img_sequences = {cam: [] for cam in camera_names}
        self.depth_sequences = (
            {cam: [] for cam in camera_names} if include_depth else None
        )
        self.start_time = time.time()
        self.fps = fps
        self.frame_timestamps = []  # Store timestamps for each frame set
        self.include_depth = include_depth
        self.max_frames = 10000  # Limit to prevent memory issues

        def record_frames():
            try:
                frame_interval = 1.0 / fps
                next_frame_time = time.time() + frame_interval

                while not self.stop_recording:
                    if (
                        max_duration_seconds is not None
                        and time.time() - self.start_time > max_duration_seconds
                    ):
                        break

                    # Check memory bounds
                    if len(self.frame_timestamps) >= self.max_frames:
                        print(
                            f"Reached maximum frame limit ({self.max_frames}). Stopping recording."
                        )
                        break

                    # Get current timestamp
                    current_time = time.time()

                    # If we're ahead of schedule, sleep until next frame time
                    if current_time < next_frame_time:
                        time.sleep(next_frame_time - current_time)

                    # Capture frames from all cameras simultaneously
                    data = self.get_data(include_timestamp=True)
                    timestamp = None

                    # Store frames and timestamp
                    for cam_name in camera_names:
                        if cam_name in data:
                            img, depth, timestamp = data[cam_name]
                            if include_depth:
                                self.img_sequences[cam_name].append(img)
                                self.depth_sequences[cam_name].append(depth)
                            else:
                                self.img_sequences[cam_name].append(img)

                    if timestamp is not None:
                        self.frame_timestamps.append(timestamp)

                    # Calculate next frame time
                    next_frame_time = current_time + frame_interval

            except Exception as e:
                print(f"Error during recording: {e}")
                self.stop_recording = True

        self.recording_thread = threading.Thread(target=record_frames)
        self.recording_thread.start()

    def stop_recording_video(self, save_path: str) -> None:
        """
        Stop recording and save videos to the specified path/directory.

        Args:
            save_path: Path to save the video(s). If recording multiple cameras, this is treated as a directory
                      and camera names will be appended to create individual video files.
        """
        if not hasattr(self, "recording_thread") or self.recording_thread is None:
            print("No recording in progress")
            return

        self.stop_recording = True
        if self.recording_thread.is_alive():
            self.recording_thread.join()

        # Check if we have any recorded frames
        if not any(self.img_sequences.values()):
            print("No frames recorded")
            return

        try:
            # If recording multiple cameras, treat save_path as directory
            if len(self.img_sequences) > 1:
                os.makedirs(save_path, exist_ok=True)
                base_path = os.path.join(save_path, "camera_{}")
            else:
                # Handle single camera case - ensure directory exists
                save_dir = os.path.dirname(save_path)
                if save_dir:  # Only create directory if path contains a directory
                    os.makedirs(save_dir, exist_ok=True)
                base_path = save_path.replace(
                    ".mp4", ""
                )  # Remove extension for base path

            # Save videos and depth data for each camera
            for cam_name, img_seq in self.img_sequences.items():
                if not img_seq:
                    print(f"No frames recorded for camera {cam_name}")
                    continue

                # Determine the save paths for this camera
                if len(self.img_sequences) > 1:
                    video_path = f"{base_path.format(cam_name)}.mp4"
                    if self.include_depth:
                        depth_path = f"{base_path.format(cam_name)}_depth.npz"
                else:
                    video_path = f"{base_path}.mp4"
                    if self.include_depth:
                        depth_path = f"{base_path}_depth.npz"

                # Save video with error handling
                try:
                    fourcc = VideoWriter_fourcc(*"mp4v")
                    first_img = img_seq[0]
                    height, width, _ = first_img.shape

                    video_writer = VideoWriter(
                        video_path, fourcc, self.fps, (width, height)
                    )

                    if not video_writer.isOpened():
                        raise RuntimeError(
                            f"Failed to create video writer for {video_path}"
                        )

                    for img in img_seq:
                        img = img[:, :, ::-1]  # RGB to BGR for OpenCV
                        video_writer.write(img)

                    video_writer.release()
                    print(f"Video for camera {cam_name} saved to {video_path}")
                except Exception as e:
                    print(f"Error saving video for camera {cam_name}: {e}")
                    continue

                # Save depth data if included
                if self.include_depth and self.depth_sequences is not None:
                    try:
                        depth_seq = self.depth_sequences[cam_name]
                        if len(depth_seq) != len(img_seq):
                            print(
                                f"Warning: Depth sequence length ({len(depth_seq)}) doesn't match video frame count ({len(img_seq)}) for camera {cam_name}"
                            )

                        # Save depth data as compressed numpy array
                        np.savez_compressed(
                            depth_path, depth_frames=np.array(depth_seq)
                        )
                        print(f"Depth data for camera {cam_name} saved to {depth_path}")
                    except Exception as e:
                        print(f"Error saving depth data for camera {cam_name}: {e}")

            # Save timestamps for analysis
            if len(self.frame_timestamps) > 0:
                try:
                    timestamp_path = (
                        os.path.join(save_path, "frame_timestamps.txt")
                        if len(self.img_sequences) > 1
                        else f"{base_path}_timestamps.txt"
                    )
                    np.savetxt(timestamp_path, self.frame_timestamps, fmt="%.6f")
                    print(f"Frame timestamps saved to {timestamp_path}")
                except Exception as e:
                    print(f"Error saving timestamps: {e}")

        except Exception as e:
            print(f"Error saving videos: {e}")
        finally:
            # Cleanup
            self.img_sequences = {}
            self.depth_sequences = None
            self.frame_timestamps = []
            self.recording_thread = None


def get_available_presets():
    """
    Get the available presets for the depth sensor.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start the pipeline
    profile = pipeline.start(config)

    # Set high accuracy preset
    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.visual_preset):
        presets = depth_sensor.get_option_range(rs.option.visual_preset)
        print("Available Presets:")
        for i in range(int(presets.max + 1)):
            try:
                depth_sensor.set_option(rs.option.visual_preset, i)
                print(f"{i}: {depth_sensor.get_option_value_description(rs.option.visual_preset, i)}")
            except:
                continue
        # Set to high accuracy (usually preset index 1)
        depth_sensor.set_option(rs.option.visual_preset, 1)  # 1 is often 'High Accuracy'
    else:
        print("Visual presets not supported by this sensor.")

