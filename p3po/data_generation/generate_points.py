from pathlib import Path
import sys

sys.path.append("../")

import pickle
import cv2
import yaml
import imageio
from natsort import natsorted

import torch

from points_class import PointsClass
from pathlib import Path
import h5py
from tqdm import tqdm

def generate_points_h5(task_name, cfg, task_path):
    pixel_key = "pixels0"
    depth_key = "depth0"
    subsample = 1
    use_gt_depth = True
    write_videos = False
    mark_every = 8
    video_paths = []

    points_class = PointsClass(**cfg)
    episode_list = []

    with h5py.File(task_path, "r") as h5f:
        demos = h5f["observations"]
        demo_keys = natsorted(demos.keys())

        pbar = tqdm(
            total=len(demo_keys),
            desc=f"Generating points for {task_name}",
            leave=True,
            dynamic_ncols=True,
        )

        for i, demo_key in enumerate(demo_keys):
            demo = demos[demo_key]
            frames = demo[pixel_key][...]
            depth = demo[depth_key][...]

            points_class.add_to_image_list(frames[0])
            points_class.find_semantic_similar_points()
            points_class.track_points(is_first_step=True)
            points_class.track_points(one_frame=(mark_every == 1))

            if use_gt_depth:
                points_class.set_depth(depth[0])
            else:
                points_class.get_depth()

            points_list = []
            points = points_class.get_points()
            points_list.append(points[0])

            if write_videos:
                video_list = []
                image = points_class.plot_image()
                video_list.append(image[0])

            for idx, image in enumerate(frames[1:]):
                points_class.add_to_image_list(image)
                if use_gt_depth:
                    points_class.set_depth(depth[idx + 1])

                if (idx + 1) % mark_every == 0 or idx == (len(frames) - 2):
                    to_add = mark_every - (idx + 1) % mark_every
                    if to_add < mark_every:
                        for j in range(to_add):
                            points_class.add_to_image_list(image)
                    else:
                        to_add = 0

                    points_class.track_points(one_frame=(mark_every == 1))
                    if not use_gt_depth:
                        points_class.get_depth(last_n_frames=mark_every)

                    points = points_class.get_points(last_n_frames=mark_every)
                    for j in range(mark_every - to_add):
                        points_list.append(points[j])

                    if write_videos:
                        images = points_class.plot_image(last_n_frames=mark_every)
                        for j in range(mark_every - to_add):
                            video_list.append(images[j])


            if write_videos:
                imageio.mimsave(f"videos/{task_name}_%d.mp4" % i, video_list, fps=30)

            episode_list.append(torch.stack(points_list))
            points_class.reset_episode()
            pbar.update(1)

        pbar.close()

    final_graph = {}
    final_graph["episode_list"] = episode_list
    final_graph["subsample"] = subsample
    final_graph["pixel_key"] = pixel_key
    final_graph["use_gt_depth"] = use_gt_depth
    final_graph["gt_depth_key"] = depth_key
    final_graph["pickle_path"] = task_file_path
    final_graph["video_paths"] = video_paths
    final_graph["cfg"] = cfg

    Path(f"{cfg['root_dir']}/processed_data/points").mkdir(parents=True, exist_ok=True)
    pickle.dump(
        final_graph,
        open(f"{cfg['root_dir']}/processed_data/points/{task_name}.pkl", "wb"),
    )


def generate_points(task_name, cfg, pickle_path, read_from_pickle = True):
    # TODO: Set if you want to read from a pickle or from mp4 files
    # If you are reading from a pickle please make sure that the images are RGB not BGR


    pickle_image_key = "pixels0"

    # TODO: If you want to use gt depth, set to True and set the key for the depth in the pickle
    # To use gt depth, the depth must be in the same pickle as the images
    # We assume the input depth is in the form width x height
    use_gt_depth = True
    gt_depth_key = "depth0"

    # Otherwise we need to add videos to a list
    # TODO: A list of videos to read from if you are not loading data from a pickle
    video_paths = []

    # TODO: Set to true if you want to save a video of the points being tracked
    write_videos = False

    # TODO:  If you want to subsample the frames, set the subsample rate here. Note you will have to update your dataset to
    # reflect the subsampling rate, we do not do this for you.
    subsample = 1

    if read_from_pickle:
        examples = pickle.load(open(pickle_path, "rb"))
        num_demos = len(examples["observations"])
    else:
        num_demos = len(video_paths)

    if write_videos:
        Path(f"{cfg['root_dir']}/p3po/data_generation/videos").mkdir(
            parents=True, exist_ok=True
        )

    # Initialize the PointsClass object
    points_class = PointsClass(**cfg)
    episode_list = []

    mark_every = 8
    for i in range(num_demos):
        # Read the frames from the pickle or video, these frames must be in RGB so if reading from a pickle make sure to convert if necessary
        if read_from_pickle:
            frames = examples["observations"][i][pickle_image_key][0::subsample]
            if use_gt_depth:
                depth = examples["observations"][i][gt_depth_key][0::subsample]
        else:
            frames = []
            video = cv2.VideoCapture(video_paths[i])
            subsample_counter = 0
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                if subsample_counter % subsample == 0:
                    # CV2 reads in BGR format, so we need to convert to RGB
                    frames.append(frame[:, :, ::-1])
                subsample_counter += 1
            video.release()

        points_class.add_to_image_list(frames[0])
        points_class.find_semantic_similar_points()
        points_class.track_points(is_first_step=True)
        points_class.track_points(one_frame=(mark_every == 1))

        if use_gt_depth:
            points_class.set_depth(depth[0])
        else:
            points_class.get_depth()

        points_list = []
        points = points_class.get_points()
        points_list.append(points[0])

        if write_videos:
            video_list = []
            image = points_class.plot_image()
            video_list.append(image[0])

        for idx, image in enumerate(frames[1:]):
            points_class.add_to_image_list(image)
            if use_gt_depth:
                points_class.set_depth(depth[idx + 1])

            if (idx + 1) % mark_every == 0 or idx == (len(frames) - 2):
                to_add = mark_every - (idx + 1) % mark_every
                if to_add < mark_every:
                    for j in range(to_add):
                        points_class.add_to_image_list(image)
                else:
                    to_add = 0

                points_class.track_points(one_frame=(mark_every == 1))
                if not use_gt_depth:
                    points_class.get_depth(last_n_frames=mark_every)

                points = points_class.get_points(last_n_frames=mark_every)
                for j in range(mark_every - to_add):
                    points_list.append(points[j])

                if write_videos:
                    images = points_class.plot_image(last_n_frames=mark_every)
                    for j in range(mark_every - to_add):
                        video_list.append(images[j])

        if write_videos:
            imageio.mimsave(f"videos/{task_name}_%d.mp4" % i, video_list, fps=30)

        episode_list.append(torch.stack(points_list))
        points_class.reset_episode()

    final_graph = {}
    final_graph["episode_list"] = episode_list
    final_graph["subsample"] = subsample
    final_graph["pixel_key"] = pickle_image_key
    final_graph["use_gt_depth"] = use_gt_depth
    final_graph["gt_depth_key"] = gt_depth_key
    final_graph["pickle_path"] = pickle_path
    final_graph["video_paths"] = video_paths
    final_graph["cfg"] = cfg

    Path(f"{cfg['root_dir']}/processed_data/points").mkdir(parents=True, exist_ok=True)
    pickle.dump(
        final_graph,
        open(f"{cfg['root_dir']}/processed_data/points/{task_name}.pkl", "wb"),
    )


with open("../cfgs/suite/p3po.yaml") as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

tasks_dir = Path("/scratch/data/open_teach/training_data/expert_demos_w_gt_lite/frames")
task_names = natsorted([f.stem for f in tasks_dir.iterdir()])
for task_name in task_names:
    # pickle_path = "/fs/vulcan-projects/verigraph_3d/data/P3PO/expert_demos/put_green_block_in_chest.pkl"
    task_file_path = tasks_dir / f"{task_name}.h5"
    # generate_points(task_name, cfg, pickle_path, read_from_pickle=True)
    generate_points_h5(task_name, cfg, task_file_path)
