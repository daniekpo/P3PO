import pickle
from pathlib import Path
import imageio
from p3po.read_data.p3po_xarm import BCDataset, get_relative_action, get_quaternion_orientation
from p3po.suite.ur5e_env import UR5Env

from dm_env import specs
import numpy as np
from p3po.replay_buffer import make_expert_replay_loader

path = "/scratch/data/open_teach/processed_data_pkl"
processed_path = "./processed_data"
tasks = ["grab_ball"]
num_demos_per_task = 100
obs_type = "features"
history = True
history_len = 100
prompt =  "text"
temporal_agg = False
num_future_actions = 1
img_size = 128 # should not matter
action_after_steps = 1
intermediate_goal_step = 30
store_actions = True
training_keys = ["graph"]
subsample = 1
skip_first_n = 0
relative_actions = True
debug_idx = 0

env_idx=0
episode_idx=35
start_idx=0
end_idx=500

debug_pickle_path = Path(f"debug/sample_env_{env_idx}_episode_{episode_idx}.pkl")
if debug_pickle_path.exists():
    with open(debug_pickle_path, "rb") as f:
        sample = pickle.load(f)
    with open("debug/action_spec.pkl", "rb") as f:
        action_spec = pickle.load(f)
    with open("debug/stats.pkl", "rb") as f:
        stats = pickle.load(f)
else:
    dataset = BCDataset(
        path=path,
        processed_path=processed_path,
        tasks=tasks,
        num_demos_per_task=num_demos_per_task,
        obs_type=obs_type,
        history=history,
        history_len=history_len,
        prompt=prompt,
        temporal_agg=temporal_agg,
        num_future_actions=num_future_actions,
        img_size=img_size,
        action_after_steps=action_after_steps,
        intermediate_goal_step=intermediate_goal_step,
        store_actions=store_actions,
        training_keys=training_keys,
        subsample=subsample,
        skip_first_n=skip_first_n,
        relative_actions=relative_actions,
        use_quaternion_orientation=True
    )

    action_spec = specs.BoundedArray(
        (dataset._max_action_dim,),
        np.float32,
        dataset.stats["actions"]["min"],
        dataset.stats["actions"]["max"],
        "action",
    )
    batch_size=1 # actual training used 64
    sample = dataset._sample(env_idx, episode_idx, start_idx, end_idx)
    stats = dataset.stats # stats passed to agent act function

    with open(debug_pickle_path, "wb") as f:
        pickle.dump(sample, f)
    with open("debug/action_spec.pkl", "wb") as f:
        pickle.dump(action_spec, f)
    with open("debug/stats.pkl", "wb") as f:
        pickle.dump(stats, f)


actions = sample["actions"] # B x history_len [x num_future_actions] x action_dim

# Used only when use_priopro is set
proprioceptive = sample["proprioceptive"] # B x history_len x 8
graph = sample["graph"] # B x history_len x (n_points * 3)


proprio_key = "proprioceptive" # only when use_prioprio is set
def preprocess(pose, norm_stats):
    norm_min =norm_stats[proprio_key]['min']
    norm_max = norm_stats[proprio_key]["max"]

    new_pose = (pose - norm_min) / (norm_max - norm_min + 1e-5)
    return new_pose

def postprocess(action, norm_stats):
    actions_max = norm_stats["actions"]["max"]
    actions_min = norm_stats["actions"]["min"]

    action = action * (actions_max - actions_min) + actions_min
    return action


print("Initializing robot")
env = UR5Env(camera_names=["front_right"], include_depth=False, training=False, relative_actions=True)
env.reset()
print("Initialized")

for i in range(len(actions)):
    action = actions[i]
    post_processed_action = postprocess(action, stats)
    env.step(post_processed_action)