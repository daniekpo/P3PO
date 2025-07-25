import pickle
import numpy as np


def main(file_path, output_path):
    demonstrations = pickle.load(open(file_path, 'rb'))

    observations = demonstrations['observations']

    # Separate lists for actions and observations
    actions_list = []
    observations_list = []

    for observation in observations:
        cartesian_state = observation['cartesian_states']
        gripper_state = observation['gripper_states']

        # append gripper state to cartesion state in the first dimension to go from 6d to 7d
        actions = np.concatenate([cartesian_state, gripper_state.reshape(-1, 1)], axis=1)

        obs_data = {
            'pixels': observation['pixels0'],
            # 'depth': observation['depth0'],
        }

        actions_list.append(actions)
        observations_list.append(obs_data)

    # Structure as P3PO expects
    converted_data = {
        'actions': np.array(actions_list, dtype=object),  # Array of arrays as P3PO expects
        'observations': observations_list
    }

    with open(output_path, 'wb') as f:
        pickle.dump(converted_data, f)


if __name__ == '__main__':
    file_path = '/scratch/repos/P3PO/expert_demos/xarm_env/xarm_test.pkl'
    output_path = './data/xarm_test_converted.pkl'
    main(file_path, output_path)