# import cv2
import numpy as np
import os
import h5py

def prepare_data(data_folder, hdf5_file, acc_time, frame_size, debug=False):
    "Returns frames (N, 64, 64) which can be used as input for an SNN and the GT labels (N, 2, 1)."
    hdf5file = data_folder + hdf5_file
    frames = []
    h5_data = h5py.File(hdf5file)

    gt_data = []
    positions = []

    nr_of_skipped_gt = 0
    last_position = np.array([h5_data['gt_x'][0], h5_data['gt_y'][0]])
    for index, (x, y, ts) in enumerate(zip(h5_data['gt_x'], h5_data['gt_y'], h5_data['gt_t'])):
        if x < 0 or y < 0 or x > 1280 or y > 720:
            nr_of_skipped_gt += 1
            print(f"Skipped {index} x: {x}, y: {y}")
            continue
        gt_data.append((x, y, ts))

        if abs(last_position[0] - x) > frame_size or abs(last_position[1] - y) > frame_size:
            print(f"Position jumped x: {last_position[0] - x}, y: {last_position[1] - y}, reset `last_position` to current gt position!")
            last_position = np.array([x, y])
        position = np.array([x, y]) - last_position + np.array([frame_size/2, frame_size/2])
        if 0 > position[0] or position[0] > frame_size or 0 > position[1] or position[1]> frame_size:
            print(f"position[0]: {position[0]}, position[1]: {position[1]}")
            import ipdb; ipdb.set_trace()
        positions.append(position)
        last_position = np.array([x, y])

    positions = np.array(positions)
    positions = positions.reshape((positions.shape[0], positions.shape[1], 1))

    if nr_of_skipped_gt > 0:
        print(f"Number of skipped ground truth data points for {hdf5file}: {nr_of_skipped_gt} (total data points: {len(gt_data)})")

    ts_index = 0
    frame = np.zeros((frame_size, frame_size))
    last_position = gt_data[0][0:2]
    for x, y, p, ts in zip(h5_data['event_x'], h5_data['event_y'], h5_data['event_p'], h5_data['event_t']):
        if ts > gt_data[ts_index][2]:
            if debug: 
                cv2.imshow("image", frame)
                cv2.waitKey()  
            frames.append(frame)
            frame = np.zeros((frame_size, frame_size))
            if ts_index+1 >= len(gt_data):
                break
            ts_index += 1

        # Make sure to only consider events within the accumulation time
        if (gt_data[ts_index][2] - acc_time) <= ts <= gt_data[ts_index][2]:
            # if (round(gt_data[ts_index][0]) - frame_size/2) <= x < (round(gt_data[ts_index][0]) + frame_size/2) and\
            #        (round(gt_data[ts_index][1]) - frame_size/2) <= y < (round(gt_data[ts_index][1]) + frame_size/2):
            if (round(last_position[0]) - frame_size/2) <= x < (round(last_position[0]) + frame_size/2) and\
                   (round(last_position[1]) - frame_size/2) <= y < (round(last_position[1]) + frame_size/2):
                x_index = int(x - round(last_position[0]) + frame_size/2)
                y_index = int(y - round(last_position[1]) + frame_size/2)

                frame[x_index, y_index] = 1

        last_position = gt_data[ts_index][0:2]

    return np.array(frames), positions

def stack_data(dataset):
    data = dataset[0]
    for i in range(1, len(dataset)):
        data = np.vstack((data, dataset[i]))

    return data

def store_data(data_folder):
    hdf5_files = os.listdir(data_folder)
    dataset_frames = []
    dataset_positions = []
    nr_data_points = 0
    for hdf5_file in hdf5_files:
        frames, positions = prepare_data(data_folder, hdf5_file, acc_time=1000, frame_size=64, debug=False)
        dataset_frames.append(frames)
        dataset_positions.append(positions)
        nr_data_points += positions.shape[0] 
        # print(f"Total number of data points: {nr_data_points}")
        # print(f"{positions.shape[0]} data points form {data_folder + hdf5_file}")
    
    dataset_frames = stack_data(dataset_frames)
    dataset_positions = stack_data(dataset_positions)

    with open(data_folder + 'dataset_frames.npy', 'wb') as file:
        np.save(file, dataset_frames)

    with open(data_folder + 'dataset_positions.npy', 'wb') as file:
        np.save(file, dataset_positions)

def load_data(data_folder):
    with open(data_folder + 'dataset_frames.npy', 'rb') as file:
        dataset_frames = np.load(file)

    with open(data_folder + 'dataset_positions.npy', 'rb') as file:
        dataset_positions = np.load(file)

    return dataset_frames, dataset_positions
