import argparse
import json
import os
import pickle

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def main(args):
    splits = json.load(open(os.path.join(args.data_dir, 'train-val-test_split.json'), 'r'))
    train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]
    followings = pickle.load(open(os.path.join(args.data_dir, 'following_cache4.pkl'), 'rb'))
    annotations = json.load(open(os.path.join(args.data_dir, 'flintstones_annotations_v1-0.json')))
    descriptions = dict()
    for sample in annotations:
        descriptions[sample["globalID"]] = sample["description"]

    f = h5py.File(args.save_path, "w")
    for subset, ids in {'train': train_ids, 'val': val_ids, 'test': test_ids}.items():
        ids = [i for i in ids if i in followings and len(followings[i]) == 4]
        length = len(ids)

        group = f.create_group(subset)
        images = list()
        for i in range(5):
            images.append(
                group.create_dataset('image{}'.format(i), (length,), dtype=h5py.vlen_dtype(np.dtype('uint8'))))
        text = group.create_dataset('text', (length,), dtype=h5py.string_dtype(encoding='utf-8'))
        for i, item in enumerate(tqdm(ids, leave=True, desc="saveh5")):
            globalIDs = [item] + followings[item]
            txt = list()
            for j, globalID in enumerate(globalIDs):
                img = np.load(os.path.join(args.data_dir, 'video_frames_sampled', '{}.npy'.format(globalID)))
                img = np.concatenate(img, axis=0).astype(np.uint8)
                img = cv2.imencode('.png', img)[1].tobytes()
                img = np.frombuffer(img, np.uint8)
                images[j][i] = img
                txt.append(descriptions[globalID])
            text[i] = '|'.join([t.replace('\n', '').replace('\t', '').strip() for t in txt])
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for flintstones hdf5 file saving')
    parser.add_argument('--data_dir', type=str, required=True, help='flintstones data directory')
    parser.add_argument('--save_path', type=str, required=True, help='path to save hdf5')
    args = parser.parse_args()
    main(args)
