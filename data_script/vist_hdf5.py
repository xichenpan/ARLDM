import argparse
import json
import os

import cv2
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def main(args):
    train_data = json.load(open(os.path.join(args.sis_json_dir, 'train.story-in-sequence.json')))
    val_data = json.load(open(os.path.join(args.sis_json_dir, 'val.story-in-sequence.json')))
    test_data = json.load(open(os.path.join(args.sis_json_dir, 'test.story-in-sequence.json')))

    prefix = ["train", "val", "test"]
    whole_album = {}
    for i, data in enumerate([train_data, val_data, test_data]):
        album_mapping = {}
        for annot_new in data["annotations"]:
            annot = annot_new[0]
            assert len(annot_new) == 1
            if annot['story_id'] not in album_mapping:
                album_mapping[annot['story_id']] = {"flickr_id": [annot['photo_flickr_id']],
                                                    "sis": [annot['original_text']],
                                                    "length": 1}
            else:
                album_mapping[annot['story_id']]["flickr_id"].append(annot['photo_flickr_id'])
                album_mapping[annot['story_id']]["sis"].append(
                    annot['original_text'])
                album_mapping[annot['story_id']]["length"] += 1
        whole_album[prefix[i]] = album_mapping

    for p in prefix:
        deletables = []
        for story_id, story in whole_album[p].items():
            if story['length'] != 5:
                print("deleting {}".format(story_id))
                deletables.append(story_id)
                continue
            d = [os.path.exists(os.path.join(args.img_dir, "{}.jpg".format(_))) for _ in story["flickr_id"]]
            if sum(d) < 5:
                print("deleting {}".format(story_id))
                deletables.append(story_id)
            else:
                pass
        for i in deletables:
            del whole_album[p][i]

    train_data = json.load(open(os.path.join(args.dii_json_dir, 'train.description-in-isolation.json')))
    val_data = json.load(open(os.path.join(args.dii_json_dir, 'val.description-in-isolation.json')))
    test_data = json.load(open(os.path.join(args.dii_json_dir, 'test.description-in-isolation.json')))

    flickr_id2text = {}
    for i, data in enumerate([train_data, val_data, test_data]):
        for l in data['annotations']:
            assert len(l) == 1
            if l[0]['photo_flickr_id'] in flickr_id2text:
                flickr_id2text[l[0]['photo_flickr_id']] = \
                    max([flickr_id2text[l[0]['photo_flickr_id']], l[0]['original_text']], key=len)
            else:
                flickr_id2text[l[0]['photo_flickr_id']] = l[0]['original_text']

    for p in prefix:
        deletables = []
        for story_id, story in whole_album[p].items():
            story['dii'] = []
            for i, flickr_id in enumerate(story['flickr_id']):
                if flickr_id not in flickr_id2text:
                    print("{} not found in story {}".format(flickr_id, story_id))
                    deletables.append(story_id)
                    break
                story['dii'].append(flickr_id2text[flickr_id])
        for i in deletables:
            del whole_album[p][i]

    f = h5py.File(args.save_path, "w")
    for p in prefix:
        group = f.create_group(p)
        story_dict = whole_album[p]
        length = len(story_dict)
        images = list()
        for i in range(5):
            images.append(
                group.create_dataset('image{}'.format(i), (length,), dtype=h5py.vlen_dtype(np.dtype('uint8'))))
        sis = group.create_dataset('sis', (length,), dtype=h5py.string_dtype(encoding='utf-8'))
        dii = group.create_dataset('dii', (length,), dtype=h5py.string_dtype(encoding='utf-8'))
        for i, (story_id, story) in enumerate(tqdm(story_dict.items(), leave=True, desc="saveh5")):
            imgs = [Image.open('{}/{}.jpg'.format(args.img_dir, flickr_id)).convert('RGB') for flickr_id in
                    story['flickr_id']]
            for j, img in enumerate(imgs):
                img = np.array(img).astype(np.uint8)
                img = cv2.imencode('.png', img)[1].tobytes()
                img = np.frombuffer(img, np.uint8)
                images[j][i] = img
            sis[i] = '|'.join([t.replace('\n', '').replace('\t', '').strip() for t in story['sis']])
            txt_dii = [t.replace('\n', '').replace('\t', '').strip() for t in story['dii']]
            txt_dii = sorted(set(txt_dii), key=txt_dii.index)
            dii[i] = '|'.join(txt_dii)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for vist hdf5 file saving')
    parser.add_argument('--sis_json_dir', type=str, required=True, help='sis json file directory')
    parser.add_argument('--dii_json_dir', type=str, required=True, help='dii json file directory')
    parser.add_argument('--img_dir', type=str, required=True, help='json file directory')
    parser.add_argument('--save_path', type=str, required=True, help='path to save hdf5')
    args = parser.parse_args()
    main(args)
