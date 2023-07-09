import json
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from multiprocessing import Process
import os
import argparse


def download_subprocess(dii, save_dir):
    for image in tqdm(dii):
        key, value = image.popitem()
        try:
            img_data = requests.get(value).content
            img = Image.open(BytesIO(img_data)).convert('RGB')
            h = img.size[0]
            w = img.size[1]
            if min(h, w) > 512:
                img = img.resize((int(h / (w / 512)), 512) if h > w else (512, int(w / (h / 512))))
            img.save('{}/{}.jpg'.format(save_dir, key))
        except:
            print(key, value)


def main(args):
    train_data = json.load(open(os.path.join(args.json_dir, 'train.description-in-isolation.json')))
    val_data = json.load(open(os.path.join(args.json_dir, 'val.description-in-isolation.json')))
    test_data = json.load(open(os.path.join(args.json_dir, 'test.description-in-isolation.json')))
    dii = []
    for subset in [train_data, val_data, test_data]:
        for image in subset["images"]:
            try:
                dii.append({image['id']: image['url_o']})
            except:
                dii.append({image['id']: image['url_m']})

    dii = [image for image in dii if not os.path.exists('{}/{}.jpg'.format(args.img_dir, list(image)[0]))]
    print('total images: {}'.format(len(dii)))

    def splitlist(inlist, chunksize):
        return [inlist[x:x + chunksize] for x in range(0, len(inlist), chunksize)]

    dii_splitted = splitlist(dii, int((len(dii) / args.num_process)))
    process_list = []
    for dii_sub_list in dii_splitted:
        p = Process(target=download_subprocess, args=(dii_sub_list,args.img_dir))
        process_list.append(p)
        p.Daemon = True
        p.start()
    for p in process_list:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arguments for vist images downloading')
    parser.add_argument('--json_dir', type=str, required=True, help='dii json file directory')
    parser.add_argument('--img_dir', type=str, required=True, help='images saving directory')
    parser.add_argument('--num_process', type=int, default=32)
    args = parser.parse_args()
    main(args)
