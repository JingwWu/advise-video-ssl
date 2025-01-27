import os
from megfile import smart_open, smart_exists

import decord
decord.bridge.set_bridge('torch')
import pandas as pd
import random
import json

import torch
from torch.utils.data import Dataset

def read_data(ptc, video_path):
    if ptc == 'local':
        dfmt = video_path.split('.')[-1]
        if dfmt in ["avi", "mp4"]:
            video_reader = decord.VideoReader(str(video_path), width=-1, height=-1, num_threads=1)
        elif dfmt in ["pt"]:
            video_reader = torch.load(str(video_path), map_location=torch.device('cpu'))
        else:
            raise NotImplementedError
    elif ptc == 's3':
        file_like_obj = smart_open(str(video_path), mode='rb')
        dfmt = video_path.split('.')[-1]
        if dfmt in ["avi", "mp4"]:
            video_reader = decord.VideoReader(file_like_obj, width=-1, height=-1, num_threads=1)
        elif dfmt in ["pt"]:
            video_reader = torch.load(file_like_obj, map_location=torch.device('cpu'))
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return video_reader, dfmt



class BasicDataset(Dataset):
    def __init__(self, name, data_dir, label_dir, split_name):
        self.name = name
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.split_name = split_name

        if self.data_dir.startswith('s3'):
            self.protocol = 's3'
            assert smart_exists(self.data_dir) is True
        else:
            self.protocol = 'local'

        if self.name in ['kinetics', 'ucf']:
            split_path = os.path.join(self.label_dir, self.split_name)
            self.data_list = pd.read_csv(split_path, header=None)[0]

        elif self.name == 'diving':
            split_path = os.path.join(self.label_dir, self.split_name)
            with open(split_path) as f:
                video_infos = json.load(f)
            data_list = []
            for video_info in video_infos:
                video_file = video_info['vid_name'] + '.avi'
                class_idx = video_info['label']
                data_list.append(f"{video_file} {class_idx}")
            self.data_list = data_list

        elif self.name == 'something':
            class_idx_path = os.path.join(
                self.label_dir, 'something-something-v2-labels.json'
            )
            with open(class_idx_path) as f:
                class_dict = json.load(f)
            self.class_idx_dict = class_dict

            assert self.split_name in [
                'something-something-v2-train.json',
                'something-something-v2-validation.json'
                'something-something-v2-test.json',
            ]
            split_path = os.path.join(self.label_dir, self.split_name)
            with open(split_path) as f:
                video_infos = json.load(f)
            data_list = []
            for video_info in video_infos:
                video_idx = int(video_info['id'])
                video_file, video_name = f'{video_idx}.avi', video_info['label']
                class_name = video_info['template'].replace('[', '').replace(']', '')
                class_idx = int(self.class_idx_dict[class_name])
                data_list.append(
                    f"{video_file}%{video_name}%{class_name}%{class_idx}"
                )
            self.data_list = data_list

        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.name in ['kinetics', 'ucf', 'diving']:
            video_item, video_cls = self.data_list[idx].split(' ')
            video_path = os.path.join(self.data_dir, video_item)
        elif self.name == 'something':
            video_item, video_name, class_name, video_cls = self.data_list[idx].split('%')
            video_path = os.path.join(self.data_dir, video_item)

        video_reader, dfmt = read_data(self.protocol, video_path)

        if dfmt in ["avi", "mp4"]:
            video_length = len(video_reader)
            # [T, H, W, C]
        elif dfmt in ["pt"]:
            video_length = int(video_reader['info']['length'][0])
            video_reader = video_reader['data']
        else:
            raise NotImplementedError

        if self.name in ['kinetics', 'ucf', 'diving']:
            infos = {
                'item': video_item,
                'item_id': idx,
                'cls_id': [int(video_cls)],
                'length': video_length,
                'backend': 'decord'
            }
        elif self.name == 'something':
            infos = {
                'item': video_item,
                'item_id': idx,
                'cls_id': [int(video_cls)],
                'vid_name': video_name,
                'cls_name': class_name,
                'length': video_length,
                'backend': 'decord'
            }
        return video_reader, infos


class SamplingDataset(BasicDataset):
    def __init__(self, name, data_dir, label_dir, split_name, spl_func=None):
        super().__init__(name, data_dir, label_dir, split_name)
        self.sampling = spl_func

    def __getitem__(self, idx):
        if self.sampling is not None:
            for _ in range(10):
                video_reader, infos = super().__getitem__(idx)
                video_data, infos = self.sampling(video_reader, infos)
                if video_data is not None:
                    break
                else:
                    print(f"fail to decode, or the video length cant satisfy lowest requirement, try another")
                    idx = random.randint(0, self.__len__() - 1)
            else:
                raise RuntimeError("After 10 times retried, still failed.")
            return video_data, infos

        # NOTE: only for debug
        video_reader, infos = super().__getitem__(idx)
        video_data = video_reader.get_batch(range(infos['length']))
        return video_data, infos


if __name__ == '__main__':
    vid_dataset = SamplingDataset(
            data_dir='/home/wjw/Datasets/k400/vid_resize256',
            label_dir='/home/wjw/Workspace/projects/data_list/kinetics',
            split_name='k200_train.txt')
    for data, infos in vid_dataset:
        print(data.shape, infos)

