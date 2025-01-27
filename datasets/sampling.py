import torch

import numpy as np
import random

# import datasets.st_dynamics as stdyn

import logging
logger = logging.getLogger(__name__)


### Time-Stamp Generator Constructors
class SimpleUniformGenerator:
    def __init__(self, nums, length, jitter=0.0):
        self.clip_nums = nums
        self.frame_nums = length
        self.jitter = jitter

    def __call__(self, video_length):
        curr_nums = self.clip_nums
        while curr_nums:
            curr_nums -= 1
            yield self.gen_stamps(video_length)

    def gen_stamps(self, video_length):
        interval = video_length // self.frame_nums
        start_idx, end_idx = random.randint(0, interval), video_length - random.randint(1, interval)
        bound = int(interval * self.jitter)
        delta_idx = [random.randint(-bound, bound) for _ in range(self.frame_nums)]
        stamps = np.linspace(
            start_idx, end_idx,
            num=self.frame_nums,
            endpoint=True,
            dtype=np.int64
        )
        stamps = np.clip(stamps + delta_idx, 0, video_length - 1)
        return stamps

class StampsGenerator:
    def __init__(self, nums, length, stride, method):
        self.clip_nums = nums
        self.frame_nums = length
        self.stride = stride
        self.method = method

        self.clip_lens = length * stride

    def __call__(self, video_length):
        if self.method == 'random':
            curr_nums = self.clip_nums
            while curr_nums:
                curr_nums -= 1
                yield self.gen_stamps(video_length)
        else:
            raise NotImplementedError

    def gen_stamps(self, video_length):
        start_idx = random.randint(0, video_length - self.clip_lens)
        stamps = np.linspace(
            start_idx, start_idx + self.clip_lens,
            num=self.frame_nums,
            endpoint=False,
            dtype=np.int64
        )
        return stamps


class IntervalsStampsGenerator(StampsGenerator):
    def __init__(
            self,
            nums, length, stride, method,
            speed_range=None, jitter=None,
            intervals=None
        ):
        super().__init__(nums, length, stride, method)
        self.intervals = intervals
        self.speed_range = speed_range
        self.jitter = jitter

        assert speed_range is not None and jitter is not None
        assert intervals is None

    def gen_stamps(self, video_length):
        if video_length == self.clip_lens:
            start_idx = 0
        else:
            start_idx = random.randint(0, video_length - self.clip_lens)

        if self.speed_range is not None:
            return self.speed_modeling(start_idx, video_length)

        elif self.intervals is not None:
            return self.interval_modeling(start_idx, video_length)

        else:
            raise NotImplementedError

    def interval_modeling(self, start_idx, video_length):
        for _ in range(10):
            interval_list = random.choices(self.intervals, k=self.frame_nums-1)
            stamps, curr_idx = [start_idx], start_idx
            for itv in interval_list:
                curr_idx += itv * self.stride
                stamps.append(curr_idx)
            if curr_idx < video_length:
                break
        else:
            interval_list = [self.intervals[0]] * (self.frame_nums - 1)
            stamps, curr_idx = [start_idx], start_idx
            for itv in interval_list:
                curr_idx += itv * self.stride
                stamps.append(curr_idx)
            assert curr_idx < video_length
        return start_idx, stamps

    def speed_modeling(self, start_idx, video_length):
        speed_jit = 1 + random.uniform(-self.jitter, self.jitter)
        speed_rate = random.choice(self.speed_range)

        if speed_rate < 0:
            speed_rate = -speed_rate
            rev_flag = True
        else:
            rev_flag = False

        stamps = np.mod(np.linspace(
                start_idx,
                start_idx + self.clip_lens * speed_rate * speed_jit,
                num=self.frame_nums,
                endpoint=False,
                dtype=np.int64
                ), video_length).tolist()
        if rev_flag:
            stamps = stamps[::-1]

        return start_idx, stamps, speed_rate


class SelfPaddingGenerator(StampsGenerator):
    def gen_stamps(self, video_length):
        start_idx = random.randint(0, video_length - 1)
        stamps = np.mod(np.linspace(
                start_idx,
                start_idx + self.clip_lens,
                num=self.frame_nums,
                endpoint=False,
                dtype=np.int64
                ), video_length)
        return stamps


class SpeedStampsGenerator(StampsGenerator):
    def __init__(self, nums, length, stride, method, jitter):
        super().__init__(nums, length, stride, method)
        self.jitter = jitter

    def __call__(self, video_length, speed_rate):
        if self.method == 'random':
            curr_nums = self.clip_nums
            while curr_nums:
                curr_nums -= 1
                yield self.gen_stamps(video_length, speed_rate)
        else:
            raise NotImplementedError

    def gen_stamps(self, video_length, speed_rate):
        speed_jit = 1 + np.random.uniform(-self.jitter, self.jitter)
        start_idx = random.randint(0, video_length - 1)

        if speed_rate < 0:
            speed_rate = -speed_rate
            rev_flag = True
        else:
            rev_flag = False

        stamps = np.mod(np.linspace(
                start_idx,
                start_idx + self.clip_lens * speed_rate * speed_jit,
                num=self.frame_nums,
                endpoint=False,
                dtype=np.int64
                ), video_length)
        if rev_flag:
            stamps = stamps[::-1]

        return stamps


class DuplicSpeedStampsGenerator(SpeedStampsGenerator):
    def __init__(self, nums, length, stride, method, jitter, duplic):
        super().__init__(nums, length, stride, method, jitter)
        self.duplic = duplic

    def __call__(self, video_length, speed_rate):
        if self.method == 'random':
            curr_nums = self.clip_nums
            while curr_nums:
                curr_nums -= 1
                speed_jit = 1 + np.random.uniform(-self.jitter, self.jitter)
                start_idx = random.randint(0, video_length - 1)
                for _ in range(self.duplic):
                    yield self.gen_stamps(video_length, speed_rate, start_idx, speed_jit)
        elif self.method == 'uniform':
            start_list = np.linspace(
                0, video_length - self.clip_lens - 1,
                num=self.clip_nums, endpoint=True, dtype=np.int64).tolist()
            curr_nums = self.clip_nums
            while curr_nums:
                curr_nums -= 1
                speed_jit = 1 + np.random.uniform(-self.jitter, self.jitter)
                start_idx = start_list.pop(0)
                for _ in range(self.duplic):
                    yield self.gen_stamps(video_length, speed_rate, start_idx, speed_jit)
        else:
            raise NotImplementedError


    def gen_stamps(self, video_length, speed_rate, start_idx, speed_jit):
        if speed_rate < 0:
            speed_rate = -speed_rate
            rev_flag = True
        else:
            rev_flag = False

        stamps = np.mod(np.linspace(
            start_idx,
            start_idx + self.clip_lens * speed_rate * speed_jit,
            num=self.frame_nums,
            endpoint=False,
            dtype=np.int64
            ), video_length)
        if rev_flag:
            stamps = stamps[::-1]

        return stamps


class SpeedStampsValGenerator(StampsGenerator):
    def __init__(self, length, stride, jitter):
        # HACK: dummy arguments
        nums = 1
        method = 'random'
        super().__init__(nums, length, stride, method)
        self.jitter = jitter

    def __call__(self, video_length, speed_rate, num_label):
        self.clip_nums = video_length // 25 # HACK: assert frame_rate == 25
        if self.clip_nums == 0: self.clip_nums = 1
        start_points = np.linspace(
                0, video_length, num=self.clip_nums,
                endpoint=False, dtype=np.int64
                )
        start_points = np.repeat(start_points, num_label).tolist()
        curr_nums = len(start_points)
        while curr_nums:
            curr_nums -= 1
            yield self.gen_stamps(video_length, speed_rate, start_points.pop(0))

    def gen_stamps(self, video_length, speed_rate, start_idx):
        speed_jit = 1 + np.random.uniform(-self.jitter, self.jitter)

        if speed_rate < 0:
            speed_rate = -speed_rate
            rev_flag = True
        else:
            rev_flag = False

        stamps = np.mod(np.linspace(
                start_idx,
                start_idx + self.clip_lens * speed_rate * speed_jit,
                num=self.frame_nums,
                endpoint=False,
                dtype=np.int64
                ), video_length)
        if rev_flag:
            stamps = stamps[::-1]

        return stamps


### Sampling Function Constructors
class BasicSampling:
    def __init__(self, stps_gen, aug_func, aug_mode='frame'):
        self.stamps_generator = stps_gen
        self.augmentation = aug_func
        self.aug_mode = aug_mode

    def __call__(self, video_reader, infos):
        # decode all video
        # NOTE: support partial decode
        video_length = infos['length']
        container = video_reader.get_batch(range(0, video_length))
        # [T, H, W, C]

        # sample multi clips in one video
        results = []
        for time_stamps in self.stamps_generator(video_length):
            clip = self.sample_clip_from_video(container, time_stamps)
            results.append(clip)
        results = torch.stack(results).permute(0, 2, 1, 3, 4)
        # [K, C, T, H, W]
        return results, infos

    def sample_clip_from_video(self, container, time_stamps):
        assert max(time_stamps) < len(container)
        clip = [container[int(frame_idx), ::] for frame_idx in time_stamps]
        clip = torch.stack(clip).permute(0, 3, 1, 2)
        # [T, C, H, W]
        clip = self.apply_aug(clip) # apply augmentation for a clip
        # [T, C, H, W]
        return clip

    def apply_aug(self, clip):
        result = []

        if self.aug_mode == 'frame':
            # fix seed for all frames in one clip
            local_seed = torch.randint(0, 2**32, (1,)).item()
            for frame_idx in range(clip.shape[0]):
                random.seed(local_seed)
                np.random.seed(local_seed)
                torch.manual_seed(local_seed)
                frame = self.augmentation(clip[frame_idx, ::])
                result.append(frame)
            clip = torch.stack(result)

        elif self.aug_mode == 'clip':
            clip = self.augmentation(clip)

        else:
            raise NotImplementedError

        return clip


class TemporalModelingSampling(BasicSampling):
    def __init__(self, stps_gen, aug_func, aug_mode='frame'):
        super().__init__(stps_gen, aug_func, aug_mode)
        self.clip_nums = getattr(stps_gen, 'clip_nums')
        self.clip_lens = getattr(stps_gen, 'clip_lens')

    def __call__(self, video_reader, infos):
        video_length = infos['length']
        if video_length < self.clip_lens:
            return None, None

        container = video_reader.get_batch(range(0, video_length))

        clips, clips_adj, starts, stamps, speeds = [], [], [], [], []
        for idx in range(self.clip_nums):
            start_idx, time_stamps, speed_rate = next(self.stamps_generator(video_length))

            clips.append(self.sample_clip_from_video(container, time_stamps))

            starts.append(start_idx)
            stamps.append(time_stamps)
            speeds.append(speed_rate)

        clips = torch.stack(clips).permute(0, 2, 1, 3, 4)

        infos['starts'] = torch.Tensor(starts).type(torch.long)
        infos['stamps'] = torch.Tensor(stamps).type(torch.long)
        infos['speeds'] = torch.Tensor(speeds).type(torch.long)


class SpeedSampling(BasicSampling):
    "Work with SpeedStampsGenerator"
    def __init__(self, stps_gen, aug_func, speed_range, aug_mode='frame'):
        super().__init__(stps_gen, aug_func, aug_mode)
        self.speed_range = speed_range
        self.clip_nums = getattr(stps_gen, 'clip_nums')
        if hasattr(stps_gen, 'duplic'):
            self.clip_nums = self.clip_nums * getattr(stps_gen, 'duplic')

    def __call__(self, video_reader, infos):
        # decode all video
        # NOTE: support partial decode
        video_length = infos['length']
        container = video_reader.get_batch(range(0, video_length))
        # [T, H, W, C]

        # sample multi clips in one video
        results = []
        speed_rate, infos = self.gen_labels(
            nums=self.clip_nums, _range=self.speed_range, infos=infos,
        )
        for idx in range(self.clip_nums):
            time_stamps = next(self.stamps_generator(video_length, speed_rate[idx]))
            clip = self.sample_clip_from_video(container, time_stamps)
            results.append(clip)
        results = torch.stack(results).permute(0, 2, 1, 3, 4)
        # [K, C, T, H, W]
        return results, infos

    def gen_labels(self, nums, _range, infos):
        labels = [random.randint(0, len(_range) - 1) for _ in range(nums)]
        speed_rate = [_range[idx] for idx in labels]
        infos['spd_label'] = labels
        return speed_rate, infos


class SimpleUniformSampling(BasicSampling):
    "Work with SimpleUniformGenerator"
    def __init__(self, stps_gen, aug_func, aug_mode='frame', duplic=1):
        super().__init__(stps_gen, aug_func, aug_mode)
        self.clip_nums = getattr(stps_gen, 'clip_nums')
        self.clip_lens = getattr(stps_gen, 'frame_nums')
        self.duplic = duplic

    def __call__(self, video_reader, infos):
        video_length = infos['length']
        if video_length < self.clip_lens:
            return None, None

        container = video_reader.get_batch(range(0, video_length))
        # [T, H, W, C]

        # sample multi clips in one video
        results = []
        for idx in range(self.clip_nums):
            time_stamps = next(self.stamps_generator(video_length))
            for _ in range(self.duplic):
                clip = self.sample_clip_from_video(container, time_stamps)
                results.append(clip)
        results = torch.stack(results).permute(0, 2, 1, 3, 4)
        # [K, C, T, H, W]
        return results, infos


class SpeedValSampling(BasicSampling):
    "Work with SpeedStampsValGenerator"
    def __init__(self, stps_gen, aug_func, speed_range, aug_mode='frame'):
        super().__init__(stps_gen, aug_func, aug_mode)
        self.speed_range = speed_range
        # self.clip_nums = getattr(stps_gen, 'clip_nums')   No need for this

    def __call__(self, video_reader, infos):
        # decode all video
        # NOTE: support partial decode
        video_length = infos['length']
        container = video_reader.get_batch(range(0, video_length))
        # [T, H, W, C]
        self.clip_nums = video_length // 25 # HACK: assert frame_rate == 25
        if self.clip_nums == 0: self.clip_nums = 1

        # sample multi clips in one video
        results = []
        speed_rate, infos = self.gen_labels(
                nums=self.clip_nums, _range=self.speed_range, infos=infos,)
        for idx in range(len(speed_rate)):
            time_stamps = next(self.stamps_generator(
                video_length, speed_rate[idx], len(self.speed_range)))
            clip = self.sample_clip_from_video(container, time_stamps)
            results.append(clip)
        results = torch.stack(results).permute(0, 2, 1, 3, 4)
        # [K, C, T, H, W]
        return results, infos

    def gen_labels(self, nums, _range, infos):
        labels = [x for x in range(len(_range))] * nums
        speed_rate = [_range[idx] for idx in labels]
        infos['spd_label'] = labels
        return speed_rate, infos


if __name__ == '__main__':
    gen = SimpleUniformGenerator(4, 16)
    for stamps in gen(220):
        print(stamps)

