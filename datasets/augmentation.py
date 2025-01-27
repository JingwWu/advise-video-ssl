import torch

import random
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image, ImageEnhance, ImageOps

import math


class RandomGrayscale(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.grayscale(img).convert('RGB')
        else:
            return img


class RandomColorJitter(object):
    def __init__(self, p, b, c, s, h):
        self.p = p
        self.b = b
        self.c = c
        self.s = s
        self.h = h

    def __call__(self, img):
        b_factor = random.uniform(max(0, 1.0 - self.b), 1.0 + self.b)
        c_factor = random.uniform(max(0, 1.0 - self.c), 1.0 + self.c)
        s_factor = random.uniform(max(0, 1.0 - self.s), 1.0 + self.s)
        h_factor = random.uniform(0 - self.h, self.h)

        fn_idx = [0, 1, 2, 3]
        random.shuffle(fn_idx)

        if random.random() < self.p:
            for fn_id in fn_idx:
                if fn_id == 0:
                    img = ImageEnhance.Brightness(img).enhance(b_factor)
                if fn_id == 1:
                    img = ImageEnhance.Contrast(img).enhance(c_factor)
                if fn_id == 2:
                    img = ImageEnhance.Color(img).enhance(s_factor)
                if fn_id == 3: # adjust hue
                    img_hsv = np.array(img.convert('HSV'))
                    img_hsv[:, :, 0] = img_hsv[:, :, 0] + h_factor
                    img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
            return img
        else:
            return img


class Resize(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, img):
        return img.resize((self.w, self.h), resample=Image.BILINEAR)


class RandomCrop(object):
    def __init__(self, k):
        self.k = k

    def __call__(self, img):
        self.w, self.h = img.width, img.height
        top, left = 0, 0
        if self.h != self.k:
            top = random.randint(1, self.h - self.k) - 1
        if self.w != self.k:
            left = random.randint(1, self.w - self.k) - 1
        bottom, right = (top + self.k), (left + self.k)
        return img.crop((left, top, right, bottom))


class RandomCropTensor:
    def __init__(self, k):
        self.k = k

    def __call__(self, img):
        h, w = img.shape[2], img.shape[3]
        top, left = random.randint(1, h - self.k) - 1, random.randint(1, w - self.k) - 1
        return F.crop(img, top, left, self.k, self.k)


_pil_interpolation_to_str = {
    Image.NEAREST: "PIL.Image.NEAREST",
    Image.BILINEAR: "PIL.Image.BILINEAR",
    Image.BICUBIC: "PIL.Image.BICUBIC",
    Image.LANCZOS: "PIL.Image.LANCZOS",
    Image.HAMMING: "PIL.Image.HAMMING",
    Image.BOX: "PIL.Image.BOX",
}


_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def _pil_interp(method):
    if method == "bicubic":
        return Image.BICUBIC
    elif method == "lanczos":
        return Image.LANCZOS
    elif method == "hamming":
        return Image.HAMMING
    else:
        return Image.BILINEAR


_pil_interp_to_tv_enum = {
    Image.NEAREST: InterpolationMode.NEAREST,
    Image.LANCZOS: InterpolationMode.LANCZOS,
    Image.BILINEAR: InterpolationMode.BILINEAR,
    Image.BICUBIC: InterpolationMode.BICUBIC,
    Image.HAMMING: InterpolationMode.HAMMING,
    Image.BOX: InterpolationMode.BOX,
}


class ClipRandomResizedCrop:
    """Crop the given tensor clip to random size and aspect ratio with random interpolation.
    A jitter operation is applied to the cropping location, with arg: `jitter` to control.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        jitter: (float)
        shift: (float)
        Zoom: (float)
        brightness: (float)
        saturation: (float)
        white_blan: (float)
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        jitter=0.,
        shift=0.,
        zoom=0.,
        brightness=0.,
        saturation=0.,
        white_blan=0.,
        interpolation="bilinear",
    ):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            print("range should be of kind (min, max)")

        if interpolation == "random":
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)

        self.scale = scale
        self.ratio = ratio

        self.jitter = jitter
        self.shift = shift
        self.zoom = zoom

        self.brightness = brightness
        self.saturation = saturation
        self.white_blan = white_blan

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (tensor): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.shape[0] * img.shape[1]

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if h <= img.shape[0] and w <= img.shape[1]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.shape[1] / img.shape[0]
        if in_ratio < min(ratio):
            w = img.shape[1]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.shape[0]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.shape[1]
            h = img.shape[0]
        i = (img.shape[0] - h) // 2
        j = (img.shape[1] - w) // 2
        return i, j, h, w

    def check_interpolation(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        if isinstance(interpolation, int):
            interpolation = _pil_interp_to_tv_enum[interpolation]
        return interpolation

    def apply_shake(self, clip):
        interpolation = self.check_interpolation()
        i, j, h, w = self.get_params(clip[0, 0, ::], self.scale, self.ratio)

        mg_h = int(round(h * self.jitter)) // 2
        mg_w = int(round(w * self.jitter)) // 2

        frames = [clip[idx] for idx in range(clip.shape[0])]
        res = []
        for frame in frames:
            for _ in range(10):
                jitter_i = random.randint(i - mg_h, i + mg_h)
                jitter_j = random.randint(j - mg_w, j + mg_w)

                if (jitter_i + h <= clip.shape[2]) or \
                (jitter_j + w <= clip.shape[3]):
                    break
            else:
                jitter_i = i
                jitter_j = j
            frame = F.resized_crop(
                frame, jitter_i, jitter_j, h, w, self.size, interpolation
            )
            res.append(frame)

        res = torch.stack(res)
        return res

    def apply_shift(self, clip):
        interpolation = self.check_interpolation()
        si, sj, h, w = self.get_params(clip[0, 0, ::], self.scale, self.ratio)
        ei, ej, _, _ = self.get_params(clip[0, 0, ::], self.scale, self.ratio)

        # limit the max shift
        ei = int(si + (ei - si) * self.shift)
        ej = int(sj + (ej - sj) * self.shift)

        num_frame = clip.shape[0]
        i_list = np.linspace(si, ei, num=num_frame, endpoint=True, dtype=np.int32).tolist()
        j_list = np.linspace(sj, ej, num=num_frame, endpoint=True, dtype=np.int32).tolist()

        frames = [clip[idx] for idx in range(num_frame)]
        res = []
        for frame, i, j in zip(frames, i_list, j_list):
            frame = F.resized_crop(frame, i, j, h, w, self.size, interpolation)
            res.append(frame)

        res = torch.stack(res)
        return res

    def apply_zoom(self, clip):
        interpolation = self.check_interpolation()
        si, sj, sh, sw = self.get_params(clip[0, 0, ::], self.scale, self.ratio)

        eh = clip.shape[2]
        ew = sw * eh // sh

        ei = si + (sh - eh) // 2
        ej = sj + (sw - ew) // 2

        num_frame = clip.shape[0]
        i_list = np.linspace(si, ei, num=num_frame, endpoint=True, dtype=np.int32).tolist()
        j_list = np.linspace(sj, ej, num=num_frame, endpoint=True, dtype=np.int32).tolist()
        h_list = np.linspace(sh, eh, num=num_frame, endpoint=True, dtype=np.int32).tolist()
        w_list = np.linspace(sw, ew, num=num_frame, endpoint=True, dtype=np.int32).tolist()

        if random.random() < 0.5:
            i_list = i_list[::-1]
            j_list = j_list[::-1]
            h_list = h_list[::-1]
            w_list = w_list[::-1]

        frames = [clip[idx] for idx in range(num_frame)]
        res = []
        for frame, i, j, h, w in zip(frames, i_list, j_list, h_list, w_list):
            frame = F.resized_crop(frame, i, j, h, w, self.size, interpolation)
            res.append(frame)

        res = torch.stack(res)
        return res

    def change_brightness(self, clip):
        sb = random.uniform(1 - self.brightness, 1 + self.brightness)
        eb = random.uniform(1 - self.brightness, 1 + self.brightness)

        num_frame = clip.shape[0]
        b_list = np.linspace(sb, eb, num=num_frame, endpoint=True).tolist()

        frames = [clip[idx] for idx in range(num_frame)]
        res = []
        for frame, bri in zip(frames, b_list):
            frame = F.adjust_brightness(frame, bri)
            res.append(frame)

        res = torch.stack(res)
        return res

    def change_saturation(self, clip):
        ss = random.uniform(1 - self.saturation, 1 + self.saturation)
        es = random.uniform(1 - self.saturation, 1 + self.saturation)

        num_frame = clip.shape[0]
        s_list = np.linspace(ss, es, num=num_frame, endpoint=True).tolist()

        frames = [clip[idx] for idx in range(num_frame)]
        res = []
        for frame, sat in zip(frames, s_list):
            frame = F.adjust_saturation(frame, sat)
            res.append(frame)

        res = torch.stack(res)
        return res

    def change_white_balance(self, clip):
        sw = random.uniform(1 - self.white_blan, 1 + self.white_blan)
        ew = random.uniform(1 - self.white_blan, 1 + self.white_blan)

        num_frame = clip.shape[0]
        w_list = np.linspace(sw, ew, num=num_frame, endpoint=True).tolist()

        color_ch = random.randint(0, 2)
        frames = [clip[idx] for idx in range(num_frame)]
        res = []
        for frame, wb in zip(frames, w_list):
            frame[color_ch, :, :] = torch.clamp(frame[color_ch, :, :] * wb, min=0, max=1)
            res.append(frame)

        res = torch.stack(res)
        return res

    def __call__(self, clip):
        """
        Args:
            clip (tensor): Clip to be cropped and resized.
                shape: (#frame, #color-ch, H, W)
        Returns:
            tensor: Randomly cropped and resized clip.
        """
        choices_weight = [0 if x == 0. else 1 for x in [self.jitter, self.shift, self.zoom]]
        if sum(choices_weight) != 0:
            cam_dyn_fn = random.choices(
                    [self.apply_shake, self.apply_shift, self.apply_zoom],
                    weights = choices_weight, k = 1)[0]
            res = cam_dyn_fn(clip)
        else:
            interpolation = self.check_interpolation()
            i, j, h, w = self.get_params(clip[0, 0, ::], self.scale, self.ratio)

            frames = [clip[idx] for idx in range(clip.shape[0])]
            res = []
            for frame in frames:
                res.append(F.resized_crop(frame, i, j, h, w, self.size, interpolation))

            res = torch.stack(res)

        res = res.float() / 255.

        choices_weight = [0 if x == 0. else 1 for x in [self.brightness, self.saturation, self.white_blan]]
        if sum(choices_weight) != 0:
            color_dyn_fn = random.choices(
                    [self.change_brightness, self.change_saturation, self.change_white_balance],
                    weights = choices_weight, k = 1)[0]
            res = color_dyn_fn(res)

        return res


class AugsWarper:
    def __init__(self, aug_list, aug_mode='frame'):
        self.toPIL = transforms.ToPILImage()
        self.toTensor = transforms.ToTensor()
        self.norm = transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                )
        self.aug_list = aug_list
        self.aug_mode = aug_mode

        # Compose augs
        if self.aug_mode == 'frame':
            self.aug = transforms.Compose([
                self.toPIL,
                *self.aug_list,
                self.toTensor,
                self.norm,
            ])
        elif self.aug_mode == 'clip':
            self.aug = transforms.Compose([
                *self.aug_list,
                self.norm,
            ])
        elif self.aug_mode == 'none':
            self.aug = self.aug_list
        else:
            raise NotImplementedError

    def __call__(self, img):
        return self.aug(img)


if __name__ == '__main__':
    aug_list = [
            Resize(w=171, h=128),
            RandomCrop(k=112),
            RandomColorJitter(p=0.8, b=0.5, c=0.5, s=0.5, h=0.1),
            ]
    aug = AugsWarper(aug_list)
    import torch, decord
    decord.bridge.set_bridge('torch')
    video_path="/home/wjw/Datasets/ucf101/UCF-101/Diving/v_Diving_g05_c02.avi"
    video_reader = decord.VideoReader(str(video_path), width=-1, height=-1, num_threads=1)
    container = video_reader.get_batch(range(0, len(video_reader))).permute(0, 3, 1, 2)
    img = container[4, ::]; print(img.shape)
    print(img)
    img = aug(img)
    print(torch.mean(img), torch.std(img))
    print(img)
