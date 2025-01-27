import torch
import torch.multiprocessing as mp

import numpy as np
import random
import math

from datasets.augmentation import ClipRandomResizedCrop
import torchvision.transforms.functional as F


speed_group_0 = {0: 4, 1: 8, 2: 12, 3: 16,}

sub_clip_len = 8


def metric_shuffle(data):
    nvideo, nclip, nch, nframe, h, w = data.shape
    random_perm = torch.randperm(nvideo * nclip)
    data = data.flatten(0, 1)[random_perm].reshape(nvideo, nclip, nch, nframe, h, w)

    return data, random_perm


def gen_fg_for_one_clip(clip, group):
    """data shape
    clip: [3, 16, 224, 224]
    group: list: nframe // sub_clip_len
    """
    nch, nframe, h, w = clip.shape
    nSubClip = nframe // sub_clip_len                           # e.g. 16 // 4 = 4
    i, j, ph, pw = _get_params(clip[0, 0, ::], (0.01, 0.04), (3./4., 4./3.))
    patch = F.resized_crop(clip[:, 0, ::], 0, 0, h, w, (ph, pw))        # resize the frist frame as path

    for idx in range(nSubClip):
        interval = speed_group_0[group[idx]] * sub_clip_len
        i_stop, j_stop = None, None
        for _ in range(10):
            degree = math.radians(random.randint(0, 360))
            delta_i, delta_j = int(interval * math.sin(degree)), int(interval * math.cos(degree))
            i_stop, j_stop = i + delta_i, j + delta_j
            if i_stop in range(0, h - ph) and j_stop in range(0, w - pw):
                break
        else:
            margin_i, margin_j = i - h // 2, j - w // 2
            if margin_j == 0: margin_j += 1
            degree = math.atan(margin_i / margin_j)
            delta_i, delta_j = int(interval * math.sin(degree)), int(interval * math.cos(degree))
            i_stop, j_stop = i + delta_i, j + delta_j
            i_stop, j_stop = max(0, min(h - ph, i_stop)), max(0, min(w - pw, j_stop))

        i_list = np.linspace(i, i_stop, num=sub_clip_len, endpoint=False, dtype=np.int32).tolist()
        j_list = np.linspace(j, j_stop, num=sub_clip_len, endpoint=False, dtype=np.int32).tolist()
        for fidx, (ti, tj) in enumerate(zip(i_list, j_list)):
            clip[:, idx * sub_clip_len + fidx, ti: ti + ph, tj: tj + pw] = patch

        i, j = i_stop, j_stop

    return clip


def gen_pos_pair(data, label):
    nvideo, nclip, nch, nframe, h, w = data.shape
    result = []
    for vidx in range(nvideo):
        pos_data, label_group = data[vidx], label[vidx]
        for cidx in range(nclip):       # in one positive pair group
            clip = gen_fg_for_one_clip(pos_data[cidx], label_group)
            result.append(clip)
    result = torch.stack(result).reshape(nvideo, nclip, nch, nframe, h, w)
    return result


def gen_fg_patch(data, infos):
    nvideo, nclip, nch, nframe, h, w = data.shape
    data, random_perm = metric_shuffle(data)
    speed_label = random.choices(
        range(0, len(speed_group_0)),
        k=nvideo * nframe // sub_clip_len,
    )
    speed_label = torch.Tensor(speed_label).to(dtype=torch.int64)
    speed_label = speed_label.reshape(nvideo, nframe // sub_clip_len).tolist()

    data = gen_pos_pair(data, speed_label)

    return data, infos


interpolation = F.InterpolationMode.BILINEAR
_get_params = ClipRandomResizedCrop.get_params


def only_crop(clip, size, scale, ratio):
    i, j, h, w = _get_params(clip[0, 0, ::], scale, ratio)
    frames = [clip[idx] for idx in range(clip.shape[0])]
    res = []
    for frame in frames:
        res.append(F.resized_crop(frame, i, j, h, w, size, interpolation))

    res = torch.stack(res)
    return res


def apply_shk(clip, size, scale, ratio, shk):
    i, j, h, w = _get_params(clip[0, 0, ::], scale, ratio)

    mg_h = int(round(h * shk)) // 2
    mg_w = int(round(w * shk)) // 2

    frames = [clip[idx] for idx in range(clip.shape[0])]
    res = []
    for frame in frames:
        ri = random.randint(i - mg_h, i + mg_h)
        rj = random.randint(j - mg_w, j + mg_w)

        frame = F.resized_crop(frame, ri, rj, h, w, size, interpolation)
        res.append(frame)

    res = torch.stack(res)
    return res


def apply_sft(clip, size, scale, ratio, sft):
    si, sj, h, w = _get_params(clip[0, 0, ::], scale, ratio)
    ei, ej, _, _ = _get_params(clip[0, 0, ::], scale, ratio)

    # limit the max shift
    ei = int(si + (ei - si) * sft)
    ej = int(sj + (ej - sj) * sft)

    num_frame = clip.shape[0]
    i_list = np.linspace(si, ei, num=num_frame, endpoint=True, dtype=np.int32).tolist()
    j_list = np.linspace(sj, ej, num=num_frame, endpoint=True, dtype=np.int32).tolist()

    frames = [clip[idx] for idx in range(num_frame)]
    res = []
    for frame, i, j in zip(frames, i_list, j_list):
        frame = F.resized_crop(frame, i, j, h, w, size, interpolation)
        res.append(frame)

    res = torch.stack(res)
    return res


def apply_zm(clip, size, scale, ratio, zm):
    si, sj, sh, sw = _get_params(clip[0, 0, ::], scale, ratio)

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
        frame = F.resized_crop(frame, i, j, h, w, size, interpolation)
        res.append(frame)

    res = torch.stack(res)
    return res


def change_bri(clip, scale, ratio, bri):
    i, j, h, w = _get_params(clip[0, 0, ::], scale, ratio)
    sb = random.uniform(1 - bri, 1 + bri)
    eb = random.uniform(1 - bri, 1 + bri)

    num_frame = clip.shape[0]
    b_list = np.linspace(sb, eb, num=num_frame, endpoint=True).tolist()

    frames = [clip[idx] for idx in range(num_frame)]
    res = []
    for frame, bri in zip(frames, b_list):
        frame[:, i:i+h, j:j+w] = F.adjust_brightness(frame[:, i:i+h, j:j+w], bri)
        res.append(frame)

    res = torch.stack(res)
    return res, eb - sb


def change_sat(clip, scale, ratio, sat):
    i, j, h, w = _get_params(clip[0, 0, ::], scale, ratio)
    ss = random.uniform(1 - sat, 1 + sat)
    es = random.uniform(1 - sat, 1 + sat)

    num_frame = clip.shape[0]
    s_list = np.linspace(ss, es, num=num_frame, endpoint=True).tolist()

    frames = [clip[idx] for idx in range(num_frame)]
    res = []
    for frame, sat in zip(frames, s_list):
        frame[:, i:i+h, j:j+w] = F.adjust_saturation(frame[:, i:i+h, j:j+w], sat)
        res.append(frame)

    res = torch.stack(res)
    return res, es - ss


def change_wb(clip, scale, ratio, wb):
    i, j, h, w = _get_params(clip[0, 0, ::], scale, ratio)
    sw = random.uniform(1 - wb, 1 + wb)
    ew = random.uniform(1 - wb, 1 + wb)

    num_frame = clip.shape[0]
    w_list = np.linspace(sw, ew, num=num_frame, endpoint=True).tolist()

    clr_ch = random.randint(0, 2)
    frames = [clip[idx] for idx in range(num_frame)]
    res = []
    for frame, wb in zip(frames, w_list):
        frame[clr_ch, i:i+h, j:j+w] = torch.clamp(
                frame[clr_ch, i:i+h, j:j+w] * wb, min=0, max=1)
        res.append(frame)

    res = torch.stack(res)
    return res, ew - sw
