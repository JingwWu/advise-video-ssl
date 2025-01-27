import torch
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage, ToTensor

from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageSequenceClip
import numpy as np

import logging
logger = logging.getLogger(__name__)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


spd_label_to_text = {
    -2: "Rev 2x",
    -1: "Rev 1x",
    1: "1x",
    2: "2x",
    4: "4x",
    8: "8x",
}

cam_label_to_text = {
    0: "Shake",
    1: "Shift",
    2: "Zoom",
}

clr_label_to_text = {
    0: "Bri",
    1: "Sat",
    2: "Wb",
}


def clipsAddText(inp, text):
    """(#color-ch, #frame, H, W)"""
    to_pil_image = ToPILImage()
    to_tensor = ToTensor()

    dat = inp.permute(1, 0, 2, 3)
    dat_list = [to_pil_image(dat[idx]) for idx in range(dat.shape[0])]

    res = []
    for pil_dat in dat_list:
        draw = ImageDraw.Draw(pil_dat)

        draw.text((10, 5), text, fill=(255, 255, 255))
        res.append(to_tensor(pil_dat))

    res = torch.stack(res).permute(1, 0, 2, 3)
    return res


def denormalization(inp):
    if inp.ndim == 6:
        _mean = torch.tensor(mean).reshape(1, 1, 3, 1, 1, 1)
        _std = torch.tensor(mean).reshape(1, 1, 3, 1, 1, 1)
    elif inp.ndim == 5:
        _mean = torch.tensor(mean).reshape(1, 3, 1, 1, 1)
        _std = torch.tensor(mean).reshape(1, 3, 1, 1, 1)
    else:
        raise NotImplementedError

    denormed = inp * _std + _mean
    return denormed


def clips2images(inp, filename='output'):
    """(#video, #clip, #color-ch, #frame, H, W)"""
    dat = denormalization(inp)

    dat = dat.flatten(0, 1).permute(1, 2, 3, 0, 4)
    dat = dat.flatten(1, 2).flatten(2, 3)

    save_image(dat, f"{filename}.jpg")


def clips2gifs(inp, label_list=None, filename='output'):
    """
    inp: (#sample, #color-ch, #frame, H, W)
    label_list: (#sample)
    Usage:
        from utils.visualization import clips2gifs
        label_list = torch.stack(infos['spd_label']).permute(1, 0).flatten(0, 1)
        label_list = [cfg.SSL.RANGE[label_list[idx]] * cfg.DATA.STRIDE for idx in range(len(label_list))]
        clips2gifs(data.flatten(0, 1), label_list, f"output_rk{cfg.RANK}"); exit(0)
    """
    dat = denormalization(inp)

    _min, _max = torch.min(dat), torch.max(dat)
    dat = (dat - _min) / (_max - _min)

    dat_list = [dat[idx] for idx in range(dat.shape[0])]
    dat = []
    for idx in range(len(dat_list)):
        clip = dat_list[idx]
        if label_list is not None:
            label = spd_label_to_text[label_list[idx]]
            clip = clipsAddText(clip, label)
        dat.append(clip)
    dat = torch.stack(dat)

    dat = dat.permute(2, 3, 0, 4, 1).flatten(2, 3)
    frames = [(dat[idx].numpy() * 255).astype(np.uint8) for idx in range(dat.shape[0])]

    clip = ImageSequenceClip(frames, fps=len(frames))
    clip.write_gif(f"{filename}.gif", fps=len(frames)*4)


def get_feat(model, loader, load_path, rank=0):
    from megfile import smart_open, smart_exists
    assert load_path is not None
    mdl_path = load_path
    if mdl_path.startswith("s3"):
        assert smart_exists(mdl_path), f"File does not exist at path: {mdl_path}"
        with smart_open(mdl_path, mode='rb') as f:
            ckpt = torch.load(f, map_location='cpu')
    else:
        assert os.path.exists(mdl_path), f"File does not exist at path: {mdl_path}"
        ckpt = torch.load(mdl_path, map_location='cpu')
    logger.info(f"Loading checkpoint from: {mdl_path}")

    model_state = ckpt['model_state']
    model.module.load_state_dict(model_state)
    model.eval()

    results = []
    for it, (data, infos) in enumerate(loader):
        nv, nc, ch, nf, H, W = data.shape
        input_tensor = [data]
        cls_id = infos['cls_id'][0]

        features_blobs = []
        def hook_feature(module, inp, out):
            features_blobs.append(inp[0].data)

        model.module.head_cls.projection[3].register_forward_hook(hook_feature)
        outputs = model(input_tensor)

        feats = features_blobs[0]
        feat_list = [feats[idx] for idx in range(nv)]
        for idx in range(nv):
            data = (feat_list[idx].cpu().float(), cls_id[idx].item())
            results.append(data)
        if rank == 0:
            print(f"Process: [{(it + 1):04d}/{len(loader):04d}]: len: {len(results)}")
    torch.save(results, f'OnlyIFM_feats_mlp_rk{rank}.pt')


def cam(model, loader, load_path, rank=0):
    import cv2
    from megfile import smart_open, smart_exists
    assert load_path is not None
    mdl_path = load_path
    if mdl_path.startswith("s3"):
        assert smart_exists(mdl_path), f"File does not exist at path: {mdl_path}"
        with smart_open(mdl_path, mode='rb') as f:
            ckpt = torch.load(f, map_location='cpu')
    else:
        assert os.path.exists(mdl_path), f"File does not exist at path: {mdl_path}"
        ckpt = torch.load(mdl_path, map_location='cpu')
    logger.info(f"Loading checkpoint from: {mdl_path}")

    model_state = ckpt['model_state']
    model.module.load_state_dict(model_state)
    model.eval()

    # for key in model.state_dict().keys():
    #     print(key)
    # exit(0)

    X_1 = model.module.head_cls.projection[0].weight
    X_2 = model.module.head_cls.projection[3].weight
    Y = model.module.head_cls.projection[3].bias
    X = torch.mm(X_1.t(), X_2.t())

    for it, (data, infos) in enumerate(loader):
        nv, nc, ch, nf, H, W = data.shape
        input_tensor = [data]

        features_blobs = []
        def hook_feature(module, inp, out):
            features_blobs.append(out.data)

        # model.module.temporal_encoder.pathway0_res0.branch2.c_bn.register_forward_hook(hook_feature)
        model.module.spatial_encoder.ln_post.register_forward_hook(hook_feature)
        outputs = model(input_tensor)

        imgs = data.flatten(0, 1).permute(0, 2, 3, 4, 1)
        feats = features_blobs[0].reshape(nv, nf, 257, 1024)[:, :, 1:, :].reshape(4, 16, 16, 16, 1024).type(torch.float)
        # feats = features_blobs[0].permute(0, 2, 1, 3, 4)
        feats = feats.permute(0, 1, 4, 2, 3)
        for v_idx in range(nv):
            cls_id = infos['cls_id'][0][v_idx].item()
            X_vid, Y_vid = X[:, cls_id], Y[cls_id]
            heat_map_list = []
            ori_imgs_list = []
            for f_idx in range(nf):
                img, feat = imgs[v_idx, f_idx], feats[v_idx, f_idx]
                feat = feat.mean(0)
                # feat = torch.einsum('dij, d -> ij', feat, X_vid) + Y_vid

                img = img.cpu().detach().numpy()
                img = img - np.min(img)
                img = img / np.max(img)
                img = np.uint8(255 * img)
                ori_imgs_list.append(img)

                feat = feat.cpu().detach().numpy()
                feat = feat - np.min(feat)
                # feat = (1.0 - feat / np.max(feat))
                feat = feat / np.max(feat)
                feat = cv2.resize(feat, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
                feat = np.uint8(255 * feat)
                feat = cv2.applyColorMap(feat, cv2.COLORMAP_JET)

                heat_map = np.uint8(0.6 * img + 0.4 * feat)
                heat_map_list.append(heat_map)

            heat_map = np.concatenate(heat_map_list, axis=1)
            ori_imgs = np.concatenate(ori_imgs_list, axis=1)
            heat_map = Image.fromarray(heat_map)
            ori_imgs = Image.fromarray(ori_imgs)

            h_id = it * nv + v_idx
            ori_imgs.save(f'/data/visual/cam/ori_imgs/ori_imgs_id{h_id:04d}_rk{rank}.jpg')
        print(it)
        if it >= 25: exit(0)


