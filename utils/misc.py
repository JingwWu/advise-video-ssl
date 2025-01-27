import os

import torch
import torchvision.io as io

from megfile import smart_open, smart_exists
from megfile.s3 import s3_listdir

import random
import numpy as np
import yaml
import time
import types
import subprocess
import logging
logger = logging.getLogger(__name__)

from collections import deque

from torch.utils.data.dataloader import DataLoader
from utils.tensorboard_utils import TensorboardLogger
from megfile import smart_open, smart_exists


class Timer:
    def __init__(self, *args):
        self.names = args
        self.times = {}
        for n in self.names:
            meter = AverageMeter(n, ':.4f')
            setattr(self, n, meter)

    def reset_meter(self, name):
        assert name in self.names
        meter = AverageMeter(name, ':.4f')
        setattr(self, name, meter)

    def tic(self, name):
        self.times[name] = time.time()

    def toc(self, name):
        old_t = self.times[name]
        interval = time.time() - old_t
        meter = getattr(self, name)
        meter.update(interval)

    def get_timer(self, name):
        return getattr(self, name)


class TopKAccuracyCalculator:
    def __init__(self, k):
        self.k = k

    def __call__(self, logits, labels):
        _, predicted = torch.topk(logits, self.k, dim=1)
        correct = torch.sum(predicted == labels.view(-1, 1), dim=1).float()
        accuracy = correct.mean()
        return accuracy


def model_init(cfg, model):
    mdl_path = cfg.RESUME
    assert mdl_path is not None
    if mdl_path.startswith("s3"):
        smart_exists(mdl_path)
        logger.info(f"Loading checkpoint from: {mdl_path}")
        with smart_open(mdl_path, mode='rb') as f:
            ckpt = torch.load(f, map_location='cpu', weights_only=True)
    else:
        assert os.path.exists(mdl_path)
        logger.info(f"Loading checkpoint from: {mdl_path}")
        ckpt = torch.load(mdl_path, map_location='cpu', weights_only=True)

    summary, model_state = ckpt['summary'], ckpt['model_state']
    logger.info(f"ckpt summary: {summary}")

    # load model state dict
    try:
        model.module.load_state_dict(model_state, strict=True)
        logger.info("Model Params Matched!")
    except:
        raise NotImplementedError


def training_resume(cfg, _tr, _ckpt, model, optimizer):
    if cfg.TRAIN.RESUME is not None:
        # resume from specific model
        mdl_path = cfg.TRAIN.RESUME
        if mdl_path.startswith("s3"):
            assert smart_exists(mdl_path), f"File does not exist at path: {mdl_path}"
            with smart_open(mdl_path, mode='rb') as f:
                ckpt = torch.load(f, map_location='cpu', weights_only=True)
        else:
            assert os.path.exists(mdl_path), f"File does not exist at path: {mdl_path}"
            ckpt = torch.load(mdl_path, map_location='cpu', weights_only=True)
        logger.info(f"Loading checkpoint from: {mdl_path}")
    else:
        # AUTO_RESUME Activated
        mdl_path = _ckpt.get_last_checkpoint()
        if mdl_path.startswith("s3"):
            with smart_open(mdl_path, mode='rb') as f:
                ckpt = torch.load(f, map_location='cpu', weights_only=True)
        else:
            ckpt = torch.load(mdl_path, map_location='cpu', weights_only=True)
        logger.info(f"Auto resumed from last checkpoint: {mdl_path}")

    # try loading summary
    summary = None
    try:
        summary, model_state, opt_state = ckpt['summary'], ckpt['model_state'], ckpt['opt_state']
        logger.info("ckpt summary: {}".format(summary))
    except KeyError:
        model_state = ckpt['model_state']
        logger.info("No ckpt summary found, just loading params.")

    new_model_state, skipped_keys = {}, []
    for name, param in model_state.items():

        if cfg.MODEL.MODEL_NAME == 'ResNet':
            if 'hist' in name:
                skipped_keys.append(name)
                continue
            if 'head' in name:
                skipped_keys.append(name)
                continue
            name = name.replace('backbone.', '')
        if cfg.MODEL.MODEL_NAME == 'TemporalModel':
            if 'hist' in name:
                skipped_keys.append(name)
                continue
            if 'head' in name:
                skipped_keys.append(name)
                continue

        new_model_state[name] = param
    model_state = new_model_state
    logger.info(f"\n\033[33mskipped_keys:\033[0m\n{skipped_keys}")

    # load model state dict
    if cfg.MODEL.MODEL_NAME == 'ResNet':
        _model = model.module
    if cfg.MODEL.MODEL_NAME == 'TemporalModel':
        _model = model.module

    try:
        _model.load_state_dict(model_state, strict=True)
        logger.info("Model Params Matched!")
    except RuntimeError as e:
        m_keys, u_keys = _model.load_state_dict(model_state, strict=False)
        loginfo = f"Params Mismatch Encountered!\n\033[33mmissing_keys:\033[0m\n{m_keys}\n\033[33munexpected_keys:\033[0m\n{u_keys}"
        print(loginfo); logger.info(loginfo)

    # load optimizer state dict
    if cfg.TASK != "action_recog":
        try:
            optimizer.load_state_dict(opt_state)
        except:
            pass

    # reset training infos
    if cfg.TASK != 'action_recog' and summary is not None:
        cfg.SOLVER.START_EPOCH = summary['epochs']
        _tr['curr_iters'] = summary['iteration']


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)


def set_seed_strict(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


### from: https://github.com/pytorch/pytorch/issues/15849#issuecomment-518126031
class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


# https://github.com/pytorch/pytorch/issues/15849#issuecomment-573921048
class FastDataLoader(DataLoader):
    '''for reusing cpu workers, to save time'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        # self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class checkpoint_manager:
    def __init__(self, cfg, criteria):
        self.file_path = cfg.MODELDATA
        self.ckpt_path = os.path.join(self.file_path, 'checkpoints')
        self.rank = cfg.RANK
        self.freq = cfg.TRAIN.SAVE_FREQ
        self.task = cfg.TASK
        self.criteria = criteria

        if self.file_path.startswith('s3'):
            self.protocol = 's3'
        else:
            self.protocol = 'local'

        if self.rank != 0:
            return

        if self.protocol == 'local':
            try:
                os.makedirs(self.ckpt_path, mode=0o775, exist_ok=False)
            except FileExistsError as e:
                logger.warning(e)

        logger.info("Checkpoints location: {}".format(self.ckpt_path))

    def has_checkpoint(self):
        if self.protocol == 's3':
            return smart_exists(self.ckpt_path)
        if self.protocol == 'local':
            return len(os.listdir(self.ckpt_path)) != 0
        raise NotImplementedError

    def get_last_checkpoint(self):
        max_epoch = -1
        latest_checkpoint = None

        if self.protocol == 'local':
            file_list = os.listdir(self.ckpt_path)
        elif self.protocol == 's3':
            file_list = s3_listdir(self.ckpt_path)

        for filename in file_list:
            if filename.startswith("Model_Epoch_") and filename.endswith(".pth"):
                epoch_str = filename.split("_")[-1].split(".")[0]
                epoch = int(epoch_str)
                if epoch > max_epoch:
                    max_epoch = epoch
                    latest_checkpoint = filename

        if latest_checkpoint is not None:
            return os.path.join(self.ckpt_path, latest_checkpoint)
        else:
            return os.path.join(self.ckpt_path, 'Model_Best.pth')

    def __call__(self, _tr, model, opt):
        if self.rank != 0:
            return

        curr_ep = _tr['curr_ep']
        summary = _tr['summary']
        if (curr_ep + 1) % self.freq == 0:
            self.save_ckpt(summary, model, opt)

        if not hasattr(self, 'best_result') \
        or self.criteria(summary, self.task)['value'] < self.best_result:
            self.best_result = self.criteria(summary, self.task)['value']
            _tr['best_infos'] = self.save_best(summary, model, opt)
            logger.info("Criteria: {}".format(self.criteria(summary, self.task)))

    def save_best(self, summary, model, opt):
        # save best
        file_name = "Model_Best.pth"
        save_path = os.path.join(self.ckpt_path, file_name)

        if hasattr(model, 'module'):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        dict_to_save = {
                'summary': summary,
                'model_state': model_state,
                'opt_state': opt.state_dict()
                }

        if self.protocol == 's3':
            with smart_open(save_path, "wb") as f:
                torch.save(dict_to_save, f)
        elif self.protocol == 'local':
            torch.save(dict_to_save, save_path)
        else:
            raise NotImplementedError

        logger.info("New **BEST** file: {}".format(save_path))

        # log best
        item = "best_{}".format(self.criteria(summary, self.task)['item'])
        val = self.criteria(summary, self.task)['value']
        return {item: val, 'best_ep': summary['epochs']}

    def save_ckpt(self, summary, model, opt):
        file_name = "Model_Epoch_{:04d}.pth".format(summary['epochs'])
        save_path = os.path.join(self.ckpt_path, file_name)

        if hasattr(model, 'module'):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        dict_to_save = {
                'summary': summary,
                'model_state': model_state,
                'opt_state': opt.state_dict()
                }

        if self.protocol == 's3':
            with smart_open(save_path, "wb") as f:
                torch.save(dict_to_save, f)
        elif self.protocol == 'local':
            torch.save(dict_to_save, save_path)
        else:
            raise NotImplementedError

        logger.info("New ckpt file: {}".format(save_path))


def seconds_to_ddhhmmss(seconds):
    days = seconds // (24 * 3600)
    remaining_seconds = seconds % (24 * 3600)

    hours = remaining_seconds // 3600
    remaining_seconds %= 3600

    minutes = remaining_seconds // 60
    remaining_seconds %= 60

    seconds = remaining_seconds

    result = ""
    if days > 0: result += f"{days}d "
    if hours > 0: result += f"{hours:02d}h:"
    result += f"{minutes:02d}m:"
    result += f"{seconds:02d}s"

    return result


class training_helper:
    def __init__(self, cfg, tb=None):
        self.cfg = cfg
        self.rt_dict = {}
        self.file_path = cfg.LOGDIR
        if tb:
            self.tb_lg = TensorboardLogger(
                    log_dir=self.file_path,
                    is_master=True if cfg.RANK == 0 else False,
                    prefix=tb,
                    )

    def log(self):
        # log if allowed
        if self.rt_dict['it'] == 0 \
        or (self.rt_dict['it'] + 1) % self.cfg.TRAIN.LOG_FREQ == 0 \
        or (self.rt_dict['it'] + 1) == self.rt_dict['epoch_iters']:
            infos = "Epoch: [{:04d}/{:04d}]".format(
                    self.rt_dict['curr_ep'] + 1, self.cfg.SOLVER.MAX_EPOCH) +\
                    "[{:d}/{:d}]\t".format(
                    self.rt_dict['it'] + 1, self.rt_dict['epoch_iters']
                    )
            meters_dict = self.print_meters(
                    _pre=[
                        self.rt_dict['timer'].get_timer('dt_time'),
                        self.rt_dict['timer'].get_timer('it_time'),
                        ],
                    _post=[
                        self.rt_dict['grad_norm'],
                        ]
                    )
            infos += str(meters_dict)
            infos += "cur_lr: {:.3e} max_lr: {:.3e}".format(
                    self.rt_dict['curr_lr'], self.cfg.SOLVER.BASE_LR
                    )
            infos += " ETA: {}".format(self.cal_eta())
            logger.info(infos)
            if hasattr(self, 'tb_lg'):
                self.tb_lg.flush()

    def cal_eta(self):
        meter = self.rt_dict['timer'].get_timer('it_time')
        avg_it_time = meter.avg
        iters = self.rt_dict['total_iters'] - self.rt_dict['curr_iters'] - 1
        seconds = int(iters * avg_it_time)
        return seconds_to_ddhhmmss(seconds)


    def train_start(self):
        if self.cfg.SOLVER.START_EPOCH == 0:
            logger.info('Training start from scratch.')
            self.rt_dict['curr_iters'] = -1
        else:
            if self.cfg.SOLVER.START_EPOCH < self.cfg.SOLVER.MAX_EPOCH:
                logger.info("Training resumed from epoch: {:04d}".format(self.cfg.SOLVER.START_EPOCH + 1))
            else:
                logger.info("No training performed, please check configs.")

        logger.info(self.rt_dict)

        torch.distributed.barrier()
        self.rt_dict['timer'] = Timer('ep_time', 'it_time', 'dt_time')

    def train_end(self):
        if 'best_infos' in self.rt_dict:
            logger.info("Training Ends, {}\n".format(self.rt_dict['best_infos']))
        else:
            logger.info("Training Ends.\n")

    def epoch_start(self, curr_ep):
        self.rt_dict['curr_ep'] = curr_ep

        infos = ' Epoch: [{:04d}] Starts '.format(curr_ep + 1)
        infos = '\n' + '>'*11 + infos + '<'*11
        logger.info(infos)
        self.reset_meters(self.cfg.SSL.STAT)
        self.rt_dict['grad_norm'] = AverageMeter('grad_norm', ':.4f')

        if hasattr(self, 'tb_lg'):
            self.tb_lg.set_step(curr_ep * self.rt_dict['epoch_iters'])

        self.rt_dict['timer'].reset_meter('it_time')
        self.rt_dict['timer'].reset_meter('dt_time')
        self.rt_dict['timer'].tic('ep_time')
        self.rt_dict['timer'].tic('dt_time')
        self.rt_dict['timer'].tic('it_time')

    def epoch_end(self):
        summary = {}
        summary['epochs'] = self.rt_dict['curr_ep'] + 1
        summary['iteration'] = self.rt_dict['curr_iters'] + 1
        for item in self.stat:
            meter = getattr(self, item)
            summary[meter.name] = meter.avg

        self.rt_dict['timer'].toc('ep_time')
        meter = self.rt_dict['timer'].get_timer('ep_time')
        summary['ep_time'] = meter.val
        logger.info("Epoch Summary: {}".format(summary))

        if hasattr(self, 'tb_lg'):
            self.tb_lg.update(
                    head='ep/scalar',
                    step=self.rt_dict['curr_ep'] + 1,
                    **summary
                    )
            self.tb_lg.flush()
        self.rt_dict['summary'] = summary


    def iter_step(self, it):
        self.rt_dict['timer'].toc('dt_time')
        self.rt_dict['curr_iters'] += 1
        self.rt_dict['it'] = it

    def iter_end(self):
        self.rt_dict['timer'].toc('it_time')
        self.rt_dict['timer'].tic('it_time')

        if hasattr(self, 'tb_lg'):
            self.tb_lg.update(head='it/scalar', **{
                'lr': self.rt_dict['curr_lr'],
                'grad_norm': self.rt_dict['grad_norm'].val,
                })
            self.tb_lg.set_step()

        torch.cuda.synchronize()
        self.log()

        self.rt_dict['timer'].tic('dt_time')

    def reset_meters(self, stat):
        self.stat = stat
        for item in self.stat:
            meter = AverageMeter(item, ':.4f')
            setattr(self, item, meter)
        logger.info("Reset meters: {}".format(stat))

    def update_meters(self, inp):
        for item in inp.keys():
            meter = getattr(self, item)
            if isinstance(inp[item], torch.Tensor):
                value = inp[item].item()
            elif isinstance(inp[item], float):
                value = inp[item]
            meter.update(value)
        if hasattr(self, 'tb_lg'):
            self.tb_lg.update(head='it/scalar', **inp)

    def print_meters(self, _pre=[], _post=[]):
        infos = {}
        for meter in _pre:
            item = meter.name
            infos[item] = '{:.2f}({:.2f})'.format(meter.val, meter.avg)
        for item in self.stat:
            meter = getattr(self, item)
            infos[item] = '{:.4f}({:.4f})'.format(meter.val, meter.avg)
        for meter in _post:
            item = meter.name
            infos[item] = '{:.4f}({:.4f})'.format(meter.val, meter.avg)
        results = ""
        for k, v in infos.items():
            results += "{}: {} ".format(k, v)
        return results

    def __setitem__(self, key, value):
        self.rt_dict[key] = value

    def __getitem__(self, key):
        return self.rt_dict[key]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name='null', fmt=':.4f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.local_history = deque([])
        self.local_avg = 0
        self.history = []
        self.dict = {} # save all data values here
        self.save_dict = {} # save mean and std here, for summary table

    def update(self, val, n=1, history=0, step=5):
        self.val = val
        self.sum += val * n
        self.count += n
        if n == 0: return
        self.avg = self.sum / self.count
        if history:
            self.history.append(val)
        if step > 0:
            self.local_history.append(val)
            if len(self.local_history) > step:
                self.local_history.popleft()
            self.local_avg = np.average(self.local_history)


    def dict_update(self, val, key):
        if key in self.dict.keys():
            self.dict[key].append(val)
        else:
            self.dict[key] = [val]

    def print_dict(self, title='IoU', save_data=False):
        """Print summary, clear self.dict and save mean+std in self.save_dict"""
        total = []
        for key in self.dict.keys():
            val = self.dict[key]
            avg_val = np.average(val)
            len_val = len(val)
            std_val = np.std(val)

            if key in self.save_dict.keys():
                self.save_dict[key].append([avg_val, std_val])
            else:
                self.save_dict[key] = [[avg_val, std_val]]

            print('Activity:%s, mean %s is %0.4f, std %s is %0.4f, length of data is %d' \
                % (key, title, avg_val, title, std_val, len_val))

            total.extend(val)

        self.dict = {}
        avg_total = np.average(total)
        len_total = len(total)
        std_total = np.std(total)
        print('\nOverall: mean %s is %0.4f, std %s is %0.4f, length of data is %d \n' \
            % (title, avg_total, title, std_total, len_total))

        if save_data:
            print('Save %s pickle file' % title)
            with open('img/%s.pickle' % title, 'wb') as f:
                pickle.dump(self.save_dict, f) # type: ignore

    def __len__(self):
        return self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def get_current_commit():
    try:
        result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
                )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return ''


def setup_logger(rank, log_file):
    data_fmt = '[%(asctime)s][%(levelname)s] %(message)s'
    time_fmt = '%y/%m/%d %H:%M:%S'

    if rank != 0:
        logging.basicConfig(
                level=logging.ERROR,
                format=data_fmt,
                datefmt=time_fmt,
                )

    os.makedirs(os.path.dirname(log_file), mode=0o775, exist_ok=True)
    if not os.path.exists(log_file):
        open(log_file, 'w').close()

    logging.basicConfig(
            level=logging.INFO,
            format=data_fmt,
            datefmt=time_fmt,
            filename=log_file,
            filemode='a'
            )

def plot_input_normed(
    tensor,
    bboxes=(),
    texts=(),
    path="./tmp_vis.png",
    folder_path="",
    make_grids=False,
    output_video=False,
):
    """
    Plot the input tensor with the optional bounding box and save it to disk.
    Args:
        tensor (tensor): a tensor with shape of `NxCxHxW`.
        bboxes (tuple): bounding boxes with format of [[x, y, h, w]].
        texts (tuple): a tuple of string to plot.
        path (str): path to the image to save to.
    """
    tensor = tensor.float()
    try:
        os.mkdir(folder_path)
    except Exception as e:
        pass
    tensor = convert_normalized_images(tensor)
    if output_video:
        # assert make_grids, "video needs to have make_grids on"
        assert tensor.ndim == 5
        sz = tensor.shape

        if make_grids:
            vid = tensor.reshape([sz[0], sz[1] * sz[2], sz[3], sz[4]])
            vid = make_grid(vid, padding=8, pad_value=1.0, nrow=sz[0]) # type: ignore
            vid = vid.reshape([sz[1], sz[2], vid.shape[1], vid.shape[2]])
        else:
            vid = tensor.reshape([sz[0] * sz[1], sz[2], sz[3], sz[4]])

        vid = vid.permute([0, 2, 3, 1])
        vid *= 255.0
        vid = vid.to(torch.uint8)
        fps = 30.0 * vid.shape[0] / 64.0
        io.video.write_video(path, vid, fps, video_codec="libx264")
    elif make_grids:
        if tensor.ndim > 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze()
            nrow = 1
        elif tensor.ndim == 5:
            nrow = tensor.shape[1]
            tensor = tensor.reshape(
                shape=(-1, tensor.shape[2], tensor.shape[3], tensor.shape[4])
            )
        vis2 = (
            make_grid(tensor, padding=8, pad_value=1.0, nrow=nrow) # type: ignore
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        plt.imsave(fname=path, arr=vis2, format="png") # type: ignore
    else:
        f, ax = plt.subplots( # type: ignore
            nrows=tensor.shape[0],
            ncols=tensor.shape[1],
            figsize=(10 * tensor.shape[1], 10 * tensor.shape[0]),
        )

        if tensor.shape[0] == 1:
            for i in range(tensor.shape[1]):
                ax[i].axis("off")
                ax[i].imshow(tensor[0][i].permute(1, 2, 0))
                # ax[1][0].axis('off')
                if bboxes is not None and len(bboxes) > i:
                    for box in bboxes[i]:
                        x1, y1, x2, y2 = box
                        ax[i].vlines(x1, y1, y2, colors="g", linestyles="solid")
                        ax[i].vlines(x2, y1, y2, colors="g", linestyles="solid")
                        ax[i].hlines(y1, x1, x2, colors="g", linestyles="solid")
                        ax[i].hlines(y2, x1, x2, colors="g", linestyles="solid")

            if texts is not None and len(texts) > i:
                ax[i].text(0, 0, texts[i])
        else:
            for i in range(tensor.shape[0]):
                for j in range(tensor.shape[1]):
                    ax[i][j].axis("off")
                    ax[i][j].imshow(tensor[i][j].permute(1, 2, 0))
                    # ax[1][0].axis('off')
                    if bboxes is not None and len(bboxes) > i:
                        for box in bboxes[i]:
                            x1, y1, x2, y2 = box
                            ax[i].vlines(
                                x1, y1, y2, colors="g", linestyles="solid"
                            )
                            ax[i].vlines(
                                x2, y1, y2, colors="g", linestyles="solid"
                            )
                            ax[i].hlines(
                                y1, x1, x2, colors="g", linestyles="solid"
                            )
                            ax[i].hlines(
                                y2, x1, x2, colors="g", linestyles="solid"
                            )

                    if texts is not None and len(texts) > i:
                        ax[i].text(0, 0, texts[i])
        print(f"{path}")
        f.tight_layout(pad=0.0)
        with pathmgr.open(path, "wb") as h: # type: ignore
            f.savefig(h)


def convert_normalized_images(tensor):

    tensor = tensor * 0.225
    tensor = tensor + 0.45

    tensor = tensor.clamp(min=0.0, max=1.0)

    return tensor
