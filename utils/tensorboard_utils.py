from queue import Queue
from threading import Thread
from tensorboardX import SummaryWriter
from torch import Tensor


class TensorboardLogger(object):
    def __init__(self, log_dir, is_master, prefix='pt'):
        self.is_master = is_master
        self.writer = SummaryWriter(log_dir=log_dir) if self.is_master else None
        self.step = 0
        self.prefix = prefix
        self.log_freq = 1

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def get_loggable(self, step=None):
        if step is None:  # iter wise
            step = self.step
            loggable = step % self.log_freq == 0
        else:  # epoch wise
            loggable = True
        return step, (loggable and self.is_master)

    def update(self, head='scalar', step=None, **kwargs):
        step, loggable = self.get_loggable(step)
        if loggable:
            head = f'{self.prefix}_{head}'
            for k, v in kwargs.items():
                if v is None:
                    continue
                if isinstance(v, Tensor):
                    v = v.item()
                assert isinstance(v, (float, int)), (type(v), v)
                self.writer.add_scalar(head + "/" + k, v, step)

    def log_distribution(self, tag, values, step=None):
        step, loggable = self.get_loggable(step)
        if loggable:
            if not isinstance(values, Tensor):
                values = tensor(values)
            self.writer.add_histogram(tag=tag, values=values, global_step=step)

    def log_image(self, tag, img, step=None, dataformats='NCHW'):
        step, loggable = self.get_loggable(step)
        if loggable:
            # img = img.cpu().numpy()
            self.writer.add_image(tag, img, step, dataformats=dataformats)

    def flush(self):
        if self.is_master: self.writer.flush()

    def close(self):
        if self.is_master: self.writer.close()



class PlotterThread():
    '''log tensorboard data in a background thread to save time'''
    def __init__(self, writer):
        self.writer = writer
        self.task_queue = Queue(maxsize=0)
        worker = Thread(target=self.do_work, args=(self.task_queue,))
        worker.setDaemon(True)
        worker.start()

    def do_work(self, q):
        while True:
            content = q.get()
            if content[-1] == 'image':
                self.writer.add_image(*content[:-1])
            elif content[-1] == 'scalar':
                self.writer.add_scalar(*content[:-1])
            else:
                raise ValueError
            q.task_done()

    def add_data(self, name, value, step, data_type='scalar'):
        self.task_queue.put([name, value, step, data_type])

    def __len__(self):
        return self.task_queue.qsize()
