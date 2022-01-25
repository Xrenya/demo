from torch.utils.tensorboard import SummaryWriter


class NoSummaryWriter:
    def __init__(self, iterable=None):
        self._iterable = iterable

    def __getattr__(self, _):
        return self._nop

    @staticmethod
    def _nop(*args, **kwargs):
        pass

    def __iter__(self):
        self._iter = iter(self._iterable)
        return self

    def __next__(self):
        return next(self._iter)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class TensorBoard:
    def __init__(self, initialize: bool = False):
        self.initialize = initialize

    def init(self, *args, **kwargs):
        if self.initialize:
            return SummaryWriter(*args, **kwargs)
        return NoSummaryWriter()
