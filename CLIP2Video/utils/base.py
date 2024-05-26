from abc import ABCMeta, abstractmethod
from utils.registry.build_functions import CONNECTOR


@CONNECTOR.register_module()
class BaseConnector(metaclass=ABCMeta):
    """Base class for video loader."""

    def __init__(self):
        super(BaseConnector, self).__init__()

    @abstractmethod
    def __next__(self, *args, **kwargs):
        """Make Connector iterable.

        return None, imgs, img0s, frame_idxs
            imgs : (torch.tensor) transformed image
            img0s : (numpy.ndarray) original image
            frame_idxs : (list) frmae indexs
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self, *args, **kwargs):
        """Make Connector iterable."""
        raise NotImplementedError
