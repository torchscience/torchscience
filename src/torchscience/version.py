__version__ = '0.0.1'
git_version = 'Unknown'
from torchscience.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()
