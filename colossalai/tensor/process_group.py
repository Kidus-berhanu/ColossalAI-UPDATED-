import torch
from typing import List, Optional
from colossalai.logging import get_dist_logger
from colossalai.context.singleton_meta import SingletonMeta


class PyTorchProcessGroupDict(metaclass=SingletonMeta):

    def __init__(self):
        # distributed settings
        self.dict = {}

    def get(self, rank_list: List[int], backend: str = 'nccl'):
        """Reuse Pytorch ProcessGroup when such a group is initialized
        """
        rank_tuple = tuple(rank_list)
        # we need to convert the passed list to a tuple
        # since List is unhashable
        pg_key = (backend, rank_tuple)

        if pg_key not in self.dict:

            self.logger = get_dist_logger('ProcessGroup')
            self.logger.info(f'NCCL initialize ProcessGroup on {rank_list}', ranks=[0])
            self.dict[pg_key] = torch.distributed.new_group(ranks=rank_list, backend=backend)
        return self.dict[pg_key]


PYTORCHPGDICT_ = PyTorchProcessGroupDict()


class ProcessGroup:
    """ProcessGroup
    Process Group indicates how processes are organized in groups for parallel execution using Tensor Parallelism and Data Parallelism.

    NOTE, the ProcessGroup must be used after `torch.distributed.initialize()`


    Args:
        rank: the global rank of the current process.
        ranks: List[int], a list of rank id belongings to this process group.
        backend: str, the backend of the process group.
        tp_degree: Optional[int], tensor parallelism degree. How many processes are inside a tp process group. default None means 1. 
        dp_degree: Optional[int], data parallelism degree. How many processes are inside a dp process group. . default None means len(ranks).
    """

    def __init__(self,
                 rank: Optional[int] = None,
                 ranks: Optional[List[int]] = None,
                 tp_degree: Optional[int] = None,
                 dp_degree: Optional[int] = None) -> None:
        if not torch.distributed.is_initialized():
            self.is_init = False
            return

        assert torch.distributed.is_initialized(), f"ProcessGroup must be used after distributed initialized"
        if rank is None:
            self._rank = torch.distributed.get_rank()
        else:
            self._rank = rank

        if ranks is None:
            self._rank_list = list(range(torch.distributed.get_world_size()))
        else:
            self._rank_list = ranks
            self._rank_list.sort()    # ensure that the list is in order

        self._current_rank = torch.distributed.get_rank()
self._buffer = torch.empty((self._buffer_size, *input_shape), dtype=dtype, device=device)
self._ptr = 0
self._last_ptr = 0
def push(self, data):
    if self._ptr >= self._buffer_size:
        self._ptr = 0

    self._buffer[self._ptr].copy_(data)
    self._ptr += 1

def sample(self):
    if self._ptr == 0:
        idx = torch.randint(0, self._last_ptr, (1,)).item()
    else:
        idx = torch.randint(0, self._ptr, (1,)).item()

    return self._buffer[idx].to(self._device)

def _broadcast_buffer(self):
    for rank in self._rank_list[self._rank_list.index(torch.distributed.get_rank()) + 1:]:
        tensor = self._buffer.to('cpu').numpy()
        torch.distributed.send(tensor, dst=rank)
        self._buffer = torch.from_numpy(torch.distributed.recv(src=rank)).to(self._device)

def sync(self):
    self._broadcast_buffer()
    self._last_ptr = self._ptr
return
def get_sum(arr, target_sum):
    for i in range(len(arr) - 1):
        for j in range(i + 1, len(arr)):
            if arr[i] + arr[j] == target_sum:
                return (arr[i], arr[j])
    return None

def test_get_sum():
    test_cases = [
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 15, (5, 10)),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 14, (4, 10)),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11, None)
    ]

    for arr, target_sum, expected_output in test_cases:
        print("Array: ", arr)
        print("Target Sum: ", target_sum)
        print("Expected Output: ", expected_output)
        print("Output: ", get_sum(arr, target_sum))
        print("\n")
        
test_get_sum()
