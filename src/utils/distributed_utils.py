import Tree_Dataset
import torch
import torchvision

@functools.lru_cache(maxsize=None)
def is_root_process():
    """Return whether this process is the root process."""
    return torch.distributed.get_rank() == 0

# The reason we define this is that `torch.distributed` does not
# implement it; for the global rank, there's
# `torch.distributed.get_rank()`.
# Some notes on what this means:
# The world is the group containing all the processes in the distributed training.
# The worlde size as you may have guessed is the number of processes in the system.
# Typically each GPU corresponds to a process.
# Global rank means the process ID across all nodes/servers
# Local rank means the ID of the process on the current node/server
@functools.lru_cache(maxsize=None)
def get_local_rank():
    """Return the local rank of this process."""
    return int(os.getenv('LOCAL_RANK'))

def print0(*args, **kwargs):
    """Print something only on the root process."""
    if is_root_process():
        print(*args, **kwargs)


def save0(*args, **kwargs):
    """Pass the given arguments to `torch.save`, but only on the root
    process.
    """
    # We do *not* want to write to the same location with multiple
    # processes at the same time.
    if is_root_process():
        torch.save(*args, **kwargs)

def prepare_datasets(args, device):
    """ Return the train, validation and test datasets wrapped in dataloaders using distributed samplers for parallel GPU training.
    Args:
        --dataset_path: path of directory containing "Trees.json" an images.
        --val_ratio, test_ratio: portion of examples to use for train/val/test split.
        --seed: seed used for splitting and shuffling datasets.
        --batch_size: 'How many samples to use per batch. Note that this is the local batch size; the effective, or global, batch size will be obtained by multiplying this number with the number of processes.
        --train-num-workers, valid-num-workers: How many workers to use for processing the training/validation dataset.
    """
    dataset = Tree_Dataset.TreesDataset(args.dataset_path)

    dataset_size = len(dataset)
    
    assert (args.val_ratio >= 0 and args.test_ratio >= 0 and args.val_ratio+test_ratio <= 1), "Invalid val/test ratios."

    val_size = int(args.val_ratio * dataset_size)
    test_size = int(args.test_ratio * dataset_size)
    train_size = dataset_size-val_size-test_size

    generator = torch.Generator().manual_seed(args.seed)

    train_dset, val_dset, test_dset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dset,
        shuffle=True,
        seed=args.seed
    )
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dset,
        shuffle=False,
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dset,
        shuffle=False,
    )

    train_dset = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size,
        collate_fn=Tree_Dataset.collate_trees, # This collate_fn is a big assumption for now
        sampler=train_sampler,
        # Use multiple processes for loading data.
        num_workers=args.train_num_workers,
        # Use pinned memory on GPUs for faster device-copy.
        pin_memory=True,
        persistent_workers=args.train_num_workers > 0,
    )
    valid_dset = torch.utils.data.DataLoader(
        valid_dset,
        batch_size=args.batch_size,
        collate_fn=Tree_Dataset.collate_trees,
        sampler=valid_sampler,
        num_workers=args.valid_num_workers,
        # Use pinned memory on GPUs for faster device-copy.
        pin_memory=True,
        persistent_workers=args.valid_num_workers > 0,
    )
    test_dset = torch.utils.data.DataLoader(
        test_dset,
        batch_size=args.batch_size,
        collate_fn=Tree_Dataset.collate_trees,
        sampler=test_sampler,
        # Use pinned memory on GPUs for faster device-copy.
        pin_memory=True,
    )
    return train_dset, valid_dset, test_dset