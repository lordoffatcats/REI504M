#!/usr/bin/env bash

# This `sbatch` script provides a template for using an in-memory
# dataset by creating and extracting a tar archive into `/dev/shm`.
#
# Places that need adaptation depending on your needs are marked with
# "TODO".
#
# The advantage of using an in-memory dataset is blazing performance.
# The disadvantage is that your data has to fit into RAM next to the
# program state. If your data does not fit into RAM, prefer
# `squashfs.sbatch` instead.
# Another disadvantage is that if your data changes, you will have to
# add it to the tar archive manually. You should also avoid writing to
# the same location where you read data from since the read location
# is node-local and temporary; you will not be able to access the data
# after the job ends.

#SBATCH --nodes=# TODO
#SBATCH --ntasks-per-node=# TODO
#SBATCH --account=# TODO
#SBATCH --partition=# TODO

[...]  # TODO

# Path to your data.
export DATA_PATH="..."  # TODO
export TAR_PATH="$DATA_PATH.tar"
# Where you will access your data from.
export SHM_PATH="/dev/shm/$(whoami)/$(basename "$DATA_PATH")"

# Create the tar file if it doesn't exist yet.
if ! [ -e "$TAR_PATH" ]; then
   tar cf "$TAR_PATH" -C "$(dirname "$DATA_PATH")" "$(basename "$DATA_PATH")"
fi

extract_to_shm() {
    # Do nothing on tasks with node-local rank other than 0.
    ((SLURM_LOCALID)) && return 0

    # We completely remove `SHM_PATH` before doing anything else to
    # prevent dirty data (maybe there are still some remains from
    # previous jobs).
    rm -rf "$SHM_PATH"

    mkdir -p "$SHM_PATH"
    # Extract data to `SHM_PATH`.
    tar xf "$TAR_PATH" -C "$(dirname "$SHM_PATH")"
    # Make `SHM_PATH` private.
    chmod 700 "$SHM_PATH"
}
export -f extract_to_shm

# Extract the data to `SHM_PATH`; only once per node.
srun bash -c extract_to_shm
# Use the data at `SHM_PATH`.
srun python main.py --input_path="$SHM_PATH"  # TODO