import lmdb
import os
import pickle

def create_small_lmdb(src_lmdb, dest_lmdb, num_samples=100):
    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(dest_lmdb), exist_ok=True)

    # Open the original LMDB file directly by setting subdir=False
    env_in = lmdb.open(src_lmdb, readonly=True, lock=False, subdir=False)

    # Create the new smaller LMDB file directly by setting subdir=False
    env_out = lmdb.open(dest_lmdb, map_size=1024**3, subdir=False) # 1GB map size limit

    with env_in.begin() as txn_in, env_out.begin(write=True) as txn_out:
        # LMDBs in fairchem look for the "length" key to know the total size
        txn_out.put(b"length", pickle.dumps(num_samples))

        # Copy over only the first 'num_samples' items
        for i in range(num_samples):
            key = str(i).encode("ascii")
            data = txn_in.get(key)
            if data is not None:
                txn_out.put(key, data)

    env_in.close()
    env_out.close()
    print(f"Created {dest_lmdb} with {num_samples} samples.")

# Example usage:
# Point directly to the .lmdb file and use subdir=False inside the function
create_small_lmdb(
    src_lmdb="/home/ryoji/equiformer_v3/src/fairchem/data/s2ef/all/val_id/data.0000.lmdb",
    dest_lmdb="/home/ryoji/equiformer_v3/src/fairchem/data/s2ef/all/val_id_mini/data.0000.lmdb",
    num_samples=30000
)
