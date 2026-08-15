import argparse
import os
import lmdb
from tqdm import tqdm

def create_lmdb(img_dir, lmdb_path, max_size=1099511627776): # 1TB default map_size
    """
    Creates an LMDB file containing raw image bytes, with the image filename as the key.
    
    Args:
        img_dir: Path to the directory containing images.
        lmdb_path: Path to the output LMDB file (e.g., 'data.mdb').
        max_size: Maximum size of the LMDB map.
    """
    # Define valid image extensions
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    
    # Collect all image paths
    image_paths = []
    for root, _, files in os.walk(img_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                image_paths.append(os.path.join(root, f))
                
    image_paths.sort()
    num_images = len(image_paths)
    print(f"Found {num_images} images in {img_dir}.")
    
    if num_images == 0:
        print("No images found. Exiting.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(lmdb_path)), exist_ok=True)
    
    # Open LMDB environment
    # subdir=False means lmdb_path points directly to the file, not a directory
    env = lmdb.open(lmdb_path, map_size=max_size, subdir=False, readonly=False, meminit=False, map_async=True)
    
    txn = env.begin(write=True)
    write_frequency = 5000
    
    print("Writing images to LMDB...")
    for idx, img_path in enumerate(tqdm(image_paths)):
        # Use the image filename as the key
        key = os.path.relpath(img_path, img_dir).encode('utf-8')
        
        # Read raw bytes of the image
        with open(img_path, 'rb') as f:
            val = f.read()
            
        txn.put(key, val)
        
        # Commit periodically to free up memory and persist changes
        if (idx + 1) % write_frequency == 0:
            txn.commit()
            txn = env.begin(write=True)
            
    # The snippet uses `txn.stat()['entries'] - 1` to get length, 
    # which implies one extra metadata entry exists. 
    # We add a 'length' key as the metadata entry here to be consistent.
    txn.put(b'length', str(num_images).encode('utf-8'))
    
    # Final commit
    txn.commit()
    env.sync()
    env.close()
    
    print(f"Successfully created LMDB at {lmdb_path} with {num_images} entries.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create an LMDB dataset from a directory of images.")
    parser.add_argument('--img_dir', type=str, required=True, help="Directory containing the images to pack")
    parser.add_argument('--lmdb_path', type=str, required=True, help="Output LMDB path")
    
    args = parser.parse_args()
    create_lmdb(args.img_dir, args.lmdb_path)
