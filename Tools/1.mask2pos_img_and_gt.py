import os
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Batch resize ISIC images and masks")

    parser.add_argument("--image_folder", type=str, required=True,
                        help="Input image folder path (jpg/png format) or npy file path")
    parser.add_argument("--gt_folder", type=str, required=True,
                        help="Input mask folder path (png format) or npy file path")
    parser.add_argument("--out_image_folder", type=str, default="ISIC_2016_resized_images_256",
                        help="Output image folder path (default: ISIC_2016_resized_images_256)")
    parser.add_argument("--out_gt_folder", type=str, default="ISIC_2016_resized_groundtruth_256",
                        help="Output mask folder path (default: ISIC_2016_resized_groundtruth_256)")
    parser.add_argument("--target_size", type=int, nargs=2, default=[256, 256],
                        help="Target size (width height), must be multiple of 16 (default: 256 256)")
    parser.add_argument("--data_config_json_path", type=str, default="dataset_config.json")
    parser.add_argument("--dataset_name", type=str, default="None")

    return parser.parse_args()


def save_mask_as_png(mask_array, base_name, out_gt_folder):
    """
    Save mask as png format.
    - Single channel: save directly under out_gt_folder/
    - Multi-channel: save under out_gt_folder/channel_X/

    Args:
        mask_array: numpy array, shape (H, W) or (H, W, C)
        base_name: base filename (without extension)
        out_gt_folder: output folder path
    """
    if mask_array.ndim == 2:
        # Single channel mask (H, W), save directly
        mask_pil = Image.fromarray(mask_array)
        mask_pil.save(os.path.join(out_gt_folder, f"{base_name}.png"))
    elif mask_array.ndim == 3 and mask_array.shape[2] == 1:
        # (H, W, 1) shape, treat as single channel, save directly
        mask_pil = Image.fromarray(mask_array[:, :, 0])
        mask_pil.save(os.path.join(out_gt_folder, f"{base_name}.png"))
    elif mask_array.ndim == 3:
        # Multi-channel mask (H, W, C), save each channel to corresponding folder
        num_channels = mask_array.shape[2]
        for c in range(num_channels):
            channel_folder = os.path.join(out_gt_folder, f"channel_{c}")
            os.makedirs(channel_folder, exist_ok=True)

            channel_data = mask_array[:, :, c]
            mask_pil = Image.fromarray(channel_data)
            mask_pil.save(os.path.join(channel_folder, f"{base_name}.png"))
    else:
        raise ValueError(f"Unsupported mask array shape: {mask_array.shape}")

def process_single_npy_files(image_folder, gt_folder, out_image_folder, out_gt_folder, target_size):
    """
    Process case where each image is a separate .npy file (single format).
    Output: Each image/mask saved as separate png file, preserving original filename.
    """
    os.makedirs(out_image_folder, exist_ok=True)
    os.makedirs(out_gt_folder, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.npy')])
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith('.npy')])

    if len(image_files) != len(gt_files):
        raise ValueError(f"Image and mask .npy file count mismatch: {len(image_files)} vs {len(gt_files)}")

    print(f"Total {len(image_files)} image-mask pairs to process")

    for img_file, gt_file in tqdm(zip(image_files, gt_files), total=len(image_files), desc="Processing"):
        img_path = os.path.join(image_folder, img_file)
        gt_path = os.path.join(gt_folder, gt_file)

        # Load npy arrays
        img_array = np.load(img_path)
        gt_array = np.load(gt_path)

        # Resize
        resized_img = resize_image_array(img_array, target_size, is_mask=False)
        resized_gt = resize_image_array(gt_array, target_size, is_mask=True)

        # Generate output filename (remove .npy suffix, keep original name)
        base_name = os.path.splitext(img_file)[0]
        out_img_name = f"{base_name}.png"

        # Save image as png
        img_pil = Image.fromarray(resized_img)
        img_pil.save(os.path.join(out_image_folder, out_img_name))

        # Save mask as png (auto-handle single/multi-channel)
        save_mask_as_png(resized_gt, base_name, out_gt_folder)

    print(f"Processing completed!")


def resize_image_array(img_array, target_size, is_mask=False):
    """
    Resize a single image array (for NPY format).

    Args:
        img_array: numpy array, shape (H, W) or (H, W, C)
        target_size: target size (width, height)
        is_mask: whether it's a mask (mask uses NEAREST interpolation, image uses BICUBIC)

    Returns:
        resized_array: resized numpy array
    """
    # Data type conversion: ensure uint8
    if img_array.dtype in [np.float32, np.float64]:
        # If float and in [0,1] range, multiply by 255
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    elif img_array.dtype != np.uint8:
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    if is_mask and img_array.max() == 1:
        img_array = img_array * 255

    # Handle multi-channel mask (e.g., 6 channels) - resize per channel
    if is_mask and img_array.ndim == 3 and img_array.shape[2] > 1:
        resized_channels = []
        for c in range(img_array.shape[2]):
            channel = img_array[:, :, c]
            img = Image.fromarray(channel)
            resized_channel = img.resize(target_size, Image.NEAREST)
            resized_channels.append(np.array(resized_channel))
        return np.stack(resized_channels, axis=2)
    else:
        # Single channel or RGB image
        img = Image.fromarray(img_array)
        resample_method = Image.NEAREST if is_mask else Image.BICUBIC
        resized_img = img.resize(target_size, resample_method)
        return np.array(resized_img)


def process_npy_files(image_npy_path, mask_npy_path, out_image_folder, out_gt_folder, target_size):
    """
    Process npy format image and mask files (multi format: whole npy file).
    Output: Each image/mask saved as separate png file, using sequential numbering.

    Args:
        image_npy_path: image npy file path
        mask_npy_path: mask npy file path
        out_image_folder: output image folder
        out_gt_folder: output mask folder
        target_size: target size (width, height)
    """
    # If folder is passed, find .npy file
    if os.path.isdir(image_npy_path):
        npy_files = [f for f in os.listdir(image_npy_path) if f.endswith('.npy')]
        if 'images.npy' in npy_files:
            image_npy_path = os.path.join(image_npy_path, 'images.npy')
        elif len(npy_files) == 1:
            image_npy_path = os.path.join(image_npy_path, npy_files[0])
        elif len(npy_files) > 1:
            print(f"Warning: Found multiple npy files {npy_files}, using first one")
            image_npy_path = os.path.join(image_npy_path, npy_files[0])
        else:
            raise FileNotFoundError(f"No npy file found in {image_npy_path}")

    if os.path.isdir(mask_npy_path):
        npy_files = [f for f in os.listdir(mask_npy_path) if f.endswith('.npy')]
        if 'masks.npy' in npy_files:
            mask_npy_path = os.path.join(mask_npy_path, 'masks.npy')
        elif len(npy_files) == 1:
            mask_npy_path = os.path.join(mask_npy_path, npy_files[0])
        elif len(npy_files) > 1:
            print(f"Warning: Found multiple npy files {npy_files}, using first one")
            mask_npy_path = os.path.join(mask_npy_path, npy_files[0])
        else:
            raise FileNotFoundError(f"No npy file found in {mask_npy_path}")

    print(f"Loading image file: {image_npy_path}")
    images = np.load(image_npy_path)
    print(f"Image array shape: {images.shape}, dtype: {images.dtype}")

    print(f"Loading mask file: {mask_npy_path}")
    masks = np.load(mask_npy_path)
    print(f"Mask array shape: {masks.shape}, dtype: {masks.dtype}")

    # Check if image and mask counts match
    if len(images) != len(masks):
        raise ValueError(f"Image count ({len(images)}) does not match mask count ({len(masks)})!")

    num_samples = len(images)
    print(f"Total {num_samples} image-mask pairs to process")

    # Check if resize is needed
    current_size = (images.shape[2], images.shape[1])  # (width, height)
    if current_size == target_size:
        print(f"Note: Current image size {current_size} equals target size {target_size}")
        print("Will process data directly (performing data type check and conversion)")

    # Create output folders
    os.makedirs(out_image_folder, exist_ok=True)
    os.makedirs(out_gt_folder, exist_ok=True)

    # Calculate zero-padding digits needed for filename
    num_digits = len(str(num_samples))

    # Process each image-mask pair
    for idx in tqdm(range(num_samples), desc="Processing"):
        img_array = images[idx]
        mask_array = masks[idx]

        # Resize image
        resized_img = resize_image_array(img_array, target_size, is_mask=False)
        resized_mask = resize_image_array(mask_array, target_size, is_mask=True)

        # Generate filename (using zero-padded index, starting from 1)
        base_name = f"image_{str(idx + 1).zfill(num_digits)}"

        # Save image as png
        img_pil = Image.fromarray(resized_img)
        img_pil.save(os.path.join(out_image_folder, f"{base_name}.png"))

        # Save mask as png (auto-handle single/multi-channel)
        save_mask_as_png(resized_mask, base_name, out_gt_folder)

    print(f"Processing completed!")
    print(f"Images saved to: {out_image_folder}")
    print(f"Masks saved to: {out_gt_folder}")


def process_image_files(image_folder, gt_folder, out_image_folder, out_gt_folder,
                       target_size, mask_suffix, image_extensions):
    """
    Process jpg/png format image and mask files (keep original logic unchanged).
    """
    # Create output folders
    os.makedirs(out_image_folder, exist_ok=True)
    os.makedirs(out_gt_folder, exist_ok=True)

    # Iterate all jpg/png images
    image_files = [f for f in os.listdir(image_folder)
                   if f.endswith(".jpg") or f.endswith(".png")]

    for img_filename in tqdm(image_files, desc="Processing"):
        img_name = os.path.splitext(img_filename)[0]
        gt_filename = f"{img_name}{mask_suffix}"
        gt_path = os.path.join(gt_folder, gt_filename)

        if not os.path.exists(gt_path):
            print(f"Warning: Mask file {gt_filename} not found, skipping image {img_filename}")
            continue

        with Image.open(os.path.join(image_folder, img_filename)) as img, \
             Image.open(gt_path) as gt:

            resized_img = img.resize(target_size, Image.BICUBIC)
            resized_gt = gt.resize(target_size, Image.NEAREST)

            resized_img.save(os.path.join(out_image_folder, img_filename))
            resized_gt.save(os.path.join(out_gt_folder, gt_filename))


def main():
    args = parse_args()

    image_folder = args.image_folder
    gt_folder = args.gt_folder
    out_image_folder = args.out_image_folder
    out_gt_folder = args.out_gt_folder
    target_size = tuple(args.target_size)
    data_config_json_path = args.data_config_json_path
    dataset_name = args.dataset_name
  	
    with open(data_config_json_path, 'r') as f:
        data = json.load(f)
    
    mask_suffix = data['datasets'][dataset_name]['mask_suffix']
    image_extensions = data['datasets'][dataset_name]['image_extensions']
    npy_format = None
    if 'npy' in image_extensions:
    	npy_format = data['datasets'][dataset_name]['npy_format']
    
    print(f"Dataset: {dataset_name}")
    print(f"Image format: {image_extensions}")
    print(f"Mask suffix: {mask_suffix}")
    print(f"Target size: {target_size}")

    # Validate target size is multiple of 16
    if (target_size[0] % 16 != 0) or (target_size[1] % 16 != 0):
        raise ValueError(f"Target size {target_size} must be multiple of 16 (both width and height must be divisible by 16)")

    # Determine processing method based on image_extensions
    if isinstance(image_extensions, list):
        ext_list = image_extensions
    else:
        ext_list = [image_extensions]
    
    if 'npy' in ext_list:
        # Process npy files
        if os.path.isdir(image_folder):
            npy_files = [f for f in os.listdir(image_folder) if f.endswith('.npy')]
            if len(npy_files) == 0:
                raise FileNotFoundError(f"No .npy file found in {image_folder}")
            elif len(npy_files) == 1:
                image_npy_path = os.path.join(image_folder, npy_files[0])
            else:
                if 'images.npy' in npy_files:
                    image_npy_path = os.path.join(image_folder, 'images.npy')
                else:
                    image_npy_path = os.path.join(image_folder, npy_files[0])
                    print(f"Warning: Found multiple .npy files, using {npy_files[0]}")
        else:
            image_npy_path = image_folder

        if os.path.isdir(gt_folder):
            npy_files = [f for f in os.listdir(gt_folder) if f.endswith('.npy')]
            if len(npy_files) == 0:
                raise FileNotFoundError(f"No .npy file found in {gt_folder}")
            elif len(npy_files) == 1:
                mask_npy_path = os.path.join(gt_folder, npy_files[0])
            else:
                if 'masks.npy' in npy_files:
                    mask_npy_path = os.path.join(gt_folder, 'masks.npy')
                else:
                    mask_npy_path = os.path.join(gt_folder, npy_files[0])
                    print(f"Warning: Found multiple .npy files, using {npy_files[0]}")
        else:
            mask_npy_path = gt_folder

        # Determine if whole .npy or separate .npy files
        if os.path.isfile(image_folder) and image_folder.endswith('.npy'):
            # Single .npy file (whole)
            process_npy_files(image_folder, gt_folder, out_image_folder, out_gt_folder, target_size)
        elif os.path.isdir(image_folder):
            # Check if contains single .npy file (whole)
            npy_files = [f for f in os.listdir(image_folder) if f.endswith('.npy')]
            mask_npy_files = [f for f in os.listdir(gt_folder) if f.endswith('.npy')]
            if npy_format == 'multi':
                # Whole .npy
                image_npy_path = os.path.join(image_folder, 'images.npy' if 'images.npy' in npy_files else npy_files[0])
                mask_npy_path = os.path.join(gt_folder, 'masks.npy' if 'masks.npy' in mask_npy_files else mask_npy_files[0])
                process_npy_files(image_npy_path, mask_npy_path, out_image_folder, out_gt_folder, target_size)
            elif npy_format == 'single':
                # Each image is a separate .npy
                process_single_npy_files(image_folder, gt_folder, out_image_folder, out_gt_folder, target_size)
            else:
                raise ValueError(f"Unsupported npy_format: {npy_format}, must be 'multi' or 'single'")
        else:
            raise ValueError("Cannot identify .npy data organization")

    elif any(ext in ext_list for ext in ['jpg', 'png']):
        # Process jpg/png files (keep original logic)
        process_image_files(image_folder, gt_folder, out_image_folder,
                           out_gt_folder, target_size, mask_suffix, image_extensions)
    else:
        raise ValueError(f"Unsupported image format: {image_extensions}")

    print("All files processed!")

if __name__ == "__main__":
    main()
