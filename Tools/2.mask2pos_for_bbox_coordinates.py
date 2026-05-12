# python calculate_bbox.py \
#   --mask_folder /mnt/fast/nobackup/scratch4weeks/xxx/MedSAM/work_dir/medsam-vit-base/ISIC_Data/Part-1-Lesion-Segmentation/Training/ISBI2016_ISIC_Part1_Training_GroundTruth \
#   --target_size 256 256 \
#   --output_json bbox_coordinates_256.json


import os
import argparse
import numpy as np
import json
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate bounding box coordinates from masks")

    parser.add_argument("--mask_folder", type=str, required=True,
                        help="Input mask folder path (png format) or npy file path")
    parser.add_argument("--target_size", type=int, nargs=2, default=[256, 256],
                        help="Target size (width height), must be multiple of 16 (default: 256 256)")
    parser.add_argument("--output_json", type=str, default="bbox_coordinates_256.json",
                        help="Output bounding box JSON file path (default: bbox_coordinates_256.json)")
    parser.add_argument("--merge_channels", action="store_true",
                        help="For multi-channel masks, whether to merge all channels before calculating bbox (default: False, calculate each channel separately)")

    return parser.parse_args()

def process_single_npy_masks(mask_folder, target_size, merge_channels):
    """Each mask is a separate .npy file."""
    bbox_dict = {}
    npy_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.npy')])
    if not npy_files:
        raise FileNotFoundError("No .npy mask files found in directory")

    for fname in npy_files:
        mask_array = np.load(os.path.join(mask_folder, fname))   # (H,W) or (H,W,C)
        h_original, w_original = mask_array.shape[:2]
        original_size = (w_original, h_original)

        # Following logic is identical to single sample in process_npy_masks
        if mask_array.ndim == 3 and mask_array.shape[2] > 1:   # Multi-channel
            if merge_channels:
                merged = np.sum(mask_array, axis=-1)
                bbox = compute_bbox_from_mask(merged, target_size, original_size)
                if bbox is not None:
                    bbox_dict[fname] = bbox
            else:
                channel_bboxes = {f"channel_{c}": compute_bbox_from_mask(mask_array[:,:,c],
                                                                         target_size, original_size)
                                  for c in range(mask_array.shape[2])
                                  if compute_bbox_from_mask(mask_array[:,:,c], target_size, original_size) is not None}
                if channel_bboxes:
                    bbox_dict[fname] = channel_bboxes
        else:                                                  # Single channel
            bbox = compute_bbox_from_mask(mask_array, target_size, original_size)
            if bbox is not None:
                bbox_dict[fname] = bbox
    return bbox_dict

def compute_bbox_from_mask(mask_np, target_size, original_size):
    """
    Compute bounding box coordinates from mask.

    Args:
        mask_np: 2D numpy array
        target_size: (width, height) target size
        original_size: (width, height) original size

    Returns:
        (x1, y1, x2, y2) or None (if mask is empty)
    """
    non_zero_coords = np.argwhere(mask_np > 0)
    if len(non_zero_coords) == 0:
        return None

    y_coords, x_coords = non_zero_coords[:, 0], non_zero_coords[:, 1]
    x1_original, y1_original = np.min(x_coords), np.min(y_coords)
    x2_original, y2_original = np.max(x_coords), np.max(y_coords)

    # Calculate scale ratio
    scale_x = target_size[0] / original_size[0]
    scale_y = target_size[1] / original_size[1]

    # Calculate resized bounding box coordinates (round to integer)
    x1 = round(x1_original * scale_x)
    y1 = round(y1_original * scale_y)
    x2 = round(x2_original * scale_x)
    y2 = round(y2_original * scale_y)

    # Ensure coordinates are within resized dimensions
    x1 = max(0, min(x1, target_size[0]-1))
    y1 = max(0, min(y1, target_size[1]-1))
    x2 = max(x1, min(x2, target_size[0]-1))
    y2 = max(y1, min(y2, target_size[1]-1))

    return (x1, y1, x2, y2)

def process_png_masks(mask_folder, target_size):
    """Process PNG format mask files."""
    bbox_dict = {}
    
    for mask_filename in os.listdir(mask_folder):
        if not mask_filename.endswith(".png") and not mask_filename.endswith(".jpg"):
            continue
        
        mask_path = os.path.join(mask_folder, mask_filename)
        with Image.open(mask_path) as mask:
            mask_np = np.array(mask)

            # if mask_np.ndim == 3:
            	# mask_np = mask_np[:, :, 0]  # Take first channel, or use np.mean(mask_np, axis=2)

            if mask_np.ndim == 3:
                # Check if channels are identical
                channels_equal = np.allclose(mask_np[:,:,0], mask_np[:,:,1]) and np.allclose(mask_np[:,:,1], mask_np[:,:,2])

                if channels_equal:
                    print(f"{mask_filename}: Three channels identical, taking first channel")
                    mask_np = mask_np[:, :, 0]
                else:
                    print(f"{mask_filename}: Three channels different - R mean:{mask_np[:,:,0].mean():.2f}, "
                          f"G mean:{mask_np[:,:,1].mean():.2f}, B mean:{mask_np[:,:,2].mean():.2f}")
                    # Options: take max / mean / merge after binarization
                    mask_np = np.max(mask_np, axis=2)  # Or np.mean() or other strategy
            
            h_original, w_original = mask_np.shape
            original_size = (w_original, h_original)
            
            bbox = compute_bbox_from_mask(mask_np, target_size, original_size)
            if bbox is None:
                print(f"Warning: {mask_filename} has no target region, skipping")
                continue
            
            bbox_dict[mask_filename] = bbox
    
    return bbox_dict

def process_npy_masks(mask_npy_path, target_size, merge_channels):
    """Process NPY format mask files."""
    print(f"Loading mask file: {mask_npy_path}")
    masks = np.load(mask_npy_path)
    print(f"Mask array shape: {masks.shape}")

    bbox_dict = {}
    num_samples = len(masks)

    # Check if multi-channel
    is_multichannel = masks.ndim == 4 and masks.shape[3] > 1

    if is_multichannel:
        num_channels = masks.shape[3]
        print(f"Detected multi-channel mask, total {num_channels} channels")

        if merge_channels:
            print("Will merge all channels before calculating bbox")
        else:
            print("Will calculate bbox for each channel separately")
    
    for idx in range(num_samples):
        mask_array = masks[idx]
        h_original, w_original = mask_array.shape[0], mask_array.shape[1]
        original_size = (w_original, h_original)
        
        if is_multichannel:
            if merge_channels:
                # Merge all channels (sum operation)
                merged_mask = np.sum(mask_array, axis=-1)
                bbox = compute_bbox_from_mask(merged_mask, target_size, original_size)

                if bbox is None:
                    print(f"Warning: Sample {idx} has no target region, skipping")
                    continue

                bbox_dict[f"sample_{idx:04d}"] = bbox
            else:
                # Process each channel separately
                channel_bboxes = {}
                has_valid_bbox = False

                for c in range(num_channels):
                    channel_mask = mask_array[:, :, c]
                    bbox = compute_bbox_from_mask(channel_mask, target_size, original_size)

                    if bbox is not None:
                        channel_bboxes[f"channel_{c}"] = bbox
                        has_valid_bbox = True

                if not has_valid_bbox:
                    print(f"Warning: Sample {idx} has no target region in all channels, skipping")
                    continue

                bbox_dict[f"sample_{idx:04d}"] = channel_bboxes
        else:
            # Single channel mask
            bbox = compute_bbox_from_mask(mask_array, target_size, original_size)
            
            if bbox is None:
                print(f"Warning: Sample {idx} has no target region, skipping")
                continue
            
            bbox_dict[f"sample_{idx:04d}"] = bbox
    
    return bbox_dict

def main():
    args = parse_args()
    
    mask_folder = args.mask_folder
    target_size = tuple(args.target_size)
    output_json = args.output_json
    merge_channels = args.merge_channels
    
    # Validate target size is multiple of 16
    if (target_size[0] % 16 != 0) or (target_size[1] % 16 != 0):
        raise ValueError(f"Target size {target_size} must be multiple of 16 (both width and height must be divisible by 16)")

    # Determine input type
    if os.path.isfile(mask_folder) and mask_folder.endswith('.npy'):
        # Process npy file
        bbox_dict = process_npy_masks(mask_folder, target_size, merge_channels)
    elif os.path.isdir(mask_folder):
        # Check if contains npy files
        npy_files = [f for f in os.listdir(mask_folder) if f.endswith('.npy')]

        if npy_files:
            if 'masks.npy' in npy_files or len(npy_files) < 3 and npy_files[0].endswith('.npy'):
                mask_npy_path = os.path.join(mask_folder, 'masks.npy' if 'masks.npy' in npy_files else npy_files[0])
                bbox_dict = process_npy_masks(mask_npy_path, target_size, merge_channels)
            else:
          	    bbox_dict = process_single_npy_masks(mask_folder, target_size, merge_channels)
        else:
            bbox_dict = process_png_masks(mask_folder, target_size)
        
        # if npy_files:
        #     # Folder contains npy files
        #     if 'masks.npy' in npy_files:
        #         mask_npy_path = os.path.join(mask_folder, 'masks.npy')
        #     else:
        #         mask_npy_path = os.path.join(mask_folder, npy_files[0])
        #         print(f"Using npy file: {npy_files[0]}")

        #     bbox_dict = process_npy_masks(mask_npy_path, target_size, merge_channels)
        # else:
        #     # Process png files
        #     bbox_dict = process_png_masks(mask_folder, target_size)
        
    else:
        raise ValueError(f"Invalid input path: {mask_folder}")

    # Print result examples
    print(f"\nProcessing completed, total {len(bbox_dict)} samples")
    print("Bounding box dictionary examples:")
    for filename, bbox in list(bbox_dict.items())[:3]:
        print(f"{filename}: {bbox}")

    # Save dictionary as json file
    with open(output_json, "w") as f:
        json.dump(bbox_dict, f, indent=2)

    print(f"\nBounding boxes saved to: {output_json}")

if __name__ == "__main__":
    main()
