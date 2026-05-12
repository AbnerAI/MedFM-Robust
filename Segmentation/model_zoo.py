# model_zoo.py
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, List
from segment_anything import sam_model_registry

# ---------------------------- Base Class ----------------------------
class SegmentationModelBase(nn.Module, ABC):
    def __init__(self, device, cfg):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.name = cfg.get("name", "Unknown")
        self.prompt_type = cfg.get("prompt_type", "none")
        self.prompt_required = cfg.get("prompt_required", False)

    @abstractmethod
    def forward(self, img_tensor, input_boxes=None, target_mask=None):
        """
        Args:
            img_tensor: [B, C, H, W]
            input_boxes: prompt info (box/point/none)
            target_mask: [B, 1, H, W] for computing loss during training
        Returns:
            pred_masks: [B, 1, H, W]
            loss: Optional[Tensor] if target_mask is provided
        """
        pass

    def predict(self, img_tensor, input_boxes=None):
        """Inference mode, only returns prediction results."""
        with torch.no_grad():
            return self.forward(img_tensor, input_boxes)


# ---------------------------- SAM Series Models ----------------------------
class MedSAMModel(SegmentationModelBase):
    def __init__(self, device, cfg):
        super().__init__(device, cfg)
        try:
            from transformers import SamModel, SamProcessor
            self.model = SamModel.from_pretrained(cfg["repo_id"]).to(device)
            self.processor = SamProcessor.from_pretrained(cfg["repo_id"])
            self.processor.image_processor.do_rescale = False
            print(f"[OK] Successfully loaded {cfg['name']}")
        except Exception as e:
            print(f"[ERROR] Failed to load {cfg['name']}: {e}")
            raise

    def forward(self, img_tensor, input_boxes=None, target_mask=None):
        img_tensor = img_tensor.to(self.device)

        # Check if gradients are needed
        if img_tensor.requires_grad:
            # Adversarial attack mode, ensure gradients are preserved
            inputs = self.processor(
                images=img_tensor,
                input_boxes=[[input_boxes]] if input_boxes is not None else None,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            # If processor disconnected gradients, try to fix
            if not inputs['pixel_values'].requires_grad:
                processed_shape = inputs['pixel_values'].shape
                other_inputs = {k: v for k, v in inputs.items() if k != 'pixel_values'}

                # Manually process image to preserve gradients
                if processed_shape[-2:] != img_tensor.shape[-2:]:
                    processed_img = F.interpolate(
                        img_tensor,
                        size=processed_shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                else:
                    processed_img = img_tensor

                inputs = {**other_inputs, 'pixel_values': processed_img}
        else:
            # Inference mode, use processor normally
            inputs = self.processor(
                images=img_tensor,
                input_boxes=[[input_boxes]] if input_boxes is not None else None,
                return_tensors="pt",
                padding=True
            ).to(self.device)

        # Model forward pass
        outputs = self.model(**inputs, multimask_output=False)
        pred_masks = outputs.pred_masks.sigmoid()

        # Handle dimensions
        if len(pred_masks.shape) == 5:
            pred_masks = pred_masks.squeeze(2)

        # Resize to original image size
        pred_masks = F.interpolate(
            pred_masks,
            size=img_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # Compute loss
        if target_mask is not None:
            target_mask = target_mask.to(self.device)
            if target_mask.shape != pred_masks.shape:
                target_mask = F.interpolate(
                    target_mask,
                    size=pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
            with torch.amp.autocast('cuda', enabled=False):
            	loss = F.binary_cross_entropy(pred_masks.float(), target_mask.float())
            return pred_masks, loss

        return pred_masks


class SAMModel(MedSAMModel):
    """Same as MedSAM, only different weights."""
    pass


class SAMPointModel(SegmentationModelBase):
    """SAM using single point prompt."""
    def __init__(self, device, cfg):
        super().__init__(device, cfg)
        try:
            from transformers import SamModel, SamProcessor
            self.model = SamModel.from_pretrained(cfg["repo_id"]).to(device)
            self.processor = SamProcessor.from_pretrained(cfg["repo_id"])
            self.processor.image_processor.do_rescale = False
            print(f"[OK] Successfully loaded {cfg['name']}")
        except Exception as e:
            print(f"[ERROR] Failed to load {cfg['name']}: {e}")
            raise

    def forward(self, img_tensor, input_points=None, target_mask=None):
        img_tensor = img_tensor.to(self.device)

        inputs = self.processor(
            images=img_tensor,
            input_points=[[input_points]] if input_points is not None else None,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        outputs = self.model(**inputs, multimask_output=False)
        pred_masks = outputs.pred_masks.sigmoid()

        if len(pred_masks.shape) == 5:
            pred_masks = pred_masks.squeeze(2)

        pred_masks = F.interpolate(
            pred_masks,
            size=img_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        if target_mask is not None:
            target_mask = target_mask.to(self.device)
            # Use autocast-safe way to compute loss
            with torch.amp.autocast('cuda', enabled=False):
                loss = F.binary_cross_entropy(pred_masks.float(), target_mask.float())
            return pred_masks, loss

        return pred_masks


# ---------------------------- Traditional Segmentation Models ----------------------------
class UNetModel(SegmentationModelBase):
    def __init__(self, device, cfg):
        super().__init__(device, cfg)
        try:
            # Use simple UNet implementation or pretrained model
            import torchvision.models.segmentation as seg_models
            self.model = seg_models.fcn_resnet50(pretrained=True, num_classes=1).to(device)

            # If local weights exist, load them
            if cfg.get("local_path") and os.path.exists(cfg["local_path"]):
                self.model.load_state_dict(torch.load(cfg["local_path"], map_location=device))
                print(f"[OK] Loaded weights from {cfg['local_path']}")
            else:
                print(f"[OK] Loaded {cfg['name']} with pretrained weights")
        except Exception as e:
            print(f"[ERROR] Failed to load {cfg['name']}: {e}")
            raise

    def forward(self, img_tensor, input_boxes=None, target_mask=None):
        img_tensor = img_tensor.to(self.device)

        # UNet doesn't need prompts, ignore input_boxes
        out = self.model(img_tensor)['out']
        pred_masks = torch.sigmoid(out)

        if target_mask is not None:
            target_mask = target_mask.to(self.device)
            if target_mask.shape != pred_masks.shape:
                target_mask = F.interpolate(
                    target_mask,
                    size=pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
            # Use autocast-safe way to compute loss
            with torch.amp.autocast('cuda', enabled=False):
                loss = F.binary_cross_entropy(pred_masks.float(), target_mask.float())
            return pred_masks, loss

        return pred_masks


class DeepLabV3Model(SegmentationModelBase):
    def __init__(self, device, cfg):
        super().__init__(device, cfg)
        try:
            import torchvision.models.segmentation as seg_models
            self.model = seg_models.deeplabv3_resnet101(pretrained=True, num_classes=1).to(device)

            if cfg.get("local_path") and os.path.exists(cfg["local_path"]):
                self.model.load_state_dict(torch.load(cfg["local_path"], map_location=device))
                print(f"[OK] Loaded weights from {cfg['local_path']}")
            else:
                print(f"[OK] Loaded {cfg['name']} with pretrained weights")
        except Exception as e:
            print(f"[ERROR] Failed to load {cfg['name']}: {e}")
            raise

    def forward(self, img_tensor, input_boxes=None, target_mask=None):
        img_tensor = img_tensor.to(self.device)

        out = self.model(img_tensor)['out']
        pred_masks = torch.sigmoid(out)

        if target_mask is not None:
            target_mask = target_mask.to(self.device)
            if target_mask.shape != pred_masks.shape:
                target_mask = F.interpolate(
                    target_mask,
                    size=pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
            loss = F.binary_cross_entropy(pred_masks, target_mask)
            return pred_masks, loss

        return pred_masks

# ---------------------------- SAM-Med2D ----------------------------

def create_official_args(model_type, ckpt_path, use_adapter, device):
    """Fully replicate official argument creation method."""
    import argparse

    # Fully replicate official parse_args function
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="sammed", help="run model name")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument("--data_path", type=str, default="data_demo", help="train data path")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument("--model_type", type=str, default=model_type, help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default=ckpt_path, help="sam checkpoint")
    parser.add_argument("--boxes_prompt", type=bool, default=True, help="use boxes prompt")
    parser.add_argument("--point_num", type=int, default=1, help="point num")
    parser.add_argument("--iter_point", type=int, default=1, help="iter num")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=use_adapter, help="use adapter")
    parser.add_argument("--prompt_path", type=str, default=None, help="fix prompt path")
    parser.add_argument("--save_pred", type=bool, default=False, help="save reslut")

    # Parse empty command line arguments, use all defaults
    args = parser.parse_args([])

    # Apply official post-processing logic
    if args.iter_point > 1:
        args.iter_point = 1

    return args

class SAMMed2DModel(SegmentationModelBase):
    def __init__(self, device, cfg):
        super().__init__(device, cfg)
        from segment_anything import sam_model_registry
        from segment_anything.utils.transforms import ResizeLongestSide

        # Load configuration
        model_type = cfg.get("model_type", "vit_b")
        ckpt_path = cfg.get("local_path", "./pretrain_model/sam-med2d_b.pth")
        image_size = cfg.get("image_size", 256)
        use_adapter = cfg.get("encoder_adapter", True)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[ERROR] SAM-Med2D weights not found: {ckpt_path}")

        # Create args (consistent with standalone test script)
        args = argparse.Namespace(
            image_size=image_size,
            encoder_adapter=use_adapter,
            sam_checkpoint=ckpt_path,
            model_type=model_type,
            device=device
        )

        # Load model
        self.model = sam_model_registry[model_type](args).to(device)
        self.model.eval()  # Evaluation mode, but will temporarily switch to training mode during adversarial attack

        # Initialize official transformer (for bbox coordinate scaling)
        self.transform = ResizeLongestSide(image_size)
        self.image_size = image_size

        # [Fix 1] Ensure pixel_mean/std are initialized on correct device
        pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(-1, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(-1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

        print(f"[OK] Successfully loaded SAM-Med2D: {model_type}, image size: {image_size}")
        print(f"[OK] Registered correct normalization parameters, consistent with standalone test script")

    def forward(self, img_tensor, input_boxes=None, target_mask=None):
        """
        Pure PyTorch implementation, correctly handles normalization, maintains gradient flow for adversarial attacks.
        """
        # [Fix 2] Ensure all tensors are on the same device
        img_tensor = img_tensor.to(self.device)
        if target_mask is not None:
            target_mask = target_mask.to(self.device)

        # Get dimensions
        batch_size, _, H, W = img_tensor.shape
        original_size = (H, W)

        # ============================================================
        # Phase 1: Image preprocessing
        # ============================================================
        # Scaling logic
        scale = self.image_size / max(original_size)
        new_h, new_w = int(H * scale), int(W * scale)
        resized_tensor = F.interpolate(
            img_tensor,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False
        )

        # Padding logic
        h_pad = self.image_size - new_h
        w_pad = self.image_size - new_w
        input_image_torch = F.pad(resized_tensor, (0, w_pad, 0, h_pad, 0, 0), value=0)

        # [Key] Apply correct normalization
        # Convert [0,1] range to [0,255], then normalize
        if input_image_torch.max() <= 1.0:
            input_image_torch = input_image_torch * 255.0

        # [Fix 3] Ensure pixel_mean/std are on correct device
        input_image_torch = (input_image_torch - self.pixel_mean) / self.pixel_std

        # Image encoding
        image_embedding = self.model.image_encoder(input_image_torch)

        # ============================================================
        # Phase 2: Prompt processing
        # ============================================================
        # [Key] bbox coordinate transformation must use official transform.apply_boxes()
        box_tensor = None
        if input_boxes is not None:
            # Convert to numpy (because apply_boxes expects numpy input)
            if isinstance(input_boxes, (list, tuple, np.ndarray)):
                box_np = np.array(input_boxes).reshape(1, 4)
            else:  # tensor
                box_np = input_boxes.detach().cpu().numpy().reshape(1, 4)
            # Use official transformation (exact reproduction)
            box_tf = self.transform.apply_boxes(box_np, original_size)
            box_tensor = torch.tensor(box_tf, device=self.device).unsqueeze(0).float()

        # Prompt encoding
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None,
            boxes=box_tensor,
            masks=None
        )

        # ============================================================
        # Phase 3: Mask decoding and post-processing
        # ============================================================
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Post-processing: interpolate logits to original size first, then apply sigmoid
        # Keep logits for loss computation (BCE_with_logits is more stable, supports autocast)
        low_res_masks_resized = F.interpolate(
            low_res_masks,
            size=original_size,
            mode="bilinear",
            align_corners=False
        )

        # Apply sigmoid to get final prediction
        pred_masks = torch.sigmoid(low_res_masks_resized)

        # ============================================================
        # Phase 4: Adversarial attack loss computation (using logits, supports mixed precision)
        # ============================================================
        if target_mask is not None:
            if target_mask.shape != pred_masks.shape:
                target_mask = F.interpolate(
                    target_mask,
                    size=pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
            # Use binary_cross_entropy_with_logits (autocast safe)
            loss = F.binary_cross_entropy_with_logits(low_res_masks_resized, target_mask)
            return pred_masks, loss

        return pred_masks

    def predict(self, img_tensor, input_boxes=None):
        """Inference mode (maintain interface consistency)."""
        with torch.no_grad():
            return self.forward(img_tensor, input_boxes)

# ---------------------------- Model Factory ----------------------------
MODEL_REGISTRY = {
    "medsam": MedSAMModel,
    "sam": SAMModel,
  	"sammed2d": SAMMed2DModel,
    "sam_point": SAMPointModel,
    "unet": UNetModel,
    "deeplab": DeepLabV3Model,
}

def create_model(model_name: str, device: str, model_cfg: dict) -> SegmentationModelBase:
    """Model factory function."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}, supported models: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_name]
    return model_class(device, model_cfg)

def list_available_models():
    """List all available models."""
    return list(MODEL_REGISTRY.keys())
