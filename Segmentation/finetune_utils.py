#!/usr/bin/env python3
"""
Fine-tuning Utilities Module - Complete Fixed Version
Contains detailed diagnostic output to identify issues after running

[Fixes]
1. warmup_epochs changed from 1 to 0.1 (reduce warmup time)
2. learning_rate default changed from 1e-4 to 5e-4
3. SegmentationLoss added numerical stability handling

[New Features - LoRA+]
4. Added lora_plus_encoder and lora_plus_decoder strategies
5. LoRA+ uses different learning rates for A/B matrices
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class FinetuneStrategy(Enum):
    DECODER_ONLY = "decoder_only"
    DECODER_PROMPT = "decoder_prompt"
    ADAPTER_ONLY = "adapter_only"
    ENCODER_PARTIAL = "encoder_partial"
    FULL = "full"
    LORA = "lora"
    # New LoRA+ strategies
    LORA_PLUS_ENCODER = "lora_plus_encoder"  # LoRA+ applied to encoder, with regular training for decoder
    LORA_PLUS_DECODER = "lora_plus_decoder"  # LoRA+ applied to decoder

@dataclass
class FinetuneConfig:
    strategy: str = "decoder_only"
    learning_rate: float = 5e-4  # [Fix] increased from 1e-4 to 5e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    batch_size: int = 4
    encoder_lr_scale: float = 0.1
    unfreeze_encoder_layers: int = 4  # default unfreeze 4 layers
    lora_r: int = 8  # unified default value, consistent with argparse
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    # [New] LoRA+ specific parameters
    lora_plus_lr_ratio: float = 16.0  # B matrix lr = A matrix lr * ratio (LoRA+ paper recommends 16)
    warmup_epochs: float = 0.1  # [Fix] changed from 1 to 0.1, only warmup 10% of first epoch
    save_interval: int = 1
    eval_interval: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    output_dir: str = "./finetune_output"
    checkpoint_name: str = "finetuned_model.pth"
    
    def to_dict(self) -> Dict:
        return {
            "strategy": self.strategy,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "encoder_lr_scale": self.encoder_lr_scale,
            "unfreeze_encoder_layers": self.unfreeze_encoder_layers,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "lora_plus_lr_ratio": self.lora_plus_lr_ratio,  # [New]
            "warmup_epochs": self.warmup_epochs,
            "use_amp": self.use_amp,
            "output_dir": self.output_dir,
        }


def freeze_module(module: nn.Module):
    """Freeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module):
    """Unfreeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def print_trainable_parameters(model: nn.Module, name: str = "Model"):
    """Print trainable parameter statistics for a model."""
    trainable, total = count_parameters(model)
    pct = 100 * trainable / total if total > 0 else 0
    print(f"\n{'='*70}")
    print(f"[STATS] {name} Parameter Statistics:")
    print(f"   Trainable params: {trainable:,}")
    print(f"   Total params:     {total:,}")
    print(f"   Trainable ratio:  {pct:.2f}%")
    print(f"{'='*70}")


def diagnose_model_structure(model, model_name):
    """Diagnose model structure - key diagnostic function."""
    print(f"\n{'='*70}")
    print(f"[DIAG] Diagnosing {model_name} model structure")
    print(f"{'='*70}")

    # Get internal SAM model
    if hasattr(model, 'model'):
        sam = model.model
    else:
        sam = model

    print(f"\n[INFO] Top-level modules:")
    for name, child in sam.named_children():
        param_count = sum(p.numel() for p in child.parameters())
        print(f"   {name}: {type(child).__name__} ({param_count:,} params)")

    # Check encoder structure
    if hasattr(sam, 'vision_encoder'):
        encoder = sam.vision_encoder
        print(f"\n[INFO] vision_encoder structure:")
        for name, child in encoder.named_children():
            if isinstance(child, nn.ModuleList):
                print(f"   {name}: ModuleList (length={len(child)})")
            else:
                print(f"   {name}: {type(child).__name__}")

        # Check layers attribute
        print(f"\n[CHECK] Encoder layer attribute check:")
        for attr in ['layers', 'blocks', 'layer', 'encoder_layers']:
            if hasattr(encoder, attr):
                obj = getattr(encoder, attr)
                if isinstance(obj, nn.ModuleList):
                    print(f"   [OK] encoder.{attr} exists, length={len(obj)}")
                else:
                    print(f"   [WARN] encoder.{attr} exists but not ModuleList: {type(obj)}")
            else:
                print(f"   [NO] encoder.{attr} does not exist")


def diagnose_trainable_params(model, strategy):
    """Diagnose trainable parameter distribution."""
    print(f"\n{'='*70}")
    print(f"[DIAG] Diagnosing trainable params (strategy: {strategy})")
    print(f"{'='*70}")

    if hasattr(model, 'model'):
        sam = model.model
    else:
        sam = model

    # Statistics by module
    module_stats = {}
    for name, param in sam.named_parameters():
        parts = name.split('.')
        module = parts[0] if parts else 'other'

        if module not in module_stats:
            module_stats[module] = {'total': 0, 'trainable': 0}

        module_stats[module]['total'] += param.numel()
        if param.requires_grad:
            module_stats[module]['trainable'] += param.numel()

    print(f"\n[STATS] Trainable params by module:")
    for module, stats in sorted(module_stats.items()):
        pct = 100 * stats['trainable'] / stats['total'] if stats['total'] > 0 else 0
        status = "[OK]" if stats['trainable'] > 0 else "[NO]"
        print(f"   {status} {module}: {stats['trainable']:,} / {stats['total']:,} ({pct:.1f}%)")

    # Verify strategy is correctly executed
    print(f"\n[CHECK] Strategy verification:")

    if strategy == "decoder_only":
        if module_stats.get('mask_decoder', {}).get('trainable', 0) == 0:
            print(f"   [ERROR] mask_decoder should be trainable but is not!")
        else:
            print(f"   [OK] mask_decoder unfrozen")
        if module_stats.get('vision_encoder', {}).get('trainable', 0) > 0:
            print(f"   [WARN] vision_encoder should not be trained!")

    elif strategy == "decoder_prompt":
        if module_stats.get('mask_decoder', {}).get('trainable', 0) == 0:
            print(f"   [ERROR] mask_decoder should be trainable!")
        else:
            print(f"   [OK] mask_decoder unfrozen")
        if module_stats.get('prompt_encoder', {}).get('trainable', 0) == 0:
            print(f"   [ERROR] prompt_encoder should be trainable!")
        else:
            print(f"   [OK] prompt_encoder unfrozen")

    elif strategy == "encoder_partial":
        if module_stats.get('vision_encoder', {}).get('trainable', 0) == 0:
            print(f"   [ERROR] vision_encoder should be partially trainable!")
            print(f"   [WARN] encoder_partial is effectively the same as decoder_prompt!")
        else:
            print(f"   [OK] vision_encoder partially unfrozen")

    elif strategy == "full":
        total_trainable = sum(s['trainable'] for s in module_stats.values())
        total_params = sum(s['total'] for s in module_stats.values())
        if total_trainable < total_params * 0.99:
            print(f"   [WARN] full strategy but not all params are trainable")
        else:
            print(f"   [OK] All params unfrozen")

    elif strategy == "lora":
        # Check if there are lora parameters
        lora_params = sum(1 for n, p in sam.named_parameters() if 'lora' in n.lower() and p.requires_grad)
        if lora_params == 0:
            print(f"   [ERROR] No trainable LoRA params found!")
        else:
            print(f"   [OK] Found {lora_params} LoRA params")

    # [New] LoRA+ strategy verification
    elif strategy in ["lora_plus_encoder", "lora_plus_decoder"]:
        lora_params = sum(1 for n, p in sam.named_parameters() if 'lora' in n.lower() and p.requires_grad)
        if lora_params == 0:
            print(f"   [ERROR] No trainable LoRA params found!")
        else:
            print(f"   [OK] Found {lora_params} LoRA params")

        # Check if decoder is unfrozen (for lora_plus_encoder)
        if strategy == "lora_plus_encoder":
            if module_stats.get('mask_decoder', {}).get('trainable', 0) == 0:
                print(f"   [WARN] mask_decoder not unfrozen")
            else:
                print(f"   [OK] mask_decoder unfrozen (joint training)")

    return module_stats


# ========================== LoRA+ Implementation ==========================

def get_lora_target_modules_for_model(model_name: str, component: str = "encoder") -> List[str]:
    """
    Return correct LoRA target_modules based on model type.

    Args:
        model_name: Model name ('medsam' or 'sammed2d')
        component: 'encoder' or 'decoder'

    Returns:
        List of target_modules

    Note - SAM-Med2D qkv layer specifics:
        SAM-Med2D encoder's qkv layer is nn.Linear(dim, dim * 3), output dim is 3x input (Q/K/V merged).
        Adding LoRA to qkv layer adjusts Q, K, V projections together, unlike MedSAM where only Q and V are adjusted.
        This means:
        1. Parameter count is ~1.5x more than separate q_proj/v_proj (because K adjustment is included)
        2. Semantically it's a low-rank adjustment to the entire attention projection, not just Q/V
    """
    if model_name.lower() == "sammed2d":
        # SAM-Med2D uses segment_anything library
        # ViT encoder Attention uses merged qkv layer: self.qkv = nn.Linear(dim, dim * 3)
        # mask_decoder Attention uses: self.q_proj, self.k_proj, self.v_proj
        if component == "encoder":
            return ["qkv", "proj"]  # encoder uses merged qkv
        else:
            return ["q_proj", "v_proj", "out_proj"]  # decoder uses separate projections
    else:
        # MedSAM (HuggingFace transformers.SamModel)
        # vision_encoder and mask_decoder both use separate q_proj, v_proj
        return ["q_proj", "v_proj"]


class LoRAPlusLinear(nn.Module):
    """
    LoRA+ Linear layer implementation.

    Core innovation: A matrix and B matrix use different learning rates.
    B matrix lr = A matrix lr * lr_ratio (paper recommends lr_ratio=16)
    """
    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze original weights
        original_linear.weight.requires_grad = False
        if original_linear.bias is not None:
            original_linear.bias.requires_grad = False

        # LoRA matrices - these will be assigned different learning rates
        # lora_A: (r, in_features) - uses base learning rate
        # lora_B: (out_features, r) - uses lr * lr_ratio
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)  # B initialized to 0, ensures initial output unchanged

        # Dropout
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output
        result = self.original_linear(x)

        # LoRA increment: (x @ A^T) @ B^T * scaling
        lora_out = self.lora_dropout(x)
        lora_out = lora_out @ self.lora_A.T  # (batch, seq, r)
        lora_out = lora_out @ self.lora_B.T  # (batch, seq, out)
        lora_out = lora_out * self.scaling
        
        return result + lora_out


def _get_submodule(model, target_name: str):
    """Get submodule by dot-separated name."""
    parts = target_name.split('.')
    module = model
    for part in parts:
        module = getattr(module, part)
    return module


def _set_submodule(model, target_name: str, new_module: nn.Module):
    """Set submodule by dot-separated name."""
    parts = target_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _setup_lora_plus(model, config: FinetuneConfig, component: str, model_name: str):
    """
    Setup LoRA+ fine-tuning.

    Args:
        model: Model wrapper
        config: Fine-tuning configuration
        component: 'encoder' or 'decoder'
        model_name: Model name, used to auto-select target_modules
    """
    if hasattr(model, 'model'):
        sam = model.model
    else:
        sam = model

    # [Fix] Auto-select target_modules based on model type
    target_modules = get_lora_target_modules_for_model(model_name, component)

    print(f"\n{'='*70}")
    print(f"[SETUP] Setting up LoRA+ ({component}) - Model: {model_name}")
    print(f"   r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"   lr_ratio={config.lora_plus_lr_ratio} (B matrix lr multiplier)")
    print(f"   target_modules={target_modules} (auto-detected)")
    print(f"{'='*70}")

    # Determine search scope
    if component == "encoder":
        if hasattr(sam, 'vision_encoder'):
            search_root = sam.vision_encoder
            root_name = 'vision_encoder'
        elif hasattr(sam, 'image_encoder'):
            search_root = sam.image_encoder
            root_name = 'image_encoder'
        else:
            print(f"   [ERROR] Cannot find encoder module!")
            return
        search_roots = [(search_root, root_name)]
    elif component == "decoder":
        # [Fix] Only search in mask_decoder, prompt_encoder has basically no attention layers
        search_roots = []
        if hasattr(sam, 'mask_decoder'):
            search_roots.append((sam.mask_decoder, 'mask_decoder'))
        # Removed prompt_encoder search (it's mainly Embedding layers, no q_proj/v_proj)
    else:
        raise ValueError(f"Unknown component: {component}")

    applied_count = 0

    for search_root, root_name in search_roots:
        for name, module in search_root.named_modules():
            if isinstance(module, nn.Linear):
                # Check if it's a target module
                full_name = f"{root_name}.{name}" if name else root_name
                is_target = False
                for target in target_modules:
                    if target in name or name.endswith(target):
                        is_target = True
                        break

                if is_target:
                    # Create LoRA+ module
                    lora_module = LoRAPlusLinear(
                        original_linear=module,
                        r=config.lora_r,
                        alpha=config.lora_alpha,
                        dropout=config.lora_dropout
                    )

                    # Replace original module
                    _set_submodule(search_root, name, lora_module)

                    applied_count += 1
                    print(f"      [OK] {full_name}")

    if applied_count == 0:
        print(f"   [WARN] No target modules found!")
        print(f"   Check target_modules: {target_modules}")
    else:
        print(f"\n   [OK] LoRA+ applied to {applied_count} modules")

    # Count LoRA parameters
    lora_a_params = 0
    lora_b_params = 0
    for name, param in sam.named_parameters():
        if 'lora_A' in name and param.requires_grad:
            lora_a_params += param.numel()
        elif 'lora_B' in name and param.requires_grad:
            lora_b_params += param.numel()

    print(f"\n   [STATS] LoRA+ parameter statistics:")
    print(f"      lora_A params: {lora_a_params:,}")
    print(f"      lora_B params: {lora_b_params:,}")
    print(f"      Total LoRA params: {lora_a_params + lora_b_params:,}")


# ========================== MedSAM Fine-tuning ==========================

def setup_medsam_finetune(model, config: FinetuneConfig):
    """Setup MedSAM fine-tuning with complete diagnostics."""
    strategy = config.strategy
    sam = model.model  # transformers.SamModel

    print(f"\n{'='*70}")
    print(f"[SETUP] Setting up MedSAM fine-tuning")
    print(f"   Strategy: {strategy}")
    print(f"   unfreeze_encoder_layers: {config.unfreeze_encoder_layers}")
    print(f"{'='*70}")

    # Diagnose model structure
    diagnose_model_structure(model, "MedSAM")

    # Step 1: Freeze all parameters
    print(f"\n[Step 1] Freezing all parameters...")
    freeze_module(sam)
    t1, _ = count_parameters(sam)
    print(f"   Trainable params after freeze: {t1:,}")

    # Step 2: Unfreeze based on strategy
    print(f"\n[Step 2] Unfreezing params based on strategy '{strategy}'...")

    if strategy == "decoder_only":
        unfreeze_module(sam.mask_decoder)
        print(f"   Unfrozen: mask_decoder")

    elif strategy == "decoder_prompt":
        unfreeze_module(sam.mask_decoder)
        unfreeze_module(sam.prompt_encoder)
        print(f"   Unfrozen: mask_decoder, prompt_encoder")

    elif strategy == "encoder_partial":
        unfreeze_module(sam.mask_decoder)
        unfreeze_module(sam.prompt_encoder)
        print(f"   Unfrozen: mask_decoder, prompt_encoder")

        # Unfreeze last N encoder layers
        n_layers = config.unfreeze_encoder_layers
        encoder = sam.vision_encoder

        # Find encoder layers
        layers = None
        layer_attr = None
        for attr in ['layers', 'blocks', 'layer', 'encoder_layers']:
            if hasattr(encoder, attr):
                candidate = getattr(encoder, attr)
                if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
                    layers = candidate
                    layer_attr = attr
                    break

        if layers is not None:
            total = len(layers)
            if n_layers > 0:
                start_idx = max(0, total - n_layers)
                for i in range(start_idx, total):
                    unfreeze_module(layers[i])
                    print(f"   Unfrozen: encoder.{layer_attr}[{i}]")
                print(f"   [OK] Unfroze {n_layers}/{total} encoder layers")
            else:
                print(f"   [WARN] n_layers=0, encoder will not be unfrozen!")
        else:
            print(f"   [ERROR] Cannot find encoder layers!")
            print(f"   [WARN] encoder_partial will be equivalent to decoder_prompt!")

    elif strategy == "full":
        unfreeze_module(sam)
        print(f"   Unfrozen: all parameters")

    elif strategy == "lora":
        print(f"   Applying LoRA...")
        _setup_lora_medsam(model, config)

    # [New] LoRA+ encoder strategy
    elif strategy == "lora_plus_encoder":
        print(f"   Applying LoRA+ (encoder)...")
        _setup_lora_plus(model, config, component="encoder", model_name="medsam")
        # Also unfreeze decoder for regular training
        unfreeze_module(sam.mask_decoder)
        unfreeze_module(sam.prompt_encoder)
        print(f"   [OK] Also unfroze decoder and prompt_encoder for regular training")

    # [New] LoRA+ decoder strategy
    elif strategy == "lora_plus_decoder":
        print(f"   Applying LoRA+ (decoder)...")
        _setup_lora_plus(model, config, component="decoder", model_name="medsam")
        # [Fix] Also unfreeze prompt_encoder to increase trainable params
        unfreeze_module(sam.prompt_encoder)
        print(f"   [OK] Also unfroze prompt_encoder for regular training")

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Step 3: Diagnose results
    print(f"\n[Step 3] Verifying unfreeze results...")
    diagnose_trainable_params(model, strategy)

    # Print final statistics
    print_trainable_parameters(sam, "MedSAM")

    return model


def _setup_lora_medsam(model, config: FinetuneConfig):
    """Setup LoRA for MedSAM."""
    sam = model.model

    # Check if already a PEFT model to avoid duplicate wrapping
    try:
        from peft import PeftModel
        if isinstance(sam, PeftModel):
            print(f"   [WARN] Model is already a PEFT model, skipping duplicate wrapping")
            return
    except ImportError:
        pass

    try:
        from peft import get_peft_model, LoraConfig

        print(f"   Using peft library to setup LoRA...")
        print(f"   r={config.lora_r}, alpha={config.lora_alpha}")
        print(f"   target_modules={config.lora_target_modules}")

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
        )

        # Apply LoRA - this modifies model structure
        model.model = get_peft_model(sam, lora_config)
        print(f"   [OK] LoRA configured successfully")

        # Print LoRA parameters
        lora_params = [(n, p.numel()) for n, p in model.model.named_parameters()
                       if 'lora' in n.lower() and p.requires_grad]
        print(f"   LoRA parameter count: {len(lora_params)}")
        for name, count in lora_params[:5]:
            print(f"      - {name}: {count:,}")
        if len(lora_params) > 5:
            print(f"      ... and {len(lora_params)-5} more")

    except ImportError as e:
        print(f"   [WARN] peft library not installed: {e}")
        print(f"   Using manual LoRA implementation...")
        _apply_manual_lora(sam, config)
    except Exception as e:
        print(f"   [ERROR] LoRA setup failed: {e}")
        raise


def _apply_manual_lora(model, config):
    """Manual LoRA implementation."""
    count = 0

    # Get model device
    device = next(model.parameters()).device
    print(f"   Model device: {device}")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for target in config.lora_target_modules:
                if target in name:
                    in_f, out_f = module.in_features, module.out_features

                    # Freeze original weights
                    module.weight.requires_grad = False
                    if module.bias is not None:
                        module.bias.requires_grad = False

                    # Add LoRA parameters - create on correct device
                    lora_A = nn.Parameter(torch.zeros(config.lora_r, in_f, device=device))
                    lora_B = nn.Parameter(torch.zeros(out_f, config.lora_r, device=device))
                    nn.init.kaiming_uniform_(lora_A, a=5**0.5)
                    nn.init.zeros_(lora_B)

                    module.register_parameter('lora_A', lora_A)
                    module.register_parameter('lora_B', lora_B)
                    module.scaling = config.lora_alpha / config.lora_r

                    # Modify forward - ensure device consistency
                    original_forward = module.forward
                    def new_forward(x, orig_fwd=original_forward, mod=module):
                        result = orig_fwd(x)
                        # Ensure LoRA params are on same device as input
                        lora_A = mod.lora_A.to(x.device)
                        lora_B = mod.lora_B.to(x.device)
                        lora_out = (x @ lora_A.T @ lora_B.T) * mod.scaling
                        return result + lora_out
                    module.forward = new_forward

                    count += 1
                    print(f"      Applied LoRA: {name}")
                    break

    print(f"   Manual LoRA applied to {count} layers")


# ========================== SAM-Med2D Fine-tuning ==========================

def setup_sammed2d_finetune(model, config: FinetuneConfig):
    """Setup SAM-Med2D fine-tuning."""
    strategy = config.strategy
    sam = model.model

    print(f"\n{'='*70}")
    print(f"[SETUP] Setting up SAM-Med2D fine-tuning")
    print(f"   Strategy: {strategy}")
    print(f"{'='*70}")

    diagnose_model_structure(model, "SAM-Med2D")

    freeze_module(sam)

    if strategy == "decoder_only":
        unfreeze_module(sam.mask_decoder)

    elif strategy == "decoder_prompt":
        unfreeze_module(sam.mask_decoder)
        unfreeze_module(sam.prompt_encoder)

    elif strategy == "adapter_only":
        unfreeze_module(sam.mask_decoder)
        unfreeze_module(sam.prompt_encoder)

        adapter_count = 0
        for name, param in sam.image_encoder.named_parameters():
            if 'adapter' in name.lower() or 'Adapter' in name:
                param.requires_grad = True
                adapter_count += 1
        print(f"   Unfroze {adapter_count} adapter parameters")

    elif strategy == "encoder_partial":
        unfreeze_module(sam.mask_decoder)
        unfreeze_module(sam.prompt_encoder)

        n_layers = config.unfreeze_encoder_layers
        if hasattr(sam.image_encoder, 'blocks') and n_layers > 0:
            total = len(sam.image_encoder.blocks)
            for i in range(total - n_layers, total):
                unfreeze_module(sam.image_encoder.blocks[i])

        for name, param in sam.image_encoder.named_parameters():
            if 'adapter' in name.lower():
                param.requires_grad = True

    elif strategy == "full":
        unfreeze_module(sam)

    elif strategy == "lora":
        # [Fix] SAM-Med2D needs different target_modules
        config.lora_target_modules = get_lora_target_modules_for_model("sammed2d", "encoder")
        print(f"   SAM-Med2D LoRA target_modules: {config.lora_target_modules}")
        _setup_lora_medsam(model, config)

    # [New] LoRA+ encoder strategy
    elif strategy == "lora_plus_encoder":
        print(f"   Applying LoRA+ (encoder)...")
        _setup_lora_plus(model, config, component="encoder", model_name="sammed2d")
        # Also unfreeze decoder for regular training
        unfreeze_module(sam.mask_decoder)
        unfreeze_module(sam.prompt_encoder)
        print(f"   [OK] Also unfroze decoder and prompt_encoder for regular training")

    # [New] LoRA+ decoder strategy
    elif strategy == "lora_plus_decoder":
        print(f"   Applying LoRA+ (decoder)...")
        _setup_lora_plus(model, config, component="decoder", model_name="sammed2d")
        # [Fix] Also unfreeze prompt_encoder to increase trainable params
        unfreeze_module(sam.prompt_encoder)
        print(f"   [OK] Also unfroze prompt_encoder for regular training")

    diagnose_trainable_params(model, strategy)
    print_trainable_parameters(sam, "SAM-Med2D")

    return model


# ========================== Entry Function ==========================

def setup_finetune(model, model_name: str, config: FinetuneConfig):
    """Unified entry point for fine-tuning setup."""
    print(f"\n{'#'*70}")
    print(f"# Fine-tuning Setup Start")
    print(f"# Model: {model_name}")
    print(f"# Strategy: {config.strategy}")
    print(f"# Learning rate: {config.learning_rate}")
    print(f"# warmup_epochs: {config.warmup_epochs}")
    print(f"# unfreeze_encoder_layers: {config.unfreeze_encoder_layers}")
    # [New] Print LoRA+ parameters
    if config.strategy.startswith("lora"):
        print(f"# LoRA r: {config.lora_r}")
        print(f"# LoRA alpha: {config.lora_alpha}")
    if config.strategy.startswith("lora_plus"):
        print(f"# LoRA+ lr_ratio: {config.lora_plus_lr_ratio}")
    print(f"{'#'*70}")

    if model_name.lower() == "medsam":
        result = setup_medsam_finetune(model, config)
    elif model_name.lower() == "sammed2d":
        result = setup_sammed2d_finetune(model, config)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    print(f"\n{'#'*70}")
    print(f"# Fine-tuning Setup Complete")
    print(f"{'#'*70}\n")

    return result


# ========================== Optimizer ==========================

def build_optimizer(model, config: FinetuneConfig) -> torch.optim.Optimizer:
    """
    Build optimizer.

    [New] For LoRA+ strategies, A and B matrices use different learning rates:
    - lora_A: base learning rate lr
    - lora_B: lr * lora_plus_lr_ratio (paper recommends 16x)
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    print(f"\n[STATS] Building optimizer:")
    print(f"   Trainable param tensors: {len(trainable_params)}")

    if len(trainable_params) == 0:
        print(f"   [ERROR] No trainable parameters!")
        raise RuntimeError("No trainable parameters! Check freeze/unfreeze logic")

    total_params = sum(p.numel() for p in trainable_params)
    print(f"   Total trainable params: {total_params:,}")

    # [New] LoRA+ mode: separate A and B matrix parameter groups
    if config.strategy.startswith("lora_plus"):
        lora_a_params = []
        lora_b_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if 'lora_A' in name:
                lora_a_params.append(param)
            elif 'lora_B' in name:
                lora_b_params.append(param)
            else:
                other_params.append(param)

        param_groups = []

        if lora_a_params:
            param_groups.append({
                "params": lora_a_params,
                "lr": config.learning_rate,
                "name": "lora_A"
            })
            print(f"   lora_A group: {len(lora_a_params)} tensors, lr={config.learning_rate}")

        if lora_b_params:
            lora_b_lr = config.learning_rate * config.lora_plus_lr_ratio
            param_groups.append({
                "params": lora_b_params,
                "lr": lora_b_lr,
                "name": "lora_B"
            })
            print(f"   lora_B group: {len(lora_b_params)} tensors, lr={lora_b_lr} ({config.lora_plus_lr_ratio}x)")

        if other_params:
            param_groups.append({
                "params": other_params,
                "lr": config.learning_rate,
                "name": "other"
            })
            print(f"   other group: {len(other_params)} tensors, lr={config.learning_rate}")

        if len(param_groups) == 0:
            raise RuntimeError("No trainable parameters! Check freeze/unfreeze logic")

    # Layer-wise learning rate (original logic)
    elif config.strategy in ["encoder_partial", "full"]:
        encoder_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "encoder" in name.lower():
                encoder_params.append(param)
            else:
                other_params.append(param)

        param_groups = []
        if encoder_params:
            param_groups.append({
                "params": encoder_params,
                "lr": config.learning_rate * config.encoder_lr_scale,
                "name": "encoder"
            })
            print(f"   encoder group: {len(encoder_params)} tensors, lr={config.learning_rate * config.encoder_lr_scale}")
        if other_params:
            param_groups.append({
                "params": other_params,
                "lr": config.learning_rate,
                "name": "other"
            })
            print(f"   other group: {len(other_params)} tensors, lr={config.learning_rate}")
    else:
        param_groups = [{"params": trainable_params, "lr": config.learning_rate}]
        print(f"   Unified learning rate: {config.learning_rate}")

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    return optimizer


def build_scheduler(optimizer, config: FinetuneConfig, steps_per_epoch: int):
    """Build learning rate scheduler."""
    total_steps = config.num_epochs * steps_per_epoch
    warmup_steps = int(config.warmup_epochs * steps_per_epoch)  # [Fix] support fractional warmup_epochs

    print(f"\n[STATS] Building learning rate scheduler:")
    print(f"   Total steps: {total_steps}")
    print(f"   Warmup steps: {warmup_steps} ({config.warmup_epochs} epochs)")

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)  # [Fix] start from step+1 to avoid lr=0
        else:
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


# ========================== Loss Function ==========================

class SegmentationLoss(nn.Module):
    """BCE + Dice loss - [Fix] supports AMP mixed precision training."""
    def __init__(self, bce_weight: float = 0.3, dice_weight: float = 0.7):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # [Fix] Exit autocast before computing BCE to ensure numerical stability
        with torch.amp.autocast('cuda', enabled=False):
            # Convert to float32 for precision
            pred = pred.float()
            target = target.float()

            # Ensure predictions are in valid range to avoid log(0)
            pred = torch.clamp(pred, min=1e-7, max=1-1e-7)

            # BCE Loss
            bce_loss = F.binary_cross_entropy(pred, target, reduction='mean')

            # Dice Loss
            smooth = 1e-5
            pred_flat = pred.view(-1)
            target_flat = target.view(-1)
            intersection = (pred_flat * target_flat).sum()
            dice_loss = 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

            total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return total_loss


# ========================== Model Save/Load ==========================

def save_finetuned_model(model, config: FinetuneConfig, epoch: int, optimizer=None, metrics=None):
    """Save fine-tuned model checkpoint."""
    os.makedirs(config.output_dir, exist_ok=True)
    save_path = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch}.pth")

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if metrics is not None:
        checkpoint["metrics"] = metrics

    torch.save(checkpoint, save_path)
    print(f"[OK] Model saved: {save_path}")
    return save_path


def load_finetuned_model(model, checkpoint_path: str, strict: bool = True):
    """Load fine-tuned model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    print(f"[OK] Loaded checkpoint: {checkpoint_path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    if "metrics" in checkpoint:
        print(f"   Metrics: {checkpoint['metrics']}")

    return model, checkpoint


# ========================== Strategy Recommendation ==========================

def recommend_strategy(dataset_size: int, gpu_memory_gb: float, target_domain: str = "medical") -> str:
    """Recommend fine-tuning strategy based on dataset size and GPU memory."""
    print(f"\n[STATS] Analyzing fine-tuning conditions:")
    print(f"   Dataset size: {dataset_size}")
    print(f"   GPU memory: {gpu_memory_gb} GB")

    if dataset_size < 100:
        strategy = "decoder_only"
    elif dataset_size < 500:
        strategy = "decoder_prompt"
    elif dataset_size < 2000:
        strategy = "encoder_partial"
    else:
        strategy = "full" if gpu_memory_gb >= 24 else "encoder_partial"

    print(f"   Recommended strategy: {strategy}")
    return strategy
