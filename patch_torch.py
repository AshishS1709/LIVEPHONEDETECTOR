"""
Monkey patch for PyTorch 2.6 to allow ultralytics globals
This file patches torch.load before any models are loaded
"""

import torch
import torch.serialization

def patch_torch_load():
    """Patch torch.load to handle ultralytics models safely"""
    
    # Check PyTorch version and patch accordingly
    pytorch_version = torch.__version__
    major, minor = pytorch_version.split('.')[:2]
    major, minor = int(major), int(minor)
    
    print(f"🔍 Detected PyTorch version: {pytorch_version}")
    
    # For PyTorch 2.6+ with weights_only security feature
    if major >= 2 and minor >= 6:
        # Try to use add_safe_globals if available
        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([
                'ultralytics.nn.tasks.DetectionModel',
                'ultralytics.models.yolo.detect.DetectionPredictor',
                'ultralytics.models.yolo.detect.DetectionValidator', 
                'ultralytics.models.yolo.detect.DetectionTrainer',
                'ultralytics.nn.modules.block.C2f',
                'ultralytics.nn.modules.conv.Conv',
                'ultralytics.nn.modules.head.Detect',
                'ultralytics.nn.modules.block.Bottleneck',
                'ultralytics.nn.modules.block.SPPF',
                'torch.nn.modules.upsampling.Upsample',
                'torch.nn.modules.pooling.MaxPool2d',
                'torch.nn.modules.activation.SiLU'
            ])
            print("✅ PyTorch safe globals added via add_safe_globals")
        else:
            # Fallback: Monkey patch torch.load to use weights_only=False
            original_load = torch.load
            
            def patched_load(*args, **kwargs):
                # Force weights_only=False for ultralytics models
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            torch.load = patched_load
            print("⚠️  PyTorch patched to use weights_only=False (fallback method)")
    
    else:
        # For older PyTorch versions, no patching needed
        print("ℹ️  PyTorch version doesn't require patching")

# Apply the patch
patch_torch_load()
print("🔧 PyTorch compatibility patch applied successfully")