#!/usr/bin/env python3
"""
Torch initialization wrapper to handle PyTorch 2.6 security changes
Run this before importing any YOLO models
"""

import torch.serialization

# Add safe globals for ultralytics before any model loading occurs
torch.serialization.add_safe_globals([
    'ultralytics.nn.tasks.DetectionModel',
    'ultralytics.models.yolo.detect.DetectionPredictor', 
    'ultralytics.models.yolo.detect.DetectionValidator',
    'ultralytics.models.yolo.detect.DetectionTrainer',
    'ultralytics.nn.modules.block.C2f',
    'ultralytics.nn.modules.conv.Conv',
    'ultralytics.nn.modules.head.Detect'
])

print("✅ PyTorch safe globals configured for ultralytics")

# Now import and run the main application
if __name__ == "__main__":
    import main  # This will import your FastAPI app
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")