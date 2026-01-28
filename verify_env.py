import os
import torch
import sys

def verify():
    print("Verifying environment and project structure...")
    
    # 1. Imports
    try:
        from models.encoder import get_encoder
        from models.classifier import get_brain_classifier
        from utils.dataset import BrainMRIDataset, KneeOADataset
        print("✅ Imports successful.")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return

    # 2. Dataset Paths
    brain_path = "datasets/brain_mri"
    knee_path = "datasets/knee_oa"
    
    if os.path.exists(brain_path):
        print(f"✅ Brain MRI dataset found at {brain_path}")
        try:
            ds = BrainMRIDataset(brain_path)
            print(f"   Found {len(ds)} images in Brain MRI dataset.")
        except Exception as e:
            print(f"   ⚠️ Could not load Brain MRI dataset: {e}")
    else:
        print(f"❌ Brain MRI dataset NOT found at {brain_path}")

    if os.path.exists(knee_path):
        print(f"✅ Knee OA dataset found at {knee_path}")
        try:
            ds = KneeOADataset(knee_path, split='train')
            print(f"   Found {len(ds)} images in Knee OA train dataset.")
        except Exception as e:
             print(f"   ⚠️ Could not load Knee OA dataset: {e}")
    else:
        print(f"❌ Knee OA dataset NOT found at {knee_path}")

    # 3. Model construction
    try:
        model = get_encoder()
        print("✅ ResNet-18 Encoder constructed successfully.")
    except Exception as e:
        print(f"❌ Model construction failed: {e}")

if __name__ == "__main__":
    verify()
