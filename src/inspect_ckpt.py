#!/usr/bin/env python3
"""
Checkpoint Inspector Tool
Quickly extracts and formats the Epoch, Accuracy, and Config from a PyTorch .pt file.

Usage:
    python inspect_ckpt.py path/to/best_model_tsm.pt
"""

import argparse
import pprint
from pathlib import Path

import torch
from omegaconf import OmegaConf

def inspect_checkpoint(ckpt_path: str):
    path = Path(ckpt_path).resolve()
    
    if not path.is_file():
        print(f"❌ Error: Checkpoint not found at {path}")
        return

    print(f"Loading {path.name} to CPU...")
    try:
        # map_location='cpu' ensures this script doesn't crash if your GPU is full
        ckpt = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"❌ Failed to load checkpoint. Error: {e}")
        return

    print("\n" + "="*50)
    print(f" 📊 CHECKPOINT METADATA: {path.name}")
    print("="*50)

    # 1. Extract Epoch
    epoch = ckpt.get("epoch", "Not Found")
    print(f"▶ Epoch: {epoch}")

    # 2. Extract Accuracy (Checking common keys used in your pipeline)
    val_acc = ckpt.get("val_accuracy", ckpt.get("best_acc", "Not Found"))
    if isinstance(val_acc, float):
        print(f"▶ Validation Accuracy: {val_acc * 100:.2f}%")
    else:
        print(f"▶ Validation Accuracy: {val_acc}")

    # 3. Extract and Format Config
    print("\n" + "="*50)
    print(" ⚙️  STORED CONFIGURATION")
    print("="*50)
    
    config = ckpt.get("config")
    if config is None:
        print("❌ No configuration dictionary found in this checkpoint.")
    else:
        try:
            # Attempt to parse and pretty-print using OmegaConf (Hydra)
            cfg = OmegaConf.create(config)
            print(OmegaConf.to_yaml(cfg))
        except Exception:
            # Fallback to standard Python pretty-print if it's a raw dict
            pprint.pprint(config)

    # Modify config.model.name if provided
    if False:
        print("\n" + "="*50)
        print(" 🔧 MODIFYING CONFIG")
        print("="*50)
        
        if config is None:
            print("❌ Cannot modify: No configuration found in checkpoint.")
        else:
            try:
                # Access and modify the model name
                if isinstance(config, dict):
                    if "model" not in config:
                        config["model"] = {}
                    if not isinstance(config["model"], dict):
                        config["model"] = {}
                    old_name = config["model"].get("name", "Not Found")
                    config["model"]["new_tsm"] = True
                    print(f"▶ Changed model.name from '{old_name}' to '{config['model']['name']}'")
                    
                    # Save the modified checkpoint
                    torch.save(ckpt, path)
                    print(f"✅ Checkpoint saved successfully to {path}")
                else:
                    print("❌ Config is not a dictionary format.")
            except Exception as e:
                print(f"❌ Failed to modify config. Error: {e}")

    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a trained PyTorch checkpoint.")
    parser.add_argument("ckpt_path", type=str, help="Path to the .pt checkpoint file")    
    args = parser.parse_args()
    inspect_checkpoint(args.ckpt_path)