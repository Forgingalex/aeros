"""Export PyTorch model to ONNX format."""

import torch
import argparse
from pathlib import Path
from src.model.cnn import HeadingCNN


def export_onnx(
    checkpoint_path: Path,
    output_path: Path,
    input_size: tuple = (224, 224),
):
    """Export PyTorch model to ONNX.
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Path to save ONNX model
        input_size: Input image size (height, width)
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load model
    model = HeadingCNN()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    
    print(f"Exporting to ONNX format...")
    print(f"  Input shape: {dummy_input.shape}")
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    
    print(f"ONNX model saved to {output_path}")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification: PASSED")
    except ImportError:
        print("Warning: onnx package not installed, skipping verification")
    except Exception as e:
        print(f"Warning: ONNX verification failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/heading_model.onnx",
        help="Output path for ONNX model",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=224,
        help="Input image height",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=224,
        help="Input image width",
    )
    
    args = parser.parse_args()
    
    export_onnx(
        checkpoint_path=Path(args.checkpoint),
        output_path=Path(args.output),
        input_size=(args.input_height, args.input_width),
    )

