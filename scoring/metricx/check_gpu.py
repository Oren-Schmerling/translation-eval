#!/usr/bin/env python3
"""Pre-flight diagnostic check for GPU and bfloat16 capability."""
import argparse
import sys

def main() -> None:
    parser = argparse.ArgumentParser(description="Check GPU and bfloat16 capability.")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit non-zero if CUDA or bfloat16 is unavailable.",
    )
    args = parser.parse_args()

    print(f"Python Version: {sys.version.split()[0]}")

    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except ImportError:
        torch_version = "NOT_INSTALLED"
        cuda_available = False

    print(f"PyTorch Version: {torch_version}")
    print(f"torch.cuda.is_available(): {cuda_available}")

    bf16_supported = False
    device_name = "N/A"
    total_vram_gb = 0.0

    if cuda_available:
        try:
            import torch
            bf16_supported = torch.cuda.is_bf16_supported()
            device_name = torch.cuda.get_device_name(0)
            total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
            total_vram_gb = total_vram_bytes / (1024 ** 3)
        except Exception as e:
            print(f"Warning probing CUDA device: {e}")

    print(f"torch.cuda.is_bf16_supported(): {bf16_supported}")
    print(f"GPU Device Name: {device_name}")
    print(f"Total VRAM: {total_vram_gb:.2f} GB")

    if cuda_available and bf16_supported:
        print("MODE=CUDA_BF16")
    else:
        print("MODE=CPU_ONLY")

    if args.require_cuda:
        if not cuda_available:
            print("ERROR: --require-cuda specified but torch.cuda.is_available() is False.", file=sys.stderr)
            sys.exit(1)
        if not bf16_supported:
            print("ERROR: --require-cuda specified but torch.cuda.is_bf16_supported() is False.", file=sys.stderr)
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
