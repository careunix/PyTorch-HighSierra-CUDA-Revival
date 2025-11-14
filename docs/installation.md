# Installation Guide (macOS High Sierra + CUDA 10.2)

## Requirements
- macOS 10.13.6 High Sierra
- NVIDIA Web Drivers (387.x – 387.10.10.10.40.x)
- CUDA Toolkit 10.2 (last macOS release)
- cuDNN 7.6.5
- Python 3.8 (Conda recommended)

## Install the Wheel (Release Asset)

Option A — install directly from URL:
```
pip install "https://github.com/careunix/PyTorch-HighSierra-CUDA-Revival/releases/download/v0.1.0/torch-1.7.0a0-cp38-cp38-macosx_10_13_x86_64.whl"
```

Option B — download then install:
```
curl -L -o torch-1.7.0a0-cp38-cp38-macosx_10_13_x86_64.whl \
  https://github.com/careunix/PyTorch-HighSierra-CUDA-Revival/releases/download/v0.1.0/torch-1.7.0a0-cp38-cp38-macosx_10_13_x86_64.whl
pip install ./torch-1.7.0a0-cp38-cp38-macosx_10_13_x86_64.whl
```

## Verify Integrity & Signature

Expected SHA256:
```
38da4acfe780a041b1f73f67c66efcdb37e9773615446f6a02ed2586f3cff9c7
```

Check locally:
```
shasum -a 256 torch-1.7.0a0-cp38-cp38-macosx_10_13_x86_64.whl | awk '{print $1}'
```

Verify GPG signature:
```
# Import public key (from repo or raw)
gpg --import archive/keys/careunix.pub.asc
curl -L -o careunix.pub.asc \
  https://raw.githubusercontent.com/careunix/PyTorch-HighSierra-CUDA-Revival/main/archive/keys/careunix.pub.asc
gpg --import careunix.pub.asc

# Detached signature (raw)
curl -L -o SIGNATURE.asc \
  https://raw.githubusercontent.com/careunix/PyTorch-HighSierra-CUDA-Revival/main/wheel/SIGNATURE.asc
gpg --verify SIGNATURE.asc torch-1.7.0a0-cp38-cp38-macosx_10_13_x86_64.whl
```

## Verify

```
python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
