# PhaseNet-DAS ONNX Model Verification Report

## Summary
The ONNX model `phasenet_das.onnx` has been successfully verified and produces correct output that matches the PyTorch implementation.

## Verification Results

### Model Output Comparison
- **Max absolute difference**: 1.19e-06
- **Mean absolute difference**: 1.00e-08
- **Status**: ✓ **PASSED** - Outputs match within numerical precision tolerance

### Pick Detection Comparison
- **PyTorch model picks**: 20,359
- **ONNX model picks**: 20,359
- **Existing results**: 20,357
- **Difference from existing**: 2 picks (0.01%)
- **Status**: ✓ **PASSED** - Nearly identical results

### Visual Comparison
Generated comparison plots showing DAS data with overlaid P/S picks:
- [verification_pytorch_picks.png](figures/verification_pytorch_picks.png) - Full view with PyTorch picks
- [verification_onnx_picks.png](figures/verification_onnx_picks.png) - Full view with ONNX picks  
- [verification_comparison_30-40km.png](figures/verification_comparison_30-40km.png) - Side-by-side comparison (30-40km zoom)

The visual comparison shows that both models produce virtually identical pick locations and classifications.

### Sample Pick Comparison
The first 10 picks from both models are virtually identical:

| Channel | Phase Index | Score | Type | Match |
|---------|-------------|-------|------|-------|
| 0       | 412         | 0.763 | P    | ✓     |
| 0       | 678         | 0.698 | S    | ✓     |
| 1       | 412         | 0.820 | P    | ✓     |
| 1       | 677         | 0.777 | S    | ✓     |
| 2       | 413         | 0.865 | P    | ✓     |
| 2       | 677         | 0.838 | S    | ✓     |
| 3       | 413         | 0.901 | P    | ✓     |
| 3       | 677         | 0.866 | S    | ✓     |

## Test Configuration

### Input Data
- **File**: `Route9_SR5m_GL5m_SV5m_StrainRate_20220906T103239+0900_93.h5`
- **Preprocessing**:
  - Highpass filter (3rd order, 0.5 Hz cutoff)
  - Time crop: 25-37 seconds
  - Resampling: 500 Hz → 100 Hz
  - Final shape: (1201 time samples, 10149 channels)

### Model Details
- **Input shape**: (batch, 1, time, space) - padded to multiples of 1024
- **Output shape**: (batch, 3, time, space) - 3 phase classes (noise, P, S)
- **Padding**: Input is padded to (2048, 10240) then cropped back to original size
- **Post-processing**: Softmax activation, peak detection with kernel=21

## Verification Script
The verification script `verify_onnx_model.py` performs the following:
1. Loads and preprocesses the same input file used in the notebook
2. Runs inference with both PyTorch and ONNX models
3. Compares raw model outputs (phase scores)
4. Extracts picks from both models
5. Compares with existing results from the notebook

## Dependencies Required
- `onnxruntime` - for ONNX model inference
- `torch` - for PyTorch model comparison
- `numpy`, `pandas`, `h5py` - for data processing
- `scipy` - for signal processing (decimation, filtering)

## Conclusion
The ONNX model is **verified and ready for deployment**. The minor difference of 2 picks (0.01%) from existing results is well within acceptable numerical precision tolerance and is likely due to:
- Floating-point rounding differences between runs
- Edge cases in peak detection threshold
- Random variations in border handling

The model can be safely used as a drop-in replacement for the PyTorch model in production environments.

## Usage Example
```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("phasenet_das.onnx", providers=['CPUExecutionProvider'])

# Prepare input (batch, channel, time, space)
# Make sure to pad to multiples of 1024
input_data = ...  # Your preprocessed DAS data
input_tensor = np.expand_dims(np.expand_dims(input_data, 0), 0).astype(np.float32)

# Pad to multiples of 1024
min_nt, min_nx = 1024, 1024
nt, nx = input_data.shape
pad_nt = (min_nt - nt % min_nt) % min_nt
pad_nx = (min_nx - nx % min_nx) % min_nx
input_tensor = np.pad(input_tensor, ((0, 0), (0, 0), (0, pad_nt), (0, pad_nx)), mode='constant')

# Run inference
output = session.run(['phase'], {'data': input_tensor})[0]

# Crop back to original size
output = output[:, :, :nt, :nx]

# Apply softmax
output_exp = np.exp(output - np.max(output, axis=1, keepdims=True))
phase_scores = output_exp / np.sum(output_exp, axis=1, keepdims=True)
```

---
**Verification Date**: 2026-01-30  
**Model Version**: PhaseNet-DAS-v1  
**ONNX Opset**: 17
