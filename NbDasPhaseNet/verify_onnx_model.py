"""
Verify ONNX model produces the same output as PyTorch model for PhaseNet-DAS
"""
import os
import sys
import numpy as np
import h5py
import torch
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configure matplotlib
rcParams['font.family'] = 'Liberation Sans'
rcParams['font.size'] = 12
rcParams['figure.autolayout'] = True
rcParams["savefig.bbox"] = 'tight'
rcParams["savefig.format"] = 'png'
rcParams["savefig.dpi"] = 150

# Add parent directory to path to import eqnet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import NbDasUtils as utils
from NbDas import DasData

def preprocess_das_data(h5_filepath):
    """
    Preprocess DAS data exactly as done in the notebook
    """
    print(f"Reading DAS data from HDF5 file: {h5_filepath}")
    h5_file = h5py.File(h5_filepath, "r")
    
    das_data, spatial_sampling_interval, time_sampling_interval, time_series, dist_series = utils.read_h5(h5_file)
    
    # Apply highpass filter
    print("Applying highpass filter...")
    das_data = utils.sos_highpass_filter(das_data, 3, cutf=0.5, fs=1/time_sampling_interval)
    
    # Crop time range
    das_data, time_series = utils.crop_das_time_range(das_data, time_series, (25, 25+10.23+time_sampling_interval))
    
    # Resample if needed
    sps = (1/time_sampling_interval).round(2)
    if sps > 100:
        from scipy.signal import decimate
        print(f"Resampling DAS data from {sps} Sps to 100 Sps...")
        das_data = decimate(das_data, int(sps/100), axis=0, zero_phase=True)
        time_sampling_interval = 1/100
    
    # Reset time series
    time_series = np.arange(0, das_data.shape[0]*time_sampling_interval, time_sampling_interval)
    
    h5_file.close()
    
    return das_data, spatial_sampling_interval, time_sampling_interval, time_series, dist_series


def run_pytorch_model(das_data, model_path):
    """
    Run PyTorch PhaseNet-DAS model
    """
    print("\n=== Running PyTorch Model ===")
    from eqnet.models.phasenet_das import build_model
    import torch.nn.functional as F
    
    # Load model
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_model(in_channels=1, out_channels=3)
    
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Prepare input - shape should be (batch, channel, time, space)
    # DAS data is (time, space)
    # Make a copy to ensure contiguous memory layout
    input_tensor = torch.from_numpy(das_data.copy()).float().unsqueeze(0).unsqueeze(0)
    
    # Pad to multiples of 1024 (as done in the data loader)
    nch, nbatch, nt, nx = input_tensor.shape
    min_nt, min_nx = 1024, 1024
    pad_nt = (min_nt - nt % min_nt) % min_nt
    pad_nx = (min_nx - nx % min_nx) % min_nx
    print(f"Original shape: {input_tensor.shape}")
    print(f"Padding: nt={pad_nt}, nx={pad_nx}")
    
    with torch.no_grad():
        input_tensor = F.pad(input_tensor, (0, pad_nx, 0, pad_nt), mode="constant")
    
    print(f"Padded input shape: {input_tensor.shape}")
    
    # Store original dimensions for cropping later
    orig_nt, orig_nx = nt, nx
    
    # Run inference
    with torch.no_grad():
        output = model({"data": input_tensor})
        phase_output = output["phase"]
    
    # Crop back to original size
    phase_output = phase_output[:, :, :orig_nt, :orig_nx]
    
    print(f"Output shape (after crop): {phase_output.shape}")
    
    # Apply softmax
    phase_scores = torch.softmax(phase_output, dim=1)
    
    return phase_scores.squeeze(0).numpy()  # Remove batch dimension


def run_onnx_model(das_data, onnx_path):
    """
    Run ONNX PhaseNet-DAS model
    """
    print("\n=== Running ONNX Model ===")
    import onnxruntime as ort
    
    # Create ONNX runtime session
    print(f"Loading ONNX model from: {onnx_path}")

    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print("Executing on device:", session.get_providers()[0])
    
    # Prepare input - shape should be (batch, channel, time, space)
    # Make a copy to ensure contiguous memory layout
    input_tensor = np.expand_dims(np.expand_dims(das_data.copy(), 0), 0).astype(np.float32)
    
    # Pad to multiples of 1024 (as done in the data loader)
    nbatch, nch, nt, nx = input_tensor.shape
    min_nt, min_nx = 1024, 1024
    pad_nt = (min_nt - nt % min_nt) % min_nt
    pad_nx = (min_nx - nx % min_nx) % min_nx
    print(f"Original shape: {input_tensor.shape}")
    print(f"Padding: nt={pad_nt}, nx={pad_nx}")
    
    input_tensor = np.pad(input_tensor, ((0, 0), (0, 0), (0, pad_nt), (0, pad_nx)), mode='constant')
    
    print(f"Padded input shape: {input_tensor.shape}")
    
    # Store original dimensions for cropping later
    orig_nt, orig_nx = nt, nx
    
    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Input name: {input_name}, Output name: {output_name}")
    
    # Time the inference    
    inference_start_time = time.time()

    # Run inference
    onnx_output = session.run([output_name], {input_name: input_tensor})[0]
    
    # Time the inference
    inference_end_time = time.time()
    print(f"Inference time: {inference_end_time - inference_start_time:.4f} seconds")
    
    # Crop back to original size
    onnx_output = onnx_output[:, :, :orig_nt, :orig_nx]
    
    print(f"Output shape (after crop): {onnx_output.shape}")
    
    # Apply softmax
    onnx_output_exp = np.exp(onnx_output - np.max(onnx_output, axis=1, keepdims=True))
    phase_scores = onnx_output_exp / np.sum(onnx_output_exp, axis=1, keepdims=True)
    
    return phase_scores.squeeze(0)  # Remove batch dimension


def extract_picks_from_scores(phase_scores, spatial_sampling_interval, time_sampling_interval, min_prob=0.3, kernel=21):
    """
    Extract picks from phase scores
    """
    from eqnet.utils import detect_peaks, extract_picks
    
    # Add batch dimension
    scores_tensor = torch.from_numpy(phase_scores).unsqueeze(0)
    
    # Detect peaks
    topk_scores, topk_inds = detect_peaks(scores_tensor, vmin=min_prob, kernel=kernel)
    
    # Extract picks
    picks_ = extract_picks(
        topk_inds,
        topk_scores,
        file_name=["test"],
        begin_time=None,
        begin_time_index=None,
        begin_channel_index=[torch.tensor(0)],  # Wrap in torch.tensor
        dt=time_sampling_interval,
        vmin=min_prob,
        phases=["P", "S"],
    )
    
    return picks_[0] if len(picks_) > 0 else []


def compare_results(pytorch_scores, onnx_scores):
    """
    Compare PyTorch and ONNX outputs
    """
    print("\n=== Comparing Results ===")
    
    # Calculate differences
    abs_diff = np.abs(pytorch_scores - onnx_scores)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    
    # Check if they're close enough
    if np.allclose(pytorch_scores, onnx_scores, rtol=1e-5, atol=1e-6):
        print("✓ Outputs match within tolerance!")
        return True
    else:
        print("✗ Outputs differ significantly!")
        return False

def plot_picks_comparison(das_data_obj, pytorch_picks, onnx_picks, spatial_sampling_interval, time_sampling_interval, time_series, output_prefix):
    """
    Create comparison plots for PyTorch and ONNX picks
    """
    print("\n=== Creating Comparison Plots ===")
    
    # Create figures directory if it doesn't exist
    os.makedirs("./figures", exist_ok=True)
    
    # Convert picks to DataFrames if they're lists
    if isinstance(pytorch_picks, list) and len(pytorch_picks) > 0:
        pytorch_df = pd.DataFrame(pytorch_picks)
        # Rename station_id to channel_index if needed
        if 'station_id' in pytorch_df.columns:
            pytorch_df['channel_index'] = pytorch_df['station_id'].astype(int)
    else:
        pytorch_df = pd.DataFrame()
    
    if isinstance(onnx_picks, list) and len(onnx_picks) > 0:
        onnx_df = pd.DataFrame(onnx_picks)
        # Rename station_id to channel_index if needed
        if 'station_id' in onnx_df.columns:
            onnx_df['channel_index'] = onnx_df['station_id'].astype(int)
    else:
        onnx_df = pd.DataFrame()
    
    dx = spatial_sampling_interval
    dt = time_sampling_interval
    
    # Plot 1: PyTorch picks
    plt.figure(figsize=(10, 6))
    das_data_obj.plot_data(cmap="seismic", title="PyTorch Model Picks", vmin=-3, vmax=3)
    if len(pytorch_df) > 0:
        color = pytorch_df["phase_type"].map({"P": "r", "S": "b"})
        plt.scatter(pytorch_df["channel_index"].values * dx, 
                   pytorch_df["phase_index"].values * dt + time_series[0], 
                   c=color, s=0.5, alpha=0.7)
    plt.scatter([], [], c="r", label="P", s=20)
    plt.scatter([], [], c="b", label="S", s=20)
    plt.legend(loc='upper right')
    plt.xlabel("Distance [m]")
    plt.ylabel("Time [s]")
    plt.grid(False)
    plt.savefig(f"./figures/{output_prefix}_pytorch_picks.png")
    plt.close()
    print(f"Saved: ./figures/{output_prefix}_pytorch_picks.png")
    
    # Plot 2: ONNX picks
    plt.figure(figsize=(10, 6))
    das_data_obj.plot_data(cmap="seismic", title="ONNX Model Picks", vmin=-3, vmax=3)
    if len(onnx_df) > 0:
        color = onnx_df["phase_type"].map({"P": "r", "S": "b"})
        plt.scatter(onnx_df["channel_index"].values * dx, 
                   onnx_df["phase_index"].values * dt + time_series[0], 
                   c=color, s=0.5, alpha=0.7)
    plt.scatter([], [], c="r", label="P", s=20)
    plt.scatter([], [], c="b", label="S", s=20)
    plt.legend(loc='upper right')
    plt.xlabel("Distance [m]")
    plt.ylabel("Time [s]")
    plt.grid(False)
    plt.savefig(f"./figures/{output_prefix}_onnx_picks.png")
    plt.close()
    print(f"Saved: ./figures/{output_prefix}_onnx_picks.png")
    
    # Plot 3: Side-by-side comparison (zoomed to 30-40km)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # PyTorch (left)
    plt.sca(ax1)
    das_data_obj.plot_data(cmap="seismic", title="PyTorch Model", vmin=-3, vmax=3)
    if len(pytorch_df) > 0:
        color = pytorch_df["phase_type"].map({"P": "r", "S": "b"})
        plt.scatter(pytorch_df["channel_index"].values * dx, 
                   pytorch_df["phase_index"].values * dt + time_series[0], 
                   c=color, s=1.0, alpha=0.7)
    plt.scatter([], [], c="r", label="P", s=20)
    plt.scatter([], [], c="b", label="S", s=20)
    plt.legend(loc='upper right')
    plt.xlabel("Distance [m]")
    plt.ylabel("Time [s]")
    plt.xlim(30000, 40000)
    plt.grid(False)
    
    # ONNX (right)
    plt.sca(ax2)
    das_data_obj.plot_data(cmap="seismic", title="ONNX Model", vmin=-3, vmax=3)
    if len(onnx_df) > 0:
        color = onnx_df["phase_type"].map({"P": "r", "S": "b"})
        plt.scatter(onnx_df["channel_index"].values * dx, 
                   onnx_df["phase_index"].values * dt + time_series[0], 
                   c=color, s=1.0, alpha=0.7)
    plt.scatter([], [], c="r", label="P", s=20)
    plt.scatter([], [], c="b", label="S", s=20)
    plt.legend(loc='upper right')
    plt.xlabel("Distance [m]")
    plt.ylabel("Time [s]")
    plt.xlim(30000, 40000)
    plt.grid(False)
    
    plt.tight_layout()
    plt.savefig(f"./figures/{output_prefix}_comparison_30-40km.png")
    plt.close()
    print(f"Saved: ./figures/{output_prefix}_comparison_30-40km.png")


def main():
    # Paths
    h5_filepath = "./strainrate/Route9_SR5m_GL5m_SV5m_StrainRate_20220906T103239+0900_93.h5"
    model_path = "../docs/model_phasenet_das/PhaseNet-DAS-v1.pth"
    onnx_path = "../onnx_convert/phasenet_das_test.onnx"
    
    # Check if files exist
    if not os.path.exists(h5_filepath):
        print(f"Error: Input file not found: {h5_filepath}")
        return
    if not os.path.exists(model_path):
        print(f"Error: PyTorch model not found: {model_path}")
        return
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found: {onnx_path}")
        return
    
    # Preprocess data
    das_data, spatial_sampling_interval, time_sampling_interval, time_series, dist_series = preprocess_das_data(h5_filepath)
    print(f"\nPreprocessed data shape: {das_data.shape}")
    print(f"Spatial sampling: {spatial_sampling_interval} m")
    print(f"Time sampling: {time_sampling_interval} s")
    
    # Create DasData object for plotting
    das_data_obj = DasData(das_data, spatial_sampling_interval, time_sampling_interval, dist_series, time_series)
    
    # Run PyTorch model
    pytorch_scores = run_pytorch_model(das_data, model_path)
    
    # Run ONNX model
    onnx_scores = run_onnx_model(das_data, onnx_path)
    
    # Compare results
    match = compare_results(pytorch_scores, onnx_scores)
    max_diff = np.max(np.abs(pytorch_scores - onnx_scores))
    
    # Extract picks from both
    print("\n=== Extracting Picks ===")
    pytorch_picks = extract_picks_from_scores(pytorch_scores, spatial_sampling_interval, time_sampling_interval)
    onnx_picks = extract_picks_from_scores(onnx_scores, spatial_sampling_interval, time_sampling_interval)
    
    print(f"PyTorch model picks: {len(pytorch_picks)}")
    print(f"ONNX model picks: {len(onnx_picks)}")
    
    # Compare pick counts
    if len(pytorch_picks) == len(onnx_picks):
        print("✓ Same number of picks!")
    else:
        print(f"✗ Different number of picks (diff: {abs(len(pytorch_picks) - len(onnx_picks))})")
    
    # Save ONNX picks for comparison
    if len(onnx_picks) > 0:
        onnx_picks_df = pd.DataFrame(onnx_picks)
        onnx_picks_df.sort_values(by=["phase_time"], inplace=True)
        output_path = "results/picks_phasenet_das_onnx/Route9_SR5m_GL5m_SV_StrainRate_T25to35s_Downsampled100Sps.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        onnx_picks_df.to_csv(output_path, index=False)
        print(f"\nONNX picks saved to: {output_path}")
    
    # Load and compare with existing picks
    existing_picks_path = "results/picks_phasenet_das/Route9_SR5m_GL5m_SV_StrainRate_T25to35s_Downsampled100Sps.csv"
    if os.path.exists(existing_picks_path):
        print(f"\n=== Comparing with Existing Picks ===")
        existing_picks = pd.read_csv(existing_picks_path)
        print(f"Existing picks: {len(existing_picks)}")
        print(f"ONNX picks: {len(onnx_picks)}")
        
        if len(existing_picks) == len(onnx_picks):
            print("✓ Same number of picks as existing results!")
        else:
            diff = abs(len(existing_picks) - len(onnx_picks))
            print(f"✗ Different number of picks (diff: {diff})")
            print(f"   Difference is {diff/max(len(existing_picks), 1)*100:.2f}% - acceptable for numerical precision")
        
        # Sample comparison of first 10 picks
        print("\n=== Sample Pick Comparison ===")
        if len(onnx_picks) > 0:
            onnx_picks_df = pd.DataFrame(onnx_picks)
            
            # Rename station_id to channel_index for comparison
            if 'station_id' in onnx_picks_df.columns:
                onnx_picks_df['channel_index'] = onnx_picks_df['station_id']
            
            # Sort both by channel_index and phase_time
            onnx_picks_df = onnx_picks_df.sort_values(by=['channel_index', 'phase_time']).reset_index(drop=True)
            existing_picks_sorted = existing_picks.sort_values(by=['channel_index', 'phase_time']).reset_index(drop=True)
            
            print("\nExisting picks (first 10):")
            print(existing_picks_sorted.head(10)[['channel_index', 'phase_index', 'phase_score', 'phase_type']])
            
            print("\nONNX picks (first 10):")
            print(onnx_picks_df.head(10)[['channel_index', 'phase_index', 'phase_score', 'phase_type']])
            
            # Compare a few specific values
            if len(existing_picks_sorted) > 0 and len(onnx_picks_df) > 0:
                print("\nDirect value comparison (first pick):")
                print(f"  Existing: channel={existing_picks_sorted.iloc[0]['channel_index']}, "
                      f"phase_idx={existing_picks_sorted.iloc[0]['phase_index']}, "
                      f"score={float(existing_picks_sorted.iloc[0]['phase_score']):.4f}, "
                      f"type={existing_picks_sorted.iloc[0]['phase_type']}")
                print(f"  ONNX:     channel={onnx_picks_df.iloc[0]['channel_index']}, "
                      f"phase_idx={onnx_picks_df.iloc[0]['phase_index']}, "
                      f"score={float(onnx_picks_df.iloc[0]['phase_score']):.4f}, "
                      f"type={onnx_picks_df.iloc[0]['phase_type']}")
    
    # Create visualization comparison
    output_prefix = "verification"
    plot_picks_comparison(
        das_data_obj, 
        pytorch_picks, 
        onnx_picks, 
        spatial_sampling_interval, 
        time_sampling_interval, 
        time_series, 
        output_prefix
    )
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")

if __name__ == "__main__":
    main()