import cv2
import numpy as np
import os
import argparse
import glob
import time # For basic benchmarking

# --- Optical Flow Parameters ---
# For Lucas-Kanade (Sparse) - CPU
lk_params_cpu = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def estimate_water_velocity_gpu(video_path, meters_per_pixel, roi=None, use_dense_flow=False, skip_frames=0, max_frames_to_process=None):
    """
    Estimates water velocity from a video using GPU-accelerated optical flow if available.
    """
    # Check for CUDA availability
    cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if cuda_available:
        print(f"CUDA is available. Using GPU for {os.path.basename(video_path)}.")
    else:
        print(f"CUDA not available or OpenCV not built with CUDA. Falling back to CPU for {os.path.basename(video_path)}.")
        # Optionally, you could call the original CPU-only function here
        # return estimate_water_velocity_cpu(...) # Assuming you have a CPU version

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Error: Could not get FPS for {video_path}. Assuming 30 FPS.")
        fps = 30.0

    for _ in range(skip_frames):
        ret, _ = cap.read()
        if not ret:
            print(f"Error: Not enough frames to skip in {video_path}")
            cap.release()
            return None

    ret, prev_frame_cpu = cap.read()
    if not ret:
        print(f"Error: Could not read the first frame from {video_path} after skipping.")
        cap.release()
        return None

    if roi:
        x, y, w, h = roi
        prev_frame_roi_cpu = prev_frame_cpu[y:y+h, x:x+w]
    else:
        prev_frame_roi_cpu = prev_frame_cpu

    prev_gray_cpu = cv2.cvtColor(prev_frame_roi_cpu, cv2.COLOR_BGR2GRAY)

    # --- GPU Specific Initialization ---
    if cuda_available:
        prev_gray_gpu = cv2.cuda_GpuMat()
        prev_gray_gpu.upload(prev_gray_cpu)

        if use_dense_flow:
            # Farneback on GPU
            # You can also use TVL1 optical flow: cv2.cuda.OpticalFlowDual_TVL1_create()
            # Farneback is generally faster but TVL1 can be more accurate.
            gpu_flow_calculator = cv2.cuda_FarnebackOpticalFlow.create(
                numLevels=5, pyrScale=0.5, fastPyramids=False, winSize=15,
                numIters=3, polyN=5, polySigma=1.2, flags=0)
        else:
            # Lucas-Kanade on GPU
            # goodFeaturesToTrack is usually run on CPU, then points uploaded
            p0_cpu = cv2.goodFeaturesToTrack(prev_gray_cpu, mask=None, maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7)
            if p0_cpu is None or len(p0_cpu) == 0:
                print(f"Warning: No good features to track (GPU) in {video_path}. Try dense flow or adjust ROI/params.")
                cap.release()
                return None
            p0_gpu = cv2.cuda_GpuMat()
            p0_gpu.upload(p0_cpu.astype(np.float32)) # LK on GPU expects float32 points

            gpu_sparse_flow = cv2.cuda.SparsePyrLKOpticalFlow_create(
                winSize=(15, 15), maxLevel=2, iters=10 # criteria (eps, count) is simplified to iters
            )
    else: # CPU Path
        if not use_dense_flow:
            p0_cpu = cv2.goodFeaturesToTrack(prev_gray_cpu, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            if p0_cpu is None or len(p0_cpu) == 0:
                print(f"Warning: No good features to track (CPU) in {video_path}.")
                cap.release()
                return None

    velocities_mps = []
    frame_count = 0

    while True:
        ret, frame_cpu = cap.read()
        if not ret:
            break

        if roi:
            x, y, w, h = roi
            frame_roi_cpu = frame_cpu[y:y+h, x:x+w]
        else:
            frame_roi_cpu = frame_cpu

        current_gray_cpu = cv2.cvtColor(frame_roi_cpu, cv2.COLOR_BGR2GRAY)
        frame_velocities_px_per_frame = []

        if cuda_available:
            current_gray_gpu = cv2.cuda_GpuMat()
            current_gray_gpu.upload(current_gray_cpu)

            if use_dense_flow:
                flow_gpu = gpu_flow_calculator.calc(prev_gray_gpu, current_gray_gpu, None)
                flow_xy = flow_gpu.download() # Download the flow field (dx, dy)
                mag, _ = cv2.cartToPolar(flow_xy[..., 0], flow_xy[..., 1])
                valid_flow_magnitudes = mag[(mag > 0.1) & (mag < 100)] # Thresholds
                if len(valid_flow_magnitudes) > 0:
                    avg_pixel_displacement = np.mean(valid_flow_magnitudes)
                    frame_velocities_px_per_frame.append(avg_pixel_displacement)
            else: # Sparse GPU
                if p0_gpu is None or p0_gpu.empty() or p0_gpu.cols() == 0: # Check if p0_gpu is valid
                    # Re-detect features if lost
                    p0_cpu_new = cv2.goodFeaturesToTrack(prev_gray_cpu, mask=None, maxCorners=200, qualityLevel=0.01, minDistance=10)
                    if p0_cpu_new is None or len(p0_cpu_new) == 0:
                        prev_gray_gpu.upload(current_gray_cpu)
                        prev_gray_cpu = current_gray_cpu.copy()
                        continue
                    p0_gpu.upload(p0_cpu_new.astype(np.float32))

                # p1_gpu, status_gpu, err_gpu = gpu_sparse_flow.calc(prev_gray_gpu, current_gray_gpu, p0_gpu, None)
                # status and err are often combined or handled differently in CUDA API.
                # The calc method for SparsePyrLKOpticalFlow returns (nextPts, status, error)
                p1_gpu, status_gpu, _ = gpu_sparse_flow.calc(prev_gray_gpu, current_gray_gpu, p0_gpu, None)

                if p1_gpu is not None and not p1_gpu.empty():
                    p0_cpu_tracked = p0_gpu.download().reshape(-1, 2) # Download points for CPU calculations
                    p1_cpu_tracked = p1_gpu.download().reshape(-1, 2)
                    status_cpu = status_gpu.download().ravel()

                    good_new = p1_cpu_tracked[status_cpu == 1]
                    good_old = p0_cpu_tracked[status_cpu == 1]

                    for (new_pt, old_pt) in zip(good_new, good_old):
                        dx = new_pt[0] - old_pt[0]
                        dy = new_pt[1] - old_pt[1]
                        displacement = np.sqrt(dx**2 + dy**2)
                        frame_velocities_px_per_frame.append(displacement)
                    
                    if len(good_new) > 0:
                         p0_gpu.upload(good_new.reshape(-1, 1, 2).astype(np.float32)) # Update points for next iteration
                    else: # No points tracked successfully
                        p0_gpu = None # Will trigger re-detection

                else: # No points outputted by LK GPU
                    p0_gpu = None # Will trigger re-detection

            prev_gray_gpu.upload(current_gray_cpu) # Update prev_gray_gpu for next iteration

        else: # CPU Path
            if use_dense_flow:
                flow_cpu = cv2.calcOpticalFlowFarneback(prev_gray_cpu, current_gray_cpu, None,
                                                    0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow_cpu[..., 0], flow_cpu[..., 1])
                valid_flow_magnitudes = mag[(mag > 0.1) & (mag < 100)]
                if len(valid_flow_magnitudes) > 0:
                    avg_pixel_displacement = np.mean(valid_flow_magnitudes)
                    frame_velocities_px_per_frame.append(avg_pixel_displacement)
            else: # Sparse CPU
                if p0_cpu is None or len(p0_cpu) == 0:
                    p0_cpu = cv2.goodFeaturesToTrack(prev_gray_cpu, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
                    if p0_cpu is None or len(p0_cpu) == 0:
                        prev_gray_cpu = current_gray_cpu.copy()
                        continue

                p1_cpu, st_cpu, _ = cv2.calcOpticalFlowPyrLK(prev_gray_cpu, current_gray_cpu, p0_cpu, None, **lk_params_cpu)
                if p1_cpu is not None and st_cpu is not None:
                    good_new = p1_cpu[st_cpu == 1]
                    good_old = p0_cpu[st_cpu == 1]
                    for (new, old) in zip(good_new, good_old):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        dx = a - c
                        dy = b - d
                        displacement = np.sqrt(dx**2 + dy**2)
                        frame_velocities_px_per_frame.append(displacement)
                    p0_cpu = good_new.reshape(-1, 1, 2)
                else:
                    p0_cpu = None # Re-detect

            prev_gray_cpu = current_gray_cpu.copy()


        if frame_velocities_px_per_frame:
            avg_frame_displacement_px = np.mean(frame_velocities_px_per_frame)
            velocity_px_per_s = avg_frame_displacement_px * fps
            velocity_m_per_s = velocity_px_per_s * meters_per_pixel
            velocities_mps.append(velocity_m_per_s)

        frame_count += 1
        if max_frames_to_process and frame_count >= max_frames_to_process:
            break

    cap.release()
    cv2.destroyAllWindows() # Just in case any debug windows were opened

    if not velocities_mps:
        print(f"Warning: No valid velocities could be calculated for {video_path}.")
        return None
    return np.mean(velocities_mps)


def main():
    parser = argparse.ArgumentParser(description="Estimate water velocity from videos in a folder (GPU enabled).")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing video files.")
    parser.add_argument("meters_per_pixel", type=float, help="Calibration: real-world meters per pixel.")
    parser.add_argument("--roi", type=str, help="Region of Interest: x,y,w,h. Optional.", default=None)
    parser.add_argument("--video_extensions", type=str, help="Comma-separated video extensions. Default: mp4,avi,mov,mkv", default="mp4,avi,mov,mkv")
    parser.add_argument("--use_dense_flow", action="store_true", help="Use Farneback dense optical flow instead of Lucas-Kanade.")
    parser.add_argument("--skip_frames", type=int, default=0, help="Number of initial frames to skip.")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process per video.")

    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        print(f"Error: Folder not found at {args.folder_path}")
        return

    parsed_roi = None
    if args.roi:
        try:
            parsed_roi = tuple(map(int, args.roi.split(',')))
            if len(parsed_roi) != 4: raise ValueError
        except ValueError:
            print("Error: Invalid ROI format. Use x,y,w,h")
            return

    extensions = [ext.strip().lower() for ext in args.video_extensions.split(',')]
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(args.folder_path, f"*.{ext}")))
        video_files.extend(glob.glob(os.path.join(args.folder_path, f"*.{ext.upper()}")))

    if not video_files:
        print(f"No video files found in {args.folder_path}")
        return

    print(f"Found {len(video_files)} video(s) to process.\n")
    results = {}

    for video_file in video_files:
        start_time = time.time()
        print(f"Processing {os.path.basename(video_file)}...")
        # Use the GPU enabled function
        avg_velocity = estimate_water_velocity_gpu(
            video_file,
            args.meters_per_pixel,
            roi=parsed_roi,
            use_dense_flow=args.use_dense_flow,
            skip_frames=args.skip_frames,
            max_frames_to_process=args.max_frames
        )
        end_time = time.time()
        proc_time = end_time - start_time

        if avg_velocity is not None:
            print(f"Estimated average water velocity for {os.path.basename(video_file)}: {avg_velocity:.3f} m/s (Processed in {proc_time:.2f}s)\n")
            results[os.path.basename(video_file)] = (avg_velocity, proc_time)
        else:
            print(f"Could not estimate velocity for {os.path.basename(video_file)}. (Processed in {proc_time:.2f}s)\n")
            results[os.path.basename(video_file)] = ("N/A", proc_time)

    print("\n--- Summary ---")
    for video_name, (vel, ptime) in results.items():
        if isinstance(vel, float):
            print(f"{video_name}: {vel:.3f} m/s (Time: {ptime:.2f}s)")
        else:
            print(f"{video_name}: {vel} (Time: {ptime:.2f}s)")

if __name__ == "__main__":
    main()