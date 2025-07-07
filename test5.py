import cv2
import numpy as np
import os
import glob
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import math

# --- Global CUDA Availability Check ---
CUDA_AVAILABLE = False
try:
    if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        CUDA_AVAILABLE = True
        print("INFO: CUDA is available. GPU acceleration for Dense Flow can be used.")
    elif hasattr(cv2, 'cuda'):
        print("INFO: OpenCV CUDA module is present, but no CUDA-enabled GPU device found or count is 0. GPU acceleration for Dense Flow unavailable.")
    else:
        print("INFO: OpenCV was not compiled with CUDA support. GPU acceleration for Dense Flow unavailable.")
except cv2.error as e:
    print(f"WARNING: Error checking CUDA availability: {e}. GPU acceleration for Dense Flow unavailable.")
except Exception as e: # Catch any other unexpected error during the check
    print(f"WARNING: Unexpected error during CUDA check: {e}. GPU acceleration for Dense Flow unavailable.")

# --- Optical Flow Parameters ---
# For Lucas-Kanade (Sparse)
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# For Farneback (Dense) CPU
farneback_params_cpu = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)

# For Farneback (Dense) GPU
# Parameters for cv2.cuda_FarnebackOpticalFlow.create()
# numLevels, pyrScale, fastPyramids, winSize, numIters, polyN, polySigma, flags
farneback_params_gpu_create = dict(
    numLevels=farneback_params_cpu['levels'],
    pyrScale=farneback_params_cpu['pyr_scale'],
    fastPyramids=False, # A GPU-specific option, False is often a good default
    winSize=farneback_params_cpu['winsize'],
    numIters=farneback_params_cpu['iterations'],
    polyN=farneback_params_cpu['poly_n'],
    polySigma=farneback_params_cpu['poly_sigma'],
    flags=farneback_params_cpu['flags']
)

def estimate_water_velocity(video_path, meters_per_pixel, roi=None, use_dense_flow=False, use_gpu=False, skip_frames=0, max_frames_to_process=None, log_callback=None):
    """
    Estimates water velocity from a video using optical flow.
    Args:
        video_path (str): Path to the video file.
        meters_per_pixel (float): Calibration factor (real-world meters represented by one pixel).
        roi (tuple or list, optional): ROI definition. Defaults to None.
        use_dense_flow (bool): If True, use Farneback dense optical flow. Else, Lucas-Kanade sparse.
        use_gpu (bool): If True and use_dense_flow is True, attempt to use GPU for Farneback.
        skip_frames (int): Number of initial frames to skip.
        max_frames_to_process (int, optional): Max frames to process after skipping.
        log_callback (function, optional): Callback function to log messages.
    Returns:
        float: Estimated average water velocity in m/s, or None if processing failed.
    """
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"Error: Could not open video {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        log(f"Warning: Could not get FPS for {video_path}. Assuming 30 FPS.")
        fps = 30.0

    for i in range(skip_frames):
        ret, _ = cap.read()
        if not ret:
            log(f"Error: Not enough frames to skip ({skip_frames} requested, {i} skipped) in {video_path}")
            cap.release()
            return None

    ret, prev_frame = cap.read()
    if not ret:
        log(f"Error: Could not read the first frame from {video_path} after skipping {skip_frames} frames.")
        cap.release()
        return None

    fh_orig, fw_orig = prev_frame.shape[:2]
    is_quad_roi = False
    roi_points_relative_to_crop = None
    x_b, y_b, w_b, h_b = 0, 0, fw_orig, fh_orig

    if roi:
        roi_message_prefix = ""
        if isinstance(roi, (list, tuple)) and len(roi) == 4 and \
           isinstance(roi[0], (list, tuple)) and len(roi[0]) == 2:
            is_quad_roi = True
            roi_message_prefix = "Quadrilateral "
            roi_points_np = np.array(roi, dtype=np.int32)
            for pt_idx, (px, py) in enumerate(roi_points_np):
                if not (0 <= px < fw_orig and 0 <= py < fh_orig):
                    log(f"Error: {roi_message_prefix}ROI point {pt_idx+1} ({px},{py}) is outside original frame dimensions ({fw_orig}x{fh_orig}) for {video_path}.")
                    cap.release()
                    return None
            x_b_calc, y_b_calc, w_b_calc, h_b_calc = cv2.boundingRect(roi_points_np)
            if not (w_b_calc > 0 and h_b_calc > 0):
                log(f"Error: {roi_message_prefix}ROI results in zero width/height bounding box for {video_path}.")
                cap.release()
                return None
            if not (0 <= x_b_calc < fw_orig and 0 <= y_b_calc < fh_orig and \
                    x_b_calc + w_b_calc <= fw_orig and y_b_calc + h_b_calc <= fh_orig):
                log(f"Error: {roi_message_prefix}ROI's bounding box ({x_b_calc},{y_b_calc},{w_b_calc},{h_b_calc}) is outside frame dimensions ({fw_orig}x{fh_orig}) for {video_path}.")
                cap.release()
                return None
            x_b, y_b, w_b, h_b = x_b_calc, y_b_calc, w_b_calc, h_b_calc
            prev_frame_processed = prev_frame[y_b:y_b+h_b, x_b:x_b+w_b]
            roi_points_relative_to_crop = roi_points_np.copy()
            roi_points_relative_to_crop[:, 0] -= x_b
            roi_points_relative_to_crop[:, 1] -= y_b
        elif isinstance(roi, tuple) and len(roi) == 4:
            roi_message_prefix = "Rectangular "
            x_rect, y_rect, w_rect, h_rect = roi
            if x_rect < 0 or y_rect < 0 or w_rect <= 0 or h_rect <= 0 or \
               (x_rect + w_rect) > fw_orig or (y_rect + h_rect) > fh_orig:
                log(f"Error: {roi_message_prefix}ROI ({x_rect},{y_rect},{w_rect},{h_rect}) is outside frame dimensions ({fw_orig}x{fh_orig}) or invalid for {video_path}.")
                cap.release()
                return None
            x_b, y_b, w_b, h_b = x_rect, y_rect, w_rect, h_rect
            prev_frame_processed = prev_frame[y_b:y_b+h_b, x_b:x_b+w_b]
        else:
            log(f"Error: Invalid ROI format provided for {video_path}.")
            cap.release()
            return None
    else:
        prev_frame_processed = prev_frame

    prev_gray = cv2.cvtColor(prev_frame_processed, cv2.COLOR_BGR2GRAY)
    p0 = None

    # --- GPU / Dense Flow Specific Initialization ---
    gpu_farneback_calculator = None
    gpu_active_for_this_video = False # Tracks if GPU is successfully initialized and active for this video

    if use_dense_flow and use_gpu:
        if CUDA_AVAILABLE:
            try:
                gpu_farneback_calculator = cv2.cuda_FarnebackOpticalFlow.create(**farneback_params_gpu_create)
                log(f"INFO: Successfully initialized GPU for Farneback dense optical flow on {os.path.basename(video_path)}.")
                gpu_active_for_this_video = True
            except cv2.error as e:
                log(f"WARNING: Error creating CUDA Farneback object for {os.path.basename(video_path)}: {e}. Falling back to CPU.")
        else:
            log(f"INFO: GPU requested for dense flow on {os.path.basename(video_path)}, but CUDA is not available. Using CPU.")
    elif use_dense_flow and not use_gpu:
        log(f"INFO: Using CPU for Farneback dense optical flow on {os.path.basename(video_path)} (as requested).")
    # --- End GPU/Dense Flow Init ---

    if not use_dense_flow: # Sparse LK
        gftt_mask = None
        if is_quad_roi and roi_points_relative_to_crop is not None:
            gftt_mask = np.zeros(prev_gray.shape, dtype=np.uint8)
            cv2.fillPoly(gftt_mask, [np.array(roi_points_relative_to_crop, dtype=np.int32)], 255)
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=gftt_mask, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        if p0 is None or len(p0) == 0:
            log(f"Warning: No good features to track in the first ROI/frame of {video_path} (LK). Processing cannot continue for this video with LK.")
            cap.release()
            return None

    velocities_mps = []
    frame_count = 0

    # Pre-allocate GpuMat for GPU flow output if using GPU, to potentially save a little time
    gpu_flow_output_mat = None
    if gpu_active_for_this_video and gpu_farneback_calculator:
        try:
            # Create a GpuMat of the expected size for the flow output.
            # Flow has 2 channels (dx, dy).
            # Note: This assumes prev_gray is representative of all gray frames in size.
            gpu_flow_output_mat = cv2.cuda_GpuMat(prev_gray.shape[0], prev_gray.shape[1], cv2.CV_32FC2)
        except cv2.error as e:
            log(f"WARNING: Error pre-allocating GpuMat for flow output: {e}. Will allocate dynamically.")
            gpu_flow_output_mat = None # Fallback to dynamic allocation by calc method

    gpu_prev_gray_mat = None
    gpu_current_gray_mat = None
    if gpu_active_for_this_video and gpu_farneback_calculator:
        try:
            gpu_prev_gray_mat = cv2.cuda_GpuMat()
            gpu_current_gray_mat = cv2.cuda_GpuMat()
        except cv2.error as e:
            log(f"WARNING: Error creating GpuMat for gray frames: {e}. GPU use might fail.")
            gpu_active_for_this_video = False # Disable GPU path if core GpuMats cannot be created


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if roi:
            frame_processed = frame[y_b:y_b+h_b, x_b:x_b+w_b]
        else:
            frame_processed = frame
        
        current_gray = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2GRAY)
        frame_velocities_px_per_frame = []

        if use_dense_flow:
            flow = None
            if gpu_active_for_this_video and gpu_farneback_calculator and \
               gpu_prev_gray_mat is not None and gpu_current_gray_mat is not None:
                try:
                    gpu_prev_gray_mat.upload(prev_gray)
                    gpu_current_gray_mat.upload(current_gray)
                    
                    if gpu_flow_output_mat is not None:
                        gpu_farneback_calculator.calc(gpu_prev_gray_mat, gpu_current_gray_mat, gpu_flow_output_mat)
                        flow = gpu_flow_output_mat.download()
                    else: # Fallback if pre-allocation failed
                        temp_gpu_flow = gpu_farneback_calculator.calc(gpu_prev_gray_mat, gpu_current_gray_mat, None)
                        flow = temp_gpu_flow.download()

                except cv2.error as e:
                    log(f"WARNING: Error during CUDA Farneback calculation for {os.path.basename(video_path)} (frame {frame_count}): {e}. Switching to CPU for remaining frames.")
                    gpu_active_for_this_video = False # Fallback for the rest of this video
                    # Calculate this frame on CPU
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, **farneback_params_cpu)
            else: # CPU path for dense flow
                flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, **farneback_params_cpu)

            if flow is not None:
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                relevant_magnitudes = []
                if is_quad_roi and roi_points_relative_to_crop is not None:
                    flow_filter_mask = np.zeros(prev_gray.shape, dtype=np.uint8)
                    cv2.fillPoly(flow_filter_mask, [np.array(roi_points_relative_to_crop, dtype=np.int32)], 255)
                    relevant_magnitudes = mag[flow_filter_mask == 255]
                else:
                    relevant_magnitudes = mag.flatten()
                
                valid_flow_magnitudes = relevant_magnitudes[(relevant_magnitudes > 0.1) & (relevant_magnitudes < 100)]
                if len(valid_flow_magnitudes) > 0:
                    avg_pixel_displacement = np.mean(valid_flow_magnitudes)
                    frame_velocities_px_per_frame.append(avg_pixel_displacement)
        else: # Sparse (Lucas-Kanade)
            if p0 is None or len(p0) == 0:
                gftt_mask_runtime = None
                if is_quad_roi and roi_points_relative_to_crop is not None:
                    gftt_mask_runtime = np.zeros(prev_gray.shape, dtype=np.uint8)
                    cv2.fillPoly(gftt_mask_runtime, [np.array(roi_points_relative_to_crop, dtype=np.int32)], 255)
                p0 = cv2.goodFeaturesToTrack(prev_gray, mask=gftt_mask_runtime, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
                if p0 is None or len(p0) == 0:
                    prev_gray = current_gray.copy()
                    frame_count += 1
                    if max_frames_to_process and frame_count >= max_frames_to_process:
                        break
                    continue

            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, p0, None, **lk_params)

            if p1 is not None and st is not None:
                good_new_crop = p1[st == 1]
                good_old_crop = p0[st == 1]
                
                good_new_final = good_new_crop
                good_old_final = good_old_crop

                if is_quad_roi and roi_points_relative_to_crop is not None and len(good_old_crop) > 0:
                    filtered_new = []
                    filtered_old = []
                    for i, pt_old_crop_coords in enumerate(good_old_crop):
                        if cv2.pointPolygonTest(np.array(roi_points_relative_to_crop, dtype=np.int32), 
                                                tuple(pt_old_crop_coords.ravel().astype(float)), False) >= 0:
                            filtered_old.append(pt_old_crop_coords)
                            filtered_new.append(good_new_crop[i])
                    good_new_final = np.array(filtered_new)
                    good_old_final = np.array(filtered_old)

                if len(good_new_final) > 0:
                    for i, (new, old) in enumerate(zip(good_new_final, good_old_final)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        displacement = np.sqrt((a - c)**2 + (b - d)**2)
                        frame_velocities_px_per_frame.append(displacement)
                    p0 = good_new_final.reshape(-1, 1, 2)
                else:
                    p0 = None 
            else:
                p0 = None

        if frame_velocities_px_per_frame:
            avg_frame_displacement_px = np.mean(frame_velocities_px_per_frame)
            velocity_px_per_s = avg_frame_displacement_px * fps
            velocity_m_per_s = velocity_px_per_s * meters_per_pixel
            velocities_mps.append(velocity_m_per_s)

        prev_gray = current_gray.copy()
        frame_count += 1
        if max_frames_to_process and frame_count >= max_frames_to_process:
            break

    cap.release()

    if not velocities_mps:
        log(f"Warning: No valid velocities could be calculated for {video_path} after processing {frame_count} frames.")
        return None

    final_avg_velocity = np.mean(velocities_mps)
    return final_avg_velocity


class WaterVelocityGUI:
    def __init__(self, master):
        self.master = master
        master.title("Water Velocity Estimator")

        self.folder_path_var = tk.StringVar()
        self.mpp_var = tk.StringVar(value="0.01")
        self.known_distance_var = tk.StringVar(value="100.0") 
        self.roi_var = tk.StringVar()
        self.extensions_var = tk.StringVar(value="mp4,avi,mov,mkv")
        self.use_dense_flow_var = tk.BooleanVar(value=False)
        self.use_gpu_var = tk.BooleanVar(value=False) # For GPU option
        self.skip_frames_var = tk.StringVar(value="0")
        self.max_frames_var = tk.StringVar()
        self.selected_video_for_setup_path = tk.StringVar()

        self.roi_drawing_points = []
        self.roi_drawing_frame_copy = None
        self.roi_drawing_window_name = "Draw ROI - Click 4 points. ENTER: Confirm, R: Reset, ESC/C: Cancel"

        self.mpp_calib_points = []
        self.mpp_calib_frame_copy = None
        self.mpp_calib_window_name = "Calibrate Meters/Pixel - Click 2 points. ENTER: Confirm, R: Reset, ESC/C: Cancel"

        self.max_display_width = 1280
        self.max_display_height = 720

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        input_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="10")
        input_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        input_frame.columnconfigure(1, weight=1)

        row_idx = 0
        ttk.Label(input_frame, text="Video Folder:").grid(row=row_idx, column=0, sticky="w", pady=2)
        ttk.Entry(input_frame, textvariable=self.folder_path_var, width=40).grid(row=row_idx, column=1, sticky="ew", pady=2)
        ttk.Button(input_frame, text="Browse...", command=self.browse_folder).grid(row=row_idx, column=2, padx=5, pady=2)
        row_idx += 1

        mpp_calib_group = ttk.LabelFrame(input_frame, text="Meters/Pixel Calibration", padding="5")
        mpp_calib_group.grid(row=row_idx, column=0, columnspan=3, sticky="ew", pady=(10,5))
        mpp_calib_group.columnconfigure(1, weight=1)
        mpp_calib_row = 0

        ttk.Label(mpp_calib_group, text="Known Distance (cm):").grid(row=mpp_calib_row, column=0, sticky="w", pady=2, padx=2)
        ttk.Entry(mpp_calib_group, textvariable=self.known_distance_var).grid(row=mpp_calib_row, column=1, sticky="ew", pady=2, padx=2)
        mpp_calib_row += 1

        ttk.Label(mpp_calib_group, text="Meters/Pixel:").grid(row=mpp_calib_row, column=0, sticky="w", pady=2, padx=2)
        ttk.Entry(mpp_calib_group, textvariable=self.mpp_var).grid(row=mpp_calib_row, column=1, sticky="ew", pady=2, padx=2)
        self.calibrate_mpp_button = ttk.Button(mpp_calib_group, text="Calibrate...", command=self.calibrate_mpp_on_frame, state=tk.DISABLED)
        self.calibrate_mpp_button.grid(row=mpp_calib_row, column=2, padx=(5,2), pady=2)
        mpp_calib_row += 1
        
        self.select_setup_video_button = ttk.Button(mpp_calib_group, text="Select Video for Setup", command=self.select_video_for_setup)
        self.select_setup_video_button.grid(row=mpp_calib_row, column=0, pady=(5,2), sticky="w", padx=2)
        self.selected_setup_video_label = ttk.Label(mpp_calib_group, text="No video selected.", wraplength=350, justify=tk.LEFT)
        self.selected_setup_video_label.grid(row=mpp_calib_row, column=1, columnspan=2, pady=(5,2), sticky="ew", padx=2)
        row_idx += 1

        roi_group = ttk.LabelFrame(input_frame, text="Region of Interest (ROI)", padding="5")
        roi_group.grid(row=row_idx, column=0, columnspan=3, sticky="ew", pady=5)
        roi_group.columnconfigure(1, weight=1)
        roi_group_row = 0

        ttk.Label(roi_group, text="ROI (4-points):").grid(row=roi_group_row, column=0, sticky="w", pady=2, padx=2)
        ttk.Entry(roi_group, textvariable=self.roi_var).grid(row=roi_group_row, column=1, sticky="ew", padx=2, pady=2)
        self.draw_roi_button = ttk.Button(roi_group, text="Draw ROI (4 Clicks)", command=self.draw_roi_on_frame, state=tk.DISABLED)
        self.draw_roi_button.grid(row=roi_group_row, column=2, pady=2, padx=(5,2))
        row_idx += 1
        
        ttk.Label(input_frame, text="Video Ext (comma-sep):").grid(row=row_idx, column=0, sticky="w", pady=2)
        ttk.Entry(input_frame, textvariable=self.extensions_var).grid(row=row_idx, column=1, columnspan=2, sticky="ew", pady=2)
        row_idx += 1

        ttk.Label(input_frame, text="Skip Initial Frames:").grid(row=row_idx, column=0, sticky="w", pady=2)
        ttk.Entry(input_frame, textvariable=self.skip_frames_var).grid(row=row_idx, column=1, columnspan=2, sticky="ew", pady=2)
        row_idx += 1

        ttk.Label(input_frame, text="Max Frames (optional):").grid(row=row_idx, column=0, sticky="w", pady=2)
        ttk.Entry(input_frame, textvariable=self.max_frames_var).grid(row=row_idx, column=1, columnspan=2, sticky="ew", pady=2)
        row_idx += 1
        
        ttk.Checkbutton(input_frame, text="Use Dense Optical Flow (Farneback)", variable=self.use_dense_flow_var).grid(row=row_idx, column=0, columnspan=3, sticky="w", pady=5)
        row_idx += 1

        # GPU Checkbutton
        self.gpu_checkbutton = ttk.Checkbutton(input_frame, text="Use GPU for Dense Flow (if available, requires CUDA)", variable=self.use_gpu_var)
        self.gpu_checkbutton.grid(row=row_idx, column=0, columnspan=3, sticky="w", pady=5)
        if not CUDA_AVAILABLE:
            self.gpu_checkbutton.config(state=tk.DISABLED)
            self.use_gpu_var.set(False) # Ensure it's false if disabled
        row_idx += 1


        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.grid(row=1, column=0, sticky="ew", pady=5)
        
        self.start_button = ttk.Button(control_frame, text="Start Processing", command=self.start_processing_thread)
        self.start_button.pack(pady=5)

        output_frame = ttk.LabelFrame(main_frame, text="Log & Results", padding="10")
        output_frame.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")
        
        self.log_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=15, width=80)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

    def browse_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.folder_path_var.set(folder_selected)

    def select_video_for_setup(self):
        video_path = filedialog.askopenfilename(
            title="Select Video for Setup (MPP Calibration & ROI)",
            filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*"))
        )
        if video_path:
            self.selected_video_for_setup_path.set(video_path)
            self.calibrate_mpp_button.config(state=tk.NORMAL)
            self.draw_roi_button.config(state=tk.NORMAL)
            label_text = f"Setup Video: {os.path.basename(video_path)}"
            cap_temp = cv2.VideoCapture(video_path)
            if cap_temp.isOpened():
                ret_temp, frame_temp = cap_temp.read()
                if ret_temp:
                    h, w = frame_temp.shape[:2]
                    label_text += f" ({w}x{h} px)"
                else:
                    label_text += " (Error reading frame)"
                cap_temp.release()
            else:
                label_text += " (Error opening video)"
            self.selected_setup_video_label.config(text=label_text)
            self._log_message_gui(f"Video selected for setup: {os.path.basename(video_path)}")
        else:
            self.selected_video_for_setup_path.set("")
            self.calibrate_mpp_button.config(state=tk.DISABLED)
            self.draw_roi_button.config(state=tk.DISABLED)
            self.selected_setup_video_label.config(text="No video selected.")

    def _mouse_callback_mpp(self, event, x, y, flags, param):
        window_name, original_frame = param
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.mpp_calib_points) < 2:
                self.mpp_calib_points.append((x, y))
                cv2.circle(self.mpp_calib_frame_copy, (x, y), 5, (0, 255, 0), -1)
                if len(self.mpp_calib_points) == 2:
                    cv2.line(self.mpp_calib_frame_copy, self.mpp_calib_points[0], self.mpp_calib_points[1], (0, 255, 0), 2)
                cv2.imshow(window_name, self.mpp_calib_frame_copy)

    def calibrate_mpp_on_frame(self):
        video_path = self.selected_video_for_setup_path.get()
        if not video_path:
            self._log_message_gui("Error: No video selected for setup. Use 'Select Video for Setup' first.")
            return
        try:
            known_distance_cm = float(self.known_distance_var.get())
            if known_distance_cm <= 0:
                self._log_message_gui("Error: Known Distance (cm) must be a positive number.")
                return
        except ValueError:
            self._log_message_gui("Error: Invalid Known Distance (cm). Please enter a valid number.")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log_message_gui(f"Error: Could not open video {os.path.basename(video_path)} for MPP calibration.")
            return
        ret, frame_for_mpp = cap.read()
        cap.release()
        if not ret:
            self._log_message_gui(f"Error: Could not read a frame from {os.path.basename(video_path)} for MPP calibration.")
            return

        self.mpp_calib_points = []
        self.mpp_calib_frame_copy = frame_for_mpp.copy()
        self._log_message_gui(f"MPP Calibration window '{self.mpp_calib_window_name}' opening. Click 2 points. \nENTER to confirm, R to reset, ESC/C to cancel.")
        self.master.withdraw() 
        cv2.namedWindow(self.mpp_calib_window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.mpp_calib_window_name, self._mouse_callback_mpp, (self.mpp_calib_window_name, frame_for_mpp))
        cv2.imshow(self.mpp_calib_window_name, self.mpp_calib_frame_copy)
        frame_h, frame_w = frame_for_mpp.shape[:2]
        if frame_w > self.max_display_width or frame_h > self.max_display_height:
            scale_w = self.max_display_width / frame_w
            scale_h = self.max_display_height / frame_h
            scale = min(scale_w, scale_h)
            display_w = int(frame_w * scale)
            display_h = int(frame_h * scale)
            cv2.resizeWindow(self.mpp_calib_window_name, display_w, display_h)
        cv2.waitKey(1) 
        calib_confirmed = False
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 13: 
                if len(self.mpp_calib_points) == 2:
                    calib_confirmed = True; break
                else: self._log_message_gui("Please click 2 points before pressing Enter for MPP calibration.")
            elif key == ord('r') or key == ord('R'):
                self.mpp_calib_points = []; self.mpp_calib_frame_copy = frame_for_mpp.copy()
                cv2.imshow(self.mpp_calib_window_name, self.mpp_calib_frame_copy)
                self._log_message_gui("MPP calibration points reset. Click 2 new points.")
            elif key == 27 or key == ord('c') or key == ord('C'): break
        cv2.destroyWindow(self.mpp_calib_window_name)
        self.master.deiconify(); self.master.attributes('-topmost', True); self.master.focus_force(); self.master.attributes('-topmost', False)
        if calib_confirmed and len(self.mpp_calib_points) == 2:
            p1, p2 = self.mpp_calib_points
            pixel_distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if pixel_distance > 0:
                known_distance_m = known_distance_cm / 100.0 
                mpp_calculated = known_distance_m / pixel_distance
                self.mpp_var.set(f"{mpp_calculated:.8f}")
                self._log_message_gui(f"MPP calibrated: {mpp_calculated:.8f} m/pixel (Known: {known_distance_cm}cm, Pixels: {pixel_distance:.2f}px)")
            else: self._log_message_gui("MPP calibration failed: Pixel distance is zero.")
        else: self._log_message_gui("MPP calibration cancelled or incomplete.")

    def _mouse_callback_roi(self, event, x, y, flags, param):
        window_name, original_frame = param
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.roi_drawing_points) < 4:
                self.roi_drawing_points.append((x, y))
                cv2.circle(self.roi_drawing_frame_copy, (x, y), 5, (0, 255, 0), -1)
                if len(self.roi_drawing_points) > 1:
                    cv2.line(self.roi_drawing_frame_copy, self.roi_drawing_points[-2], self.roi_drawing_points[-1], (0, 255, 0), 2)
                if len(self.roi_drawing_points) == 4:
                    cv2.line(self.roi_drawing_frame_copy, self.roi_drawing_points[3], self.roi_drawing_points[0], (0, 0, 255), 2)
                cv2.imshow(window_name, self.roi_drawing_frame_copy)

    def draw_roi_on_frame(self):
        video_path = self.selected_video_for_setup_path.get()
        if not video_path:
            self._log_message_gui("Error: No video selected for setup. Use 'Select Video for Setup' first."); return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log_message_gui(f"Error: Could not open video {os.path.basename(video_path)} for ROI."); return
        ret, frame_for_roi = cap.read(); cap.release()
        if not ret:
            self._log_message_gui(f"Error: Could not read a frame from {os.path.basename(video_path)} for ROI."); return
        self.roi_drawing_points = []; self.roi_drawing_frame_copy = frame_for_roi.copy()
        self._log_message_gui(f"ROI selection window '{self.roi_drawing_window_name}' opening. Click 4 points. \nENTER to confirm, R to reset, ESC/C to cancel.")
        self.master.withdraw() 
        cv2.namedWindow(self.roi_drawing_window_name, cv2.WINDOW_NORMAL) 
        cv2.setMouseCallback(self.roi_drawing_window_name, self._mouse_callback_roi, (self.roi_drawing_window_name, frame_for_roi))
        cv2.imshow(self.roi_drawing_window_name, self.roi_drawing_frame_copy)
        frame_h, frame_w = frame_for_roi.shape[:2]
        if frame_w > self.max_display_width or frame_h > self.max_display_height:
            scale_w = self.max_display_width / frame_w; scale_h = self.max_display_height / frame_h
            scale = min(scale_w, scale_h)
            display_w = int(frame_w * scale); display_h = int(frame_h * scale)
            cv2.resizeWindow(self.roi_drawing_window_name, display_w, display_h)
        cv2.waitKey(1)
        roi_confirmed = False
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 13: 
                if len(self.roi_drawing_points) == 4: roi_confirmed = True; break
                else: self._log_message_gui("Please click 4 points before pressing Enter for ROI.")
            elif key == ord('r') or key == ord('R'):
                self.roi_drawing_points = []; self.roi_drawing_frame_copy = frame_for_roi.copy()
                cv2.imshow(self.roi_drawing_window_name, self.roi_drawing_frame_copy)
                self._log_message_gui("ROI points reset. Click 4 new points.")
            elif key == 27 or key == ord('c') or key == ord('C'): break
        cv2.destroyWindow(self.roi_drawing_window_name)
        self.master.deiconify(); self.master.attributes('-topmost', True); self.master.focus_force(); self.master.attributes('-topmost', False)
        if roi_confirmed and len(self.roi_drawing_points) == 4:
            flat_coords = [coord for point in self.roi_drawing_points for coord in point]
            self.roi_var.set(",".join(map(str, flat_coords)))
            self._log_message_gui(f"ROI selected: {self.roi_var.get()}")
        else: self.roi_var.set(""); self._log_message_gui("ROI selection cancelled or incomplete.")

    def _log_message_gui(self, message):
        if self.master.winfo_exists():
            self.log_text.insert(tk.END, str(message) + "\n")
            self.log_text.see(tk.END)
            self.master.update_idletasks() 

    def log_message_thread_safe(self, message):
        if self.master.winfo_exists():
            self.master.after(0, self._log_message_gui, message)

    def _processing_job(self):
        try:
            folder_path = self.folder_path_var.get()
            if not folder_path or not os.path.isdir(folder_path):
                self.log_message_thread_safe("Error: Please select a valid video folder."); return
            try:
                mpp = float(self.mpp_var.get())
                if mpp <= 0: raise ValueError("Meters per pixel must be positive.")
            except ValueError as e: self.log_message_thread_safe(f"Error: Invalid Meters/Pixel. {e}"); return
            parsed_roi = None; roi_str = self.roi_var.get().strip()
            if roi_str:
                try:
                    coords = list(map(int, roi_str.split(',')))
                    if len(coords) == 8:
                        parsed_roi = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
                        if any(val < 0 for val in coords): raise ValueError("ROI coordinates must be non-negative.")
                    else: raise ValueError("ROI string must contain 8 comma-separated integers for 4 points.")
                except ValueError as e: self.log_message_thread_safe(f"Error: Invalid ROI string '{roi_str}'. {e}"); return
            extensions_str = self.extensions_var.get()
            if not extensions_str: self.log_message_thread_safe("Error: Video extensions cannot be empty."); return
            extensions = [ext.strip().lower() for ext in extensions_str.split(',') if ext.strip()]
            if not extensions: self.log_message_thread_safe("Error: No valid video extensions provided."); return
            try:
                skip_frames = int(self.skip_frames_var.get())
                if skip_frames < 0: raise ValueError("Skip frames must be non-negative.")
            except ValueError as e: self.log_message_thread_safe(f"Error: Invalid Skip Frames. {e}"); return
            max_frames = None; max_frames_str = self.max_frames_var.get().strip()
            if max_frames_str:
                try:
                    max_frames = int(max_frames_str)
                    if max_frames <= 0: raise ValueError("Max frames must be positive if set.")
                except ValueError as e: self.log_message_thread_safe(f"Error: Invalid Max Frames. {e}"); return
            use_dense = self.use_dense_flow_var.get()
            use_gpu_flag = self.use_gpu_var.get() # Get GPU preference

            video_files = []
            for ext in extensions:
                video_files.extend(glob.glob(os.path.join(folder_path, f"*.{ext}")))
                video_files.extend(glob.glob(os.path.join(folder_path, f"*.{ext.upper()}")))
            video_files = sorted(list(set(video_files)))
            if not video_files: self.log_message_thread_safe(f"No videos with specified extensions found in {folder_path}"); return

            self.log_message_thread_safe(f"Found {len(video_files)} video(s) to process.\n")
            results = {}
            for video_file in video_files:
                self.log_message_thread_safe(f"--- Processing {os.path.basename(video_file)} ---")
                avg_velocity = estimate_water_velocity(
                    video_file, mpp, roi=parsed_roi, use_dense_flow=use_dense,
                    use_gpu=use_gpu_flag, # Pass GPU preference
                    skip_frames=skip_frames, max_frames_to_process=max_frames,
                    log_callback=self.log_message_thread_safe
                )
                if avg_velocity is not None:
                    self.log_message_thread_safe(f"==> Estimated avg velocity for {os.path.basename(video_file)}: {avg_velocity:.3f} m/s\n")
                    results[os.path.basename(video_file)] = avg_velocity
                else:
                    self.log_message_thread_safe(f"==> Could not estimate velocity for {os.path.basename(video_file)}.\n")
                    results[os.path.basename(video_file)] = "N/A"
            self.log_message_thread_safe("\n--- Summary ---")
            if results:
                for video_name, vel in results.items():
                    if isinstance(vel, float): self.log_message_thread_safe(f"{video_name}: {vel:.3f} m/s")
                    else: self.log_message_thread_safe(f"{video_name}: {vel}")
            else: self.log_message_thread_safe("No videos yielded results.")
            self.log_message_thread_safe("\nProcessing complete.")
        except Exception as e:
            self.log_message_thread_safe(f"Unexpected error in processing job: {e}")
            import traceback
            self.log_message_thread_safe(traceback.format_exc())
        finally:
            if self.master.winfo_exists():
                self.master.after(0, lambda: self.start_button.config(state=tk.NORMAL) if self.master.winfo_exists() else None)

    def start_processing_thread(self):
        if not self.master.winfo_exists(): return
        self.log_text.delete('1.0', tk.END) 
        self._log_message_gui("Starting processing... Please wait.")
        self.start_button.config(state=tk.DISABLED)
        thread = threading.Thread(target=self._processing_job, daemon=True)
        thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    gui = WaterVelocityGUI(root)
    root.mainloop()