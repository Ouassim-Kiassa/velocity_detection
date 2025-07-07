import cv2
import numpy as np
import os
import glob
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, simpledialog
import threading
import math
import queue

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
except Exception as e:
    print(f"WARNING: Unexpected error during CUDA check: {e}. GPU acceleration for Dense Flow unavailable.")

# --- Optical Flow Parameters ---
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

farneback_params_cpu = dict(
    pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

farneback_params_gpu_create = dict(
    numLevels=farneback_params_cpu['levels'], pyrScale=farneback_params_cpu['pyr_scale'],
    fastPyramids=False, winSize=farneback_params_cpu['winsize'],
    numIters=farneback_params_cpu['iterations'], polyN=farneback_params_cpu['poly_n'],
    polySigma=farneback_params_cpu['poly_sigma'], flags=farneback_params_cpu['flags'])

def estimate_water_velocity(video_path, roi=None,
                            use_dense_flow=False, use_gpu=False,
                            skip_frames=0, max_frames_to_process=None,
                            log_callback=None, homography_matrix=None):
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)

    if homography_matrix is None:
        log(f"WARNING [{os.path.basename(video_path)}]: Homography matrix not provided. Cannot calculate metric velocity.")

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
    x_b, y_b, w_b, h_b = 0, 0, fw_orig, fh_orig # Bounding box of ROI in original frame

    if roi:
        if isinstance(roi, (list, tuple)) and len(roi) == 4 and \
           isinstance(roi[0], (list, tuple)) and len(roi[0]) == 2:
            is_quad_roi = True
            roi_points_np_orig_frame = np.array(roi, dtype=np.int32)
            for pt_idx, (px, py) in enumerate(roi_points_np_orig_frame):
                if not (0 <= px < fw_orig and 0 <= py < fh_orig):
                    log(f"Error: Quad ROI point {pt_idx+1} ({px},{py}) is outside original frame ({fw_orig}x{fh_orig}).")
                    cap.release(); return None
            x_b_calc, y_b_calc, w_b_calc, h_b_calc = cv2.boundingRect(roi_points_np_orig_frame)
            if not (w_b_calc > 0 and h_b_calc > 0):
                log(f"Error: Quad ROI results in zero width/height box."); cap.release(); return None
            x_b, y_b, w_b, h_b = x_b_calc, y_b_calc, w_b_calc, h_b_calc
            prev_frame_processed = prev_frame[y_b:y_b+h_b, x_b:x_b+w_b]
            roi_points_relative_to_crop = roi_points_np_orig_frame.copy()
            roi_points_relative_to_crop[:, 0] -= x_b
            roi_points_relative_to_crop[:, 1] -= y_b
        else:
            log(f"INFO [{os.path.basename(video_path)}]: ROI provided is not a 4-point quadrilateral. Homography-based correction will not be applied. No metric velocity will be calculated.")
            prev_frame_processed = prev_frame
            is_quad_roi = False
    else:
        prev_frame_processed = prev_frame
        is_quad_roi = False

    if prev_frame_processed.size == 0:
        log(f"Error: Frame to process is empty (ROI might be invalid or out of bounds).")
        cap.release(); return None
    prev_gray = cv2.cvtColor(prev_frame_processed, cv2.COLOR_BGR2GRAY)

    p0 = None
    gpu_farneback_calculator = None
    gpu_active_for_this_video = False
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

    if not use_dense_flow:
        gftt_mask_initial = None
        if is_quad_roi and roi_points_relative_to_crop is not None:
            gftt_mask_initial = np.zeros(prev_gray.shape, dtype=np.uint8)
            cv2.fillPoly(gftt_mask_initial, [np.array(roi_points_relative_to_crop, dtype=np.int32)], 255)
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=gftt_mask_initial, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        if p0 is None or len(p0) == 0:
            log(f"Warning [{os.path.basename(video_path)}]: No good LK features in first frame/ROI.")

    velocities_mps = []
    frame_count = 0
    gpu_flow_output_mat = None; gpu_prev_gray_mat = None; gpu_current_gray_mat = None
    if gpu_active_for_this_video and gpu_farneback_calculator:
        try:
            gpu_flow_output_mat = cv2.cuda_GpuMat(prev_gray.shape[0], prev_gray.shape[1], cv2.CV_32FC2)
            gpu_prev_gray_mat = cv2.cuda_GpuMat(); gpu_current_gray_mat = cv2.cuda_GpuMat()
        except cv2.error as e:
            log(f"WARNING: GpuMat prealloc error: {e}. GPU use might fail or be slower.")
            gpu_active_for_this_video = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if is_quad_roi:
            frame_processed = frame[y_b:y_b+h_b, x_b:x_b+w_b]
        else:
            frame_processed = frame

        if frame_processed.size == 0:
            log(f"Warning: Frame to process is empty in loop (frame {frame_count}). Skipping.")
            frame_count +=1; continue
        current_gray = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2GRAY)

        current_frame_displacements_meters = []

        if use_dense_flow:
            flow = None
            if gpu_active_for_this_video and gpu_farneback_calculator and \
               gpu_prev_gray_mat is not None and gpu_current_gray_mat is not None:
                try:
                    gpu_prev_gray_mat.upload(prev_gray); gpu_current_gray_mat.upload(current_gray)
                    if gpu_flow_output_mat is not None:
                        gpu_farneback_calculator.calc(gpu_prev_gray_mat, gpu_current_gray_mat, gpu_flow_output_mat)
                        flow = gpu_flow_output_mat.download()
                    else:
                        temp_gpu_flow = gpu_farneback_calculator.calc(gpu_prev_gray_mat, gpu_current_gray_mat, None)
                        flow = temp_gpu_flow.download()
                except cv2.error as e:
                    log(f"WARNING: CUDA Farneback calc error (frame {frame_count}): {e}. Fallback CPU.")
                    gpu_active_for_this_video = False
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, **farneback_params_cpu)
            else:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, **farneback_params_cpu)

            if flow is not None and homography_matrix is not None:
                step = 15
                y_coords_crop, x_coords_crop = np.mgrid[step//2:prev_gray.shape[0]:step, step//2:prev_gray.shape[1]:step]
                active_grid_mask = np.ones(x_coords_crop.shape, dtype=bool)
                if is_quad_roi and roi_points_relative_to_crop is not None:
                    poly_mask = np.zeros(prev_gray.shape, dtype=np.uint8)
                    cv2.fillPoly(poly_mask, [np.array(roi_points_relative_to_crop, dtype=np.int32)], 255)
                    active_grid_mask = poly_mask[y_coords_crop, x_coords_crop] == 255
                
                x_c_flat = x_coords_crop[active_grid_mask].flatten()
                y_c_flat = y_coords_crop[active_grid_mask].flatten()
                if len(x_c_flat) > 0:
                    flow_dx = flow[y_c_flat, x_c_flat, 0]
                    flow_dy = flow[y_c_flat, x_c_flat, 1]
                    p1_orig = np.vstack((x_c_flat + x_b, y_c_flat + y_b)).T.reshape(-1,1,2).astype(np.float32)
                    p2_orig = np.vstack((x_c_flat + flow_dx + x_b, y_c_flat + flow_dy + y_b)).T.reshape(-1,1,2).astype(np.float32)
                    if p1_orig.size > 0:
                        world_coords_p1 = cv2.perspectiveTransform(p1_orig, homography_matrix)
                        world_coords_p2 = cv2.perspectiveTransform(p2_orig, homography_matrix)
                        if world_coords_p1 is not None and world_coords_p2 is not None:
                            dxm = world_coords_p2[:,0,0] - world_coords_p1[:,0,0]
                            dym = world_coords_p2[:,0,1] - world_coords_p1[:,0,1]
                            disp_m = np.sqrt(dxm**2 + dym**2)
                            valid_m = disp_m[(disp_m > 1e-4) & (disp_m < 5.0)]
                            current_frame_displacements_meters.extend(valid_m)
        else: # Sparse (Lucas-Kanade)
            if p0 is None or len(p0) < 10:
                gftt_mask_rt = None
                if is_quad_roi and roi_points_relative_to_crop is not None:
                    gftt_mask_rt = np.zeros(prev_gray.shape, dtype=np.uint8)
                    cv2.fillPoly(gftt_mask_rt, [np.array(roi_points_relative_to_crop, dtype=np.int32)], 255)
                p0_new = cv2.goodFeaturesToTrack(prev_gray, mask=gftt_mask_rt, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
                if p0_new is not None and len(p0_new) > 0:
                    p0 = p0_new
                else:
                    prev_gray = current_gray.copy()
                    frame_count += 1
                    if max_frames_to_process and frame_count >= max_frames_to_process:
                        break
                    continue
            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, p0, None, **lk_params)
                if p1 is not None and st is not None:
                    good_new_c = p1[st==1]
                    good_old_c = p0[st==1]
                    good_new_f = good_new_c
                    good_old_f = good_old_c
                    if is_quad_roi and roi_points_relative_to_crop is not None and len(good_old_c) > 0:
                        filt_new = []; filt_old = []
                        for i, pt_old_c in enumerate(good_old_c):
                            if cv2.pointPolygonTest(np.array(roi_points_relative_to_crop,dtype=np.int32), tuple(pt_old_c.ravel()), False) >=0 and \
                               cv2.pointPolygonTest(np.array(roi_points_relative_to_crop,dtype=np.int32), tuple(good_new_c[i].ravel()), False) >=0:
                                filt_old.append(pt_old_c)
                                filt_new.append(good_new_c[i])
                        if filt_new:
                            good_new_f=np.array(filt_new)
                            good_old_f=np.array(filt_old)
                        else:
                            good_new_f=np.array([])
                            good_old_f=np.array([])
                    if len(good_new_f) > 0 and homography_matrix is not None:
                        p_old_orig = good_old_f + np.array([x_b,y_b],dtype=np.float32)
                        p_new_orig = good_new_f + np.array([x_b,y_b],dtype=np.float32)
                        world_coords_old = cv2.perspectiveTransform(p_old_orig.reshape(-1,1,2), homography_matrix)
                        world_coords_new = cv2.perspectiveTransform(p_new_orig.reshape(-1,1,2), homography_matrix)
                        if world_coords_old is not None and world_coords_new is not None:
                            dxm = world_coords_new[:,0,0]-world_coords_old[:,0,0]
                            dym = world_coords_new[:,0,1]-world_coords_old[:,0,1]
                            disp_m = np.sqrt(dxm**2 + dym**2)
                            valid_m = disp_m[(disp_m > 1e-4) & (disp_m < 5.0)]
                            current_frame_displacements_meters.extend(valid_m)
                        p0 = good_new_f.reshape(-1,1,2)
                    elif len(good_new_f) > 0:
                        p0 = good_new_f.reshape(-1,1,2)
                    else:
                        p0 = None
                else:
                    p0 = None
            else:
                pass # p0 was None, skip LK
        if homography_matrix is not None and current_frame_displacements_meters:
            avg_disp_m = np.mean(current_frame_displacements_meters)
            vel_mps_frame = avg_disp_m * fps
            velocities_mps.append(vel_mps_frame)
        prev_gray = current_gray.copy()
        frame_count += 1
        if max_frames_to_process and frame_count >= max_frames_to_process:
            break
    cap.release()
    if not velocities_mps:
        log(f"Warning [{os.path.basename(video_path)}]: No valid METRIC velocities calculated ({frame_count} frames). Homography setup might be missing/failed or no valid flow.")
        return None
    final_avg_vel = np.mean(velocities_mps)
    log(f"INFO [{os.path.basename(video_path)}]: Final avg METRIC vel ({len(velocities_mps)} valid frame avgs): {final_avg_vel:.3f} m/s")
    return final_avg_vel


class WaterVelocityGUI:
    def __init__(self, master):
        self.master = master
        master.title("Water Velocity Estimator")
        self.folder_path_var = tk.StringVar()
        self.roi_var = tk.StringVar()
        self.dist_12_cm = tk.DoubleVar(value=0.0); self.dist_23_cm = tk.DoubleVar(value=0.0)
        self.dist_34_cm = tk.DoubleVar(value=0.0); self.dist_41_cm = tk.DoubleVar(value=0.0)
        self.roi_side_lengths_set = False
        self.extensions_var = tk.StringVar(value="mp4,avi,mov,mkv")
        self.use_dense_flow_var = tk.BooleanVar(value=False); self.use_gpu_var = tk.BooleanVar(value=False)
        self.skip_frames_var = tk.StringVar(value="0"); self.max_frames_var = tk.StringVar()
        self.selected_video_for_setup_path = tk.StringVar()
        self.roi_drawing_points_pixels = []
        self.roi_drawing_frame_copy = None
        self.roi_drawing_window_name = "Draw ROI & Set Distances"
        self.max_display_width = 1280; self.max_display_height = 720
        self.homography_matrix = None
        self.ui_queue = queue.Queue()
        self.roi_interaction_active = False
        self.setup_ui()
        self.master.after(100, self.process_ui_queue)

    def process_ui_queue(self):
        try:
            while True: 
                task, args = self.ui_queue.get_nowait()
                if task == "get_distance":
                    pt_idx1, pt_idx2, next_task_info = args
                    self._show_distance_dialog(pt_idx1, pt_idx2, next_task_info)
                elif task == "log": self._log_message_gui(args)
                elif task == "finish_roi_interaction":
                    self._finalize_roi_interaction_from_thread(args) 
                elif task == "reset_distances_and_log":
                    self.dist_12_cm.set(0.0); self.dist_23_cm.set(0.0)
                    self.dist_34_cm.set(0.0); self.dist_41_cm.set(0.0)
                    self.roi_side_lengths_set = False
                    self.roi_var.set("")
                    self._log_message_gui(args)
                self.ui_queue.task_done()
        except queue.Empty: pass 
        finally:
            if self.master.winfo_exists(): self.master.after(100, self.process_ui_queue)

    def _show_distance_dialog(self, pt_idx1, pt_idx2, next_task_info):
        self._log_message_gui(f"[DIALOG_SHOW] Attempting to show for P{pt_idx1}-P{pt_idx2}.")
        if self.master.state() == 'withdrawn':
            self._log_message_gui("[DIALOG_SHOW] Main window was withdrawn, deiconifying.")
            self.master.deiconify()
        self.master.lift()
        self.master.focus_force() # Attempt to force focus to main window before dialog
        self.master.update_idletasks() 

        title = f"Distance P{pt_idx1}-P{pt_idx2}"
        prompt_msg = f"Enter real-world distance between Point {pt_idx1} and Point {pt_idx2} (in cm):"
        distance_cm_input = simpledialog.askfloat(title, prompt_msg, parent=self.master, minvalue=0.001)
        
        self._log_message_gui(f"[DIALOG_SHOW] P{pt_idx1}-P{pt_idx2} askfloat returned: {distance_cm_input}")
        self.master.attributes('-topmost', False) 

        target_var = None # Determine which DoubleVar to update
        if (pt_idx1, pt_idx2) == (1, 2): target_var = self.dist_12_cm
        elif (pt_idx1, pt_idx2) == (2, 3): target_var = self.dist_23_cm
        elif (pt_idx1, pt_idx2) == (3, 4): target_var = self.dist_34_cm
        elif (pt_idx1, pt_idx2) == (4, 1): target_var = self.dist_41_cm

        if distance_cm_input is not None and distance_cm_input > 0:
            if target_var:
                target_var.set(distance_cm_input)
                # Log the value directly from the DoubleVar after setting it
                self._log_message_gui(f"Distance P{pt_idx1}-P{pt_idx2} successfully set to: {target_var.get():.2f} cm")
            
            if next_task_info and next_task_info.get("type") == "next_distance":
                next_pt1, next_pt2 = next_task_info["points"]
                self.ui_queue.put(("get_distance", (next_pt1, next_pt2, None))) 
        elif distance_cm_input is not None:
            if target_var: target_var.set(0.0) # Reset to 0 if invalid positive value was entered
            self._log_message_gui(f"Invalid distance for P{pt_idx1}-P{pt_idx2} ({distance_cm_input}). Must be > 0. Value set to 0.")
        else:
            if target_var: target_var.set(0.0) # Reset to 0 if dialog cancelled
            self._log_message_gui(f"Distance input for P{pt_idx1}-P{pt_idx2} cancelled by user (dialog returned None). Value set to 0.")


    def setup_ui(self):
        main_frame = ttk.Frame(self.master, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        input_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="10"); input_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew"); input_frame.columnconfigure(1, weight=1)
        row_idx = 0
        ttk.Label(input_frame, text="Video Folder:").grid(row=row_idx, column=0, sticky="w", pady=2)
        ttk.Entry(input_frame, textvariable=self.folder_path_var, width=40).grid(row=row_idx, column=1, sticky="ew", pady=2)
        ttk.Button(input_frame, text="Browse...", command=self.browse_folder).grid(row=row_idx, column=2, padx=5, pady=2); row_idx += 1
        setup_video_group = ttk.LabelFrame(input_frame, text="Setup Video for Interactive ROI", padding="5"); setup_video_group.grid(row=row_idx, column=0, columnspan=3, sticky="ew", pady=5); setup_video_group.columnconfigure(1, weight=1); sv_row = 0
        self.select_setup_video_button = ttk.Button(setup_video_group, text="Select Video for Setup", command=self.select_video_for_setup); self.select_setup_video_button.grid(row=sv_row, column=0, pady=(5,2), sticky="w", padx=2)
        self.selected_setup_video_label = ttk.Label(setup_video_group, text="No video selected.", wraplength=350, justify=tk.LEFT); self.selected_setup_video_label.grid(row=sv_row, column=1, columnspan=2, pady=(5,2), sticky="ew", padx=2); row_idx += 1
        roi_dist_group = ttk.LabelFrame(input_frame, text="Interactive ROI & Side Lengths (for Perspective Correction)", padding="5"); roi_dist_group.grid(row=row_idx, column=0, columnspan=3, sticky="ew", pady=5); roi_dist_group.columnconfigure(1, weight=1); rd_row = 0
        self.draw_roi_button = ttk.Button(roi_dist_group, text="Define ROI & Enter Side Lengths...", command=self.start_roi_interaction_thread, state=tk.DISABLED); self.draw_roi_button.grid(row=rd_row, column=0, columnspan=3, pady=5, padx=2, sticky="ew"); rd_row += 1
        ttk.Label(roi_dist_group, text="ROI Pixels (x1,y1..):").grid(row=rd_row, column=0, sticky="w", pady=1, padx=2)
        roi_display_entry = ttk.Entry(roi_dist_group, textvariable=self.roi_var, state='readonly', width=40); roi_display_entry.grid(row=rd_row, column=1, columnspan=2, sticky="ew", pady=1, padx=2); rd_row += 1
        ttk.Label(roi_dist_group, text="Dist P1-P2 (cm):").grid(row=rd_row, column=0, sticky="w", pady=1, padx=2); ttk.Entry(roi_dist_group, textvariable=self.dist_12_cm, state='readonly', width=10).grid(row=rd_row, column=1, sticky="w", pady=1, padx=2)
        ttk.Label(roi_dist_group, text="Dist P2-P3 (cm):").grid(row=rd_row, column=2, sticky="w", pady=1, padx=2); ttk.Entry(roi_dist_group, textvariable=self.dist_23_cm, state='readonly', width=10).grid(row=rd_row, column=3, sticky="w", pady=1, padx=2); rd_row += 1
        ttk.Label(roi_dist_group, text="Dist P3-P4 (cm):").grid(row=rd_row, column=0, sticky="w", pady=1, padx=2); ttk.Entry(roi_dist_group, textvariable=self.dist_34_cm, state='readonly', width=10).grid(row=rd_row, column=1, sticky="w", pady=1, padx=2)
        ttk.Label(roi_dist_group, text="Dist P4-P1 (cm):").grid(row=rd_row, column=2, sticky="w", pady=1, padx=2); ttk.Entry(roi_dist_group, textvariable=self.dist_41_cm, state='readonly', width=10).grid(row=rd_row, column=3, sticky="w", pady=1, padx=2); row_idx += 1
        other_params_group = ttk.LabelFrame(input_frame, text="Processing Options", padding="5"); other_params_group.grid(row=row_idx, column=0, columnspan=3, sticky="ew", pady=5); op_row = 0
        ttk.Label(other_params_group, text="Video Ext (comma-sep):").grid(row=op_row, column=0, sticky="w", pady=2, padx=2); ttk.Entry(other_params_group, textvariable=self.extensions_var).grid(row=op_row, column=1, columnspan=2, sticky="ew", pady=2, padx=2); op_row += 1
        ttk.Label(other_params_group, text="Skip Initial Frames:").grid(row=op_row, column=0, sticky="w", pady=2, padx=2); ttk.Entry(other_params_group, textvariable=self.skip_frames_var).grid(row=op_row, column=1, columnspan=2, sticky="ew", pady=2, padx=2); op_row += 1
        ttk.Label(other_params_group, text="Max Frames (optional):").grid(row=op_row, column=0, sticky="w", pady=2, padx=2); ttk.Entry(other_params_group, textvariable=self.max_frames_var).grid(row=op_row, column=1, columnspan=2, sticky="ew", pady=2, padx=2); op_row += 1
        ttk.Checkbutton(other_params_group, text="Use Dense Optical Flow (Farneback)", variable=self.use_dense_flow_var).grid(row=op_row, column=0, columnspan=3, sticky="w", pady=5, padx=2); op_row += 1
        self.gpu_checkbutton = ttk.Checkbutton(other_params_group, text="Use GPU for Dense Flow (if available, requires CUDA)", variable=self.use_gpu_var); self.gpu_checkbutton.grid(row=op_row, column=0, columnspan=3, sticky="w", pady=5, padx=2)
        if not CUDA_AVAILABLE: self.gpu_checkbutton.config(state=tk.DISABLED); self.use_gpu_var.set(False)
        row_idx +=1 
        control_frame = ttk.Frame(main_frame, padding="5"); control_frame.grid(row=row_idx, column=0, sticky="ew", pady=5)
        self.start_button = ttk.Button(control_frame, text="Start Processing", command=self.start_processing_thread); self.start_button.pack(pady=5)
        output_frame = ttk.LabelFrame(main_frame, text="Log & Results", padding="10"); output_frame.grid(row=row_idx + 1, column=0, padx=5, pady=5, sticky="nsew") 
        self.log_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=10, width=80); self.log_text.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=1); main_frame.rowconfigure(row_idx + 1, weight=1)

    def browse_folder(self):
        f = filedialog.askdirectory();
        if f: self.folder_path_var.set(f)

    def select_video_for_setup(self):
        vp = filedialog.askopenfilename(title="Select Video for ROI Setup", filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")))
        if vp:
            self.selected_video_for_setup_path.set(vp); self.draw_roi_button.config(state=tk.NORMAL)
            lt = f"Setup Video: {os.path.basename(vp)}"; cap = cv2.VideoCapture(vp)
            if cap.isOpened(): r,fr = cap.read(); lt += f" ({fr.shape[1]}x{fr.shape[0]} px)" if r else " (Err read)"; cap.release()
            else: lt += " (Err open)"
            self.selected_setup_video_label.config(text=lt); self._log_message_gui(f"Video for setup: {os.path.basename(vp)}")
        else:
            self.selected_video_for_setup_path.set(""); self.draw_roi_button.config(state=tk.DISABLED)
            self.selected_setup_video_label.config(text="No video selected.")
            self.ui_queue.put(("reset_distances_and_log", "Video selection cancelled, ROI data cleared."))

    def start_roi_interaction_thread(self):
        video_path = self.selected_video_for_setup_path.get()
        if not video_path: self._log_message_gui("Error: No video selected for ROI setup."); return
        self.roi_drawing_points_pixels = []
        self.roi_var.set("")
        self.dist_12_cm.set(0.0); self.dist_23_cm.set(0.0); self.dist_34_cm.set(0.0); self.dist_41_cm.set(0.0)
        self.roi_side_lengths_set = False; self.roi_interaction_active = True 
        self.master.withdraw() 
        threading.Thread(target=self._roi_interaction_cv_thread, args=(video_path,), daemon=True).start()

    def _roi_interaction_cv_thread(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.ui_queue.put(("log", f"Error opening {os.path.basename(video_path)} in CV thread."))
            self.ui_queue.put(("finish_roi_interaction", False)); return
        ret, frame_for_roi_orig = cap.read(); cap.release()
        if not ret:
            self.ui_queue.put(("log", f"Error reading frame from {os.path.basename(video_path)} in CV thread."))
            self.ui_queue.put(("finish_roi_interaction", False)); return

        cv_thread_collected_points = []
        cv_thread_frame_copy = frame_for_roi_orig.copy()
        window_name = self.roi_drawing_window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        callback_data = {
            "points_list_ref": cv_thread_collected_points,
            "original_frame": frame_for_roi_orig,
            "current_display_frame_ref": [cv_thread_frame_copy], 
            "ui_queue_ref": self.ui_queue,
            "window_name_ref": window_name
        }

        def _thread_mouse_callback_internal(event, x, y, flags, param_dict):
            points_list = param_dict["points_list_ref"]
            orig_frame = param_dict["original_frame"]
            display_frame_wrapper = param_dict["current_display_frame_ref"]
            q = param_dict["ui_queue_ref"]
            win_name = param_dict["window_name_ref"]

            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points_list) < 4:
                    points_list.append((x,y))
                    temp_draw_frame = orig_frame.copy()
                    for i, pt_draw in enumerate(points_list):
                        cv2.circle(temp_draw_frame, pt_draw, 7, (0,255,0), -1)
                        cv2.putText(temp_draw_frame, f"P{i+1}", (pt_draw[0]+10, pt_draw[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)
                        if i > 0: cv2.line(temp_draw_frame, points_list[i-1], pt_draw, (0,255,0),2)
                    display_frame_wrapper[0] = temp_draw_frame
                    cv2.imshow(win_name, display_frame_wrapper[0])

                    if len(points_list) == 2: q.put(("get_distance",(1,2,None)))
                    elif len(points_list) == 3: q.put(("get_distance",(2,3,None)))
                    elif len(points_list) == 4:
                        q.put(("get_distance",(3,4,{"type":"next_distance","points":(4,1)})))
                        cv2.line(display_frame_wrapper[0], points_list[3], points_list[0], (0,0,255), 2)
                        cv2.imshow(win_name, display_frame_wrapper[0])
        
        cv2.setMouseCallback(window_name, _thread_mouse_callback_internal, callback_data)
        fh,fw=frame_for_roi_orig.shape[:2]; dw,dh=fw,fh
        if fw>self.max_display_width or fh>self.max_display_height: scale=min(self.max_display_width/fw, self.max_display_height/fh); dw,dh=int(fw*scale),int(fh*scale)
        cv2.resizeWindow(window_name,dw,dh); cv2.imshow(window_name,callback_data["current_display_frame_ref"][0])
        self.ui_queue.put(("log", f"ROI Setup: {window_name}. Click 4 points. Dialogs for side lengths (cm). ENTER to Confirm, R to Reset, ESC/C to Cancel."))
        
        cv_interaction_confirmed_by_enter = False
        while True:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: 
                self.ui_queue.put(("log", "CV window closed by user (X button). ROI setup cancelled."))
                cv_interaction_confirmed_by_enter = False
                break
            key = cv2.waitKey(50) & 0xFF
            if key == 13: 
                if len(cv_thread_collected_points) == 4:
                    self.ui_queue.put(("log", "CV window: Enter pressed with 4 points."))
                    cv_interaction_confirmed_by_enter = True; break
                else:
                    self.ui_queue.put(("log", "CV window: Enter pressed, but less than 4 points clicked. Please click 4 points first."))
            elif key == ord('r') or key == ord('R'):
                self.ui_queue.put(("log", "CV window: 'R' pressed. Resetting points and distances."))
                cv_thread_collected_points.clear()
                callback_data["current_display_frame_ref"][0] = frame_for_roi_orig.copy()
                cv2.imshow(window_name, callback_data["current_display_frame_ref"][0])
                self.ui_queue.put(("reset_distances_and_log", "ROI points and distances reset. Click 4 new points."))
            elif key == 27 or key == ord('c') or key == ord('C'):
                self.ui_queue.put(("log", "CV window: ESC/C pressed. ROI setup cancelled."))
                cv_interaction_confirmed_by_enter = False
                break
        
        cv2.destroyWindow(window_name)
        if cv_interaction_confirmed_by_enter and len(cv_thread_collected_points) == 4:
            self.roi_drawing_points_pixels = list(cv_thread_collected_points)
        else:
            self.roi_drawing_points_pixels = []
            
        self.ui_queue.put(("finish_roi_interaction", cv_interaction_confirmed_by_enter and len(self.roi_drawing_points_pixels) == 4))

    def _finalize_roi_interaction_from_thread(self, cv_points_and_enter_confirmed_flag):
        self.master.deiconify()
        self.master.lift(); self.master.focus_force(); self.master.update_idletasks()
        self.master.attributes('-topmost', False)
        self.roi_interaction_active = False

        self._log_message_gui(f"[FINALIZE_ROI] CV points & Enter confirmed flag from CV thread: {cv_points_and_enter_confirmed_flag}")
        self._log_message_gui(f"[FINALIZE_ROI] Number of ROI pixel points transferred to GUI (self.roi_drawing_points_pixels): {len(self.roi_drawing_points_pixels)}")
        d12 = self.dist_12_cm.get(); d23 = self.dist_23_cm.get()
        d34 = self.dist_34_cm.get(); d41 = self.dist_41_cm.get()
        self._log_message_gui(f"[FINALIZE_ROI] Distances (cm) from Tkinter vars: P1P2:{d12:.2f}, P2P3:{d23:.2f}, P3P4:{d34:.2f}, P4P1:{d41:.2f}")

        all_distances_entered_validly = (d12 > 0 and d23 > 0 and d34 > 0 and d41 > 0)
        self._log_message_gui(f"[FINALIZE_ROI] All 4 distances entered validly (>0 cm): {all_distances_entered_validly}")

        final_setup_success = cv_points_and_enter_confirmed_flag and \
                              len(self.roi_drawing_points_pixels) == 4 and \
                              all_distances_entered_validly
        
        self._log_message_gui(f"[FINALIZE_ROI] Overall setup success determination: {final_setup_success}")

        if final_setup_success:
            flat_coords = [coord for point in self.roi_drawing_points_pixels for coord in point]
            self.roi_var.set(",".join(map(str, flat_coords)))
            self.roi_side_lengths_set = True
            self._log_message_gui(f"SUCCESS: ROI (pixels) set: {self.roi_var.get()}")
            self._log_message_gui(f"SUCCESS: Side lengths (cm) also set and valid. Perspective correction WILL be attempted.")
        else:
            self.roi_var.set("")
            self.roi_side_lengths_set = False
            log_msg = "FAILURE: Interactive ROI & Distance setup FAILED or was Incomplete. Conditions for homography NOT met:\n"
            if not cv_points_and_enter_confirmed_flag: log_msg += "  - CV interaction (4 points + Enter key) was not completed or window closed.\n"
            # Redundant check, covered by cv_points_and_enter_confirmed_flag being true only if len is 4
            # elif len(self.roi_drawing_points_pixels) != 4: log_msg += f"  - Expected 4 ROI pixel points, got {len(self.roi_drawing_points_pixels)}.\n"
            if not all_distances_entered_validly: log_msg += "  - Not all 4 side distances were entered correctly (must be >0cm).\n"
            self._log_message_gui(log_msg.strip())

    def _log_message_gui(self, message):
        if self.master.winfo_exists():
            self.log_text.insert(tk.END, str(message) + "\n")
            self.log_text.see(tk.END); self.master.update_idletasks()

    def log_message_thread_safe(self, message):
        if self.master.winfo_exists():
            self.master.after(0, self._log_message_gui, message)

    def _processing_job(self):
        self.homography_matrix = None
        try:
            folder_path = self.folder_path_var.get()
            if not folder_path or not os.path.isdir(folder_path):
                self.log_message_thread_safe("Error: Please select a valid video folder."); return

            parsed_roi_pixel_coords_list = None
            src_points_for_homography_pixels_np = None
            roi_str = self.roi_var.get().strip()

            if self.roi_side_lengths_set and roi_str:
                self.log_message_thread_safe("[PROC_JOB] self.roi_side_lengths_set is TRUE and roi_str is not empty.")
                try:
                    coords = list(map(int, roi_str.split(',')))
                    if len(coords) == 8:
                        parsed_roi_pixel_coords_list = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
                        src_points_for_homography_pixels_np = np.array(parsed_roi_pixel_coords_list, dtype=np.float32)
                        
                        self.log_message_thread_safe("[PROC_JOB] ROI pixels parsed successfully. Attempting homography calculation.")
                        l12m,l23m,l34m,l41m = self.dist_12_cm.get()/100.0, self.dist_23_cm.get()/100.0, \
                                             self.dist_34_cm.get()/100.0, self.dist_41_cm.get()/100.0
                        if not (l12m > 0 and l23m > 0 and l34m > 0 and l41m > 0):
                            self.log_message_thread_safe("WARNING [PROC_JOB]: Distances became invalid after finalization. No homography.")
                            self.homography_matrix = None
                        else:
                            avg_width_m = (l12m + l34m) / 2.0
                            avg_height_m = (l23m + l41m) / 2.0
                            dst_points_world_meters_np = np.array([
                                [0, 0], [avg_width_m, 0],
                                [avg_width_m, avg_height_m], [0, avg_height_m]
                            ], dtype=np.float32)
                            self.homography_matrix, status = cv2.findHomography(
                                src_points_for_homography_pixels_np, dst_points_world_meters_np)
                            if self.homography_matrix is None:
                                self.log_message_thread_safe("WARNING [PROC_JOB]: Homography calculation failed. Metric velocity cannot be calculated.")
                            else:
                                self.log_message_thread_safe(f"INFO [PROC_JOB]: Homography matrix calculated. Target world rect: {avg_width_m:.2f}m x {avg_height_m:.2f}m.")
                    else:
                        self.log_message_thread_safe(f"WARNING [PROC_JOB]: roi_var string '{roi_str}' not valid for 4 points. No homography.")
                except ValueError as e:
                    self.log_message_thread_safe(f"WARNING [PROC_JOB]: Error processing ROI/Distances for homography: {e}. No homography.")
                except Exception as e:
                    self.log_message_thread_safe(f"WARNING [PROC_JOB]: General error during homography matrix calculation: {e}. No homography.")
            else:
                self.log_message_thread_safe(f"[PROC_JOB] Conditions for homography not met. self.roi_side_lengths_set: {self.roi_side_lengths_set}, roi_str empty: {not roi_str}. No homography.")

            extensions = [ext.strip().lower() for ext in self.extensions_var.get().split(',') if ext.strip()]
            if not extensions: self.log_message_thread_safe("Error: No valid video extensions provided."); return
            try: skip_frames = int(self.skip_frames_var.get()); assert skip_frames >= 0
            except: self.log_message_thread_safe("Error: Invalid Skip Frames."); return
            max_frames_str = self.max_frames_var.get().strip(); max_frames = None
            if max_frames_str:
                try: max_frames = int(max_frames_str); assert max_frames > 0
                except: self.log_message_thread_safe("Error: Invalid Max Frames."); return
            use_dense = self.use_dense_flow_var.get(); use_gpu_flag = self.use_gpu_var.get()
            video_files = sorted(list(set(sum([glob.glob(os.path.join(folder_path, f"*.{ext}")) + \
                                               glob.glob(os.path.join(folder_path, f"*.{ext.upper()}")) for ext in extensions], []))))
            if not video_files: self.log_message_thread_safe(f"No videos with specified extensions found in {folder_path}"); return
            
            self.log_message_thread_safe(f"Found {len(video_files)} video(s) to process.\n")
            if self.homography_matrix is None:
                 self.log_message_thread_safe("CRITICAL INFO: No valid Homography Matrix for this run. Metric velocities will NOT be calculated.")

            results = {}
            for video_file in video_files:
                self.log_message_thread_safe(f"--- Processing {os.path.basename(video_file)} ---")
                avg_velocity = estimate_water_velocity(
                    video_file, roi=parsed_roi_pixel_coords_list,
                    use_dense_flow=use_dense, use_gpu=use_gpu_flag,
                    skip_frames=skip_frames, max_frames_to_process=max_frames,
                    log_callback=self.log_message_thread_safe,
                    homography_matrix=self.homography_matrix)
                if avg_velocity is not None:
                    self.log_message_thread_safe(f"==> Estimated avg METRIC velocity for {os.path.basename(video_file)}: {avg_velocity:.3f} m/s\n")
                    results[os.path.basename(video_file)] = avg_velocity
                else:
                    self.log_message_thread_safe(f"==> Could not estimate METRIC velocity for {os.path.basename(video_file)}. Check logs.\n")
                    results[os.path.basename(video_file)] = "N/A (Metric)"
            
            self.log_message_thread_safe("\n--- Summary ---")
            if results:
                for video_name, vel in results.items():
                    if isinstance(vel, float): self.log_message_thread_safe(f"{video_name}: {vel:.3f} m/s")
                    else: self.log_message_thread_safe(f"{video_name}: {vel}")
            else: self.log_message_thread_safe("No videos yielded results (or metric calculation was not possible).")
            self.log_message_thread_safe("\nProcessing complete.")
        except Exception as e:
            self.log_message_thread_safe(f"Unexpected error in processing job: {e}")
            import traceback; self.log_message_thread_safe(traceback.format_exc())
        finally:
            if self.master.winfo_exists():
                self.master.after(0, lambda: self.start_button.config(state=tk.NORMAL) if self.master.winfo_exists() else None)

    def start_processing_thread(self):
        if not self.master.winfo_exists(): return
        self.log_text.delete('1.0', tk.END)
        self._log_message_gui("Starting processing... Please wait.")
        self.start_button.config(state=tk.DISABLED)
        threading.Thread(target=self._processing_job, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    gui = WaterVelocityGUI(root)
    root.mainloop()