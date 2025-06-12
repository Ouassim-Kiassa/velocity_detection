import cv2
import numpy as np
import os
import glob
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading

# --- Optical Flow Parameters ---
# For Lucas-Kanade (Sparse)
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# For Farneback (Dense)
# No specific params needed here, but can be tuned in cv2.calcOpticalFlowFarneback

def estimate_water_velocity(video_path, meters_per_pixel, roi=None, use_dense_flow=False, skip_frames=0, max_frames_to_process=None, log_callback=None):
    """
    Estimates water velocity from a video using optical flow.
    Args:
        video_path (str): Path to the video file.
        meters_per_pixel (float): Calibration factor (real-world meters represented by one pixel).
        roi (tuple, optional): (x, y, w, h) for the Region of Interest. Defaults to None (full frame).
        use_dense_flow (bool): If True, use Farneback dense optical flow. Else, Lucas-Kanade sparse.
        skip_frames (int): Number of initial frames to skip.
        max_frames_to_process (int, optional): Maximum number of frames to process after skipping.
                                              None means process all available frames.
        log_callback (function, optional): Callback function to log messages. Takes a string argument.
    Returns:
        float: Estimated average water velocity in m/s, or None if processing failed.
    """
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message) # Fallback to console print

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"Error: Could not open video {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        log(f"Error: Could not get FPS for {video_path}. Assuming 30 FPS.")
        fps = 30.0 # Fallback

    # Skip initial frames
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

    if roi:
        x, y, w, h = roi
        fh, fw = prev_frame.shape[:2]
        if x < 0 or y < 0 or w <= 0 or h <= 0 or (x + w) > fw or (y + h) > fh:
            log(f"Error: ROI ({x},{y},{w},{h}) is outside frame dimensions ({fw}x{fh}) or invalid for {video_path}.")
            cap.release()
            return None
        prev_frame_roi = prev_frame[y:y+h, x:x+w]
    else:
        prev_frame_roi = prev_frame

    prev_gray = cv2.cvtColor(prev_frame_roi, cv2.COLOR_BGR2GRAY)

    p0 = None 
    if not use_dense_flow:
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        if p0 is None or len(p0) == 0:
            log(f"Warning: No good features to track in the first ROI frame of {video_path} (LK). Processing cannot continue for this video with LK.")
            cap.release()
            return None

    velocities_mps = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break 

        if roi:
            x, y, w, h = roi 
            frame_roi = frame[y:y+h, x:x+w]
        else:
            frame_roi = frame

        current_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
        frame_velocities_px_per_frame = []

        if use_dense_flow:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None,
                                                pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            valid_flow_magnitudes = mag[(mag > 0.1) & (mag < 100)] 
            if len(valid_flow_magnitudes) > 0:
                avg_pixel_displacement = np.mean(valid_flow_magnitudes)
                frame_velocities_px_per_frame.append(avg_pixel_displacement)

        else: # Sparse (Lucas-Kanade)
            if p0 is None or len(p0) == 0: 
                p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
                if p0 is None or len(p0) == 0:
                    prev_gray = current_gray.copy() 
                    frame_count += 1
                    if max_frames_to_process and frame_count >= max_frames_to_process:
                        break
                    continue 

            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, p0, None, **lk_params)

            if p1 is not None and st is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                if len(good_new) > 0:
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        displacement = np.sqrt((a - c)**2 + (b - d)**2)
                        frame_velocities_px_per_frame.append(displacement)
                    p0 = good_new.reshape(-1, 1, 2) 
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
        self.roi_var = tk.StringVar()
        self.extensions_var = tk.StringVar(value="mp4,avi,mov,mkv")
        self.use_dense_flow_var = tk.BooleanVar(value=False)
        self.skip_frames_var = tk.StringVar(value="0")
        self.max_frames_var = tk.StringVar()
        self.selected_video_for_roi_path = tk.StringVar()

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

        ttk.Label(input_frame, text="Meters/Pixel:").grid(row=row_idx, column=0, sticky="w", pady=2)
        ttk.Entry(input_frame, textvariable=self.mpp_var).grid(row=row_idx, column=1, columnspan=2, sticky="ew", pady=2)
        row_idx += 1

        # --- ROI Selection Frame ---
        roi_outer_frame = ttk.Frame(input_frame) # Use a simple frame for better layout control
        roi_outer_frame.grid(row=row_idx, column=0, columnspan=3, sticky="ew", pady=2)
        roi_outer_frame.columnconfigure(1, weight=1) # Make entry field expand

        ttk.Label(roi_outer_frame, text="ROI (x,y,w,h):").grid(row=0, column=0, sticky="w", pady=(0,2))
        ttk.Entry(roi_outer_frame, textvariable=self.roi_var).grid(row=0, column=1, columnspan=2, sticky="ew", padx=(0,5), pady=(0,2))

        self.select_roi_video_button = ttk.Button(roi_outer_frame, text="Select Video for ROI", command=self.select_video_for_roi_drawing)
        self.select_roi_video_button.grid(row=1, column=0, pady=(2,0), sticky="w")
        
        self.draw_roi_button = ttk.Button(roi_outer_frame, text="Draw ROI on Video", command=self.draw_roi_on_frame, state=tk.DISABLED)
        self.draw_roi_button.grid(row=1, column=1, pady=(2,0), sticky="w", padx=(0,5))
        
        self.selected_roi_video_label = ttk.Label(roi_outer_frame, text="No video selected for ROI.", wraplength=200, justify=tk.LEFT)
        self.selected_roi_video_label.grid(row=1, column=2, pady=(2,0), sticky="ew")
        row_idx += 1
        # --- End ROI Selection Frame ---

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

    def select_video_for_roi_drawing(self):
        video_path = filedialog.askopenfilename(
            title="Select Video for ROI Drawing",
            filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*"))
        )
        if video_path:
            self.selected_video_for_roi_path.set(video_path)
            self.draw_roi_button.config(state=tk.NORMAL)
            self.selected_roi_video_label.config(text=f"For ROI: {os.path.basename(video_path)}")
            self._log_message_gui(f"Video selected for ROI drawing: {os.path.basename(video_path)}")
        else:
            self.draw_roi_button.config(state=tk.DISABLED)
            self.selected_roi_video_label.config(text="No video selected for ROI.")

    def draw_roi_on_frame(self):
        video_path = self.selected_video_for_roi_path.get()
        if not video_path:
            self._log_message_gui("Error: No video selected. Use 'Select Video for ROI' first.")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log_message_gui(f"Error: Could not open video {os.path.basename(video_path)} for ROI.")
            return

        ret, frame = cap.read()
        cap.release() 

        if not ret:
            self._log_message_gui(f"Error: Could not read a frame from {os.path.basename(video_path)} for ROI.")
            return
        
        window_name = "Draw ROI - Press ENTER to confirm, C or ESC to cancel"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
        # Optional: Resize window to be smaller if frames are very large
        # h, w = frame.shape[:2]
        # if w > 1280 or h > 720:
        #    cv2.resizeWindow(window_name, w // 2, h // 2)


        self._log_message_gui("ROI selection window opened. Draw rectangle, press ENTER (confirm) or C/ESC (cancel).")
        
        self.master.withdraw() # Hide main Tkinter window

        r = None
        try:
            # Create a copy for drawing so the original `frame` isn't modified if needed later
            # Though in this case, `frame` is local and not reused.
            r = cv2.selectROI(window_name, frame.copy(), fromCenter=False, showCrosshair=True)
        except Exception as e:
            self._log_message_gui(f"Error during cv2.selectROI: {e}")
            r = (0,0,0,0) # Treat as cancel
        finally:
            cv2.destroyWindow(window_name)
            self.master.deiconify() # Bring back main Tkinter window
            self.master.attributes('-topmost', True) # Try to bring to front
            self.master.focus_force()
            self.master.attributes('-topmost', False)


        if r and r[2] > 0 and r[3] > 0:  
            x, y, w, h = r
            self.roi_var.set(f"{x},{y},{w},{h}")
            self._log_message_gui(f"ROI selected: x={x}, y={y}, w={w}, h={h}")
        else:
            self._log_message_gui("ROI selection cancelled or invalid (width/height is zero).")

    def _log_message_gui(self, message):
        if self.master.winfo_exists(): # Check if window still exists
            self.log_text.insert(tk.END, str(message) + "\n")
            self.log_text.see(tk.END)
            self.master.update_idletasks() 

    def log_message_thread_safe(self, message):
        if self.master.winfo_exists():
            self.master.after(0, self._log_message_gui, message)

    def _processing_job(self):
        try:
            # Parameter validation (as before)
            folder_path = self.folder_path_var.get()
            if not folder_path or not os.path.isdir(folder_path):
                self.log_message_thread_safe("Error: Please select a valid video folder.")
                return

            try:
                mpp = float(self.mpp_var.get())
                if mpp <= 0: raise ValueError("Meters per pixel must be positive.")
            except ValueError as e:
                self.log_message_thread_safe(f"Error: Invalid Meters/Pixel. {e}")
                return

            parsed_roi = None
            roi_str = self.roi_var.get().strip()
            if roi_str:
                try:
                    parsed_roi = tuple(map(int, roi_str.split(',')))
                    if len(parsed_roi) != 4: raise ValueError("ROI: 4 integers.")
                    if any(val < 0 for val in parsed_roi[:2]) or any(val <= 0 for val in parsed_roi[2:]):
                         raise ValueError("ROI x,y >= 0; w,h > 0.")
                except ValueError as e:
                    self.log_message_thread_safe(f"Error: Invalid ROI. {e}")
                    return
            
            extensions_str = self.extensions_var.get()
            if not extensions_str:
                self.log_message_thread_safe("Error: Video extensions cannot be empty.")
                return
            extensions = [ext.strip().lower() for ext in extensions_str.split(',') if ext.strip()]
            if not extensions:
                self.log_message_thread_safe("Error: No valid video extensions provided.")
                return

            try:
                skip_frames = int(self.skip_frames_var.get())
                if skip_frames < 0: raise ValueError("Skip frames >= 0.")
            except ValueError as e:
                self.log_message_thread_safe(f"Error: Invalid Skip Frames. {e}")
                return

            max_frames = None
            max_frames_str = self.max_frames_var.get().strip()
            if max_frames_str:
                try:
                    max_frames = int(max_frames_str)
                    if max_frames <= 0: raise ValueError("Max frames > 0 if set.")
                except ValueError as e:
                    self.log_message_thread_safe(f"Error: Invalid Max Frames. {e}")
                    return

            use_dense = self.use_dense_flow_var.get()
            # End parameter validation

            video_files = []
            for ext in extensions:
                video_files.extend(glob.glob(os.path.join(folder_path, f"*.{ext}")))
                video_files.extend(glob.glob(os.path.join(folder_path, f"*.{ext.upper()}")))
            video_files = sorted(list(set(video_files)))

            if not video_files:
                self.log_message_thread_safe(f"No videos with specified extensions found in {folder_path}")
                return

            self.log_message_thread_safe(f"Found {len(video_files)} video(s) to process.\n")
            results = {}

            for video_file in video_files:
                self.log_message_thread_safe(f"--- Processing {os.path.basename(video_file)} ---")
                avg_velocity = estimate_water_velocity(
                    video_file, mpp, roi=parsed_roi, use_dense_flow=use_dense,
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
                    if isinstance(vel, float):
                        self.log_message_thread_safe(f"{video_name}: {vel:.3f} m/s")
                    else:
                        self.log_message_thread_safe(f"{video_name}: {vel}")
            else:
                self.log_message_thread_safe("No videos yielded results.")
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