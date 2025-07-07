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
        CUDA_AVAILABLE = True; print("INFO: CUDA is available.")
    elif hasattr(cv2, 'cuda'): print("INFO: CUDA module present, no CUDA GPU.")
    else: print("INFO: OpenCV not compiled with CUDA.")
except Exception as e: print(f"WARNING: CUDA check error: {e}")

# --- Optical Flow Parameters ---
lk_params = dict(winSize=(15, 15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
farneback_params_cpu = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
farneback_params_gpu_create = dict(numLevels=3, pyrScale=0.5, fastPyramids=False, winSize=15, numIters=3, polyN=5, polySigma=1.2, flags=0)


# --- estimate_water_velocity function (from previous corrected version) ---
def estimate_water_velocity(video_path, roi_orig_frame_pixels=None,
                            use_dense_flow=False, use_gpu=False,
                            skip_frames=0, max_frames_to_process=None,
                            log_callback=None, homography_matrix_to_world=None,
                            show_bev_visualization=False,
                            bev_img_size=(500, 500),
                            H_bev_mapping=None):
    def log(message):
        if log_callback: log_callback(message)
        else: print(message)

    if homography_matrix_to_world is None:
        log_msg = f"WARNING [{os.path.basename(video_path)}]: Metric homography (homography_matrix_to_world) not provided. "
        if show_bev_visualization: log_msg += "BEV might not be to scale. "
        log_msg += "Cannot calculate metric velocity."
        log(log_msg)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): log(f"Error: Could not open video {video_path}"); return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: log(f"Warning: Could not get FPS for {video_path}. Assuming 30 FPS."); fps = 30.0

    for i in range(skip_frames):
        ret, _ = cap.read()
        if not ret: log(f"Error: Not enough frames to skip in {video_path}"); cap.release(); return None
    
    ret, prev_full_frame = cap.read()
    if not ret: log(f"Error: Could not read first frame after skipping in {video_path}."); cap.release(); return None

    fh_orig, fw_orig = prev_full_frame.shape[:2]
    is_quad_roi_for_processing = False
    roi_pts_relative_to_crop = None
    x_b, y_b, w_b, h_b = 0, 0, fw_orig, fh_orig
    prev_frame_for_flow = prev_full_frame

    if roi_orig_frame_pixels:
        if isinstance(roi_orig_frame_pixels, (list, tuple)) and len(roi_orig_frame_pixels) == 4 and \
           isinstance(roi_orig_frame_pixels[0], (list, tuple)) and len(roi_orig_frame_pixels[0]) == 2:
            roi_points_np_orig = np.array(roi_orig_frame_pixels, dtype=np.int32)
            if not all(0 <= px < fw_orig and 0 <= py < fh_orig for px, py in roi_points_np_orig):
                log(f"Error: ROI points outside original frame dimensions for {video_path}."); cap.release(); return None
            x_b, y_b, w_b, h_b = cv2.boundingRect(roi_points_np_orig)
            if not (w_b > 0 and h_b > 0): log(f"Error: ROI bounding box has zero area for {video_path}."); cap.release(); return None
            prev_frame_for_flow = prev_full_frame[y_b:y_b+h_b, x_b:x_b+w_b]
            roi_pts_relative_to_crop = roi_points_np_orig - np.array([x_b, y_b])
            is_quad_roi_for_processing = True
        else:
            log(f"INFO [{os.path.basename(video_path)}]: ROI not 4 points. Processing full frame for flow. No metric velocity/BEV.")
    
    if prev_frame_for_flow.size == 0: log(f"Error: Frame for flow is empty for {video_path}."); cap.release(); return None
    prev_gray_for_flow = cv2.cvtColor(prev_frame_for_flow, cv2.COLOR_BGR2GRAY)

    bev_background = None
    if show_bev_visualization:
        if H_bev_mapping is not None:
            try: bev_background = cv2.warpPerspective(prev_full_frame, H_bev_mapping, bev_img_size)
            except cv2.error as e: log(f"Error warping BEV background: {e}"); bev_background = np.zeros((bev_img_size[1], bev_img_size[0], 3), dtype=np.uint8)
        else: bev_background = np.zeros((bev_img_size[1], bev_img_size[0], 3), dtype=np.uint8)

    p0 = None; gpu_farneback_calculator = None; gpu_active_for_this_video = False
    if use_dense_flow and use_gpu and CUDA_AVAILABLE:
        try: gpu_farneback_calculator = cv2.cuda_FarnebackOpticalFlow.create(**farneback_params_gpu_create); gpu_active_for_this_video = True
        except cv2.error as e: log(f"WARN: CUDA Farneback create error: {e}. Fallback CPU.")
    if not use_dense_flow:
        gftt_mask = None
        if is_quad_roi_for_processing and roi_pts_relative_to_crop is not None:
            gftt_mask = np.zeros(prev_gray_for_flow.shape, dtype=np.uint8)
            cv2.fillPoly(gftt_mask, [np.array(roi_pts_relative_to_crop, dtype=np.int32)], 255)
        p0 = cv2.goodFeaturesToTrack(prev_gray_for_flow, mask=gftt_mask, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        if p0 is None or len(p0) == 0: log(f"WARN [{os.path.basename(video_path)}]: No good LK features in first frame/ROI.")
    
    gpu_flow_output_mat=None; gpu_prev_gray_mat=None; gpu_current_gray_mat=None
    if gpu_active_for_this_video:
        try: gpu_flow_output_mat=cv2.cuda_GpuMat(prev_gray_for_flow.shape[0],prev_gray_for_flow.shape[1],cv2.CV_32FC2);gpu_prev_gray_mat=cv2.cuda_GpuMat();gpu_current_gray_mat=cv2.cuda_GpuMat()
        except cv2.error as e: log(f"WARN: GpuMat alloc error: {e}"); gpu_active_for_this_video=False

    velocities_mps = []; frame_count_processed = 0
    bev_window_name = "Bird's Eye View Flow"

    while True:
        ret, current_full_frame = cap.read();
        if not ret: break

        current_frame_for_flow = current_full_frame[y_b:y_b+h_b, x_b:x_b+w_b] if is_quad_roi_for_processing else current_full_frame
        if current_frame_for_flow.size == 0: log(f"WARN: Empty frame for flow (frame {frame_count_processed}). Skip."); frame_count_processed+=1; continue
        current_gray_for_flow = cv2.cvtColor(current_frame_for_flow, cv2.COLOR_BGR2GRAY)
        current_frame_displacements_meters = []
        
        bev_image_with_vectors = None
        if show_bev_visualization:
            if H_bev_mapping is not None:
                 try: bev_image_with_vectors = cv2.warpPerspective(current_full_frame, H_bev_mapping, bev_img_size)
                 except cv2.error as e: log(f"Error warping BEV for frame {frame_count_processed}: {e}"); bev_image_with_vectors = np.zeros((bev_img_size[1], bev_img_size[0], 3),dtype=np.uint8) if bev_background is None else bev_background.copy()
            elif bev_background is not None: bev_image_with_vectors = bev_background.copy()
            else: bev_image_with_vectors = np.zeros((bev_img_size[1], bev_img_size[0], 3), dtype=np.uint8)

        if use_dense_flow:
            flow = None
            if gpu_active_for_this_video:
                try: gpu_prev_gray_mat.upload(prev_gray_for_flow);gpu_current_gray_mat.upload(current_gray_for_flow);gpu_farneback_calculator.calc(gpu_prev_gray_mat,gpu_current_gray_mat,gpu_flow_output_mat);flow=gpu_flow_output_mat.download()
                except cv2.error as e: log(f"WARN: CUDA Farneback calc error (frame {frame_count_processed}): {e}. Fallback CPU."); gpu_active_for_this_video=False; flow=cv2.calcOpticalFlowFarneback(prev_gray_for_flow,current_gray_for_flow,None,**farneback_params_cpu)
            else: flow=cv2.calcOpticalFlowFarneback(prev_gray_for_flow,current_gray_for_flow,None,**farneback_params_cpu)

            if flow is not None:
                if homography_matrix_to_world is not None:
                    step=15; y_crop,x_crop=np.mgrid[step//2:prev_gray_for_flow.shape[0]:step, step//2:prev_gray_for_flow.shape[1]:step]
                    grid_mask = np.ones(x_crop.shape,dtype=bool)
                    if is_quad_roi_for_processing and roi_pts_relative_to_crop is not None:
                        poly_m = np.zeros(prev_gray_for_flow.shape,dtype=np.uint8); cv2.fillPoly(poly_m,[np.array(roi_pts_relative_to_crop, dtype=np.int32)],255); grid_mask = poly_m[y_crop,x_crop]==255
                    xcf,ycf = x_crop[grid_mask].flatten(), y_crop[grid_mask].flatten()
                    if len(xcf)>0:
                        dx_c,dy_c = flow[ycf,xcf,0],flow[ycf,xcf,1]
                        p1_orig_dense = np.vstack((xcf+x_b, ycf+y_b)).T.reshape(-1,1,2).astype(np.float32)
                        p2_orig_dense = np.vstack((xcf+dx_c+x_b, ycf+dy_c+y_b)).T.reshape(-1,1,2).astype(np.float32)
                        if p1_orig_dense.size > 0:
                            w_p1_d=cv2.perspectiveTransform(p1_orig_dense,homography_matrix_to_world); w_p2_d=cv2.perspectiveTransform(p2_orig_dense,homography_matrix_to_world)
                            if w_p1_d is not None and w_p2_d is not None: dxm,dym=w_p2_d[:,0,0]-w_p1_d[:,0,0],w_p2_d[:,0,1]-w_p1_d[:,0,1];d_m=np.sqrt(dxm**2+dym**2);v_m=d_m[(d_m>1e-4)&(d_m<5.0)]; current_frame_displacements_meters.extend(v_m)
                if show_bev_visualization and H_bev_mapping is not None and bev_image_with_vectors is not None:
                    step_bev = 25 
                    y_cb,x_cb = np.mgrid[step_bev//2:prev_gray_for_flow.shape[0]:step_bev, step_bev//2:prev_gray_for_flow.shape[1]:step_bev]
                    gm_bev = np.ones(x_cb.shape,dtype=bool)
                    if is_quad_roi_for_processing and roi_pts_relative_to_crop is not None:
                        pm_bev = np.zeros(prev_gray_for_flow.shape,dtype=np.uint8); cv2.fillPoly(pm_bev,[np.array(roi_pts_relative_to_crop, dtype=np.int32)],255); gm_bev = pm_bev[y_cb,x_cb]==255
                    xcf_b,ycf_b = x_cb[gm_bev].flatten(), y_cb[gm_bev].flatten()
                    if len(xcf_b)>0:
                        dx_cb,dy_cb = flow[ycf_b,xcf_b,0],flow[ycf_b,xcf_b,1]
                        p1ofb = np.vstack((xcf_b+x_b, ycf_b+y_b)).T.reshape(-1,1,2).astype(np.float32)
                        p2ofb = np.vstack((xcf_b+dx_cb+x_b, ycf_b+dy_cb+y_b)).T.reshape(-1,1,2).astype(np.float32)
                        if p1ofb.size > 0:
                            p1bpts = cv2.perspectiveTransform(p1ofb, H_bev_mapping); p2bpts = cv2.perspectiveTransform(p2ofb, H_bev_mapping)
                            if p1bpts is not None and p2bpts is not None:
                                for i in range(p1bpts.shape[0]):
                                    pt1,pt2 = tuple(p1bpts[i,0,:].astype(int)), tuple(p2bpts[i,0,:].astype(int))
                                    if 0<=pt1[0]<bev_img_size[0] and 0<=pt1[1]<bev_img_size[1] and 0<=pt2[0]<bev_img_size[0] and 0<=pt2[1]<bev_img_size[1]:
                                        cv2.arrowedLine(bev_image_with_vectors, pt1, pt2, (0,200,0), 1, tipLength=0.3)
        else: # Sparse LK
            if p0 is None or len(p0) < 10:
                gftt_m_rt = None
                if is_quad_roi_for_processing and roi_pts_relative_to_crop is not None:
                    gftt_m_rt = np.zeros(prev_gray_for_flow.shape,dtype=np.uint8)
                    cv2.fillPoly(gftt_m_rt,[np.array(roi_pts_relative_to_crop, dtype=np.int32)],255)
                p0_new = cv2.goodFeaturesToTrack(prev_gray_for_flow,mask=gftt_m_rt,maxCorners=100,qualityLevel=0.3,minDistance=7,blockSize=7)
                if p0_new is not None and len(p0_new)>0:
                    p0 = p0_new
                else:
                    prev_gray_for_flow=current_gray_for_flow.copy()
                    frame_count_processed+=1
                    if max_frames_to_process and frame_count_processed >= max_frames_to_process:
                        break 
                    continue
            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray_for_flow, current_gray_for_flow, p0, None, **lk_params)
                if p1 is not None and st is not None:
                    g_new_c,g_old_c = p1[st==1],p0[st==1]; g_new_f,g_old_f = g_new_c,g_old_c
                    if is_quad_roi_for_processing and roi_pts_relative_to_crop is not None and len(g_old_c)>0:
                        f_n,f_o=[],[]
                        for i,pt_o_c in enumerate(g_old_c):
                            if cv2.pointPolygonTest(np.array(roi_pts_relative_to_crop,dtype=np.int32), tuple(pt_o_c.ravel()),False)>=0 and \
                               cv2.pointPolygonTest(np.array(roi_pts_relative_to_crop,dtype=np.int32), tuple(g_new_c[i].ravel()),False)>=0:
                                f_o.append(pt_o_c);f_n.append(g_new_c[i])
                        if f_n: g_new_f,g_old_f=np.array(f_n),np.array(f_o)
                        else: g_new_f,g_old_f=np.array([]),np.array([])
                    if len(g_new_f) > 0:
                        if homography_matrix_to_world is not None:
                            p_old_o_lk = g_old_f + np.array([x_b,y_b],dtype=np.float32); p_new_o_lk = g_new_f + np.array([x_b,y_b],dtype=np.float32)
                            w_p_o_lk=cv2.perspectiveTransform(p_old_o_lk.reshape(-1,1,2),homography_matrix_to_world); w_p_n_lk=cv2.perspectiveTransform(p_new_o_lk.reshape(-1,1,2),homography_matrix_to_world)
                            if w_p_o_lk is not None and w_p_n_lk is not None:
                                dxm,dym=w_p_n_lk[:,0,0]-w_p_o_lk[:,0,0],w_p_n_lk[:,0,1]-w_p_o_lk[:,0,1]
                                d_m=np.sqrt(dxm**2+dym**2);v_m=d_m[(d_m>1e-4)&(d_m<5.0)];current_frame_displacements_meters.extend(v_m)
                        if show_bev_visualization and H_bev_mapping is not None and bev_image_with_vectors is not None:
                            p_o_b_lk=g_old_f+np.array([x_b,y_b],dtype=np.float32);p_n_b_lk=g_new_f+np.array([x_b,y_b],dtype=np.float32)
                            p1b_lk=cv2.perspectiveTransform(p_o_b_lk.reshape(-1,1,2),H_bev_mapping);p2b_lk=cv2.perspectiveTransform(p_n_b_lk.reshape(-1,1,2),H_bev_mapping)
                            if p1b_lk is not None and p2b_lk is not None:
                                for i in range(p1b_lk.shape[0]):
                                    pt1,pt2=tuple(p1b_lk[i,0,:].astype(int)),tuple(p2b_lk[i,0,:].astype(int))
                                    if 0<=pt1[0]<bev_img_size[0] and 0<=pt1[1]<bev_img_size[1] and 0<=pt2[0]<bev_img_size[0] and 0<=pt2[1]<bev_img_size[1]:
                                        cv2.arrowedLine(bev_image_with_vectors,pt1,pt2,(0,0,255),1,tipLength=0.4)
                        p0 = g_new_f.reshape(-1,1,2)
                    else:
                        p0 = None
                else:
                    p0 = None
            
        if homography_matrix_to_world is not None and current_frame_displacements_meters:
            avg_disp_m=np.mean(current_frame_displacements_meters); vel_mps_f=avg_disp_m*fps; velocities_mps.append(vel_mps_f)
        
        if show_bev_visualization and bev_image_with_vectors is not None:
            cv2.imshow(bev_window_name, bev_image_with_vectors)
            key_bev = cv2.waitKey(1) & 0xFF 
            if key_bev == ord('q') or key_bev == 27: 
                show_bev_visualization = False; cv2.destroyWindow(bev_window_name)
        
        prev_gray_for_flow = current_gray_for_flow.copy()
        prev_full_frame = current_full_frame.copy() 
        frame_count_processed += 1
        if max_frames_to_process and frame_count_processed >= max_frames_to_process:
            break
    
    cap.release()
    if cv2.getWindowProperty(bev_window_name, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow(bev_window_name)
    if not velocities_mps:
        log(f"WARN [{os.path.basename(video_path)}]: No valid METRIC vels ({frame_count_processed} frames).")
        return None
    final_avg_vel = np.mean(velocities_mps)
    log(f"INFO [{os.path.basename(video_path)}]: Final avg METRIC vel ({len(velocities_mps)} avgs): {final_avg_vel:.3f} m/s")
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
        self.homography_matrix_to_world = None
        self.H_bev_mapping = None            
        self.bev_output_size_px = (600, 400) 
        self.ui_queue = queue.Queue()
        self.roi_interaction_active = False
        self.show_bev_var = tk.BooleanVar(value=False)
        self.setup_ui()
        self.master.after(100, self.process_ui_queue)

    def process_ui_queue(self):
        try:
            while True: 
                task, args = self.ui_queue.get_nowait()
                if task == "get_distance": self._show_distance_dialog(*args)
                elif task == "log": self._log_message_gui(args)
                elif task == "finish_roi_interaction": self._finalize_roi_interaction_from_thread(args) 
                elif task == "reset_distances_and_log":
                    self.dist_12_cm.set(0.0); self.dist_23_cm.set(0.0); self.dist_34_cm.set(0.0); self.dist_41_cm.set(0.0)
                    self.roi_side_lengths_set = False; self.roi_var.set(""); self._log_message_gui(args)
                self.ui_queue.task_done()
        except queue.Empty: pass 
        finally:
            if self.master.winfo_exists(): self.master.after(100, self.process_ui_queue)

    def _show_distance_dialog(self, pt_idx1, pt_idx2, next_task_info):
        self._log_message_gui(f"[DIALOG_SHOW] Attempting for P{pt_idx1}-P{pt_idx2}.")
        if self.master.state() == 'withdrawn': self.master.deiconify(); self._log_message_gui("[DIALOG_SHOW] Main window deiconified.")
        self.master.lift(); self.master.focus_force(); self.master.update_idletasks()
        title=f"Dist P{pt_idx1}-P{pt_idx2}"; prompt=f"Real dist P{pt_idx1}-P{pt_idx2} (cm):"
        dist_cm_in = simpledialog.askfloat(title,prompt,parent=self.master,minvalue=1e-3)
        self._log_message_gui(f"[DIALOG_SHOW] P{pt_idx1}-P{pt_idx2} askfloat returned: {dist_cm_in}")
        self.master.attributes('-topmost',False)
        target_var=None
        if(pt_idx1,pt_idx2)==(1,2):target_var=self.dist_12_cm
        elif(pt_idx1,pt_idx2)==(2,3):target_var=self.dist_23_cm
        elif(pt_idx1,pt_idx2)==(3,4):target_var=self.dist_34_cm
        elif(pt_idx1,pt_idx2)==(4,1):target_var=self.dist_41_cm
        if dist_cm_in is not None and dist_cm_in>0:
            if target_var:target_var.set(dist_cm_in);self._log_message_gui(f"Dist P{pt_idx1}-P{pt_idx2} set to: {target_var.get():.2f} cm")
            if next_task_info and next_task_info.get("type")=="next_distance": self.ui_queue.put(("get_distance",(*next_task_info["points"],None)))
        elif dist_cm_in is not None:
            if target_var: target_var.set(0.0); self._log_message_gui(f"Invalid dist for P{pt_idx1}-P{pt_idx2} ({dist_cm_in}). Set to 0.")
        else:
            if target_var: target_var.set(0.0); self._log_message_gui(f"Dist input P{pt_idx1}-P{pt_idx2} cancelled. Set to 0.")

    def setup_ui(self):
        main_frame=ttk.Frame(self.master,padding="10");main_frame.pack(fill=tk.BOTH,expand=True)
        input_frame=ttk.LabelFrame(main_frame,text="Parameters",padding="10");input_frame.grid(row=0,column=0,padx=5,pady=5,sticky="ew");input_frame.columnconfigure(1,weight=1)
        r_idx=0
        ttk.Label(input_frame,text="Video Folder:").grid(row=r_idx,column=0,sticky="w",pady=2)
        ttk.Entry(input_frame,textvariable=self.folder_path_var,width=40).grid(row=r_idx,column=1,sticky="ew",pady=2)
        ttk.Button(input_frame,text="Browse...",command=self.browse_folder).grid(row=r_idx,column=2,padx=5,pady=2);r_idx+=1
        su_vid_grp=ttk.LabelFrame(input_frame,text="Setup Video for Interactive ROI",padding="5");su_vid_grp.grid(row=r_idx,column=0,columnspan=3,sticky="ew",pady=5);su_vid_grp.columnconfigure(1,weight=1);sv_r=0
        self.su_vid_btn=ttk.Button(su_vid_grp,text="Select Video for Setup",command=self.select_video_for_setup);self.su_vid_btn.grid(row=sv_r,column=0,pady=(5,2),sticky="w",padx=2)
        self.su_vid_lbl=ttk.Label(su_vid_grp,text="No video selected.",wraplength=350,justify=tk.LEFT);self.su_vid_lbl.grid(row=sv_r,column=1,columnspan=2,pady=(5,2),sticky="ew",padx=2);r_idx+=1
        roi_d_grp=ttk.LabelFrame(input_frame,text="Interactive ROI & Side Lengths (for Perspective Correction)",padding="5");roi_d_grp.grid(row=r_idx,column=0,columnspan=3,sticky="ew",pady=5);roi_d_grp.columnconfigure(1,weight=1);rd_r=0
        self.draw_roi_btn=ttk.Button(roi_d_grp,text="Define ROI & Enter Side Lengths...",command=self.start_roi_interaction_thread,state=tk.DISABLED);self.draw_roi_btn.grid(row=rd_r,column=0,columnspan=3,pady=5,padx=2,sticky="ew");rd_r+=1
        ttk.Label(roi_d_grp,text="ROI Pixels (x1,y1..):").grid(row=rd_r,column=0,sticky="w",pady=1,padx=2)
        ttk.Entry(roi_d_grp,textvariable=self.roi_var,state='readonly',width=40).grid(row=rd_r,column=1,columnspan=2,sticky="ew",pady=1,padx=2);rd_r+=1
        ttk.Label(roi_d_grp,text="Dist P1-P2 (cm):").grid(row=rd_r,column=0,sticky="w",pady=1,padx=2);ttk.Entry(roi_d_grp,textvariable=self.dist_12_cm,state='readonly',width=10).grid(row=rd_r,column=1,sticky="w",pady=1,padx=2)
        ttk.Label(roi_d_grp,text="Dist P2-P3 (cm):").grid(row=rd_r,column=2,sticky="w",pady=1,padx=2);ttk.Entry(roi_d_grp,textvariable=self.dist_23_cm,state='readonly',width=10).grid(row=rd_r,column=3,sticky="w",pady=1,padx=2);rd_r+=1
        ttk.Label(roi_d_grp,text="Dist P3-P4 (cm):").grid(row=rd_r,column=0,sticky="w",pady=1,padx=2);ttk.Entry(roi_d_grp,textvariable=self.dist_34_cm,state='readonly',width=10).grid(row=rd_r,column=1,sticky="w",pady=1,padx=2)
        ttk.Label(roi_d_grp,text="Dist P4-P1 (cm):").grid(row=rd_r,column=2,sticky="w",pady=1,padx=2);ttk.Entry(roi_d_grp,textvariable=self.dist_41_cm,state='readonly',width=10).grid(row=rd_r,column=3,sticky="w",pady=1,padx=2);r_idx+=1
        op_grp=ttk.LabelFrame(input_frame,text="Processing Options",padding="5");op_grp.grid(row=r_idx,column=0,columnspan=3,sticky="ew",pady=5);op_r=0
        ttk.Label(op_grp,text="Video Ext (comma-sep):").grid(row=op_r,column=0,sticky="w",pady=2,padx=2);ttk.Entry(op_grp,textvariable=self.extensions_var).grid(row=op_r,column=1,columnspan=2,sticky="ew",pady=2,padx=2);op_r+=1
        ttk.Label(op_grp,text="Skip Initial Frames:").grid(row=op_r,column=0,sticky="w",pady=2,padx=2);ttk.Entry(op_grp,textvariable=self.skip_frames_var).grid(row=op_r,column=1,columnspan=2,sticky="ew",pady=2,padx=2);op_r+=1
        ttk.Label(op_grp,text="Max Frames (optional):").grid(row=op_r,column=0,sticky="w",pady=2,padx=2);ttk.Entry(op_grp,textvariable=self.max_frames_var).grid(row=op_r,column=1,columnspan=2,sticky="ew",pady=2,padx=2);op_r+=1
        ttk.Checkbutton(op_grp,text="Use Dense Optical Flow (Farneback)",variable=self.use_dense_flow_var).grid(row=op_r,column=0,columnspan=3,sticky="w",pady=5,padx=2);op_r+=1
        self.gpu_cb=ttk.Checkbutton(op_grp,text="Use GPU for Dense Flow (if CUDA)",variable=self.use_gpu_var);self.gpu_cb.grid(row=op_r,column=0,columnspan=3,sticky="w",pady=5,padx=2)
        if not CUDA_AVAILABLE:self.gpu_cb.config(state=tk.DISABLED);self.use_gpu_var.set(False)
        op_r+=1; self.bev_cb=ttk.Checkbutton(op_grp,text="Show Bird's-Eye Flow Visualization",variable=self.show_bev_var);self.bev_cb.grid(row=op_r,column=0,columnspan=3,sticky="w",pady=5,padx=2)
        r_idx+=1
        ctrl_fr=ttk.Frame(main_frame,padding="5");ctrl_fr.grid(row=r_idx,column=0,sticky="ew",pady=5)
        self.start_btn=ttk.Button(ctrl_fr,text="Start Processing",command=self.start_processing_thread);self.start_btn.pack(pady=5)
        out_fr=ttk.LabelFrame(main_frame,text="Log & Results",padding="10");out_fr.grid(row=r_idx+1,column=0,padx=5,pady=5,sticky="nsew")
        self.log_txt=scrolledtext.ScrolledText(out_fr,wrap=tk.WORD,height=10,width=80);self.log_txt.pack(fill=tk.BOTH,expand=True)
        main_frame.columnconfigure(0,weight=1);main_frame.rowconfigure(r_idx+1,weight=1)

    def browse_folder(self):
        f=filedialog.askdirectory();
        if f:self.folder_path_var.set(f)

    def select_video_for_setup(self):
        vp=filedialog.askopenfilename(title="Select Video for ROI Setup",filetypes=(("Video files","*.mp4 *.avi *.mov *.mkv"),("All files","*.*")))
        if vp:
            self.selected_video_for_setup_path.set(vp);self.draw_roi_btn.config(state=tk.NORMAL)
            lt=f"Setup Video: {os.path.basename(vp)}";cap=cv2.VideoCapture(vp)
            if cap.isOpened():r,fr=cap.read();lt+=f" ({fr.shape[1]}x{fr.shape[0]} px)" if r else " (Err read)";cap.release()
            else: lt+=" (Err open)"
            self.selected_setup_video_label.config(text=lt);self._log_message_gui(f"Video for setup: {os.path.basename(vp)}")
        else:
            self.selected_video_for_setup_path.set("");self.draw_roi_btn.config(state=tk.DISABLED)
            self.selected_setup_video_label.config(text="No video selected.")
            self.ui_queue.put(("reset_distances_and_log","Video selection cancelled, ROI data cleared."))

    def start_roi_interaction_thread(self):
        vp=self.selected_video_for_setup_path.get()
        if not vp: self._log_message_gui("Error: No video for ROI setup."); return
        self.roi_drawing_points_pixels=[]; self.roi_var.set("")
        self.dist_12_cm.set(0.0);self.dist_23_cm.set(0.0);self.dist_34_cm.set(0.0);self.dist_41_cm.set(0.0)
        self.roi_side_lengths_set=False; self.roi_interaction_active=True 
        self.master.withdraw() 
        threading.Thread(target=self._roi_interaction_cv_thread,args=(vp,),daemon=True).start()

    def _roi_interaction_cv_thread(self, video_path):
        cap=cv2.VideoCapture(video_path)
        if not cap.isOpened(): self.ui_queue.put(("log",f"Err open {os.path.basename(video_path)}")); self.ui_queue.put(("finish_roi_interaction",False)); return
        ret,frame_orig=cap.read(); cap.release()
        if not ret: self.ui_queue.put(("log",f"Err read {os.path.basename(video_path)}")); self.ui_queue.put(("finish_roi_interaction",False)); return
        cv_pts_local=[]; cv_fr_copy_local=frame_orig.copy(); win_name=self.roi_drawing_window_name
        cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
        cb_data={"pts_ref":cv_pts_local,"orig_fr_ref":frame_orig,"disp_fr_wrap":[cv_fr_copy_local],"q_ref":self.ui_queue,"win_ref":win_name}
        def _thread_mouse_cb_int(ev,x,y,fl,p_dict):
            pts,o_fr,d_fr_w,q,wn = p_dict["pts_ref"],p_dict["orig_fr_ref"],p_dict["disp_fr_wrap"],p_dict["q_ref"],p_dict["win_ref"]
            if ev==cv2.EVENT_LBUTTONDOWN and len(pts)<4:
                pts.append((x,y));tmp_draw_fr=o_fr.copy()
                for i,pt_d in enumerate(pts): cv2.circle(tmp_draw_fr,pt_d,7,(0,255,0),-1);cv2.putText(tmp_draw_fr,f"P{i+1}",(pt_d[0]+10,pt_d[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2);_=(cv2.line(tmp_draw_fr,pts[i-1],pt_d,(0,255,0),2) if i>0 else 0)
                d_fr_w[0]=tmp_draw_fr;cv2.imshow(wn,d_fr_w[0])
                if len(pts)==2: q.put(("get_distance",(1,2,None)))
                elif len(pts)==3: q.put(("get_distance",(2,3,None)))
                elif len(pts)==4: q.put(("get_distance",(3,4,{"type":"next_distance","points":(4,1)})));cv2.line(d_fr_w[0],pts[3],pts[0],(0,0,255),2);cv2.imshow(wn,d_fr_w[0])
        cv2.setMouseCallback(win_name,_thread_mouse_cb_int,cb_data)
        fh,fw=frame_orig.shape[:2];dw,dh=fw,fh
        if fw>self.max_display_width or fh>self.max_display_height: sc=min(self.max_display_width/fw,self.max_display_height/fh);dw,dh=int(fw*sc),int(fh*sc)
        cv2.resizeWindow(win_name,dw,dh);cv2.imshow(win_name,cb_data["disp_fr_wrap"][0])
        self.ui_queue.put(("log",f"ROI Setup: {win_name}. Click 4 points. Dialogs for side lengths (cm). ENTER to Confirm, R to Reset, ESC/C to Cancel."))
        cv_enter_confirm=False
        while True:
            if cv2.getWindowProperty(win_name,cv2.WND_PROP_VISIBLE)<1: self.ui_queue.put(("log","CV window closed by X. ROI cancelled."));cv_enter_confirm=False;break
            key=cv2.waitKey(50)&0xFF
            if key==13:
                if len(cv_pts_local)==4: self.ui_queue.put(("log","CV: Enter with 4 points."));cv_enter_confirm=True;break
                else: self.ui_queue.put(("log","CV: Enter, but <4 points. Click 4 points."))
            elif key==ord('r') or key==ord('R'):
                self.ui_queue.put(("log","CV: 'R' pressed. Resetting."));cv_pts_local.clear();cb_data["disp_fr_wrap"][0]=frame_orig.copy();cv2.imshow(win_name,cb_data["disp_fr_wrap"][0])
                self.ui_queue.put(("reset_distances_and_log","ROI points/distances reset. Click 4 new points."))
            elif key==27 or key==ord('c') or key==ord('C'): self.ui_queue.put(("log","CV: ESC/C pressed. ROI cancelled."));cv_enter_confirm=False;break
        cv2.destroyWindow(win_name)
        if cv_enter_confirm and len(cv_pts_local)==4: self.roi_drawing_points_pixels=list(cv_pts_local)
        else: self.roi_drawing_points_pixels=[]
        self.ui_queue.put(("finish_roi_interaction",cv_enter_confirm and len(self.roi_drawing_points_pixels)==4))

    def _finalize_roi_interaction_from_thread(self, cv_points_and_enter_confirmed_flag):
        self.master.deiconify();self.master.lift();self.master.focus_force();self.master.update_idletasks();self.master.attributes('-topmost',False)
        self.roi_interaction_active=False
        self._log_message_gui(f"[FINALIZE_ROI] CV points & Enter confirmed: {cv_points_and_enter_confirmed_flag}")
        self._log_message_gui(f"[FINALIZE_ROI] GUI ROI points count: {len(self.roi_drawing_points_pixels)}")
        d12,d23,d34,d41=self.dist_12_cm.get(),self.dist_23_cm.get(),self.dist_34_cm.get(),self.dist_41_cm.get()
        self._log_message_gui(f"[FINALIZE_ROI] Dists(cm): P1P2:{d12:.2f},P2P3:{d23:.2f},P3P4:{d34:.2f},P4P1:{d41:.2f}")
        all_dists_ok=(d12>0 and d23>0 and d34>0 and d41>0)
        self._log_message_gui(f"[FINALIZE_ROI] All 4 dists >0: {all_dists_ok}")
        final_ok = cv_points_and_enter_confirmed_flag and len(self.roi_drawing_points_pixels)==4 and all_dists_ok
        self._log_message_gui(f"[FINALIZE_ROI] Overall setup success: {final_ok}")
        if final_ok:
            self.roi_var.set(",".join(map(str,[c for p in self.roi_drawing_points_pixels for c in p])))
            self.roi_side_lengths_set=True; self._log_message_gui(f"SUCCESS: ROI pixels set: {self.roi_var.get()}. Side lengths valid. Perspective correction ON.")
        else:
            self.roi_var.set(""); self.roi_side_lengths_set=False
            msg="FAILURE: ROI & Dist setup FAILED/Incomplete. Homography conditions NOT met:\n"
            if not cv_points_and_enter_confirmed_flag: msg+="  - CV interaction (4pts+Enter) not completed/cancelled.\n"
            if not all_dists_ok: msg+="  - Not all 4 side distances entered correctly (>0cm).\n"
            self._log_message_gui(msg.strip())

    def _log_message_gui(self, message):
        if self.master.winfo_exists(): self.log_txt.insert(tk.END,str(message)+"\n");self.log_txt.see(tk.END);self.master.update_idletasks()

    def log_message_thread_safe(self, message):
        if self.master.winfo_exists(): self.master.after(0,self._log_message_gui,message)

    def _processing_job(self):
        self.homography_matrix_to_world = None 
        self.H_bev_mapping = None            
        try:
            folder_path = self.folder_path_var.get()
            if not folder_path or not os.path.isdir(folder_path):
                self.log_message_thread_safe("Error: Please select a valid video folder.")
                return

            parsed_roi_pixel_coords_list = None
            src_points_for_homography_np = None
            roi_str = self.roi_var.get().strip() 
            show_bev_flag = self.show_bev_var.get()

            if self.roi_side_lengths_set and roi_str:
                self.log_message_thread_safe("[PROC_JOB] ROI and side lengths are set. Attempting homography.")
                try:
                    coords = list(map(int, roi_str.split(',')))
                    if len(coords) == 8:
                        parsed_roi_pixel_coords_list = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
                        src_points_for_homography_np = np.array(parsed_roi_pixel_coords_list, dtype=np.float32)
                        
                        l12m,l23m,l34m,l41m = self.dist_12_cm.get()/100.0, self.dist_23_cm.get()/100.0, \
                                             self.dist_34_cm.get()/100.0, self.dist_41_cm.get()/100.0
                        if not (l12m > 0 and l23m > 0 and l34m > 0 and l41m > 0):
                            self.log_message_thread_safe("WARN [PROC_JOB]: Distances invalid. No metric homography.")
                        else:
                            avg_width_m = (l12m + l34m) / 2.0
                            avg_height_m = (l23m + l41m) / 2.0
                            dst_pts_world_meters_np = np.array([[0,0],[avg_width_m,0],[avg_width_m,avg_height_m],[0,avg_height_m]],dtype=np.float32)
                            self.homography_matrix_to_world, _ = cv2.findHomography(src_points_for_homography_np, dst_pts_world_meters_np)
                            if self.homography_matrix_to_world is None:
                                self.log_message_thread_safe("WARN [PROC_JOB]: Metric homography failed.")
                            else:
                                self.log_message_thread_safe(f"INFO [PROC_JOB]: Metric homography OK. Target world: {avg_width_m:.2f}m x {avg_height_m:.2f}m.")
                        
                        if show_bev_flag and src_points_for_homography_np is not None: # Ensure src_points are valid for BEV H
                            bev_w, bev_h = self.bev_output_size_px
                            dst_pts_bev_img_np = np.float32([[0,0], [bev_w-1,0], [bev_w-1,bev_h-1], [0,bev_h-1]])
                            self.H_bev_mapping, _ = cv2.findHomography(src_points_for_homography_np, dst_pts_bev_img_np)
                            if self.H_bev_mapping is None:
                                self.log_message_thread_safe("WARN [PROC_JOB]: BEV mapping homography failed.")
                            else:
                                self.log_message_thread_safe(f"INFO [PROC_JOB]: BEV mapping homography OK for {bev_w}x{bev_h}px output.")
                    else:
                        self.log_message_thread_safe(f"WARN [PROC_JOB]: roi_var string invalid (expected 8 coords, got {len(coords)}). No homography.")
                except ValueError as e: # Catches issues from map(int,...) or .get() if DoubleVar contains non-numeric
                    self.log_message_thread_safe(f"WARN [PROC_JOB]: ValueError processing ROI/Distances for homography: {e}. No homography.")
                except Exception as e: # General catch-all
                    self.log_message_thread_safe(f"WARN [PROC_JOB]: Error in homography setup: {e}. No homography.")
            else:
                self.log_message_thread_safe(f"[PROC_JOB] Conditions for homography not met. self.roi_side_lengths_set:{self.roi_side_lengths_set}, roi_str empty:{not roi_str}. No homography.")

            extensions_str = self.extensions_var.get()
            extensions = [ext.strip().lower() for ext in extensions_str.split(',') if ext.strip()]
            if not extensions: self.log_message_thread_safe("Error: No valid video extensions provided."); return

            try:
                skip_frames_val = int(self.skip_frames_var.get())
                if skip_frames_val < 0: raise ValueError("Skip frames must be non-negative.")
            except ValueError as e:
                self.log_message_thread_safe(f"Error: Invalid Skip Frames value: {e}"); return
            
            max_frames_val = None
            max_frames_str_val = self.max_frames_var.get().strip()
            if max_frames_str_val:
                try:
                    max_frames_val = int(max_frames_str_val)
                    if max_frames_val <= 0: raise ValueError("Max frames must be positive if set.")
                except ValueError as e:
                    self.log_message_thread_safe(f"Error: Invalid Max Frames value: {e}"); return
            
            use_dense_val = self.use_dense_flow_var.get()
            use_gpu_val = self.use_gpu_var.get()
            
            video_files = []
            for ext_item in extensions: # Corrected variable name
                video_files.extend(glob.glob(os.path.join(folder_path, f"*.{ext_item}")))
                video_files.extend(glob.glob(os.path.join(folder_path, f"*.{ext_item.upper()}")))
            video_files = sorted(list(set(video_files)))

            if not video_files: self.log_message_thread_safe(f"No videos with specified extensions found in {folder_path}"); return
            
            self.log_message_thread_safe(f"Found {len(video_files)} video(s) to process.\n")
            if self.homography_matrix_to_world is None:
                 self.log_message_thread_safe("CRITICAL INFO: No valid METRIC Homography for this run. Metric velocities will NOT be calculated.")
            if show_bev_flag and self.H_bev_mapping is None:
                 self.log_message_thread_safe("CRITICAL INFO: BEV requested but BEV Homography FAILED. No BEV visualization.")

            results = {}
            for video_file_item in video_files: # Corrected variable name
                self.log_message_thread_safe(f"--- Processing {os.path.basename(video_file_item)} ---")
                avg_velocity = estimate_water_velocity(
                    video_file_item, roi_orig_frame_pixels=parsed_roi_pixel_coords_list, 
                    use_dense_flow=use_dense_val, use_gpu=use_gpu_val, 
                    skip_frames=skip_frames_val, max_frames_to_process=max_frames_val,
                    log_callback=self.log_message_thread_safe, 
                    homography_matrix_to_world=self.homography_matrix_to_world,
                    show_bev_visualization=show_bev_flag, 
                    bev_img_size=self.bev_output_size_px, 
                    H_bev_mapping=self.H_bev_mapping
                )
                if avg_velocity is not None:
                    self.log_message_thread_safe(f"==> Est. METRIC vel for {os.path.basename(video_file_item)}: {avg_velocity:.3f} m/s\n")
                    results[os.path.basename(video_file_item)] = avg_velocity
                else:
                    self.log_message_thread_safe(f"==> No METRIC vel for {os.path.basename(video_file_item)}.\n")
                    results[os.path.basename(video_file_item)] = "N/A (Metric)"
            
            self.log_message_thread_safe("\n--- Summary ---")
            if results:
                for video_name_item, vel_item in results.items(): # Corrected variable names
                    if isinstance(vel_item, float):
                        self.log_message_thread_safe(f"{video_name_item}: {vel_item:.3f} m/s")
                    else:
                        self.log_message_thread_safe(f"{video_name_item}: {vel_item}")
            else:
                self.log_message_thread_safe("No videos yielded results.")
            self.log_message_thread_safe("\nProcessing complete.")

        except Exception as e:
            self.log_message_thread_safe(f"Unexpected error in processing job: {e}")
            import traceback; self.log_message_thread_safe(traceback.format_exc())
        finally:
            if self.master.winfo_exists():
                self.master.after(0, lambda: self.start_btn.config(state=tk.NORMAL) if self.master.winfo_exists() else None)

    def start_processing_thread(self):
        if not self.master.winfo_exists(): return
        self.log_txt.delete('1.0',tk.END); self._log_message_gui("Starting processing... Please wait.")
        self.start_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._processing_job,daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    gui = WaterVelocityGUI(root)
    root.mainloop()