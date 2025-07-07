  def draw_roi_on_frame(self):
        video_path = self.selected_video_for_roi_path.get()
        if not video_path:
            self._log_message_gui("Error: No video selected. Use 'Select Video for ROI' first."); return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log_message_gui(f"Error: Could not open video {os.path.basename(video_path)} for ROI."); return
        ret, frame_for_roi = cap.read()
        cap.release()
        if not ret:
            self._log_message_gui(f"Error: Could not read a frame from {os.path.basename(video_path)} for ROI."); return
        
        self.roi_drawing_points = [] 
        self.roi_drawing_frame_copy = frame_for_roi.copy()

        # MODIFICATION HERE: Use WINDOW_NORMAL
        cv2.namedWindow(self.roi_drawing_window_name, cv2.WINDOW_NORMAL) 
        cv2.setMouseCallback(self.roi_drawing_window_name, self._mouse_callback_roi, (self.roi_drawing_window_name, frame_for_roi))
        cv2.imshow(self.roi_drawing_window_name, self.roi_drawing_frame_copy)

        # MODIFICATION HERE: Optionally resize window if frame is too large
        frame_h, frame_w = frame_for_roi.shape[:2]
        max_display_w, max_display_h = 1280, 720 # Adjust as needed
        if frame_w > max_display_w or frame_h > max_display_h:
            scale_w = max_display_w / frame_w
            scale_h = max_display_h / frame_h
            scale = min(scale_w, scale_h)
            display_w = int(frame_w * scale)
            display_h = int(frame_h * scale)
            cv2.resizeWindow(self.roi_drawing_window_name, display_w, display_h)