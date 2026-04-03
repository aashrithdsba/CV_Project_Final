import cv2
import numpy as np
import os

class ClassicalCVPipeline:
    def __init__(self, sequence_dir):
        self.method = 'SIFT'
        self.target_name = os.path.basename(sequence_dir)
        self.sequence_dir = sequence_dir
        
        self.detector = None
        self.matcher = None
        self.hog = None
        self.template_data = {'current_status': 'Out of View', 'overall_status': 'Not Found'}
        
        img_dir = os.path.join(sequence_dir, 'img')
        gt_path = os.path.join(sequence_dir, 'groundtruth_rect.txt')
        
        img1_path = os.path.join(img_dir, '0001.jpg')
        if not os.path.exists(img1_path) and os.path.exists(os.path.join(img_dir, '00001.jpg')):
            img1_path = os.path.join(img_dir, '00001.jpg')
            
        img1 = cv2.imread(img1_path)
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1 is not None else None
        
        x, y, w, h = 0, 0, 0, 0
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                first_line = f.readline().strip().replace('\t', ',')
                parts = first_line.split(',')
                if len(parts) >= 4:
                    x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    
        if img1_gray is not None and w > 0 and h > 0:
            template_img = img1_gray[y:y+h, x:x+w]
            if max(w,h) < 150:
                scale = 150 / max(w,h)
                template_img = cv2.resize(template_img, (int(w*scale), int(h*scale)))
                
            self.template_data['img'] = template_img
        else:
            self.template_data['img'] = None
            
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        if self.template_data['img'] is not None:
            kp, des = self.detector.detectAndCompute(self.template_data['img'], None)
            self.template_data['kp'] = kp
            self.template_data['des'] = des
                
        self.history_bbox = None
        self.smooth_bbox = None
        self.missing_counter = 100
        self.consecutive_found = 0
        
    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = False
        t_data = self.template_data
        
        if 'des' in t_data and t_data['des'] is not None:
            kp_frame, des_frame = self.detector.detectAndCompute(gray, None)
            if des_frame is not None:
                matches = self.matcher.knnMatch(t_data['des'], des_frame, k=2)
                good_matches = []
                for m_n in matches:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                            
                if len(good_matches) >= 12:
                    src_pts = np.float32([t_data['kp'][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if M is not None:
                        h_t, w_t = t_data['img'].shape
                        pts = np.float32([[0,0],[0,h_t-1],[w_t-1,h_t-1],[w_t-1,0]]).reshape(-1,1,2)
                        dst = cv2.perspectiveTransform(pts, M)
                        
                        arr = np.int32(dst)
                        xs = arr[:,0,0]
                        ys = arr[:,0,1]
                        x1, y1, x2, y2 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
                        
                        box_w = x2 - x1
                        box_h = y2 - y1
                        gw, gh = gray.shape[1], gray.shape[0]
                        
                        if (box_w > 10 and box_h > 10 and box_w < gw*0.8 and box_h < gh*0.8 and x1 >= -gw*0.2 and y1 >= -gh*0.2):
                            found = True
                            self.missing_counter = 0
                            self.consecutive_found += 1
                            
                            if self.consecutive_found >= 3:
                                t_data['overall_status'] = 'Tracking Active'
                            t_data['current_status'] = 'In View'
                            
                            alpha = 0.3
                            if self.smooth_bbox is not None:
                                sx1, sy1, sx2, sy2 = self.smooth_bbox
                                x1 = int(alpha * x1 + (1 - alpha) * sx1)
                                y1 = int(alpha * y1 + (1 - alpha) * sy1)
                                x2 = int(alpha * x2 + (1 - alpha) * sx2)
                                y2 = int(alpha * y2 + (1 - alpha) * sy2)
                                
                            self.smooth_bbox = (x1, y1, x2, y2)
                            self.history_bbox = (x1, y1, x2, y2)
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(frame, f"{self.target_name} ({self.method})", (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if not found:
            self.missing_counter += 1
            self.consecutive_found = 0
            if self.history_bbox:
                hx1, hy1, hx2, hy2 = self.history_bbox
                if self.missing_counter > 15:
                    t_data['current_status'] = 'Lost'
                else:
                    t_data['current_status'] = 'Occluded'
                    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 255), 2)
                    cv2.putText(frame, f"{self.target_name} (Predicting)", (hx1, max(hy1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
        status_dict = {self.target_name: {'current': t_data['current_status'], 'overall': t_data['overall_status']}}
        return frame, status_dict
