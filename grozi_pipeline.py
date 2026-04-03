import cv2
import numpy as np
import os

class GroziPipeline:
    """SIFT-based product detection for Grozi-120 shelf videos."""

    def __init__(self, video_name, product_ids, data_root=None):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_root = data_root if data_root else os.path.join(self.base_dir, 'data')
        
        self.template_dir = os.path.join(self.data_root, 'inVitro', 'inVitro')
        self.insitu_dir = os.path.join(self.data_root, 'inSitu', 'inSitu')
        self.video_dir = os.path.join(self.data_root, 'videos', 'video')

    COLORS = [
        (0, 255, 0), (255, 100, 0), (0, 200, 255),
        (255, 0, 200), (100, 255, 100), (255, 255, 0),
    ]

    def __init__(self, video_name, product_ids):
        self.video_name = video_name
        self.product_ids = product_ids
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        self.templates = {}
        for i, pid in enumerate(product_ids):
            tpl = self._load_template(pid)
            if tpl is not None:
                tpl['color'] = self.COLORS[i % len(self.COLORS)]
                tpl['current_status'] = 'Searching'
                tpl['overall_status'] = 'Not Found'
                tpl['missing_counter'] = 100
                tpl['consecutive_found'] = 0
                tpl['smooth_bbox'] = None
                tpl['history_bbox'] = None
                self.templates[pid] = tpl

    def _load_template(self, pid):
        img_path = os.path.join(self.template_dir, str(pid), 'web', 'JPEG', 'web1.jpg')
        if not os.path.exists(img_path):
            return None

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        h, w = img.shape
        if max(h, w) > 300:
            scale = 300 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        kp, des = self.detector.detectAndCompute(img, None)
        if des is None:
            return None

        return {
            'pid': pid,
            'img': img,
            'kp': kp,
            'des': des,
            'name': f'Product {pid}',
        }

    def get_video_path(self):
        return os.path.join(self.video_dir, self.video_name)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.detector.detectAndCompute(gray, None)
        statuses = {}

        for pid, tpl in self.templates.items():
            found = False

            if des_frame is not None and tpl['des'] is not None:
                matches = self.matcher.knnMatch(tpl['des'], des_frame, k=2)
                good = []
                for m_n in matches:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < 0.75 * n.distance:
                            good.append(m)

                if len(good) >= 10:
                    src_pts = np.float32([tpl['kp'][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if M is not None:
                        h_t, w_t = tpl['img'].shape
                        pts = np.float32([[0, 0], [0, h_t-1], [w_t-1, h_t-1], [w_t-1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)
                        arr = np.int32(dst)
                        xs, ys = arr[:, 0, 0], arr[:, 0, 1]
                        x1, y1, x2, y2 = int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))

                        box_w, box_h = x2 - x1, y2 - y1
                        gh, gw = gray.shape

                        if (box_w > 10 and box_h > 10 and box_w < gw * 0.8 and box_h < gh * 0.8
                                and x1 >= -gw * 0.2 and y1 >= -gh * 0.2):
                            found = True
                            tpl['missing_counter'] = 0
                            tpl['consecutive_found'] += 1
                            if tpl['consecutive_found'] >= 3:
                                tpl['overall_status'] = 'Detected'
                            tpl['current_status'] = 'In View'

                            alpha = 0.3
                            if tpl['smooth_bbox'] is not None:
                                sx1, sy1, sx2, sy2 = tpl['smooth_bbox']
                                x1 = int(alpha * x1 + (1 - alpha) * sx1)
                                y1 = int(alpha * y1 + (1 - alpha) * sy1)
                                x2 = int(alpha * x2 + (1 - alpha) * sx2)
                                y2 = int(alpha * y2 + (1 - alpha) * sy2)

                            tpl['smooth_bbox'] = (x1, y1, x2, y2)
                            tpl['history_bbox'] = (x1, y1, x2, y2)

                            color = tpl['color']
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, tpl['name'], (x1, max(y1-10, 0)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if not found:
                tpl['missing_counter'] += 1
                tpl['consecutive_found'] = 0
                if tpl['history_bbox']:
                    if tpl['missing_counter'] > 15:
                        tpl['current_status'] = 'Lost'
                    else:
                        tpl['current_status'] = 'Occluded'
                        hx1, hy1, hx2, hy2 = tpl['history_bbox']
                        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 255), 1)

            statuses[pid] = {
                'name': tpl['name'],
                'current': tpl['current_status'],
                'overall': tpl['overall_status'],
            }

        return frame, statuses

    @staticmethod
    def get_available_videos(data_root=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        root = data_root if data_root else os.path.join(base_dir, 'data')
        video_dir = os.path.join(root, 'videos', 'video')
        
        if not os.path.exists(video_dir):
            return []
        return sorted([f for f in os.listdir(video_dir) if f.endswith('.avi')])

    @staticmethod
    def get_products_for_video(video_name, data_root=None):
        """Return product IDs whose info.txt maps to this video."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        root = data_root if data_root else os.path.join(base_dir, 'data')
        insitu_dir = os.path.join(root, 'inSitu', 'inSitu')
        template_dir = os.path.join(root, 'inVitro', 'inVitro')
        
        products = []
        for pid in range(1, 121):
            info_path = os.path.join(insitu_dir, str(pid), 'info.txt')
            if not os.path.exists(info_path):
                continue
            with open(info_path, 'r') as f:
                for line in f:
                    if 'Shelf_' in line:
                        parts = line.split()
                        if len(parts) >= 2 and parts[1] + '.avi' == video_name:
                            tpl_path = os.path.join(template_dir, str(pid), 'web', 'JPEG', 'web1.jpg')
                            if os.path.exists(tpl_path):
                                products.append(pid)
                        break
        return products
