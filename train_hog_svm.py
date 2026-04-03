import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'inSitu', 'inSitu')
VIDEO_DIR = os.path.join(BASE_DIR, 'data', 'videos', 'video')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(MODEL_DIR, exist_ok=True)

HOG_WIN_SIZE = (64, 128)
hog = cv2.HOGDescriptor(
    _winSize=(64, 128),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)

def extract_features(img_crop):
    resized = cv2.resize(img_crop, HOG_WIN_SIZE)
    if len(resized.shape) == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return hog.compute(resized).flatten()

def train_for_product(pid):
    info_path = os.path.join(DATA_DIR, str(pid), 'info.txt')
    coord_path = os.path.join(DATA_DIR, str(pid), 'coordinates.txt')
    
    if not os.path.exists(info_path):
        print(f"[{pid}] Missing info.txt")
        return False
    if not os.path.exists(coord_path):
        print(f"[{pid}] Missing coordinates.txt")
        return False
        
    with open(info_path, 'r') as f:
        content = f.read()
        vid_name = None
        for line in content.split('\n'):
            if 'Shelf_' in line:
                parts = line.split()
                if len(parts) >= 2:
                    vid_name = parts[1] + '.avi'
                break
    
    if not vid_name:
        print(f"[{pid}] Failed to parse video name from info.txt")
        return False
        
    vid_path = os.path.join(VIDEO_DIR, vid_name)
    if not os.path.exists(vid_path):
        print(f"[{pid}] Video not found: {vid_path}")
        return False
    
    coords = []
    with open(coord_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                coords.append({
                    'frame': int(parts[1]),
                    'x': int(parts[2]), 'y': int(parts[3]),
                    'w': int(parts[4]), 'h': int(parts[5])
                })
                
    if not coords:
        print(f"[{pid}] No valid coords found in coordinates.txt")
        return False
        
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"[{pid}] cv2 failed to open video {vid_path}")
        return False

    X, y = [], []
    coord_dict = {c['frame']: c for c in coords}
    frames_to_read = max(coord_dict.keys()) + 1
    
    actual_pos_extracted = 0
    
    for f_idx in range(frames_to_read):
        ret, frame = cap.read()
        if not ret: 
            print(f"[{pid}] Video ended prematurely at frame {f_idx}")
            break
        
        if f_idx in coord_dict:
            c = coord_dict[f_idx]
            x, y_c, w, h = c['x'], c['y'], c['w'], c['h']
            
            if x>=0 and y_c>=0 and x+w<=frame.shape[1] and y_c+h<=frame.shape[0] and w>10 and h>10:
                pos_crop = frame[y_c:y_c+h, x:x+w]
                try:
                    features = extract_features(pos_crop)
                    X.append(features)
                    y.append(1)
                    actual_pos_extracted += 1
                except Exception as e:
                    print(f"[{pid}] Error extracting features: {e}")
                    continue
                
                for _ in range(5): 
                    nx = np.random.randint(0, frame.shape[1] - w)
                    ny = np.random.randint(0, frame.shape[0] - h)
                    if not (nx < x+w and nx+w > x and ny < y_c+h and ny+h > y_c):
                        neg_crop = frame[ny:ny+h, nx:nx+w]
                        try:
                            X.append(extract_features(neg_crop))
                            y.append(0)
                        except:
                            pass
                            
    cap.release()
    
    print(f"[{pid}] Extracted {actual_pos_extracted} positive patches.")
    if sum(y) == 0 or sum(y) == len(y):
        print(f"[{pid}] Invalid target array composition (sum={sum(y)}, len={len(y)})")
        return False
    
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2.67) 
    
    train_data = np.array(X, dtype=np.float32)
    response = np.array(y, dtype=np.int32)
    
    svm.train(train_data, cv2.ml.ROW_SAMPLE, response)
    svm.save(os.path.join(MODEL_DIR, f'svm_{pid}.xml'))
    return True

if __name__ == '__main__':
    trained_count = 0
    for pid in range(1, 121):
        if train_for_product(pid):
            trained_count += 1
            print(f"Successfully trained Model for Product {pid}")
    print(f"\nDone. Successfully trained {trained_count} products.")
