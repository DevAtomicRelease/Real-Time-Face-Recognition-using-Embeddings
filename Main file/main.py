import os, time, pickle, numpy as np, torch, cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = "inception_resnet_v1_vggface2.pth"   # Saved model weights
EMB_PATH   = "face_embeddings.pkl"                # Saved embeddings
THRESHOLD  = 0.65  
CAM_INDEX  = 0                                    # Change if webcam not detected

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# Load model
resnet = InceptionResnetV1(pretrained=None)
state = torch.load(MODEL_PATH, map_location=device)
for k in list(state.keys()):
    if k.startswith("logits."):
        del state[k]
resnet.load_state_dict(state, strict=False)
resnet.eval().to(device)
print("Model loaded from local file:", MODEL_PATH)

# Load embeddings
if os.path.exists(EMB_PATH):
    with open(EMB_PATH, "rb") as f:
        data = pickle.load(f)
    db_emb = data.get("embeddings", [])
    db_names = np.array(data.get("names", []))
    print(f"Loaded {len(db_names)} embeddings from {EMB_PATH}")
else:
    print(f"⚠️ Embedding file not found at {EMB_PATH}. Starting empty database.")
    db_emb = np.empty((0, 512))
    db_names = np.array([])


# MTCNN for face detection
mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, post_process=True, device=device)

# HELPER FUNCTIONS
def l2_normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def best_match(emb_vec, db_emb, db_names, thr):
    sims = cosine_similarity([emb_vec], db_emb)[0]
    idx = int(np.argmax(sims))
    return (db_names[idx], float(sims[idx])) if sims[idx] >= thr else ("Unknown", float(sims[idx]))


# REAL-TIME FACE RECOGNITION
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Try CAM_INDEX = 1 or 2.")

t_last = time.time()
fps_avg = 0.0

print(" Press 'Q' or 'Esc' to exit the window.")

with torch.no_grad():
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)

        boxes, _ = mtcnn.detect(img_pil)
        if boxes is not None:
            faces = mtcnn.extract(img_pil, boxes, save_path=None)
            if isinstance(faces, torch.Tensor) and faces.ndim == 3:
                faces = faces.unsqueeze(0)
            if isinstance(faces, torch.Tensor) and faces.shape[0] > 0:
                emb = resnet(faces.to(device)).cpu().numpy()
                emb = l2_normalize(emb)

                for (box, vec) in zip(boxes, emb):
                    name, score = best_match(vec, db_emb, db_names, THRESHOLD)
                    x1, y1, x2, y2 = [int(v) for v in box]
                    color = (0,255,0) if name != "Unknown" else (0,0,255)
                    cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), color, 2)
                    label = f"{name} {score:.2f}"
                    cv2.rectangle(frame_bgr, (x1,y1-22), (x1+max(120,10*len(label)), y1), (0,0,0), -1)
                    cv2.putText(frame_bgr, label, (x1+4, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        # FPS counter
        now = time.time()
        fps = 1.0 / max(1e-6, (now - t_last))
        t_last = now
        fps_avg = 0.9*fps_avg + 0.1*fps if fps_avg else fps
        cv2.putText(frame_bgr, f"FPS: {fps_avg:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.imshow("Face Recognition", frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or 'q'
            break

cap.release()
cv2.destroyAllWindows()
print("Exited cleanly.")
