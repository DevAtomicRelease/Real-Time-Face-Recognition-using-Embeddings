# Real-Time-Face-Recognition-using-Embeddings
A Python-based face embedding system for real-time face recognition and similarity comparison.  
A practical pipeline to detect faces, generate 512-dim embeddings, and perform identity matching using **facenet-pytorch** (MTCNN + InceptionResnetV1). Supports local runs and Google Colab.
The system works in two stages:

1. **Embedding Generation (Colab)**  
   Use `face_embedding_modal.ipynb` to detect faces from your dataset, generate embeddings using **InceptionResnetV1**, and save:
   - `inception_resnet_v1_vggface2.pth` (model weights)
   - `face_embeddings.pkl` (processed embedding database)

2. **Real-Time Recognition (Local Machine)**  
   Download these two files from Colab NoteBook `face_embeddings.pkl` and `inception_resnet_v1_vggface2.pth` and place them in the same folder  `main.py` .  
   Then, run `main.py` to perform **real-time face recognition** via webcam.

## Folder Structure

face-recognition/
│
├── main.py # Runs real-time face recognition (run locally)
├── face_embeddings.pkl # Download from Colab (embedding database)
├── inception_resnet_v1_vggface2.pth # Download from Colab (model weights)

 **Important:**  
To run `main.py`, the files `face_embeddings.pkl` and `inception_resnet_v1_vggface2.pth` must be in the same folder as `main.py`.

## Features

Face Detection with MTCNN – Detects multiple faces in images or real-time video frames with high accuracy.

Face Embedding using InceptionResnetV1 – Converts detected faces into 512-dimensional feature vectors for identity comparison.

Real-Time Recognition – Recognizes and labels known faces live using your webcam.

Pretrained Model Support – Uses VGGFace2 weights for robust, pretrained feature extraction.

Offline Local Execution – Once embeddings and the model are downloaded from Colab, the system runs fully offline.

Embedding Reusability – Generated embeddings (face_embeddings.pkl) can be reused without rebuilding for every session.

Adjustable Threshold – Fine-tune similarity matching sensitivity (THRESHOLD = 0.65) as per dataset precision.

Cross-Platform Compatibility – Works on both GPU and CPU environments across Windows, Linux, and macOS.

Lightweight Dependencies – Built using only PyTorch, facenet-pytorch, and OpenCV — easy to install and maintain.


## Tech Stack

- **Languages:** Python  
- **Libraries:** OpenCV, TensorFlow, Keras, NumPy, Scikit-learn  
- **Models:** FaceNet (for generating embeddings)  
- **Techniques:** Face Detection, Embedding Extraction, Similarity Matching  

## How It Works

- Detect faces from a live webcam feed using OpenCV.
- Generate embeddings for each detected face using the FaceNet model.
- Compare these embeddings with pre-stored ones using cosine similarity.
- Display the recognized name and level of recognition in real time.

## Results

- Achieved reliable real-time recognition with minimal delay.
- High accuracy in identifying known faces under varying lighting conditions.

Contributions are welcome!
Feel free to fork this repository, create an issue, or submit a pull request to improve the project.

## Author
DevAtomicRelease
