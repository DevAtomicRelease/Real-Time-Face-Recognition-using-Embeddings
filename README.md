# Real-Time-Face-Recognition-using-Embeddings
This project implements a **real-time facial recognition system** using deep learning–based embeddings for accurate and efficient face identification and verification.   
It leverages a **pre-trained FaceNet model** to generate face embeddings and **OpenCV** for real-time video capture and detection.

## Features

- Real-time face detection using **OpenCV**  
- Facial embeddings generation using **FaceNet**  
- Face matching through **cosine similarity**
- Live video stream for instant recognition and labeling  
- Modular code design for easy retraining or adding new faces

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

## Important Note
keep the modal, embedding file and the main file in the same folder.

## Author

DevAtomicRelease
