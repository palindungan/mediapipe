import cv2
import mediapipe as mp
import numpy as np
from Util import BasicToolModule

def draw_outer_face_edges(image_path):
    # Inisialisasi MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=2,
        min_detection_confidence=0.5
    )

    # Baca gambar dari path
    image = cv2.imread(image_path)

    # Konversi gambar menjadi RGB (MediaPipe menggunakan gambar dalam format RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Deteksi wajah pada gambar
    results = mp_face_mesh.process(image_rgb)

    # Mendapatkan koordinat tepi wajah terluar
    face_edges_list = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_edges = []
            outer_edges = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                           361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                           176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                           162, 21, 54, 103, 67, 109, 10]
            for idx in outer_edges:
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                face_edges.append((x, y))
            face_edges_list.append(face_edges)

    # Gambar poligon wajah terluar pada semua wajah
    mask = np.zeros_like(image)
    for face_edges in face_edges_list:
        points = np.array(face_edges, np.int32)
        cv2.fillPoly(mask, [points], (255, 255, 255))

    # Gambar hitam di luar poligon wajah terluar
    image[~mask.any(axis=2)] = 0

    # Gambar titik-titik tepi wajah terluar pada semua wajah
    for face_edges in face_edges_list:
        for x, y in face_edges:
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    return image

if __name__ == "__main__":
    basicTools = BasicToolModule.BasicTools()

    image_path = basicTools.getBaseUrl() + '/Resource/Images/2.jpg'

    result_image = draw_outer_face_edges(image_path)

    # Tampilkan gambar dengan titik-titik tepi wajah terluar dan latar belakang hitam pada semua wajah
    cv2.imshow("Outer Face Edges", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
