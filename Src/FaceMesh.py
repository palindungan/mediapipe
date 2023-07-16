import cv2
import mediapipe as mp
from Util import BasicToolModule

def draw_outer_face_edges(image_path):
    # Inisialisasi MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,  # Hanya mencari satu wajah pada gambar
        min_detection_confidence=0.5
    )

    # Baca gambar dari path
    image = cv2.imread(image_path)

    # Konversi gambar menjadi RGB (MediaPipe menggunakan gambar dalam format RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Deteksi wajah pada gambar
    results = mp_face_mesh.process(image_rgb)

    # Mendapatkan koordinat tepi wajah terluar
    face_edges = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            outer_edges = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                           361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                           176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                           162, 21, 54, 103, 67, 109, 10]
            for idx in outer_edges:
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                face_edges.append((x, y))

                # Gambar titik-titik tepi wajah pada gambar
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    return image

if __name__ == "__main__":
    basicTools = BasicToolModule.BasicTools()

    image_path = basicTools.getBaseUrl() + '/Resource/Images/1.png'

    result_image = draw_outer_face_edges(image_path)

    # Tampilkan gambar dengan titik-titik tepi wajah terluar
    cv2.imshow("Outer Face Edges", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
