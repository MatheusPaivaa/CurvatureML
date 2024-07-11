import pandas as pd
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import BaseOptions

# Set environment variable to suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def draw_landmarks_on_image(rgb_image, detection_result):
    """
    Draws face landmarks on the provided RGB image.
    """
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Convert the face landmarks to the required format.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        # Draw the face landmarks on the image.
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

    return annotated_image

def process_images(images_folder, destination_folder):
    """
    Processes images from the images folder and saves them as .obj files in the destination folder.
    """
    # Configure options for the FaceLandmarker.
    base_options = BaseOptions(model_asset_path='./include/face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # Get image paths.
    image_paths = [os.path.join(images_folder, file) for file in os.listdir(images_folder) if file.endswith(('.png', '.jpg', '.jpeg', '.ppm'))]

    for idx, image_path in enumerate(image_paths, start=1):
        try:
            print(f"Processing image {idx}/{len(image_paths)}: {image_path}")

            image = mp.Image.create_from_file(image_path)

            detection_result = detector.detect(image)

            file_path = os.path.join(destination_folder, os.path.splitext(os.path.basename(image_path))[0] + ".obj")
            with open(file_path, 'w+') as file:
                for i in range(len(detection_result.face_landmarks[0])-10):
                    file.write("v %.10e %.10e %.10e\n" % (
                        detection_result.face_landmarks[0][i].x,
                        detection_result.face_landmarks[0][i].y,
                        detection_result.face_landmarks[0][i].z))
                    file.flush()

                eqlfile = './include/face_mesh.csv'
                df = pd.read_csv(eqlfile, sep=',', header=None)
                EQL = df.to_numpy()

                for i in range(len(EQL)):
                    a = EQL[i][0] + 1
                    b = EQL[i][1] + 1
                    c = EQL[i][2] + 1
                    file.write("f %d %d %d\n" % (a, b, c))
                    file.flush()

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
