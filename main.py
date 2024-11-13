import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Static Images
# IMAGES = ["download.jpeg", "download (1).jpeg", "download (2).jpeg"]
# with mp_face_detection.FaceDetection(
#     model_selection=1, min_detection_confidence=0.5) as face_detection:
#     for ind, file in enumerate(IMAGES):
#         image = cv2.imread(file)

#         if image is None:
#             print(f"Error loading image at {file}. Check the path")
#             continue

#         result = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#         if not result.detections:
#             continue
#         imcopy = image.copy()
#         for detection in result.detections:
#             print('NoseTip:')
#             print(mp_face_detection.get_key_point(
#                 detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
#             mp_drawing.draw_detection(imcopy, detection)
#         cv2.imwrite(f'{file[0:-5]}_Image_Copy' + str(ind) + '.png', imcopy)

# Web Cam Live

cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        mp_drawing.draw_detection(image, detection)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows() 