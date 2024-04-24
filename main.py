import cv2 # OpenCV
import mediapipe as mp # MediaPipe
from contextlib import ExitStack

def inicializar_recursos():
  mp_drawing = mp.solutions.drawing_utils
  mp_hands_detection = mp.solutions.hands
  mp_face_detection = mp.solutions.face_detection   
  face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
  hands_detection = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
  
  return mp_drawing, mp_hands_detection, face_detection, hands_detection
  
def processar_frame(image, face_detection, hands_detection):
  image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
  image.flags.writeable = False
  face_results = face_detection.process(image)
  hands_results = hands_detection.process(image)
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  return image, face_results, hands_results
  
def desenhar_faces_e_maos(image, face_results, mp_drawing, mp_hands, hands_results):
  if face_results.detections:
      for detection in face_results.detections:
          bboxC = detection.location_data.relative_bounding_box
          ih, iw, _ = image.shape
          x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
          cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

  if hands_results.multi_hand_landmarks:
    for hand_landmarks in hands_results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(
          image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
      
  return image

def main():
  
  mp_drawing, mp_hands_detection, face_detection, hands_detection = inicializar_recursos()
  
  cap = cv2.VideoCapture(0)

  while cap.isOpened():
    ret, frame = cap.read()
   
    if not ret:
        print("erro ao capturar o frame")
        continue

    frame, face_results, hands_results = processar_frame(frame, face_detection, hands_detection)
    
    frame = desenhar_faces_e_maos(frame, face_results, mp_drawing, mp_hands_detection, hands_results)

    cv2.imshow('Projeto IA', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()