import cv2 # OpenCV
import mediapipe as mp # MediaPipe

def inicializar_recursos():
  mp_drawing = mp.solutions.drawing_utils
  mp_hands_detection = mp.solutions.hands
  mp_face_detection = mp.solutions.face_detection   
  face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
  hands_detection = mp.solutions.hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9)
  
  return mp_drawing, mp_hands_detection, face_detection, hands_detection
  
def processar_frame(image, face_detection, hands_detection):
  image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
  image.flags.writeable = False
  face_results = face_detection.process(image)
  hands_results = hands_detection.process(image)
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  return image, face_results, hands_results

def processar_estado_mao(hand_landmarks, mp_hands, x_face, y_face):
    dedo_medio, dedo_indicador, dedo_anelar, dedo_mindinho, face_mao_dir = False, False, False, False, False

    if (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x and
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x > x_face and
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > x_face and
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y < y_face):
        face_mao_dir = True

    # Verifica se o dedo medio esta esticado
    if (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y <
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y and
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y <
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y <
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y):
        dedo_medio = True
    
    # Verifica se o dedo está esticado
    if (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y <
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y and
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y <
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y <
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y):
        dedo_indicador = True
    
    #Verifica se o dedo anelar está esticado
    if (hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y <
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y and
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y <
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y <
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y):
        dedo_anelar = True
    
    #verifica se o dedo mindinho está esticado
    if (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y <
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y and
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y <
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y and
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y <
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y):
        dedo_mindinho = True

    if dedo_medio and dedo_indicador and dedo_anelar and dedo_mindinho and face_mao_dir:
        return "aberta"

def main():
  
  nao_esta_detectando_face = False
  estado_mao = None
  x_face = 0
  mp_drawing, mp_hands_detection, face_detection, hands_detection = inicializar_recursos()  # Inicializa os recursos
  
  cap = cv2.VideoCapture(0)# Captura a câmera

  # Loop para capturar os frames
  while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret: # Se não conseguir capturar o frame
        print("erro ao capturar o frame")
        continue

    frame, face_results, hands_results = processar_frame(frame, face_detection, hands_detection) # Processa o frame

    # Se o rosto for detectado
    if face_results.detections:
      nao_esta_detectando_face = False
      for detection in face_results.detections: # Para cada rosto detectado
        # Desenha um retângulo ao redor do rosto
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = frame.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        x_face = bboxC.xmin
        y_face = bboxC.ymin
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
      nao_esta_detectando_face = True
        
    # Se a mão for detectada
    if hands_results.multi_hand_landmarks:
      for hand_landmarks in hands_results.multi_hand_landmarks:# Para cada mão detectada
        estado_mao  = processar_estado_mao(hand_landmarks, mp_hands_detection, x_face, y_face) # Verifica se a mão está aberta
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands_detection.HAND_CONNECTIONS) # Desenha as conexões das mãos

    mao_aberta = False
    if estado_mao == "aberta":
      mao_aberta = True
    else:
      mao_aberta = False

    if mao_aberta:
      print("Mão aberta, pediu pausa nos estudos")
    if nao_esta_detectando_face:
      print("Não está detectando o rosto, não esta estudando")

    cv2.imshow('Projeto IA', frame)# Exibe o frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()