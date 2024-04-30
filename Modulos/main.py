import cv2 # OpenCV
import tracker as tc # Tracker
import time
from multiprocessing import Process, Queue


def tracker(update_queue):
  global nao_esta_detectando_face
  global mao_aberta
  nao_esta_detectando_face = False
  mao_aberta = False
  cap = cv2.VideoCapture(0)# Captura a câmera
  mp_drawing, mp_hands_detection, face_detection, hands_detection = tc.inicializar_recursos()  # Inicializa os recursos
  while cap.isOpened():
    frame, mao_aberta, nao_esta_detectando_face = tc.comecar_captura(cap, face_detection,hands_detection,mp_hands_detection,mp_drawing)# Começa a captura     
    cv2.imshow('Projeto IA', frame)# Exibe o frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    update_queue.put(nao_esta_detectando_face)
    update_queue.put(mao_aberta)
  cap.release()
  cv2.destroyAllWindows()

def controlador(update_queue):
  tempo_trabalho = 25 * 60  # 25 minutos em segundos
  tempo_pausa_curta = 5 * 60  # 5 minutos em segundos
  nao_esta_detectando_face = False
  mao_aberta = False
  start_time = time.monotonic()

  while True:  
    if not update_queue.empty():
      nao_esta_detectando_face = update_queue.get()
      mao_aberta = update_queue.get()
   
    if nao_esta_detectando_face or mao_aberta:
      if time.monotonic() - start_time < tempo_pausa_curta:
        tempo_restante = tempo_pausa_curta - (time.monotonic() - start_time)
        minutos_restantes, segundos_restantes = divmod(tempo_restante, 60)
        minutos_restantes, segundos_restantes = divmod(tempo_restante, 60)
        minutos_restantes = int(minutos_restantes)
        segundos_restantes = int(segundos_restantes)
        tempo_formatado = f"{minutos_restantes:02d}:{segundos_restantes:02d}"
        # Limpa a tela e imprime o tempo restante
        print("\033[2J\033[H" + " Tempo pausa curta : " + f"{tempo_formatado}")
    
    if not nao_esta_detectando_face and not mao_aberta: 
      if time.monotonic() - start_time < tempo_trabalho:
        tempo_restante = tempo_trabalho - (time.monotonic() - start_time)
        minutos_restantes, segundos_restantes = divmod(tempo_restante, 60)
        minutos_restantes, segundos_restantes = divmod(tempo_restante, 60)
        minutos_restantes = int(minutos_restantes)
        segundos_restantes = int(segundos_restantes)
        tempo_formatado = f"{minutos_restantes:02d}:{segundos_restantes:02d}"
        # Limpa a tela e imprime o tempo restante
        print("\033[2J\033[H" + " Tempo trabalho : " + f"{tempo_formatado}")


def main():

  update_queue = Queue()
  controlador_process =  Process(target=controlador, args=(update_queue,))
  tracker_process =  Process(target=tracker, args=(update_queue,))

  tracker_process.start()
  controlador_process.start()
  tracker_process.join()
  controlador_process.join()
  

if __name__ == "__main__":
  main()

