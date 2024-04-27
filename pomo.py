import time

# Define os tempos de trabalho e pausa
tempo_trabalho = 25 * 60  # 25 minutos em segundos
tempo_pausa_curta = 5 * 60  # 5 minutos em segundos
tempo_pausa_longa = 20 * 60  # 20 minutos em segundos


def iniciar_timer(duration, tipo_timer):
    start_time = time.monotonic()
    while time.monotonic() - start_time < duration:
        tempo_restante = duration - (time.monotonic() - start_time)
        minutos_restantes, segundos_restantes = divmod(tempo_restante, 60)
        minutos_restantes, segundos_restantes = divmod(tempo_restante, 60)
        minutos_restantes = int(minutos_restantes)
        segundos_restantes = int(segundos_restantes)
        tempo_formatado = f"{minutos_restantes:02d}:{segundos_restantes:02d}"
        # Limpa a tela e imprime o tempo restante
        print("\033[2J\033[H" + f"{tipo_timer}: {tempo_formatado}")


# Ciclo principal
def main():
    # Inicia o timer de pausa longa
    iniciar_timer(tempo_pausa_longa, "Pausa Longa")

    # Inicia o timer de pausa curta
    iniciar_timer(tempo_pausa_curta, "Pausa Curta")

    # Inicia o timer de trabalho
    iniciar_timer(tempo_trabalho, "Trabalho")

if __name__ == "__main__":
    main()
