"""Interfaz Gradio simplificada: entrevista corta con voz en el navegador."""

from __future__ import annotations

import socket

import gradio as gr

from asistente import (
    DISCLAIMER,
    ETIQUETAS_LEGIBLES,
    MAX_TURNOS_CONVERSACION,
    SALUDO_INICIAL,
    SALUDO_VOZ,
    cargar_beto,
    grabar_desde_pc,
    normalizar_contenido,
    precargar_whisper,
    preparar_ruta_audio,
    procesar_turno_conversacion,
    procesar_mensaje_libre,
    sintetizar_voz_gradio,
    transcribir_audio,
)

SEGUNDOS_GRABACION = 6


def _mensajes_usuario(historial: list) -> list[str]:
    if not historial:
        return []
    return [
        normalizar_contenido(m.get("content"))
        for m in historial
        if m.get("role") == "user" and normalizar_contenido(m.get("content"))
    ]


def _agregar_turno(historial: list, usuario: str, asistente: str) -> list:
    return historial + [
        {"role": "user", "content": usuario},
        {"role": "assistant", "content": asistente},
    ]


def _voz(texto: str):
    return sintetizar_voz_gradio(texto)


def _panel_corto(r: dict) -> str:
    if "error" in r:
        return r["error"]
    if r.get("clase") == "crisis":
        return "⚠️ Señal de crisis — busca ayuda ahora."
    if r.get("modo") == "libre":
        return "Chat libre. Escribe si tienes dudas."

    turno = r.get("turno", 1)
    nombre = ETIQUETAS_LEGIBLES.get(r["clase"], r["clase"])
    conf = r.get("confianza", 0)

    if r.get("completada"):
        return f"✅ Listo · Perfil: {nombre} ({conf:.0f}%) · Revisa el informe en el chat."

    return f"Pregunta {turno}/{MAX_TURNOS_CONVERSACION} · {nombre} ({conf:.0f}%)"


def _estado_corto(r: dict, n_prev: int) -> str:
    if "error" in r:
        return "Escribe o graba tu respuesta."
    if r.get("clase") == "crisis":
        return "Prioriza buscar ayuda."
    if r.get("completada") or r.get("modo") == "libre":
        return "Entrevista terminada. Puedes seguir escribiendo."
    n = r.get("turno", n_prev + 1)
    return f"Responde la pregunta {n} de {MAX_TURNOS_CONVERSACION}."


def _procesar(texto: str, historial: list):
    texto = (texto or "").strip()
    if not texto:
        return None

    n_prev = len(_mensajes_usuario(historial))

    if n_prev >= MAX_TURNOS_CONVERSACION:
        r = procesar_mensaje_libre(texto)
        if "error" in r:
            return {"historial": historial, "voz": None, "panel": r["error"], "estado": "Escribe algo."}
        historial = _agregar_turno(historial, texto, r["respuesta"])
        return {
            "historial": historial,
            "voz": _voz(r["respuesta"]),
            "panel": _panel_corto(r),
            "estado": _estado_corto(r, n_prev),
        }

    r = procesar_turno_conversacion(texto, _mensajes_usuario(historial))
    if "error" in r:
        return {"historial": historial, "voz": None, "panel": r["error"], "estado": "Intenta de nuevo."}

    historial = _agregar_turno(historial, texto, r["respuesta"])
    return {
        "historial": historial,
        "voz": _voz(r["respuesta"]),
        "panel": _panel_corto(r),
        "estado": _estado_corto(r, n_prev),
    }


def _salida(out, historial):
    if out is None:
        n = len(_mensajes_usuario(historial))
        estado = (
            f"Responde la pregunta {n + 1} de {MAX_TURNOS_CONVERSACION}."
            if n < MAX_TURNOS_CONVERSACION
            else "Entrevista terminada."
        )
        return historial, None, "Esperando tu respuesta...", estado
    return out["historial"], out["voz"], out["panel"], out["estado"]


def responder_texto(mensaje, historial):
    h, v, p, est = _salida(_procesar(mensaje, historial), historial)
    return h, v, p, est, ""


def grabar_y_responder(historial):
    try:
        ruta = grabar_desde_pc(SEGUNDOS_GRABACION)
        texto = transcribir_audio(ruta)
    except Exception as e:
        n = len(_mensajes_usuario(historial))
        return (
            historial,
            None,
            f"No pude usar el micrófono. Escribe tu respuesta.\n({e})",
            f"Responde la pregunta {min(n + 1, MAX_TURNOS_CONVERSACION)} de {MAX_TURNOS_CONVERSACION}.",
        )

    if not texto.strip():
        return (
            historial,
            None,
            "No escuché nada. Intenta otra vez o escribe.",
            "Habla más cerca del micrófono o usa el cuadro de texto.",
        )

    out = _procesar(texto, historial)
    h, v, p, est = _salida(out, historial)
    return h, v, p, est


def enviar_audio_navegador(audio, historial):
    ruta = preparar_ruta_audio(audio)
    if ruta is None:
        n = len(_mensajes_usuario(historial))
        return historial, None, "Graba audio primero.", f"Pregunta {min(n + 1, MAX_TURNOS_CONVERSACION)} de {MAX_TURNOS_CONVERSACION}."
    try:
        texto = transcribir_audio(ruta)
    except Exception as e:
        return historial, None, f"Error: {e}", "Usa el cuadro de texto."

    if not texto.strip():
        return historial, None, "Sin voz detectada.", "Intenta de nuevo."

    h, v, p, est = _salida(_procesar(texto, historial), historial)
    return h, v, p, est


def reiniciar_chat():
    historial = [{"role": "assistant", "content": SALUDO_INICIAL}]
    return (
        historial,
        _voz(SALUDO_VOZ),
        f"Pregunta 1 de {MAX_TURNOS_CONVERSACION}",
        "Responde la primera pregunta.",
        "",
        None,
    )


def crear_demo() -> gr.Blocks:
    print("Cargando modelos...")
    cargar_beto()
    precargar_whisper()
    print("Generando bienvenida...")
    voz_inicial = sintetizar_voz_gradio(SALUDO_VOZ)
    print("Listo.")

    historial_inicial = [{"role": "assistant", "content": SALUDO_INICIAL}]

    with gr.Blocks(title="Asistente de bienestar") as demo:
        gr.Markdown(
            f"### Asistente de bienestar · {MAX_TURNOS_CONVERSACION} preguntas\n"
            f"*Escribe o graba · {DISCLAIMER}*"
        )

        with gr.Row():
            chat = gr.Chatbot(
                value=historial_inicial,
                height=420,
                show_label=False,
            )

        resumen = gr.Textbox(
            value=f"Pregunta 1 de {MAX_TURNOS_CONVERSACION}",
            lines=1,
            interactive=False,
            show_label=False,
        )

        estado = gr.Textbox(
            value="Responde la primera pregunta.",
            lines=1,
            interactive=False,
            show_label=False,
        )

        # Oculto: solo reproduce en el navegador (una vez por actualización)
        voz_asistente = gr.Audio(
            value=voz_inicial,
            autoplay=True,
            interactive=False,
            visible=False,
            type="numpy",
        )

        with gr.Row():
            btn_grabar = gr.Button(f"🎤 Grabar ({SEGUNDOS_GRABACION}s)", variant="primary", scale=1)
            entrada = gr.Textbox(
                placeholder="O escribe aquí...",
                scale=3,
                show_label=False,
            )
            btn_texto = gr.Button("Enviar", variant="primary", scale=1)

        with gr.Row():
            btn_nueva = gr.Button("Nueva entrevista", variant="secondary")

        with gr.Accordion("Micrófono del navegador", open=False):
            mic_nav = gr.Audio(sources=["microphone", "upload"], type="filepath")
            btn_nav = gr.Button("Enviar audio")

        outs = [chat, voz_asistente, resumen, estado]
        btn_grabar.click(grabar_y_responder, [chat], outs)
        btn_texto.click(responder_texto, [entrada, chat], outs + [entrada])
        entrada.submit(responder_texto, [entrada, chat], outs + [entrada])
        btn_nav.click(enviar_audio_navegador, [mic_nav, chat], outs)
        btn_nueva.click(
            reiniciar_chat,
            None,
            [chat, voz_asistente, resumen, estado, entrada, mic_nav],
        )

    return demo


def puerto_libre(inicio: int = 7860, intentos: int = 20) -> int:
    for puerto in range(inicio, inicio + intentos):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", puerto))
                return puerto
            except OSError:
                continue
    raise OSError("No hay puertos libres.")


def lanzar_demo(inline: bool = False, inbrowser: bool = True):
    demo = crear_demo()
    puerto = puerto_libre()
    print(f"Demo: http://127.0.0.1:{puerto}")
    return demo.launch(
        inline=inline,
        inbrowser=inbrowser and not inline,
        share=False,
        server_name="127.0.0.1",
        server_port=puerto,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="teal"),
    )
