"""
Módulo central del asistente de salud mental.
Usado por el notebook (entrenamiento/evaluación) y por demo_app.py (demo en vivo).

Flujo: texto o audio → BETO → clase + señales → respuesta empática
"""

from __future__ import annotations

import re
import shutil
import uuid
import wave
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- Rutas y modelo base ---
CARPETA_SALIDA = Path("outputs")
CARPETA_MODELO = CARPETA_SALIDA / "modelo_beto"
CARPETA_BILSTM = CARPETA_SALIDA / "modelo_bilstm"
NOMBRE_MODELO = "dccuchile/bert-base-spanish-wwm-cased"
LONGITUD_MAX = 256

DISCLAIMER = "Esto NO es diagnóstico médico. Es acompañamiento preventivo."

# Palabras que activan mensaje de crisis (prioridad sobre el modelo)
PALABRAS_CRISIS = [
    "suicidio", "suicidarme", "hacerme daño", "no quiero vivir",
    "matarme", "morirme", "quiero morir",
    "quitarme la vida", "ya no quiero estar aquí", "mejor muerto", "mejor muerta",
    "no tiene sentido vivir", "desaparecer para siempre", "acabar con todo",
    "no puedo más con esto", "hacerme algo", "lastimarme",
]

MENSAJE_CRISIS = (
    "Me preocupa lo que compartes. Busca ayuda de inmediato: "
    "línea de crisis, consejería universitaria o un adulto de confianza."
)

# Señales mostradas al usuario (0-100) según clase detectada
MAPEO_SENALES = {
    "ansiedad": {"ansiedad": 85, "estres": 50, "agotamiento_academico": 30},
    "depresion": {"ansiedad": 40, "estres": 55, "agotamiento_academico": 70},
    "salud_mental_general": {"ansiedad": 45, "estres": 75, "agotamiento_academico": 65},
    "soledad": {"ansiedad": 50, "estres": 40, "agotamiento_academico": 35},
    "trauma_estres": {"ansiedad": 60, "estres": 80, "agotamiento_academico": 50},
    "uso_sustancias": {"ansiedad": 55, "estres": 60, "agotamiento_academico": 40},
    "otro": {"ansiedad": 30, "estres": 30, "agotamiento_academico": 30},
}

import random as _random

# Respuestas por clase y turno de conversación (listas → variedad aleatoria)
PLANTILLAS: dict[str, dict[int, list[str]]] = {
    "ansiedad": {
        1: [
            "Noto tensión en tus palabras. ¿Desde cuándo te sientes así?",
            "Percibo cierta angustia en lo que describes. ¿Puedes contarme más sobre esa sensación?",
            "Lo que sientes suena a una carga importante. ¿Hay algo en particular que dispare esa ansiedad?",
        ],
        2: [
            "Prueba la respiración 4-7-8: inhala 4 s, sostén 7, exhala 8. Es eficaz en momentos de tensión.",
            "Cuando la ansiedad aparece, nombrar lo que sientes en voz alta puede ayudarte a ganar distancia emocional.",
            "Grounding: nombra 5 cosas que puedes ver ahora mismo. Ancla tu mente al presente.",
        ],
        3: [
            "Pedir ayuda es una fortaleza, no una debilidad. La consejería universitaria puede orientarte.",
            "Hablar con alguien de confianza —amigo, familiar, orientador— puede aliviar mucho la carga.",
            "No tienes que manejar esto solo/a. Los servicios de bienestar estudiantil existen para esto.",
        ],
        4: [
            "Identificar los desencadenantes es clave. ¿Hay patrones que notes antes de sentirte ansioso/a?",
            "Llevar un pequeño diario de emociones puede ayudarte a entender cuándo y por qué aparece la ansiedad.",
        ],
        5: [
            "El descanso y la alimentación influyen mucho en la ansiedad. ¿Cómo estás durmiendo últimamente?",
            "La actividad física ligera —aunque sea 20 minutos— reduce significativamente el cortisol.",
        ],
        6: [
            "Recuerda: la ansiedad es una señal de tu cuerpo, no una señal de que estás fallando.",
            "Has dado un gran paso al hablar sobre esto. Considera buscar apoyo profesional como siguiente paso.",
        ],
    },
    "depresion": {
        1: [
            "Parece un cansancio emocional profundo. ¿Has podido descansar esta semana?",
            "Lo que describes suena a mucho peso acumulado. ¿Cuánto tiempo llevas sintiéndote así?",
            "Ese agotamiento que describes merece atención. ¿Hay algo que antes te gustaba y ahora no disfrutas?",
        ],
        2: [
            "Divide las tareas en bloques muy pequeños. Un paso mínimo al día suma mucho.",
            "Establecer una rutina mínima —levantarte a la misma hora, salir 10 minutos— puede ser un ancla.",
            "No te exijas el 100% ahora. Hacer el mínimo viable y cuidarte es suficiente por hoy.",
        ],
        3: [
            "No tienes que enfrentarlo solo/a. La consejería universitaria está para acompañarte.",
            "Hablar con un profesional no significa que estés 'loco/a'. Significa que te importas.",
            "El apoyo externo marca una diferencia real. ¿Tienes acceso a servicios de salud mental en tu institución?",
        ],
        4: [
            "¿Hay pequeñas cosas que antes te daban alegría? Retomar una de ellas, aunque sea brevemente, puede ayudar.",
            "La conexión social, aunque cueste, puede romper el ciclo. ¿Hay alguien con quien puedas reunirte esta semana?",
        ],
        5: [
            "El sueño y la depresión se retroalimentan. Intentar mantener un horario de sueño estable puede ayudar.",
            "La exposición a la luz natural, aunque sea 15 minutos, tiene efecto comprobado en el estado de ánimo.",
        ],
        6: [
            "Lo que sientes es real y merece atención profesional. Dar ese paso es un acto de valentía.",
            "Has compartido mucho hoy. Eso requiere coraje. Considera hablar con un profesional de salud mental.",
        ],
    },
    "salud_mental_general": {
        1: [
            "El estrés académico puede acumularse sin que lo notemos. ¿Qué te preocupa más ahora mismo?",
            "Parece que llevas bastante carga. ¿Sientes que el ritmo universitario te está sobrepasando?",
            "El bienestar emocional afecta el rendimiento. ¿Cuándo fue la última vez que te sentiste bien?",
        ],
        2: [
            "Prueba pausas activas de 5 minutos cada hora de estudio. Tu concentración mejorará.",
            "La técnica Pomodoro (25 min de trabajo, 5 de pausa) puede reducir el agotamiento mental.",
            "Delegar o priorizar tareas es una habilidad clave. ¿Qué podrías dejar para después o pedir ayuda?",
        ],
        3: [
            "Cuidar tu salud mental es tan importante como aprobar un examen.",
            "Los servicios de bienestar estudiantil ofrecen recursos específicos para esto. ¿Los conoces?",
            "No todo depende de ti. Pedir ayuda también es parte del proceso.",
        ],
        4: [
            "¿Estás durmiendo y comiendo bien? El cuerpo y la mente están conectados más de lo que creemos.",
            "¿Hay actividades no académicas que te recarguen? Proteger ese tiempo es fundamental.",
        ],
        5: [
            "El agotamiento académico es real y reconocido. No es falta de ganas, es saturación del sistema.",
            "Hablar con un tutor o consejero académico puede aliviar la presión y darte opciones que no veías.",
        ],
        6: [
            "Recuerda que el rendimiento académico fluctúa y eso es normal. Una mala época no define tu capacidad.",
            "Dar este paso de reflexionar sobre tu bienestar ya es un avance importante.",
        ],
    },
    "soledad": {
        1: [
            "Parece que te sientes solo/a. ¿Tienes alguien con quien hablar en tu entorno?",
            "La soledad en la universidad es más común de lo que parece. ¿Cómo son tus relaciones sociales ahora?",
            "Sentirse desconectado puede ser muy pesado. ¿Hay alguna persona con quien antes te sintieras bien?",
        ],
        2: [
            "Los grupos de estudio o actividades extracurriculares son una forma natural de conectar sin presión.",
            "Incluso interacciones pequeñas —saludar a alguien en clase, escribir un mensaje— pueden romper el aislamiento.",
            "Las comunidades online de interés común también pueden ser un punto de partida para conectar.",
        ],
        3: [
            "Construir conexiones lleva tiempo, especialmente en entornos nuevos. La paciencia contigo mismo/a es clave.",
            "La consejería universitaria ofrece grupos de apoyo donde puedes conocer a otros en situaciones similares.",
            "No estás solo/a en sentirte solo/a. Es una paradoja muy universitaria. Buscar apoyo es válido.",
        ],
        4: [
            "¿Hay intereses o hobbies tuyos que podrías explorar en grupos o clubes de tu institución?",
            "A veces la soledad viene de no sentirnos comprendidos. ¿Sientes que puedes ser tú mismo/a con las personas que te rodean?",
        ],
        5: [
            "La soledad prolongada puede afectar la salud física y mental. Tomártela en serio es importante.",
            "¿Has considerado el voluntariado o actividades comunitarias? Dan sentido de pertenencia.",
        ],
        6: [
            "Conectar auténticamente con otros requiere vulnerabilidad, y eso da miedo. Pero vale la pena intentarlo.",
            "Un profesional puede ayudarte a entender qué barreras internas pueden estar dificultando las conexiones.",
        ],
    },
    "trauma_estres": {
        1: [
            "Escucho mucha presión en lo que describes. ¿Hay algo reciente que te haya sobrepasado?",
            "Lo que describes suena a un estrés intenso. ¿Cuándo comenzó esta sensación de estar al límite?",
            "Ese nivel de estrés merece atención. ¿Has podido hablar con alguien sobre lo que estás viviendo?",
        ],
        2: [
            "Cuando el estrés es muy alto, prioriza lo básico: dormir, comer y moverte un poco.",
            "Reducir la carga de estímulos —tiempo en redes, noticias— puede aliviar la sobrecarga del sistema nervioso.",
            "Técnicas de regulación como la respiración diafragmática o el contacto con la naturaleza ayudan al cuerpo a calmarse.",
        ],
        3: [
            "Si el estrés es muy intenso o viene de una experiencia difícil, un profesional puede darte herramientas específicas.",
            "No hay que esperar a 'estar peor' para buscar ayuda. Ahora mismo es un buen momento.",
            "Los servicios de salud mental universitarios están preparados para situaciones de estrés agudo.",
        ],
        4: [
            "¿Hay personas en tu vida que hayan pasado por algo similar y puedan entenderte?",
            "A veces el cuerpo acumula el estrés antes de que la mente lo procese. ¿Notas síntomas físicos?",
        ],
        5: [
            "La escritura expresiva —escribir libremente sobre lo que sientes— puede ser muy liberadora.",
            "¿Hay rituales o actividades que te den sensación de control o calma? Identificarlos ayuda mucho.",
        ],
        6: [
            "Has mostrado mucha fortaleza al hablar de esto. El siguiente paso es buscar apoyo profesional.",
            "El trauma y el estrés agudo son condiciones que responden bien al apoyo temprano. No lo dejes para después.",
        ],
    },
    "uso_sustancias": {
        1: [
            "Gracias por compartir eso. ¿Puedes contarme más sobre cómo ha afectado esto tu vida cotidiana?",
            "Lo que describes requiere atención. ¿Has podido hablar con alguien de confianza sobre esto?",
            "Entiendo que puede ser difícil hablar de esto. ¿Desde cuándo sientes que esto es una preocupación?",
        ],
        2: [
            "El apoyo temprano mejora mucho el pronóstico. Hablar con un profesional de salud es un primer paso.",
            "No estás solo/a en esto. Hay recursos especializados que pueden ayudarte sin juzgarte.",
            "Identificar los momentos o emociones que desencadenan el uso puede ser un primer paso importante.",
        ],
        3: [
            "Los servicios de salud universitarios cuentan con orientación confidencial. Es un espacio seguro.",
            "Buscar ayuda especializada no implica que hayas fracasado. Implica que te importas.",
            "Hay líneas de ayuda confidenciales disponibles las 24 horas si necesitas hablar con alguien ahora.",
        ],
        4: [
            "¿Hay situaciones específicas que asocias con el uso? Identificarlas puede ayudar a gestionarlas.",
            "El apoyo social es fundamental en este proceso. ¿Hay personas en tu vida que puedan acompañarte?",
        ],
        5: [
            "Cuidar la salud física —sueño, alimentación, ejercicio— es parte del proceso de bienestar integral.",
            "Los grupos de apoyo entre pares pueden ser muy poderosos. ¿Has considerado explorar esa opción?",
        ],
        6: [
            "Estás haciendo algo importante al reflexionar sobre esto. El siguiente paso es hablar con un profesional.",
            "Recuerda que pedir ayuda es una decisión valiente. Hay personas preparadas para acompañarte.",
        ],
    },
    "otro": {
        1: [
            "Gracias por compartir. ¿Puedes contarme más sobre cómo te sientes en este momento?",
            "Quiero entenderte mejor. ¿Qué es lo que más te pesa últimamente?",
            "Lo que describes merece atención. ¿Hay algo concreto que te esté preocupando?",
        ],
        2: [
            "Tus emociones son válidas, sea cual sea su origen. Pequeñas pausas pueden ayudarte.",
            "A veces no necesitamos etiquetar lo que sentimos para saber que necesitamos ayuda.",
            "¿Hay algo que antes te funcionara para sentirte mejor que puedas retomar?",
        ],
        3: [
            "Si los síntomas continúan, la consejería universitaria es un buen primer paso sin compromiso.",
            "No hace falta saber exactamente qué pasa para buscar orientación. El profesional puede ayudarte a entenderlo.",
            "Hablar con alguien, aunque sea para desahogarte, puede hacer una diferencia.",
        ],
        4: [
            "¿Hay actividades cotidianas que te den sensación de logro o bienestar? Protegerlas es importante.",
            "A veces el bienestar emocional mejora cuando atendemos lo básico: descanso, movimiento, conexión social.",
        ],
        5: [
            "¿Hay personas en tu vida con quienes puedas ser honesto/a sobre cómo te sientes?",
            "La autocompasión —tratarte con la misma amabilidad que a un amigo— es una herramienta poderosa.",
        ],
        6: [
            "Gracias por tomarte este tiempo para reflexionar. Es un paso importante.",
            "Si sientes que necesitas más apoyo, los servicios de salud mental están disponibles para ti.",
        ],
    },
}

ALIAS_PLANTILLAS: dict[str, str] = {}  # ya no se necesita alias, uso_sustancias tiene sus propias plantillas

MAX_TURNOS_CONVERSACION = 4

ETIQUETAS_LEGIBLES = {
    "ansiedad": "Ansiedad",
    "depresion": "Depresión",
    "salud_mental_general": "Estrés académico / bienestar general",
    "soledad": "Soledad",
    "trauma_estres": "Trauma o estrés agudo",
    "uso_sustancias": "Uso de sustancias",
    "otro": "Otro / no especificado",
}

PREGUNTAS_ENTREVISTA = [
    "¿Cómo describirías tu estado emocional en los últimos días?",
    "¿Desde cuándo notas estos síntomas y con qué frecuencia aparecen?",
    "¿Cómo ha afectado esto tu sueño, tus estudios o tus relaciones personales?",
    "¿Cuentas con apoyo de alguien de confianza o has probado alguna estrategia para sentirte mejor?",
]

RECOMENDACIONES = {
    "ansiedad": "Considera técnicas de respiración, pausas activas y hablar con consejería universitaria.",
    "depresion": "Prioriza descanso, rutinas breves y apoyo profesional si el ánimo persiste.",
    "salud_mental_general": "Organiza tus tareas en bloques, descansa y pide orientación en servicios de bienestar.",
    "soledad": "Explora grupos de estudio, actividades sociales y espacios de escucha en tu institución.",
    "trauma_estres": "Reduce exigencias hoy, busca contención emocional y apoyo especializado si es necesario.",
    "uso_sustancias": "Habla con un profesional de salud; el apoyo temprano mejora el pronóstico.",
    "otro": "Monitorea cómo te sientes y consulta consejería si los síntomas continúan.",
}


# =============================================================================
# BiLSTM from scratch (PyTorch)
# =============================================================================

def tokenizar_palabras(texto: str) -> list[str]:
    return re.findall(r"\w+", str(texto).lower())


class TextoDataset(Dataset):
    def __init__(self, textos, labels, vocab, max_len):
        self.textos = textos
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):
        ids = [self.vocab.get(t, 1) for t in tokenizar_palabras(self.textos[idx])][: self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), self.labels[idx]


class BiLSTMClasificador(nn.Module):
    """Embeddings + BiLSTM + capa densa. Entrenado desde cero."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=2, batch_first=True,
            bidirectional=True, dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        contexto = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(self.dropout(contexto))


def construir_vocab(textos, max_vocab=15000):
    contador = Counter()
    for t in textos:
        contador.update(tokenizar_palabras(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for palabra, _ in contador.most_common(max_vocab - 2):
        vocab[palabra] = len(vocab)
    return vocab


# =============================================================================
# BETO — carga e inferencia
# =============================================================================

_modelo_beto = None
_tokenizer_beto = None
_id_a_etiqueta: dict[int, str] = {}
_etiqueta_a_id: dict[str, int] = {}
_dispositivo = "cuda" if torch.cuda.is_available() else "cpu"


def cargar_beto(carpeta: Path | None = None):
    """Carga BETO fine-tuned desde disco (una sola vez)."""
    global _modelo_beto, _tokenizer_beto, _id_a_etiqueta, _etiqueta_a_id
    ruta = carpeta or CARPETA_MODELO
    _tokenizer_beto = AutoTokenizer.from_pretrained(ruta)
    _modelo_beto = AutoModelForSequenceClassification.from_pretrained(ruta)
    _modelo_beto.to(_dispositivo)
    _modelo_beto.eval()
    _id_a_etiqueta = {int(k): v for k, v in _modelo_beto.config.id2label.items()}
    _etiqueta_a_id = {v: k for k, v in _id_a_etiqueta.items()}
    return _modelo_beto, _tokenizer_beto


def detectar_crisis(texto: str) -> bool:
    t = texto.lower()
    return any(p in t for p in PALABRAS_CRISIS)


def analizar_texto(texto: str) -> dict:
    """
    Clasifica un texto con BETO.
    Retorna: clase, confianza (%), señales, probabilidades.
    """
    if _modelo_beto is None:
        cargar_beto()

    entradas = _tokenizer_beto(texto, return_tensors="pt", truncation=True, max_length=LONGITUD_MAX)
    entradas = {k: v.to(_dispositivo) for k, v in entradas.items()}

    with torch.no_grad():
        logits = _modelo_beto(**entradas).logits[0]
    probs = torch.softmax(logits, dim=0).cpu().numpy()

    id_max = int(probs.argmax())
    clase = _id_a_etiqueta[id_max]
    confianza = float(probs[id_max]) * 100

    senales = MAPEO_SENALES.get(clase, MAPEO_SENALES["otro"]).copy()
    for idx, etiqueta in _id_a_etiqueta.items():
        if etiqueta == "ansiedad":
            senales["ansiedad"] = min(100, senales["ansiedad"] + probs[idx] * 30)
        elif etiqueta == "salud_mental_general":
            senales["estres"] = min(100, senales["estres"] + probs[idx] * 30)
            senales["agotamiento_academico"] = min(100, senales["agotamiento_academico"] + probs[idx] * 30)
        elif etiqueta == "depresion":
            senales["agotamiento_academico"] = min(100, senales["agotamiento_academico"] + probs[idx] * 30)

    return {
        "clase": clase,
        "confianza": confianza,
        "senales": senales,
        "probs": probs,
        "crisis": detectar_crisis(texto),
    }


def dialogar(turno: int, clase: str, senales: dict) -> str:
    """Elige respuesta empática según clase y turno."""
    respuesta = _elegir_respuesta_plantilla(clase, turno)
    if float(senales.get("ansiedad", 0)) >= 70 and turno == 1:
        respuesta += " Respira hondo; estoy aquí."
    return respuesta


def normalizar_contenido(contenido) -> str:
    """Convierte contenido Gradio/chat a string seguro."""
    if contenido is None:
        return ""
    if isinstance(contenido, str):
        return contenido.strip()
    if isinstance(contenido, list):
        partes = [normalizar_contenido(c) for c in contenido]
        return " ".join(p for p in partes if p)
    return str(contenido).strip()


def balancear_train_df(df, columna: str = "label", semilla: int = 42):
    """Oversampling de clases minoritarias hasta igualar la clase mayoritaria."""
    import pandas as pd
    from sklearn.utils import resample

    conteos = df[columna].value_counts()
    maximo = conteos.max()
    partes = []
    for etiqueta in conteos.index:
        subset = df[df[columna] == etiqueta]
        if len(subset) < maximo:
            subset = resample(
                subset, replace=True, n_samples=maximo, random_state=semilla,
            )
        partes.append(subset)
    return (
        pd.concat(partes, ignore_index=True)
        .sample(frac=1, random_state=semilla)
        .reset_index(drop=True)
    )


_EMPATICOS = {
    "tristeza": [
        "Lamento que te sientas así. Gracias por confiar en mí.",
        "Eso suena muy difícil. Aprecio que lo compartas conmigo.",
        "Entiendo que es un momento doloroso. Estoy aquí para escucharte.",
    ],
    "ansiedad": [
        "Comprendo que estés pasando por un momento de tensión.",
        "Esa sensación de angustia es real y merece atención.",
        "Escucho que hay mucha presión en lo que describes. Gracias por compartirlo.",
    ],
    "agotamiento": [
        "Entiendo el agotamiento que describes.",
        "Ese cansancio que sientes es una señal que merece atención.",
        "Tiene sentido sentirse así después de tanto. Gracias por decírmelo.",
    ],
    "soledad": [
        "Sentirse solo/a puede ser muy pesado. Gracias por compartirlo.",
        "La soledad es una de las cargas más difíciles. No estás solo/a en sentirla.",
    ],
    "miedo": [
        "El miedo a veces es una señal de que algo importante está en juego. Te escucho.",
        "Es valiente hablar de lo que nos asusta. Gracias por la confianza.",
    ],
    "frustracion": [
        "La frustración acumulada puede ser agotadora. Gracias por contármelo.",
        "Entiendo que la situación te está pesando. Eso tiene sentido.",
    ],
    "neutral": [
        "Gracias por compartirlo. Te escucho con atención.",
        "Aprecio que me cuentes esto. Estoy aquí para acompañarte.",
        "Gracias por confiar en mí. Vamos a explorar esto juntos/as.",
    ],
}


def _reconocimiento_empatico(texto: str) -> str:
    t = texto.lower()
    if any(w in t for w in ("triste", "deprim", "llor", "llanto", "pena", "dolor")):
        return _random.choice(_EMPATICOS["tristeza"])
    if any(w in t for w in ("solo", "sola", "aisla", "nadie", "incomprendid")):
        return _random.choice(_EMPATICOS["soledad"])
    if any(w in t for w in ("miedo", "temor", "asusta", "pánico", "panico")):
        return _random.choice(_EMPATICOS["miedo"])
    if any(w in t for w in ("frustr", "harto", "harta", "rabia", "ira", "enojad")):
        return _random.choice(_EMPATICOS["frustracion"])
    if any(w in t for w in ("ansios", "estrés", "estres", "nervios", "preocup", "angust")):
        return _random.choice(_EMPATICOS["ansiedad"])
    if any(w in t for w in ("cansad", "agotad", "sin energ", "exhaust", "no puedo")):
        return _random.choice(_EMPATICOS["agotamiento"])
    return _random.choice(_EMPATICOS["neutral"])


def informe_final(r: dict) -> str:
    """Informe preliminar profesional al cerrar la entrevista."""
    clase = r["clase"]
    nombre = ETIQUETAS_LEGIBLES.get(clase, clase.replace("_", " ").title())
    senales = r.get("senales") or {}
    senales_txt = ", ".join(f"{k.replace('_', ' ')} {int(v)}%" for k, v in senales.items())
    rec = RECOMENDACIONES.get(clase, RECOMENDACIONES["otro"])

    return (
        "─── Informe preliminar (modelo BETO) ───\n"
        f"Perfil detectado: {nombre}\n"
        f"Confianza del modelo: {r['confianza']:.1f}%\n"
        f"Indicadores: {senales_txt or 'no disponibles'}\n\n"
        f"Interpretación: Con base en tus respuestas, el sistema identifica señales "
        f"compatibles con {nombre.lower()}. Esto es orientativo, no un diagnóstico clínico.\n\n"
        f"Recomendación: {rec}\n\n"
        f"{DISCLAIMER}"
    )


def _elegir_respuesta_plantilla(clase: str, turno: int) -> str:
    """Selecciona una respuesta empática de la plantilla con variedad aleatoria."""
    clase_tpl = ALIAS_PLANTILLAS.get(clase, clase)
    if clase_tpl not in PLANTILLAS:
        clase_tpl = "otro"
    turno_clamped = min(turno, max(PLANTILLAS[clase_tpl].keys()))
    opciones = PLANTILLAS[clase_tpl].get(turno_clamped, PLANTILLAS[clase_tpl][1])
    return _random.choice(opciones)


def respuesta_entrevista(turno: int, texto_usuario: str, r: dict) -> str:
    """Genera la respuesta del asistente según el turno de la entrevista (1-4)."""
    ack = _reconocimiento_empatico(texto_usuario)
    consejo = _elegir_respuesta_plantilla(r["clase"], turno)
    if turno >= MAX_TURNOS_CONVERSACION:
        return f"{ack}\n\n{consejo}\n\n{informe_final(r)}"
    pregunta = PREGUNTAS_ENTREVISTA[min(turno, len(PREGUNTAS_ENTREVISTA) - 1)]
    return f"{ack} {consejo}\n\n➤ {pregunta}"


SALUDO_INICIAL = (
    f"Hola. Cuatro preguntas cortas sobre cómo te sientes. "
    f"Puedes escribir o usar el micrófono.\n\n➤ {PREGUNTAS_ENTREVISTA[0]}"
)

SALUDO_VOZ = (
    "Hola. Bienvenido al asistente de bienestar. "
    "Te haré cuatro preguntas. Puedes hablar o escribir. "
    "Cuando estés listo, responde."
)

# Respuestas para el modo de chat libre post-entrevista
_RESPUESTAS_POST = [
    "Gracias por seguir compartiendo. Recuerda que lo conversado es orientativo. "
    "¿Hay algo más que quieras explorar o alguna pregunta sobre las recomendaciones?",
    "Entiendo. Si tienes dudas sobre los recursos mencionados o quieres hablar más, estoy aquí.",
    "Es valioso que sigas reflexionando. ¿Hay algo concreto del informe que te haya generado preguntas?",
    "Cada persona vive estas experiencias de forma única. Si necesitas más orientación, "
    "te recomiendo dar el paso de hablar con un profesional de salud mental.",
    "Gracias por tu confianza. Si algo de lo que sientes se intensifica, por favor busca apoyo inmediato.",
]


def procesar_mensaje_libre(texto: str) -> dict:
    """
    Modo chat libre post-entrevista.
    Detecta crisis con prioridad. Si no, da orientación general sin reclasificar.
    """
    texto = (texto or "").strip()
    if not texto:
        return {"error": "Escribe un mensaje."}

    if detectar_crisis(texto):
        return {
            "texto_usuario": texto,
            "clase": "crisis",
            "respuesta": MENSAJE_CRISIS,
            "disclaimer": DISCLAIMER,
            "modo": "libre",
        }

    return {
        "texto_usuario": texto,
        "clase": "libre",
        "respuesta": _random.choice(_RESPUESTAS_POST),
        "disclaimer": DISCLAIMER,
        "modo": "libre",
    }


def procesar_turno_conversacion(texto: str, historial_usuario: list[str] | None = None) -> dict:
    """
    Un turno de la entrevista (1-4). Analiza el contexto acumulado con BETO.
    """
    texto = (texto or "").strip()
    if not texto:
        return {"error": "Escribe o graba un mensaje."}

    historial_usuario = [normalizar_contenido(m) for m in (historial_usuario or []) if normalizar_contenido(m)]
    turno = len(historial_usuario) + 1
    contexto = " ".join(historial_usuario + [texto])

    if detectar_crisis(contexto):
        return {
            "texto_usuario": texto,
            "contexto": contexto,
            "turno": turno,
            "clase": "crisis",
            "confianza": 100.0,
            "senales": {},
            "respuesta": MENSAJE_CRISIS,
            "disclaimer": DISCLAIMER,
            "completada": True,
        }

    r = analizar_texto(contexto)
    return {
        "texto_usuario": texto,
        "contexto": contexto,
        "turno": turno,
        "clase": r["clase"],
        "confianza": r["confianza"],
        "senales": r["senales"],
        "respuesta": respuesta_entrevista(turno, texto, r),
        "disclaimer": DISCLAIMER,
        "completada": turno >= MAX_TURNOS_CONVERSACION,
        "modo": "entrevista",
    }


def _barra_ascii(valor: float, ancho: int = 15) -> str:
    """Genera una barra de progreso ASCII para el panel de análisis."""
    llenas = int(round(valor / 100 * ancho))
    vacias = ancho - llenas
    return f"[{'█' * llenas}{'░' * vacias}] {int(valor)}%"


def resumen_analisis(r: dict) -> str:
    """Panel de análisis en tiempo real durante la entrevista."""
    if "error" in r:
        return r["error"]
    if r.get("clase") == "crisis":
        return "⚠️  SEÑAL DE CRISIS DETECTADA\nPrioriza buscar ayuda inmediata."
    if r.get("modo") == "libre":
        return "💬 Modo chat libre activo.\nLa entrevista ha concluido. Puedes seguir escribiendo."

    turno_actual = r.get('turno', 1)
    # Indicador visual de progreso ●●●○○○
    completados = min(turno_actual, MAX_TURNOS_CONVERSACION)
    indicador = '●' * completados + '○' * (MAX_TURNOS_CONVERSACION - completados)

    nombre = ETIQUETAS_LEGIBLES.get(r["clase"], r["clase"])
    lineas = [
        f"Progreso: {indicador} ({turno_actual}/{MAX_TURNOS_CONVERSACION})",
        "",
        f"🔍 Perfil provisional: {nombre}",
        f"Confianza: {_barra_ascii(r['confianza'])}",
    ]
    senales = r.get("senales") or {}
    if senales:
        lineas.append("")
        lineas.append("📊 Indicadores:")
        etiq = {"ansiedad": "Ansiedad", "estres": "Estrés", "agotamiento_academico": "Agotamiento"}
        for k, v in senales.items():
            nombre_s = etiq.get(k, k.replace('_', ' ').title())
            lineas.append(f"  {nombre_s}: {_barra_ascii(float(v))}")
    lineas.append("")
    if r.get("completada"):
        lineas.append("✅ Entrevista completada.")
        lineas.append("Revisa el informe en el chat.")
        lineas.append("Puedes seguir escribiendo en modo libre.")
    else:
        restantes = MAX_TURNOS_CONVERSACION - turno_actual
        lineas.append(f"⏳ Quedan {restantes} pregunta(s). Continúa respondiendo.")
    return "\n".join(lineas)


def procesar_mensaje(texto: str, turno: int = 1) -> dict:
    """Pipeline completo: texto → análisis → respuesta."""
    texto = (texto or "").strip()
    if not texto:
        return {"error": "Escribe o graba un mensaje."}

    if detectar_crisis(texto):
        return {
            "texto_usuario": texto,
            "clase": "crisis",
            "confianza": 100.0,
            "senales": {},
            "respuesta": MENSAJE_CRISIS,
            "disclaimer": DISCLAIMER,
        }

    r = analizar_texto(texto)
    return {
        "texto_usuario": texto,
        "clase": r["clase"],
        "confianza": r["confianza"],
        "senales": r["senales"],
        "respuesta": dialogar(turno, r["clase"], r["senales"]),
        "disclaimer": DISCLAIMER,
    }


# =============================================================================
# Whisper — voz a texto (opcional, para demo con micrófono)
# =============================================================================

_modelo_whisper = None
_whisper_tamano = "base"  # más rápido que "small" en GPU/CPU


def _asegurar_ffmpeg() -> bool:
    """
    Garantiza que 'ffmpeg' est\u00e9 disponible como comando en el PATH.

    Problema en Windows: imageio_ffmpeg empaqueta el binario como
    'ffmpeg-win-x86_64-v7.1.exe', no como 'ffmpeg.exe'.
    Soluci\u00f3n: creamos una copia llamada 'ffmpeg.exe' en un dir temporal.
    """
    import os
    import shutil
    import tempfile

    # 1. Ya est\u00e1 en PATH con el nombre correcto
    if shutil.which("ffmpeg"):
        return True

    # 2. Intentar v\u00eda imageio_ffmpeg (binario con nombre no est\u00e1ndar en Windows)
    try:
        import imageio_ffmpeg
        exe_orig = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if exe_orig.exists():
            alias_dir = Path(tempfile.gettempdir()) / "ffmpeg_alias"
            alias_dir.mkdir(exist_ok=True)
            alias_exe = alias_dir / "ffmpeg.exe"
            if not alias_exe.exists():
                shutil.copy2(exe_orig, alias_exe)
            os.environ["PATH"] = str(alias_dir) + os.pathsep + os.environ.get("PATH", "")
            return True
    except (ImportError, Exception):
        pass

    return False  # ffmpeg no disponible



def _cargar_audio_sin_ffmpeg(ruta: Path) -> np.ndarray:
    """
    Carga un archivo WAV como array float32 a 16 kHz SIN necesitar ffmpeg.
    Usa soundfile primero, luego wave (stdlib) como fallback.
    """
    # --- Intento 1: soundfile (la mejor opción para WAV) ---
    try:
        import soundfile as sf
        data, sr = sf.read(str(ruta), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)  # mono
        # Remuestrear a 16 kHz si es necesario
        if sr != 16000 and len(data) > 0:
            n_samples_16k = int(len(data) / sr * 16000)
            data = np.interp(
                np.linspace(0, len(data), n_samples_16k),
                np.arange(len(data)),
                data,
            ).astype(np.float32)
        return data.astype(np.float32)
    except Exception:
        pass

    # --- Intento 2: wave (stdlib Python, solo WAV PCM) ---
    try:
        import wave as _wave
        with _wave.open(str(ruta), "rb") as wf:
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            raw = wf.readframes(n_frames)

        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        dtype = dtype_map.get(sampwidth, np.int16)
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if n_channels > 1:
            data = data.reshape(-1, n_channels).mean(axis=1)
        # Normalizar a [-1, 1]
        max_val = float(2 ** (8 * sampwidth - 1))
        data = data / max_val
        # Remuestrear a 16 kHz
        if sr != 16000 and len(data) > 0:
            n_samples_16k = int(len(data) / sr * 16000)
            data = np.interp(
                np.linspace(0, len(data), n_samples_16k),
                np.arange(len(data)),
                data,
            ).astype(np.float32)
        return data.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el audio WAV sin ffmpeg: {e}")


def precargar_whisper(tamano: str = "base"):
    """Carga Whisper al iniciar la demo (evita espera en la primera grabación)."""
    global _modelo_whisper, _whisper_tamano
    import whisper

    _asegurar_ffmpeg()
    if _modelo_whisper is None or _whisper_tamano != tamano:
        print(f"Cargando Whisper ({tamano})...")
        _modelo_whisper = whisper.load_model(tamano, device=_dispositivo)
        _whisper_tamano = tamano
        print("Whisper listo.")
    return _modelo_whisper


def transcribir_audio(ruta_wav: str | Path, dispositivo: str | None = None) -> str:
    """
    Transcribe audio con Whisper-base (español).
    Carga el WAV con soundfile/wave primero para evitar dependencia de ffmpeg.
    """
    global _modelo_whisper
    import whisper

    ruta = Path(ruta_wav)
    if not ruta.exists():
        raise FileNotFoundError(f"Archivo de audio no encontrado: {ruta}")

    dev = dispositivo or _dispositivo
    if _modelo_whisper is None:
        _asegurar_ffmpeg()
        _modelo_whisper = whisper.load_model(_whisper_tamano, device=dev)

    # --- Estrategia 1: cargar WAV con soundfile/wave (sin ffmpeg) ---
    audio_input: str | np.ndarray = str(ruta)
    ext = ruta.suffix.lower()
    if ext in (".wav", ".wave"):
        try:
            audio_input = _cargar_audio_sin_ffmpeg(ruta)
        except Exception:
            # Fallback: intentar con ffmpeg si está disponible
            _asegurar_ffmpeg()
            audio_input = str(ruta)
    else:
        # Para mp3/webm/ogg necesitamos ffmpeg
        if not _asegurar_ffmpeg():
            raise RuntimeError(
                "ffmpeg no encontrado. Instala ffmpeg o usa grabación WAV. "
                "Alternativa: pip install imageio-ffmpeg"
            )

    resultado = _modelo_whisper.transcribe(
        audio_input,
        language="es",
        fp16=(dev == "cuda"),
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
    )
    return resultado["text"].strip()


def guardar_wav(audio_array, ruta: Path, sample_rate=16000):
    """Guarda array float32 como WAV int16."""
    audio = np.clip(audio_array.flatten(), -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(str(ruta), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def preparar_ruta_audio(audio, carpeta: Path | None = None) -> Path | None:
    """Copia el audio de Gradio a una ruta estable (evita WinError 2 por temp borrado)."""
    if audio is None:
        return None

    destino = carpeta or (CARPETA_SALIDA / "audio_temp")
    destino.mkdir(parents=True, exist_ok=True)

    origen: Path | None = None
    if isinstance(audio, (str, Path)):
        origen = Path(str(audio).split("?")[0])
    elif isinstance(audio, dict):
        origen = Path(str(audio.get("path") or audio.get("name") or ""))
    elif isinstance(audio, tuple) and len(audio) == 2:
        sr, data = audio
        ruta = destino / f"entrada_{uuid.uuid4().hex}.wav"
        guardar_wav(data, ruta, sample_rate=sr)
        return ruta

    if origen is None or not origen.exists():
        return None

    ext = origen.suffix.lower() or ".wav"
    dest = destino / f"entrada_{uuid.uuid4().hex}{ext}"
    shutil.copy2(origen, dest)
    return dest


def texto_para_voz(respuesta: str) -> str:
    """Texto corto para TTS: una sola frase clave por turno."""
    if "─── Informe preliminar" in respuesta:
        return "Gracias por tus respuestas. Tu informe está en pantalla."
    for linea in respuesta.split("\n"):
        limpia = linea.strip().lstrip("➤").strip()
        if linea.strip().startswith("➤") and limpia:
            return limpia[:350]
    lineas = [l.strip() for l in respuesta.split("\n") if l.strip() and not l.startswith("───")]
    if lineas:
        return lineas[0][:350]
    return respuesta[:350]


def sintetizar_voz(texto: str, ruta: Path | None = None) -> str | None:
    """Texto a MP3/WAV en disco."""
    ruta_str = _generar_archivo_voz(texto, ruta)
    return ruta_str


def _generar_archivo_voz(texto: str, ruta: Path | None = None) -> str | None:
    texto = (texto or "").strip()
    if not texto:
        return None

    salida = ruta or (CARPETA_SALIDA / "audio_temp" / f"respuesta_{uuid.uuid4().hex}.wav")
    salida.parent.mkdir(parents=True, exist_ok=True)

    try:
        import asyncio
        import edge_tts

        mp3 = salida.with_suffix(".mp3")

        async def _generar():
            com = edge_tts.Communicate(texto[:500], "es-MX-DaliaNeural")
            await com.save(str(mp3))

        asyncio.run(_generar())
        if mp3.exists() and mp3.stat().st_size > 0:
            return _convertir_a_wav(mp3, salida)
    except Exception:
        pass

    try:
        import pyttsx3

        motor = pyttsx3.init()
        for voz in motor.getProperty("voices"):
            nombre = voz.name.lower()
            if any(x in nombre for x in ("spanish", "espa", "helena", "sabina", "heloisa")):
                motor.setProperty("voice", voz.id)
                break
        motor.setProperty("rate", 160)
        motor.save_to_file(texto[:500], str(salida))
        motor.runAndWait()
        return str(salida) if salida.exists() and salida.stat().st_size > 0 else None
    except Exception:
        return None


def _convertir_a_wav(origen: Path, destino: Path) -> str:
    """Convierte mp3/webm a wav mono 16 kHz (Whisper/Gradio)."""
    _asegurar_ffmpeg()
    try:
        import whisper
        audio = whisper.load_audio(str(origen))
        import soundfile as sf
        destino.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(destino), audio, 16000)
        return str(destino)
    except Exception:
        shutil.copy2(origen, destino)
        return str(destino)


def audio_a_gradio(ruta: str | Path | None) -> tuple[int, np.ndarray] | None:
    """Carga audio como (sample_rate, array) para el componente Audio de Gradio."""
    if not ruta:
        return None
    p = Path(ruta)
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        import soundfile as sf
        data, sr = sf.read(str(p), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return int(sr), data
    except Exception:
        pass
    try:
        import whisper
        audio = whisper.load_audio(str(p))
        return 16000, audio.astype(np.float32)
    except Exception:
        return None


def reproducir_en_pc(ruta: str | Path | None):
    """Reproduce en los altavoces del PC (respaldo si el navegador bloquea autoplay)."""
    arr = audio_a_gradio(ruta)
    if arr is None:
        return
    sr, data = arr
    try:
        import sounddevice as sd
        sd.play(data, sr)
        sd.wait()
    except Exception:
        pass


def sintetizar_voz_gradio(
    texto: str,
    *,
    reproducir_pc: bool = False,
) -> tuple[int, np.ndarray] | None:
    """TTS en formato Gradio. Por defecto solo suena en el navegador (autoplay)."""
    ruta = _generar_archivo_voz(texto_para_voz(texto))
    if ruta and reproducir_pc:
        reproducir_en_pc(ruta)
    return audio_a_gradio(ruta)


def grabar_desde_pc(segundos: int = 6, sample_rate: int = 16000) -> Path:
    """
    Graba del micrófono del PC directamente (sin navegador).
    Usa sounddevice + soundfile. Si soundfile no está, usa wave (stdlib).
    """
    import sounddevice as sd

    destino = CARPETA_SALIDA / "audio_temp"
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / f"grabacion_{uuid.uuid4().hex}.wav"

    print(f"Grabando {segundos} s... Habla ahora.")
    try:
        audio = sd.rec(
            int(segundos * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
    except Exception as e:
        raise RuntimeError(
            f"Error al grabar con sounddevice: {e}\n"
            "Verifica que el micrófono esté conectado y que PortAudio esté instalado."
        )

    # Intentar guardar con soundfile; fallback con wave stdlib
    try:
        import soundfile as sf
        sf.write(str(ruta), audio, sample_rate)
    except Exception:
        # Fallback: guardar como WAV PCM 16-bit con wave stdlib
        guardar_wav(audio, ruta, sample_rate=sample_rate)

    if not ruta.exists() or ruta.stat().st_size == 0:
        raise RuntimeError("El archivo de grabación quedó vacío. Verifica el micrófono.")

    print(f"Grabación lista: {ruta.name} ({ruta.stat().st_size} bytes)")
    return ruta
