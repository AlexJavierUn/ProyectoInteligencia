import re
import nltk
import spacy
from nltk.corpus import stopwords

NLP_ES = spacy.blank("es")

try:
    stopwords.words("spanish")
except LookupError:
    nltk.download("stopwords")

# Palabras que normalmente son "stopwords" pero cambian el sentimiento:
# "no me gustó" no es lo mismo que "me gustó".
CONSERVAR = {"no", "ni", "nunca", "sin", "pero", "muy", "más", "menos", "tampoco", "nada"}
PARADAS = set(stopwords.words("spanish")) - CONSERVAR

def limpiar(texto):
    """Normaliza una reseña y conserva negaciones importantes para el sentimiento."""
    texto = "" if texto is None else str(texto)
    texto = texto.lower()
    texto = re.sub(r"http\S+", " ", texto)
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
    palabras = [token.text for token in NLP_ES(texto)
                if token.is_alpha and token.text not in PARADAS]
    return " ".join(palabras)
