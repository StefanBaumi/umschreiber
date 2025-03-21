# feature_engineering.py

import re
import numpy as np
import textstat
from collections import Counter
from typing import List

def get_sentences(text: str) -> List[str]:
    """
    Extrahiert Sätze aus einem Text basierend auf Satzzeichen.
    Leert dabei Leerstrings.
    """
    sentences = re.split(r'[.!?]', text)
    return [s.strip() for s in sentences if s.strip()]

def compute_average_sentence_length(text: str) -> float:
    """Berechnet die durchschnittliche Satzlänge (in Wörtern)."""
    sentences = get_sentences(text)
    if not sentences:
        return 0.0
    word_counts = [len(s.split()) for s in sentences]
    return np.mean(word_counts)

def compute_variance_sentence_length(text: str) -> float:
    """Berechnet die Varianz der Satzlängen (in Wörtern)."""
    sentences = get_sentences(text)
    if not sentences:
        return 0.0
    word_counts = [len(s.split()) for s in sentences]
    return np.var(word_counts)

def compute_repeated_words_ratio(text: str) -> float:
    """
    Berechnet den Anteil der einzigartigen Wörter, die mehrfach vorkommen.
    Verwendet dazu collections.Counter.
    """
    words = re.findall(r'\w+', text.lower())
    if not words:
        return 0.0
    word_counts = Counter(words)
    repeated = sum(1 for count in word_counts.values() if count > 1)
    return repeated / len(word_counts)

def compute_flesch_reading_ease(text: str) -> float:
    """
    Berechnet den Flesch Reading Ease Score.
    Höhere Werte deuten auf einen leichter lesbaren Text hin.
    """
    return textstat.flesch_reading_ease(text)

def compute_type_token_ratio(text: str) -> float:
    """
    Type-Token-Ratio: Anzahl einzigartiger Wörter geteilt durch Gesamtanzahl Wörter.
    Misst die Wortvielfalt.
    """
    words = re.findall(r'\w+', text.lower())
    if not words:
        return 0.0
    unique_words = set(words)
    return len(unique_words) / len(words)

def compute_filler_word_ratio(text: str) -> float:
    """
    Anteil an Füllwörtern am Gesamttext.
    Die Füllwortliste kann nach Bedarf erweitert werden.
    """
    filler_words = {"also", "halt", "naja", "irgendwie", "sozusagen",
                    "quasi", "eigentlich", "ok", "äh", "mh", "ach", "nun"}
    words = re.findall(r'\w+', text.lower())
    if not words:
        return 0.0
    filler_count = sum(1 for w in words if w in filler_words)
    return filler_count / len(words)

def compute_personal_pronoun_ratio(text: str) -> float:
    """
    Anteil persönlicher Pronomen (1. und 2. Person) am Text.
    Typische deutsche Pronomen werden hier erfasst.
    """
    pronouns = {"ich", "du", "wir", "uns", "euch", "dir", "mich", "dich"}
    words = re.findall(r'\w+', text.lower())
    if not words:
        return 0.0
    pronoun_count = sum(1 for w in words if w in pronouns)
    return pronoun_count / len(words)

def compute_passive_indicator_ratio(text: str) -> float:
    """
    Schätzt den Anteil von passiv-indizierenden Wörtern.
    Einfache Indikatoren: 'wird', 'wurden', 'worden'.
    """
    passive_markers = {"wird", "wurden", "worden"}
    words = re.findall(r'\w+', text.lower())
    if not words:
        return 0.0
    count_passives = sum(1 for w in words if w in passive_markers)
    return count_passives / len(words)

def compute_emoji_ratio(text: str) -> float:
    """
    Zählt klassische Emoticons im Text.
    Für echte Unicode-Emojis wäre ein komplexerer Regex nötig.
    """
    emoticons = re.findall(r'(:\)|:\(|:\^|\^_\^|;-\)|;\)|:D|xD)', text)
    words = re.findall(r'\w+', text)
    total_words = len(words)
    if total_words == 0:
        return 0.0
    return len(emoticons) / total_words

def extract_features(text: str) -> np.ndarray:
    """
    Führt alle Feature-Berechnungen aus und gibt ein NumPy-Array zurück.
    Reihenfolge der Features:
      1. Durchschnittliche Satzlänge
      2. Varianz der Satzlängen
      3. Wiederholte Wörter Ratio
      4. Flesch Reading Ease
      5. Type-Token-Ratio
      6. Füllwort-Rate
      7. Persönliche Pronomen-Rate
      8. Passiv-Indikator-Rate
      9. Emoji-/Emoticon-Rate
    """
    return np.array([
        compute_average_sentence_length(text),
        compute_variance_sentence_length(text),
        compute_repeated_words_ratio(text),
        compute_flesch_reading_ease(text),
        compute_type_token_ratio(text),
        compute_filler_word_ratio(text),
        compute_personal_pronoun_ratio(text),
        compute_passive_indicator_ratio(text),
        compute_emoji_ratio(text)
    ])
