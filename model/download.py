import spacy
from spacy.cli import download

def ensure_spacy_model():
    model_name = "en_core_web_sm"

    try:
        # Try loading the model
        nlp = spacy.load(model_name)
        print(f"Model '{model_name}' is already installed.")
    except OSError:
        # If not installed, download it
        print(f"Model '{model_name}' not found. Downloading...")
        download(model_name)
        nlp = spacy.load(model_name)
        print(f"Model '{model_name}' successfully installed.")

    return nlp


# Test
