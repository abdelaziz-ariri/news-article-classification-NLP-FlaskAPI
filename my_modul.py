import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# S'assurer que les ressources NLTK sont téléchargées
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Nettoyage et prétraitement d'un texte :
    - Mise en minuscule
    - Suppression des caractères non alphabétiques
    - Suppression des stopwords
    - Lemmatisation
    """
    if not isinstance(text, str):
        return ""

    # Mise en minuscule
    text = text.lower()

    # Supprimer les URLs et les mentions (utile pour des données de réseaux sociaux)
    text = re.sub(r"http\S+|www\S+|@\w+", " ", text)

    # Supprimer tout sauf les lettres
    text = re.sub('[^a-z]', ' ', text)

    # Tokenisation simple
    words = text.split()

    # Supprimer les stopwords + lemmatisation
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return ' '.join(words)
