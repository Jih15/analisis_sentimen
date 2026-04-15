import re

# ---------------------------------------------------------------------------
# Indonesian stopwords
# ---------------------------------------------------------------------------
STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk",
    "pada", "adalah", "dalam", "tidak", "juga", "sudah", "saya", "kamu",
    "kami", "mereka", "akan", "bisa", "ada", "lebih", "seperti", "dapat",
    "oleh", "karena", "sehingga", "namun", "tetapi", "atau", "jika", "maka",
    "sangat", "telah", "belum", "masih", "hanya", "saja", "pun", "bukan",
    "agar", "supaya", "ketika", "sebelum", "sesudah", "setelah", "antara",
    "atas", "bawah", "lain", "semua", "setiap", "beberapa", "satu", "dua",
    "tiga", "ia", "nya", "mu", "ku", "jadi", "hal", "cara", "kita", "ya",
    "ga", "gak", "tak", "nggak", "si", "pak", "bu", "bang", "kak", "mas",
    "mbak", "lah", "pun", "kok", "deh", "sih", "dong", "nih", "tuh",
    "udah", "udh", "sdh", "sdh", "banget", "bgt", "jg", "yg", "dgn",
    "utk", "krn", "tp", "tapi", "dr", "pd", "ke", "nya", "sy", "km",
    "mrk", "klo", "kalo", "kalau", "emang", "memang", "gimana", "bagaimana",
    "kenapa", "mengapa", "dimana", "kapan", "siapa", "apa", "mana", "sama",
    "iya", "tidak", "bisa", "mau", "msh", "blm", "jgn", "jangan",
    "ini", "itu", "ada", "saat", "waktu", "sekarang", "nanti", "sudah",
    "tapi", "tdk", "gak", "g", "u", "d", "y", "yg", "n",
}


# ---------------------------------------------------------------------------
# Step-by-step preprocessing
# ---------------------------------------------------------------------------
def case_folding(text: str) -> str:
    """Ubah teks ke huruf kecil."""
    return str(text).lower()


def clean_text(text: str) -> str:
    """Hapus URL, mention, hashtag, angka, dan karakter khusus."""
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_stopwords(text: str) -> str:
    """Hapus kata-kata stopword Bahasa Indonesia."""
    words = text.split()
    return " ".join(w for w in words if w not in STOPWORDS_ID and len(w) > 1)


def stemming(text: str) -> str:
    """Stemming menggunakan PySastrawi. Fallback ke teks asli jika tidak tersedia."""
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        return stemmer.stem(text)
    except ImportError:
        return text


def preprocess(text: str, use_stemming: bool = True) -> str:
    """Pipeline preprocessing lengkap."""
    text = case_folding(text)
    text = clean_text(text)
    text = remove_stopwords(text)
    if use_stemming:
        text = stemming(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_batch(texts, use_stemming: bool = True) -> list:
    """Proses banyak teks sekaligus."""
    return [preprocess(t, use_stemming) for t in texts]
