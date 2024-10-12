import regex
import unicodedata


def clean_text(text: str, *, strip: bool = True, keep_punct: bool = True) -> str:
    # NFC normalization
    text = unicodedata.normalize('NFC', text)
    # remove non-latin characters (but keep numbers, punctuations, and whitespaces)
    if keep_punct:
        text = regex.sub(r'([^\p{Latin}\p{Punctuation}0-9\s]+)', r'', text)
    else:
        text = regex.sub(r'([^\p{Latin}0-9\s]+)', r'', text)
    # normalize tone
    text = normalize_tone(text)
    if strip:
        text = text.strip()
    return text


tone_normalization_map = {
    "òa": "oà",
    "Òa": "Oà",
    "ÒA": "OÀ",
    "óa": "oá",
    "Óa": "Oá",
    "ÓA": "OÁ",
    "ỏa": "oả",
    "Ỏa": "Oả",
    "ỎA": "OẢ",
    "õa": "oã",
    "Õa": "Oã",
    "ÕA": "OÃ",
    "ọa": "oạ",
    "Ọa": "Oạ",
    "ỌA": "OẠ",
    "òe": "oè",
    "Òe": "Oè",
    "ÒE": "OÈ",
    "óe": "oé",
    "Óe": "Oé",
    "ÓE": "OÉ",
    "ỏe": "oẻ",
    "Ỏe": "Oẻ",
    "ỎE": "OẺ",
    "õe": "oẽ",
    "Õe": "Oẽ",
    "ÕE": "OẼ",
    "ọe": "oẹ",
    "Ọe": "Oẹ",
    "ỌE": "OẸ",
    "ùy": "uỳ",
    "Ùy": "Uỳ",
    "ÙY": "UỲ",
    "úy": "uý",
    "Úy": "Uý",
    "ÚY": "UÝ",
    "ủy": "uỷ",
    "Ủy": "Uỷ",
    "ỦY": "UỶ",
    "ũy": "uỹ",
    "Ũy": "Uỹ",
    "ŨY": "UỸ",
    "ụy": "uỵ",
    "Ụy": "Uỵ",
    "ỤY": "UỴ",
}

def normalize_tone(text: str) -> str:
    """
    Tone normalization for Vietnamese (source: https://github.com/VinAIResearch/BARTpho/blob/main/VietnameseToneNormalization.md)
    """
    for orig, repl in tone_normalization_map.items():
        text = text.replace(orig, repl)
    return text
