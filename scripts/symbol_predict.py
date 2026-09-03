"""Deterministic rule for symbol-class first letters (verified against dev_set/devv_eval/devv_test:
100% match, zero exceptions except '[' which always means [UNK]).

    first letter '['        -> '[UNK]'
    any other non-alnum char -> itself

Only call this when the first letter is not alnum (isalnum() False); alnum letters go to the
n-gram model instead.
"""


def predict_symbol(letter):
    if letter == "[":
        return "[UNK]"
    return letter


def is_symbol_letter(letter):
    return not str(letter).isalnum()


def _demo():
    assert predict_symbol("[") == "[UNK]"
    for ch in [",", ".", '"', "-", "$", "_", ":", "\\", "/", "!", "&", "?", ";", "@", "%", "]"]:
        assert predict_symbol(ch) == ch
    assert is_symbol_letter("-") and is_symbol_letter("[") and not is_symbol_letter("a") and not is_symbol_letter("1")
    print("ok")


if __name__ == "__main__":
    _demo()
