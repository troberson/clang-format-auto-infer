"""Qualifier alignment detection."""


def detect_qualifier_alignment(files: list[str]) -> str:
    """Detect qualifier alignment style (const/volatile positioning).

    - Left:  'const int', 'volatile char'
    - Right: 'int const', 'char volatile'
    - Leave:      mixed or unclear
    """
    type_keywords = {
        "int",
        "char",
        "float",
        "double",
        "void",
        "long",
        "short",
        "unsigned",
        "signed",
        "bool",
        "auto",
        "size_t",
        "ssize_t",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        "int8_t",
        "int16_t",
        "int32_t",
        "int64_t",
    }
    qualifiers = {"const", "volatile"}

    align_left = 0  # qualifier before type: 'const int'
    align_right = 0  # type before qualifier: 'int const'

    for fpath in files:
        try:
            with open(fpath, errors="replace") as f:
                for line in f:
                    # Skip comments
                    stripped = line.strip()
                    if (
                        stripped.startswith("//")
                        or stripped.startswith("/*")
                        or stripped.startswith("*")
                    ):
                        continue
                    words = stripped.split()
                    for i, word in enumerate(words):
                        clean = word.strip("*,&;()")
                        if clean in qualifiers:
                            # Check if next word is a type keyword -> AlignLeft
                            if i + 1 < len(words):
                                next_word = words[i + 1].strip("*,&;()")
                                if next_word in type_keywords:
                                    align_left += 1
                            # Check if previous word is a type keyword -> AlignRight
                            # (independent check — a qualifier can have both prev and next words)
                            if i > 0:
                                prev_word = words[i - 1].strip("*,&;()")
                                if prev_word in type_keywords:
                                    align_right += 1
        except OSError:  # pragma: no cover
            continue

    total = align_left + align_right
    if total == 0:
        return "Leave"

    # Require at least 60% dominance to suggest a specific alignment
    if align_left / total >= 0.6:
        return "Left"
    if align_right / total >= 0.6:
        return "Right"
    return "Leave"
