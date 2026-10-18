"""WCAG AA contrast checks against the actual palette used by the UI."""
from app.theme import DARK_PALETTE, LIGHT_PALETTE


def _linearize(channel: float) -> float:
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def _relative_luminance(color: str) -> float:
    hex_value = color.removeprefix("#")
    channels = [int(hex_value[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    r, g, b = (_linearize(channel) for channel in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_ratio(first: str, second: str) -> float:
    lighter, darker = sorted((_relative_luminance(first), _relative_luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def test_text_tokens_meet_wcag_aa_normal_text_contrast_in_both_themes():
    for theme_name, palette in (("light", LIGHT_PALETTE), ("dark", DARK_PALETTE)):
        pairs = {
            "body text on page": (palette["INK"], palette["PAPER"]),
            "muted copy on page": (palette["MUTED"], palette["PAPER"]),
            "input text on input surface": (palette["INK"], palette["CONTROL"]),
            "primary button label": (palette["PRIMARY_TEXT"], palette["PRIMARY"]),
        }
        for label, (foreground, background) in pairs.items():
            ratio = _contrast_ratio(foreground, background)
            assert ratio >= 4.5, f"{theme_name} theme {label} contrast is {ratio:.2f}:1"
