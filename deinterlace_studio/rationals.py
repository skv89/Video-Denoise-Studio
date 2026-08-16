from __future__ import annotations

from fractions import Fraction
from math import ceil


class RationalError(ValueError):
    """Raised when a media rational cannot be parsed safely."""


def parse_fraction(value: object, *, allow_zero: bool = False) -> Fraction | None:
    """Parse FFmpeg/user rational forms without converting through float."""

    if value is None:
        return None
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, int):
        result = Fraction(value, 1)
    elif isinstance(value, float):
        result = Fraction(str(value))
    else:
        text = str(value).strip().lower().replace("÷", "/").replace(":", "/")
        if not text or text in {"n/a", "na", "unknown", "0/0"}:
            return None
        try:
            result = Fraction(text)
        except (ValueError, ZeroDivisionError) as exc:
            raise RationalError(f"Invalid rational value: {value!r}") from exc

    if result < 0 or (result == 0 and not allow_zero):
        raise RationalError(f"Rational value must be positive: {value!r}")
    return result


def fraction_text(value: Fraction | None) -> str:
    if value is None:
        return "unknown"
    return f"{value.numerator}:{value.denominator}"


def rate_text(value: Fraction | None, places: int = 6) -> str:
    if value is None:
        return "unknown"
    return f"{value.numerator}/{value.denominator} ({float(value):.{places}f})"


def derive_dar(width: int, height: int, sar: Fraction | None) -> Fraction:
    if width <= 0 or height <= 0:
        raise RationalError("Stored dimensions must be positive")
    return Fraction(width, height) * (sar or Fraction(1, 1))


def exact_square_pixel_raster(
    dar: Fraction,
    source_width: int,
    source_height: int,
    *,
    modulus: int = 2,
    no_downscale: bool = True,
) -> tuple[int, int]:
    """Return the smallest exact-DAR square-pixel raster meeting constraints.

    Exact display geometry sometimes cannot retain the source height. For
    example, 349:192 at 576 lines requires an odd 1047-pixel width. This
    routine raises both dimensions to the smallest codec-safe exact multiple.
    """

    if dar <= 0 or source_width <= 0 or source_height <= 0 or modulus <= 0:
        raise RationalError("Invalid square-pixel raster inputs")
    minimum = 1
    if no_downscale:
        minimum = max(
            ceil(source_width / dar.numerator),
            ceil(source_height / dar.denominator),
            1,
        )
    multiplier = minimum
    while (
        (dar.numerator * multiplier) % modulus != 0
        or (dar.denominator * multiplier) % modulus != 0
    ):
        multiplier += 1
    return dar.numerator * multiplier, dar.denominator * multiplier


def fractions_close(left: Fraction | None, right: Fraction | None, tolerance: float = 1e-5) -> bool:
    if left is None or right is None:
        return left is right
    if left == right:
        return True
    return abs(float(left) - float(right)) <= tolerance * max(abs(float(left)), abs(float(right)), 1.0)

