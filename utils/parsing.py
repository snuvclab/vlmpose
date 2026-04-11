import json
from typing import Any, Dict, Optional, List


class PoseParseError(ValueError):
    pass


class TargetSelectParseError(ValueError):
    pass


def _extract_first_json_object(text: str) -> str:
    """
    Extract the first top-level JSON object {...} from an LLM response.
    Handles markdown code fences and extra surrounding text.
    """
    # 1) If there's a fenced block, prefer its contents
    if "```" in text:
        parts = text.split("```")
        # parts: [pre, fence_content, post, fence_content, ...]
        # Take the first fenced content that looks like it contains a JSON object
        for i in range(1, len(parts), 2):
            block = parts[i]
            # Drop possible language hint at the first line: "json\n{...}"
            if "\n" in block:
                first_line, rest = block.split("\n", 1)
                if first_line.strip().lower() in {"json", "javascript"}:
                    block = rest
            if "{" in block and "}" in block:
                text = block
                break

    # 2) Find the first balanced {...} while respecting quoted strings
    start = text.find("{")
    if start == -1:
        raise PoseParseError("No '{' found in response (cannot locate JSON object).")

    in_str = False
    esc = False
    depth = 0
    end = None

    for idx in range(start, len(text)):
        ch = text[idx]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        # not in string
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break

    if end is None:
        raise PoseParseError("Found '{' but could not find matching '}' (unbalanced braces).")

    return text[start:end].strip()


def _extract_first_json_object2(text: str) -> str:
    """
    Extract the first top-level JSON object {...} from an LLM/VLM response.
    Handles markdown code fences and extra surrounding text.
    """
    # 1) Prefer fenced blocks
    if "```" in text:
        parts = text.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i]
            # remove language hint line e.g. "json\n{...}"
            if "\n" in block:
                first_line, rest = block.split("\n", 1)
                if first_line.strip().lower() in {"json", "javascript"}:
                    block = rest
            if "{" in block and "}" in block:
                text = block
                break

    # 2) Find first balanced {...} while respecting quoted strings
    start = text.find("{")
    if start == -1:
        raise TargetSelectParseError("No '{' found (cannot locate JSON object).")

    in_str = False
    esc = False
    depth = 0
    end = None

    for idx in range(start, len(text)):
        ch = text[idx]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break

    if end is None:
        raise TargetSelectParseError("Unbalanced braces: couldn't find matching '}'.")

    return text[start:end].strip()


def parse_pose_response(response_text: str, strict: bool = True) -> Dict[str, Any]:
    """
    Parse and validate the LLM response according to your instruction schema.

    Returns a normalized dict with:
      - Target object: Optional[str]
      - Reasoning: str
      - Translation: Optional[List[float]]  # length 3
      - Dominant rotation axis: Optional[str]  # 'x','y','z'
      - Angle: Optional[float]

    strict=True:
      - Errors on unknown/missing keys, wrong types, invalid axis.
    strict=False:
      - Attempts small normalizations (e.g., axis lowercasing), but still errors on structural issues.
    """
    EXPECTED_KEYS = [
        "Target object",
        "Reasoning",
        "Translation",
        "Dominant rotation axis",
        "Angle",
    ]

    AXES = {"x", "y", "z"}

    raw_json = _extract_first_json_object(response_text)

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise PoseParseError(f"JSON parsing failed: {e.msg} (line {e.lineno}, col {e.colno})") from e

    if not isinstance(data, dict):
        raise PoseParseError("Top-level JSON must be an object/dict.")

    # Key checks
    keys = list(data.keys())
    missing = [k for k in EXPECTED_KEYS if k not in data]
    extra = [k for k in keys if k not in EXPECTED_KEYS]

    if missing:
        raise PoseParseError(f"Missing required keys: {missing}")
    if strict and extra:
        raise PoseParseError(f"Unexpected extra keys: {extra}")

    # (Optional) enforce key order if you truly want "in order"
    if strict:
        filtered_in_order = [k for k in keys if k in EXPECTED_KEYS]
        if filtered_in_order != EXPECTED_KEYS:
            raise PoseParseError(
                "Keys are not in the required order.\n"
                f"Expected: {EXPECTED_KEYS}\n"
                f"Got:      {filtered_in_order}"
            )

    # Validate fields
    target = data["Target object"]
    reasoning = data["Reasoning"]
    translation = data["Translation"]
    axis = data["Dominant rotation axis"]
    angle = data["Angle"]

    # Reasoning
    if not isinstance(reasoning, str) or reasoning.strip() == "":
        raise PoseParseError('"Reasoning" must be a non-empty string.')

    # Target object
    if target is not None and not isinstance(target, str):
        raise PoseParseError('"Target object" must be a string or null.')
    if isinstance(target, str) and target.strip() == "":
        raise PoseParseError('"Target object" must be a non-empty string or null.')

    # Translation
    norm_translation: Optional[List[float]] = None
    if translation is None:
        norm_translation = None
    else:
        if not (isinstance(translation, list) and len(translation) == 3):
            raise PoseParseError('"Translation" must be null or a list of 3 numbers [Tx, Ty, Tz].')
        try:
            norm_translation = [float(translation[0]), float(translation[1]), float(translation[2])]
        except Exception as e:
            raise PoseParseError('"Translation" entries must be numeric (floats).') from e

    # Axis
    norm_axis: Optional[str] = None
    if axis is None:
        norm_axis = None
    else:
        if not isinstance(axis, str):
            raise PoseParseError('"Dominant rotation axis" must be a string or null.')
        ax = axis.strip()
        if not strict:
            ax = ax.lower()
        if ax not in AXES:
            raise PoseParseError(f'"Dominant rotation axis" must be one of {sorted(AXES)} or null.')
        norm_axis = ax

    # Angle
    norm_angle: Optional[float] = None
    if angle is None:
        norm_angle = None
    else:
        try:
            norm_angle = float(angle)
        except Exception as e:
            raise PoseParseError('"Angle" must be a float (number) or null.') from e

    # --- Disallow ANY nulls ---
    if (target is None) or (norm_translation is None) or (norm_axis is None) or (norm_angle is None):
        raise PoseParseError("Null is not allowed: all fields must be non-null.")

    out = {
        "Target object": (target.strip() if isinstance(target, str) else None),
        "Reasoning": reasoning,
        "Translation": norm_translation,
        "Dominant rotation axis": norm_axis,
        "Angle": norm_angle,
    }
    return out


def parse_target_select_response(
    response_text: str,
    strict: bool = True,
    allow_label_str: bool = False,
) -> Dict[str, Any]:
    """
    Parse + validate VLM output for INSTRUCTIONS_TARGET_SELECT.

    Expected JSON keys (in order):
      - "Target label" (int)
      - "Target object" (str)
      - "Related labels" (list[int])
      - "Related objects" (list[str])
      - "Reasoning" (str)

    strict=True:
      - errors on extra keys and wrong key order.
    allow_label_str=True:
      - allows digit strings for labels, e.g.
        "Target label": "2"
        "Related labels": ["3", 5]

    Returns:
      dict with normalized types:
        {
          "Target label": int,
          "Target object": str,
          "Related labels": list[int],
          "Related objects": list[str],
          "Reasoning": str
        }
    """
    EXPECTED_KEYS = [
        "Target label",
        "Target object",
        "Related labels",
        "Related objects",
        "Reasoning",
    ]

    raw_json = _extract_first_json_object2(response_text)

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise TargetSelectParseError(
            f"JSON parsing failed: {e.msg} (line {e.lineno}, col {e.colno})"
        ) from e

    if not isinstance(data, dict):
        raise TargetSelectParseError("Top-level JSON must be an object/dict.")

    keys = list(data.keys())
    missing = [k for k in EXPECTED_KEYS if k not in data]
    extra = [k for k in keys if k not in EXPECTED_KEYS]
    if missing:
        raise TargetSelectParseError(f"Missing required keys: {missing}")
    if strict and extra:
        raise TargetSelectParseError(f"Unexpected extra keys: {extra}")

    if strict:
        filtered_in_order = [k for k in keys if k in EXPECTED_KEYS]
        if filtered_in_order != EXPECTED_KEYS:
            raise TargetSelectParseError(
                "Keys are not in the required order.\n"
                f"Expected: {EXPECTED_KEYS}\n"
                f"Got:      {filtered_in_order}"
            )

    target_label = data["Target label"]
    target = data["Target object"]
    related_labels = data["Related labels"]
    related_objects = data["Related objects"]
    reasoning = data["Reasoning"]

    def _parse_positive_int(value: Any, field_name: str) -> int:
        if isinstance(value, bool):
            raise TargetSelectParseError(
                f'"{field_name}" must be an integer, not boolean.'
            )

        if isinstance(value, int):
            value_int = value
        elif allow_label_str and isinstance(value, str) and value.strip().isdigit():
            value_int = int(value.strip())
        else:
            raise TargetSelectParseError(
                f'"{field_name}" must be an integer '
                f'(or digit string if allow_label_str=True).'
            )

        if value_int <= 0:
            raise TargetSelectParseError(
                f'"{field_name}" must be a positive integer (>=1).'
            )

        return value_int

    # Target label validation
    target_label_int = _parse_positive_int(target_label, "Target label")

    # Target object validation
    if not isinstance(target, str) or target.strip() == "":
        raise TargetSelectParseError('"Target object" must be a non-empty string.')
    target_str = target.strip()

    # Related labels validation
    if not isinstance(related_labels, list):
        raise TargetSelectParseError('"Related labels" must be a list.')
    related_label_ints = [
        _parse_positive_int(v, "Related labels")
        for v in related_labels
    ]

    # Related objects validation
    if not isinstance(related_objects, list):
        raise TargetSelectParseError('"Related objects" must be a list.')

    related_object_strs = []
    for obj in related_objects:
        if not isinstance(obj, str) or obj.strip() == "":
            raise TargetSelectParseError(
                '"Related objects" must be a list of non-empty strings.'
            )
        related_object_strs.append(obj.strip())

    # Cross-check lengths
    if len(related_label_ints) != len(related_object_strs):
        raise TargetSelectParseError(
            '"Related labels" and "Related objects" must have the same length.'
        )

    # Reasoning validation
    if not isinstance(reasoning, str) or reasoning.strip() == "":
        raise TargetSelectParseError('"Reasoning" must be a non-empty string.')
    reasoning_str = reasoning.strip()

    return {
        "Target label": target_label_int,
        "Target object": target_str,
        "Related labels": related_label_ints,
        "Related objects": related_object_strs,
        "Reasoning": reasoning_str,
    }


def parse_best_view_response(
    response_text: str,
    N: int,
    strict: bool = True,
    allow_number_str: bool = True,
) -> Dict[str, Any]:
    """
    Parse + validate VLM output for INSTRUCTIONS_TARGET_SELECT1_TMPL.

    Expected JSON keys (in order):
      - "Image number" (int in [1..N])
      - "Reasoning" (non-empty str)

    Args:
      response_text: resp.output_text
      N: number of views (e.g., 10)
      strict:
        - True: error on extra keys, enforce key order exactly
        - False: allow extra keys (still requires both keys)
      allow_number_str:
        - True: allow "Image number": "10" and convert to int

    Returns:
      {
        "Image number": int,
        "Reasoning": str
      }
    """
    if not isinstance(N, int) or N <= 0:
        raise TargetSelectParseError(f"N must be a positive int, got: {N}")

    EXPECTED_KEYS = ["Image number", "Reasoning"]

    # reuse existing extractor
    raw_json = _extract_first_json_object2(response_text)

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise TargetSelectParseError(
            f"JSON parsing failed: {e.msg} (line {e.lineno}, col {e.colno})"
        ) from e

    if not isinstance(data, dict):
        raise TargetSelectParseError("Top-level JSON must be an object/dict.")

    keys = list(data.keys())
    missing = [k for k in EXPECTED_KEYS if k not in data]
    extra = [k for k in keys if k not in EXPECTED_KEYS]

    if missing:
        raise TargetSelectParseError(f"Missing required keys: {missing}")
    if strict and extra:
        raise TargetSelectParseError(f"Unexpected extra keys: {extra}")

    if strict:
        filtered_in_order = [k for k in keys if k in EXPECTED_KEYS]
        if filtered_in_order != EXPECTED_KEYS:
            raise TargetSelectParseError(
                "Keys are not in the required order.\n"
                f"Expected: {EXPECTED_KEYS}\n"
                f"Got:      {filtered_in_order}"
            )

    img_num = data["Image number"]
    reasoning = data["Reasoning"]

    # ---- Image number validation ----
    if isinstance(img_num, bool):
        # bool is subclass of int -> reject explicitly
        raise TargetSelectParseError('"Image number" must be an integer, not boolean.')

    if isinstance(img_num, int):
        img_int = img_num
    elif allow_number_str and isinstance(img_num, str):
        s = img_num.strip()
        if s.isdigit():
            img_int = int(s)
        else:
            raise TargetSelectParseError(
                '"Image number" string must contain only digits (e.g., "10").'
            )
    else:
        raise TargetSelectParseError(
            '"Image number" must be an integer (or digit string if allow_number_str=True).'
        )

    if not (1 <= img_int <= N):
        raise TargetSelectParseError(
            f'"Image number" must be within [1, {N}], got: {img_int}'
        )

    # ---- Reasoning validation ----
    if not isinstance(reasoning, str) or reasoning.strip() == "":
        raise TargetSelectParseError('"Reasoning" must be a non-empty string.')
    reasoning_str = reasoning.strip()

    return {"Image number": img_int, "Reasoning": reasoning_str}


def parse_best_view_response_v2(
    response_text: str,
    N: int,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Parse + validate VLM output for INSTRUCTIONS_TARGET_SELECT1_1_TMPL.

    Expected JSON keys (in order):
      - "Faithfulness" : exactly "Yes" or "No"
      - "Image number" : integer in [1..N]  (NOTE: string like "10" is NOT allowed)
      - "Reasoning"    : non-empty string

    strict=True:
      - errors on extra keys and wrong key order.
    strict=False:
      - allows extra keys (but still requires all required keys).
    """
    if not isinstance(N, int) or N <= 0:
        raise TargetSelectParseError(f"N must be a positive int, got: {N}")

    EXPECTED_KEYS = ["Faithfulness", "Image number", "Reasoning"]
    ALLOWED_FAITH = {"Yes", "No"}

    raw_json = _extract_first_json_object2(response_text)

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise TargetSelectParseError(
            f"JSON parsing failed: {e.msg} (line {e.lineno}, col {e.colno})"
        ) from e

    if not isinstance(data, dict):
        raise TargetSelectParseError("Top-level JSON must be an object/dict.")

    keys = list(data.keys())
    missing = [k for k in EXPECTED_KEYS if k not in data]
    extra = [k for k in keys if k not in EXPECTED_KEYS]
    if missing:
        raise TargetSelectParseError(f"Missing required keys: {missing}")
    if strict and extra:
        raise TargetSelectParseError(f"Unexpected extra keys: {extra}")

    if strict:
        filtered_in_order = [k for k in keys if k in EXPECTED_KEYS]
        if filtered_in_order != EXPECTED_KEYS:
            raise TargetSelectParseError(
                "Keys are not in the required order.\n"
                f"Expected: {EXPECTED_KEYS}\n"
                f"Got:      {filtered_in_order}"
            )

    # ---- Faithfulness ----
    faith = data["Faithfulness"]
    if not isinstance(faith, str):
        raise TargetSelectParseError('"Faithfulness" must be a string ("Yes" or "No").')
    faith_str = faith.strip()
    if faith_str not in ALLOWED_FAITH:
        raise TargetSelectParseError('"Faithfulness" must be exactly "Yes" or "No".')

    # ---- Image number ----
    img_num = data["Image number"]

    # bool is subclass of int -> reject explicitly
    if isinstance(img_num, bool):
        raise TargetSelectParseError('"Image number" must be an integer, not boolean.')

    # IMPORTANT: DO NOT allow strings like "10"
    if not isinstance(img_num, int):
        raise TargetSelectParseError('"Image number" must be an integer (string is not allowed).')

    if not (1 <= img_num <= N):
        raise TargetSelectParseError(f'"Image number" must be within [1, {N}], got: {img_num}')

    # ---- Reasoning ----
    reasoning = data["Reasoning"]
    if not isinstance(reasoning, str) or reasoning.strip() == "":
        raise TargetSelectParseError('"Reasoning" must be a non-empty string.')
    reasoning_str = reasoning.strip()

    return {
        "Faithfulness": faith_str,
        "Image number": img_num,
        "Reasoning": reasoning_str,
    }
