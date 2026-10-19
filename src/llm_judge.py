"""Evidence-grounded, secondary evaluation of Qwen assessment output via Ollama."""

import json
import ollama

# ============================================================
# CONFIG
# ============================================================

DEFAULT_JUDGE_MODEL = "qwen3.5:4b"
JUDGE_CONTEXT_WINDOW = 32000
JUDGE_MAX_OUTPUT_TOKENS = 2048
JUDGE_TEMPERATURE = 0

# ============================================================
# EVIDENCE HELPERS
# ============================================================

def _text(value, default=""):
    """Return a safe prompt string without serialising Python None as JSON null."""

    if value is None:
        return default

    if isinstance(value, str):
        return value

    return str(value)


def _list(value):
    """Return a list or an empty list for malformed/empty input."""

    return value if isinstance(value, list) else []


def _chunk_id(value):
    """Return a non-empty chunk identifier, or None when it cannot be cited."""

    if not isinstance(value, str):
        return None

    value = value.strip()
    return value or None

def _evidence_for_prompt(items):
    """Prepare evidence text and chunk IDs for the judge prompt."""

    prepared = []

    for item in _list(items):
        if not isinstance(item, dict):
            continue

        chunk_id = _chunk_id(item.get("chunk_id"))
        if chunk_id is None:
            continue

        prepared.append({
            "chunk_id": chunk_id,
            "text": _text(item.get("text")),
        })

    return prepared

def _cited_ids(items):
    """Extract chunk IDs cited by Qwen."""

    return [
        chunk_id
        for item in _list(items)
        if isinstance(item, dict)
        for chunk_id in [_chunk_id(item.get("chunk_id"))]
        if chunk_id is not None
    ]


# ============================================================
# PROMPT
# ============================================================

def build_judge_prompt(
    evidence_package,
    assessment_result
):
    """
    Build an evidence-grounded prompt for Qwen.

    Qwen evaluates the assessment output only.
    It must not perform independent retrieval or add
    requirements from external knowledge.
    """

    package = (
        evidence_package
        if isinstance(evidence_package, dict)
        else {}
    )

    result = (
        assessment_result
        if isinstance(assessment_result, dict)
        else {}
    )

    payload = {

        "sub_kriteria": {

            "kode":
                _text(package.get("kode") or result.get("kode")),

            "nama":
                _text(package.get("nama") or result.get("nama")),

            "objective":
                _text(package.get("objective")),

            "assessment_question":
                _text(package.get("assessment_question")),

            "evidence_to_find":
                [
                    _text(item)
                    for item in _list(package.get("evidence_to_find"))
                    if item is not None
                ],
        },

        "evidence_juknis":
            _evidence_for_prompt(
                package.get("juknis", [])
            ),

        "evidence_regulasi":
            _evidence_for_prompt(
                package.get("regulasi", [])
            ),

        "output_sistem_qwen": {

            "status":
                _text(result.get("status")),

            "assessment_level":
                result.get("assessment_level"),

            "alasan":
                _text(result.get("alasan")),

            "evidence_juknis_ids":
                _cited_ids(
                    result.get(
                        "evidence_juknis",
                        []
                    )
                ),

            "evidence_regulasi_ids":
                _cited_ids(
                    result.get(
                        "evidence_regulasi",
                        []
                    )
                ),
        },
    }

    return f"""
Anda adalah LLM-as-a-Judge yang mengevaluasi OUTPUT
sistem assessment Qwen.

Anda BUKAN melakukan assessment ulang dokumen
berdasarkan pengetahuan Anda sendiri.

Gunakan HANYA informasi yang terdapat di dalam
PAKET EVALUASI.

ATURAN WAJIB:

1. Jangan gunakan pengetahuan eksternal.

2. Jangan melakukan retrieval.

3. Jangan menambah requirement yang tidak terdapat
   dalam paket evaluasi.

4. Jangan mengasumsikan requirement tertentu wajib
   ada jika requirement tersebut tidak disebutkan
   dalam sub-kriteria, objective, assessment question,
   evidence_to_find, atau evidence yang diberikan.

5. Jangan memberikan nilai lebih tinggi hanya karena
   alasan Qwen panjang atau terlihat meyakinkan.

6. Evaluasi apakah kesimpulan Qwen benar-benar
   didukung oleh evidence yang tersedia.

7. Jangan membuat chunk_id baru.

8. supported_chunk_ids HANYA boleh berisi chunk_id
   yang tersedia di paket evaluasi.

9. Jika evidence tidak cukup untuk menentukan suatu
   penilaian, gunakan null.

10. Jangan menebak.

11. Jangan memperbaiki assessment Qwen berdasarkan
    pengetahuan di luar paket evaluasi.

12. Fokus pada evidence yang benar-benar tersedia,
    bukan evidence yang seharusnya ada.

PENILAIAN:

1. decision_correctness

Apakah status Qwen konsisten dengan sub-kriteria
dan evidence?

- 1 = ya
- 0 = tidak
- null = evidence tidak cukup


2. reasoning_faithfulness

Apakah alasan Qwen hanya membuat klaim yang didukung
oleh evidence?

- 1 = ya
- 0 = tidak
- null = evidence tidak cukup


3. citation_groundedness

Apakah chunk_id yang dikutip Qwen:

- tersedia dalam paket evidence, DAN
- relevan dengan alasan/status yang diberikan?

Nilai:

- 1 = ya
- 0 = tidak
- null = tidak dapat dinilai


4. completeness

Apakah alasan Qwen mencakup requirement inti
yang dapat dinilai berdasarkan paket evaluasi?

- 2 = lengkap
- 1 = sebagian
- 0 = tidak memadai
- null = tidak dapat dinilai


VERDICT:

- "pass"
  Jika output Qwen secara keseluruhan dapat dipercaya
  berdasarkan evidence yang tersedia.

- "fail"
  Jika terdapat kesalahan material pada keputusan,
  reasoning, atau citation.

- "undeterminable"
  Jika evidence tidak cukup untuk menentukan.


OUTPUT:

Keluarkan JSON VALID SAJA.

Jangan menggunakan markdown.
Jangan menggunakan ```json.
Jangan menambahkan penjelasan di luar JSON.

Format:

{{
  "kode": "...",
  "decision_correctness": 1,
  "reasoning_faithfulness": 1,
  "citation_groundedness": 1,
  "completeness": 2,
  "verdict": "pass",
  "supported_chunk_ids": ["..."],
  "rationale": "ringkas dan berbasis evidence"
}}

PAKET EVALUASI:

{json.dumps(
    payload,
    ensure_ascii=False,
    indent=2
)}
""".strip()


# ============================================================
# VALIDATION HELPERS
# ============================================================

def _binary(value):
    """Accept only integer 0 or 1."""

    return (
        value
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value in (0, 1)
        )
        else None
    )


def _parse_json_response(raw):
    """
    Parse judge response safely.

    Supports:
    - normal JSON
    - JSON wrapped in ```json ... ```
    """

    if isinstance(raw, dict):
        return raw

    if not isinstance(raw, str):
        raise ValueError(
            "Response judge bukan string atau dictionary."
        )

    raw = raw.strip()

    if not raw:
        raise ValueError(
            "Response judge kosong."
        )

    # --------------------------------------------------------
    # Attempt 1: normal JSON
    # --------------------------------------------------------

    try:
        return json.loads(raw)

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Attempt 2: Markdown JSON fence
    # --------------------------------------------------------

    if raw.startswith("```"):

        lines = raw.splitlines()

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip() == "```"
        ):
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Judge mengembalikan Markdown JSON "
                "tetapi isinya tidak valid."
            ) from exc

    raise ValueError(
        "Response judge bukan JSON yang valid."
    )


def _response_text(response):
    """Extract the assistant text from an Ollama chat response."""

    if isinstance(response, dict):
        text = response.get("message", {}).get("content")
    else:
        message = getattr(response, "message", None)
        text = getattr(message, "content", None)

    if isinstance(text, str) and text.strip():
        return text

    raise RuntimeError("Ollama tidak mengembalikan response text.")


# ============================================================
# NORMALISE JUDGMENT
# ============================================================

def normalise_judgment(
    parsed,
    kode,
    allowed_chunk_ids
):
    """
    Reject malformed output and hallucinated citations.
    """

    fallback = {

        "kode":
            kode,

        "decision_correctness":
            None,

        "reasoning_faithfulness":
            None,

        "citation_groundedness":
            None,

        "completeness":
            None,

        "verdict":
            "undeterminable",

        "supported_chunk_ids":
            [],

        "rationale":
            "Output judge tidak valid.",
    }

    if not isinstance(parsed, dict):
        return fallback

    # --------------------------------------------------------
    # Completeness
    # --------------------------------------------------------

    completeness = parsed.get(
        "completeness"
    )

    if (
        not isinstance(
            completeness,
            int
        )
        or isinstance(
            completeness,
            bool
        )
        or completeness not in (0, 1, 2)
    ):
        completeness = None

    # --------------------------------------------------------
    # Verdict
    # --------------------------------------------------------

    verdict = parsed.get(
        "verdict"
    )

    if verdict not in {
        "pass",
        "fail",
        "undeterminable",
    }:
        verdict = "undeterminable"

    # --------------------------------------------------------
    # Supported chunk IDs
    # --------------------------------------------------------

    ids = parsed.get(
        "supported_chunk_ids",
        []
    )

    if isinstance(ids, list):

        ids = [
            chunk_id

            for chunk_id in ids

            if (
                isinstance(
                    chunk_id,
                    str
                )
                and chunk_id in allowed_chunk_ids
            )
        ]

    else:
        ids = []

    # Remove duplicates
    ids = list(
        dict.fromkeys(ids)
    )

    # --------------------------------------------------------
    # Rationale
    # --------------------------------------------------------

    rationale = parsed.get(
        "rationale"
    )

    if (
        isinstance(
            rationale,
            str
        )
        and rationale.strip()
    ):
        rationale = rationale.strip()

    else:
        rationale = fallback[
            "rationale"
        ]

    # --------------------------------------------------------
    # Final normalised result
    # --------------------------------------------------------

    return {

        "kode":
            kode,

        "decision_correctness":
            _binary(
                parsed.get(
                    "decision_correctness"
                )
            ),

        "reasoning_faithfulness":
            _binary(
                parsed.get(
                    "reasoning_faithfulness"
                )
            ),

        "citation_groundedness":
            _binary(
                parsed.get(
                    "citation_groundedness"
                )
            ),

        "completeness":
            completeness,

        "verdict":
            verdict,

        "supported_chunk_ids":
            ids,

        "rationale":
            rationale,
    }


# ============================================================
# OLLAMA
# ============================================================

def _call_ollama(
    prompt,
    model
):
    """Call a local Ollama model as the secondary LLM judge."""

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Anda adalah LLM-as-a-Judge. Keluarkan JSON valid saja tanpa markdown.",
            },
            {"role": "user", "content": prompt},
        ],
        think=False,
        stream=False,
        format="json",
        options={
            "temperature": JUDGE_TEMPERATURE,
            "num_ctx": JUDGE_CONTEXT_WINDOW,
            "num_predict": JUDGE_MAX_OUTPUT_TOKENS,
        },
    )
    return _response_text(response)


# ============================================================
# MAIN JUDGE
# ============================================================

def judge_all_subcategories(
    evidence_by_subcategory,
    assessment_results,
    model=DEFAULT_JUDGE_MODEL,
    call_model=None,
):
    """
    Judge all Qwen results using a local Ollama model.

    call_model dapat digunakan untuk testing
    tanpa benar-benar memanggil API.
    """

    if (
        not isinstance(
            evidence_by_subcategory,
            dict
        )
        or not isinstance(
            assessment_results,
            dict
        )
    ):
        raise TypeError(
            "evidence_by_subcategory dan "
            "assessment_results harus berupa dict."
        )

    # --------------------------------------------------------
    # Select model caller
    # --------------------------------------------------------

    call_model = call_model or (
        lambda prompt:
            _call_ollama(
                prompt,
                model
            )
    )

    judgments = {}

    # --------------------------------------------------------
    # Process each subcategory
    # --------------------------------------------------------

    for kode, result in (
        assessment_results.items()
    ):

        package = (
            evidence_by_subcategory.get(
                kode,
                {}
            )
        )

        # ----------------------------------------------------
        # Allowed evidence IDs
        # ----------------------------------------------------

        if not isinstance(package, dict):
            package = {}

        allowed_ids = {
            chunk_id
            for source in (
                _list(package.get("juknis")),
                _list(package.get("regulasi")),
            )
            for item in source
            if isinstance(item, dict)
            for chunk_id in [_chunk_id(item.get("chunk_id"))]
            if chunk_id is not None
        }

        # ----------------------------------------------------
        # Call judge
        # ----------------------------------------------------

        try:

            prompt = build_judge_prompt(
                package,
                result
            )

            raw = call_model(
                prompt
            )

            parsed = _parse_json_response(
                raw
            )

            judgment = normalise_judgment(
                parsed,
                kode,
                allowed_ids
            )

        # ----------------------------------------------------
        # Handle errors
        # ----------------------------------------------------

        except Exception as exc:

            judgment = normalise_judgment(
                None,
                kode,
                allowed_ids
            )

            judgment["rationale"] = (
                f"Judge gagal dijalankan: {exc}"
            )

        judgments[kode] = judgment

    return judgments
