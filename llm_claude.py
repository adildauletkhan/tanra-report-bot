"""LLM-структурирование голосовых отчётов через Anthropic Claude.

Публичный интерфейс:

    async def structure(transcript, project_id, default_zone_hint,
                        active_wbs_tasks, resources_catalog) -> dict

Возвращает structured_data по схеме voice-report-system-prompt.md §5.1, дополненную
полем summary_line (человекочитаемая сводка для Telegram).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from pathlib import Path

from anthropic import APIError, AsyncAnthropic

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# Актуальная модель Claude Sonnet. Переопределяется через env при выходе новой.
CLAUDE_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
_DEFAULT_PROMPT = Path(__file__).resolve().parent / "voice-report-system-prompt.md"
SYSTEM_PROMPT_PATH = os.environ.get("SYSTEM_PROMPT_PATH", str(_DEFAULT_PROMPT))
MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "2000"))

_SYSTEM_PROMPT: str | None = None
_client: AsyncAnthropic | None = None


def ensure_config() -> None:
    """Проверка обязательных переменных окружения и наличия файла системного промпта."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "Не задана переменная окружения ANTHROPIC_API_KEY. Добавьте её в Railway → Variables."
        )
    _load_system_prompt()  # проверяем, что файл читается на старте


def _load_system_prompt() -> str:
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        path = Path(SYSTEM_PROMPT_PATH)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        try:
            _SYSTEM_PROMPT = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(
                f"Не удалось прочитать системный промпт '{path}': {exc}"
            ) from exc
        if not _SYSTEM_PROMPT.strip():
            raise RuntimeError(f"Системный промпт пуст: {path}")
        logger.info("Loaded system prompt from %s (%d chars)", path, len(_SYSTEM_PROMPT))
    return _SYSTEM_PROMPT


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def _build_user_payload(
    transcript: str,
    project_id: str | None,
    default_zone_hint: str | None,
    active_wbs_tasks: list[dict],
    resources_catalog: list[dict],
) -> dict:
    return {
        "audio_transcript": transcript,
        "language_hint": "mixed",
        "context": {
            "project_id": project_id,
            "report_date": dt.date.today().isoformat(),
            "default_zone_hint": default_zone_hint,
            "active_wbs_tasks": active_wbs_tasks,
            "resources_catalog": resources_catalog,
        },
    }


async def _call_claude(user_payload: dict, strict: bool = False) -> str:
    """Один вызов модели. Возвращает сырой текст ответа."""
    instruction = (
        "Ответь ТОЛЬКО валидным JSON объектом structured_data по схеме из системного "
        "промпта (§5.1), без markdown, без пояснений, без обёртки в ```json. "
        "Первый символ ответа должен быть '{', последний — '}'."
    )
    if strict:
        instruction += (
            " Предыдущий ответ не распарсился как JSON. Верни строго один JSON-объект."
        )

    user_content = instruction + "\n\nВходные данные:\n" + json.dumps(
        user_payload, ensure_ascii=False
    )

    # Без assistant-prefill: на Claude Sonnet 4.6+ / Sonnet 5 prefill даёт HTTP 400.
    resp = await _get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=_load_system_prompt(),
        messages=[
            {"role": "user", "content": user_content},
        ],
    )
    text = "".join(
        block.text for block in resp.content if getattr(block, "type", None) == "text"
    )
    return text


def _summary_line(structured: dict) -> str:
    entries = structured.get("entries") or []
    if not entries:
        # Покажем хотя бы сырой фрагмент, если он есть в needs_clarification.
        clar = structured.get("needs_clarification") or []
        if clar and clar[0].get("raw_fragment"):
            return f"Распознано: {clar[0]['raw_fragment']}"
        return "Отчёт не распознан по зонам — требуется уточнение."
    parts: list[str] = []
    for e in entries[:3]:
        zone = e.get("zone") or "зона не указана"
        work = e.get("work_type") or "работа"
        fact = e.get("fact_pct")
        fact_s = f"{fact}%" if fact is not None else "?"
        risk = (e.get("risk") or {}).get("description")
        seg = f"{zone} · {work}: факт {fact_s}"
        if risk:
            seg += f" · риск: {risk}"
        parts.append(seg)
    return "\n".join(parts)


def _synthetic_entry(
    transcript: str, default_zone_hint: str | None, reason: str
) -> dict:
    """Минимальная запись для журнала, когда LLM не смог разобрать отчёт."""
    return {
        "task_id": None,
        "wbs_code": None,
        "zone": default_zone_hint,
        "work_type": "голосовой отчёт",
        "match_confidence": "no_context",
        "plan_pct": None,
        "fact_pct": None,
        "delta_pct": None,
        "blocker": {"type": "none", "description": None},
        "risk": {"severity": "none", "delay_days": None, "description": None},
        "responsible": None,
        "recommended_actions": [],
        "notes": (transcript or "").strip()[:1000] or reason,
    }


def fallback_structure(
    transcript: str, reason: str, default_zone_hint: str | None = None
) -> dict:
    """Публичный fallback: всегда ≥1 entry, чтобы журнал не оставался пустым."""
    snippet = (transcript or "").strip()
    if len(snippet) > 500:
        snippet = snippet[:497] + "…"
    entry = _synthetic_entry(transcript, default_zone_hint, reason)
    return {
        "entries": [entry],
        "needs_clarification": [{"reason": reason, "raw_fragment": (transcript or "")[:200]}],
        "summary_line": f"Распознано (черновик без структуры):\n{snippet}",
    }


async def structure(
    transcript: str,
    project_id: str | None,
    default_zone_hint: str | None,
    active_wbs_tasks: list[dict],
    resources_catalog: list[dict],
) -> dict:
    """Структурирует транскрипт в structured_data (+ summary_line).

    При любой ошибке API/парсинга возвращает fallback с сырым текстом —
    бот всё равно покажет черновик, а не «сервис недоступен».
    """
    user_payload = _build_user_payload(
        transcript, project_id, default_zone_hint, active_wbs_tasks, resources_catalog
    )

    try:
        raw = await _call_claude(user_payload, strict=False)
        structured = _parse(raw)
        if structured is None:
            logger.warning(
                "Claude вернул невалидный JSON, повтор. raw[:300]=%r", (raw or "")[:300]
            )
            raw = await _call_claude(user_payload, strict=True)
            structured = _parse(raw)
    except APIError as exc:
        logger.exception("Anthropic API error (model=%s): %s", CLAUDE_MODEL, exc)
        return fallback_structure(
            transcript,
            f"anthropic_api_error:{getattr(exc, 'status_code', '?')}",
            default_zone_hint,
        )
    except Exception as exc:
        logger.exception("Claude structure unexpected error: %s", exc)
        return fallback_structure(
            transcript, f"structure_error:{type(exc).__name__}", default_zone_hint
        )

    if structured is None:
        logger.error("Claude не вернул валидный JSON после повторной попытки")
        return fallback_structure(transcript, "parse_error", default_zone_hint)

    structured.setdefault("entries", [])
    structured.setdefault("needs_clarification", [])

    # Claude иногда возвращает валидный JSON, но с пустым entries[] — не даём
    # подтвердить «пустой» отчёт: кладём синтетическую запись в очередь ПТО.
    if not structured["entries"] and (transcript or "").strip():
        logger.warning("Claude вернул пустой entries[], добавляем synthetic entry")
        structured["entries"] = [
            _synthetic_entry(transcript, default_zone_hint, "empty_entries")
        ]
        structured["needs_clarification"].append(
            {"reason": "empty_entries", "raw_fragment": transcript[:200]}
        )

    structured["summary_line"] = _summary_line(structured)

    entries = structured["entries"]
    confidences = [e.get("match_confidence") for e in entries]
    logger.info(
        "Claude structure: model=%s transcript_len=%d entries=%d confidences=%s needs_clarification=%d",
        CLAUDE_MODEL,
        len(transcript),
        len(entries),
        ",".join(str(c) for c in confidences) or "n/a",
        len(structured["needs_clarification"]),
    )
    logger.debug("Claude structured_data: %s", json.dumps(structured, ensure_ascii=False))
    return structured


def _parse(raw: str) -> dict | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    # Срезаем markdown-обёртку, если модель всё же добавила.
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Иногда модель добавляет текст до/после JSON — ищем первый объект.
        start = raw.find("{")
        if start < 0:
            return None
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw, start)
            parsed = obj
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None
