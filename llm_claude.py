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

from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# Актуальная модель Claude Sonnet. Переопределяется через env при выходе новой.
CLAUDE_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
SYSTEM_PROMPT_PATH = os.environ.get("SYSTEM_PROMPT_PATH", "voice-report-system-prompt.md")
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
        try:
            with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as fh:
                _SYSTEM_PROMPT = fh.read()
        except OSError as exc:
            raise RuntimeError(
                f"Не удалось прочитать системный промпт '{SYSTEM_PROMPT_PATH}': {exc}"
            ) from exc
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
    """Один вызов модели. Возвращает сырой текст ответа (с prefill '{' в начале)."""
    instruction = (
        "Ответь ТОЛЬКО валидным JSON объектом structured_data по схеме из системного "
        "промпта (§5.1), без markdown, без пояснений, без обёртки в ```json."
    )
    if strict:
        instruction += (
            " Предыдущий ответ не распарсился как JSON. Верни строго один JSON-объект, "
            "начинающийся с { и заканчивающийся }."
        )

    user_content = instruction + "\n\nВходные данные:\n" + json.dumps(user_payload, ensure_ascii=False)

    resp = await _get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=_load_system_prompt(),
        messages=[
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "{"},  # prefill — форсируем JSON-объект
        ],
    )
    text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")
    # prefill "{" не входит в ответ модели — возвращаем обратно.
    return "{" + text


def _summary_line(structured: dict) -> str:
    entries = structured.get("entries") or []
    if not entries:
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


def _fallback(transcript: str, reason: str) -> dict:
    return {
        "entries": [],
        "needs_clarification": [{"reason": reason, "raw_fragment": transcript[:200]}],
        "summary_line": "Не удалось разобрать отчёт, требуется ручная проверка",
    }


async def structure(
    transcript: str,
    project_id: str | None,
    default_zone_hint: str | None,
    active_wbs_tasks: list[dict],
    resources_catalog: list[dict],
) -> dict:
    """Структурирует транскрипт в structured_data (+ summary_line)."""
    user_payload = _build_user_payload(
        transcript, project_id, default_zone_hint, active_wbs_tasks, resources_catalog
    )

    raw = await _call_claude(user_payload, strict=False)
    structured = _parse(raw)
    if structured is None:
        logger.warning("Claude вернул невалидный JSON, повторная попытка со строгой инструкцией")
        raw = await _call_claude(user_payload, strict=True)
        structured = _parse(raw)

    if structured is None:
        logger.error("Claude не вернул валидный JSON после повторной попытки")
        return _fallback(transcript, "parse_error")

    structured.setdefault("entries", [])
    structured.setdefault("needs_clarification", [])
    structured["summary_line"] = _summary_line(structured)

    entries = structured["entries"]
    confidences = [e.get("match_confidence") for e in entries]
    logger.info(
        "Claude structure: transcript_len=%d entries=%d confidences=%s needs_clarification=%d",
        len(transcript),
        len(entries),
        ",".join(str(c) for c in confidences) or "n/a",
        len(structured["needs_clarification"]),
    )
    logger.debug("Claude structured_data: %s", json.dumps(structured, ensure_ascii=False))
    return structured


def _parse(raw: str) -> dict | None:
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Иногда модель добавляет текст после JSON — пробуем взять первый объект.
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw)
            parsed = obj
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None
