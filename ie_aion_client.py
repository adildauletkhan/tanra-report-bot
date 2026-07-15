"""HTTP-клиент backend IE:AION (строительный домен, Фаза 1).

Все функции async, используют aiohttp. Базовый URL — из IE_AION_BACKEND_URL,
аутентификация машинного доступа — заголовок X-Bot-Api-Key (BOT_BACKEND_API_KEY).

Эндпоинты (см. cursor-prompt-backend-phase1.md):
    GET  /api/construction/projects/{id}/active-tasks
    POST /api/construction/projects/{id}/journal
    GET  /api/construction/foremen/by-telegram/{telegram_user_id}
    GET  /api/construction/foremen/by-invite/{invite_code}
    POST /api/construction/foremen/{foreman_id}/link-telegram
"""

from __future__ import annotations

import logging
import os

import aiohttp

logger = logging.getLogger(__name__)

IE_AION_BACKEND_URL = os.environ.get("IE_AION_BACKEND_URL", "").rstrip("/")
BOT_BACKEND_API_KEY = os.environ.get("BOT_BACKEND_API_KEY", "")

_HTTP_TIMEOUT_SECONDS = float(os.environ.get("IE_AION_HTTP_TIMEOUT_SECONDS", "20"))


class IeAionBackendError(Exception):
    """Ошибка обращения к backend IE:AION (сеть/таймаут/5xx/невалидный ответ)."""


def ensure_config() -> None:
    """Проверка обязательных переменных окружения на старте (вызывается из bot.py)."""
    missing = [
        name
        for name, value in (
            ("IE_AION_BACKEND_URL", IE_AION_BACKEND_URL),
            ("BOT_BACKEND_API_KEY", BOT_BACKEND_API_KEY),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Не заданы переменные окружения для backend IE:AION: "
            + ", ".join(missing)
            + ". Добавьте их в Railway → Variables."
        )


def _headers() -> dict[str, str]:
    return {"X-Bot-Api-Key": BOT_BACKEND_API_KEY}


def _url(path: str) -> str:
    return f"{IE_AION_BACKEND_URL}{path}"


async def _request(method: str, path: str, *, json: dict | None = None, allow_404: bool = False):
    timeout = aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_SECONDS)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(method, _url(path), json=json, headers=_headers()) as resp:
                if resp.status == 404 and allow_404:
                    return None
                body = await resp.text()
                if resp.status >= 400:
                    logger.error("IE:AION backend %s %s -> HTTP %s: %s", method, path, resp.status, body[:300])
                    raise IeAionBackendError(f"HTTP {resp.status}: {body[:300]}")
                if not body:
                    return None
                return await resp.json() if resp.content_type == "application/json" else body
    except aiohttp.ClientError as exc:
        logger.error("IE:AION backend network error %s %s: %s", method, path, exc)
        raise IeAionBackendError(f"Сетевая ошибка при обращении к backend: {exc}") from exc
    except TimeoutError as exc:
        logger.error("IE:AION backend timeout %s %s", method, path)
        raise IeAionBackendError("Таймаут при обращении к backend IE:AION") from exc


# --------------------------------------------------------------------------
# WBS / отчёты
# --------------------------------------------------------------------------


async def get_active_tasks(project_id: str) -> list[dict]:
    """Активные задачи WBS в формате active_wbs_tasks (поле в поле для LLM)."""
    data = await _request("GET", f"/api/construction/projects/{project_id}/active-tasks")
    return data or []


async def get_resources_catalog(project_id: str) -> list[dict]:
    """
    Справочник ресурсов (материалы/подрядчики) для сопоставления в LLM.

    TODO: соответствующего backend-эндпоинта в Фазе 1 нет. Когда появится
    GET /api/construction/projects/{id}/resources — заменить заглушку на реальный
    вызов. Пока возвращаем пустой список, чтобы не блокировать структурирование.
    """
    return []


async def submit_journal_entries(
    project_id: str,
    entries: list[dict],
    author_foreman_id: str | None,
    raw_quote: str | None,
    report_date: str,
) -> dict:
    """POST /journal — применяет подтверждённые entries к WBS/журналу."""
    payload = {
        "entries": entries,
        "author_foreman_id": author_foreman_id,
        "raw_quote": raw_quote,
        "report_date": report_date,
    }
    result = await _request("POST", f"/api/construction/projects/{project_id}/journal", json=payload)
    return result or {}


# --------------------------------------------------------------------------
# Бригадиры / привязка Telegram
# --------------------------------------------------------------------------


async def find_foreman_by_telegram(telegram_user_id: int) -> dict | None:
    """Найти привязанного бригадира по telegram_user_id (404 -> None)."""
    return await _request(
        "GET", f"/api/construction/foremen/by-telegram/{telegram_user_id}", allow_404=True
    )


async def find_foreman_by_invite(invite_code: str) -> dict | None:
    """Найти бригадира по коду приглашения перед привязкой (404 -> None)."""
    return await _request(
        "GET", f"/api/construction/foremen/by-invite/{invite_code}", allow_404=True
    )


async def link_foreman_telegram(foreman_id: str, telegram_user_id: int, invite_code: str) -> dict:
    """Привязать Telegram-аккаунт к бригадиру (проверяет соответствие invite_code)."""
    payload = {"telegram_user_id": telegram_user_id, "invite_code": invite_code}
    result = await _request(
        "POST", f"/api/construction/foremen/{foreman_id}/link-telegram", json=payload
    )
    return result or {}
