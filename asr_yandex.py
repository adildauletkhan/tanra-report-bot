"""
ASR через Yandex SpeechKit — AsyncRecognizer, API v3 REST (регион Казахстан).

Публичный интерфейс — единственная функция:

    async def transcribe(audio_path: str, language_hint: str = "mixed") -> str

Импортируется в bot.py как:

    from asr_yandex import transcribe as asr_transcribe

Поток работы (туториал KZ API v3):
    1. POST /stt/v3/recognizeFileAsync — аудио (inline base64) → operation id.
    2. Повторяем GET /stt/v3/getRecognition?operationId=... пока результат
       не готов (404/«not ready» → sleep и retry). Отдельный Operation API
       (GET /operations/{id}) для SpeechKit KZ отдаёт 404 на реальные id —
       его не используем.
    3. Разбираем поток StreamingResponse → транскрипт.

Автоопределение языка (ru/kk и смешанные фразы) включается через
languageRestriction = WHITELIST + ["auto"] — SpeechKit определяет язык по каждому
предложению отдельно.

Формат аудио: Telegram отдаёт голосовые в .ogg (Opus). SpeechKit v3 поддерживает
OggOpus нативно (containerAudioType = OGG_OPUS) — конвертация в WAV/PCM НЕ требуется.
Функция convert_ogg_if_needed() оставлена как точка расширения: если однажды на вход
придёт не-ogg формат, там можно включить конвертацию через ffmpeg. В этом случае
ffmpeg нужно установить в окружении Railway (например, через Dockerfile с
`apt-get install -y ffmpeg` вместо чистого Nixpacks-автодетекта).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time

import aiohttp

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Конфигурация (переменные окружения, задаются в Railway → Variables)
# --------------------------------------------------------------------------
YANDEX_SPEECHKIT_API_KEY = os.environ.get("YANDEX_SPEECHKIT_API_KEY")
YANDEX_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID")

# Эндпойнты региона Казахстан (можно переопределить через env при смене региона).
STT_SERVICE_URL = os.environ.get(
    "YANDEX_STT_SERVICE_URL", "https://stt.api.ml.yandexcloud.kz"
).rstrip("/")

# Параметры опроса getRecognition до готовности результата.
ASR_POLL_TIMEOUT_SECONDS = float(os.environ.get("ASR_POLL_TIMEOUT_SECONDS", "90"))
ASR_POLL_INTERVAL_SECONDS = float(os.environ.get("ASR_POLL_INTERVAL_SECONDS", "1.5"))
# Первая пауза после submit — типично ~10с на минуту аудио; для коротких
# Telegram-голосовых хватает пары секунд.
ASR_INITIAL_WAIT_SECONDS = float(os.environ.get("ASR_INITIAL_WAIT_SECONDS", "2.0"))

# Общий сетевой таймаут одного HTTP-запроса.
_HTTP_TIMEOUT_SECONDS = float(os.environ.get("ASR_HTTP_TIMEOUT_SECONDS", "30"))

# Лимит на inline-передачу аудио в теле запроса. Для более длинных записей нужно
# сначала заливать файл в Object Storage и передавать uri вместо content.
# TODO: при поддержке длинных отчётов (>~10 МБ .ogg) реализовать загрузку в
#       Object Storage и передачу поля "uri" в recognizeFileAsync.
MAX_INLINE_AUDIO_BYTES = int(os.environ.get("ASR_MAX_INLINE_BYTES", str(10 * 1024 * 1024)))


# --------------------------------------------------------------------------
# Типы исключений — чтобы вызывающий код (process_report в bot.py) мог отличать
# ошибки аутентификации/квот от прочих и реагировать по-разному.
# --------------------------------------------------------------------------


class YandexSpeechKitError(Exception):
    """Базовая ошибка интеграции с Yandex SpeechKit."""


class YandexSpeechKitAuthError(YandexSpeechKitError):
    """401/403 — неверный ключ или отсутствие роли ai.speechkit-stt.user."""


class YandexSpeechKitRateLimitError(YandexSpeechKitError):
    """429 — превышен лимит запросов."""


# --------------------------------------------------------------------------
# Конфигурация / заголовки
# --------------------------------------------------------------------------


def ensure_config() -> None:
    """
    Проверка обязательных переменных окружения на старте (аналогично проверке
    TELEGRAM_BOT_TOKEN в bot.py). Вызывается из bot.py при запуске.
    """
    missing = [
        name
        for name, value in (
            ("YANDEX_SPEECHKIT_API_KEY", YANDEX_SPEECHKIT_API_KEY),
            ("YANDEX_FOLDER_ID", YANDEX_FOLDER_ID),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Не заданы переменные окружения для Yandex SpeechKit: "
            + ", ".join(missing)
            + ". Добавьте их в Railway → Variables."
        )


def _auth_headers() -> dict[str, str]:
    headers = {"Authorization": f"Api-Key {YANDEX_SPEECHKIT_API_KEY}"}
    if YANDEX_FOLDER_ID:
        # Для аутентификации по API-ключу каталог обычно определяется самим ключом,
        # но x-folder-id не мешает и полезен для явности/аудита.
        headers["x-folder-id"] = YANDEX_FOLDER_ID
    return headers


async def _raise_for_status(resp: aiohttp.ClientResponse) -> None:
    if 200 <= resp.status < 300:
        return

    body = await resp.text()
    if resp.status == 401:
        logger.error("Yandex SpeechKit 401: неверный API-ключ")
        raise YandexSpeechKitAuthError("401: неверный API-ключ Yandex SpeechKit")
    if resp.status == 403:
        logger.error("Yandex SpeechKit 403: нет прав (роль ai.speechkit-stt.user?)")
        raise YandexSpeechKitAuthError(
            "403: нет прав. Проверьте роль ai.speechkit-stt.user у сервисного аккаунта"
        )
    if resp.status == 429:
        logger.error("Yandex SpeechKit 429: превышен лимит запросов")
        raise YandexSpeechKitRateLimitError("429: превышен лимит запросов SpeechKit")

    logger.error("Yandex SpeechKit HTTP %s: %s", resp.status, body[:500])
    raise YandexSpeechKitError(f"HTTP {resp.status}: {body[:500]}")


# --------------------------------------------------------------------------
# Аудио
# --------------------------------------------------------------------------


def convert_ogg_if_needed(audio_path: str) -> str:
    """
    Возвращает путь к файлу в формате, пригодном для отправки в SpeechKit.

    Telegram присылает .ogg (Opus), который SpeechKit v3 принимает нативно
    (containerAudioType = OGG_OPUS), поэтому конвертация не требуется и возвращается
    исходный путь.

    Если однажды на вход придёт другой формат, здесь можно включить конвертацию в
    LPCM 16-bit / 48000 Hz / mono через ffmpeg (потребует установленного ffmpeg в
    окружении — см. модульный docstring).
    """
    return audio_path


# --------------------------------------------------------------------------
# Шаги API
# --------------------------------------------------------------------------


def _is_not_ready_status(status: int, body: str) -> bool:
    """getRecognition ещё не готов — нужно подождать и повторить."""
    if status in (404, 409, 425, 429):
        return True
    if status == 400:
        lower = (body or "").lower()
        return any(
            marker in lower
            for marker in (
                "not ready",
                "not_ready",
                "not found",
                "not_found",
                "failed_precondition",
                "unavailable",
                "try again",
            )
        )
    return False


async def _submit_recognition(session: aiohttp.ClientSession, audio_bytes: bytes) -> str:
    """recognizeFileAsync → возвращает operation_id.

    Тело в camelCase — как в REST OpenAPI SpeechKit v3 KZ.
    """
    payload = {
        "content": base64.b64encode(audio_bytes).decode("ascii"),
        "recognitionModel": {
            "model": "general",
            "audioFormat": {
                "containerAudio": {"containerAudioType": "OGG_OPUS"},
            },
            "textNormalization": {
                "textNormalization": "TEXT_NORMALIZATION_ENABLED",
                "profanityFilter": False,
                "literatureText": False,
            },
            "languageRestriction": {
                "restrictionType": "WHITELIST",
                "languageCode": ["ru-RU", "kk", "auto"],
            },
        },
    }

    url = f"{STT_SERVICE_URL}/stt/v3/recognizeFileAsync"
    async with session.post(url, json=payload, headers=_auth_headers()) as resp:
        await _raise_for_status(resp)
        data = await resp.json()

    operation_id = data.get("id")
    if not operation_id:
        raise YandexSpeechKitError(
            f"recognizeFileAsync не вернул operation id: {json.dumps(data)[:300]}"
        )
    return operation_id


async def _wait_recognition_result(
    session: aiohttp.ClientSession, operation_id: str
) -> str:
    """
    Опрашивает getRecognition до готовности результата.

    Как в туториале KZ: после submit ждём и забираем результат через
    getRecognition, без Operation.Get (тот для SpeechKit KZ отвечает 404).
    """
    url = f"{STT_SERVICE_URL}/stt/v3/getRecognition"
    params = {"operationId": operation_id, "operation_id": operation_id}
    deadline = time.monotonic() + ASR_POLL_TIMEOUT_SECONDS

    await asyncio.sleep(ASR_INITIAL_WAIT_SECONDS)

    last_status = None
    last_body = ""
    while time.monotonic() < deadline:
        async with session.get(url, params=params, headers=_auth_headers()) as resp:
            last_status = resp.status
            last_body = await resp.text()

            if 200 <= resp.status < 300:
                if last_body.strip():
                    return last_body
                # Пустое тело — ещё рано или тишина; подождём ещё немного.
                logger.debug("getRecognition 200 but empty body, retry")
            elif resp.status in (401, 403):
                await _raise_for_status(resp)
            elif _is_not_ready_status(resp.status, last_body):
                logger.debug(
                    "getRecognition not ready yet: HTTP %s body[:200]=%r",
                    resp.status,
                    last_body[:200],
                )
            else:
                logger.error(
                    "Yandex SpeechKit getRecognition HTTP %s: %s",
                    resp.status,
                    last_body[:500],
                )
                raise YandexSpeechKitError(
                    f"HTTP {resp.status}: {last_body[:500]}"
                )

        await asyncio.sleep(ASR_POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        "Yandex SpeechKit: превышено время ожидания распознавания "
        f"(last HTTP {last_status}: {last_body[:200]})"
    )


# --------------------------------------------------------------------------
# Разбор результата
# --------------------------------------------------------------------------


def _iter_stream_objects(raw_text: str):
    """
    getRecognition (REST server-streaming) возвращает последовательность JSON-объектов.
    Разбираем устойчиво: сначала пробуем весь ответ как один JSON (объект или массив),
    иначе — как склеенные/построчные JSON-объекты через raw_decode.
    """
    raw_text = raw_text.strip()
    if not raw_text:
        return

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(parsed, list):
            yield from parsed
        else:
            yield parsed
        return

    decoder = json.JSONDecoder()
    idx, length = 0, len(raw_text)
    while idx < length:
        while idx < length and raw_text[idx] in " \r\n\t":
            idx += 1
        if idx >= length:
            break
        try:
            obj, end = decoder.raw_decode(raw_text, idx)
        except json.JSONDecodeError:
            logger.debug("Не удалось разобрать фрагмент ответа getRecognition, пропуск")
            break
        yield obj
        idx = end


def _confidence_value(alt: dict) -> float:
    raw = alt.get("confidence")
    if raw is None or raw == "":
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _best_alternative(alternative_update: dict) -> tuple[str, list[str]]:
    """Из AlternativeUpdate берём альтернативу с максимальным confidence."""
    alternatives = alternative_update.get("alternatives") or []
    if not alternatives:
        return "", []

    best = max(alternatives, key=_confidence_value)
    text = (best.get("text") or "").strip()
    languages = []
    for lang in best.get("languages") or []:
        code = lang.get("languageCode") or lang.get("language_code")
        if code:
            languages.append(code)
    return text, languages


def _extract_transcript(raw_text: str) -> tuple[str, list[str]]:
    """
    Склеивает финальный транскрипт из ответа getRecognition.

    Предпочитаем finalRefinement (нормализованный текст, есть при включённой
    нормализации), иначе — final. Возвращаем (транскрипт, список кодов языков).
    """
    refined_segments: list[str] = []
    final_segments: list[str] = []
    languages: list[str] = []

    for obj in _iter_stream_objects(raw_text):
        # REST-обёртка кладёт событие в "result"; gRPC-стиль — на верхний уровень.
        event = obj.get("result", obj) if isinstance(obj, dict) else {}
        if not isinstance(event, dict):
            continue

        refinement = event.get("finalRefinement") or event.get("final_refinement")
        if refinement:
            normalized = (
                refinement.get("normalizedText")
                or refinement.get("normalized_text")
                or {}
            )
            text, langs = _best_alternative(normalized)
            if text:
                refined_segments.append(text)
            languages.extend(langs)
            continue

        final = event.get("final")
        if final:
            text, langs = _best_alternative(final)
            if text:
                final_segments.append(text)
            languages.extend(langs)

    segments = refined_segments or final_segments
    transcript = " ".join(segments).strip()
    # Сохраняем порядок, убираем дубли языков.
    unique_languages = list(dict.fromkeys(languages))
    return transcript, unique_languages


# --------------------------------------------------------------------------
# Публичная функция
# --------------------------------------------------------------------------


async def transcribe(audio_path: str, language_hint: str = "mixed") -> str:
    """
    Распознаёт речь из аудиофайла через Yandex SpeechKit AsyncRecognizer (v3 REST).

    language_hint оставлен для совместимости сигнатуры; автоопределение языка
    включено на стороне SpeechKit (languageRestriction = auto), поэтому подсказка
    сейчас не используется, но может пригодиться для сужения WHITELIST в будущем.

    Возвращает строку транскрипта. Для тишины/шума (пустой результат) возвращает
    пустую строку (без исключения) — решение о дальнейших действиях принимает
    вызывающий код (process_report).
    """
    ensure_config()

    prepared_path = convert_ogg_if_needed(audio_path)
    with open(prepared_path, "rb") as fh:
        audio_bytes = fh.read()

    if len(audio_bytes) > MAX_INLINE_AUDIO_BYTES:
        # TODO: для длинных записей заливать в Object Storage и передавать uri.
        raise YandexSpeechKitError(
            f"Аудио слишком большое для inline-распознавания "
            f"({len(audio_bytes)} байт > {MAX_INLINE_AUDIO_BYTES}). "
            "Требуется загрузка в Object Storage (см. TODO в asr_yandex.py)."
        )

    started = time.monotonic()
    # sock_read отдельно: upload/base64 и длинный getRecognition не должны
    # укладываться в короткий total на весь session.
    timeout = aiohttp.ClientTimeout(
        total=None,
        sock_connect=min(15.0, _HTTP_TIMEOUT_SECONDS),
        sock_read=max(_HTTP_TIMEOUT_SECONDS, ASR_POLL_TIMEOUT_SECONDS),
    )

    async with aiohttp.ClientSession(timeout=timeout) as session:
        operation_id = await _submit_recognition(session, audio_bytes)
        logger.info("SpeechKit operation_id=%s, audio_bytes=%d", operation_id, len(audio_bytes))
        raw_result = await _wait_recognition_result(session, operation_id)

    transcript, languages = _extract_transcript(raw_result)

    elapsed = time.monotonic() - started
    logger.info(
        "SpeechKit: распознано за %.2fs, длина транскрипта=%d, языки=%s",
        elapsed,
        len(transcript),
        ",".join(languages) if languages else "n/a",
    )
    if not transcript:
        logger.warning(
            "SpeechKit: пустой транскрипт, raw[:500]=%r",
            (raw_result or "")[:500],
        )
    else:
        logger.debug("SpeechKit transcript: %s", transcript)

    return transcript
