"""
Tanra AI Report Bot — сбор голосовых отчётов бригадиров.

Интеграция с backend IE:AION (Фаза 1):
- бригадир привязывается к объекту через код приглашения (/start INV-XXXXXX);
- активные задачи WBS и справочник ресурсов тянутся из backend перед LLM;
- подтверждённый отчёт пишется в суточный журнал (POST /journal).

Команды:
    /start [INV-код] — регистрация/вход; с кодом — привязка к объекту
    /object          — показать текущий привязанный объект (сменить нельзя — это действие ПТО)
    /status          — последний черновик в этой сессии
    /record          — явно начать запись отчёта (голос + опционально фото)

Требует: pip install aiogram aiohttp anthropic
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

import ie_aion_client
import llm_claude
from asr_yandex import ensure_config as ensure_yandex_config
from asr_yandex import transcribe as asr_transcribe
from ie_aion_client import IeAionBackendError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

# Проверяем обязательные переменные окружения всех интеграций на старте.
ensure_yandex_config()
ie_aion_client.ensure_config()
llm_claude.ensure_config()

BACKEND_UNAVAILABLE_MSG = "Сервис временно недоступен, попробуйте позже."


# --------------------------------------------------------------------------
# Локальный кеш привязанных бригадиров (телеграм-сессия процесса).
# Полноценное хранилище — backend IE:AION (Foreman). Кеш ускоряет работу и
# хранит foreman_id/project_id/default_zone между сообщениями.
# TODO: при рестарте процесса кеш пуст — восстанавливается лениво через
#       find_foreman_by_telegram при первом действии пользователя.
# --------------------------------------------------------------------------


@dataclass
class ForemanSession:
    telegram_user_id: int
    foreman_id: str
    project_id: str
    full_name: str
    default_zone: str | None = None


@dataclass
class VoiceReportLog:
    report_id: str
    telegram_user_id: int
    project_id: str | None
    raw_audio_path: str
    raw_transcript: str | None = None
    structured_data: dict | None = None
    confirmed: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)


foreman_cache: dict[int, ForemanSession] = {}
# Последний черновик на процесс (для /status и подтверждения). Полная история
# отчётов теперь живёт в суточном журнале IE:AION.
# TODO: /status показывает только черновик текущей сессии; за историей —
#       модуль «Суточный журнал» в IE:AION.
pending_report: dict[int, VoiceReportLog] = {}


async def resolve_foreman(telegram_user_id: int) -> ForemanSession | None:
    """Вернуть сессию бригадира из кеша или подтянуть из backend по telegram_user_id."""
    cached = foreman_cache.get(telegram_user_id)
    if cached is not None:
        return cached

    data = await ie_aion_client.find_foreman_by_telegram(telegram_user_id)
    if not data or data.get("telegram_link_status") != "linked":
        return None

    session = ForemanSession(
        telegram_user_id=telegram_user_id,
        foreman_id=data["foreman_id"],
        project_id=data["project_id"],
        full_name=data.get("full_name", ""),
        default_zone=data.get("default_zone"),
    )
    foreman_cache[telegram_user_id] = session
    return session


# --------------------------------------------------------------------------
# FSM-состояния
# --------------------------------------------------------------------------


class ReportFlow(StatesGroup):
    waiting_voice = State()
    waiting_confirmation = State()
    waiting_correction = State()


router = Router()


def _confirm_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Подтвердить", callback_data="confirm_report"),
                InlineKeyboardButton(text="✏️ Исправить", callback_data="fix_report"),
                InlineKeyboardButton(text="🗑 Отменить", callback_data="cancel_report"),
            ]
        ]
    )


# --------------------------------------------------------------------------
# /start — регистрация по коду приглашения / вход
# --------------------------------------------------------------------------


@router.message(CommandStart())
async def cmd_start(message: Message, command: CommandObject, state: FSMContext) -> None:
    await state.clear()
    telegram_user_id = message.from_user.id
    invite_code = (command.args or "").strip()

    if invite_code:
        await _handle_invite(message, telegram_user_id, invite_code)
        return

    try:
        session = await resolve_foreman(telegram_user_id)
    except IeAionBackendError:
        await message.answer(BACKEND_UNAVAILABLE_MSG)
        return

    if session is not None:
        await message.answer(
            f"С возвращением, {session.full_name or 'коллега'}!\n"
            "Присылайте голосовые отчёты — я сам разберу зону, работу и статус.\n\n"
            "Команды: /object — текущий объект, /status — последний черновик, "
            "/record — начать запись."
        )
    else:
        await message.answer(
            "Вы ещё не зарегистрированы. Попросите код приглашения у вашего ПТО/прораба "
            "и отправьте его командой:\n/start <код>"
        )


async def _handle_invite(message: Message, telegram_user_id: int, invite_code: str) -> None:
    try:
        foreman = await ie_aion_client.find_foreman_by_invite(invite_code)
        if foreman is None:
            await message.answer(
                "Код приглашения не найден или уже использован. Уточните код у ПТО/прораба."
            )
            return

        linked = await ie_aion_client.link_foreman_telegram(
            foreman["foreman_id"], telegram_user_id, invite_code
        )
    except IeAionBackendError:
        await message.answer(BACKEND_UNAVAILABLE_MSG)
        return

    session = ForemanSession(
        telegram_user_id=telegram_user_id,
        foreman_id=linked.get("foreman_id", foreman["foreman_id"]),
        project_id=linked.get("project_id", foreman["project_id"]),
        full_name=linked.get("full_name", foreman.get("full_name", "")),
        default_zone=linked.get("default_zone", foreman.get("default_zone")),
    )
    foreman_cache[telegram_user_id] = session

    zone_note = f"\nЗона по умолчанию: {session.default_zone}" if session.default_zone else ""
    await message.answer(
        f"Готово! Вы привязаны к объекту.{zone_note}\n\n"
        "Присылайте голосовые отчёты — я сам разберу зону, работу и статус."
    )


# --------------------------------------------------------------------------
# /object — показать текущий привязанный объект (смена — только через ПТО в IE:AION)
# --------------------------------------------------------------------------


@router.message(Command("object"))
async def cmd_object(message: Message) -> None:
    try:
        session = await resolve_foreman(message.from_user.id)
    except IeAionBackendError:
        await message.answer(BACKEND_UNAVAILABLE_MSG)
        return

    if session is None:
        await message.answer(
            "Вы ещё не зарегистрированы. Отправьте код приглашения: /start <код>"
        )
        return

    zone_note = f"\nЗона по умолчанию: {session.default_zone}" if session.default_zone else ""
    await message.answer(
        f"Ваш объект: {session.project_id}{zone_note}\n\n"
        "Сменить объект может только ПТО в системе IE:AION."
    )


# --------------------------------------------------------------------------
# /status — последний черновик в этой сессии
# --------------------------------------------------------------------------


@router.message(Command("status"))
async def cmd_status(message: Message) -> None:
    report = pending_report.get(message.from_user.id)
    if report is None or not report.structured_data:
        await message.answer(
            "Нет активного черновика в этой сессии. Отправьте голосовое сообщение "
            "или используйте /record. История отчётов — в модуле «Суточный журнал» IE:AION."
        )
        return

    status_label = "✅ подтверждён" if report.confirmed else "⏳ ожидает подтверждения"
    summary = report.structured_data.get("summary_line") or report.raw_transcript
    await message.answer(
        f"Последний черновик ({report.created_at:%d.%m.%Y %H:%M})\n"
        f"Статус: {status_label}\n\n{summary or 'обрабатывается...'}"
    )


# --------------------------------------------------------------------------
# /record — явный старт записи
# --------------------------------------------------------------------------


@router.message(Command("record"))
async def cmd_record(message: Message, state: FSMContext) -> None:
    try:
        session = await resolve_foreman(message.from_user.id)
    except IeAionBackendError:
        await message.answer(BACKEND_UNAVAILABLE_MSG)
        return
    if session is None:
        await message.answer("Сначала зарегистрируйтесь: /start <код приглашения>")
        return

    await state.set_state(ReportFlow.waiting_voice)
    await message.answer(
        "🎤 Записываю. Отправьте голосовое сообщение с отчётом "
        "(можно приложить фото захватки)."
    )


# --------------------------------------------------------------------------
# Обработка голосового отчёта
# --------------------------------------------------------------------------


async def download_voice(bot: Bot, message: Message) -> str:
    file = await bot.get_file(message.voice.file_id)
    local_path = f"/tmp/{message.voice.file_id}.ogg"
    await bot.download_file(file.file_path, local_path)
    return local_path


async def _structure_transcript(session: ForemanSession, transcript: str) -> dict:
    """Подтягивает контекст WBS/ресурсов из backend и структурирует через Claude."""
    try:
        active_wbs_tasks = await ie_aion_client.get_active_tasks(session.project_id)
        resources_catalog = await ie_aion_client.get_resources_catalog(session.project_id)
    except IeAionBackendError:
        # WBS-контекст недоступен — структурируем без него (LLM пометит no_context).
        logger.warning("Не удалось получить active-tasks/resources, работаем без WBS-контекста")
        active_wbs_tasks, resources_catalog = [], []

    return await llm_claude.structure(
        transcript=transcript,
        project_id=session.project_id,
        default_zone_hint=session.default_zone,
        active_wbs_tasks=active_wbs_tasks,
        resources_catalog=resources_catalog,
    )


async def process_report(
    bot: Bot, message: Message, state: FSMContext, audio_path: str
) -> None:
    session = await resolve_foreman(message.from_user.id)
    if session is None:
        await message.answer("Сначала зарегистрируйтесь: /start <код приглашения>")
        await state.clear()
        return

    await message.answer("⏳ Обрабатываю...")

    transcript = await asr_transcribe(audio_path, language_hint="mixed")
    if not transcript.strip():
        await message.answer(
            "Не удалось распознать речь, попробуйте ещё раз в тихом месте."
        )
        await state.clear()
        return

    structured = await _structure_transcript(session, transcript)

    report = VoiceReportLog(
        report_id=f"r-{datetime.utcnow().timestamp()}",
        telegram_user_id=message.from_user.id,
        project_id=session.project_id,
        raw_audio_path=audio_path,
        raw_transcript=transcript,
        structured_data=structured,
    )
    pending_report[message.from_user.id] = report

    draft_text = structured.get("summary_line") or transcript
    needs_clarification = structured.get("needs_clarification") or []
    clarification_note = ""
    if needs_clarification:
        clarification_note = "\n\n⚠ Требует уточнения: " + "; ".join(
            str(item.get("reason", "")) for item in needs_clarification
        )

    await message.answer(
        f"📋 Черновик отчёта:\n{draft_text}{clarification_note}\n\nВсё верно?",
        reply_markup=_confirm_keyboard(),
    )
    await state.set_state(ReportFlow.waiting_confirmation)


@router.message(ReportFlow.waiting_voice, F.voice)
async def handle_voice_recording(message: Message, state: FSMContext, bot: Bot) -> None:
    audio_path = await download_voice(bot, message)
    await process_report(bot, message, state, audio_path)


@router.message(F.voice)
async def handle_voice_default(message: Message, state: FSMContext, bot: Bot) -> None:
    current_state = await state.get_state()
    if current_state == ReportFlow.waiting_correction.state:
        return  # обработается отдельным хендлером

    session = await resolve_foreman(message.from_user.id)
    if session is None:
        await message.answer("Сначала зарегистрируйтесь: /start <код приглашения>")
        return

    audio_path = await download_voice(bot, message)
    await process_report(bot, message, state, audio_path)


@router.message(ReportFlow.waiting_voice, F.photo)
@router.message(F.photo)
async def handle_photo(message: Message) -> None:
    report = pending_report.get(message.from_user.id)
    if report is None:
        await message.answer(
            "Фото получено, но нет активного отчёта — сначала отправьте голосовое."
        )
        return
    # TODO: сохранить file_id/URL фото и передать в POST /journal (поле photos).
    await message.answer("📷 Фото прикреплено к текущему черновику отчёта.")


# --------------------------------------------------------------------------
# Подтверждение / исправление / отмена
# --------------------------------------------------------------------------


@router.callback_query(F.data == "confirm_report")
async def on_confirm(callback: CallbackQuery, state: FSMContext) -> None:
    report = pending_report.get(callback.from_user.id)
    if report is None or not report.structured_data:
        await callback.answer("Черновик не найден.")
        return

    session = foreman_cache.get(callback.from_user.id)
    entries = report.structured_data.get("entries") or []
    if session is None:
        await callback.answer("Сессия истекла, отправьте /start.")
        return

    try:
        result = await ie_aion_client.submit_journal_entries(
            project_id=session.project_id,
            entries=entries,
            author_foreman_id=session.foreman_id,
            raw_quote=report.raw_transcript,
            report_date=date.today().isoformat(),
        )
    except IeAionBackendError:
        # НЕ удаляем черновик — пользователь сможет повторить подтверждение.
        await callback.message.answer(
            "Не удалось сохранить отчёт, попробуйте подтвердить ещё раз."
        )
        await callback.answer()
        return

    report.confirmed = True
    applied = result.get("applied_count", len(entries))
    clarify = result.get("clarification_count", 0)
    tail = f" ({clarify} в очереди верификации ПТО)" if clarify else ""
    await callback.message.edit_text(
        f"✅ Отчёт записан в суточный журнал: {applied} записей{tail}."
    )
    await callback.answer()
    await state.clear()


@router.callback_query(F.data == "fix_report")
async def on_fix(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(ReportFlow.waiting_correction)
    await callback.message.answer("Что поправить? Скажите голосом или напишите текстом.")
    await callback.answer()


@router.callback_query(F.data == "cancel_report")
async def on_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    pending_report.pop(callback.from_user.id, None)
    await callback.message.edit_text("🗑 Черновик отменён.")
    await callback.answer()
    await state.clear()


async def _reprocess_correction(message: Message, state: FSMContext, correction_text: str) -> None:
    report = pending_report.get(message.from_user.id)
    if report is None:
        await message.answer("Активный черновик не найден, начните заново: /record")
        await state.clear()
        return

    session = await resolve_foreman(message.from_user.id)
    if session is None:
        await message.answer("Сначала зарегистрируйтесь: /start <код приглашения>")
        await state.clear()
        return

    merged_transcript = f"{report.raw_transcript}\n[Исправление] {correction_text}"
    structured = await _structure_transcript(session, merged_transcript)
    report.raw_transcript = merged_transcript
    report.structured_data = structured

    await message.answer(
        f"📋 Обновлённый черновик:\n{structured.get('summary_line') or correction_text}\n\nВсё верно?",
        reply_markup=_confirm_keyboard(),
    )
    await state.set_state(ReportFlow.waiting_confirmation)


@router.message(ReportFlow.waiting_correction, F.voice)
async def handle_correction_voice(message: Message, state: FSMContext, bot: Bot) -> None:
    audio_path = await download_voice(bot, message)
    correction_text = await asr_transcribe(audio_path, language_hint="mixed")
    if not correction_text.strip():
        await message.answer("Не удалось распознать исправление, попробуйте ещё раз.")
        return
    await _reprocess_correction(message, state, correction_text)


@router.message(ReportFlow.waiting_correction, F.text)
async def handle_correction_text(message: Message, state: FSMContext) -> None:
    await _reprocess_correction(message, state, message.text)


# --------------------------------------------------------------------------
# Точка входа
# --------------------------------------------------------------------------


async def main() -> None:
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    logger.info("Бот запущен (long polling)")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
