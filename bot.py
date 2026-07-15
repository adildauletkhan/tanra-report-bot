"""
Tanra AI Report Bot — сбор голосовых отчётов бригадиров.

Команды (зарегистрированы в BotFather):
    /start   — начать, привязать пользователя к объекту
    /object  — выбрать/сменить объект
    /status  — показать последний отчёт по пользователю
    /record  — явно начать запись отчёта (голос + опционально фото)

Требует: pip install aiogram aiohttp
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from asr_yandex import ensure_config as ensure_yandex_config
from asr_yandex import transcribe as asr_transcribe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
# Проверяем переменные Yandex SpeechKit на старте (аналогично TELEGRAM_BOT_TOKEN).
ensure_yandex_config()

# --------------------------------------------------------------------------
# "База" объектов — в проде это должен быть запрос к backend API платформы
# (см. construction-overview.md, fetchProject / getCurrentProjectId).
# --------------------------------------------------------------------------
PROJECTS = {
    "proj-highvill-astana": "ЖК «Highvill Astana» · 2-я очередь",
    "proj-alatau-a": "ЖК «Алатау А»",
}

# --------------------------------------------------------------------------
# In-memory "БД" пользователей и отчётов для примера.
# В проде — таблицы TelegramUser / VoiceReportLog (см. telegram-bot-tech-spec.md).
# --------------------------------------------------------------------------


@dataclass
class UserProfile:
    telegram_user_id: int
    default_project_id: str | None = None
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


users: dict[int, UserProfile] = {}
reports: dict[int, list[VoiceReportLog]] = {}
pending_report: dict[int, VoiceReportLog] = {}  # черновик, ждущий подтверждения


def get_or_create_user(telegram_user_id: int) -> UserProfile:
    if telegram_user_id not in users:
        users[telegram_user_id] = UserProfile(telegram_user_id=telegram_user_id)
    return users[telegram_user_id]


# --------------------------------------------------------------------------
# FSM-состояния
# --------------------------------------------------------------------------


class ReportFlow(StatesGroup):
    waiting_object = State()      # /object — ожидаем выбор объекта
    waiting_voice = State()       # /record — ожидаем голосовое (+фото)
    waiting_confirmation = State()  # ожидаем ✅/✏️ по черновику
    waiting_correction = State()  # ожидаем уточняющее голосовое/текст


router = Router()


# --------------------------------------------------------------------------
# /start
# --------------------------------------------------------------------------


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    user = get_or_create_user(message.from_user.id)
    if user.default_project_id is None:
        await message.answer(
            "Здравствуйте! Я собираю ежедневные отчёты по объектам.\n"
            "Для начала выберите объект командой /object."
        )
    else:
        project_name = PROJECTS.get(user.default_project_id, user.default_project_id)
        await message.answer(
            f"С возвращением! Текущий объект: {project_name}\n\n"
            "Присылайте голосовые отчёты — я сам разберу зону, работу и статус.\n"
            "Можно приложить фото захватки к тому же сообщению.\n\n"
            "Команды: /object — сменить объект, /status — последний отчёт, "
            "/record — начать запись отчёта."
        )
    await state.clear()


# --------------------------------------------------------------------------
# /object — выбор/смена объекта
# --------------------------------------------------------------------------


def build_projects_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(text=name, callback_data=f"set_object:{pid}")]
        for pid, name in PROJECTS.items()
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


@router.message(Command("object"))
async def cmd_object(message: Message, state: FSMContext) -> None:
    await state.set_state(ReportFlow.waiting_object)
    await message.answer(
        "На каком объекте вы работаете?", reply_markup=build_projects_keyboard()
    )


@router.callback_query(F.data.startswith("set_object:"))
async def on_object_selected(callback: CallbackQuery, state: FSMContext) -> None:
    project_id = callback.data.split(":", 1)[1]
    user = get_or_create_user(callback.from_user.id)
    user.default_project_id = project_id
    user.default_zone = None  # сбрасываем зону при смене объекта

    project_name = PROJECTS.get(project_id, project_id)
    await callback.message.edit_text(f"Объект выбран: {project_name}")
    await callback.answer()
    await state.clear()


# --------------------------------------------------------------------------
# /status — последний отчёт пользователя
# --------------------------------------------------------------------------


@router.message(Command("status"))
async def cmd_status(message: Message) -> None:
    user_reports = reports.get(message.from_user.id, [])
    if not user_reports:
        await message.answer(
            "Пока нет ни одного отчёта. Отправьте голосовое сообщение или "
            "используйте /record."
        )
        return

    last = user_reports[-1]
    status_label = "✅ подтверждён" if last.confirmed else "⏳ ожидает подтверждения"
    project_name = PROJECTS.get(last.project_id, last.project_id or "не указан")

    summary = last.structured_data.get("summary_line") if last.structured_data else last.raw_transcript
    await message.answer(
        f"Последний отчёт ({last.created_at:%d.%m.%Y %H:%M})\n"
        f"Объект: {project_name}\n"
        f"Статус: {status_label}\n\n"
        f"{summary or 'обрабатывается...'}"
    )


# --------------------------------------------------------------------------
# /record — явный старт записи отчёта
# --------------------------------------------------------------------------


@router.message(Command("record"))
async def cmd_record(message: Message, state: FSMContext) -> None:
    user = get_or_create_user(message.from_user.id)
    if user.default_project_id is None:
        await message.answer("Сначала выберите объект: /object")
        return

    await state.set_state(ReportFlow.waiting_voice)
    await message.answer(
        "🎤 Записываю. Отправьте голосовое сообщение с отчётом "
        "(можно приложить фото захватки)."
    )


# --------------------------------------------------------------------------
# Приём голосового сообщения — работает и по /record, и без явной команды,
# если пользователь уже выбрал объект (см. handle_voice_default ниже)
# --------------------------------------------------------------------------


async def download_voice(bot: Bot, message: Message) -> str:
    file = await bot.get_file(message.voice.file_id)
    local_path = f"/tmp/{message.voice.file_id}.ogg"
    await bot.download_file(file.file_path, local_path)
    return local_path


async def process_report(
    bot: Bot, message: Message, state: FSMContext, audio_path: str
) -> None:
    """
    Полный конвейер: ASR -> LLM-структурирование по промпту
    voice-report-system-prompt.md -> черновик на подтверждение.
    Заглушки asr_transcribe / llm_structure нужно заменить на реальные вызовы.
    """
    user = get_or_create_user(message.from_user.id)

    await message.answer("⏳ Обрабатываю...")

    transcript = await asr_transcribe(audio_path, language_hint="mixed")
    if not transcript.strip():
        # Тишина/шум — ASR вернул пустую строку. Не вызываем llm_structure.
        await message.answer(
            "Не удалось распознать речь, попробуйте ещё раз в тихом месте."
        )
        await state.clear()
        return

    structured = await llm_structure(
        transcript=transcript,
        project_id=user.default_project_id,
        default_zone_hint=user.default_zone,
    )

    report = VoiceReportLog(
        report_id=f"r-{datetime.utcnow().timestamp()}",
        telegram_user_id=message.from_user.id,
        project_id=user.default_project_id,
        raw_audio_path=audio_path,
        raw_transcript=transcript,
        structured_data=structured,
    )
    pending_report[message.from_user.id] = report

    draft_text = structured.get("summary_line", transcript)
    needs_clarification = structured.get("needs_clarification") or []

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Подтвердить", callback_data="confirm_report"),
                InlineKeyboardButton(text="✏️ Исправить", callback_data="fix_report"),
                InlineKeyboardButton(text="🗑 Отменить", callback_data="cancel_report"),
            ]
        ]
    )

    clarification_note = ""
    if needs_clarification:
        clarification_note = "\n\n⚠ Требует уточнения: " + "; ".join(
            item["reason"] for item in needs_clarification
        )

    await message.answer(
        f"📋 Черновик отчёта:\n{draft_text}{clarification_note}\n\nВсё верно?",
        reply_markup=keyboard,
    )
    await state.set_state(ReportFlow.waiting_confirmation)


@router.message(ReportFlow.waiting_voice, F.voice)
async def handle_voice_recording(
    message: Message, state: FSMContext, bot: Bot
) -> None:
    audio_path = await download_voice(bot, message)
    await process_report(bot, message, state, audio_path)


@router.message(F.voice)
async def handle_voice_default(message: Message, state: FSMContext, bot: Bot) -> None:
    """Голосовое без явной команды /record — тоже принимаем, если объект выбран."""
    current_state = await state.get_state()
    if current_state == ReportFlow.waiting_correction.state:
        return  # обработается отдельным хендлером ниже

    user = get_or_create_user(message.from_user.id)
    if user.default_project_id is None:
        await message.answer("Сначала выберите объект: /object")
        return

    audio_path = await download_voice(bot, message)
    await process_report(bot, message, state, audio_path)


@router.message(ReportFlow.waiting_voice, F.photo)
@router.message(F.photo)
async def handle_photo(message: Message) -> None:
    """Фото прикрепляется к последнему черновику отчёта пользователя, если есть."""
    report = pending_report.get(message.from_user.id)
    if report is None:
        await message.answer(
            "Фото получено, но нет активного отчёта — сначала отправьте голосовое."
        )
        return
    # В проде: сохранить file_id/URL фото в report.structured_data["photos"]
    await message.answer("📷 Фото прикреплено к текущему черновику отчёта.")


# --------------------------------------------------------------------------
# Подтверждение / исправление / отмена черновика
# --------------------------------------------------------------------------


@router.callback_query(F.data == "confirm_report")
async def on_confirm(callback: CallbackQuery, state: FSMContext) -> None:
    report = pending_report.pop(callback.from_user.id, None)
    if report is None:
        await callback.answer("Черновик не найден.")
        return

    report.confirmed = True
    reports.setdefault(callback.from_user.id, []).append(report)

    # TODO: вызвать backend API платформы, например:
    # await apply_report_to_wbs(report.structured_data)

    await callback.message.edit_text("✅ Отчёт подтверждён и записан.")
    await callback.answer()
    await state.clear()


@router.callback_query(F.data == "fix_report")
async def on_fix(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(ReportFlow.waiting_correction)
    await callback.message.answer(
        "Что поправить? Скажите голосом или напишите текстом."
    )
    await callback.answer()


@router.callback_query(F.data == "cancel_report")
async def on_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    pending_report.pop(callback.from_user.id, None)
    await callback.message.edit_text("🗑 Черновик отменён.")
    await callback.answer()
    await state.clear()


@router.message(ReportFlow.waiting_correction, F.voice)
async def handle_correction_voice(
    message: Message, state: FSMContext, bot: Bot
) -> None:
    audio_path = await download_voice(bot, message)
    report = pending_report.get(message.from_user.id)
    if report is None:
        await message.answer("Активный черновик не найден, начните заново: /record")
        await state.clear()
        return

    correction_text = await asr_transcribe(audio_path, language_hint="mixed")
    merged_transcript = f"{report.raw_transcript}\n[Исправление] {correction_text}"

    user = get_or_create_user(message.from_user.id)
    structured = await llm_structure(
        transcript=merged_transcript,
        project_id=user.default_project_id,
        default_zone_hint=user.default_zone,
    )
    report.raw_transcript = merged_transcript
    report.structured_data = structured

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Подтвердить", callback_data="confirm_report"),
                InlineKeyboardButton(text="✏️ Исправить", callback_data="fix_report"),
                InlineKeyboardButton(text="🗑 Отменить", callback_data="cancel_report"),
            ]
        ]
    )
    await message.answer(
        f"📋 Обновлённый черновик:\n{structured.get('summary_line', correction_text)}\n\nВсё верно?",
        reply_markup=keyboard,
    )
    await state.set_state(ReportFlow.waiting_confirmation)


@router.message(ReportFlow.waiting_correction, F.text)
async def handle_correction_text(message: Message, state: FSMContext) -> None:
    report = pending_report.get(message.from_user.id)
    if report is None:
        await message.answer("Активный черновик не найден, начните заново: /record")
        await state.clear()
        return

    merged_transcript = f"{report.raw_transcript}\n[Исправление] {message.text}"
    user = get_or_create_user(message.from_user.id)
    structured = await llm_structure(
        transcript=merged_transcript,
        project_id=user.default_project_id,
        default_zone_hint=user.default_zone,
    )
    report.raw_transcript = merged_transcript
    report.structured_data = structured

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Подтвердить", callback_data="confirm_report"),
                InlineKeyboardButton(text="✏️ Исправить", callback_data="fix_report"),
                InlineKeyboardButton(text="🗑 Отменить", callback_data="cancel_report"),
            ]
        ]
    )
    await message.answer(
        f"📋 Обновлённый черновик:\n{structured.get('summary_line', message.text)}\n\nВсё верно?",
        reply_markup=keyboard,
    )
    await state.set_state(ReportFlow.waiting_confirmation)


# --------------------------------------------------------------------------
# Заглушки внешних сервисов — заменить на реальную интеграцию
# (asr_transcribe теперь реализован в asr_yandex.py и импортирован выше)
# --------------------------------------------------------------------------


async def llm_structure(
    transcript: str, project_id: str | None, default_zone_hint: str | None
) -> dict:
    """
    TODO: заменить на реальный вызов LLM с системным промптом из
    voice-report-system-prompt.md. Должен вернуть structured_data
    (см. схему в §5.1 промпта), сюда добавлено поле summary_line
    для удобного вывода в чат.
    """
    raise NotImplementedError("Подключите LLM-структурирование (см. voice-report-system-prompt.md)")


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
