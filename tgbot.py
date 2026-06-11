```text
model_exp_3.pth
```

Токен бота лучше хранить не в коде, а в переменной окружения `TELEGRAM_BOT_TOKEN`.

## 1. Установка зависимостей
"""

# Выполнять один раз, если пакеты ещё не установлены
# Лучше запускать в conda-окружении Python 3.10

# !pip install python-telegram-bot==20.7 nest-asyncio
# !pip install torch torchvision diffusers transformers accelerate safetensors sentencepiece pillow tqdm

"""## 2. Импорты"""

import os
import math
import logging
import asyncio
import textwrap
import tempfile
from pathlib import Path

import nest_asyncio
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from diffusers import AltDiffusionPipeline
from telegram import Update, ReplyKeyboardRemove
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ConversationHandler,
    ContextTypes,
)

"""## 3. Конфигурация"""

MODEL_PATH = "AltDiffusion-m9"       # локальная папка или HF id
WEIGHTS_PATH = "/bot/pzad/model_exp_3.pth"     # файл с обученными весами
#WEIGHTS_PATH = "/Users/alfa/Desktop/proj_pzad/pzad/model_exp_3.pth"
IMAGE_SIZE = 512
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
LORA_RANK = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print("DEVICE:", DEVICE)
print("DTYPE:", DTYPE)

"""## 4. LoRA-слои, как в модельном ноутбуке"""

class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=8, alpha=8.0):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling


def apply_lora_to_unet(unet, r=8, alpha=8.0, device="cpu"):
    replaced = []

    for module_name, module in unet.named_modules():
        for attr in ["to_q", "to_k", "to_v"]:
            if hasattr(module, attr):
                child = getattr(module, attr)
                if isinstance(child, nn.Linear):
                    setattr(module, attr, LoRALinear(child, r=r, alpha=alpha).to(device))
                    replaced.append(f"{module_name}.{attr}")

        if hasattr(module, "to_out"):
            to_out = getattr(module, "to_out")
            if isinstance(to_out, (nn.ModuleList, nn.Sequential)) and len(to_out) > 0:
                if isinstance(to_out[0], nn.Linear):
                    to_out[0] = LoRALinear(to_out[0], r=r, alpha=alpha).to(device)
                    replaced.append(f"{module_name}.to_out.0")

    return replaced

"""## 5. Загрузка модели и весов"""

import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=8, alpha=8.0):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
pipe = pipe.to(device)
pipe.unet.eval()
pipe.vae.eval()
pipe.text_encoder.eval()



def strip_prefix_if_present(state_dict, prefix):
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}



"""## 6. Проверка генерации"""

@torch.inference_mode()
def generate_image(prompt: str) -> Image.Image:
    result = pipe(
        prompt=prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
    )
    return result.images[0]

# Быстрый тест
# test_img = generate_image("Кот сидит за ноутбуком и пишет диплом")
# test_img.save("test_generation.png")
# test_img

"""## 7. Добавление текста на картинку"""

def get_font(font_size=32):
    candidates = [
        "DejaVuSans.ttf",
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]

    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:
            pass

    return ImageFont.load_default()


def render_meme(image, caption):
    if not caption:
        return image

    image = image.convert("RGB")

    width = image.width

    padding = 20

    max_lines = 3

    font_size = 32

    font = get_font(font_size)

    wrapped_lines = textwrap.wrap(caption, width=25)
    wrapped_lines = wrapped_lines[:max_lines]

    dummy_img = Image.new("RGB", (width, 100))
    draw = ImageDraw.Draw(dummy_img)

    line_heights = []
    for line in wrapped_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_heights.append(bbox[3] - bbox[1])

    text_height = sum(line_heights) + padding * 2 + 10 * max(0, len(wrapped_lines) - 1)

    new_image = Image.new("RGB", (width, image.height + text_height), "white")
    new_image.paste(image, (0, 0))

    draw = ImageDraw.Draw(new_image)
    y_text = image.height + padding

    for line in wrapped_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x_text = (width - text_width) // 2
        draw.text((x_text, y_text), line, fill="black", font=font)
        y_text += (bbox[3] - bbox[1]) + 10

    return new_image

"""## 8. Telegram-бот"""

nest_asyncio.apply()

# Безопаснее задать токен через переменную окружения:
# Windows PowerShell:
#   $env:TELEGRAM_BOT_TOKEN="сюда_токен"
#
# Или временно можно вписать строкой:
# TOKEN = "сюда_токен"

TOKEN = '8718579709:AAGjBJacPxzMcxIqMZFbLg8np3cgS2k0E0c'

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

WAITING_FOR_DESCRIPTION = 1
WAITING_FOR_OVERLAY_TEXT = 2

user_memes = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Привет! Я бот-мемогенератор*\n\n"
        "📋 *Доступные команды:*\n"
        "/creatememe - создать новый мем\n"
        "/help - список команд\n"
        "/cancel - отменить создание\n"
        "/status - статус генерации\n\n"
        "🎮 *Погнали!*",
        parse_mode="Markdown",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *Справочная информация*\n\n"
        "🔹 */creatememe* - запустить создание мема\n"
        "🔹 */cancel* - отменить текущее действие\n"
        "🔹 */status* - проверить статус бота\n"
        "🔹 */start* - приветствие\n\n"
        "*Как создать мем:*\n"
        "1️⃣ Введи /creatememe\n"
        "2️⃣ Опиши изображение\n"
        "3️⃣ Напиши текст для подписи или '-' без текста\n"
        "4️⃣ Бот отправит результат\n",
        parse_mode="Markdown",
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"✅ *Бот работает*\n"
        f"🧠 device: `{DEVICE}`\n"
        f"🐍 weights: `{WEIGHTS_PATH}`",
        parse_mode="Markdown",
    )


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_memes.pop(update.effective_user.id, None)

    await update.message.reply_text(
        "❌ *Действие отменено*\n\n"
        "Чтобы создать новый мем — введи /creatememe",
        parse_mode="Markdown",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ConversationHandler.END


async def creatememe_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎨 *Шаг 1 из 2: опиши изображение*\n\n"
        "✏️ *Примеры:*\n"
        "• «кот грустит из-за дождя»\n"
        "• «человек-паук опаздывает на работу»\n"
        "• «собака пытается понять физику»\n\n"
        "💬 *Напиши своё описание:*\n"
        "(Чтобы отменить — введи /cancel)",
        parse_mode="Markdown",
    )
    return WAITING_FOR_DESCRIPTION


async def process_meme_description(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    description = update.message.text.strip()

    user_memes[user_id] = {"description": description}

    await update.message.reply_text(
        "✅ *Описание принято!*\n\n"
        "🎨 *Шаг 2 из 2: какой текст добавить на мем?*\n\n"
        "Напиши текст или отправь `-`, если текст не нужен.\n"
        "(Чтобы отменить — введи /cancel)",
        parse_mode="Markdown",
    )
    return WAITING_FOR_OVERLAY_TEXT


async def process_overlay_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    overlay_text = update.message.text.strip()

    if user_id not in user_memes:
        await update.message.reply_text(
            "❌ *Что-то пошло не так*\n"
            "Начни заново с /creatememe",
            parse_mode="Markdown",
        )
        return ConversationHandler.END

    description = user_memes[user_id]["description"]

    if overlay_text == "-":
        overlay_text = None
        text_info = "без текста"
    else:
        text_info = f"«{overlay_text}»"

    thinking_msg = await update.message.reply_text(
        "🎨 *Генерирую мем...*\n\n"
        "⏳ Это может занять некоторое время.",
        parse_mode="Markdown",
    )

    final_image_path = None

    try:
        generated_img = generate_image(description)

        if overlay_text:
            generated_img = render_meme(generated_img, overlay_text)

        tmp = tempfile.NamedTemporaryFile(
            suffix=f"_meme_{user_id}.png",
            delete=False,
        )
        final_image_path = tmp.name
        tmp.close()

        generated_img.save(final_image_path)

    except Exception as e:
        logging.exception("Ошибка генерации")
        await thinking_msg.delete()
        await update.message.reply_text(
            f"❌ *Ошибка генерации*\n\n"
            f"`{type(e).__name__}: {e}`\n\n"
            f"Попробуй ещё раз: /creatememe",
            parse_mode="Markdown",
        )
        user_memes.pop(user_id, None)
        return ConversationHandler.END

    await thinking_msg.delete()

    with open(final_image_path, "rb") as photo:
        await update.message.reply_photo(
            photo=photo,
            caption=(
                "✅ *Мем сгенерирован!*\n\n"
                f"📝 *Описание:* «{description}»\n"
                f"💬 *Текст:* {text_info}\n\n"
                "✨ Хочешь ещё? Введи /creatememe"
            ),
            parse_mode="Markdown",
        )

    if final_image_path and os.path.exists(final_image_path):
        os.remove(final_image_path)

    user_memes.pop(user_id, None)
    return ConversationHandler.END


async def handle_random_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "❓ *Я тебя не понял*\n\n"
        "🎨 /creatememe - создать мем\n"
        "📖 /help - все команды\n"
        "❌ /cancel - отменить действие",
        parse_mode="Markdown",
    )


async def post_init(application: Application):
    print("✅ Бот успешно запущен.")
    print("💡 Команды: /start, /creatememe, /help, /cancel, /status")


def main():
    app = Application.builder().token(TOKEN).post_init(post_init).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("creatememe", creatememe_start)],
        states={
            WAITING_FOR_DESCRIPTION: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_meme_description)
            ],
            WAITING_FOR_OVERLAY_TEXT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_overlay_text)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        name="meme_creation",
        persistent=False,
    )

    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_random_text))

    app.run_polling()

if __name__ == "__main__":
    main()
