"""Emotion Detection From Text using Hugging Face Transformers and Telegram.

Reconstructed and formatted from the source-code section of the supplied PDF.
For safety, the Telegram bot token is read from an environment variable instead
of being written directly in this file.

Set TELEGRAM_BOT_TOKEN before running this program.
"""

import logging
import os

from telegram import Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    CallbackContext,
)
from transformers import pipeline

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Load the pretrained emotion-detection model
emotion_classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
)


def start(update: Update, context: CallbackContext) -> None:
    """Send a welcome message when the user starts the bot."""
    update.message.reply_text(
        "Hi! Send me a message and I'll tell you what emotion I detect in it."
    )


def detect_emotion(update: Update, context: CallbackContext) -> None:
    """Predict the emotion in the received text and reply to the user."""
    user_text = update.message.text
    result = emotion_classifier(user_text)[0]

    label = result["label"]
    score = result["score"]
    response = f"Emotion: {label} (confidence: {score:.2f})"

    update.message.reply_text(response)


def main() -> None:
    """Start the Telegram bot."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set. Set your Telegram bot token "
            "as an environment variable before running this program."
        )

    updater = Updater(bot_token, use_context=True)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(
        MessageHandler(Filters.text & ~Filters.command, detect_emotion)
    )

    updater.start_polling()
    logger.info("Emotion detection Telegram bot is running.")
    updater.idle()


if __name__ == "__main__":
    main()
