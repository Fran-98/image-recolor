import logging
from telegram import Bot
import asyncio
import os

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
CHAT_ID = os.environ['CHAT_ID']

async def send_text_async(message: str):
    try:
        async with Bot(token=TELEGRAM_BOT_TOKEN) as bot:
            await bot.send_message(chat_id=CHAT_ID, text=message)
    except Exception as e:
        logger.error(f"Error sending message: {e}")

async def send_image_async(image_path: str, text: str):
    try:
        async with Bot(token=TELEGRAM_BOT_TOKEN) as bot:
            with open(image_path, 'rb') as image_file:
                await bot.send_photo(chat_id=CHAT_ID, photo=image_file, caption=text)
    except Exception as e:
        logger.error(f"Error sending image: {e}")


def get_or_create_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

def send_text(message: str):
    loop = get_or_create_event_loop()
    if loop.is_running():
        loop.create_task(send_text_async(message))
    else:
        loop.run_until_complete(send_text_async(message))

def send_image(image_path: str, text: str=None):
    loop = get_or_create_event_loop()
    if loop.is_running():
        loop.create_task(send_image_async(image_path, text))
    else:
        loop.run_until_complete(send_image_async(image_path, text))

