import logging
from telegram import Bot
import asyncio
import os

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
CHAT_ID = os.environ['CHAT_ID']

async def send_text_async(message: str):
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        async with bot:
            await bot.send_message(chat_id=CHAT_ID, text=message)
    except Exception as e:
        logger.error(f"Error sending message: {e}")

async def send_image_async(image_path: str, text: str):
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        with open(image_path, 'rb') as image_file:
            async with bot:
                await bot.send_photo(chat_id=CHAT_ID, photo=image_file, caption=text)

    except Exception as e:
        logger.error(f"Error sending image: {e}")


def send_text(message: str):
    asyncio.run(send_text_async(message))

def send_image(image_path: str, text: str=None):
    asyncio.run(send_image_async(image_path, text))


