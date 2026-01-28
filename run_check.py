import asyncio
import numpy as np
import cv2
import pytesseract
import re
import requests
import json
import os
from datetime import datetime, date
from io import BytesIO

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from aiogram import Bot
from aiogram.types import BufferedInputFile

# --- НАЛАШТУВАННЯ ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROUP_ID = os.getenv("GROUP_ID") 
SITE_URL = "https://voe-poweron.inneti.net/schedule_queues"

# Налаштування кольору (синій графік)
LOWER_BLUE = np.array([80, 60, 40])
UPPER_BLUE = np.array([255, 180, 120])

# --- ВАЖЛИВО: ІНДЕКС ЧЕРГИ ---
# 0->1.1, 1->1.2, 2->2.1, 3->2.2, 4->3.1, 5->3.2, 6->4.1, 7->4.2
TARGET_QUEUE_INDEX = 6 
STATE_FILE = "state.json"

# --- ФУНКЦІЇ ---

def get_image_links_headless():
    """Запускає Chrome (Headless) і шукає картинки."""
    print("🚀 Selenium: Start...")
    chrome_options = Options()
    chrome_options.add_argument("--headless=new") 
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    found_urls = []
    try:
        driver.get(SITE_URL)
        # Чекаємо поки завантажиться хоча б одна картинка
        try:
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "img")))
        except: pass
        
        images = driver.find_elements(By.TAG_NAME, "img")
        for img in images:
            src = img.get_attribute("src")
            # Фільтруємо лінки, шукаємо схожі на графіки
            if src and (("GPV" in src) or ("media" in src and ("png" in src or "jpg" in src))):
                 found_urls.append(src)
    except Exception as e:
        print(f"Selenium Error: {e}")
    finally:
        driver.quit()
    return list(set(found_urls))

def parse_date_only(img):
    """Витягує дату з заголовка картинки через OCR."""
    try:
        h, w, _ = img.shape
        header_crop = img[0:int(h*0.15), 0:int(w*0.50)]
        gray = cv2.cvtColor(header_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Використовуємо tesseract. На GitHub Actions він встановлюється в систему.
        text = pytesseract.image_to_string(gray, lang='ukr+eng')
        dm = re.findall(r'(\d{2})\.(\d{2})\.(\d{4})', text)
        if dm:
            return datetime.strptime(f"{dm[0][0]}.{dm[0][1]}.{dm[0][2]}", "%d.%m.%Y").date()
    except: pass
    return None

def analyze_schedule_image(img):
    """
    Аналізує графік, використовуючи HSV для точнішого визначення кольору.
    """
    height, width, _ = img.shape
    debug_img = img.copy()
    
    # 1. Конвертуємо зображення в HSV (це значно краще для розпізнавання кольорів)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Визначаємо діапазон синього в HSV
    # Hue (відтінок): 100-140 (синій колір в OpenCV це ~120)
    # Saturation (насиченість): > 50 (щоб ігнорувати білий/сірий)
    # Value (яскравість): > 50 (щоб ігнорувати чорний)
    lower_blue_hsv = np.array([90, 50, 50])
    upper_blue_hsv = np.array([130, 255, 255])
    
    # Створюємо маску: 255 там де синій, 0 де ні
    mask = cv2.inRange(hsv_img, lower_blue_hsv, upper_blue_hsv)

    rows_total = 12
    # Координати блоків (залишаємо як було, вони виглядають правильними)
    top_y_start = int(height * 0.19)
    top_y_end = int(height * 0.51)
    bottom_y_start = int(height * 0.58)
    bottom_y_end = int(height * 0.90)
    
    outage_intervals = []

    def scan_block(y_start, y_end, hour_offset):
        block_h = y_end - y_start
        row_h = block_h / rows_total
        
        # Центр рядка нашої черги
        y_center = int(y_start + (TARGET_QUEUE_INDEX * row_h) + (row_h / 2))
        
        # Малюємо лінію сканування (зелена)
        cv2.line(debug_img, (0, y_center), (width, y_center), (0, 255, 0), 2)
        
        # Трохи зсуваємо початок вправо (0.10 замість 0.095), щоб не зачепити рамку
        x_start = int(width * 0.10) 
        x_end = width
        col_w = (x_end - x_start) / 24
        
        current_start = None
        for i in range(24):
            x_center = int(x_start + (i * col_w) + (col_w / 2))
            
            # Малюємо точки перевірки
            cv2.circle(debug_img, (x_center, y_center), 3, (0, 0, 255), -1)
            
            if y_center < height and x_center < width:
                # Перевіряємо маску замість пікселів
                # Якщо mask[y, x] > 0, значить це синій колір
                is_blue = mask[y_center, x_center] > 0
                
                time_val = hour_offset + (i * 0.5)
                
                if is_blue:
                    if current_start is None: current_start = time_val
                else:
                    if current_start is not None:
                        outage_intervals.append((current_start, time_val))
                        current_start = None
                        
        if current_start is not None: 
            outage_intervals.append((current_start, hour_offset + 12))

    scan_block(top_y_start, top_y_end, 0)
    scan_block(bottom_y_start, bottom_y_end, 12)
    
    return outage_intervals, debug_img

def format_intervals_to_string(intervals):
    """Створює унікальний рядок-підпис для порівняння змін."""
    if not intervals: return "CLEAR"
    res = []
    for start, end in intervals:
        res.append(f"{int(start):02}:00-{int(end):02}:00")
    return "|".join(res)

def format_intervals_pretty(intervals):
    """Форматує текст для Telegram."""
    if not intervals: return "✅ Світло є (графік білий)."
    text = ""
    for start, end in intervals:
        end_str = f"{int(end):02}:00" if end != 24 else "24:00"
        text += f"⚫ `{int(start):02}:00 - {end_str}`\n"
    return text

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            try: return json.load(f)
            except: return {}
    return {}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

# --- ГОЛОВНА ЛОГІКА ---

async def main():
    if not BOT_TOKEN:
        print("❌ Немає токена. Перевірте Secrets в GitHub.")
        return

    bot = Bot(token=BOT_TOKEN)
    
    # 1. Отримуємо посилання
    urls = await asyncio.to_thread(get_image_links_headless)
    
    if not urls:
        print("❌ Selenium не знайшов картинок.")
        await bot.session.close()
        return

    history = load_state()
    something_sent = False

    print(f"🔍 Знайдено URL: {len(urls)}")

    for url in urls:
        try:
            # Завантажуємо картинку
            resp = requests.get(url, timeout=20)
            img_arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img is None: continue

            # 2. Визначаємо дату
            sched_date = parse_date_only(img)
            if not sched_date: 
                print(f"⚠️ Дата не прочиталась: {url}")
                continue
            date_str = sched_date.strftime("%d.%m.%Y")

            # 3. Аналізуємо графік
            intervals, debug_img = await asyncio.to_thread(analyze_schedule_image, img)
            
            # 4. Перевіряємо зміни
            current_signature = format_intervals_to_string(intervals)
            last_saved_signature = history.get(date_str)

            if last_saved_signature == current_signature:
                print(f"💤 {date_str}: Без змін.")
                continue
            
            # Якщо є зміни
            if last_saved_signature:
                print(f"🔥 ЗМІНИ! {date_str}")
                status_text = "🔄 **ЗМІНИ В ГРАФІКУ! (Черга 4.1)**"
            else:
                print(f"✅ Новий графік: {date_str}")
                status_text = "⚡️ **Новий графік (Черга 4.1)**"

            text_schedule = format_intervals_pretty(intervals)
            
            caption = (
                f"{status_text}\n"
                f"📅 Дата: **{date_str}**\n\n"
                f"{text_schedule}"
            )

            # Відправляємо фото в Telegram
            is_success, buffer = cv2.imencode(".png", debug_img)
            if is_success:
                io_buf = BytesIO(buffer)
                await bot.send_photo(
                    chat_id=GROUP_ID,
                    photo=BufferedInputFile(io_buf.getvalue(), filename="schedule.png"),
                    caption=caption,
                    parse_mode="Markdown"
                )
                
                history[date_str] = current_signature
                something_sent = True

        except Exception as e:
            print(f"Error on URL {url}: {e}")

    # Зберігаємо стан, щоб GitHub Actions зафіксував зміни
    if something_sent: 
        save_state(history)
        
    await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())
