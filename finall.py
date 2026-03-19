import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
import nest_asyncio
import asyncio
import os
import re
import logging
import json
import base64
from datetime import datetime
import requests
from paddleocr import PaddleOCR

# ---------------- ODOO CONFIG ----------------
ODOO_API_KEY = "3Y6PV5OZF6IBLESDUT0DVSUMMJE7A0P5"  # replace with your real API key
ODOO_BASE_URL = "https://erp.wittmannindia.com/api"
ODOO_HEADERS = {"API-KEY": ODOO_API_KEY, "Content-Type": "application/json"}

# ---------------- CONFIG ----------------
MODEL_PATH = r"D:\MyModels\Qwen-7B\models--Qwen--Qwen2.5-7B-Instruct"
BOT_TOKEN = "8198712963:AAEwbjWFNhb5viDKoxqPwYtFPAiZyRFvIgE"

# Added WRO and GRO groups (from user's provided chat IDs)
REGIONAL_GROUPS = {
    "nro": -1003114676732,
    "sro": -1003126363672,
    "wro": -1003760116097,
    "gro": -1003899776346
}
ADMIN_GROUP = -1002870999915

# Map region keys to the short codes used in Odoo
REGION_CODE = {
    "nro": "NRO",
    "sro": "SRO",
    "wro": "WRO",
    "intl": "INTL",
    "other": "OTHER",
    "gro": "GRO"
}

# --- LOGGING SETUP ---
os.makedirs("logs", exist_ok=True)
log_filename = datetime.now().strftime("logs/merged_bot_%Y-%m-%d_%H-%M-%S.log")

logger = logging.getLogger("merged_bot")
logger.setLevel(logging.INFO)
# Ensure file handler writes utf-8 to avoid encoding errors
file_handler = logging.FileHandler(log_filename, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%H:%M:%S"))
logger.addHandler(console_handler)

logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.WARNING)

# ---------------- MODEL LOAD ----------------
print("🚀 Using device:", "CUDA" if torch.cuda.is_available() else "CPU")
logger.info(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        quantization_config=bnb_config,
        dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("✅ Model loaded (4-bit).")
    logger.info("Model loaded (4-bit).")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    logger.critical(f"Failed to load model: {e}")
    exit()

# ---------------- OCR (PaddleOCR) ----------------
try:
    ocr = PaddleOCR(lang="en")
    print("✅ PaddleOCR initialized.")
    logger.info("PaddleOCR initialized.")
except Exception as e:
    print(f"❌ Failed to initialize PaddleOCR: {e}")
    logger.critical(f"Failed to initialize PaddleOCR: {e}")
    exit()


def extract_text_from_image(image_path):
    """Use PaddleOCR to extract raw lines of text from the image."""
    result = ocr.ocr(image_path)
    lines = []
    try:
        for line in result[0]:
            text_line = line[1][0]
            # normalize whitespace and remove non-printable chars
            text_line = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text_line)
            text_line = text_line.strip()
            if text_line:
                lines.append(text_line)
    except Exception:
        return ""

    return "\n".join(lines)


def extract_info_with_llm(text):
    """Use your local LLM to extract structured fields from OCR text."""
    tokens = tokenizer.encode(text)
    if len(tokens) > 300:
        tokens = tokens[:300]
    reduced_text = tokenizer.decode(tokens, skip_special_tokens=True)

    messages = [
        {"role": "system",
         "content": """You are an expert data extraction assistant.
The business card text is from an OCR and may contain garbage or hallucinated words.
correct the abrevitaion in comapnies pvt ltd yourself if wrong and contact name would mostly match the email.
choose mobile number and not the telephone number.
Be critical and ignore text that is clearly not part of a real name, title, or address.
Extract and format the given text into this format ONLY:

contact_name:<Name>
name:<company name>
email:<email>
phoneno:<mobile no>
desigination:<Designation>
address:<complete Address>
steet:<Street only>
city:<city only>
area:<area only>
state:<State only>
country:<country only>
website:<website>

If a field is not present, write "N/A".
Do NOT include any extra text or explanations.
Start your response directly with "Contact_name:".
""" },
        {"role": "user",
         "content": f"Business card text:\n{reduced_text}"}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids

    eos_ids = [tokenizer.eos_token_id]
    im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_token_id, int):
        eos_ids.append(im_end_token_id)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_ids
        )

    generated_ids = output[0][input_ids.shape[1]:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if "Name:" in result:
        result = result[result.index("Name:"):]
    result = result.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    return result.strip()


def process_image(image_path):
    try:
        # 1. Get raw OCR text ONLY
        text = extract_text_from_image(image_path)

        # 2. Pass raw OCR text directly to LLM
        llm_result = extract_info_with_llm(text)

        return llm_result

    except Exception as e:
        logger.error(f"Error in process_image: {e}")
        return f"An error occurred during OCR/LLM processing: {e}"


def save_extracted_data(extracted_data_text):
    os.makedirs("extracted_data", exist_ok=True)

    data = {
        "contact_name": "N/A",
        "desigination": "N/A",
        "name": "N/A",
        "phoneno": "N/A",
        "email": "N/A",
        "website": "N/A",
        "address": "N/A",
        "street": "N/A",
        "city": "N/A",
        "area": "N/A",
        "state": "N/A",
        "country": "N/A",
        "date": datetime.now().strftime("%Y-%m-%d")
    }

    for line in extracted_data_text.splitlines():
        parts = re.match(r"^(\w+)\s*:\s*(.*)", line)
        if parts:
            key = parts.group(1).lower()
            val = parts.group(2).strip()
            if key in data:
                data[key] = val

    filename = re.sub(r"[^a-zA-Z0-9]+", "_", data.get("name", "Unknown"))[:30] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    filepath = os.path.join("extracted_data", filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    logger.info(f" Extracted data saved to {filepath}")
    return filepath, data


def get_city_id_from_odoo(city_name):
    """Fetch the city ID from Odoo based on the city name."""
    if not city_name or city_name == "N/A":
        return None

    url = f"{ODOO_BASE_URL}/exbhition_city_detail/search"
    params = {"domain": f"[('name','=',\"{city_name}\")]", "fields": "['id','name']"}
    try:
        res = requests.get(url, headers=ODOO_HEADERS, params=params, timeout=10)
        res_json = res.json()
        if res_json.get("success") and res_json.get("data"):
            return res_json["data"][0]["id"]
        else:
            logger.info(f" City '{city_name}' not found in Odoo.")
    except Exception as e:
        logger.error(f" Error fetching city ID for '{city_name}': {e}")
    return None


def get_category_id_from_odoo(category_name):
    """
    Fetch category_detail ID by name (A, A+, B, etc.).
    Exact match first. Fallback to ilike if not found.
    """
    if not category_name:
        return None

    cat = category_name.strip()
    if not cat:
        return None

    try:
        url = f"{ODOO_BASE_URL}/witt_crm_exbhition_category_detail/search"
        params = {"domain": f"[('name','=',\"{cat}\")]", "fields": "['id','name']"}
        res = requests.get(url, headers=ODOO_HEADERS, params=params, timeout=10)
        res_json = res.json()
        if res_json.get("success") and res_json.get("data"):
            logger.info(f"Exact category match found: {cat}")
            return res_json["data"][0]["id"]
    except Exception as e:
        logger.debug(f"Exact category lookup error for '{cat}': {e}")

    # fallback ilike
    try:
        params = {"domain": f"[('name','ilike','{cat}')]", "fields": "['id','name']"}
        res = requests.get(url, headers=ODOO_HEADERS, params=params, timeout=10)
        res_json = res.json()
        if res_json.get("success") and res_json.get("data"):
            logger.info(f"Fallback fuzzy category match: {cat} -> {res_json['data'][0]['name']}")
            return res_json["data"][0]["id"]
    except Exception as e:
        logger.debug(f"Fuzzy category lookup error for '{cat}': {e}")

    return None


def get_product_id_from_odoo(product_name):
    """
    Lookup a single product by name (returns the first match id) using ilike.
    Returns None if not found.
    """
    if not product_name:
        return None

    pn = product_name.strip()
    if not pn:
        return None

    try:
        url = f"{ODOO_BASE_URL}/exbhition_product_detail/search"
        params = {"domain": f"[('name','ilike','{pn}')]", "fields": "['id','name']"}
        res = requests.get(url, headers=ODOO_HEADERS, params=params, timeout=10)
        res_json = res.json()
        if res_json.get("success") and res_json.get("data"):
            return res_json["data"][0]["id"]
    except Exception as e:
        logger.debug(f"Product lookup error for '{pn}': {e}")

    return None


def get_region_from_city_id(city_id):
    """Fetch 'region' directly from exbhition_city_detail using city_id."""
    if not city_id:
        return None

    url = f"{ODOO_BASE_URL}/exbhition_city_detail/search"
    params = {
        "domain": f"[('id','=',{city_id})]",
        "fields": "['id','region']"
    }

    try:
        res = requests.get(url, headers=ODOO_HEADERS, params=params, timeout=10)
        data = res.json()
        if data.get("success") and data.get("data"):
            return data["data"][0].get("region")
    except Exception as e:
        logger.error(f"Error fetching region for city_id={city_id}: {e}")

    return None


def get_location_from_city_id(city_id):
    """Fetch 'location' directly from exbhition_city_detail using city_id."""
    if not city_id:
        return None

    url = f"{ODOO_BASE_URL}/exbhition_city_detail/search"
    params = {
        "domain": f"[('id','=',{city_id})]",
        "fields": "['id','location']"
    }

    try:
        res = requests.get(url, headers=ODOO_HEADERS, params=params, timeout=10)
        data = res.json()
        if data.get("success") and data.get("data"):
            return data["data"][0].get("location")
    except Exception as e:
        logger.error(f"Error fetching location for city_id={city_id}: {e}")

    return None


def verify_odoo_entry(data):
    """Check if record was actually created in Odoo."""
    url = f"{ODOO_BASE_URL}/crm_exbhition_visitor_detail/search"
    domain_parts = []
    if data.get("email") and data["email"] != "N/A":
        domain_parts.append(f"('email','=','{data['email']}')")
    if data.get("phoneno") and data["phoneno"] != "N/A":
        domain_parts.append(f"('phoneno','=','{data['phoneno']}')")
    if data.get("name") and data["name"] != "N/A":
        domain_parts.append(f"('name','=','{data['name']}')")

    if not domain_parts:
        return {"success": False, "message": "No valid domain for verification."}

    domain = "[" + ",".join(domain_parts) + "]"
    params = {"domain": domain, "fields": "['id','name','email','phoneno']"}

    try:
        res = requests.get(url, headers=ODOO_HEADERS, params=params, timeout=10)
        res_json = res.json()
        if res_json.get("success") and res_json.get("data"):
            logger.info(" Verified record exists in Odoo DB.")
            return {"success": True, "data": res_json["data"][0]}
        else:
            logger.warning(" Record not found in Odoo DB.")
            return {"success": False, "message": "Record not found in Odoo DB."}
    except Exception as e:
        logger.error(f" Verification error: {e}")
        return {"success": False, "message": str(e)}


# ---------------- CAPTION PARSER ----------------
def parse_caption_lines(caption_text):
    """
    Split caption into clean non-empty lines.
    Preserves order.
    """
    if not caption_text:
        return []

    lines = []
    for line in caption_text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return lines


# ---------------- ODOO PUSH ----------------
def push_to_odoo(data, image_path=None, caption_lines=None, posting_region_hint=None):
    """
    Push visitor data to Odoo.

    Region selection priority:
      1) region returned by Odoo using city_id (get_region_from_city_id)
      2) if caption mentions one of the known region keys -> corresponding code (NRO/SRO/WRO/GRO)
      3) else use posting_region_hint to choose
    posting_region_hint should be string 'north'/'south'/'wro'/'gro' (derived from which telegram group message was posted)
    """
    url = f"{ODOO_BASE_URL}/crm_exbhition_visitor_detail/create"
    city_id = get_city_id_from_odoo(data.get("city"))

    # 1) try get Odoo region
    region = get_region_from_city_id(city_id)

    # Prepare caption lowercase aggregate for simple region detection
    caption_text_lower = ""
    if caption_lines:
        caption_text_lower = " ".join([c.lower() for c in caption_lines])

    # 2) determine region from caption if Odoo did not return region
    if not region:
        for rkey, rcode in REGION_CODE.items():
            if rkey in caption_text_lower:
                region = rcode
                logger.info(f"Region determined from caption -> {region}")
                break

    # 3) fallback to posting group hint
    if not region and posting_region_hint:
        code = REGION_CODE.get(posting_region_hint.lower())
        if code:
            region = code
            logger.info(f"Region fallback to posting group -> {region}")

    # location fallback L10
    location = get_location_from_city_id(city_id)
    if not location:
        location = "L10"
        logger.info("Location not found in Odoo; using fallback location 'L10'.")

    # caption parsing for category and product
    resolved_category_id = None
    resolved_product_ids = []
    seen_products = set()

    if caption_lines:
        for raw in caption_lines:
            line = raw.strip()
            # strip prefixes like "category:" or "prod:"
            line = re.sub(r"^(category|cat|prod|product)\s*[:\-]\s*", "", line, flags=re.IGNORECASE).strip()
            if not line:
                continue

            # try category (exact then ilike)
            if not resolved_category_id:
                cat_id = get_category_id_from_odoo(line)
                if not cat_id:
                    cat_id = get_category_id_from_odoo(line.upper())
                if cat_id:
                    resolved_category_id = cat_id
                    logger.info(f"Category resolved: {line}")
                    continue

            # try product full phrase
            prod_id = get_product_id_from_odoo(line)
            if prod_id and prod_id not in seen_products:
                resolved_product_ids.append(prod_id)
                seen_products.add(prod_id)
                logger.info(f"Product resolved (full): {line}")
                continue

            # fallback: split by words and try individual words
            for w in re.split(r"[,\s\/\-]+", line):
                if not w:
                    continue
                pid = get_product_id_from_odoo(w)
                if pid and pid not in seen_products:
                    resolved_product_ids.append(pid)
                    seen_products.add(pid)
                    logger.info(f"Product resolved (word): {w}")

    # build payload
    odoo_data = {
        "name": data.get("name", "N/A"),
        "email": data.get("email", "N/A"),
        "contact_name": data.get("contact_name", "N/A"),
        "phoneno": data.get("phoneno", "N/A"),
        "desigination": data.get("desigination", "N/A"),
        "address": data.get("address", "N/A"),
        "city": city_id,
        "customer_region": region,
        "customer_location": location,
        "street": data.get("street", "N/A"),
        "state": data.get("state", "N/A"),
        "country": data.get("country", "N/A"),
        "website": data.get("website", "N/A"),
        "date": data.get("date"),
        "salesperson": 2,
        "Exbhition_id": 60
    }

    if resolved_category_id:
        odoo_data["category_detail"] = resolved_category_id

    if resolved_product_ids:
        odoo_data["product"] = [(6, 0, resolved_product_ids)]

    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, "rb") as f:
                odoo_data["business_card_front"] = base64.b64encode(f.read()).decode()
        except Exception as e:
            logger.error(f"Image encoding failed: {e}")

    try:
        res = requests.post(url, headers=ODOO_HEADERS, json=odoo_data, timeout=10)
        logger.info(f" Odoo Response: {res.text}")
        try:
            res_json = res.json()
        except Exception:
            res_json = {"success": False, "error": "Non-JSON response from Odoo"}

        if res_json.get("success"):
            res_json["verification"] = verify_odoo_entry(data)
        return res_json
    except Exception as e:
        logger.error(f" Failed to push to Odoo: {e}")
        return {"success": False, "error": str(e)}


# ---------------- TELEGRAM HANDLER ----------------
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.photo:
        return

    chat_id = msg.chat.id
    if chat_id not in REGIONAL_GROUPS.values():
        return

    # region_hint will be one of the keys in REGIONAL_GROUPS (north/south/wro/gro)
    region_hint = [k for k, v in REGIONAL_GROUPS.items() if v == chat_id][0]
    caption_original = msg.caption or ""
    caption_lines = parse_caption_lines(caption_original)

    logger.info(f"Caption lines: {caption_lines}")

    photo = msg.photo[-1]
    file = await photo.get_file()
    image_path = f"temp_{photo.file_unique_id}.jpg"
    await file.download_to_drive(image_path)

    # send a processing message (we keep replies safe if original message gets deleted)
    processing_msg = await context.bot.send_message(chat_id=chat_id, text="Processing business card...")

    try:
        extracted_text = process_image(image_path)
        _, data = save_extracted_data(extracted_text)

        result = push_to_odoo(
            data,
            image_path=image_path,
            caption_lines=caption_lines,
            posting_region_hint=region_hint
        )

        if result.get("success"):
            await context.bot.send_message(chat_id=chat_id, text="✅ Uploaded & verified in Wittmann India CRM")
        else:
            await context.bot.send_message(chat_id=chat_id, text=f"❌ Upload failed: {result.get('error')}")

        # ---------------- ADMIN / MAIN GROUP FORWARD ----------------
        try:
            # 1) Forward original image message (keeps original sender & timestamp)
            try:
                await context.bot.forward_message(
                    chat_id=ADMIN_GROUP,
                    from_chat_id=chat_id,
                    message_id=msg.message_id
                )
                logger.info("Forwarded original image message to ADMIN_GROUP.")
            except Exception as e:
                logger.debug(f"Forward failed, will resend manually: {e}")

            # 2) Send extracted text (always searchable)
            admin_text = f"{region_hint} upload:\n\n{extracted_text}"
            if caption_original:
                admin_text += f"\n\nCaption:\n{caption_original}"

            try:
                await context.bot.send_message(chat_id=ADMIN_GROUP, text=admin_text)
                logger.info("Sent extracted text to ADMIN_GROUP.")
            except Exception as e:
                logger.debug(f"Failed to send extracted text to ADMIN_GROUP: {e}")


        except Exception as e:
            logger.error(f"Admin forwarding error: {e}")

        # delete original image message to keep group clean
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=msg.message_id)
        except Exception as e:
            logger.debug(f"Could not delete original message: {e}")

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
        # delete processing message
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=processing_msg.message_id)
        except Exception:
            pass


# ---------------- RUN BOT ----------------
async def run_bot():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    print("🤖 Bot running")
    await app.run_polling()


if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(run_bot())
