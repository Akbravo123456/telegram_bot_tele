from flask import Flask, request
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import requests
import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
import telebot

nltk.download('punkt_tab')
nltk.download('stopwords')
TOKEN = "8041755254:AAGKsVYSmhkomBbTSIfc2idIHG-2ISXRHg0"
bot = telebot.TeleBot(TOKEN)

HF_TOKEN = "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"  
login(HF_TOKEN) 

model_name = "EleutherAI/gpt-neo-1.3B"  
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN)

llama_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  
)
user_data = {}

questions = [
    "What industry is your business in?",
    "What is your business objective? (e.g., lead generation, sales)",
    "Do you have a website? If yes, please provide the URL.",
    "Do you have any social media platforms? If yes, please provide the URL.",
    "Do you use PPC campaigns? (yes/no)",
    "Who is your target audience? (e.g., young adults, professionals)",
    "What location would you like to target?",
]

def extract_keywords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    keywords = [word for word in word_tokens if word.isalnum() and word.lower() not in stop_words]
    return keywords

def scrape_website(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            texts = soup.findAll(text=True)
            visible_texts = filter(lambda t: t.parent.name not in ["style", "script", "head", "meta", "[document]"], texts)
            return " ".join(visible_texts)
        else:
            return None
    except Exception as e:
        return f"Error: {str(e)}"

def fetch_ppc_benchmarks(industry):
    try:
        url = "https://databox.com/ppc-industry-benchmarks"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            rows = soup.find_all("tr")
            benchmark_data = {}

            for row in rows:
                cols = row.find_all("td")
                if cols and industry.lower() in cols[0].text.lower():
                    benchmark_data["industry"] = cols[0].text.strip()
                    benchmark_data["CPC"] = cols[1].text.strip()
                    benchmark_data["CTR"] = cols[2].text.strip()
                    benchmark_data["CTC"] = cols[3].text.strip()
                    break

            if benchmark_data:
                return benchmark_data
            else:
                return "No benchmark data found for the specified industry."
        else:
            return f"Failed to fetch benchmark data. Status code: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def generate_llama_answer(question):
    try:
        response = llama_pipeline(
            question,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7
        )
        return response[0]["generated_text"].strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Flask App
app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    json_data = request.get_json()
    if json_data:
        update = telebot.types.Update.de_json(json_data)
        bot.process_new_updates([update])
    return "OK", 200

@bot.message_handler(commands=["start"])
def start(message):
    chat_id = message.chat.id
    user_data[chat_id] = {"step": 0}
    bot.send_message(chat_id, "Welcome to Tele_Ad Bot!")
    bot.send_message(chat_id, questions[0])

@bot.message_handler(commands=["faq"])
def faq(message):
    chat_id = message.chat.id
    user_data[chat_id]["faq"] = True
    bot.send_message(chat_id, "Ask any digital marketing-related question, and I'll provide an answer!")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    chat_id = message.chat.id
    if user_data.get(chat_id, {}).get("faq"):
        question = message.text
        answer = generate_llama_answer(question)
        bot.send_message(chat_id, answer)
        user_data[chat_id]["faq"] = False
    elif chat_id in user_data:
        step = user_data[chat_id]["step"]
        if step == 0:
            user_data[chat_id]["industry"] = message.text
        elif step == 1:
            user_data[chat_id]["objective"] = message.text
        elif step == 2:
            user_data[chat_id]["website"] = message.text if message.text.startswith("http") else None
        elif step == 3:
            user_data[chat_id]["social_media"] = message.text
        elif step == 4:
            user_data[chat_id]["ppc"] = message.text.lower() in ["yes", "y"]
        elif step == 5:
            user_data[chat_id]["audience"] = message.text
        elif step == 6:
            user_data[chat_id]["location"] = message.text

        step += 1
        if step < len(questions):
            user_data[chat_id]["step"] = step
            bot.send_message(chat_id, questions[step])
        else:
            bot.send_message(chat_id, "Generating keywords and fetching benchmark data...")
            keywords = extract_keywords(" ".join(map(str, user_data[chat_id].values())))
            bot.send_message(chat_id, f"Here are your suggested keywords:\n{', '.join(keywords)}")

            industry = user_data[chat_id].get("industry", "general")
            benchmark_data = fetch_ppc_benchmarks(industry)
            if isinstance(benchmark_data, dict):
                bot.send_message(chat_id, f"PPC Benchmark Data for {benchmark_data['industry']}:\n"
                                          f"CPC: {benchmark_data['CPC']}\n"
                                          f"CTR: {benchmark_data['CTR']}\n"
                                          f"CTC: {benchmark_data['CTC']}")
            else:
                bot.send_message(chat_id, benchmark_data)

            del user_data[chat_id]

if __name__ == "__main__":
    bot.remove_webhook()
    bot.set_webhook(url="https://Tele_Ad/webhook")
    app.run(host="0.0.0.0", port=5000)