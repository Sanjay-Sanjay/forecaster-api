from flask import Flask, jsonify, request
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
import torch
from peft import PeftModel
import re
from datetime import datetime, timedelta
import feedparser
import urllib.parse
import json

app = Flask(__name__)

try:
    print("Downloading base model")
    base_model = "NousResearch/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side="right"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, "./models/llama_forecaster_q4bit_5e_v2")
    pipe = pipeline(
        task="text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_length=4096
    )
    print("Model and Tokenizer loaded")
except Exception as e:
    print("error loading token and model",e)

with open("./company_names.json", 'r') as json_file:
    company_names = json.load(json_file)

def company_wise_news(keyword):
    try:
        base_url = "https://news.google.com/rss/search?q="
        encoded_keyword = urllib.parse.quote(keyword) 
        keyword_url = base_url + encoded_keyword
        feed = feedparser.parse(keyword_url)
        today = datetime.now()
        week_ago = today - timedelta(days=3)
        articles = []
        news_count = 0
        if feed.entries:
            for entry in feed.entries:
                published_date = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %Z")
                if published_date >= week_ago and news_count < 10:
                    title = entry.title
                    articles.append({"title": title, "published_date": published_date})
                    news_count += 1
            if not articles:
                print(f"No latest news found for the keyword '{keyword}'.")
                return ""
        else:
                print(f"No latest news found for the keyword '{keyword}'.")
                return ""
        sorted_articles = sorted(articles, key=lambda x: x["published_date"], reverse=True)
        output = ""
        for article in sorted_articles:
            output += f"{article['published_date'].strftime('%Y-%m-%d %H:%M:%S')} : {article['title']}\n"
        return output
    except Exception as e:
        print(f"An error occurred while getting news: {str(e)}")
        return ""

def forecaster_inference(question,company_key):
    try:
        fundamental_data = ""
        technical_data = ""
        news_data = ""
        twitter_data = ""
        company_name = company_names[company_key]
        if company_name:
            news_data = company_wise_news(company_name)
        
        print("Fetching fundamental, technical, news and twitter data...")

        with open (f"./fundamental/{company_key}.txt") as file: 
            fundamental_data = file.read()
        with open (f"./technical/{company_key}.txt") as file: 
            technical_data = file.read()
        with open (f"./twitter/{company_key}.txt") as file: 
            twitter_data = file.read()
        
        if len(news_data) == 0:
            print("No latest news data found, loading data from cached data")
            with open (f"./news/{company_key}.txt") as file: 
                news_data = file.read()
            
        prompt = f"### Instruction:\n{question}\n### Input:\n"
        if len(news_data) != 0:
            prompt += "[Current company news for the last 1 week]:\n" + news_data
        if len(twitter_data) != 0:
            prompt += "\n[Current Tweets for last 1 week]:\n" + twitter_data
        if len(fundamental_data) != 0:
            prompt += "\n[Fundamental analysis]:\n" + fundamental_data
        if len(technical_data) != 0:
            prompt += "\n[Technical analysis]:\n"  + technical_data
        prompt += "\n[Positive developments]:"
        output = pipe(prompt)
        answer = output[0]['generated_text'][len(prompt):]
        if ("### End of Reponse") in answer:
            answer.split("### End of Reponse")
            return "[Positive developments]:\n"+answer[0]
        return "[Positive developments]:\n"+answer
    except Exception as e:
        return f"Error in forecaster inference {e}"

@app.route('/forecaster', methods=['POST'])
def inference():
    if request.method == 'POST':
        try:
            data = request.json  
            prompt = data.get('prompt')
            symbol = data.get('symbol')
            print(f"received input request: {prompt}")
            if prompt is None or symbol is None:
                return jsonify({'error': 'Invalid data. Both prompt and symbol are required.'}), 400

            result = forecaster_inference(prompt, symbol)
            return jsonify({'result': result})
        except Exception as e:
            return jsonify({'error': f'Exception occurred: {e}'}), 500
    else:
        return jsonify({'error': 'Method Not Allowed'}), 405

if __name__ == '__main__':
    app.run(port=9008,host="127.0.0.1",debug=True)