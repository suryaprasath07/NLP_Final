from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import json
import random
import time
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

lemmatizer = WordNetLemmatizer()

# Download NLTK data on startup
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'u/\w+', '', text)
    text = re.sub(r'r/\w+', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    return ' '.join(tokens)

def get_subreddit_urls(subreddit, limit=100, max_posts=500):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RedditScraper/1.0)"}
    urls = []
    after = None

    while len(urls) < max_posts:
        params = {"limit": limit}
        if after:
            params["after"] = after

        url = f"https://www.reddit.com/r/{subreddit}/new.json"
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error fetching URLs: {e}")
            break

        children = data["data"]["children"]
        if not children:
            break

        for post in children:
            urls.append("https://www.reddit.com" + post["data"]["permalink"])
            if len(urls) >= max_posts:
                break

        after = data["data"]["after"]
        if not after:
            break

    return urls

def fetch_reddit_post_data(subreddit, urls):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RedditScraper/1.0)"}

    def parse_replies(children):
        parsed = []
        for child in children:
            if 'data' not in child:
                continue
            data = child['data']
            body = data.get('body', "")
            author = data.get('author', "")
            replies_data = data.get('replies', {})
            nested_replies = []
            if isinstance(replies_data, dict):
                nested_children = replies_data.get('data', {}).get('children', [])
                nested_replies = parse_replies(nested_children)
            parsed.append({
                "text": preprocess_text(body),
                "author": author,
                "replies": nested_replies
            })
        return parsed

    result = []

    for url in urls:
        if not url.startswith(f"https://www.reddit.com/r/{subreddit}/comments/"):
            continue

        json_url = url.rstrip("/") + ".json"
        try:
            response = requests.get(json_url, headers=headers, timeout=10)
            response.raise_for_status()
            post_json = response.json()
        except Exception as e:
            print(f"Error fetching post {url}: {e}")
            continue

        post_data = post_json[0]['data']['children'][0]['data']
        topic = post_data.get('title', "")
        text = post_data.get('selftext', "")
        author = post_data.get('author', "")

        comments = post_json[1]['data']['children']
        replies = parse_replies(comments)

        result.append({
            "topic": preprocess_text(topic),
            "text": preprocess_text(text),
            "author": author,
            "replies": replies
        })

        time.sleep(random.uniform(0.5, 1))

    return result

@app.route('/api/subreddit/<subreddit>', methods=['GET'])
def get_subreddit_data(subreddit):
    """
    Fetch data from a subreddit.
    Query params:
        - max_posts: Maximum number of posts to fetch per request (default: 20)
        - offset: Starting position for pagination (default: 0)
    """
    try:
        max_posts = int(request.args.get('max_posts', 20))
        offset = int(request.args.get('offset', 0))
        max_posts = min(max_posts, 100)  # Limit to 100 posts max per request
        
        # Fetch more URLs than needed to account for offset
        total_needed = offset + max_posts
        urls = get_subreddit_urls(subreddit, limit=100, max_posts=total_needed)
        
        if not urls:
            return jsonify({
                "error": "No posts found or invalid subreddit",
                "subreddit": subreddit
            }), 404
        
        # Apply offset and limit
        urls_slice = urls[offset:offset + max_posts]
        
        if not urls_slice:
            return jsonify({
                "subreddit": subreddit,
                "post_count": 0,
                "posts": [],
                "offset": offset,
                "has_more": False
            })
        
        data = fetch_reddit_post_data(subreddit, urls_slice)
        
        return jsonify({
            "subreddit": subreddit,
            "post_count": len(data),
            "posts": data,
            "offset": offset,
            "has_more": len(urls) > (offset + max_posts)
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "subreddit": subreddit
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

@app.route('/')
def index():
    """API documentation"""
    return jsonify({
        "message": "Reddit Data API",
        "endpoints": {
            "/api/health": "Health check",
            "/api/subreddit/<subreddit>": "Get posts from subreddit (query param: max_posts)"
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)