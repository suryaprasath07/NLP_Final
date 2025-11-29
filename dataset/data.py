import requests
import json
import random
import time
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

def get_subreddit_urls(subreddit, limit=100, max_posts=500):
    """
    Fetches post URLs from the 'new' section of a subreddit with pagination.

    Args:
        subreddit (str): Name of the subreddit.
        limit (int): Number of posts per request (max 100).
        max_posts (int): Total number of posts to fetch.

    Returns:
        list: List of post URLs.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RedditScraper/1.0)"}
    urls = []
    after = None

    while len(urls) < max_posts:
        params = {"limit": limit}
        if after:
            params["after"] = after

        url = f"https://www.reddit.com/r/{subreddit}/new.json"
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        children = data["data"]["children"]
        if not children:
            break

        for post in children:
            urls.append("https://www.reddit.com" + post["data"]["permalink"])
            if len(urls) >= max_posts:
                break

        after = data["data"]["after"]
        if not after:
            break  # No more pages

    return urls

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove Reddit user/subreddit mentions
    text = re.sub(r'u/\w+', '', text)
    text = re.sub(r'r/\w+', '', text)
    
    # Remove emojis/unicode
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize + lemmatize
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    
    return ' '.join(tokens)

def fetch_reddit_post_data(subreddit, urls):
    """
    Fetch Reddit post JSONs and extract text, author, and nested replies.

    Args:
        urls (list): List of Reddit URLs.

    Returns:
        list: List of dictionaries with topic, text, author, and replies.
    """

    headers = {"User-Agent": "Mozilla/5.0 (compatible; RedditScraper/1.0)"}

    def parse_replies(children):
        """
        Recursively parse comment replies into required format.
        """
        parsed = []
        for child in children:
            if 'data' not in child:
                continue
            data = child['data']
            body = data.get('body', "")
            author = data.get('author', "")
            # Some replies can be empty or not exist
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

        # Add .json to get the JSON version of the post
        json_url = url.rstrip("/") + ".json"
        response = requests.get(json_url, headers=headers)
        response.raise_for_status()
        post_json = response.json()

        # First object is post info
        post_data = post_json[0]['data']['children'][0]['data']
        topic = post_data.get('title', "")
        text = post_data.get('selftext', "")
        author = post_data.get('author', "")

        # Second object is comments
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

def main():
    nltk.download('punkt_tab')
    nltk.download('punkt')  # standard tokenizer
    nltk.download('wordnet')  # for lemmatizer

    subreddit = "science"
    urls = get_subreddit_urls(subreddit, limit=100, max_posts=300)
    print("==============FETCHED URLS================")
    print("Number of urls fetched : ", len(urls))
    data_dcit = fetch_reddit_post_data(subreddit, urls[:20])
    print("Number of posts fetched : ", len(data_dcit))

    with open("data.json", "w") as json_file:
        json.dump(data_dcit, json_file, indent=4)

if __name__ == "__main__":
    main()