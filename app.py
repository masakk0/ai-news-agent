from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from typing import Dict, List, Any, TypedDict, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph
import feedparser
from bs4 import BeautifulSoup
from html import unescape
import requests

app = Flask(__name__, static_folder='static')
CORS(app)

# Initialize Azure LLM from env
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

class Article(BaseModel):
    title: str
    url: str
    content: str
    source: str
    published_date: Optional[str] = None

class Summary(TypedDict):
    title: str
    summary: str
    url: str
    source: str

class GraphState(TypedDict):
    articles: Optional[List[Article]]
    summaries: Optional[List[Summary]]
    report: Optional[str]

class RSSFeedReader:
    def __init__(self):
        self.rss_feeds = {
            # Major AI Companies
            'OpenAI': 'https://openai.com/news/rss.xml',
            'Anthropic': 'https://www.anthropic.com/news/rss',
            'Google DeepMind': 'https://deepmind.google/blog/rss.xml',
            'Meta AI': 'https://ai.meta.com/blog/rss/',
            'Microsoft Research': 'https://www.microsoft.com/en-us/research/feed/',
            
            # AI Frameworks & Tools
            'HuggingFace': 'https://huggingface.co/blog/feed.xml',
            'LangChain': 'https://blog.langchain.dev/rss/',
            'LlamaIndex': 'https://blog.llamaindex.ai/feed',
            
            # Model Providers
            'Mistral AI': 'https://mistral.ai/news/rss.xml',
            'Cohere': 'https://cohere.com/blog/rss.xml',
            'Stability AI': 'https://stability.ai/news/rss',
            
            # Research & Academic
            'Google AI Blog': 'https://blog.research.google/feeds/posts/default',
            'AWS Machine Learning': 'https://aws.amazon.com/blogs/machine-learning/feed/',
            'NVIDIA AI': 'https://blogs.nvidia.com/feed/',
            
            # Additional Sources
            'Weights & Biases': 'https://wandb.ai/site/blog/rss.xml',
            'PyTorch': 'https://pytorch.org/blog/feed.xml',
        }
    
    def fetch_recent_posts(self, days=7) -> List[Article]:
        """Fetch recent posts from RSS feeds"""
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for company, feed_url in self.rss_feeds.items():
            try:
                feed = feedparser.parse(feed_url)
                
                # Get up to 10 recent posts per feed
                for entry in feed.entries[:10]:
                    # Check if post is recent
                    published_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        post_date = datetime(*entry.published_parsed[:6])
                        if post_date < cutoff_date:
                            continue
                        published_date = post_date.strftime("%Y-%m-%d")
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        post_date = datetime(*entry.updated_parsed[:6])
                        if post_date < cutoff_date:
                            continue
                        published_date = post_date.strftime("%Y-%m-%d")
                    
                    title = entry.get('title', '')
                    url = entry.get('link', '')
                    content = entry.get('summary', '') or entry.get('description', '') or entry.get('content', [{}])[0].get('value', '')
                    
                    # Clean HTML from content
                    if content:
                        content = BeautifulSoup(content, 'html.parser').get_text()
                        content = unescape(content)
                        # Limit content length but keep enough for good summaries
                        content = content[:3000]
                    
                    if title and url:
                        articles.append(Article(
                            title=title,
                            url=url,
                            content=content,
                            source=company,
                            published_date=published_date
                        ))
                    
            except Exception as e:
                print(f"Error reading RSS from {company}: {e}")
                continue
        
        return articles


class ArxivReader:
    """Fetch recent AI/ML papers from arxiv"""
    
    def fetch_recent_papers(self, days=7, max_results=10) -> List[Article]:
        """Fetch recent AI papers from arxiv API"""
        articles = []
        
        # Search queries for different AI topics
        queries = [
            'cat:cs.AI OR cat:cs.LG OR cat:cs.CL',  # AI, ML, Computational Linguistics
        ]
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for query in queries:
            try:
                url = f'http://export.arxiv.org/api/query?search_query={query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending'
                response = requests.get(url)
                
                if response.status_code == 200:
                    # Parse XML response
                    from xml.etree import ElementTree as ET
                    root = ET.fromstring(response.content)
                    
                    # Namespace for arxiv API
                    ns = {'atom': 'http://www.w3.org/2005/Atom'}
                    
                    for entry in root.findall('atom:entry', ns):
                        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                        url = entry.find('atom:id', ns).text
                        summary = entry.find('atom:summary', ns).text.strip()
                        published = entry.find('atom:published', ns).text
                        
                        # Parse date
                        pub_date = datetime.strptime(published[:10], '%Y-%m-%d')
                        if pub_date < cutoff_date:
                            continue
                        
                        # Get authors
                        authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
                        author_str = authors[0] + (' et al.' if len(authors) > 1 else '')
                        
                        articles.append(Article(
                            title=f"{title}",
                            url=url,
                            content=f"Authors: {author_str}\n\n{summary[:2000]}",
                            source="arXiv",
                            published_date=pub_date.strftime("%Y-%m-%d")
                        ))
                        
            except Exception as e:
                print(f"Error fetching arxiv papers: {e}")
                continue
        
        return articles


class Summarizer:
    def __init__(self):
        self.system_prompt = """You are an AI expert summarizing official announcements and research for technical audiences.

For each article, extract and highlight:
1. NEW FEATURES: What new capabilities, features, or products were announced?
2. RESEARCH FINDINGS: What are the key discoveries or breakthroughs?
3. TECHNICAL DETAILS: Model sizes, performance metrics, architectural innovations, benchmarks
4. AVAILABILITY: When/how users can access these features (API, open-source, beta, etc.)

Guidelines:
- Summarize in 3-5 concise sentences
- Focus ONLY on concrete new developments, not general discussions
- Include specific metrics, numbers, or benchmarks when available
- If it's a research paper, highlight the main contribution and results
- If the article doesn't contain substantive new information, state: "No significant new developments"

Be technical and precise. Skip marketing language."""
    
    def summarize(self, article: Article) -> str:
        try:
            response = llm.invoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"Source: {article.source}\nTitle: {article.title}\n\nContent: {article.content}")
            ])
            return response.content
        except Exception as e:
            print(f"Error summarizing article '{article.title}': {e}")
            return "Error generating summary"


class Publisher:
    def create_report(self, summaries: List[Dict]) -> str:
        prompt = """Create a comprehensive weekly AI/ML news report organized by categories:

## 🚀 New Model Releases & Features
(New models, major version updates, or significant feature launches)

## 🔬 Research Breakthroughs & Papers
(Scientific discoveries, important papers, novel techniques)

## 📊 Performance & Benchmarks
(Benchmark improvements, capability expansions, technical specifications)

## 🛠️ Developer Tools & Platform Updates
(API changes, SDKs, frameworks, libraries, integrations)

## 🏢 Company & Strategic Announcements
(Partnerships, initiatives, significant organizational news)

For each item:
- Start with **[Source Name]** in bold
- Lead with the specific feature, discovery, or announcement
- Include key technical details, metrics, or numbers
- Add a brief note on why it matters or what's innovative
- Include the source link

Format:
**[Company/Source]** Brief headline
Technical details and key points. Why it matters.
[Read more](url)

Guidelines:
- Be technical and precise
- Skip marketing fluff
- Focus on actionable information
- Group related items together
- If a section has no items, you can omit it

Keep it professional and information-dense."""
        
        summaries_text = "\n\n".join([
            f"Source: [{item['source']}]\nTitle: {item['title']}\nSummary: {item['summary']}\nURL: {item['url']}"
            for item in summaries
        ])
        
        try:
            response = llm.invoke([
                SystemMessage(content=prompt),
                HumanMessage(content=f"Here are this week's AI/ML updates:\n\n{summaries_text}")
            ])
            return response.content
        except Exception as e:
            print(f"Error creating report: {e}")
            return "Error generating report"


def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Gather articles from RSS feeds and arxiv"""
    all_articles = []
    
    # 1. Get official announcements from RSS feeds
    print("Fetching RSS feeds...")
    rss_reader = RSSFeedReader()
    rss_articles = rss_reader.fetch_recent_posts(days=7)
    all_articles.extend(rss_articles)
    print(f"Found {len(rss_articles)} RSS articles")
    
    # 2. Get research papers from arxiv
    print("Fetching arxiv papers...")
    arxiv_reader = ArxivReader()
    arxiv_articles = arxiv_reader.fetch_recent_papers(days=7, max_results=15)
    all_articles.extend(arxiv_articles)
    print(f"Found {len(arxiv_articles)} arxiv papers")
    
    # 3. Remove duplicates by URL
    seen_urls = set()
    unique_articles = []
    for article in all_articles:
        if article.url not in seen_urls:
            seen_urls.add(article.url)
            unique_articles.append(article)
    
    print(f"Total unique articles: {len(unique_articles)}")
    state['articles'] = unique_articles
    return state


def summarize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize each article"""
    print("Summarizing articles...")
    summarizer = Summarizer()
    state['summaries'] = []
    
    for article in state['articles']:
        summary = summarizer.summarize(article)
        
        # Filter out articles with no substantial new information
        low_value_indicators = [
            "no significant new developments",
            "doesn't contain substantive new information",
            "no concrete new developments",
            "general discussion only",
            "error generating summary"
        ]
        
        # Only include if it has real substance
        if not any(indicator.lower() in summary.lower() for indicator in low_value_indicators):
            state['summaries'].append({
                'title': article.title,
                'summary': summary,
                'url': article.url,
                'source': article.source
            })
    
    print(f"Created {len(state['summaries'])} summaries")
    return state


def publish_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Create final report"""
    print("Creating report...")
    publisher = Publisher()
    state['report'] = publisher.create_report(state['summaries'])
    return state


def create_workflow() -> StateGraph:
    """Create the LangGraph workflow"""
    workflow = StateGraph(state_schema=GraphState)
    workflow.add_node("search", search_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("publish", publish_node)
    workflow.add_edge("search", "summarize")
    workflow.add_edge("summarize", "publish")
    workflow.set_entry_point("search")
    return workflow.compile()


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/generate', methods=['POST'])
def generate_report():
    """Generate the weekly AI news report"""
    try:
        print("Starting report generation...")
        
        workflow = create_workflow()
        final_state = workflow.invoke({
            "articles": None,
            "summaries": None,
            "report": None
        })
        
        print("Report generation complete!")
        
        return jsonify({
            'date': datetime.now().strftime("%Y-%m-%d"),
            'articles_count': len(final_state['summaries']),
            'articles': final_state['summaries'],
            'report': final_state['report']
        })
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/sources', methods=['GET'])
def get_sources():
    """Get list of all RSS sources being monitored"""
    reader = RSSFeedReader()
    return jsonify({
        'rss_feeds': list(reader.rss_feeds.keys()),
        'arxiv': True
    })


@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
