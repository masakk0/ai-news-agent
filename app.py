from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from typing import Dict, List, Any, TypedDict, Optional
from datetime import datetime
from pydantic import BaseModel
from tavily import TavilyClient
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph

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

class Summary(TypedDict):
    title: str
    summary: str
    url: str

class GraphState(TypedDict):
    articles: Optional[List[Article]]
    summaries: Optional[List[Summary]]
    report: Optional[str]
    tavily_key: Optional[str]

class NewsSearcher:
    def __init__(self, api_key):
        self.tavily = TavilyClient(api_key=api_key)
        self.preferred_domains = [
            "openai.com",
            "anthropic.com",
            "langchain.com",
            "huggingface.co",
            "deepmind.google",
            "ai.meta.com",
            "ai.google",
            "mistral.ai",
            "cohere.com",
            "stability.ai",
            "arxiv.org",  # For research papers
            "blog.google",
            "research.ibm.com"
        ]
    
    def search(self) -> List[Article]:
        # Search with multiple focused queries
        queries = [
            "new AI model release announcement features",
            "machine learning research breakthrough discovery",
            "LLM language model new capabilities update"
        ]
        
        all_results = []
        for query in queries:
            response = self.tavily.search(
                query=query,
                topic="news",
                time_period="2w",
                search_depth="advanced",
                max_results=10,
                include_domains=self.preferred_domains
            )
            all_results.extend(response['results'])
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_articles = []
        for r in all_results:
            if r['url'] not in seen_urls:
                seen_urls.add(r['url'])
                unique_articles.append(Article(
                    title=r['title'], 
                    url=r['url'], 
                    content=r['content']
                ))
        
        # Limit to top 7-10 articles
        return unique_articles[:10]

class Summarizer:
    def __init__(self):
        self.system_prompt = """You are an AI expert summarizing news for technical audiences.
        
        For each article, extract and highlight:
        1. NEW FEATURES: What new capabilities or features were announced?
        2. RESEARCH FINDINGS: What are the key discoveries or breakthroughs?
        3. TECHNICAL DETAILS: Model sizes, performance metrics, or architectural innovations
        4. AVAILABILITY: When/how users can access these features
        
        Summarize in 3-4 concise sentences. Focus ONLY on concrete new developments, 
        not general discussions. If the article doesn't contain substantive new information, 
        note that clearly."""
    
    def summarize(self, article: Article) -> str:
        response = llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Title: {article.title}\n\nContent: {article.content}")
        ])
        return response.content

class Publisher:
    def create_report(self, summaries: List[Dict]) -> str:
        prompt = """Create a weekly AI/ML news report organized by categories:

        ## 🚀 New Model Releases & Features
        (Models, tools, or features announced this week)

        ## 🔬 Research Breakthroughs
        (Scientific discoveries, papers, or technical advances)

        ## 📊 Performance & Capabilities
        (Benchmark improvements, new capabilities, technical specs)

        ## 🛠️ Developer Tools & Updates
        (API changes, SDKs, frameworks, integrations)

        For each item:
        - Lead with the WHO (company/org) and WHAT (the specific feature/discovery)
        - Include key technical details or metrics
        - Add a one-line takeaway of why it matters
        - Include the source link

        Keep it technical but concise. Skip marketing fluff - focus on concrete, actionable information."""
        
        summaries_text = "\n\n".join([
            f"Title: {item['title']}\nSummary: {item['summary']}\nSource: {item['url']}"
            for item in summaries
        ])
        
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=summaries_text)
        ])
        return response.content

def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    searcher = NewsSearcher(state['tavily_key'])
    state['articles'] = searcher.search()
    return state

def summarize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    summarizer = Summarizer()
    state['summaries'] = []
    
    for article in state['articles']:
        summary = summarizer.summarize(article)
        
        # Filter out articles with no substantial new information
        low_value_indicators = [
            "doesn't contain substantive new information",
            "no concrete new developments",
            "general discussion only"
        ]
        
        # Only include if it has real substance
        if not any(indicator.lower() in summary.lower() for indicator in low_value_indicators):
            state['summaries'].append({
                'title': article.title,
                'summary': summary,
                'url': article.url
            })
    
    return state

def publish_node(state: Dict[str, Any]) -> Dict[str, Any]:
    publisher = Publisher()
    state['report'] = publisher.create_report(state['summaries'])
    return state

def create_workflow() -> StateGraph:
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
    try:
        data = request.get_json(force=True, silent=True) or {}
        tavily_key = data.get('tavily_key')
        
        if not tavily_key:
            return jsonify({'error': 'Tavily API key required'}), 400
        
        workflow = create_workflow()
        final_state = workflow.invoke({
            "articles": None,
            "summaries": None,
            "report": None,
            "tavily_key": tavily_key
        })
        
        return jsonify({
            'date': datetime.now().strftime("%Y-%m-%d"),
            'articles': final_state['summaries'],
            'report': final_state['report']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
