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
    
    def search(self) -> List[Article]:
        response = self.tavily.search(
            query="artificial intelligence and machine learning news in model release and new features",
            topic="news",
            time_period="1w",
            search_depth="advanced",
            max_results=5
        )
        return [Article(title=r['title'], url=r['url'], content=r['content']) 
                for r in response['results']]

class Summarizer:
    def __init__(self):
        self.system_prompt = """You are an AI expert who makes complex topics accessible
        to general audiences. Summarize this article in 2-3 sentences, focusing on the key points
        and explaining any technical terms simply."""
    
    def summarize(self, article: Article) -> str:
        response = llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Title: {article.title}\n\nContent: {article.content}")
        ])
        return response.content

class Publisher:
    def create_report(self, summaries: List[Dict]) -> str:
        prompt = """Create a weekly AI/ML news report for the general public.
        Format it with:
        1. A brief introduction
        2. The main news items with their summaries
        3. Links for further reading
        Make it engaging and accessible to non-technical readers."""
        
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
        data = request.json
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