"""
Web Agent - Handles web-related tasks like search and scraping
"""

import asyncio
import requests
from typing import Dict, Any
from bs4 import BeautifulSoup

from utils.config import Config
from utils.logger import setup_logger

class WebAgent:
    """Agent for web-related tasks"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("web_agent")
        self.session = requests.Session()
        
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web-related task"""
        task_type = params.get("task_type", "search")
        
        if task_type == "search":
            return await self._search_web(params)
        elif task_type == "scrape":
            return await self._scrape_page(params)
        elif task_type == "summarize":
            return await self._summarize_content(params)
        else:
            return {"error": f"Unknown task type: {task_type}"}
    
    async def _search_web(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform web search"""
        query = params.get("query", "")
        if not query:
            return {"error": "No search query provided"}
        
        # Simple DuckDuckGo search (no API key required)
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={query}"
            response = self.session.get(search_url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                results = []
                
                # Extract search results
                for result in soup.find_all('div', class_='result')[:5]:
                    title_elem = result.find('a', class_='result__a')
                    if title_elem:
                        title = title_elem.get_text()
                        url = title_elem.get('href')
                        snippet = result.find('a', class_='result__snippet')
                        snippet_text = snippet.get_text() if snippet else ""
                        
                        results.append({
                            "title": title,
                            "url": url,
                            "snippet": snippet_text
                        })
                
                return {
                    "query": query,
                    "results": results,
                    "result_count": len(results)
                }
            else:
                return {"error": f"Search failed with status {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}
    
    async def _scrape_page(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Scrape content from a web page"""
        url = params.get("url", "")
        if not url:
            return {"error": "No URL provided"}
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract main content
            title = soup.find('title')
            title_text = title.get_text() if title else "No title"
            
            # Try to get main content
            content_selectors = ['article', '.content', '.main', 'main']
            content = ""
            
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    content = element.get_text()[:1000]  # Limit content length
                    break
            
            if not content:
                # Fallback to body content
                body = soup.find('body')
                content = body.get_text()[:1000] if body else "No content found"
            
            return {
                "url": url,
                "title": title_text,
                "content": content.strip(),
                "content_length": len(content)
            }
            
        except Exception as e:
            return {"error": f"Scraping failed: {str(e)}"}
    
    async def _summarize_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize web content"""
        content = params.get("content", "")
        if not content:
            return {"error": "No content to summarize"}
        
        # Simple extractive summarization
        sentences = content.split('.')
        # Take first few sentences as summary
        summary_sentences = sentences[:3]
        summary = '. '.join(summary_sentences) + '.'
        
        return {
            "original_length": len(content),
            "summary_length": len(summary),
            "summary": summary,
            "compression_ratio": len(summary) / len(content)
        }