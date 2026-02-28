import logging
import json

from langchain_core.runnables import RunnableConfig

from graph.types import State
from config.configuration import Configuration
from config import SELECTED_SEARCH_ENGINE, SearchEngine
from tools.search import LoggedTavilySearch, get_web_search_tool

logger = logging.getLogger(__name__)



def background_investigation_node(state: State, config: RunnableConfig):
    logger.info("background_investigation node is running")
    configurable = Configuration.from_runnable_config(config)

    if not configurable.enable_web_search:
        logger.info("Web search is disabled, skipping background investigation.")
        return {"background_investigation_results": json.dumps([],ensure_ascii = False)}
    
    query = state.get("clarified_research_topic") or state.get("research_topic")
    background_investigation_results = []

    if SELECTED_SEARCH_ENGINE == SearchEngine.TAVILY.value:
        searched_content = LoggedTavilySearch(
            max_results = configurable.max_search_results
        ).invoke(query)
    
        if isinstance(searched_content, tuple):
            searched_content = searched_content[0]
        
        if isinstance(searched_content,str):
            try:
                parsed = json.loads(searched_content)
                if isinstance(parsed, dict) and "error" in parsed:
                    logger.error(f"Tavily search error: {parsed['error']}")
                    background_investigation_results = []
                elif isinstance(parsed, list):
                    background_investigation_results = [
                        f"##{elem.get('title', 'Untitled')}\n\n{elem.get('content','No content')}"
                        for elem in parsed
                    ]
                else:
                    logger.error(f"Unexpected Tavily response format: {searched_content}")
                    background_investigation_results = []
            except json.JSONDecodeError:
                logger.error(f"Failed to parse Tavily response as JSON: {searched_content}")
                background_investigation_results = []


        elif isinstance(searched_content, list):
            background_investigation_results = [
                f"## {elem.get('title', 'Untitled')}\n\n{elem.get('content', '')}"
                for elem in searched_content
            ]
        else:
            logger.error(
                f"Tavily search returned malformed response: {searched_content}"
            )
            background_investigation_results = []
        
    else:
        background_investigation_results = get_web_search_tool(
            configurable.max_search_results
        ).invoke(query)
    
    return {
        "background_investigation_results": json.dumps(
            background_investigation_results, ensure_ascii = False
        )
    }


