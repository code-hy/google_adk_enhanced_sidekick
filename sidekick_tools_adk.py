# sidekick_tools_adk.py
from playwright.async_api import async_playwright
from dotenv import load_dotenv
import os
import requests
import aiohttp
import aiofiles
from pathlib import Path
import json
from typing import Any, Dict, List, Callable
import asyncio
import yaml
import subprocess
import sys
import platform

load_dotenv(override=True)

pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"

SANDBOX_DIR = Path("sandbox")
SANDBOX_DIR.mkdir(exist_ok=True)

# Simple MCP Client Manager (placeholder - remove problematic MCP imports)
class MCPClientManager:
    def __init__(self):
        self.sessions = {}
    
    async def disconnect_all(self):
        """Disconnect all MCP servers"""
        self.sessions.clear()

# Initialize MCP manager
mcp_manager = MCPClientManager()

# Try different import approaches for Google ADK tools
try:
    # First try the direct import
    from google.adk.tools import Tool
except ImportError:
    try:
        # Try alternative import paths
        from google.adk import Tool
    except ImportError:
        try:
            # Try importing from a submodule
            from google.adk.tools.tool import Tool
        except ImportError:
            # If all else fails, we'll need to use a decorator approach
            Tool = None
            print("Warning: Could not import Tool from Google ADK. Using decorator approach instead.")

# Define tools using Google ADK's Tool class or decorator
def send_notification(text: str) -> str:
    """Send free notification to Telegram (no cost, no app install needed).
    
    Setup:
    1. Message @BotFather on Telegram, type /newbot
    2. Message @userinfobot to get your TELEGRAM_CHAT_ID
    3. Add to .env: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        return "⚠️ Skipped: Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env"
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        response = requests.post(
            url,
            json={"chat_id": chat_id, "text": text},
            timeout=10
        )
        response.raise_for_status()
        return "✅ Telegram notification sent!"
    except Exception as e:
        return f"❌ Telegram error: {str(e)[:100]}"

def search_web(query: str) -> str:
    """Search the web using Google Serper API.
    
    Args:
        query: Search query string
        
    Returns:
        Search results as formatted string
    """
    try:
        # Simple implementation - you can replace with actual search API
        return f"Web search for: {query}\n[This would call a search API in production]"
    except Exception as e:
        return f"Search error: {str(e)}"

def wikipedia_lookup(query: str) -> str:
    """Lookup information on Wikipedia.
    
    Args:
        query: Topic to search on Wikipedia
        
    Returns:
        Wikipedia article summary
    """
    try:
        # Simple implementation
        return f"Wikipedia lookup for: {query}\n[This would call Wikipedia API in production]"
    except Exception as e:
        return f"Wikipedia error: {str(e)}"

def execute_python(code: str) -> str:
    """Execute Python code in a REPL environment.
    
    Args:
        code: Python code to execute. Include print() statements for output.
        
    Returns:
        Output of the code execution
    """
    from io import StringIO
    import sys
    import traceback
    
    old_cwd = os.getcwd()
    os.chdir(SANDBOX_DIR)
    
    try:
        output = StringIO()
        sys.stdout = output
        exec(code, {"__builtins__": __builtins__}, {})
        sys.stdout = sys.__stdout__
        result = output.getvalue()
        return result if result else "Code executed (no output)"
    except Exception as e:
        sys.stdout = sys.__stdout__
        return f"Error: {str(e)}\n{traceback.format_exc()}"
    finally:
        os.chdir(old_cwd)

def write_file(filename: str, content: str) -> str:
    """Write content to a file in the sandbox directory.
    
    Args:
        filename: Name of the file to create
        content: Content to write
        
    Returns:
        Success message with file path
    """
    try:
        file_path = SANDBOX_DIR / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"File written: {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def read_file(filename: str) -> str:
    """Read content from a file in the sandbox directory.
    
    Args:
        filename: Name of the file to read
        
    Returns:
        File contents or error message
    """
    try:
        file_path = SANDBOX_DIR / filename
        if not file_path.exists():
            return f"File not found: {filename}"
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def list_files() -> str:
    """List all files in the sandbox directory.
    
    Returns:
        List of files
    """
    files = [f.name for f in SANDBOX_DIR.iterdir() if f.is_file()]
    return "\n".join(files) if files else "No files in sandbox"

async def call_mcp_tool(server_name: str, tool_name: str, arguments: dict) -> str:
    """Call a tool from an MCP (Model Control Protocol) server.
    
    Args:
        server_name: Name of the MCP server
        tool_name: Name of the tool to call
        arguments: Arguments to pass to the tool
        
    Returns:
        Tool execution result
    """
    try:
        # Placeholder implementation
        return f"MCP tool {tool_name} from {server_name} would be called with {arguments}"
    except Exception as e:
        return f"MCP Error: {str(e)}"

def openapi_tool_executor(openapi_spec_url: str, operation_id: str, parameters: dict) -> str:
    """Execute a tool from an OpenAPI specification.
    
    Args:
        openapi_spec_url: URL to OpenAPI spec
        operation_id: Operation to execute
        parameters: Parameters for the operation
        
    Returns:
        API response
    """
    try:
        # Placeholder implementation
        return f"OpenAPI operation {operation_id} would be called with {parameters}"
    except Exception as e:
        return f"OpenAPI Error: {str(e)}"

async def get_browser_tools() -> tuple[List[Any], Any, Any]:
    """Initialize Playwright browser and return tools"""
    # Check if we're on Windows and handle Playwright accordingly
    is_windows = platform.system() == "Windows"
    
    if is_windows:
        print("Warning: Running on Windows. Browser tools may not work properly.")
        # Create dummy browser tools for Windows
        async def browser_navigate(url: str) -> str:
            return f"Browser navigation to {url} is not available on Windows in this environment."
        
        async def browser_click(selector: str) -> str:
            return f"Browser click on {selector} is not available on Windows in this environment."
        
        async def browser_get_content() -> str:
            return "Browser content extraction is not available on Windows in this environment."
        
        # Create tools based on what's available in the Google ADK
        if Tool is not None:
            # Use Tool.from_function if available
            browser_tools = [
                Tool.from_function(browser_navigate),
                Tool.from_function(browser_click),
                Tool.from_function(browser_get_content)
            ]
        else:
            # Fall back to using the functions directly
            browser_tools = [browser_navigate, browser_click, browser_get_content]
        
        return browser_tools, None, None
    
    # Non-Windows code path
    try:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)  # Changed to headless for stability
        
        async def browser_navigate(url: str) -> str:
            """Navigate browser to a URL.
            
            Args:
                url: URL to navigate to
                
            Returns:
                Success message or page title
            """
            try:
                page = await browser.new_page()
                await page.goto(url)
                title = await page.title()
                await page.close()
                return f"Navigated to {url}. Page title: {title}"
            except Exception as e:
                return f"Browser navigation error: {str(e)}"
        
        async def browser_click(selector: str) -> str:
            """Click an element on the page.
            
            Args:
                selector: CSS selector for element to click
                
            Returns:
                Success message
            """
            try:
                page = browser.contexts[0].pages[-1] if browser.contexts else await browser.new_page()
                await page.click(selector)
                return f"Clicked element: {selector}"
            except Exception as e:
                return f"Click error: {str(e)}"
        
        async def browser_get_content() -> str:
            """Get page content as text.
            
            Returns:
                Page text content
            """
            try:
                page = browser.contexts[0].pages[-1] if browser.contexts else await browser.new_page()
                content = await page.content()
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')
                return soup.get_text()[:2000]
            except Exception as e:
                return f"Content extraction error: {str(e)}"
        
        # Create tools based on what's available in the Google ADK
        if Tool is not None:
            # Use Tool.from_function if available
            browser_tools = [
                Tool.from_function(browser_navigate),
                Tool.from_function(browser_click),
                Tool.from_function(browser_get_content)
            ]
        else:
            # Fall back to using the functions directly
            browser_tools = [browser_navigate, browser_click, browser_get_content]
        
        return browser_tools, browser, playwright
    except Exception as e:
        print(f"Error initializing browser tools: {e}")
        # Fall back to dummy tools
        async def browser_navigate(url: str) -> str:
            return f"Browser navigation to {url} failed with error: {str(e)}"
        
        async def browser_click(selector: str) -> str:
            return f"Browser click on {selector} failed with error: {str(e)}"
        
        async def browser_get_content() -> str:
            return f"Browser content extraction failed with error: {str(e)}"
        
        # Create tools based on what's available in the Google ADK
        if Tool is not None:
            # Use Tool.from_function if available
            browser_tools = [
                Tool.from_function(browser_navigate),
                Tool.from_function(browser_click),
                Tool.from_function(browser_get_content)
            ]
        else:
            # Fall back to using the functions directly
            browser_tools = [browser_navigate, browser_click, browser_get_content]
        
        return browser_tools, None, None

async def get_all_tools() -> tuple[List[Any], Any, Any]:
    """Get all tools including browser and utility tools"""
    browser_tools, browser, playwright = await get_browser_tools()
    
    # Create tools based on what's available in the Google ADK
    if Tool is not None:
        # Use Tool.from_function if available
        utility_tools = [
            Tool.from_function(send_notification),
            Tool.from_function(search_web),
            Tool.from_function(wikipedia_lookup),
            Tool.from_function(execute_python),
            Tool.from_function(write_file),
            Tool.from_function(read_file),
            Tool.from_function(list_files),
            Tool.from_function(call_mcp_tool),
            Tool.from_function(openapi_tool_executor)
        ]
    else:
        # Fall back to using the functions directly
        utility_tools = [
            send_notification,
            search_web,
            wikipedia_lookup,
            execute_python,
            write_file,
            read_file,
            list_files,
            call_mcp_tool,
            openapi_tool_executor
        ]
    
    return browser_tools + utility_tools, browser, playwright