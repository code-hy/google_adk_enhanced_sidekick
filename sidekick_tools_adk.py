from playwright.async_api import async_playwright
from google.adk.tools import Tool, tool
from dotenv import load_dotenv
import os
import requests
import aiofiles
from pathlib import Path
import json
from typing import Any, Dict, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
import yaml

load_dotenv(override=True)

pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"

SANDBOX_DIR = Path("sandbox")
SANDBOX_DIR.mkdir(exist_ok=True)

# MCP Client Manager
class MCPClientManager:
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.processes: Dict[str, Any] = {}
    
    async def connect(self, server_name: str, command: str, args: List[str]):
        """Connect to MCP server"""
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=None
        )
        
        stdio_transport = stdio_client(server_params)
        self.processes[server_name] = stdio_transport
        
        async with stdio_transport as (read, write):
            session = ClientSession(read, write)
            await session.initialize()
            self.sessions[server_name] = session
            logger.info(f"Connected to MCP server: {server_name}")
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool from MCP server"""
        if server_name not in self.sessions:
            raise ValueError(f"MCP server {server_name} not connected")
        
        session = self.sessions[server_name]
        result = await session.call_tool(tool_name, arguments)
        return result
    
    async def disconnect_all(self):
        """Disconnect all MCP servers"""
        for session in self.sessions.values():
            await session.exit()
        self.sessions.clear()
        self.processes.clear()

# Initialize MCP manager
mcp_manager = MCPClientManager()

# Enhanced tools with observability
@tool(name="send_push_notification")
def send_push_notification(text: str) -> str:
    """Send a push notification to the user via Pushover.
    
    Args:
        text: The message text to send
        
    Returns:
        Success message or error
    """
    try:
        requests.post(
            pushover_url,
            data={"token": pushover_token, "user": pushover_user, "message": text}
        )
        return "Push notification sent successfully"
    except Exception as e:
        return f"Failed to send notification: {str(e)}"

@tool(name="search_web")
def search_web(query: str) -> str:
    """Search the web using Google Serper API.
    
    Args:
        query: Search query string
        
    Returns:
        Search results as formatted string
    """
    from google_serper import GoogleSerperAPIWrapper
    
    serper = GoogleSerperAPIWrapper()
    results = serper.run(query)
    return results

@tool(name="wikipedia_lookup")
def wikipedia_lookup(query: str) -> str:
    """Lookup information on Wikipedia.
    
    Args:
        query: Topic to search on Wikipedia
        
    Returns:
        Wikipedia article summary
    """
    from wikipedia import WikipediaAPIWrapper
    
    wikipedia = WikipediaAPIWrapper()
    return wikipedia.run(query)

@tool(name="execute_python")
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

@tool(name="write_file")
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

@tool(name="read_file")
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

@tool(name="list_files")
def list_files() -> str:
    """List all files in the sandbox directory.
    
    Returns:
        List of files
    """
    files = [f.name for f in SANDBOX_DIR.iterdir() if f.is_file()]
    return "\n".join(files) if files else "No files in sandbox"

@tool(name="mcp_tool_caller")
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
        result = await mcp_manager.call_tool(server_name, tool_name, arguments)
        return str(result)
    except Exception as e:
        return f"MCP Error: {str(e)}"

@tool(name="openapi_tool_executor")
def openapi_tool_executor(
    openapi_spec_url: str,
    operation_id: str,
    parameters: dict
) -> str:
    """Execute a tool from an OpenAPI specification.
    
    Args:
        openapi_spec_url: URL to OpenAPI spec
        operation_id: Operation to execute
        parameters: Parameters for the operation
        
    Returns:
        API response
    """
    import httpx
    from openapi3 import OpenAPI
    
    try:
        # Fetch and parse spec
        spec = OpenAPI(httpx.get(openapi_spec_url).json())
        operation = spec.spec["paths"][parameters["path"]][parameters["method"]]
        
        # Execute request
        client = httpx.Client()
        response = client.request(
            method=parameters["method"].upper(),
            url=f"{spec.spec['servers'][0]['url']}{parameters['path']}",
            params=parameters.get("query", {}),
            json=parameters.get("body", {})
        )
        
        return response.json()
    except Exception as e:
        return f"OpenAPI Error: {str(e)}"

async def get_browser_tools() -> tuple[List[Tool], Any, Any]:
    """Initialize Playwright browser and return tools"""
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    
    @tool(name="browser_navigate")
    async def browser_navigate(url: str) -> str:
        """Navigate browser to a URL.
        
        Args:
            url: URL to navigate to
            
        Returns:
            Success message or page title
        """
        page = await browser.new_page()
        await page.goto(url)
        return f"Navigated to {url}. Page title: {await page.title()}"
    
    @tool(name="browser_click")
    async def browser_click(selector: str) -> str:
        """Click an element on the page.
        
        Args:
            selector: CSS selector for element to click
            
        Returns:
            Success message
        """
        page = browser.contexts[0].pages[-1]
        await page.click(selector)
        return f"Clicked element: {selector}"
    
    @tool(name="browser_get_content")
    async def browser_get_content() -> str:
        """Get page content as text.
        
        Returns:
            Page text content
        """
        page = browser.contexts[0].pages[-1]
        content = await page.content()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        return soup.get_text()[:2000]
    
    browser_tools = [
        browser_navigate,
        browser_click,
        browser_get_content
    ]
    
    return browser_tools, browser, playwright

async def initialize_mcp_servers():
    """Initialize MCP servers from configuration"""
    mcp_config = os.getenv("MCP_CONFIG_PATH", "./mcp_servers.yaml")
    
    if os.path.exists(mcp_config):
        with open(mcp_config, 'r') as f:
            config = yaml.safe_load(f)
        
        for server_name, server_info in config.get("servers", {}).items():
            await mcp_manager.connect(
                server_name,
                server_info["command"],
                server_info.get("args", [])
            )
            logger.info(f"Initialized MCP server: {server_name}")

async def get_all_tools() -> tuple[List[Tool], Any, Any]:
    """Get all tools including browser and utility tools"""
    browser_tools, browser, playwright = await get_browser_tools()
    await initialize_mcp_servers()
    
    utility_tools = [
        send_push_notification,
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
