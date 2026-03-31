"""
mcp_client.py - The Remote Control (MCP Client)

WHAT IT DOES:
  Connects to the MCP Server and uses its tools.
  This is how external applications interact with our system.

ANALOGY:
  A TV remote control. It connects to the TV (server)
  and lets you press buttons (call tools).
"""

import asyncio  # For async operations
import json     # For parsing responses

from mcp import ClientSession  # MCP Client
from mcp.client.stdio import stdio_client, StdioServerParameters  # Communication


class MCPClient:
    """
    MCP Client that connects to our server and calls tools.
    """

    def __init__(self, server_script: str = 'src/mcp_server.py'):
        """
        Set up the client.
        
        Args:
            server_script: Path to the MCP server file
        """
        self.server_script = server_script
        self.session = None

    async def connect(self):
        """
        Connect to the MCP Server.
        
        This starts the server as a subprocess and creates
        a communication channel between client and server.
        """
        # Define how to start the server
        server_params = StdioServerParameters(
            command='python',              # Run with Python
            args=[self.server_script],      # The server file
        )

        # Create the connection
        # stdio_client handles the communication pipe
        self._transport = await stdio_client(server_params).__aenter__()
        read_stream, write_stream = self._transport

        # Create a session (like opening a phone line)
        self.session = await ClientSession(
            read_stream, write_stream
        ).__aenter__()

        # Initialize the session (handshake)
        await self.session.initialize()
        print('Connected to MCP Server!')

    async def list_tools(self):
        """List all tools the server offers."""
        if not self.session:
            await self.connect()

        result = await self.session.list_tools()
        tools = []
        for tool in result.tools:
            tools.append({
                'name': tool.name,
                'description': tool.description,
            })
        return tools

    async def call_tool(self, tool_name: str, arguments: dict):
        """
        Call a specific tool on the server.
        
        Args:
            tool_name: Which tool to call (e.g., 'ask_question')
            arguments: The inputs for the tool (e.g., {'question': 'How many...'})
        """
        if not self.session:
            await self.connect()

        result = await self.session.call_tool(tool_name, arguments)
        return result.content[0].text if result.content else 'No response'
