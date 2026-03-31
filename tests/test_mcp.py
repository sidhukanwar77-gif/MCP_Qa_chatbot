
from src.mcp_server import mcp
print('Server name:', mcp.name)
print('Server created successfully!')
print('Tools registered:', list(mcp._tool_manager._tools.keys()))

# Expected output:
# Server name: rag-memory-server
# Server created successfully!
# Tools registered: ['search_documents', 'ask_question', 'save_memory', 'get_memory']


"""Test MCP Client connecting to MCP Server."""

import asyncio
import sys
sys.path.insert(0, '.')
from src.mcp_client import MCPClient

async def test_mcp():
    print('=== Testing MCP Client + Server ===')
    print()

    client = MCPClient()
    await client.connect()

    # List available tools
    tools = await client.list_tools()
    print('Available tools:')
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description'][:60]}...")
    print()

    # Call a tool
    print('Calling ask_question...')
    result = await client.call_tool('ask_question', {
        'question': 'How many leave days?',
        'session_id': 'test',
    })
    print(f'Result: {result}')

    print('\n=== MCP Test PASSED ===')

if __name__ == '__main__':
    asyncio.run(test_mcp())
