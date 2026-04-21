import sys
from unittest.mock import MagicMock

# Mock the agent module and its internal MemoryProvider class 
# so that source/__init__.py can import without throwing ModuleNotFoundError
mock_agent = MagicMock()
mock_agent.memory_provider.MemoryProvider = MagicMock
sys.modules['agent'] = mock_agent
sys.modules['agent.memory_provider'] = mock_agent.memory_provider
