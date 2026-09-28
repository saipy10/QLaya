"""MCP (Model Context Protocol) stdio server for QLaya (optional extra).

Install the extra to use it:

    pip install "qlaya[mcp]"

Then run the server:

    qlaya-mcp-server            # console script
    python -m qlaya.mcp.server  # equivalent module form

The submodules ``device`` and ``tools`` are importable without the ``mcp``
dependency; ``server`` requires the extra.
"""

__all__ = ["device", "server", "tools"]
