"""
AstroNet: Astrocyte-Inspired Working Memory Layer for LLMs

A biologically-inspired persistent memory module that wraps around frozen LLMs,
using calcium (Ca2+) dynamics to modulate attention across context windows.
"""

__version__ = "0.1.0"

from astronet.calcium import AstroStateV0
from astronet.hooks import AstroWrappedModel, install_astro_hooks
