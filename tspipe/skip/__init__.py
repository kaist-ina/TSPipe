"""Supports efficiency with skip connections."""
from tspipe.skip.namespace import Namespace
from tspipe.skip.skippable import pop, skippable, stash, verify_skippables

__all__ = ['skippable', 'stash', 'pop', 'verify_skippables', 'Namespace']
