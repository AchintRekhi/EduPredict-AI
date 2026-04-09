"""Vercel WSGI entry point for Flask app."""
import sys
import os

# Add parent directory to path so we can import webapp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webapp.app import app

# Export for Vercel
__all__ = ['app']
