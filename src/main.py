"""
Entry point for the Customer Support AI application.

This module creates the Tkinter root window and launches the graphical
interface. Keeping the entry point small makes the rest of the project easier
to import, test, and reuse.
"""

from gui import start_gui


if __name__ == "__main__":
    start_gui()
