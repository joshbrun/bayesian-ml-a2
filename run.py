# coding=utf-8

"""
SE755 - A2
The run script for SE755 A2

Authors:
Joshua Brundan
Kevin Hira
"""

from src.common.commandinterface import CommandInterface

# Accept user arguments
user_interface = CommandInterface()
options = user_interface.get_options()

if options is not None:
    pass
