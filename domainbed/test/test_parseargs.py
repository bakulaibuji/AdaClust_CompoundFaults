import unittest
from domainbed.scripts import train
from pretty_simple_namespace import pprint


class TestParseArgs(unittest.TestCase):
    args = train.getParameters()
    pprint(args)
