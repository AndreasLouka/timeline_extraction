#!/usr/bin/env python

#
# Generated  by generateDS.py.
#
# Command line options:
#   ('--no-dates', '')
#   ('--no-versions', '')
#   ('--silence', '')
#   ('--member-specs', 'list')
#   ('-f', '')
#   ('-o', 'tests/simpletype_memberspecs2_sup.py')
#   ('-s', 'tests/simpletype_memberspecs2_sub.py')
#   ('--super', 'simpletype_memberspecs2_sup')
#
# Command line arguments:
#   tests/simpletype_memberspecs.xsd
#
# Command line:
#   generateDS.py --no-dates --no-versions --silence --member-specs="list" -f -o "tests/simpletype_memberspecs2_sup.py" -s "tests/simpletype_memberspecs2_sub.py" --super="simpletype_memberspecs2_sup" tests/simpletype_memberspecs.xsd
#
# Current working directory (os.getcwd()):
#   generateds
#

import sys
from lxml import etree as etree_

import simpletype_memberspecs2_sup as supermod

def parsexml_(infile, parser=None, **kwargs):
    if parser is None:
        # Use the lxml ElementTree compatible parser so that, e.g.,
        #   we ignore comments.
        parser = etree_.ETCompatXMLParser()
    doc = etree_.parse(infile, parser=parser, **kwargs)
    return doc

#
# Globals
#

ExternalEncoding = 'ascii'

#
# Data representation classes
#


class SpecialDateSub(supermod.SpecialDate):
    def __init__(self, SpecialProperty=None, valueOf_=None):
        super(SpecialDateSub, self).__init__(SpecialProperty, valueOf_, )
supermod.SpecialDate.subclass = SpecialDateSub
# end class SpecialDateSub


class ExtremeDateSub(supermod.ExtremeDate):
    def __init__(self, ExtremeProperty=None, valueOf_=None):
        super(ExtremeDateSub, self).__init__(ExtremeProperty, valueOf_, )
supermod.ExtremeDate.subclass = ExtremeDateSub
# end class ExtremeDateSub


def get_root_tag(node):
    tag = supermod.Tag_pattern_.match(node.tag).groups()[-1]
    rootClass = None
    rootClass = supermod.GDSClassesMapping.get(tag)
    if rootClass is None and hasattr(supermod, tag):
        rootClass = getattr(supermod, tag)
    return tag, rootClass


def parse(inFilename, silence=False):
    parser = None
    doc = parsexml_(inFilename, parser)
    rootNode = doc.getroot()
    rootTag, rootClass = get_root_tag(rootNode)
    if rootClass is None:
        rootTag = 'SpecialDate'
        rootClass = supermod.SpecialDate
    rootObj = rootClass.factory()
    rootObj.build(rootNode)
    # Enable Python to collect the space used by the DOM.
    doc = None
##     if not silence:
##         sys.stdout.write('<?xml version="1.0" ?>\n')
##         rootObj.export(
##             sys.stdout, 0, name_=rootTag,
##             namespacedef_='xmlns:pl="http://kuhlman.com/people.xsd"',
##             pretty_print=True)
    return rootObj


def parseEtree(inFilename, silence=False):
    parser = None
    doc = parsexml_(inFilename, parser)
    rootNode = doc.getroot()
    rootTag, rootClass = get_root_tag(rootNode)
    if rootClass is None:
        rootTag = 'SpecialDate'
        rootClass = supermod.SpecialDate
    rootObj = rootClass.factory()
    rootObj.build(rootNode)
    # Enable Python to collect the space used by the DOM.
    doc = None
    mapping = {}
    rootElement = rootObj.to_etree(None, name_=rootTag, mapping_=mapping)
    reverse_mapping = rootObj.gds_reverse_node_mapping(mapping)
##     if not silence:
##         content = etree_.tostring(
##             rootElement, pretty_print=True,
##             xml_declaration=True, encoding="utf-8")
##         sys.stdout.write(content)
##         sys.stdout.write('\n')
    return rootObj, rootElement, mapping, reverse_mapping


def parseString(inString, silence=False):
    from StringIO import StringIO
    parser = None
    doc = parsexml_(StringIO(inString), parser)
    rootNode = doc.getroot()
    rootTag, rootClass = get_root_tag(rootNode)
    if rootClass is None:
        rootTag = 'SpecialDate'
        rootClass = supermod.SpecialDate
    rootObj = rootClass.factory()
    rootObj.build(rootNode)
    # Enable Python to collect the space used by the DOM.
    doc = None
##     if not silence:
##         sys.stdout.write('<?xml version="1.0" ?>\n')
##         rootObj.export(
##             sys.stdout, 0, name_=rootTag,
##             namespacedef_='xmlns:pl="http://kuhlman.com/people.xsd"')
    return rootObj


def parseLiteral(inFilename, silence=False):
    parser = None
    doc = parsexml_(inFilename, parser)
    rootNode = doc.getroot()
    rootTag, rootClass = get_root_tag(rootNode)
    if rootClass is None:
        rootTag = 'SpecialDate'
        rootClass = supermod.SpecialDate
    rootObj = rootClass.factory()
    rootObj.build(rootNode)
    # Enable Python to collect the space used by the DOM.
    doc = None
##     if not silence:
##         sys.stdout.write('#from simpletype_memberspecs2_sup import *\n\n')
##         sys.stdout.write('import simpletype_memberspecs2_sup as model_\n\n')
##         sys.stdout.write('rootObj = model_.rootClass(\n')
##         rootObj.exportLiteral(sys.stdout, 0, name_=rootTag)
##         sys.stdout.write(')\n')
    return rootObj


USAGE_TEXT = """
Usage: python ???.py <infilename>
"""


def usage():
    print(USAGE_TEXT)
    sys.exit(1)


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        usage()
    infilename = args[0]
    parse(infilename)


if __name__ == '__main__':
    #import pdb; pdb.set_trace()
    main()
