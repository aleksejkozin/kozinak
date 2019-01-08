# pylint: disable=E1120
# pylint: disable=W0614

from kozinak import *
import xml.etree.ElementTree as etree
import io
import os
import time
import pytest
import csv
from tempfile import TemporaryFile
from neo4j import GraphDatabase
from itertools import chain

def test_file_sorting(tmpdir):
    source_path = './tests/test_wiki/sort_test1.data'
    answer_path = './tests/test_wiki/sort_test2.data'
    data_path = tmpdir / 'test.data'
    sort_file_io(source_path, data_path)
    answer_data = open(answer_path, 'r').read()
    data = open(data_path, 'r').read()
    assert answer_data.strip(' \n\r') == data.strip(' \n\r'), 'file sort broken'

def test_merge_csv(tmpdir):
    source_path = './tests/test_wiki/merge1'
    answer_path = './tests/test_wiki/merge2'
    data_path = tmpdir / 'test.data'
    merge_csv_io(source_path, data_path)
    answer_data = open(answer_path, 'r').read()
    data = open(data_path, 'r').read()
    assert answer_data.strip(' \n\r') == data.strip(' \n\r'), 'csv merge broken'

def test_clear_csv(tmpdir):
    source_path = './tests/test_wiki/clear1'
    answer_path = './tests/test_wiki/clear2'
    data_path = tmpdir / 'test.data'
    clear_csv_io(source_path, data_path)
    answer_data = open(answer_path, 'r').read()
    data = open(data_path, 'r').read()
    assert answer_data.strip(' \n\r') == data.strip(' \n\r'), 'csv merge broken'

# @pytest.mark.skip(reason="no way of currently testing this")
def test_wiki_dump_to_neo4j(tmpdir, benchmark):
    test_data_path = './tests/test_wiki/wiki_dump.xml'
    nodes_result_path = tmpdir / 'nodes.csv'
    relationships_result_path = tmpdir / 'relationships.csv'

    test_nodes = [
        # General
        dict(id='Albert_Einstein', labels={'WikiPage', 'Person', 'Scientist'}),
        dict(id='Aristotle', labels={'WikiPage', 'Person', 'Philosopher'}),
        dict(id='Andre_Agassi', labels={'WikiPage', 'Person', 'Alive'}),
        dict(id='Animal_Farm', labels={'WikiPage', 'Book'}),   
        dict(id='Peter_Kropotkin', labels={'WikiPage'}),

        # Redirect
        dict(id='AccessibleComputing', labels={'WikiPage'}),
        dict(id='Anarchist', labels={'WikiPage'}),
        dict(id='Anarchism', labels={'WikiPage'}),

        # Authors
        dict(id='Jon_Postel', labels={'Person'}),

        # Editor
        dict(id='Edward_N._Zalta', labels={'Person'}),
    ]

    test_relationships = [
        # General
        dict(source='Anarchism', label='LINK', target='Peter_Kropotkin'),
        dict(source='Anarchism', label='LINK', target='Edna_St._Vincent_Millay'),
        dict(source='Anarchism', label='LINK', target='William_Godwin'),
        dict(source='Anarchism', label='LINK', target='Bavarian_Soviet_Republic'),
        dict(source='Anarchism', label='LINK', target='Political_philosophy'),
        # dict(source='Anarchism', label='LINK', target='List_of_anarchist_communities'),
        dict(source='Anarchism', label='LINK', target='Anarchism_and_violence'),
        dict(source='Anarchism', label='LINK', target='Lois_scélérates'),
        dict(source='Anarchism', label='LINK', target='Routledge_Encyclopedia_of_Philosophy'),
        dict(source='Anarchism', label='LINK', target='Category:Far-left_politics'),
        dict(source='Anarchism', label='LINK', target='Anarchists_(disambiguation)'),
        dict(source='Albert_Einstein', label='LINK', target='University_of_Zurich'),
        dict(source='Albert_Einstein', label='LINK', target='Ernst_G._Straus'),
        dict(source='Albert_Einstein', label='LINK', target='Mileva_Marić'),

        # Redirect
        dict(source='Anarchist', label='LINK', target='Anarchism'),
        dict(source='AccessibleComputing', label='LINK', target='Computer_accessibility'),

        # Authors
        dict(source='Animal_Farm', label='LINK', target='Richard_Lacayo'),

        # Editor
        dict(source='Ayn_Rand', label='LINK', target='Edward_N._Zalta'),
    ]

    benchmark(
        wiki_dump_to_csv_io,
        test_data_path, 
        nodes_result_path, 
        relationships_result_path
        )

    def check_node(node):
        with open(nodes_result_path, 'r') as f:
            assert pipe(
                csv.reader(f),
                map_(
                    lambda id, labels: 
                        id == node['id']
                        and 
                        node['labels'].issubset(set(labels.split(';')))
                        ),
                any
                )
    pipe(
        test_nodes,
        map(check_node),
        forcemap
        )

    def check_relationships(rel):
        with open(relationships_result_path, 'r') as f:
            assert pipe(
                csv.reader(f),
                map_(
                    lambda s, t, l: 
                        s == rel['source']
                        and 
                        t == rel['target']
                        and 
                        l == rel['label'] 
                        ),
                any
                )
    pipe(
        test_relationships,
        map(check_relationships),
        forcemap
        )
