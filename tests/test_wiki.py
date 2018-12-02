# pylint: disable=E1120
# pylint: disable=W0614

from kozinak import *
import xml.etree.ElementTree as etree
import io
import os
from tempfile import TemporaryFile
from neo4j import GraphDatabase

def test_wiki_dump_to_neo4j():
    addr = 'bolt://localhost:7687'
    test_data_path = './tests/test_wiki/wiki_dump.xml'
    test_relationships = {
        ('Anarchism', 'Peter_Kropotkin'),
        ('Anarchism', 'Edna_St._Vincent_Millay'),
        ('Anarchism', 'William_Godwin'),
        ('Anarchism', 'Bavarian_Soviet_Republic'),
        ('Anarchism', 'Political_philosophy'),
        ('Anarchism', 'List_of_anarchist_communities'),
        ('Anarchism', 'Anarchism_and_violence'),
        ('Anarchism', 'Lois_scélérates'),
        ('Anarchism', 'Routledge_Encyclopedia_of_Philosophy'),
        ('AccessibleComputing', 'Computer_accessibility'),
        ('Anarchism', 'Category:Far-left_politics'),
        ('Anarchism', 'Kropotkin_1898'),
        ('Anarchist', 'Anarchism'),
    }

    with GraphDatabase.driver(addr) as driver:
        with driver.session() as session:
            io_wiki_dump_to_neo4j(addr, test_data_path, auth=None)

            def check(from_, to_):
                query = """
                    MATCH (a1:WikiArticle)-[:LINK]->(a2:WikiArticle) 
                    WHERE a1.name = UPPER($n1) AND a2.name = UPPER($n2)
                    RETURN EXISTS((a1)-[:LINK]->(a2))
                    """

                res = session.run(query, n1 = from_, n2 = to_).single()
                assert (res != None and res[0])

            any(check(*x) for x in test_relationships)

