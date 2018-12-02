# pylint: disable=E1120

import re
from pathlib import Path
from functools import reduce
from itertools import islice, chain
from toolz import compose, curry, pipe
from toolz.itertoolz import get
from toolz.curried import map
from tqdm import tqdm

import xml.etree.ElementTree as etree
from neo4j import GraphDatabase

reduce = curry(reduce)
get = curry(get)
filter = curry(filter)

def last_non_empty(lst):
    return next(x for x in reversed(lst) if x)

@curry
def batch(size, iterable):
    sourceiter = iter(iterable)
    while True:
        batchiter = list(islice(sourceiter, size))
        if not batchiter:
            break
        yield batchiter

@curry
def withf(something, f):
    with something as x:
        return f(x)

@curry
def wopen(t, path):
    return withf(open(path, t)) 

wopenr = wopen('r')
wopenw = wopen('w')

def fin(before, after):
    try:
        before()
    except etree.ParseError as e:
        print(e)
    finally:
        after()

@curry
def wprogb(total):
    return withf(tqdm(total=total, mininterval=1.0))

def filesize(path):
    return Path(path).stat().st_size

def io_read_xml(path):
    return etree.iterparse(path)

@curry
def io_report_progress(pb, fd, x):
    pb.update(fd.tell() - pb.n)
    return x

def tag(x): 
    return x.tag.split('}')[1]

def xml_to_dict(root):
    return {
        **{tag(x): xml_to_dict(x) for x in list(root)}, 
        **{'body': root.text, 'attributes': root.attrib}
        }

# __patterns = [
#     r'\[\[.*?\|(.*?)\]\]', 
#     r'\[\[(.*?)\]\]', 
#     r'{{.*?\|(.*?)}}',
# ]
# rlink = re.compile(f"(?:{'|'.join(__patterns)})")
# extract_links_from = compose(
#     map(lambda x: x.replace(' ', '_')),
#     map(last_non_empty),
#     rlink.findall,
#     )

rlinka = re.compile(r"\[\[([^\[]*?)\]\]")
rlinkb = re.compile(r"{{([^{]*?)}}")
def extract_links_from(text):
    def proc_link(x):
        data = x.split('|')
        if data[0].startswith('File:'):
            return
        return (True, data[0])

    def proc_pattern(x):
        data = x.split('|')
        t = data[0].lower().strip()
        if t == 'cite web':
            return []
        elif t == 'cite book':
            return []
        elif t == 'cite journal':
            return []
        elif t == 'cite news':
            return []
        elif t == 'main':
            return []
        elif t == 'cite encyclopedia':
            return []
        elif t == 'isbn':
            return []
        elif t == 'rp':
            return []
        elif t == 'redirect2':
            return [(False, v) for v in data[1:]]
        elif t == 'sfn':
            if len(data) >= 3:
                return [(True, f'{data[1]}_{data[2]}')]
            else:
                return [(True, data[1])]
        elif t == 'see also':
            return [(True, data[1])]
        return []

    extract_links = lambda x: chain(
        map(proc_link, rlinka.findall(x)), 
        chain.from_iterable(map(proc_pattern, rlinkb.findall(x)))
        )
    return pipe(
        text,
        extract_links,
        filter(lambda x: x != None),
        map(lambda x: (x[0], x[1].replace(' ', '_').split('#')[0]))
        )

def escape(x):
    return x.replace("'", "")

def is_page(x):
    event, element = x
    return event == 'end' and tag(element) == 'page'

@curry  
def process_page(x):
    _, element = x
    
    page = xml_to_dict(element)
    text = page['revision']['text']['body']

    # normalize = lambda x: str.upper(x)
    title = page['title']['body'].replace(' ', '_')
    links = extract_links_from(text)

    return (title, links)

@curry 
def send_batch_to_db(session, batch):
    def to_link(y, x):
        if x[0]:
            return {'n1': y, 'n2': x[1]}
        else:
            return {'n1': x[1], 'n2': y}

    batch = chain.from_iterable(
        [to_link(x[0], y) for y in x[1]]
        for x in batch
    )

    session.run("""
            WITH $batch AS batch
            UNWIND batch AS link
            MERGE (a1:WikiArticle { name: UPPER(link.n1) }) 
            MERGE (a2:WikiArticle { name: UPPER(link.n2) })
            MERGE (a1)-[:LINK]->(a2)""",
            batch = list(batch)
            )   

def create_constraint(session):
    session.run("CREATE CONSTRAINT ON (a:WikiArticle) ASSERT a.name IS UNIQUE")

def create_session(driver):
    session = driver.session()
    create_constraint(session)
    return session

@curry  
def io_wiki_dump_to_neo4j(addr, wiki_path, auth):
    withf(GraphDatabase.driver(addr, auth=auth), lambda driver:
    withf(create_session(driver), lambda session:
    wopenr(wiki_path)(lambda wiki_xml:
    wprogb(filesize(wiki_path))(lambda pb:
    fin(lambda:
        pipe(
            wiki_xml,
            io_read_xml,
            map(io_report_progress(pb, wiki_xml)),
            filter(is_page),
            map(process_page),
            batch(200),
            map(send_batch_to_db(session)),
            any,
            ),
        lambda: True
    )))))
