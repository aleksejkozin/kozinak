# pylint: disable=E1120

import re
import os
import bz2
import csv
import urllib.request
import mwparserfromhell as mw
import more_itertools
from lxml import etree
from more_itertools import unique_everseen
from multiprocessing import Pool
from types import SimpleNamespace
from collections import namedtuple
from pathlib import Path
from functools import reduce, lru_cache
from itertools import (islice, chain, dropwhile, takewhile, 
    groupby, zip_longest, tee, filterfalse, count, repeat)
from toolz import compose, curry, pipe
from toolz.itertoolz import get, concat
from toolz.functoolz import identity
from toolz.curried import map
from tqdm import tqdm

reduce = curry(reduce)
get = curry(get)
filter = curry(filter)
filterempty = filter(identity)
dropwhile = curry(dropwhile)
takewhile = curry(takewhile)
groupby_ = curry(lambda key, iterable: groupby(iterable, key))
flatten = chain.from_iterable
flip = lambda f: lambda *a: f(*reversed(a))
comp = compose
comp_ = flip(compose)
map_ = lambda f: map(lambda x: f(*x))
truemap = curry(comp(filterempty, map))
truemap_ = curry(comp(filterempty, map_))
mapmap = lambda f: map(map(f))
chunked = curry(flip(more_itertools.chunked))
SN = SimpleNamespace

@curry
def getattr_(name, obj, default=None):
    return getattr(obj, name, default)

@curry
def get_(n, iterable, default=None):
    """Get n element from iterable

    >>> pipe((1, 2, 3, 4), get_(3, default='nope'))
    4

    >>> pipe((1, 2, 3, 4), get_(4, default='nope'))
    'nope'
    """
    return get(n, iterable, default=default)

def islazy(x):
    """Check if x is a lazy evaluation

    >>> islazy((x for x in range(10)))
    True

    >>> islazy('')
    False

    >>> islazy(dict(a=1))
    False
    """
    return (
        hasattr(x, '__iter__')
        and not isinstance(x, str)    
        and not isinstance(x, dict)   
        and not isinstance(x, set)
        )

def force(x):
    """Force lazy evaluations
    
    >>> lazy = ((x * y for y in range(3)) for x in range(3))
    >>> comp(str, force)(lazy)
    '((0, 0, 0), (0, 1, 2), (0, 2, 4))'
    """
    if islazy(x):
        return tuple(force(y) for y in x)
    return x

@curry
def first(default, iterable):
    """Returns first element of iterable
    You should also specify default value

    >>> first('none', (1, 2, 3))
    1

    >>> first('none', ())
    'none'
    """
    return more_itertools.first(iterable, default)

@curry
def last(default, iterable):
    """Returns last element of iterable
    You should also specify default value

    >>> last('none', (1, 2, 3))
    3

    >>> last('none', ())
    'none'
    """
    return more_itertools.last(iterable, default)

@curry
def replace(predicate, substitutes, count, iterable):
    return more_itertools.replace(iterable, predicate, substitutes, count)

@curry
def zip_(a, b):
    # disabling default zip() params for curry call
    return zip(a, b)

@curry
def call(f, x):
    return f(x)

@curry
def call_(x, f):
    return f(x)

@curry
def calla(f, x):
    return f(*x)

def bottom(x):
    pass

@curry
def ifempty(value, x):
    return x if x else value

forcemap = compose(any, map(bottom))

@curry
def pmap(pool, chunksize, f, iterable):
    return pool.imap(f, iterable)

@curry
def pmap_(pool, chunksize, f):
    return pmap(pool, chunksize, calla(f))

def zipcall(*args):
    return comp_(
        zip_(args),
        map_(call),
        flatten,
        )

@curry
def sorted_(key, iterable):
    return sorted(iterable, key=key)

@curry
def reduce_(f, initializer, iterable):
    return reduce(f, iterable, initializer)

@curry
def chain_(x, iterable):
    return chain(x, iterable)

def truechain(*iterables):
    return chain(*filterempty(iterables))

def not_(x):
    return not x

def output(x):
    result = force(x)
    print(result)
    return result

@curry
def print_(v, x):
    print(v)
    return x

@curry
def branch(predicate, yes, no, x):
    if predicate(x):
        return yes(x)
    return no(x)

def apply(*fs):
    @curry
    def apply_wrap(x):
        return pipe(
            fs, 
            map(lambda f: f(x)), 
            flatten
            )
    return apply_wrap

@curry
def take(n, iterable):
    """Take n elements from an iterable

    >>> comp(tuple, take)(3, count())
    (0, 1, 2)
    """
    return islice(iterable, n)

@curry
def drop(n, iterable):
    """Drop n elements from an iterable

    >>> comp(tuple, drop)(2, (1, 2, 3))
    (3,)
    """
    return islice(iterable, n, None)

@curry
def split(predicate, iterable):
    return (list(v) for _, v in groupby(iterable, predicate))

@curry
def split_(predicate, iterable):
    return (list(v) for flag, v in groupby(iterable, predicate) if not flag)

def pool():
    return Pool(None)

def prog(total):
    return tqdm(total=total, mininterval=5.0)

def progf(source):
    return tqdm(total=filesize(source), mininterval=5.0)

@curry
def open_(t, path):
    def select_reader():
        if path.lower().endswith('.bz2'):
            return bz2.BZ2File(path, t)
        else:
            return open(path, t)
    return select_reader()

openrb = open_('rb')
openr = open_('r')
openwb = open_('wb')
openw = open_('w')

@curry
def io_report_progress(pb, fd, x):
    pb.update(fd.tell() - pb.n)
    return x

def filesize(path):
    return Path(path).stat().st_size

def string_list_to_dict(data, sep='='):
    return pipe(
        (x.split(sep, 1) for x in data),
        mapmap(lambda x: x.strip(' \n\r')),
        map(list),
        filter(lambda x: len(x) == 2),
        dict
        )

@curry
def bracket(before, after):
    try:
        before()
    except etree.ParseError as e:
        print(f'Parse error: {e}')
    finally:
        after()

def _xml_tag(x): 
    return last(None, x.tag.split('}'))

def _xml_to_dict(root):
    return {
        **{_xml_tag(x): _xml_to_dict(x) for x in list(root)}, 
        **{'body': root.text, 'attributes': root.attrib}
        }

@curry
def iterparse_xml(tag, source):
    context = etree.iterparse(
        source, 
        events=('end',), 
        tag=tag, 
        remove_comments=True,
        recover=True
        )
    for _, elem in context:
        yield _xml_to_dict(elem)
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    del context

# ------ WIKIPEDIA --------

WIKI_PAGE_LABLE = 'WikiPage'
PERSON_LABEL = 'Person'
LINK_LABLE = 'LINK'

@lru_cache()
def _infobox_to_labels():
    @curry
    def subgroup(i, labels, iterable):
        base = 'Template:Infobox '
        def is_h(x):
            return x.tag == f'h{i}'
        def is_infobox(x):
            return x.tag == 'a' and (base in x.text if x.text else False)
        def output_result(infobox):
            return (
                infobox.text.replace(base, ''), 
                tuple(labels)
                )
        def extract_lables(hs):
            return pipe(
                hs,
                map(get_(0)),
                map(getattr_('text')),
                tuple
                )
        def recursion(hs, iterable_):
            return subgroup(
                i + 1, 
                labels + extract_lables(hs), 
                iterable_
                )
        return pipe(
            iterable,
            split(is_h),
            tuple,
            branch(
                lambda x: len(x) > 1,
                comp_(
                    dropwhile(comp(not_, is_h, get(0))),
                    chunked(2),
                    map_(recursion),
                    flatten,
                    ),
                comp_(
                    flatten,
                    filter(is_infobox),
                    map(output_result),
                    )
                ),
            )
    return pipe(
        'https://en.wikipedia.org/wiki/Wikipedia:List_of_infoboxes',
        urllib.request.urlopen,
        lambda x: etree.iterparse(x, html=True, remove_comments=True),
        map(get(1)),
        dropwhile(lambda x: x.attrib.get('id', None) != 'toc'),
        dropwhile(lambda x: x.tag != 'h2'),
        takewhile(lambda x: x.tag != 'span' or x.text != 'Unsorted'),   
        subgroup(2, ()),
        dict,
        )

def _extract_title_and_text(page):
    return (
        page['title']['body'], 
        page['revision']['text']['body']
        )

def _remove_html_comments(html):
    return re.sub('(<!--.*?-->)', '', html).strip(' \n\r')

def _clear_link(lnk, fast=False):
    lnk1 = _remove_html_comments(str(lnk)) if not fast else str(lnk)
    result = (
        lnk1
        .strip()
        .replace('{{!}}', '#')
        .split('#')[0]
        .replace(' ', '_')
        .replace('"', '_')
        .replace("'''''", '')
        .replace("'''", '')
        .replace("''", '')
        .strip()
        )
    if not result:
        return
    if not fast and not _is_good_csv_field(result):
        return
    return result[0].upper() + result[1:]

def _make_relationship(source, link_type, target):
    clean_source = _clear_link(source)
    clean_target = _clear_link(target)
    if clean_source and clean_target and link_type:
        return (
            'relationship',
            dict(
                source=clean_source, 
                label=link_type, 
                target=clean_target
                )
            )

def _make_node(id, labels):
    clean_id = _clear_link(id)
    clean_labels = pipe(
        labels,
        truemap(_clear_link),
        set
        )
    if clean_id and clean_labels:
        return ('node', {'id': clean_id, 'labels': clean_labels})

def _extract_infobox(templates):
    infobox = pipe(
        templates,
        filter(lambda x: x.name.startswith('Infobox')),
        first(None)
        )
    if not infobox:
        return dict()
    box_type_ = (
        str(infobox.name)
        .replace('Infobox', '')
        .split('\n')[0]
        .strip(' \n\r')
        )
    box_type = _remove_html_comments(box_type_)
    if not _is_good_csv_field(box_type):
        return dict()
    return {
        **{'infobox_type': box_type},
        **string_list_to_dict(infobox.params)
        }

def _parse_wikipage(title, templates):
    infobox = _extract_infobox(templates)
    infobox_type = infobox.get('infobox_type', None)
    is_alive = ('birth_date' in infobox and 'death_date' not in infobox)
    labels = chain(
        (
            WIKI_PAGE_LABLE,
            infobox_type,
            'Alive' if is_alive else None
        ),
        _infobox_to_labels().get(infobox_type, ()),
        )
    return (_make_node(title, labels),)

def _parse_wikilinks(title, wikilinks):
    return pipe(
        wikilinks,
        truemap(lambda x: x.title),
        truemap(lambda x: (
            _make_node(x, [WIKI_PAGE_LABLE]),
            _make_relationship(title, LINK_LABLE, x),
            )),
        flatten,
        )

@curry
def _parse_redirect_(template_name, params):
    """Split redirect template params on (input_links, output_links)

    >>> f = lambda *x: comp(force, _parse_redirect_)(x[0], x[1:])
    >>> f('Redirect', 'Foo')
    (('Foo',), ('Foo_(disambiguation)',))

    >>> f('Redirect', 'Poe', 'other uses with the name Poe')
    (('Poe',), ('Poe_(disambiguation)',))

    >>> f('Redirect', 'R', '', 'P1')
    (('R',), ('P1',))

    >>> f('Redirect', 'R', '', 'P1', 'and', 'P2')
    (('R',), ('P1', 'P2'))

    >>> f('Redirect2', 'R1', 'R2', 'U1', 'P1')
    (('R1', 'R2'), ('P1',))
    
    >>> f('Redirect', 'R', 'U1', 'P1', 'U2', 'P2')
    (('R',), ('P1', 'P2'))

    >>> f('Redirect', 'R', 'U1', 'P1', 'and', 'P2')
    (('R',), ('P1', 'P2'))

    >>> f('Redirect3', 'R1', 'R2', 'R3', 'U1', 'P1', 'U2', 'P2', 'and', 'P3')
    (('R1', 'R2', 'R3'), ('P1', 'P2', 'P3'))

    >>> f('Redirect', 'R1', 'U1', 'P1', 'other uses', 'P2', 'and', 'P3')
    (('R1',), ('P1', 'P2', 'P3'))

    >>> f('Redirect', 'R', 'U1', 'P1', 'other uses', 'P2')
    (('R',), ('P1', 'P2'))

    >>> f('Redirect', 'R', 'U1', 'P1', 'U2', 'P2', 'other uses')
    (('R',), ('P1', 'P2', 'R_(disambiguation)'))

    Returns 'None' on other tampaltes:
    >>> f('Something')
    """

    name = template_name.strip().lower()
    if name == 'redirect':
        n = 1
    elif name == 'redirect2':
        n = 2
    elif name == 'redirect3':
        n = 3
    else:
        return

    input_links, params_ = params[:n], params[n:]

    def disambiguations():
        return pipe(
            input_links,
            take(1),
            map(lambda x: f'{x}_(disambiguation)')
            )

    if len(params_) == 0 or len(params_) == 1:
        output_links = disambiguations()
    else:
        output_links = pipe(
            zip(params_, count()),
            filter(lambda x: x[1] % 2 == 1),
            map(get(0)),
            )

    if last(None, params_) == 'other uses':
        output_links_ = chain(output_links, disambiguations())
    else:
        output_links_ = output_links

    return pipe(
        (input_links, output_links_),
        mapmap(_clear_link),
        tuple
        )

@curry
def _parse_redirect(title, template):
    result = _parse_redirect_(template.name, template.params)
    if not result:
        return
    def to_event(links, isout):
        return pipe(
            links,
            map(lambda x: (
                _make_node(x, [WIKI_PAGE_LABLE]), 
                _make_relationship(title, LINK_LABLE, x)
                if isout else
                _make_relationship(x, LINK_LABLE, title) 
                )),
            flatten
            )
    return pipe(
        zip(result, (False,  True)),
        map_(to_event),
        flatten
        )

def _parse_templates(title, templates):
    procs = (_parse_redirect, _parse_authors)
    def call_procs(template):
        return pipe(
            procs,
            map(lambda f: f(title, template)),
            dropwhile(comp(not_, identity)),
            first(None),
            )
    return pipe(
        templates,
        truemap(call_procs),
        flatten,
        )

def _make_person_keys(*keys):
    return pipe(
        chain(('',), range(1, 9)),
        map(str),
        map(lambda x: map(lambda y: y % x, keys)),
        flatten,
        set,
        )

AUTHOR_KEYS = _make_person_keys(
    'author-link%s', 'author%s-link', 'authorlink%s',
    'editor-link%s', 'editor%s-link', 'editorlink%s',
    )

@curry
def _parse_authors(title, template):
    if 'cit' not in template.name:
        return
    return pipe(
        template.params,
        filter(
            lambda x: 
                str(x.name).strip() in AUTHOR_KEYS and hasattr(x, 'value')
            ),
        truemap(lambda x: (
            _make_node(x.value, [PERSON_LABEL]),
            _make_relationship(title, LINK_LABLE, x.value))
            ),
        flatten
        )

@curry
def _process_page(title, text):
    parser = mw.parse(text)
    wikilinks = parser.filter_wikilinks()
    templates = parser.filter_templates()

    results = truechain(
        _parse_wikipage(title, templates),
        _parse_wikilinks(title, wikilinks),
        _parse_templates(title, templates),
        )
    return tuple(results)

def parse_wiki_io(path):
    with pool() as p, openrb(path) as xml, progf(path) as pb:
        yield from pipe(
            xml,
            iterparse_xml(
                '{http://www.mediawiki.org/xml/export-0.10/}page'
                ),
            map(io_report_progress(pb, xml)),
            map(_extract_title_and_text),
            pmap_(p, 30, _process_page),
            # map_(_process_page),
            map(ifempty(())),
            flatten,
            filterempty,
            )

@curry
def sort_file_io(source_path, target_path):
    os.system(f'gsort --parallel=8 -uo {target_path} {source_path}')

def merge_csv_chunk_(iterable):
    def local_merge_(iterable):
        return pipe(
            iterable,
            map(lambda x: x.split(';')),
            flatten,
            set,
            sorted,
            ';'.join
            )
    return pipe(
        zip_longest(*iterable),
        map(local_merge_),
        )

@curry
def merge_csv_io(source_path, target_path):
    with open(source_path, 'r') as source, open(target_path, 'w') as target:
        reader = csv.reader(source)
        writer = csv.writer(target)
        pipe(
            reader,
            groupby_(get_(0)),
            map(get_(1)),
            map(merge_csv_chunk_),
            map(writer.writerow),
            forcemap,
            )

def _is_good_csv_field(field):
    if not field:
        return False
    return not pipe(
        field,
        filter(lambda x: x in '<>{}[]\n\r\t'),
        any,
        )

def _is_good_csv_record(rec):
    return all(_is_good_csv_field(x) for x in rec)

@curry
def clear_csv_io(source_path, target_path, n=2):
    with open(source_path, 'r') as source, open(target_path, 'w') as target:
        reader = csv.reader(source)
        writer = csv.writer(target)
        pipe(
            reader,
            filter(lambda x: len(x) == 2),
            filter(_is_good_csv_record),
            map(writer.writerow),
            forcemap,
            )

def _node_header():
    return 'name:ID,:LABEL\n'
def _node_to_string(node):
    id = node['id']
    labels_str = ';'.join(node['labels'])
    return f'"{id}",{labels_str}\n'

def _relationship_header():
    return ':START_ID,:END_ID,:TYPE\n'
def _relationship_to_string(rel):
    source = rel['source']
    target = rel['target']
    label = rel['label']
    return f'"{source}","{target}",{label}\n'

def wiki_dump_to_csv_io(dump_path, nodes_path, relationships_path):
    make_raw = lambda x: x + '.raw'
    nodes_raw_path = make_raw(nodes_path)
    rel_raw_path = make_raw(relationships_path)

    make_sorted = lambda x: x + '.sorted'
    nodes_sorted_path = make_sorted(nodes_path)

    with open(nodes_raw_path, 'w') as fnodes:
        nodes_writer = csv.writer(fnodes)
        with open(rel_raw_path, 'w') as frel:
            rel_writer = csv.writer(frel)
            def on_message(e, data):
                if e == 'node':
                    nodes_writer.writerow(
                        (data['id'], ';'.join(data['labels']))
                        )
                elif e == 'relationship':
                    rel_writer.writerow(
                        (data['source'], data['target'], data['label'])
                        )
            pipe(
                parse_wiki_io(dump_path),
                map_(on_message),
                any
                )

    sort_file_io(rel_raw_path, relationships_path)
    os.remove(rel_raw_path)
    
    sort_file_io(nodes_raw_path, nodes_sorted_path)
    os.remove(nodes_raw_path)

    merge_csv_io(nodes_sorted_path, nodes_path)
    os.remove(nodes_sorted_path)
