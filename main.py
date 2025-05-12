import os
import json
import threading
import bisect
import re
from datetime import datetime

CONFIG = {
    "data_dir": "./data",
    "btree_degree": 2
}

# B-Tree implementation
class BTreeNode:
    def __init__(self, t, leaf=False):
        self.t = t
        self.leaf = leaf
        self.keys = []
        self.values = []

    def split_child(self, i, y):
        z = BTreeNode(y.t, y.leaf)
        t = y.t
        z.keys = y.keys[t:]
        z.values = y.values[t:]
        y.keys = y.keys[:t-1]
        y.values = y.values[:t-1] if y.leaf else y.values[:t]
        self.keys.insert(i, y.keys.pop())
        self.values.insert(i+1, z)

    def insert_non_full(self, key, value):
        if self.leaf:
            pos = bisect.bisect_left(self.keys, key)
            if pos < len(self.keys) and self.keys[pos] == key:
                self.values[pos].append(value)
            else:
                self.keys.insert(pos, key)
                self.values.insert(pos, [value])
        else:
            i = bisect.bisect_right(self.keys, key)
            child = self.values[i]
            if len(child.keys) == 2*self.t - 1:
                self.split_child(i, child)
                if key > self.keys[i]:
                    i += 1
            self.values[i].insert_non_full(key, value)

    def search(self, key):
        i = bisect.bisect_left(self.keys, key)
        if i < len(self.keys) and self.keys[i] == key:
            return self.values[i]
        if self.leaf:
            return []
        return self.values[i].search(key)

class BTreeIndex:
    def __init__(self, fields, t=2):
        self.fields = fields if isinstance(fields, tuple) else (fields,)
        self.t = t
        self.root = BTreeNode(t, leaf=True)
        self.lock = threading.Lock()

    def _extract_key(self, doc):
        return tuple(doc.get(f) for f in self.fields)

    def insert(self, doc, doc_id):
        key = self._extract_key(doc)
        with self.lock:
            r = self.root
            if len(r.keys) == 2*self.t - 1:
                s = BTreeNode(self.t, leaf=False)
                s.values.append(r)
                s.split_child(0, r)
                self.root = s
            self.root.insert_non_full(key, doc_id)

    def find(self, key_tuple):
        return self.root.search(key_tuple)

# JSON IO

def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _dump_json(path, doc):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(doc, f)

# Collection with compound index support
class Collection:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.indexes = {}
        os.makedirs(self.path, exist_ok=True)

    def create_index(self, fields):
        key = tuple(fields) if isinstance(fields, list) else (fields,)
        if key in self.indexes:
            return
        idx = BTreeIndex(key, CONFIG["btree_degree"])
        for fname in os.listdir(self.path):
            if fname.endswith('.json'):
                doc = _load_json(os.path.join(self.path, fname))
                idx.insert(doc, doc['_id'])
        self.indexes[key] = idx

    def _update_indexes(self, doc):
        for idx in self.indexes.values():
            idx.insert(doc, doc['_id'])

    def insert(self, doc):
        doc_id = doc.get('_id') or f"{threading.get_ident()}_{len(os.listdir(self.path))}"
        doc['_id'] = doc_id
        path = os.path.join(self.path, f"{doc_id}.json")
        _dump_json(path, doc)
        self._update_indexes(doc)
        return doc_id

    def find_ids_by_index(self, field_values):
        for key in self.indexes:
            if all(f in field_values for f in key):
                values = tuple(field_values[f] for f in key)
                return self.indexes[key].find(values)
        return None

    def load_docs(self, ids=None):
        files = os.listdir(self.path)
        if ids:
            files = [f"{_id}.json" for _id in ids if f"{_id}.json" in files]
        return [_load_json(os.path.join(self.path, f)) for f in files if f.endswith('.json')]

    def update(self, updates, cond, ids=None):
        count = 0
        for doc in self.load_docs(ids):
            if cond(doc):
                doc.update(updates)
                _dump_json(os.path.join(self.path, f"{doc['_id']}.json"), doc)
                count += 1
        return count

    def delete(self, cond, ids=None):
        count = 0
        for doc in self.load_docs(ids):
            if cond(doc):
                os.remove(os.path.join(self.path, f"{doc['_id']}.json"))
                count += 1
        return count

# Parser and Executor
class QueryParser:
    @staticmethod
    def parse(query):
        q = re.sub(r"\s+", " ", query.strip()).rstrip(';')
        u = q.upper()
        if u.startswith('INSERT INTO'):
            m = re.match(r"INSERT INTO (\w+)\s+(.+)", q, re.IGNORECASE)
            return {'cmd': 'insert', 'collection': m.group(1), 'doc': json.loads(m.group(2))}
        if u.startswith('CREATE INDEX'):
            m = re.match(r"CREATE INDEX ON (\w+)\(([^)]+)\)", q, re.IGNORECASE)
            fields = [f.strip() for f in m.group(2).split(',')]
            return {'cmd': 'index', 'collection': m.group(1), 'fields': fields}
        if u.startswith('DELETE FROM'):
            m = re.match(r"DELETE FROM (\w+)(?: WHERE (.+))?", q, re.IGNORECASE)
            return {'cmd': 'delete', 'collection': m.group(1), 'where': m.group(2)}
        if u.startswith('UPDATE'):
            m = re.match(r"UPDATE (\w+) SET (.+?)(?: WHERE (.+))?$", q, re.IGNORECASE)
            sets = {}
            for part in m.group(2).split(','):
                k, v = map(str.strip, part.split('=', 1))
                sets[k] = json.loads(v) if v.startswith(('"', "'")) else int(v)
            return {'cmd': 'update', 'collection': m.group(1), 'sets': sets, 'where': m.group(3)}
        m = re.match(r"SELECT (.+) FROM (\w+)(?: WHERE (.+))?", q, re.IGNORECASE)
        fields = [f.strip() for f in m.group(1).split(',')]
        return {'cmd': 'select', 'fields': fields, 'collection': m.group(2), 'where': m.group(3)}

class QueryEngine:
    def __init__(self, base):
        self.base = base

    def execute(self, q):
        coll = Collection(q['collection'], os.path.join(self.base, q['collection']))
        if q['cmd'] == 'insert':
            return {'_id': coll.insert(q['doc'])}
        if q['cmd'] == 'index':
            coll.create_index(q['fields'])
            return {'status': 'index created'}

        cond = lambda d: True
        fields_used = {}
        if q.get('where'):
            cond, fields_used = self._make_eval(q['where'])

        ids = coll.find_ids_by_index(fields_used) if fields_used else None

        if q['cmd'] == 'update':
            return {'updated': coll.update(q['sets'], cond, ids)}
        if q['cmd'] == 'delete':
            return {'removed': coll.delete(cond, ids)}
        if q['cmd'] == 'select':
            docs = coll.load_docs(ids)
            filtered = [d for d in docs if cond(d)]
            if q['fields'] == ['*']:
                return filtered
            result = []
            for d in filtered:
                result.append({f: d.get(f) for f in q['fields']})
            return result
        return {}

    def _make_eval(self, expr):
        expr = expr.strip()
        fields = {}
        def eval_fn(doc):
            parts = re.split(r"\s+AND\s+", expr, flags=re.IGNORECASE)
            for part in parts:
                if '=' in part:
                    k, v = map(str.strip, part.split('=', 1))
                    fields[k] = v.strip("'")
                    if str(doc.get(k)) != v.strip("'"):
                        return False
            return True
        return eval_fn, fields

# CLI
if __name__ == '__main__':
    print("Welcome to JSON DB CLI.\nType 'help' to view available commands or type 'exit' to exit the program.")
    cmds = [
        "SELECT <fields> FROM <collection> [WHERE ...]",
        "INSERT INTO <collection> <json>",
        "CREATE INDEX ON <collection>(field[, field2...])",
        "DELETE FROM <collection> WHERE ...",
        "UPDATE <collection> SET field=value WHERE ...",
        "help", "exit"
    ]
    while True:
        q = input('db> ').strip()
        if not q:
            continue
        if q.lower() == 'exit':
            break
        if q.lower() == 'help':
            print(*cmds, sep='\n')
            continue
        try:
            parsed = QueryParser.parse(q)
            result = QueryEngine(CONFIG['data_dir']).execute(parsed)
            print(json.dumps(result, indent=2, default=str))
        except Exception as e:
            print(f"Error: {e}")
