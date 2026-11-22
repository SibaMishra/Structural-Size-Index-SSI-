"""Baseline metrics and sensitivity features computed from the same Tree-sitter trees.

One implementation covers Java, Python, JavaScript, and Go, so no language
gets a different counting rule. Definitions are given in the appendix of the paper.
"""
import math
import re
import sys
from collections import Counter

from ssi_pipeline import NODE_MAP, build_features, parse

sys.setrecursionlimit(20000)

COMMENT_TYPES = {'comment', 'line_comment', 'block_comment'}
STRING_TYPES = {
    'python': {'string', 'concatenated_string'},
    'java': {'string_literal', 'text_block'},
    'javascript': {'string', 'template_string'},
    'go': {'interpreted_string_literal', 'raw_string_literal'},
    'cpp': {'string_literal', 'raw_string_literal', 'char_literal'},
}

# ---------------------------------------------------------------- tokens

def _docstring_spans(root, language):
    """Python docstrings are treated as comments, as cloc does."""
    spans = []
    if language != 'python':
        return spans
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == 'function_definition':
            body = n.child_by_field_name('body')
            if body is not None and body.named_child_count:
                first = body.named_children[0]
                if first.type == 'expression_statement' and first.named_child_count:
                    first = first.named_children[0]
                if first.type == 'string':
                    spans.append((first.start_byte, first.end_byte))
        stack.extend(n.children)
    return spans


def tokens(root, language):
    """Yield (kind, text, start_row, end_row) for every code token.

    kind is 'op' for unnamed leaves (keywords, punctuation, operators) and
    'operand' for named leaves and whole string literals. Comments and
    docstrings are skipped.
    """
    strings = STRING_TYPES[language]
    docs = _docstring_spans(root, language)
    out = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type in COMMENT_TYPES or n.is_missing:
            continue
        if any(s <= n.start_byte and n.end_byte <= e for s, e in docs):
            continue
        if n.type in strings:
            out.append(('operand', n.text, n.start_point[0], n.end_point[0]))
            continue
        if n.child_count == 0:
            kind = 'operand' if n.is_named else 'op'
            text = n.text if kind == 'operand' else n.type.encode()
            out.append((kind, text, n.start_point[0], n.end_point[0]))
            continue
        stack.extend(reversed(n.children))
    return out


def token_metrics(root, language):
    toks = tokens(root, language)
    ops = Counter(t[1] for t in toks if t[0] == 'op')
    opd = Counter(t[1] for t in toks if t[0] == 'operand')
    n1, n2 = len(ops), len(opd)
    big_n1, big_n2 = sum(ops.values()), sum(opd.values())
    vocab, length = n1 + n2, big_n1 + big_n2
    volume = length * math.log2(vocab) if vocab > 1 else 0.0
    rows = set()
    for _, _, a, b in toks:
        rows.update(range(a, b + 1))
    return {'token_count': length, 'halstead_volume': volume, 'logical_loc': len(rows)}


# ---------------------------------------------------------------- cognitive complexity

_CC = {
    'python': dict(loops={'for_statement', 'while_statement'}, switch={'match_statement'},
                   catch={'except_clause', 'except_group_clause'}, ternary={'conditional_expression'},
                   boolop={'boolean_operator'},
                   funcs={'function_definition', 'lambda'}),
    'java': dict(loops={'for_statement', 'enhanced_for_statement', 'while_statement', 'do_statement'},
                 switch={'switch_expression', 'switch_statement'}, catch={'catch_clause'},
                 ternary={'ternary_expression'}, boolop={'binary_expression'},
                 funcs={'method_declaration', 'constructor_declaration', 'lambda_expression'}),
    'javascript': dict(loops={'for_statement', 'for_in_statement', 'while_statement', 'do_statement'},
                       switch={'switch_statement'}, catch={'catch_clause'}, ternary={'ternary_expression'},
                       boolop={'binary_expression'},
                       funcs={'function_declaration', 'function_expression', 'function', 'arrow_function',
                              'method_definition', 'generator_function', 'generator_function_declaration'}),
    'go': dict(loops={'for_statement'},
               switch={'expression_switch_statement', 'type_switch_statement', 'select_statement'},
               catch=set(), ternary=set(), boolop={'binary_expression'},
               funcs={'function_declaration', 'method_declaration', 'func_literal'}),
}
_CC['cpp'] = dict(loops={'for_statement', 'for_range_loop', 'while_statement', 'do_statement'}, switch={'switch_statement'},
                  catch={'catch_clause'}, ternary={'conditional_expression'}, boolop={'binary_expression'},
                  funcs={'function_definition', 'lambda_expression'})
_LOGICAL = {'&&', '||', '??', 'and', 'or'}


def _bool_op(node):
    op = node.child_by_field_name('operator')
    if op is not None and op.type in _LOGICAL:
        return op.type
    return None


def cognitive_complexity(root, language):
    """Cognitive complexity after the SonarSource definition.

    Structures that break the linear flow add 1 plus the nesting level. else and
    else if add 1. A run of the same logical operator adds 1. Nested functions
    raise the nesting level. Recursion and labeled jumps are not counted.
    """
    cfg = _CC[language]
    total = 0

    def children_at(node, nesting, fdepth, skip=()):
        for c in node.children:
            if c not in skip:
                visit(c, nesting, fdepth, False)

    def visit(node, nesting, fdepth, is_elseif):
        nonlocal total
        t = node.type
        if t in cfg['funcs']:
            n2 = nesting + 1 if fdepth > 0 else nesting
            for c in node.children:
                visit(c, n2, fdepth + 1, False)
            return
        if t == 'if_statement':
            if not is_elseif:
                total += 1 + nesting
            if language == 'python':
                for c in node.children:
                    if c.type in ('elif_clause', 'else_clause'):
                        total += 1
                    visit(c, nesting + 1, fdepth, False)
                return
            alt = node.child_by_field_name('alternative')
            for c in node.children:
                if alt is not None and c == alt:
                    total += 1
                    inner = alt
                    if alt.type == 'else_clause':
                        inner = alt.named_children[0] if alt.named_child_count else None
                    if inner is not None and inner.type == 'if_statement':
                        visit(inner, nesting, fdepth, True)
                    else:
                        for cc in alt.children:
                            visit(cc, nesting + 1, fdepth, False)
                else:
                    visit(c, nesting + 1, fdepth, False)
            return
        if t in cfg['loops'] or t in cfg['switch'] or t in cfg['catch'] or t in cfg['ternary']:
            total += 1 + nesting
            for c in node.children:
                visit(c, nesting + 1, fdepth, False)
            return
        if t in cfg['boolop']:
            op = _bool_op(node)
            if op is not None:
                p = node.parent
                if not (p is not None and p.type in cfg['boolop'] and _bool_op(p) == op):
                    total += 1
        for c in node.children:
            visit(c, nesting, fdepth, False)

    visit(root, 0, 0, False)
    return total


# ---------------------------------------------------------------- named node features

class NamedVisitor:
    """Same counters as ASTVisitor, over named nodes only."""

    def __init__(self):
        self.counter = Counter()
        self.total_nodes = 0
        self.leaf_nodes = 0
        self.internal_nodes = 0
        self.max_depth = 0

    def visit(self, node, depth=0):
        self.total_nodes += 1
        self.counter[node.type] += 1
        self.max_depth = max(self.max_depth, depth)
        kids = node.named_children
        if not kids:
            self.leaf_nodes += 1
        else:
            self.internal_nodes += 1
        for c in kids:
            self.visit(c, depth + 1)


def named_features(root, language):
    v = NamedVisitor()
    v.visit(root)
    return build_features(v, language)


# ---------------------------------------------------------------- exception handling

def exception_counts(root, language):
    handlers = raises = 0
    stack = [root]
    while stack:
        n = stack.pop()
        t = n.type
        if language == 'python':
            handlers += t in ('except_clause', 'except_group_clause')
            raises += t == 'raise_statement'
        elif language in ('java', 'javascript'):
            handlers += t == 'catch_clause'
            raises += t == 'throw_statement'
        elif language == 'go':
            handlers += t == 'defer_statement'
            if t == 'call_expression':
                fn = n.child_by_field_name('function')
                raises += fn is not None and fn.text == b'panic'
        stack.extend(n.children)
    return int(handlers), int(raises)


# ---------------------------------------------------------------- text based cyclomatic complexity (repository code)

def compute_cyclomatic(code):
    """McCabe complexity by keyword matching, copied from the repository notebook."""
    complexity = 1
    patterns = [r"\bif\b", r"\belse\s+if\b", r"\belif\b", r"\bfor\b", r"\bwhile\b", r"\bcase\b",
                r"\bcatch\b", r"\bexcept\b", r"\?\s*", r"&&", r"\|\|"]
    for p in patterns:
        complexity += len(re.findall(p, code))
    return complexity


def analyze(code, language):
    """Everything needed from one function in a single parse."""
    from ssi_pipeline import ASTVisitor, compute_loc
    tree = parse(code, language)
    root = tree.root_node
    v = ASTVisitor()
    v.visit(root)
    feats = build_features(v, language)
    named = named_features(root, language)
    tm = token_metrics(root, language)
    h, r = exception_counts(root, language)
    loc = compute_loc(code)
    cc = compute_cyclomatic(code)
    vol = tm['halstead_volume']
    mi = 171 - 5.2 * math.log(max(vol, 1.0)) - 0.23 * cc - 16.2 * math.log(max(loc, 1))
    return dict(
        features=feats, named=named,
        loc=loc, cc_text=cc, cog=cognitive_complexity(root, language), mi=mi,
        handlers=h, raises=r, has_error=bool(root.has_error), counter=dict(v.counter), **tm,
    )
