"""Feature extraction for SSI, copied from the notebooks in the SSI repository.

NODE_MAP, the tree visitor, and build_features() are the repository code
without changes, so the 32 features are the ones used in the paper.
"""
from collections import Counter
import numpy as np
import pandas as pd
from tree_sitter_language_pack import get_parser

FEATURES = [
    'function_count', 'parameter_count', 'variable_count', 'declaration_count', 'if_count', 'for_count',
    'while_count', 'switch_count', 'return_count', 'call_count', 'argument_count', 'assignment_count',
    'binary_expression_count', 'identifier_count', 'block_count', 'expression_count', 'literal_count',
    'ast_nodes', 'leaf_nodes', 'internal_nodes', 'max_depth', 'leaf_ratio', 'internal_ratio', 'branch_density',
    'call_density', 'identifier_density', 'expression_density', 'variable_density', 'statement_density',
    'nesting_factor', 'cyclomatic_estimate', 'avg_parameters',
]

NODE_MAP = {
    "java": {
        "function_count": ["method_declaration"],
        "parameter_count": ["formal_parameter"],
        "variable_count": ["local_variable_declaration", "variable_declarator"],
        "declaration_count": ["variable_declarator"],
        "if_count": ["if_statement"],
        "for_count": ["for_statement"],
        "while_count": ["while_statement"],
        "switch_count": ["switch_statement", "switch_expression"],
        "return_count": ["return_statement"],
        "call_count": ["method_invocation"],
        "argument_count": ["argument_list"],
        "assignment_count": ["assignment_expression"],
        "binary_expression_count": ["binary_expression"],
        "identifier_count": ["identifier"],
        "block_count": ["block"],
        "expression_count": ["expression_statement"],
        "literal_count": ["string_literal", "null_literal", "decimal_integer_literal", "true", "false"]
    },
    "python": {
        "function_count": ["function_definition"],
        "parameter_count": ["parameter"],
        "variable_count": ["assignment"],
        "declaration_count": ["assignment"],
        "if_count": ["if_statement"],
        "for_count": ["for_statement"],
        "while_count": ["while_statement"],
        "switch_count": ["match_statement"],
        "return_count": ["return_statement"],
        "call_count": ["call"],
        "argument_count": ["argument_list"],
        "assignment_count": ["assignment"],
        "binary_expression_count": ["binary_operator"],
        "identifier_count": ["identifier"],
        "block_count": ["block"],
        "expression_count": ["expression_statement"],
        "literal_count": ["string", "integer", "float", "true", "false", "none"]
    },
    "javascript": {
        "function_count": ["function_declaration", "function_expression"],
        "parameter_count": ["formal_parameters"],
        "variable_count": ["variable_declarator"],
        "declaration_count": ["variable_declarator"],
        "if_count": ["if_statement"],
        "for_count": ["for_statement"],
        "while_count": ["while_statement"],
        "switch_count": ["switch_statement"],
        "return_count": ["return_statement"],
        "call_count": ["call_expression", "new_expression"],
        "argument_count": ["arguments"],
        "assignment_count": ["assignment_expression"],
        "binary_expression_count": ["binary_expression"],
        "identifier_count": ["identifier", "property_identifier"],
        "block_count": ["statement_block"],
        "expression_count": ["expression_statement"],
        "literal_count": ["string", "null", "true", "false", "number"]
    },
    "go": {
        "function_count": ["function_declaration", "method_declaration"],
        "parameter_count": ["parameter_declaration"],
        "variable_count": ["var_declaration", "short_var_declaration", "assignment_statement"],
        "declaration_count": ["var_spec"],
        "if_count": ["if_statement"],
        "for_count": ["for_statement"],
        "while_count": [],
        "switch_count": ["expression_switch_statement", "type_switch_statement"],
        "return_count": ["return_statement"],
        "call_count": ["call_expression"],
        "argument_count": ["argument_list"],
        "assignment_count": ["assignment_statement"],
        "binary_expression_count": ["binary_expression"],
        "identifier_count": ["identifier", "field_identifier"],
        "block_count": ["block"],
        "expression_count": ["expression_list"],
        "literal_count": ["interpreted_string_literal", "raw_string_literal", "int_literal", "float_literal", "true", "false"]
    }
}


# C++ is not in the original repository. This mapping was written for the extension study (held out language,
# never used to fit the PCA) and follows the Java mapping node for node where the grammars allow it.
NODE_MAP["cpp"] = {
    "function_count": ["function_definition"],
    "parameter_count": ["parameter_declaration"],
    "variable_count": ["declaration", "init_declarator"],
    "declaration_count": ["init_declarator"],
    "if_count": ["if_statement"],
    "for_count": ["for_statement", "for_range_loop"],
    "while_count": ["while_statement", "do_statement"],
    "switch_count": ["switch_statement"],
    "return_count": ["return_statement"],
    "call_count": ["call_expression"],
    "argument_count": ["argument_list"],
    "assignment_count": ["assignment_expression"],
    "binary_expression_count": ["binary_expression"],
    "identifier_count": ["identifier", "field_identifier"],
    "block_count": ["compound_statement"],
    "expression_count": ["expression_statement"],
    "literal_count": ["string_literal", "number_literal", "char_literal", "true", "false", "null", "nullptr"],
}


# Optional correction of the Python mapping. In the Python grammar used here the node types `parameter` and
# `expression_statement` never occur, so parameter_count and expression_count (and the three features derived
# from them) are zero for every Python function in the original repository code. With FIX_PYTHON = True the two
# counts are taken from the nodes that do occur. Default is False, which reproduces the repository exactly.
FIX_PYTHON = False
_PY_PARAM_TYPES = {'identifier', 'typed_parameter', 'default_parameter', 'typed_default_parameter',
                   'list_splat_pattern', 'dictionary_splat_pattern', 'tuple_pattern'}
_PY_NOT_EXPR = {'if_statement', 'for_statement', 'while_statement', 'try_statement', 'with_statement', 'match_statement',
                'function_definition', 'class_definition', 'decorated_definition', 'return_statement', 'pass_statement',
                'break_statement', 'continue_statement', 'raise_statement', 'assert_statement', 'delete_statement',
                'import_statement', 'import_from_statement', 'future_import_statement', 'global_statement',
                'nonlocal_statement', 'print_statement', 'exec_statement', 'comment', 'string', 'else_clause',
                'elif_clause', 'except_clause', 'finally_clause', 'type_alias_statement'}


def set_fix_python(flag):
    global FIX_PYTHON
    FIX_PYTHON = bool(flag)


class ASTVisitor:
    def __init__(self):
        self.counter = Counter()
        self.total_nodes = 0
        self.leaf_nodes = 0
        self.internal_nodes = 0
        self.max_depth = 0

    def visit(self, node, depth=0, parent_type=None):
        self.total_nodes += 1
        self.counter[node.type] += 1
        if FIX_PYTHON:
            if parent_type == 'parameters' and node.type in _PY_PARAM_TYPES:
                self.counter['__py_param'] += 1
            if parent_type in ('block', 'module') and node.is_named and node.type not in _PY_NOT_EXPR:
                self.counter['__py_expr'] += 1
        self.max_depth = max(self.max_depth, depth)
        children = len(node.children)
        if children == 0:
            self.leaf_nodes += 1
        else:
            self.internal_nodes += 1
        for child in node.children:
            self.visit(child, depth + 1, node.type)


def build_features(visitor, language):
    mapping = NODE_MAP[language]
    features = {}
    for feature, node_list in mapping.items():
        features[feature] = sum(visitor.counter[node] for node in node_list)
    if FIX_PYTHON and language == 'python':
        features['parameter_count'] = visitor.counter['__py_param']
        features['expression_count'] = visitor.counter['__py_expr']
    ast = max(visitor.total_nodes, 1)
    features['ast_nodes'] = visitor.total_nodes
    features['leaf_nodes'] = visitor.leaf_nodes
    features['internal_nodes'] = visitor.internal_nodes
    features['max_depth'] = visitor.max_depth
    features['leaf_ratio'] = visitor.leaf_nodes / ast
    features['internal_ratio'] = visitor.internal_nodes / ast
    features['branch_density'] = (features['if_count'] + features['for_count'] + features['while_count']) / ast
    features['call_density'] = features['call_count'] / ast
    features['identifier_density'] = features['identifier_count'] / ast
    features['expression_density'] = features['expression_count'] / ast
    features['variable_density'] = features['variable_count'] / ast
    features['statement_density'] = features['expression_count'] / ast
    features['nesting_factor'] = features['max_depth'] / ast
    features['cyclomatic_estimate'] = 1 + features['if_count'] + features['for_count'] + features['while_count']
    features['avg_parameters'] = features['parameter_count'] / max(features['function_count'], 1)
    return features


def parse(code, language):
    return get_parser(language).parse(bytes(code, 'utf8'))


def extract_features(code, language):
    tree = parse(code, language)
    visitor = ASTVisitor()
    visitor.visit(tree.root_node)
    return build_features(visitor, language)


def compute_loc(code):
    """Physical LOC as in the repository: non blank lines, comments included."""
    return len([l for l in code.split('\n') if l.strip() != ''])


def frozen_parameters(ref_features):
    """Means, population standard deviations, and PC1 loadings of the reference corpus."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X = ref_features[FEATURES]
    scaler = StandardScaler().fit(X)
    pca = PCA().fit(scaler.transform(X))
    w = pca.components_[0]
    if w[FEATURES.index('ast_nodes')] < 0:
        w = -w
    return pd.DataFrame({'feature': FEATURES, 'mean': scaler.mean_, 'std': scaler.scale_, 'loading': w}), pca


def ssi(features, params):
    """SSI from raw features with frozen parameters (no refit)."""
    z = (features[FEATURES].values - params['mean'].values) / params['std'].values
    return z @ params['loading'].values
