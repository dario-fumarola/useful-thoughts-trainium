"""Exact-value ordered AST identity for prospectively disjoint expressions.

Ignore whitespace,redundant parentheses,Unicode operator spelling,and decimal
spellings of the same rational constant. Preserve operator/operand structure;
this is not a symbolic-algebra or equal-answer equivalence test.
"""
import ast
from fractions import Fraction

from train.calculation_boundaries import parse_numeric


def expression_key(expression):
    text=parse_numeric(expression).expression
    def key(node):
        if isinstance(node,ast.Constant):
            value=Fraction(ast.get_source_segment(text,node))
            return 'number',value.numerator,value.denominator
        if isinstance(node,ast.UnaryOp):
            child=key(node.operand)
            if isinstance(node.op,ast.UAdd):return child
            if child[0]=='number':return 'number',-child[1],child[2]
            return 'neg',child
        return type(node.op).__name__,key(node.left),key(node.right)
    return key(ast.parse(text,mode='eval').body)
