import re
import ast
import operator as op

# supported operators for calculator
_allowed_operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
    ast.Mod: op.mod,
}


class MathEvaluator(ast.NodeVisitor):
    def visit(self, node):
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            left = self.visit(node.left)
            right = self.visit(node.right)
            op_type = type(node.op)
            if op_type in _allowed_operators:
                return _allowed_operators[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self.visit(node.operand)
            op_type = type(node.op)
            if op_type in _allowed_operators:
                return _allowed_operators[op_type](operand)
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def safe_eval(expr: str) -> str:
    """Evaluate a mathematical expression safely and return result as string."""
    try:
        tree = ast.parse(expr, mode="eval")
        evaluator = MathEvaluator()
        result = evaluator.visit(tree)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


_math_regex = re.compile(r"^[0-9\s\+\-\*\/\%\(\)\.]+$")

def is_math_expression(text: str) -> bool:
    """Determine if text looks like a math expression."""
    text = text.strip()
    return bool(_math_regex.match(text))
