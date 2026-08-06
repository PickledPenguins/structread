"""The expression engine, and the slice parser that shares its philosophy.

Schemas size arrays and locate records from values already read, for example
``shape: [$width * $height]`` or ``seek: $header.data_offset``. Both the
arithmetic here and the numpy-style slice strings are parsed by hand rather
than handed to ``eval``, so a schema file is inert data and can never execute
code.

An expression is compiled once, when the schema is validated, with its
references already resolved to full field paths. Evaluating one at read time is
therefore a dictionary lookup and some arithmetic, and cannot fail for scope
reasons - those failures all happen before any file is opened.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, cast

import numpy as np

from core import DataError, SchemaError

# --------------------------------------------------------------------------
# Expression parsing
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""\s*(?:
        (?P<ref>\$[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)
      | (?P<hex>0[xX][0-9a-fA-F]+)
      | (?P<num>\d+(?:\.\d+)?)
      | (?P<op>[-+*/()])
    )""",
    re.VERBOSE,
)


def _tokenize(expr: str) -> list[tuple[str, str]]:
    """Split an expression into (kind, text) tokens."""
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(expr):
        match = _TOKEN_RE.match(expr, pos)
        if not match:
            if not expr[pos:].strip():
                break
            raise SchemaError(f"invalid character in expression {expr!r} at offset {pos}")
        kind = cast("str", match.lastgroup)
        tokens.append((kind, match.group(kind)))
        pos = match.end()
    if not tokens:
        raise SchemaError(f"empty expression: {expr!r}")
    return tokens


class _Parser:
    """Recursive-descent parser producing a small expression tree.

    Grammar:
        expr    := term (('+' | '-') term)*
        term    := factor (('*' | '/') factor)*
        factor  := '-'? primary
        primary := NUMBER | REFERENCE | '(' expr ')'
    """

    def __init__(self, tokens: list[tuple[str, str]], source: str,
                 resolve: Callable[[str], str]):
        self.tokens = tokens
        self.pos = 0
        self.source = source
        self.resolve = resolve

    def peek(self) -> tuple[str, str] | None:
        """Return the next token without consuming it."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> tuple[str, str]:
        """Consume and return the next token."""
        if self.pos >= len(self.tokens):
            raise SchemaError(f"unexpected end of expression: {self.source!r}")
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def parse(self) -> tuple:
        """Parse the token stream, rejecting anything left over."""
        node = self.expr()
        if self.pos != len(self.tokens):
            raise SchemaError(f"trailing tokens in expression: {self.source!r}")
        return node

    def expr(self) -> tuple:
        """Parse an additive expression."""
        node = self.term()
        while (token := self.peek()) and token[1] in ("+", "-"):
            self.take()
            node = ("op", token[1], node, self.term())
        return node

    def term(self) -> tuple:
        """Parse a multiplicative expression."""
        node = self.factor()
        while (token := self.peek()) and token[1] in ("*", "/"):
            self.take()
            node = ("op", token[1], node, self.factor())
        return node

    def factor(self) -> tuple:
        """Parse an optionally negated primary."""
        token = self.peek()
        if token and token[1] == "-":
            self.take()
            return ("neg", self.factor())
        return self.primary()

    def primary(self) -> tuple:
        """Parse a literal, a reference, or a parenthesised expression."""
        kind, text = self.take()
        if kind == "num":
            return ("num", float(text) if "." in text else int(text))
        if kind == "hex":
            return ("num", int(text, 16))
        if kind == "ref":
            return ("ref", self.resolve(text[1:]))
        if text == "(":
            node = self.expr()
            if self.take()[1] != ")":
                raise SchemaError(f"expected ')' in expression: {self.source!r}")
            return node
        raise SchemaError(f"unexpected token {text!r} in expression: {self.source!r}")


class Expression:
    """A compiled schema expression with its references already resolved."""

    def __init__(self, source: str, node: tuple, refs: list[str]):
        self.source = source
        self.node = node
        self.refs = refs

    def __repr__(self) -> str:
        """Show the original text, which is what appears in diagnostics."""
        return f"Expression({self.source!r})"

    def evaluate(self, values: dict[str, Any]) -> int | float:
        """Evaluate against a mapping of full field path to value."""
        return self._eval(self.node, values)

    def _eval(self, node: tuple, values: dict[str, Any]) -> int | float:
        """Walk one node of the expression tree."""
        kind = node[0]
        if kind == "num":
            return node[1]
        if kind == "neg":
            return -self._eval(node[1], values)
        if kind == "ref":
            path = node[1]
            if path not in values:
                raise DataError(f"expression {self.source!r} refers to {path}, which has no value")
            value = values[path]
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer,
                                                                np.floating)):
                raise DataError(
                    f"expression {self.source!r}: {path} is not numeric (value {value!r})"
                )
            return int(value) if isinstance(value, (int, np.integer)) else float(value)
        left = self._eval(node[2], values)
        right = self._eval(node[3], values)
        operator = node[1]
        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        if operator == "*":
            return left * right
        if right == 0:
            raise SchemaError(f"division by zero in expression: {self.source!r}")
        if isinstance(left, int) and isinstance(right, int) and left % right == 0:
            return left // right
        return left / right

    def as_int(self, values: dict[str, Any], what: str) -> int:
        """Evaluate and require a non-negative whole number."""
        value = self.evaluate(values)
        if isinstance(value, float):
            if not value.is_integer():
                raise DataError(f"{what}: {self.source!r} evaluated to {value}, not a whole number")
            value = int(value)
        if value < 0:
            raise DataError(f"{what}: {self.source!r} evaluated to {value}, which is negative")
        return int(value)


def compile_expression(raw: Any, resolve: Callable[[str], str], what: str) -> Expression:
    """Compile a schema value that may be a literal number or an expression."""
    if isinstance(raw, bool):
        raise SchemaError(f"{what}: expected a number or expression, got a boolean")
    if isinstance(raw, int):
        return Expression(str(raw), ("num", raw), [])
    if not isinstance(raw, str):
        raise SchemaError(f"{what}: expected a number or expression string, "
                          f"got {type(raw).__name__}")
    refs: list[str] = []

    def capture(name: str) -> str:
        """Resolve one reference and record the path it resolved to."""
        path = resolve(name)
        refs.append(path)
        return path

    return Expression(raw, _Parser(_tokenize(raw), raw, capture).parse(), refs)


# --------------------------------------------------------------------------
# Slices
# --------------------------------------------------------------------------

_INT_RE = re.compile(r"^-?\d+$")


def parse_slice(text: str) -> tuple:
    """Parse a numpy-style index string such as "[:, ::2]" into index objects.

    Parsed by hand rather than with eval() so that schema files stay inert.
    """
    body = text.strip()
    if body.startswith("[") and body.endswith("]"):
        body = body[1:-1]
    if not body.strip():
        return (slice(None),)
    parts: list[Any] = []
    for raw in body.split(","):
        token = raw.strip()
        if token in ("...", "Ellipsis"):
            parts.append(Ellipsis)
        elif ":" in token:
            pieces = token.split(":")
            if len(pieces) > 3:
                raise SchemaError(f"invalid slice component {token!r} in {text!r}")
            bounds: list[int | None] = []
            for piece in pieces:
                stripped = piece.strip()
                if stripped == "":
                    bounds.append(None)
                elif _INT_RE.match(stripped):
                    bounds.append(int(stripped))
                else:
                    raise SchemaError(f"invalid slice component {token!r} in {text!r}")
            while len(bounds) < 3:
                bounds.append(None)
            parts.append(slice(bounds[0], bounds[1], bounds[2]))
        elif _INT_RE.match(token):
            parts.append(int(token))
        else:
            raise SchemaError(f"invalid slice component {token!r} in {text!r}")
    return tuple(parts)


def apply_slice(array: np.ndarray, selector: tuple | None, where: str) -> np.ndarray:
    """Apply a parsed slice to an array, reporting an unusable one clearly."""
    if selector is None:
        return array
    try:
        return array[selector]
    except (IndexError, ValueError, TypeError) as exc:
        raise DataError(f"{where}: slice cannot be applied to shape "
                        f"{list(array.shape)}: {exc}") from exc
