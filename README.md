# symbolite: a minimalistic symbolic python package

______________________________________________________________________

Symbolite allows you tu create symbolic mathematical
expressions. Just create a symbol (or more) and operate with them as you
will normally do in Python.

```python
>>> from symbolite import Symbol
>>> x = Symbol("x")
>>> y = Symbol("y")
>>> expr1 = x + 3 * y
>>> print(expr1)
(x + (3 * y))
```

You can easily replace the symbols by the desired value.

```python
>>> expr2 = expr1.replace_by_name(x=5, y=2)
>>> print(expr2)
(5 + (3 * 2))
```

The output is still a symbolic expression, which you can evaluate:

```python
>>> expr2.eval()
11
```

Notice that we also got a warning (`No libsl provided, defaulting to 'math'`).
This is because evaluating an expression requires a actual library implementation,
name usually as `libsl`. The default one just uses python's math module.

You can avoid this warning by explicitely providing an `libsl` implementation.

```python
>>> from symbolite.libimpl import math as _math
>>> expr2.eval(_math)
11
```

or importing it as `libsl` and let symbolite

```python
>>> from symbolite.libimpl import math as libsl
>>> expr2.eval()
11
```

The cool thing about this is that you can use a different implementation
but let's not get too much ahead of ourselves.

Mathematical functions are available in the `lib` module.

```python
>>> from symbolite import lib
>>> expr3 = 3. * lib.cos(0.5)
>>> print(expr3)
(3.0 * libsl.cos(0.5))
```

(Functions are named according to the python math module).
Again, this is a symbolic expression until evaluated.

```python
>>> expr3.eval()
2.6327476856711
```

Two other implementations are provided: NumPy and SymPy:

```python
>>> from symbolite.libimpl import numpy as libsl
>>> expr3.eval()
2.6327476856711
>>> from symbolite.libimpl import sympy as libsl
>>> expr3.eval()
2.6327476856711
```

(notice that the way that the different libraries round and
display may vary)

In general, all symbols must be replaced by values in order
to evaluate an expression. However, when using an implementation
like SymPy that contains a Symbol object you can still evaluate.

```python
>>> (3. * lib.cos(x).eval())
3.0*cos(x)
```

which is actually a SymPy expression with a SymPy symbol (`x`).

### Installing:

```bash
pip install -U symbolite
```

### FAQ

**Q: Is symbolite a replacement for SymPy?**

**A:** No

**Q: Does it aim to be a replacement for SymPy in the future?**

**A:** No
