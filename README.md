# symbolite: a minimalistic symbolic python package

______________________________________________________________________

Symbolite allows you to create symbolic mathematical
expressions. Just create a symbol (or more) and operate with them as you
will normally do in Python.

```python
>>> from symbolite.abstract import scalar
>>> x = scalar.Scalar("x")
>>> y = scalar.Scalar("y")
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
>>> from symbolite.impl.scalar import default
>>> expr2.eval(libscalar=default)
11
```

The cool thing about this is that you can use a different implementation
but let's not get too much ahead of ourselves.

Mathematical functions are available in the `lib` module.

```python
>>> from symbolite.abstract import scalar
>>> expr3 = 3. * scalar.cos(0.5)
>>> print(expr3)
(3.0 * libscalar.cos(0.5))
```

(Functions are named according to the python math module).
Again, this is a symbolic expression until evaluated.

```python
>>> expr3.eval()
2.6327476856711
```

Two other implementations are provided: NumPy and SymPy:

```python
>>> from symbolite.impl.scalar import numpy as libscalar
>>> expr3.eval(libscalar=libscalar)
2.6327476856711
>>> from symbolite.impl.scalar import sympy as libscalar
>>> expr3.eval(libscalar=libscalar)
2.6327476856711
```

(notice that the way that the different libraries round and
display may vary)

In general, all symbols must be replaced by values in order
to evaluate an expression. However, when using an implementation
like SymPy that contains a Scalar object you can still evaluate.

```python
>>> from symbolite.impl.scalar import sympy as libscalar
>>> (3. * scalar.cos(x).eval(libscalar=libscalar))
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
