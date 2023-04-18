# symbolite: a minimalistic symbolic python package

______________________________________________________________________

Symbolite allows you to create symbolic mathematical
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
>>> expr2 = expr1.subs_by_name(x=5, y=2)
>>> print(expr2)
(5 + (3 * 2))
```

The output is still a symbolic expression, which you can evaluate:

```python
>>> expr2.eval()
11
```

Notice that we also got a warning (`No libsl provided, defaulting to Python's standard library.`).
This is because evaluating an expression requires a actual library implementation,
name usually as `libsl`. The default one just uses python's math module.

You can avoid this warning by explicitely providing an `libsl` implementation.

```python
>>> from symbolite.impl import libstd
>>> expr2.eval(libstd)
11
```

You can also import it with the right name and it will be found

```python
>>> from symbolite.impl import libstd as libsl
>>> expr2.eval()
11
```

In addition to the `Symbol` class, there is also a `Scalar` class available
to represent integer, floats or complex numbers.

```python
>>> from symbolite import Scalar
>>> x = Scalar("x")
>>> y = Scalar("y")
>>> expr1 = x + 3 * y
>>> print(expr1)
(x + (3 * y))
```

Mathematical functions that operate on scalars are available in the `scalar` module.

```python
>>> from symbolite import scalar
>>> expr3 = 3. * scalar.cos(0.5)
>>> print(expr3)
(3.0 * scalar.cos(0.5))
```

(Functions are named according to the python math module).
Again, this is a symbolic expression until evaluated.

```python
>>> expr3.eval()
2.6327476856711
```

Two other implementations are provided: NumPy and SymPy:

```python
>>> from symbolite.impl import libnumpy
>>> expr3.eval(libsl=libnumpy)
2.6327476856711
>>> from symbolite.impl import libsympy
>>> expr3.eval(libsl=libsympy)
2.6327476856711
```

(notice that the way that the different libraries round and
display may vary)

In general, all symbols must be replaced by values in order
to evaluate an expression. However, when using an implementation
like SymPy that contains a Scalar object you can still evaluate.

```python
>>> from symbolite.impl import libsympy as libsl
>>> (3. * scalar.cos(x).eval(libsl))
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
