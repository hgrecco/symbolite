# symbolite: a minimalistic symbolic python package

______________________________________________________________________

Symbolite allows you to create symbolic mathematical
expressions. Just create a symbol (or more) and operate with them as you
will normally do in Python.

```python
>>> from symbolite.symbol.abstract import Symbol
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

Notice that we also got a warning (`No libsymbol provided, defaulting to 'math'`).
This is because evaluating an expression requires a actual library implementation,
name usually as `libsl`. The default one just uses python's math module.

You can avoid this warning by explicitely providing an `libsl` implementation.

```python
>>> from symbolite.symbol import default
>>> expr2.eval(libsymbol=default)
11
```

You can also import it with the right name and it will be found

```python
>>> from symbolite.symbol import default as libsymbol
>>> expr2.eval()
11
```

I guess you want to do some math now, right? 70 math functions
are defined can be used through `symbolite-scalar` with
implementations using the Python math module, numpy and scipy.
Check it out!

### Installing:

```bash
pip install -U symbolite
```

### FAQ

**Q: Is symbolite a replacement for SymPy?**

**A:** No

**Q: Does it aim to be a replacement for SymPy in the future?**

**A:** No
