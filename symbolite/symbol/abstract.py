from symbolite.core.operands import Named, OperandMixin, dataclass


@dataclass
class Symbol(Named, OperandMixin):
    """A symbol."""

    namespace = "libsymbol"
