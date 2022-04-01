import typing

import beartype.vale
import numpy
import numpy.typing
from typing_extensions import Annotated

ScalarType = typing.TypeVar("ScalarType", bound=numpy.generic, covariant=True)

f8 = numpy.float64
u4 = numpy.uint32
i8 = numpy.int64

Float = typing.Union[numpy.float64, float]
Int = typing.Union[numpy.int64, int]

# Mat = Annotated[
#     numpy.NDArray[numpy.ScalarType], beartype.vale.IsAttr["ndim", beartype.vale.IsEqual[2]]
# ]
# Vec = Annotated[
#     numpy.NDArray[numpy.ScalarType], beartype.vale.IsAttr["ndim", beartype.vale.IsEqual[1]]
# ]
Mat = Annotated[
    numpy.ndarray[typing.Any, numpy.dtype[ScalarType]],
    beartype.vale.IsAttr["ndim", beartype.vale.IsEqual[2]],
]
Vec = Annotated[
    numpy.ndarray[typing.Any, numpy.dtype[ScalarType]],
    beartype.vale.IsAttr["ndim", beartype.vale.IsEqual[1]],
]
