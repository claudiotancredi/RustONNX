#[derive(Debug)]
pub enum OnnxError{

    TensorNotFound(String),

    ShapeMismatch(String),

    AxisOutOfBounds(String),

    ONNXParserError(String),

    EmptyContainer(String),

    UnsupportedOperator(String),

    RuntimeError(String),

    ShapeError(String),

    DimensionalityError(String)

}