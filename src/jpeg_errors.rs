use std::backtrace::Backtrace;

// #[derive(thiserror::Error, Debug)]
// #[error("There was a Jpeg decoder error.")]
// pub struct JpegDecoderError {
//     #[from]
//     source: Box<dyn std::error::Error>,

//     // backtrace: Backtrace,
// }

#[derive(thiserror::Error, Debug)]
#[error("There was a Jpeg decoder error.")]
pub enum JpegDecoderError {
    BoundsError(OutOfBoundsError),
    MagicError(BadMagicNumberError),
    HuffmanError(HuffmanDecodingError),
    DecodeError(DecodeError),
    // backtrace: Backtrace,
}

impl From<BadMagicNumberError> for JpegDecoderError {
    fn from(err: BadMagicNumberError) -> Self {
        Self::MagicError(err)
    }
}

impl From<OutOfBoundsError> for JpegDecoderError {
    fn from(err: OutOfBoundsError) -> Self {
        Self::BoundsError(err)
    }
}

impl From<HuffmanDecodingError> for JpegDecoderError {
    fn from(err: HuffmanDecodingError) -> Self {
        Self::HuffmanError(err)
    }
}

impl From<DecodeError> for JpegDecoderError {
    fn from(err: DecodeError) -> Self {
        Self::DecodeError(err)
    }
}

#[derive(thiserror::Error, Debug)]
#[error("The data doesn't have the expected Jpeg magic number.")]
pub struct BadMagicNumberError {
    #[from]
    pub source: OutOfBoundsError,
    // backtrace: Backtrace,
}

#[derive(thiserror::Error, Debug)]
#[error("Necessery informaiton seems to be missing from the Jpeg.")]
pub struct MissingInfoError;

#[derive(thiserror::Error, Debug)]
#[error("A vec, array, or iter tired to access something that didn't exist.")]
pub struct OutOfBoundsError;

#[derive(thiserror::Error, Debug)]
#[error("There was an error decoding a Huffman code")]
pub enum HuffmanDecodingError {
    Default,
    BoundsError(OutOfBoundsError),
}

impl From<OutOfBoundsError> for HuffmanDecodingError {
    fn from(err: OutOfBoundsError) -> Self {
        Self::BoundsError(err)
    }
}

#[derive(thiserror::Error, Debug)]
#[error("There was an error while trying to decode a Jpeg")]
pub enum DecodeError {
    Default,
    HuffmanError(HuffmanDecodingError),
    BoundsError(OutOfBoundsError),
}

impl From<HuffmanDecodingError> for DecodeError {
    fn from(err: HuffmanDecodingError) -> Self {
        Self::HuffmanError(err)
    }
}

impl From<OutOfBoundsError> for DecodeError {
    fn from(err: OutOfBoundsError) -> Self {
        Self::BoundsError(err)
    }
}
