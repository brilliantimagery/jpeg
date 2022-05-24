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
    // backtrace: Backtrace,
}

impl From<BadMagicNumberError> for JpegDecoderError {
    fn from(err: BadMagicNumberError) -> Self {
        JpegDecoderError::MagicError(err)
    }
}

impl From<OutOfBoundsError> for JpegDecoderError {
    fn from(err: OutOfBoundsError) -> Self {
        JpegDecoderError::BoundsError(err)
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
