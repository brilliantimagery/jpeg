#![feature(backtrace)]
#![feature(box_into_inner)]
#![feature(test)]

#[cfg(test)]
mod test_utils;

mod decoder;
mod encoder;
mod jpeg_errors;
mod jpeg_utils;
pub use jpeg_utils::{Format, Precision};

pub fn decode_image(encoded_image: Vec<u8>) -> Result<Vec<u32>, jpeg_errors::JpegDecoderError> {
    decoder::decode(encoded_image)
}

pub fn encode_image(
    raw_image: Vec<u32>,
    width: usize,
    height: usize,
    format: Format,
    precision: Precision,
) -> Vec<u8> {
    encoder::encode(raw_image, width as u16, height as u16, format, precision)
}

#[cfg(test)]
mod tests {
    extern crate test;

    use crate::test_utils;

    use super::*;

    #[test]
    fn decode_image_test() {
        let encoded_image = test_utils::get_file_as_byte_iter("F-18.ljpg");
        let raw_image = decode_image(encoded_image).unwrap();

        // use std::io::prelude::*;
        // use std::io::BufWriter;
        // use std::fs::File;
        // let raw_image = raw_image.iter().map(|&x| x as u8).collect::<Vec<_>>();
        // let mut buffer = BufWriter::new(File::create("raw_f18.bin").unwrap());
        // buffer.write(&raw_image[..]).unwrap();
        // buffer.flush().unwrap();
    }

    #[test]
    fn encode_image_test() {
        let raw_image = test_utils::get_file_as_byte_iter("raw-F-18.bin");
        let raw_image = raw_image.iter().map(|&m| m as u32).collect::<Vec<u32>>();
        let raw_image = encode_image(raw_image, 320, 240, Format::Lossless, Precision::Sixteen);

        // use std::io::prelude::*;
        // use std::io::BufWriter;
        // use std::fs::File;
        // let raw_image = raw_image.iter().map(|&x| x as u8).collect::<Vec<_>>();
        // let mut buffer = BufWriter::new(File::create("raw_f18.bin").unwrap());
        // buffer.write(&raw_image[..]).unwrap();
        // buffer.flush().unwrap();
    }
}
