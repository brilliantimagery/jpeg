#![feature(backtrace)]
#![feature(test)]

#[cfg(test)]
mod test_utils;

mod decoder;
mod encoder;
mod jpeg_errors;
mod jpeg_utils;

pub fn decode_image(encoded_image: Vec<u8>) {
    decoder::decode(encoded_image);
}

pub fn encode_image(raw_image: Vec<u8>, width: usize, height: usize, format: String) {
    encoder::JPEGEncoder::from_rgb(raw_image, width, height, format);
}

#[cfg(test)]
mod tests {
    extern crate test;

    use crate::test_utils;

    use super::*;


    #[test]
    fn decode_image_test() {
        let encoded_image = test_utils::get_file_as_byte_iter("F-18.ljpg");
        let raw_image = decode_image(encoded_image);
    }
}
