#![feature(test)]

mod decoder;
mod encoder;

pub fn decode_image(encoded_image: Vec<u8>) {
    decoder::decode(encoded_image);
}

pub fn encode_image(raw_image: Vec<u8>, width: usize, height:usize, format: String) {
    encoder::JPEGEncoder::from_rgb(raw_image, width, height, format);
}
