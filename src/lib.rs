mod decoder;

pub fn decode_image(encoded_image: Vec<u8>) {
    decoder::decode(encoded_image)
}
