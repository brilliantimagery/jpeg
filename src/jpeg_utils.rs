pub(crate) enum Marker {
    SOF0 = 0xFFC0, // Baseline DCT
    SOF3 = 0xFFC3, // Lossless Huffman Encoding
    DHT = 0xFFC4,  // Define Huffman table(s)
    SOI = 0xFFD8,  // Start of image
    EOI = 0xFFD9,  // End of image
    SOS = 0xFFDA,  // Start of scan
    DQT = 0xFFDB,  // Define quantization table(s)
    APP = 0xFFE0,  //Reserved for application segments
    APPn = 0xFFEF, //Reserved for application segments
}

impl Marker {
    pub(crate) fn to_marker(numb: u16) -> Option<Self> {
        use Marker::*;

        match numb {
            0xFFC0 => Some(SOF0),
            0xFFC3 => Some(SOF3),
            0xFFC4 => Some(DHT),
            0xFFD8 => Some(SOI),
            0xFFD9 => Some(EOI),
            0xFFDA => Some(SOS),
            0xFFDB => Some(DQT),
            0xFFE0 => Some(APP),
            0xFFEF => Some(APPn),
            _ => None,
        }
    }
}