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
