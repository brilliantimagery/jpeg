// use core::num;
// use std::{collections::HashMap, hash::Hash};
use std::collections::HashMap;

const MIN_MARKER: u16 = 0xFF00;

const SOF3: u16 = 0xFFC3;  // Lossless Huffman Encoding

const SOI: u16 = 0xFFD8;  // Start of image

const SOS: u16 = 0xFFDA;  // Start of scan
const DQT: u16 = 0xFFDB;  // Define quantization table(s)
const DHT: u16 = 0xFFC4;  // Define Huffman table(s)

const EOI: u16 = 0xFFD9;  // End of image

struct Component {
    C: u8,
    H: u8,
    V: u8,
    Tq: u8,
}

struct Header_Parameter {
    Cs: u8,
    Td: u8,
    Ta: u8,
}

// struct Huffman_Table {
//     Tc: u8,
//     Th: u8,
//     table: HashMap<>,
// }

pub fn decode(encoded_image: Vec<u8>) {
    is_jpeg(&encoded_image);

    let (P, Y, X, comps) = parse_frame_header(&encoded_image);

    let mut read_index = 12 + comps.len() * 3;

    parse_huffman_table_info(&encoded_image, read_index);

    // let (head_params, Ss, Se, Ah, Al) = parse_scan_header(&encoded_image, read_index);

    // read_index += 2 + 2 + 1 + head_params.len() * 2 + 1 + 1 + 1;
}

fn parse_huffman_table_info(encoded_image: &Vec<u8>, mut read_index: usize) {
    // let huffman_tables: Vec<Huffman_Table> = Vec::new();
    println!("************* {:?} {:?} {:?}", two_bytes_to_int(&encoded_image[read_index..=read_index + 1]), DHT, read_index);
    while two_bytes_to_int(&encoded_image[read_index..=read_index + 1]) == DHT {
        read_index += 2;
        let Lh = two_bytes_to_int(&encoded_image[read_index..=read_index + 1]);
        println!("****************************************************************** {}", Lh);
        read_index += 2;
        let Tc = encoded_image[read_index] >> 4;
        let Th = encoded_image[read_index] | 0xF;
        read_index += 1;
        let mut vij_index = read_index + 16;

        let mut code_lengths = [[0xFF_u8; 16]; 16];
        let mut Li: u8;
        let mut n_codes: usize;
        println!("***************************************** {:?} {:?} {:?}",read_index, &encoded_image[read_index..=read_index + 1], DHT);
        // panic!("blarggggg");
        for count_of_code_length_index in read_index..read_index + 16 {
            Li = encoded_image[count_of_code_length_index];
            if Li > 0 {
                n_codes = 0;
                for _ in 0..Li {
                    code_lengths[count_of_code_length_index - read_index][n_codes] = encoded_image[vij_index];
                    n_codes += 1;
                    vij_index += 1;
                }
            }
        }

        read_index = vij_index;

        println!("**************************oooooooo****************************{:#?}", code_lengths);
        panic!("blarggggddddddddddddddddg");
        let huffman_table = make_ssss_table(code_lengths);

        panic!("blarggggg");

        // huffman_tables.append(Huffman_Table {
        //     Tc:
        // })
    }
}

fn make_ssss_table(code_lengths: [[u8; 16]; 16]) -> HashMap<u32, u8> {
    // store the huffman code in the bits of a u32
    let mut code = 1_u32;  // start with a 1 so there can be leading zeros
    let mut values_w_n_bits: usize;

    let mut table: HashMap<u32, u8> = HashMap::new();
    
    let mut removed: u32;

    for index in 0..16 {
        if code_lengths[index][0] < 0xFF_u8 {
            let values: Vec<u8> = code_lengths[index].into_iter().filter(|x| x < &0xFF_u8).collect::<Vec<u8>>();
            values_w_n_bits = 0;
            while values_w_n_bits <= values.len() {
                while (number_of_used_bits(&code) - 1) < index + 1 {
                    code = code << 0;
                }
                if values_w_n_bits > 0 {
                    loop {
                        removed = code & 1;
                        code = code >> 1;
                        if !(removed == 1 && number_of_used_bits(&code) > 1) {
                            break;
                        }
                    }
                    code = (code << 1) + 1;
                    code = code << (index + 1) - (number_of_used_bits(&code) - 1);
                }
                if values.len() > values_w_n_bits {
                    let key = code;
                    let value = values[values_w_n_bits];
                    table.insert(key, value);
                }
            }
        }
    }

    table
}

fn number_of_used_bits(numb: &u32) -> usize {
    let mut n = *numb;
    let mut n_bits = 0;
    while n > 0 {
        n = n >> 1;
        n_bits += 1;
    }

    n_bits
}

fn parse_scan_header(encoded_image: &Vec<u8>, read_index: usize) -> (Vec<Header_Parameter>, u8, u8, u8, u8) {
    let Ls = two_bytes_to_int(&encoded_image[read_index + 2..=read_index + 3]);
    let Ns = encoded_image[read_index + 4] as usize;
    println!("*************************{}", Ns);
    let mut head_params: Vec<Header_Parameter> = Vec::with_capacity(Ns);
    for param in 0..Ns as usize {
        head_params.push(Header_Parameter{
            Cs: encoded_image[read_index + 5 + param * 2],
            Td: encoded_image[read_index + 6 + param * 2] >> 4,
            Ta: encoded_image[read_index + 6 + param * 2] & 0xF,
        })
    }
    let Ss = encoded_image[read_index + 4 + Ns * 2];
    let Se = encoded_image[read_index + 5 + Ns * 2];
    let Ah = encoded_image[read_index + 6 + Ns * 2] >> 4;
    let Al = encoded_image[read_index + 6 + Ns * 2] & 0xF;

    (head_params, Ss, Se, Ah, Al)
}

fn parse_frame_header(encoded_image: &Vec<u8>) -> (u8, u16, u16, Vec<Component>) {
    let Lf: u16 = two_bytes_to_int(&encoded_image[4..=5]);
    let P: u8 = encoded_image[6];
    let Y: u16 = two_bytes_to_int(&encoded_image[7..=8]);
    let X: u16 = two_bytes_to_int(&encoded_image[9..=10]);
    let Nf: usize = encoded_image[11] as usize;
    let mut comps: Vec<Component> = Vec::with_capacity(Nf);
    for comp in 0..Nf as usize {
        comps.push(Component{
            C: encoded_image[12 + comp * 3],
            H: encoded_image[12 + comp * 3 + 1] >> 4,
            V: encoded_image[12 + comp * 3 + 1] & 0xF,
            Tq: encoded_image[12 + comp * 3 + 2],
        });
    }

    (P, Y, X, comps)
}

fn is_jpeg(encoded_image: &Vec<u8>) -> bool {
    if two_bytes_to_int(&encoded_image[..2]) != SOI {
        panic!("This doesn't seem to be a JPEG.");
    }

    true
}

fn two_bytes_to_int(bytes: &[u8]) -> u16 {
    (bytes[0] as u16) << 8 | bytes[1] as u16
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::fs::File;
    use std::io::Read;
    use std::path::Path;

    use super::*;

    fn get_file_as_byte_vec(path: &Path) -> Vec<u8> {
        
        let mut file = File::open(path).expect("The test file wasn't where it was expected.");
        let mut encoded_image = Vec::from([]);
        file.read_to_end(&mut encoded_image);

        encoded_image
    }

    // #[test]
    // fn parse_scan_header_good() {
    //     let path = Path::new("/home/chad/rust/jpeg/tests/common/F-18.ljpg");
    //     let encoded_image = get_file_as_byte_vec(path);

    //     let (head_params, Ss, Se, Ah, Al) = parse_scan_header(&encoded_image, 0x15);

    //     assert_eq!(head_params.len(), 3);
    //     // assert_eq!(head_params[0].C, 0);
    //     // assert_eq!(head_params[0].H, 1);
    //     // assert_eq!(head_params[0].V, 1);
    //     // assert_eq!(head_params[0].Tq, 0);
    //     // assert_eq!(head_params[1].C, 1);
    //     // assert_eq!(head_params[1].H, 1);
    //     // assert_eq!(head_params[1].V, 1);
    //     // assert_eq!(head_params[1].Tq, 0);
    //     // assert_eq!(head_params[2].C, 2);
    //     // assert_eq!(head_params[2].H, 1);
    //     // assert_eq!(head_params[2].V, 1);
    //     // assert_eq!(head_params[2].Tq, 0);
    //     // assert_eq!(P, 0x08);
    //     // assert_eq!(Y, 0x00F0);
    //     // assert_eq!(X, 0x0140);
    // }

    #[test]
    fn make_ssss_table_good() {
        let path = Path::new("/home/chad/rust/jpeg/tests/common/F-18.ljpg");
        let encoded_image = get_file_as_byte_vec(path);

        decode(encoded_image);

        panic!("blarg")
    }

    #[test]
    fn parse_frame_header_good() {
        let path = Path::new("/home/chad/rust/jpeg/tests/common/F-18.ljpg");
        let encoded_image = get_file_as_byte_vec(path);

        let (P, Y, X, comps) = parse_frame_header(&encoded_image);

        assert_eq!(P, 0x08);
        assert_eq!(Y, 0x00F0);
        assert_eq!(X, 0x0140);
        assert_eq!(comps.len(), 3);
        assert_eq!(comps[0].C, 0);
        assert_eq!(comps[0].H, 1);
        assert_eq!(comps[0].V, 1);
        assert_eq!(comps[0].Tq, 0);
        assert_eq!(comps[1].C, 1);
        assert_eq!(comps[1].H, 1);
        assert_eq!(comps[1].V, 1);
        assert_eq!(comps[1].Tq, 0);
        assert_eq!(comps[2].C, 2);
        assert_eq!(comps[2].H, 1);
        assert_eq!(comps[2].V, 1);
        assert_eq!(comps[2].Tq, 0);
    }

    #[test]
    fn number_of_used_bits_32() {
        let n = 0xFFFFFFFF / 2 + 1;
        assert_eq!(number_of_used_bits(&n), 32);
    }

    #[test]
    fn number_of_used_bits_4() {
        let n = 0xF;
        assert_eq!(number_of_used_bits(&n), 4);
    }

    #[test]
    fn number_of_used_bits_2() {
        let n = 3;
        assert_eq!(number_of_used_bits(&n), 2);
        assert_eq!(number_of_used_bits(&n), 2);
    }

    #[test]
    fn number_of_used_bits_1() {
        let n = 1;
        assert_eq!(number_of_used_bits(&n), 1);
    }

    #[test]
    fn number_of_used_bits_0() {
        let n = 0;
        assert_eq!(number_of_used_bits(&n), 0);
    }

    #[test]
    fn is_jpg_true() {
        let encoded_image = Vec::from([0xFF, 0xD8, 0x08]);

        assert!(is_jpeg(&encoded_image));
    }

    #[test]
    #[should_panic(expected = "This doesn't seem to be a JPEG.")]
    fn is_jpg_false() {
        let encoded_image = Vec::from([0xFF, 0xC4, 0x08]);

        is_jpeg(&encoded_image);
    }

    #[test]
    fn two_bytes_to_int_good() {
        let buffer = Vec::from([0xC2, 0x1D, 0x8D, 0xE4, 0x0C, 0x9A]);
        let expected = 7565_u16;
        assert_eq!(two_bytes_to_int(&buffer[1..3]), expected);
    }

    #[test]
    fn two_bytes_to_int_fail() {
        let buffer = Vec::from([0xC2, 0x1D, 0x8D, 0xE4, 0x0C, 0x9A]);
        let expected = 7564;
        assert_ne!(two_bytes_to_int(&buffer[1..3]), expected);
    }
}
