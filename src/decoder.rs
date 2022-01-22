use std::collections::HashMap;

const SOF3: u16 = 0xFFC3;  // Lossless Huffman Encoding

const SOI: u16 = 0xFFD8;  // Start of image

const SOS: u16 = 0xFFDA;  // Start of scan
const DQT: u16 = 0xFFDB;  // Define quantization table(s)
const DHT: u16 = 0xFFC4;  // Define Huffman table(s)

const EOI: u16 = 0xFFD9;  // End of image

struct Context {
    r_a: u32,
    r_b: u32,
    r_c: u32,
    r_ix: u32,
}

struct ContextContext<'a> {
    component: &'a usize,
    x_position: usize,
    y_position: usize,
    width: &'a usize,
    numb_of_components: &'a usize,
    point_tranform: &'a u8, 
    p_: &'a u8, 
    img: &'a Vec<u32>,
}

impl ContextContext<'_> {
    fn r_a (&self) -> u32 {
        self.img[(self.x_position-1) * self.numb_of_components + self.y_position * self.width * self.numb_of_components + self.component]
    }
    fn r_b (&self) -> u32 {
        self.img[self.x_position * self.numb_of_components + (self.y_position - 1) * self.width * self.numb_of_components + self.component]
    }
    fn r_c (&self) -> u32 {
        self.img[(self.x_position-1) * self.numb_of_components + (self.y_position - 1) * self.width * self.numb_of_components + self.component]
    }
    fn r_ix (&self) -> u32 {
        1 << (self.p_ - self.point_tranform - 1)
    }
}

struct Component {
    h_: u8,
    v_: u8,
    t_q: u8,
}

struct FrameHeader {
    p_: u8,
    y_: u16,
    x_: u16,
    components: HashMap<u8, Component>,
}

struct ScanHeader {
    head_params: HashMap<u8, HeaderParameter>,
    s_s: u8,
    s_e: u8,
    a_h: u8,
    a_l: u8,
}

struct HeaderParameter {
    t_d: u8,
    t_a: u8,
}

struct SSSSTable {
    t_c: u8,
    t_h: u8,
    table: HashMap<u32, u8>,
    min_code_length: usize,
    max_code_length: usize,
}

pub fn decode(encoded_image: Vec<u8>) {
    is_jpeg(&encoded_image);
    let mut read_index: usize = 2;
    
    let mut frame_header: FrameHeader = FrameHeader {p_:0, x_:0, y_:0, components:HashMap::new()};

    // let mut ssss_tables: Vec<SSSSTable> = Vec::with_capacity(2);
    let mut ssss_tables: HashMap<usize, SSSSTable> = HashMap::new();
    while read_index < encoded_image.len() {
        let possible_marker: u16 = bytes_to_int_two(&encoded_image[read_index..read_index+2]);
        match possible_marker {
            SOF3 => {
                let header_info = parse_frame_header(&encoded_image, read_index); 

                frame_header = header_info.0;
                read_index = header_info.1;
            },
            DHT => {
                let (huffman_table, read_index) = get_huffman_info(&encoded_image, read_index);
                // ssss_tables.push(huffman_table);
                ssss_tables.insert(huffman_table.t_h as usize, huffman_table);
            },
            SOS => {
                let (scan_header, read_index) = parse_scan_header(&encoded_image, read_index);
                decode_image(&encoded_image, &frame_header, scan_header, &ssss_tables, read_index);
            },
            marker if marker > 0xFF00 && marker < 0xFFFF => panic!("Unimplimented marker!"),
            _ => { read_index += 1; },
        }
    }
}

fn get_huffman_info(encoded_image: &Vec<u8>, read_index: usize) -> (SSSSTable, usize) {
    let (t_c, t_h, code_lengths, read_index) = parse_huffman_info(&encoded_image, read_index);

    let (table, min_code_length, max_code_length) = make_ssss_table(code_lengths);

    (SSSSTable {
        t_c,
        t_h,
        table,
        min_code_length,
        max_code_length,
    }, read_index)
}

fn parse_huffman_info(encoded_image: &Vec<u8>, mut read_index: usize) -> (u8, u8, [[u8; 16]; 16], usize) {
    read_index += 2;
    let _l_h = bytes_to_int_two(&encoded_image[read_index..=read_index + 1]);
    read_index += 2;
    let t_c = encoded_image[read_index] >> 4;
    let t_h = encoded_image[read_index] & 0xF;
    read_index += 1;
    let mut vij_index = read_index + 16;
    let mut code_lengths = [[0xFF_u8; 16]; 16];
    for code_length_index in 0..16 {
        let l_i: u8 = encoded_image[read_index + code_length_index];
        if l_i > 0 {
            let mut n_codes: usize = 0;
            for _ in 0..l_i {
                code_lengths[code_length_index][0];
                code_lengths[0][n_codes];
                encoded_image[vij_index];
                code_lengths[code_length_index][n_codes] = encoded_image[vij_index];
                n_codes += 1;
                vij_index += 1;
            }
        }
    }

    (t_c, t_h, code_lengths, vij_index)
}

fn make_ssss_table(code_lengths: [[u8; 16]; 16]) -> (HashMap<u32, u8>, usize, usize) {
    // storing the huffman code in the bits of a u32
    // the code is preceided by a 1 so there can be leading zeros
    let mut code: u32 = 1;
    let mut table: HashMap<u32, u8> = HashMap::new();
    for index in 0..16 {
        if code_lengths[index][0] < 0xFF_u8 {
            let values: Vec<u8> = code_lengths[index].into_iter().filter(|x| x < &0xFF_u8).collect::<Vec<u8>>();
            let mut values_w_n_bits: usize = 0;
            while values_w_n_bits <= values.len() {
                code = code << (index + 1 - (number_of_used_bits(&code) - 1));
                if values_w_n_bits > 0 {
                    loop {
                        let removed: u32 = code & 1;
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
                values_w_n_bits += 1;
            }
        }
    }

    let mut min_code_length: usize = 100;
    let mut max_code_length: usize = 0;

    for v in table.keys() {
        let length = number_of_used_bits(v) - 1;
        if length < min_code_length {
            min_code_length = length;
        }
        if length > max_code_length {
            max_code_length = length;
        }
    }

    (table, min_code_length, max_code_length)
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

fn parse_scan_header(encoded_image: &Vec<u8>, mut read_index: usize) -> (ScanHeader, usize) {
    read_index += 2;
    let _l_s = bytes_to_int_two(&encoded_image[read_index..read_index + 2]);
    read_index += 2;
    let n_s = encoded_image[read_index] as usize;
    read_index += 1;
    let mut head_params: HashMap<u8, HeaderParameter> = HashMap::new();
    for _ in 0..n_s {
        head_params.insert(
            encoded_image[read_index],
            HeaderParameter {
                t_d: encoded_image[read_index + 1] >> 4,
                t_a: encoded_image[read_index + 1] & 0xF,
            });
        read_index += 2;
    }
    let s_s = encoded_image[read_index];
    read_index += 1;
    let s_e = encoded_image[read_index];
    read_index += 1;
    let a_h = encoded_image[read_index] >> 4;
    let a_l = encoded_image[read_index] & 0xF;
    read_index += 1;

    (ScanHeader {
        head_params,
        s_s,
        s_e,
        a_h,
        a_l,
    },
    read_index)
}

fn parse_frame_header(encoded_image: &Vec<u8>, mut read_index: usize) -> (FrameHeader, usize) {
    read_index += 2;
    let _l_f: u16 = bytes_to_int_two(&encoded_image[read_index..read_index + 2]);
    read_index += 2;
    let p_: u8 = encoded_image[read_index];
    read_index += 1;
    let y_: u16 = bytes_to_int_two(&encoded_image[read_index..read_index + 2]);
    read_index += 2;
    let x_: u16 = bytes_to_int_two(&encoded_image[read_index..read_index + 2]);
    read_index += 2;
    let n_f: usize = encoded_image[read_index] as usize;
    read_index += 1;
    let mut components: HashMap<u8, Component> = HashMap::new();
    for _ in 0..n_f as usize {
        components.insert(
            encoded_image[read_index],
            Component {
                h_: encoded_image[read_index + 1] >> 4,
                v_: encoded_image[read_index + 1] & 0xF,
                t_q: encoded_image[read_index + 2],
            });
        read_index += 3
    }

    (FrameHeader {
        p_,
        y_,
        x_,
        components,
    },
    read_index)
}

fn is_jpeg(encoded_image: &Vec<u8>) -> bool {
    if bytes_to_int_two(&encoded_image[..2]) != SOI {
        panic!("This doesn't seem to be a JPEG.");
    }
    true
}

fn bytes_to_int_two(bytes: &[u8]) -> u16 {
    (bytes[0] as u16) << 8 | bytes[1] as u16
}

fn decode_image(encoded_image: &Vec<u8>, frame_header: &FrameHeader, scan_header: 
    ScanHeader, ssss_tables: &HashMap<usize, SSSSTable>, read_index: usize) -> Vec<u32> {
        
    let width = frame_header.x_ as usize;
    let height = frame_header.y_ as usize;

    // panic!("placeholder mumbo jumbo");
    let numb_of_components = frame_header.components.len();
    let mut raw_image: Vec<u32> = Vec::with_capacity(width * height * numb_of_components);
    let image_start_index = read_index;

    let (image_bits, read_index) = get_image_data_without_stuffed_zero_bytes(&encoded_image, read_index);
    let bit_read_index: usize = 0;

    while bit_read_index < image_bits.len() {
        let image_index = read_index - image_start_index;
        let component = image_index % numb_of_components;
        let context = ContextContext {
            component: &component,
            x_position: (image_index / numb_of_components) % width,
            y_position: (image_index / numb_of_components) / width,
            width: &width,
            numb_of_components: &&numb_of_components,
            point_tranform: &scan_header.a_h, 
            p_: &frame_header.p_, 
            img: &raw_image,
        };
        let p_x = get_prediction(context, &scan_header.s_s);
        let (pixel_delta, bit_read_index) = get_huffmaned_value(ssss_tables.get(&component).unwrap(), &image_bits, bit_read_index);
        raw_image.push(((p_x as i32 + pixel_delta) & ((1 << frame_header.p_) - 1)) as u32);
    }

    raw_image
}

fn  get_prediction(context: ContextContext, predictor: &u8) -> u32 {
    let mut used_predictor: u8 = *predictor;

    if context.x_position == 0 {
        if context.y_position == 0 {
            used_predictor = 8;
        } else {
            used_predictor = 2;
        }
    } else if context.y_position == 0 {
        used_predictor = 1;        
    }

    match used_predictor {
        0 => 0,
        1 => context.r_a(),
        2 => context.r_b(),
        3 => context.r_c(),
        4 => { context.r_a() + context.r_b() - context.r_c() },
        5 => context.r_a() + ((context.r_b() - context.r_c()) >> 1),
        6 => context.r_b() + ((context.r_a() - context.r_c()) >> 1),
        7 => (context.r_a() + context.r_b()) / 2,
        8 => context.r_ix(),
        _ => 0
    }
}

fn get_image_data_without_stuffed_zero_bytes(encoded_image: &Vec<u8>, mut read_index: usize) -> (Vec<u8>, usize) {
    // See JPG document 10918-1 P33 B.1.1.5 Note 2
    let mut image_data: Vec<u8> = Vec::with_capacity(encoded_image.len());
    let mut write_index: usize = 0;

    loop {
        if encoded_image[read_index] < 0xFF {
            // if the current element is less then 0xFF the proceide as usual
            image_data.push(encoded_image[read_index]);
            read_index += 1;
            write_index += 1;
        } else if encoded_image[read_index + 1] == 0 {
            // given that the current element is 0xFF
            // if the next element is zero then
            // this element should be read and the next is a "zero byte"
            // which was added to avoid confusion with markers and should be discarded
            image_data.push(encoded_image[read_index]);
            read_index += 2;
            write_index += 1;
        } else {
            // Hit the end of the section
            break;
        }
    }

    let mut bits: Vec<u8> = Vec::with_capacity(write_index * 8);

    for i in 0..write_index {
        bits.push((image_data[i] >> 7) & 1);
        bits.push((image_data[i] >> 6) & 1);
        bits.push((image_data[i] >> 5) & 1);
        bits.push((image_data[i] >> 4) & 1);
        bits.push((image_data[i] >> 3) & 1);
        bits.push((image_data[i] >> 2) & 1);
        bits.push((image_data[i] >> 1) & 1);
        bits.push((image_data[i] >> 0) & 1);
    }

    (bits, read_index)
}

fn get_huffmaned_value(ssss_table: &SSSSTable, image_bits: &Vec<u8>, mut bit_read_index: usize) -> (i32, usize) {
    let mut ssss: u8 = 0xFF;
    let mut guess: u32 = 1;

    for _ in 0..ssss_table.min_code_length - 1 {
        guess = (guess << 1) | (image_bits[bit_read_index] as u32);
        bit_read_index += 1;
    }

    for _ in 0..ssss_table.max_code_length {
        guess = (guess << 1) | (image_bits[bit_read_index] as u32);
        bit_read_index += 1;
        if ssss_table.table.contains_key(&guess) {
            ssss = ssss_table.table[&guess];
            break;
        }
    }

    match ssss {
        0xFF => {
            // if no code is matched return a zero, this was said to be the safest somewhere
            // TODO: should if break or be error resistant? also goes for down below
            panic!("No matching Huffman code was found for a lossless tile jpeg.")
            // warnings.warn('A Huffman coding error was found in a lossless jpeg in a dng; it may'
            //               + ' have been resolved, there may be corrupted data')
        },
        16 => (32768, bit_read_index),
        _ => {
            let mut pixel_diff: u16 = 0;
            if ssss > 0 {
                let first_bit = image_bits[bit_read_index];
                // step thru the ssss number of bits to get the coded number
                for _ in 0..ssss {
                    pixel_diff = (pixel_diff << 1) | (image_bits[bit_read_index] as u16);
                    bit_read_index += 1;
                }
                // if the first read bit is 0 the number is negative and has to be calculated
                if first_bit == 0 {
                    (-1 * ((1 << ssss) - (pixel_diff + 1)) as i32, bit_read_index)
                    // (-(1 << ssss) + pixel_diff + 1, bit_read_index)
                } else {
                    (pixel_diff as i32, bit_read_index)
                }
            } else {
                (0, bit_read_index)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use std::env;
    use std::fs::File;
    use std::io::Read;
    use std::path::Path;
    use test::Bencher;

    use super::*;

    fn get_file_as_byte_vec(path: &Path) -> Vec<u8> {
        
        let mut file = File::open(path).expect("The test file wasn't where it was expected.");
        let mut encoded_image = Vec::from([]);
        file.read_to_end(&mut encoded_image);

        encoded_image
    }

    #[test]
    fn get_huffmaned_value_0_bits() {
        let ssss_table = SSSSTable {
            t_c: 0,
            t_h: 0,
            table: HashMap::from([(4, 0), (30, 4), (6, 2), (126, 6), (254, 7), (510, 8), (14, 3), (5, 1), (62, 5)]),
            min_code_length: 2,
            max_code_length: 8,
        };
        let image_bits: Vec<u8> = Vec::from([1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ]);
        
        let (pixel_diff, bit_read_index) = get_huffmaned_value(&ssss_table, &image_bits, 1);
        assert_eq!(pixel_diff, 0);
        assert_eq!(bit_read_index, 3);
    }

    #[test]
    fn get_huffmaned_value_1_bit() {
        let ssss_table = SSSSTable {
            t_c: 0,
            t_h: 0,
            table: HashMap::from([(4, 0), (30, 4), (6, 2), (126, 6), (254, 7), (510, 8), (14, 3), (5, 1), (62, 5)]),
            min_code_length: 2,
            max_code_length: 8,
        };
        let image_bits: Vec<u8> = Vec::from([1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ]);
        
        let (pixel_diff, bit_read_index) = get_huffmaned_value(&ssss_table, &image_bits, 1);
        assert_eq!(pixel_diff, 1);
        assert_eq!(bit_read_index, 4);
    }

    #[test]
    fn get_huffmaned_value_1_bit_neg() {
        let ssss_table = SSSSTable {
            t_c: 0,
            t_h: 0,
            table: HashMap::from([(4, 0), (30, 4), (6, 2), (126, 6), (254, 7), (510, 8), (14, 3), (5, 1), (62, 5)]),
            min_code_length: 2,
            max_code_length: 8,
        };
        let image_bits: Vec<u8> = Vec::from([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ]);
        
        let (pixel_diff, bit_read_index) = get_huffmaned_value(&ssss_table, &image_bits, 1);
        assert_eq!(pixel_diff, -1);
        assert_eq!(bit_read_index, 4);
    }

    #[test]
    fn get_huffmaned_value_2_bits() {
        let ssss_table = SSSSTable {
            t_c: 0,
            t_h: 0,
            table: HashMap::from([(4, 0), (30, 4), (6, 2), (126, 6), (254, 7), (510, 8), (14, 3), (5, 1), (62, 5)]),
            min_code_length: 2,
            max_code_length: 8,
        };
        let image_bits: Vec<u8> = Vec::from([1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ]);
        
        let (pixel_diff, bit_read_index) = get_huffmaned_value(&ssss_table, &image_bits, 1);
        assert_eq!(pixel_diff, 3);
        assert_eq!(bit_read_index, 5);
    }

    #[test]
    fn get_huffmaned_value_2_bits_neg() {
        let ssss_table = SSSSTable {
            t_c: 0,
            t_h: 0,
            table: HashMap::from([(4, 0), (30, 4), (6, 2), (126, 6), (254, 7), (510, 8), (14, 3), (5, 1), (62, 5)]),
            min_code_length: 2,
            max_code_length: 8,
        };
        let image_bits: Vec<u8> = Vec::from([1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ]);
        
        let (pixel_diff, bit_read_index) = get_huffmaned_value(&ssss_table, &image_bits, 1);
        assert_eq!(pixel_diff, -3);
        assert_eq!(bit_read_index, 5);
    }

    #[test]
    fn get_huffmaned_value_16_bits() {
        let ssss_table = SSSSTable {
            t_c: 0,
            t_h: 0,
            table: HashMap::from([(4, 0), (30, 4), (6, 16), (126, 6), (254, 7), (510, 8), (14, 3), (5, 1), (62, 5)]),
            min_code_length: 2,
            max_code_length: 16,
        };
        let image_bits: Vec<u8> = Vec::from([1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ]);
        
        let (pixel_diff, bit_read_index) = get_huffmaned_value(&ssss_table, &image_bits, 1);
        assert_eq!(pixel_diff, 32768);
        assert_eq!(bit_read_index, 3);
    }

    #[test]
    #[should_panic(expected = "No matching Huffman code was found for a lossless tile jpeg.")]
    fn get_huffmaned_value_panic() {
        let ssss_table = SSSSTable {
            t_c: 0,
            t_h: 0,
            table: HashMap::from([(4, 0), (30, 4), (6, 16), (126, 6), (254, 7), (510, 8), (14, 3), (5, 1), (62, 5)]),
            min_code_length: 2,
            max_code_length: 8,
        };
        let image_bits: Vec<u8> = Vec::from([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1]);
        
        let (_pixel_diff, _bit_read_index) = get_huffmaned_value(&ssss_table, &image_bits, 1);
    }

    // fn get_ContextContext<'a>() -> ContextContext + 'a {
    //     let img = Vec::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 26, 27, 28, 29, 210, 211, 212]);
    //     let components: usize = 2;
    //     let width: usize = 4;
    //     let image_index: usize = 11;
    //     let Ah: u8 = 2;
    //     let P: u8 = 8;

    //     ContextContext {
    //         component: image_index % components,
    //         x: (image_index / components) % width,
    //         y: (image_index / components) / width,
    //         width: &width,
    //         point_tranform: &Ah, 
    //         P: &P,
    //         img: &img,
    //     }
    // }

    #[test]
    fn get_image_data_without_stuffed_zero_bytes_good() {
        let encoded_image: Vec<u8> = Vec::from([0x00, 0xFE, 0x00, 0xFF, 0x00, 0x05, 0xFF, 0xDA]);
        let expected_bits: Vec<u8> = Vec::from([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1]);
        let read_index: usize = 1;

        let (actual_bits, read_index) = get_image_data_without_stuffed_zero_bytes(&encoded_image, read_index);

        assert_eq!(actual_bits, expected_bits);
        assert_eq!(read_index, 6);
    }

    #[test]
    fn contextcontext_a_good() {
        // let context = get_ContextContext();
        let img = Vec::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 26, 27, 28, 29, 210, 211, 212]);
        // 1, 2, 3, 4, 5, 6, 7, 8, 
        // 9, 10, 11, 12, 21, 22, 23, 24, 25, 26, 27, 28, 29, 210, 211, 212
        let components: usize = 2;
        let width: usize = 4;
        let image_index: usize = 10;
        let a_h: u8 = 2;
        let p_: u8 = 8;

        let context = ContextContext {
            component: &(image_index % components),
            x_position: (image_index / components) % width,
            y_position: (image_index / components) / width,
            width: &width,
            numb_of_components: &components,
            point_tranform: &a_h, 
            p_: &p_,
            img: &img,
        };

        let a = context.r_a();

        assert_eq!(a, 9);
    }

    #[test]
    fn contextcontext_b_good() {
        // let context = get_ContextContext();
        let img = Vec::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 26, 27, 28, 29, 210, 211, 212]);
        let components: usize = 2;
        let width: usize = 4;
        let image_index: usize = 10;
        let a_h: u8 = 2;
        let p_: u8 = 8;

        let context = ContextContext {
            component: &(image_index % components),
            x_position: (image_index / components) % width,
            y_position: (image_index / components) / width,
            width: &width,
            numb_of_components: &components,
            point_tranform: &a_h, 
            p_: &p_,
            img: &img,
        };

        let b = context.r_b();

        assert_eq!(b, 3);
    }

    #[test]
    fn contextcontext_c_good() {
        // let context = get_ContextContext();
        let img = Vec::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 26, 27, 28, 29, 210, 211, 212]);
        // 1, 2, 3, 4, 5, 6, 7, 8, 
        // 9, 10, 11, 12, 21, 22, 23, 24, 
        // 25, 26, 27, 28, 29, 210, 211, 212
        let components: usize = 2;
        let width: usize = 4;
        let image_index: usize = 10;
        let a_h: u8 = 2;
        let p_: u8 = 8;

        let context = ContextContext {
            component: &(image_index % components),
            x_position: (image_index / components) % width,
            y_position: (image_index / components) / width,
            width: &width,
            numb_of_components: &components,
            point_tranform: &a_h, 
            p_: &p_,
            img: &img,
        };

        let c = context.r_c();

        assert_eq!(c, 1);
    }

    #[test]
    fn contextcontext_ix_good() {
        // let context = get_ContextContext();
        let img = Vec::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 26, 27, 28, 29, 210, 211, 212]);
        // 1, 2, 3, 4, 5, 6, 7, 8, 
        // 9, 10, 11, 12, 21, 22, 23, 24, 
        // 25, 26, 27, 28, 29, 210, 211, 212
        let components: usize = 2;
        let width: usize = 4;
        let image_index: usize = 10;
        let a_h: u8 = 2;
        let p_: u8 = 8;

        let context = ContextContext {
            component: &(image_index % components),
            x_position: (image_index / components) % width,
            y_position: (image_index / components) / width,
            width: &width,
            numb_of_components: &components,
            point_tranform: &a_h, 
            p_: &p_,
            img: &img,
        };

        let ix = context.r_ix();

        assert_eq!(ix, 1 << (p_ - a_h - 1));
    }

    #[test]
    fn parse_scan_header_good() {
        let mut path = env::current_dir().unwrap();
        path.push("tests");
        path.push("common");
        path.push("F-18.ljpg");
        let path = path.as_path();
        let encoded_image = get_file_as_byte_vec(path);

        let (scan_header, read_index) = parse_scan_header(&encoded_image, 0x70);

        assert_eq!(scan_header.head_params.len(), 3);
        assert_eq!(scan_header.head_params.get(&0).unwrap().t_d, 0);
        assert_eq!(scan_header.head_params.get(&0).unwrap().t_a, 0);
        assert_eq!(scan_header.head_params.get(&1).unwrap().t_d, 1);
        assert_eq!(scan_header.head_params.get(&1).unwrap().t_a, 0);
        assert_eq!(scan_header.head_params.get(&2).unwrap().t_d, 2);
        assert_eq!(scan_header.head_params.get(&2).unwrap().t_a, 0);
        assert_eq!(scan_header.s_s, 0x05);
        assert_eq!(scan_header.s_e, 0x00);
        assert_eq!(scan_header.a_h, 0x00);
        assert_eq!(scan_header.a_l, 0x00);
        assert_eq!(read_index, 0x7E);
    }

    #[test]
    fn parse_huffman_info_good() {
        let mut path = env::current_dir().unwrap();
        path.push("tests");
        path.push("common");
        path.push("F-18.ljpg");
        let path = path.as_path();
        let encoded_image = get_file_as_byte_vec(path);

        let read_index = 0x15;
        let (t_c, t_h, code_lengths, read_index) = parse_huffman_info(&encoded_image, read_index);

        let expected = [[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                       [0, 1, 2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                       [3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                       [4, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                       [5, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                       [6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                       [7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                       [8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                       [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                       [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                       [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                       [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                       [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                       [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                       [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                       [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]];

        assert_eq!(code_lengths, expected);
        assert_eq!(t_c, 0);
        assert_eq!(t_h, 0);
        assert_eq!(read_index, 0x33)
    }

    #[test]
    fn make_ssss_tables_good() {
        let code_lengths = [[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [0, 1, 2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [4, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [5, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]];

        let expected = HashMap::from([(4, 0), (30, 4), (6, 2), (126, 6), (254, 7), (510, 8), (14, 3), (5, 1), (62, 5)]);

        let (tables, min_code_length, max_code_length) = make_ssss_table(code_lengths);

        assert_eq!(tables, expected);   
        assert_eq!(min_code_length, 2);
        assert_eq!(max_code_length, 8);
    }

    #[bench]
    fn make_ssss_tables_bench(b: &mut Bencher) {
        let code_lengths = [[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [0, 1, 2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [4, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [5, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
                               [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]];

        b.iter(|| make_ssss_table(code_lengths));            
    }

    #[test]
    fn parse_frame_header_good() {
        let mut path = env::current_dir().unwrap();
        path.push("tests");
        path.push("common");
        path.push("F-18.ljpg");
        let path = path.as_path();
        let encoded_image = get_file_as_byte_vec(path);

        let (frame_header, read_index) = parse_frame_header(&encoded_image, 2);

        assert_eq!(frame_header.p_, 0x08);
        assert_eq!(frame_header.y_, 0x00F0);
        assert_eq!(frame_header.x_, 0x0140);
        assert_eq!(frame_header.components.len(), 3);
        assert_eq!(frame_header.components.get(&0).unwrap().h_, 1);
        assert_eq!(frame_header.components.get(&0).unwrap().v_, 1);
        assert_eq!(frame_header.components.get(&0).unwrap().t_q, 0);
        assert_eq!(frame_header.components.get(&1).unwrap().h_, 1);
        assert_eq!(frame_header.components.get(&1).unwrap().v_, 1);
        assert_eq!(frame_header.components.get(&1).unwrap().t_q, 0);
        assert_eq!(frame_header.components.get(&2).unwrap().h_, 1);
        assert_eq!(frame_header.components.get(&2).unwrap().v_, 1);
        assert_eq!(frame_header.components.get(&2).unwrap().t_q, 0);
        assert_eq!(read_index, 21);
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
    fn bytes_to_int_two_good() {
        let buffer = Vec::from([0xC2, 0x1D, 0x8D, 0xE4, 0x0C, 0x9A]);
        let expected = 7565_u16;
        assert_eq!(bytes_to_int_two(&buffer[1..3]), expected);
    }

    #[test]
    fn bytes_to_int_two_fail() {
        let buffer = Vec::from([0xC2, 0x1D, 0x8D, 0xE4, 0x0C, 0x9A]);
        let expected = 7564_u16;
        assert_ne!(bytes_to_int_two(&buffer[1..3]), expected);
    }
}



// let mut all_code_lengths: Vec<[[u8; 16]; 16]> = Vec::new();
// all_code_lengths.push([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [0, 1, 2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [4, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [5, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]]);
// all_code_lengths.push([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [0, 1, 2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [4, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [5, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [9, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]]);
// all_code_lengths.push([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [0, 1, 2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [4, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [5, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]]);


// let mut expected: Vec<[[u8; 16]; 16]> = Vec::new();
// expected.push([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [0, 1, 2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [4, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [5, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]]);
// expected.push([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [0, 1, 2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [4, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [5, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [9, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]]);
// expected.push([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [0, 1, 2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [4, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [5, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
//                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]]);
