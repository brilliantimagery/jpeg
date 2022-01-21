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

struct Context {
    a: u32,
    b: u32,
    c: u32,
    ix: u32,
}

struct ContextContext<'a> {
    component: usize,
    x: usize,
    y: usize,
    width: &'a usize,
    point_tranform: &'a u8, 
    P: &'a u8, 
    img: &'a Vec<u32>,
}

impl ContextContext<'_> {
    fn a (&self) -> u32 {
        self.img[(self.component * self.x * self.y) + (self.y * self.width) + (self.x-1)]
    }
    fn b (&self) -> u32 {
        self.img[(self.component * self.x * self.y) + ((self.y - 1) * self.width) + (self.x)]
    }
    fn c (&self) -> u32 {
        self.img[(self.component * self.x * self.y) + ((self.y - 1) * self.width) + (self.x-1)]
    }
    fn ix (&self) -> u32 {
        1 << (self.P - self.point_tranform - 1)
    }
}

struct Predictor<T> 
where
    T: Fn(u32) -> u32,
{
    middle: T,

}

struct Component {
    // C: u8,
    H: u8,
    V: u8,
    Tq: u8,
}

struct FrameHeader {
    P: u8,
    Y: u16,
    X: u16,
    components: HashMap<u8, Component>,
}

struct ScanHeader {
    // head_params: Vec<HeaderParameter>,
    head_params: HashMap<u8, HeaderParameter>,
    Ss: u8,
    Se: u8,
    Ah: u8,
    Al: u8,
}

// impl ScanHeader {
//     fn new()
// }

struct HeaderParameter {
    // Cs: u8,
    Td: u8,
    Ta: u8,
}

struct SSSSTable {
    Tc: u8,
    Th: u8,
    table: HashMap<u32, u8>,
}

pub fn decode(encoded_image: Vec<u8>) {
    is_jpeg(&encoded_image);
    let mut read_index: usize = 2;
    let mut marker: u16;
    
    let mut frame_header: FrameHeader = FrameHeader {P:0, X:0, Y:0, components:HashMap::new()};

    let mut ssss_tables: Vec<SSSSTable> = Vec::with_capacity(2);
    while read_index < encoded_image.len() {
        marker = bytes_to_int_two(&encoded_image[read_index..read_index+2]);
        match marker {
            SOF3 => {
                let header_info = parse_frame_header(&encoded_image, read_index); 

                frame_header = header_info.0;
                read_index = header_info.1;
            },
            DHT => {
                let huffman_info = get_huffman_info(&encoded_image, read_index);
                ssss_tables.push(huffman_info.0);
                read_index = huffman_info.1;
            },
            SOS => {
                let (scan_header, read_index) = parse_scan_header(&encoded_image, read_index);
                decode_image(&encoded_image, frame_header, scan_header, read_index);
            },
            x if x > MIN_MARKER => panic!("Not implimented marker!"),
            _ => { read_index += 1; },
        }
    }
}

fn get_huffman_info(encoded_image: &Vec<u8>, read_index: usize) -> (SSSSTable, usize) {
    let (Tc, Th, code_lengths, read_index) = parse_huffman_info(&encoded_image, read_index);

    let table = make_ssss_table(code_lengths);

    (SSSSTable {
        Tc,
        Th,
        table
    }, read_index)
}

fn parse_huffman_info(encoded_image: &Vec<u8>, mut read_index: usize) -> (u8, u8, [[u8; 16]; 16], usize) {
    read_index += 2;
    let Lh = bytes_to_int_two(&encoded_image[read_index..=read_index + 1]);
    read_index += 2;
    let Tc = encoded_image[read_index] >> 4;
    let Th = encoded_image[read_index] | 0xF;
    read_index += 1;
    let mut vij_index = read_index + 16;

    let mut code_lengths = [[0xFF_u8; 16]; 16];
    let mut Li: u8;
    let mut n_codes: usize;
    for code_length_index in 0..16 {
        Li = encoded_image[read_index + code_length_index];
        if Li > 0 {
            n_codes = 0;
            for _ in 0..Li {
                println!("{:?}, {:?}, {:?}, {:?}, {:?}, {:?}", Lh, Tc, Th, code_length_index, n_codes, Li);
                code_lengths[code_length_index][0];
                code_lengths[0][n_codes];
                encoded_image[vij_index];
                code_lengths[code_length_index][n_codes] = encoded_image[vij_index];
                n_codes += 1;
                vij_index += 1;
            }
        }
    }

    (Tc, Th, code_lengths, vij_index)
}

fn make_ssss_table(code_lengths: [[u8; 16]; 16]) -> HashMap<u32, u8> {
    let mut values_w_n_bits: usize;    
    let mut removed: u32;

    // store the huffman code in the bits of a u32
    let mut code = 1_u32;  // start with a 1 so there can be leading zeros
    let mut table: HashMap<u32, u8> = HashMap::new();
    for index in 0..16 {
        if code_lengths[index][0] < 0xFF_u8 {
            let values: Vec<u8> = code_lengths[index].into_iter().filter(|x| x < &0xFF_u8).collect::<Vec<u8>>();
            values_w_n_bits = 0;
            while values_w_n_bits <= values.len() {
                code = code << (index + 1 - (number_of_used_bits(&code) - 1));
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
                values_w_n_bits += 1;
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

fn parse_scan_header(encoded_image: &Vec<u8>, mut read_index: usize) -> (ScanHeader, usize) {
    read_index += 2;
    let Ls = bytes_to_int_two(&encoded_image[read_index..read_index + 2]);
    read_index += 2;
    let Ns = encoded_image[read_index] as usize;
    read_index += 1;
    // let mut head_params: Vec<HeaderParameter> = Vec::with_capacity(Ns);
    let mut head_params: HashMap<u8, HeaderParameter> = HashMap::new();
    for param in 0..Ns {
        head_params.insert(
            encoded_image[read_index],
            HeaderParameter {
                Td: encoded_image[read_index + 1] >> 4,
                Ta: encoded_image[read_index + 1] & 0xF,
            });
        // head_params.push(HeaderParameter{
        //     Cs: encoded_image[read_index],
        //     Td: encoded_image[read_index + 1] >> 4,
        //     Ta: encoded_image[read_index + 1] & 0xF,
        // });
        read_index += 2;
    }
    let Ss = encoded_image[read_index];
    read_index += 1;
    let Se = encoded_image[read_index];
    read_index += 1;
    let Ah = encoded_image[read_index] >> 4;
    let Al = encoded_image[read_index] & 0xF;
    read_index += 1;

    (ScanHeader {
        head_params,
        Ss,
        Se,
        Ah,
        Al,
    },
    read_index)
}

fn parse_frame_header(encoded_image: &Vec<u8>, mut read_index: usize) -> (FrameHeader, usize) {
    read_index += 2;
    let Lf: u16 = bytes_to_int_two(&encoded_image[read_index..read_index + 2]);
    read_index += 2;
    let P: u8 = encoded_image[read_index];
    read_index += 1;
    let Y: u16 = bytes_to_int_two(&encoded_image[read_index..read_index + 2]);
    read_index += 2;
    let X: u16 = bytes_to_int_two(&encoded_image[read_index..read_index + 2]);
    read_index += 2;
    let Nf: usize = encoded_image[read_index] as usize;
    read_index += 1;
    let mut components: HashMap<u8, Component> = HashMap::new();
    for comp in 0..Nf as usize {
        components.insert(
            encoded_image[read_index],
            Component {
                H: encoded_image[read_index + 1] >> 4,
                V: encoded_image[read_index + 1] & 0xF,
                Tq: encoded_image[read_index + 2],
            });
        // components.push(Component {
        //     C: encoded_image[read_index],
        //     H: encoded_image[read_index + 1] >> 4,
        //     V: encoded_image[read_index + 1] & 0xF,
        //     Tq: encoded_image[read_index + 2],
        // });
        read_index += 3
    }

    (FrameHeader {
        P,
        Y,
        X,
        components,
    },
    read_index)
    // (P, Y, X, components, read_index)
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

fn decode_image(encoded_image: &Vec<u8>, frame_header: FrameHeader, scan_header: ScanHeader, read_index: usize) {
    let width = frame_header.X as usize;
    let height = frame_header.Y as usize;

    let number_of_pixels = (width * height) as usize;
    let write_index = 0_usize;

    panic!("placeholder mumbo jumbo");
    let components = frame_header.components.len();
    let decoded_image: Vec<u32> = Vec::with_capacity(width * height * components);
    let image_start_index = read_index;
    while read_index < encoded_image.len() {
        let image_index = read_index - image_start_index;
        let context = ContextContext {
            component: image_index % components,
            x: (image_index / components) % width,
            y: (image_index / components) / width,
            width: &width,
            point_tranform: &scan_header.Ah, 
            P: &frame_header.P, 
            img: &decoded_image,
        };
        let px = get_prediction(context, &scan_header.Ss);
    }
}

fn get_context(component: &usize, x: &usize, y: &usize, width: &usize, point_tranform: &u8, P: &u8, img: &Vec<u32>) -> Context {
    let mut a: u32 = 0;
    let mut b: u32 = 0;
    let mut c: u32 = 0;
    let mut ix: u32 = 0;

    if *y == 0 {
        if *x == 0 {
            ix = 1 << (P - point_tranform - 1);
        } else {
            a = img[(component * x * y) + (y * width) + (x-1)];
        }
    } else {
        if *x != 0 {
            a = img[(component * x * y) + (y * width) + (x-1)];
            c = img[(component * x * y) + ((y - 1) * width) + (x - 1)];
        }
        b = img[(component * x * y) + ((y - 1) * width) + x];
    }
    Context {a, b, c, ix}
}

fn  get_prediction(context: ContextContext, predictor: &u8) -> u32 {
    let mut used_predictor: u8 = *predictor;

    if context.x == 0 {
        if context.y == 0 {
            used_predictor = 8;
        } else {
            used_predictor = 2;
        }
    } else if context.y == 0 {
        used_predictor = 1;        
    }

    if used_predictor == 8_u8 {
        let a = 0;
    }

    match used_predictor {
        0 => 0,
        1 => context.a(),
        2 => context.b(),
        3 => context.c(),
        4 => { context.a() + context.b() - context.c() },
        5 => context.a() + ((context.b() - context.c()) >> 1),
        6 => context.b() + ((context.a() - context.c()) >> 1),
        7 => (context.a() + context.b()) / 2,
        8 => context.ix(),
        _ => 0
    }
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::fs::File;
    use std::io::Read;
    use std::path::{Path, PathBuf};

    use super::*;

    fn get_file_as_byte_vec(path: &Path) -> Vec<u8> {
        
        let mut file = File::open(path).expect("The test file wasn't where it was expected.");
        let mut encoded_image = Vec::from([]);
        file.read_to_end(&mut encoded_image);

        encoded_image
    }

    #[test]
    fn get_context_top_left() {
        let mut img = vec!(vec!(vec!(0_u32; 3); 4); 2);
        img[0][0] = Vec::from([1, 2, 3]);
        img[0][1] = Vec::from([4, 5, 6]);
        img[0][2] = Vec::from([7, 8, 9]);
        img[0][3] = Vec::from([10, 11, 12]);

        let component: usize = 0;
        let x: usize = 0;
        let y: usize = 0;
        let point_transform: u8 = 2;
        let P: u8 = 8;

        let context = get_context(component, &x, &y, &point_transform, &P, &img);

        assert_eq!(context.a, 1 << (P - point_transform - 1));
        assert_eq!(context.b, 0);
        assert_eq!(context.c, 0);
        assert_eq!(context.ix, img[component][x][y]);
    }

    #[test]
    fn get_context_top_middle() {
        let mut img = vec!(vec!(vec!(0_u32; 3); 4); 2);
        img[0][0] = Vec::from([1, 2, 3]);
        img[0][1] = Vec::from([4, 5, 6]);
        img[0][2] = Vec::from([7, 8, 9]);
        img[0][3] = Vec::from([10, 11, 12]);

        let component: usize = 0;
        let x: usize = 1;
        let y: usize = 0;
        let point_transform: u8 = 2;
        let P: u8 = 8;

        let context = get_context(component, &x, &y, &point_transform, &P, &img);

        assert_eq!(context.a, img[component][x - 1][y]);
        assert_eq!(context.b, 0);
        assert_eq!(context.c, 0);
        assert_eq!(context.ix, img[component][x][y]);
    }

    #[test]
    fn get_context_middle_left() {
        let mut img = vec!(vec!(vec!(0_u32; 3); 4); 2);
        img[0][0] = Vec::from([1, 2, 3]);
        img[0][1] = Vec::from([4, 5, 6]);
        img[0][2] = Vec::from([7, 8, 9]);
        img[0][3] = Vec::from([10, 11, 12]);

        let component: usize = 0;
        let x: usize = 0;
        let y: usize = 1;
        let point_transform: u8 = 2;
        let P: u8 = 8;

        let context = get_context(component, &x, &y, &point_transform, &P, &img);

        assert_eq!(context.a, 0);
        assert_eq!(context.b, img[component][x][y - 1]);
        assert_eq!(context.c, 0);
        assert_eq!(context.ix, img[component][x][y]);
    }

    #[test]
    fn get_context_middle_middle() {
        let mut img = vec!(vec!(vec!(0_u32; 3); 4); 2);
        img[0][0] = Vec::from([1, 2, 3]);
        img[0][1] = Vec::from([4, 5, 6]);
        img[0][2] = Vec::from([7, 8, 9]);
        img[0][3] = Vec::from([10, 11, 12]);

        let component: usize = 0;
        let x: usize = 1;
        let y: usize = 1;
        let point_transform: u8 = 2;
        let P: u8 = 8;

        let context = get_context(component, &x, &y, &point_transform, &P, &img);

        assert_eq!(context.a, img[component][x - 1][y]);
        assert_eq!(context.b, img[component][x][y - 1]);
        assert_eq!(context.c, img[component][x - 1][y - 1]);
        assert_eq!(context.ix, img[component][x][y]);
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
        assert_eq!(scan_header.head_params[0].Cs, 0);
        assert_eq!(scan_header.head_params[0].Td, 0);
        assert_eq!(scan_header.head_params[0].Ta, 0);
        assert_eq!(scan_header.head_params[1].Cs, 1);
        assert_eq!(scan_header.head_params[1].Td, 1);
        assert_eq!(scan_header.head_params[1].Ta, 0);
        assert_eq!(scan_header.head_params[2].Cs, 2);
        assert_eq!(scan_header.head_params[2].Td, 2);
        assert_eq!(scan_header.head_params[2].Ta, 0);
        assert_eq!(scan_header.Ss, 0x05);
        assert_eq!(scan_header.Se, 0x00);
        assert_eq!(scan_header.Ah, 0x00);
        assert_eq!(scan_header.Al, 0x00);
    }

    #[test]
    fn parse_huffman_info_good() {
        let mut path = env::current_dir().unwrap();
        path.push("tests");
        path.push("common");
        path.push("F-18.ljpg");
        let path = path.as_path();
        let encoded_image = get_file_as_byte_vec(path);

        let mut read_index = 0x15;
        let (Tc, Th, code_lengths, read_index) = parse_huffman_info(&encoded_image, read_index);

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

        let tables = make_ssss_table(code_lengths);

        assert_eq!(tables, expected);
                       
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

        assert_eq!(frame_header.P, 0x08);
        assert_eq!(frame_header.Y, 0x00F0);
        assert_eq!(frame_header.X, 0x0140);
        assert_eq!(frame_header.components.len(), 3);
        assert_eq!(frame_header.components[0].C, 0);
        assert_eq!(frame_header.components[0].H, 1);
        assert_eq!(frame_header.components[0].V, 1);
        assert_eq!(frame_header.components[0].Tq, 0);
        assert_eq!(frame_header.components[1].C, 1);
        assert_eq!(frame_header.components[1].H, 1);
        assert_eq!(frame_header.components[1].V, 1);
        assert_eq!(frame_header.components[1].Tq, 0);
        assert_eq!(frame_header.components[2].C, 2);
        assert_eq!(frame_header.components[2].H, 1);
        assert_eq!(frame_header.components[2].V, 1);
        assert_eq!(frame_header.components[2].Tq, 0);
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
