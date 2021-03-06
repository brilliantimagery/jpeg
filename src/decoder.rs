#![allow(dead_code)]
use std::collections::{BTreeMap, HashMap};
use std::slice::Iter;

use crate::jpeg_errors::*;
use crate::jpeg_utils;
use crate::jpeg_utils::{ContextContext, Marker};

/// Quantization Table, 10918-1, B.2.4.1, P. 39
struct QuantiziationTable {
    p_q: u8,        // Element precision,
    t_q: u8,        // Destinaiton identifier
    q_k: [u16; 64], // Table element
}

struct Component {
    c_: u8,  // Component identifier, 10918-1 P. 36
    h_: u8,  // Horizontal sampling factor
    v_: u8,  // Vertical sampling factor
    t_q: u8, // Quantiziation table destination selector; Not used (0), for lossless
}

struct FrameHeader {
    // Frame Header, 10918-1, B.2.2, P. 35
    marker: u16,
    p_: u8,  // Sample precision
    y_: u16, // Number of lines
    x_: u16, // Number of samples per line
    components: HashMap<u8, Component>,
}

struct ScanHeader {
    // Scan Header, 10918-1, B.2.3, P. 35
    head_params: HashMap<u8, HeaderParameter>,
    s_s: u8, // Start of Spectral selection; predictor selector in lossless
    s_e: u8, // End of Spectral or prediction selection; 0, not used, in lossless
    a_h: u8, // Successive aproximamtion bit position high, 0, not used, in lossless
    a_l_p_t: u8, // Successive approximation bit position low; point transform, Pt, for lossless mode
}

struct HeaderParameter {
    c_s: u8, // Scan component selector
    t_d: u8, // DC entropy coding table destination selector
    t_a: u8, // AC entropy coding table destination selector
}

struct SSSSTable {
    t_c: u8, // Table class – 0 = DC table or lossless table, 1 = AC table
    t_h: u8, // Huffman table destination identifier
    table: HashMap<u32, u8>,
    min_code_length: usize, // number of bits of shorted Huffman code
    max_code_length: usize, // number of bits of longest Huffman code
}

pub(crate) fn decode(encoded_image: Vec<u8>) -> Result<Vec<u32>, JpegDecoderError> {
    let mut encoded_image = encoded_image.iter();
    is_jpeg(&mut encoded_image)?;

    let mut frame_header = FrameHeader {
        marker: 0,
        p_: 0,
        x_: 0,
        y_: 0,
        components: HashMap::new(),
    };
    let mut ssss_tables: HashMap<usize, SSSSTable> = HashMap::new();
    let mut raw_image: Vec<u32> = Vec::new();
    let mut quantization_tables: HashMap<u8, QuantiziationTable> = HashMap::new();

    use crate::jpeg_utils::Marker::*;
    while encoded_image.len() > 0 {
        match bytes_to_int_two_peeked(&mut encoded_image)? {
            marker if marker == SOF0 as u16 || marker == SOF3 as u16 => {
                encoded_image.next();
                frame_header = parse_frame_header(&mut encoded_image, marker)?;
            }
            marker if marker == DQT as u16 => {
                encoded_image.next();
                let table = parse_quantizaiton_table(&mut encoded_image)?;
                quantization_tables.insert(table.t_q, table);
            }
            marker if marker == DHT as u16 => {
                encoded_image.next();
                let huffman_info = get_huffman_info(&mut encoded_image)?;
                ssss_tables.insert(huffman_info.t_h as usize, huffman_info);
            }
            marker if marker == SOS as u16 => {
                encoded_image.next();
                let scan_header = parse_scan_header(&mut encoded_image)?;
                raw_image = decode_image(
                    &mut encoded_image,
                    &frame_header,
                    &scan_header,
                    &ssss_tables,
                )?;
            }
            marker if APP as u16 <= marker && marker <= APPn as u16 => {
                encoded_image.next();
                skip_app_marker(&mut encoded_image)?;
            }
            marker if marker == EOI as u16 => {
                break;
            }
            marker if marker > 0xFF00 && marker < 0xFFFF => panic!("Unimplimented marker!"),
            _ => {
                // Either ended up in random stuff or 0xFFFF which means that the first FF can be discarded (10918-1, P. 82)
            }
        }
    }

    Ok(raw_image)
}

fn parse_quantizaiton_table(
    encoded_image: &mut Iter<u8>,
) -> Result<QuantiziationTable, OutOfBoundsError> {
    let _l_q = bytes_to_int_two_consumed(encoded_image)?;
    let p_t_q = *encoded_image.next().ok_or(OutOfBoundsError)?;
    let p_q = p_t_q >> 4;
    let t_q = p_t_q & 0xF;
    let mut q_k = [0u16; 64];

    if p_q == 0 {
        for i in q_k.iter_mut() {
            *i = *encoded_image.next().ok_or(OutOfBoundsError)? as u16
        }
    } else {
        for i in q_k.iter_mut() {
            *i = bytes_to_int_two_consumed(encoded_image)?
        }
    }

    Ok(QuantiziationTable { p_q, t_q, q_k })
}

fn skip_app_marker(encoded_image: &mut Iter<u8>) -> Result<(), OutOfBoundsError> {
    (0..bytes_to_int_two_consumed(encoded_image)? as usize).for_each(|_| {
        encoded_image.next();
    });

    Ok(())
}

fn get_huffman_info(encoded_image: &mut Iter<u8>) -> Result<SSSSTable, OutOfBoundsError> {
    let (t_c, t_h, code_lengths) = parse_huffman_info(encoded_image)?;

    let (table, min_code_length, max_code_length) = make_ssss_table(code_lengths);

    Ok(SSSSTable {
        t_c,
        t_h,
        table,
        min_code_length,
        max_code_length,
    })
}

fn parse_huffman_info(
    encoded_image: &mut Iter<u8>,
) -> Result<(u8, u8, [[Option<u8>; 16]; 16]), OutOfBoundsError> {
    let _l_h = bytes_to_int_two_consumed(encoded_image)?;
    let t_c_h = encoded_image.next().ok_or(OutOfBoundsError)?;
    let t_c = t_c_h >> 4;
    let t_h = t_c_h & 0xF;
    let mut code_lengths: [[Option<u8>; 16]; 16] = [[None; 16]; 16];
    let mut lengths: BTreeMap<u8, u8> = BTreeMap::new();
    for code_length_index in 0..16 {
        let l_i = *encoded_image.next().ok_or(OutOfBoundsError)?;
        if l_i > 0 {
            lengths.insert(code_length_index, l_i);
        }
    }
    for (code_length_index, l_i) in lengths.iter() {
        for i in 0..*l_i {
            code_lengths[*code_length_index as usize][i as usize] =
                Some(*encoded_image.next().ok_or(OutOfBoundsError)?);
        }
    }

    Ok((t_c, t_h, code_lengths))
}

/// TODO: this algerythom presumably doesn't work for all possible tables
// fn make_ssss_table(code_lengths: [[u8; 16]; 16]) -> (HashMap<u32, u8>, usize, usize) {
fn make_ssss_table(code_lengths: [[Option<u8>; 16]; 16]) -> (HashMap<u32, u8>, usize, usize) {
    // https://www.youtube.com/watch?v=dM6us854Jk0

    // Codes start towards the top left of the tree
    // As you move to the right, trailing 1s are added
    // As you move down, bits are added
    // So the left most bit represents the top row

    // 0xFF, 0xFF, 0xFF, 0xFF
    // 0x0,  0x1,  0x2,  0xFF
    // 0x3,  0xFF, 0xFF, 0xFF
    // 0x4,  0xFF, 0xFF, 0xFF

    //                                     /   \
    // Index 0, Code Length 1, Codes:     0     1
    // Value:                            NA    NA
    //                                   /\    / \
    // Index 1, Code Length 2, Codes:  00 01  10 11
    // Value:                           0  1   2 NA
    //                                           / \
    // Index 2, Code Length 3, Codes:         110  111
    // Value:                                   3   NA
    //                                              / \
    // Index 3: Code Length 4, Codes:            1110  1111
    // Values:                                      4   NA
    // NOTE: padded/leading 1 not shown so all above codes would be stored with an
    // additional 1 in front of it.

    // let mut code: u32 = 1;
    // let mut table: HashMap<u32, u8> = HashMap::new();

    // // Iterate over all of the rows of the tree
    // for bits_in_code_minus_1 in 0..16 {
    //     // for each row, add another bit
    //     code = code << 1;
    //     // if there are no codes with that number of bits go to the next row
    //     if code_lengths[bits_in_code_minus_1][0].is_none() {
    //         continue;
    //     }

    //     // let mut values_w_n_bits: usize = 0;
    //     // let values: std::slice::Iter<Option<u8>> = code_lengths[bits_in_code_minus_1].iter();
    //     for (i, value) in code_lengths[bits_in_code_minus_1].iter().enumerate() {
    //         if value.is_none() {
    //             break;
    //         }
    //         if i > 0 {
    //             let mut only_removed_ones = true;
    //             // shouldn't need number_of_... it's just there to prevent errors
    //             while only_removed_ones && number_of_used_bits(&code) > 0 {
    //                 only_removed_ones = code & 1 == 1;
    //                 code = code >> 1;
    //             }
    //             code = (code << 1) + 1;

    //             while number_of_used_bits(&code) < bits_in_code_minus_1 + 2 {
    //                 code = code << 1;
    //             }
    //             // code = code << (bits_in_code_minus_1 + 1 - (number_of_used_bits(&code) - 1));
    //         }
    //         table.insert(code, value.unwrap());
    //     }
    // }

    // storing the huffman code in the bits of a u32
    // the code is preceided by a 1 so there can be leading zeros
    let mut code: u32 = 1;
    let mut table: HashMap<u32, u8> = HashMap::new();
    for (index, row) in code_lengths.iter().enumerate() {
        // the code lengths (number of bytes) are stored in a HashMap that was initized with 0xFF
        // and the codes only go up to 16,
        // so if the first cell has 0xFF then there are no codes with a length
        // equal to that row's index
        // so remove the rows that still have the initial value, 0xFF
        // since, as previously discussed, there aren't any codes of that length

        // probably slower than the following but it's cleaner so... if row[0].is_some() {
        // filter out the values that have 0xFF since those are initial values
        // and don't have a valid code length
        let values = row.iter().filter_map(|x| *x).collect::<Vec<u8>>();
        if !values.is_empty() {
            // for each code lengh start with the 0th code of that length
            let mut values_w_n_bits: usize = 0;
            // once all codes of a length have been processed,
            // move on
            while values_w_n_bits <= values.len() {
                // Shift the padded/leading 1 so that the code's the right length
                // index + 1 is the desired code length since index is base 0 so one less then the code length
                // number_of_used_bits(&code) - 1 is the present code length since the leading/padded one takes up a bit
                // the desired code length - the present code length is the amount it must grow to achieve the desired length
                code = code << (index + 1 - (jpeg_utils::number_of_used_bits(&code) - 1));
                // While the first code of a langth "automatically" works,
                // additionl codes of a length must have bits flipped
                if values_w_n_bits > 0 {
                    // Remove bits (move up the tree) until you remove a 0
                    // (so you can move to the right branch from the left)
                    // Or until you hit the top (again, so you can move to the right branch)
                    loop {
                        let removed: u32 = code & 1;
                        code >>= 1;
                        // if !(removed == 1 && number_of_used_bits(&code) > 1) {
                        if removed == 0 || jpeg_utils::number_of_used_bits(&code) <= 1 {
                            break;
                        }
                    }
                    // Move down and to the right one node along the tree
                    code = (code << 1) + 1;
                    // Extend the code until it's appropreately long
                    code = code << ((index + 1) - (jpeg_utils::number_of_used_bits(&code) - 1));
                }
                if values.len() > values_w_n_bits {
                    table.insert(code, values[values_w_n_bits]);
                }
                values_w_n_bits += 1;
            }
        }
    }

    let mut min_code_length: usize = 100;
    let mut max_code_length: usize = 0;

    for v in table.keys() {
        let length = jpeg_utils::number_of_used_bits(v) - 1;
        if length < min_code_length {
            min_code_length = length;
        }
        if length > max_code_length {
            max_code_length = length;
        }
    }

    (table, min_code_length, max_code_length)
}

fn parse_scan_header(encoded_image: &mut Iter<u8>) -> Result<ScanHeader, OutOfBoundsError> {
    let _l_s = bytes_to_int_two_consumed(encoded_image);
    let n_s = *encoded_image.next().ok_or(OutOfBoundsError)? as usize;
    let mut head_params: HashMap<u8, HeaderParameter> = HashMap::new();
    for _ in 0..n_s {
        let c_s = *encoded_image.next().ok_or(OutOfBoundsError)?;
        let t_d_a = *encoded_image.next().ok_or(OutOfBoundsError)?;
        head_params.insert(
            c_s,
            HeaderParameter {
                c_s,
                t_d: t_d_a >> 4,
                t_a: t_d_a & 0xF,
            },
        );
    }
    let s_s = *encoded_image.next().ok_or(OutOfBoundsError)?;
    let s_e = *encoded_image.next().ok_or(OutOfBoundsError)?;
    let a_h_l = *encoded_image.next().ok_or(OutOfBoundsError)?;
    let a_h = a_h_l >> 4;
    let a_l_p_t = a_h_l & 0xF;

    Ok(ScanHeader {
        head_params,
        s_s,
        s_e,
        a_h,
        a_l_p_t,
    })
}

fn parse_frame_header(
    encoded_image: &mut Iter<u8>,
    marker: u16,
) -> Result<FrameHeader, OutOfBoundsError> {
    let _l_f: u16 = bytes_to_int_two_consumed(encoded_image)?;
    let p_: u8 = *encoded_image.next().ok_or(OutOfBoundsError)?;
    let y_: u16 = bytes_to_int_two_consumed(encoded_image)?;
    let x_: u16 = bytes_to_int_two_consumed(encoded_image)?;
    let n_f = *encoded_image.next().ok_or(OutOfBoundsError)? as usize;
    let mut components: HashMap<u8, Component> = HashMap::new();
    for _ in 0..n_f as usize {
        let c_: u8 = *encoded_image.next().ok_or(OutOfBoundsError)?;
        let h_v: u8 = *encoded_image.next().ok_or(OutOfBoundsError)?;
        let t_q: u8 = *encoded_image.next().ok_or(OutOfBoundsError)?;
        components.insert(
            c_,
            Component {
                c_,
                h_: h_v >> 4,
                v_: h_v & 0xF,
                t_q,
            },
        );
    }

    Ok(FrameHeader {
        marker,
        p_,
        y_,
        x_,
        components,
    })
}

fn is_jpeg(encoded_image: &mut Iter<u8>) -> Result<bool, BadMagicNumberError> {
    if bytes_to_int_two_consumed(encoded_image)? == Marker::SOI as u16 {
        Ok(true)
    } else {
        Err(BadMagicNumberError {
            source: OutOfBoundsError,
        })
    }
}

fn bytes_to_int_two_consumed(bytes: &mut Iter<u8>) -> Result<u16, OutOfBoundsError> {
    Ok((*bytes.next().ok_or(OutOfBoundsError)? as u16) << 8
        | *bytes.next().ok_or(OutOfBoundsError)? as u16)
}

fn bytes_to_int_two_peeked(bytes: &mut Iter<u8>) -> Result<u16, OutOfBoundsError> {
    // let mut bytes = bytes.by_ref().peekable();
    // (*bytes.next().unwrap() as u16) << 8 | **bytes.peek().unwrap() as u16
    let mut bytes_clone = bytes.clone();
    bytes.next();
    Ok((*bytes_clone.next().ok_or(OutOfBoundsError)? as u16) << 8
        | *bytes_clone.next().ok_or(OutOfBoundsError)? as u16)
}

/// TODO: THIS SEEMS TO BE WEHRE I'VE LEFT OFF
/// 10918-1, H.2, P. 136 & H.1, P. 132
fn decode_image(
    encoded_image: &mut Iter<u8>,
    frame_header: &FrameHeader,
    scan_header: &ScanHeader,
    ssss_tables: &HashMap<usize, SSSSTable>,
) -> Result<Vec<u32>, DecodeError> {
    let width = frame_header.x_ as usize;
    let height = frame_header.y_ as usize;

    let component_count = frame_header.components.len();
    let mut raw_image: Vec<u32> = Vec::with_capacity(width * height * component_count);

    let image_bits = get_image_data_without_stuffed_zero_bytes(encoded_image)?;
    let mut image_bits = image_bits.iter();

    while width * height * component_count > raw_image.len() {
        let component = raw_image.len() % component_count;
        let p_x = jpeg_utils::make_prediciton(
            &raw_image,
            raw_image.len(),
            component_count,
            width,
            frame_header.p_,
            scan_header.a_h,
            scan_header.s_s,
        );
        let pixel_delta =
            get_huffmaned_value(ssss_tables.get(&component).unwrap(), &mut image_bits)?;
        raw_image.push(((p_x as i32 + pixel_delta) & ((1 << frame_header.p_) - 1)) as u32);
    }

    Ok(raw_image)
}

fn get_image_data_without_stuffed_zero_bytes(
    encoded_image: &mut Iter<u8>,
) -> Result<Vec<u8>, OutOfBoundsError> {
    // See JPG document 10918-1 P33 B.1.1.5 Note 2
    let mut image_data: Vec<u8> = Vec::with_capacity(encoded_image.len());
    let mut image_clone = encoded_image.clone();

    let mut this_byte = image_clone.next();
    let mut next_byte = image_clone.next();
    let mut i = 0;
    while this_byte.is_some() {
        let this_val = *this_byte.ok_or(OutOfBoundsError)?;
        let next_val = *next_byte.unwrap_or(&0);
        if this_val < 0xFF {
            // if the current element is less then 0xFF the proceide as usual
            image_data.push(this_val);
            this_byte = next_byte;
            next_byte = image_clone.next();
            encoded_image.next();
            i += 1;
        } else if next_val == 0 {
            // given that the current element is 0xFF
            // if the next element is zero then
            // this element should be read and the next is a "zero byte"
            // which was added to avoid confusion with markers and should be discarded
            image_data.push(this_val);
            this_byte = image_clone.next();
            next_byte = image_clone.next();
            encoded_image.next();
            encoded_image.next();
            i += 1;
        } else {
            // Hit the end of the section
            break;
        }
    }

    let mut bits: Vec<u8> = Vec::with_capacity(i * 8);

    for i in image_data.iter().take(i) {
        bits.push((i >> 7) & 1);
        bits.push((i >> 6) & 1);
        bits.push((i >> 5) & 1);
        bits.push((i >> 4) & 1);
        bits.push((i >> 3) & 1);
        bits.push((i >> 2) & 1);
        bits.push((i >> 1) & 1);
        bits.push(i & 1);
    }

    Ok(bits)
}

fn get_huffmaned_value(
    ssss_table: &SSSSTable,
    image_bits: &mut Iter<u8>,
) -> Result<i32, HuffmanDecodingError> {
    let mut ssss: u8 = 0xFF;
    let mut guess: u32 = 1;

    for _ in 0..ssss_table.min_code_length - 1 {
        guess = (guess << 1) | (*image_bits.next().ok_or(OutOfBoundsError)? as u32);
    }

    for _ in 0..ssss_table.max_code_length {
        guess = (guess << 1) | (*image_bits.next().ok_or(OutOfBoundsError)? as u32);
        if ssss_table.table.contains_key(&guess) {
            ssss = ssss_table.table[&guess];
            break;
        }
    }

    match ssss {
        0xFF => {
            // if no code is matched return a zero, this was said to be the safest somewhere
            // TODO: should if break or be error resistant? also goes for down below
            // panic!("No matching Huffman code was found for a lossless tile jpeg.")
            // warnings.warn('A Huffman coding error was found in a lossless jpeg in a dng; it may'
            //               + ' have been resolved, there may be corrupted data')
            Err(HuffmanDecodingError::Default)
        }
        16 => Ok(32768),
        _ => {
            let mut pixel_diff: u16 = 0;
            if ssss > 0 {
                let first_bit = *image_bits.next().ok_or(OutOfBoundsError)?;
                pixel_diff = (pixel_diff << 1) | (first_bit as u16);
                // step thru the remainder of the ssss number of bits to get the coded number
                for _ in 0..ssss - 1 {
                    pixel_diff =
                        (pixel_diff << 1) | (*image_bits.next().ok_or(OutOfBoundsError)? as u16);
                }
                // if the first read bit is 0 the number is negative and has to be calculated
                if first_bit == 0 {
                    Ok(-(((1 << ssss) - (pixel_diff + 1)) as i32))
                } else {
                    Ok(pixel_diff as i32)
                }
            } else {
                Ok(0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use crate::test_utils;

    use super::*;

    #[test]
    fn parse_quantizaiton_table_good_8_bit() {
        let encoded_image = test_utils::get_file_as_byte_iter("raw_quantization_table_8_bit.bin");
        let mut encoded_image = encoded_image.iter();

        let actual_table = parse_quantizaiton_table(&mut encoded_image).unwrap();

        let expected_p_q = 0;
        let expected_t_q = 3;
        let expected_q_k =
            &test_utils::get_file_as_byte_iter("raw_quantization_table_8_bit_q_k.bin")
                .iter()
                .map(|x| *x as u16)
                .collect::<Vec<_>>()[..];

        assert_eq!(actual_table.p_q, expected_p_q);
        assert_eq!(actual_table.t_q, expected_t_q);
        assert_eq!(actual_table.q_k[..], expected_q_k[..]);
    }

    #[test]
    fn parse_quantizaiton_table_good_16_bit() {
        let encoded_image = test_utils::get_file_as_byte_iter("raw_quantization_table_16_bit.bin");
        let mut encoded_image = encoded_image.iter();

        let actual_table = parse_quantizaiton_table(&mut encoded_image).unwrap();

        let expected_p_q = 1;
        let expected_t_q = 5;
        let expected_q_k =
            &test_utils::get_file_as_byte_iter("raw_quantization_table_16_bit_q_k.bin")
                .chunks(2)
                .map(|x| (x[0] as u16) << 8 | (x[1] as u16))
                .collect::<Vec<_>>()[..];

        assert_eq!(actual_table.p_q, expected_p_q);
        assert_eq!(actual_table.t_q, expected_t_q);
        assert_eq!(actual_table.q_k[..], expected_q_k[..]);
    }

    #[test]
    fn skip_app_marker_good() {
        let encoded_image = test_utils::get_file_as_byte_iter("F-18.ljpg");

        let mut e_i: Vec<u8> = Vec::with_capacity(encoded_image.len());
        for i in 0..encoded_image.len() {
            e_i.push(encoded_image[i]);
        }

        let mut encoded_image = e_i.iter();
        for _ in 0..0x2 {
            encoded_image.next();
        }

        skip_app_marker(&mut encoded_image).unwrap();
        assert_eq!(encoded_image.len(), 42281);
    }

    #[test]
    fn get_huffmaned_value_0_bits() {
        let ssss_table = SSSSTable {
            t_c: 0,
            t_h: 0,
            table: HashMap::from([
                (4, 0),
                (30, 4),
                (6, 2),
                (126, 6),
                (254, 7),
                (510, 8),
                (14, 3),
                (5, 1),
                (62, 5),
            ]),
            min_code_length: 2,
            max_code_length: 8,
        };
        let image_bits: Vec<u8> = Vec::from([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
        let pixel_diff = get_huffmaned_value(&ssss_table, &mut image_bits.iter());
        assert_eq!(pixel_diff.unwrap(), 0);
    }

    #[test]
    fn get_huffmaned_value_1_bit() {
        let ssss_table = SSSSTable {
            t_c: 0,
            t_h: 0,
            table: HashMap::from([
                (4, 0),
                (30, 4),
                (6, 2),
                (126, 6),
                (254, 7),
                (510, 8),
                (14, 3),
                (5, 1),
                (62, 5),
            ]),
            min_code_length: 2,
            max_code_length: 8,
        };
        let image_bits: Vec<u8> = Vec::from([0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
        let pixel_diff = get_huffmaned_value(&ssss_table, &mut image_bits.iter());
        assert_eq!(pixel_diff.unwrap(), 1);
    }

    #[test]
    fn get_huffmaned_value_1_bit_neg() {
        let ssss_table = SSSSTable {
            t_c: 0,
            t_h: 0,
            table: HashMap::from([
                (4, 0),
                (30, 4),
                (6, 2),
                (126, 6),
                (254, 7),
                (510, 8),
                (14, 3),
                (5, 1),
                (62, 5),
            ]),
            min_code_length: 2,
            max_code_length: 8,
        };
        let image_bits: Vec<u8> = Vec::from([0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
        let pixel_diff = get_huffmaned_value(&ssss_table, &mut image_bits.iter());
        assert_eq!(pixel_diff.unwrap(), -1);
    }

    #[test]
    fn get_huffmaned_value_2_bits() {
        let ssss_table = SSSSTable {
            t_c: 0,
            t_h: 0,
            table: HashMap::from([
                (4, 0),
                (30, 4),
                (6, 2),
                (126, 6),
                (254, 7),
                (510, 8),
                (14, 3),
                (5, 1),
                (62, 5),
            ]),
            min_code_length: 2,
            max_code_length: 8,
        };
        let image_bits: Vec<u8> = Vec::from([1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
        let pixel_diff = get_huffmaned_value(&ssss_table, &mut image_bits.iter());
        assert_eq!(pixel_diff.unwrap(), 3);
    }

    #[test]
    fn get_huffmaned_value_2_bits_neg() {
        let ssss_table = SSSSTable {
            t_c: 0,
            t_h: 0,
            table: HashMap::from([
                (4, 0),
                (30, 4),
                (6, 2),
                (126, 6),
                (254, 7),
                (510, 8),
                (14, 3),
                (5, 1),
                (62, 5),
            ]),
            min_code_length: 2,
            max_code_length: 8,
        };
        let image_bits: Vec<u8> = Vec::from([1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
        let pixel_diff = get_huffmaned_value(&ssss_table, &mut image_bits.iter());
        assert_eq!(pixel_diff.unwrap(), -2);
    }

    #[test]
    fn get_huffmaned_value_16_bits() {
        let ssss_table = SSSSTable {
            t_c: 0,
            t_h: 0,
            table: HashMap::from([
                (4, 0),
                (30, 4),
                (6, 16),
                (126, 6),
                (254, 7),
                (510, 8),
                (14, 3),
                (5, 1),
                (62, 5),
            ]),
            min_code_length: 2,
            max_code_length: 16,
        };
        let image_bits: Vec<u8> = Vec::from([1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
        let pixel_diff = get_huffmaned_value(&ssss_table, &mut image_bits.iter());
        assert_eq!(pixel_diff.unwrap(), 32768);
    }

    #[test]
    // #[should_panic(expected = "No matching Huffman code was found for a lossless tile jpeg.")]
    fn get_huffmaned_value_panic() {
        let ssss_table = SSSSTable {
            t_c: 0,
            t_h: 0,
            table: HashMap::from([
                (4, 0),
                (30, 4),
                (6, 16),
                (126, 6),
                (254, 7),
                (510, 8),
                (14, 3),
                (5, 1),
                (62, 5),
            ]),
            min_code_length: 2,
            max_code_length: 8,
        };
        let image_bits: Vec<u8> = Vec::from([
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1,
        ]);
        let pixel_diff = get_huffmaned_value(&ssss_table, &mut image_bits.iter());
        println!("{:?}", pixel_diff);
        assert_eq!(
            pixel_diff.unwrap_err().to_string(),
            HuffmanDecodingError::Default.to_string()
        );
    }

    #[test]
    fn get_image_data_without_stuffed_zero_bytes_good_reguar_number_then_marker() {
        let encoded_image: Vec<u8> = Vec::from([0x00, 0xFE, 0x00, 0xFF, 0x00, 0x05, 0xFF, 0xDA]);
        let expected_bits: Vec<u8> = Vec::from([
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
            1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1,
        ]);

        let mut encoded_image = encoded_image.iter();

        let actual_bits = get_image_data_without_stuffed_zero_bytes(&mut encoded_image).unwrap();

        assert_eq!(actual_bits, expected_bits);
        assert_eq!(actual_bits.len(), 40);
        assert_eq!(encoded_image.next().unwrap(), &0xFF);
        assert_eq!(encoded_image.next().unwrap(), &0xDA);
    }

    #[test]
    fn get_image_data_without_stuffed_zero_bytes_good_padding_then_marker() {
        let encoded_image: Vec<u8> = Vec::from([0x00, 0xFE, 0x00, 0xFF, 0x00, 0xFF, 0xDA]);
        let expected_bits: Vec<u8> = Vec::from([
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
            1, 1, 1,
        ]);

        let mut encoded_image = encoded_image.iter();

        let actual_bits = get_image_data_without_stuffed_zero_bytes(&mut encoded_image).unwrap();

        assert_eq!(actual_bits, expected_bits);
        assert_eq!(actual_bits.len(), 32);
        assert_eq!(encoded_image.next().unwrap(), &0xFF);
        assert_eq!(encoded_image.next().unwrap(), &0xDA);
    }

    #[test]
    fn get_image_data_without_stuffed_zero_bytes_good_reguar_number_with_no_marker() {
        let encoded_image: Vec<u8> = Vec::from([0x00, 0xFE, 0x00, 0xFF, 0x00, 0x05]);
        let expected_bits: Vec<u8> = Vec::from([
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
            1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1,
        ]);

        let mut encoded_image = encoded_image.iter();

        let actual_bits = get_image_data_without_stuffed_zero_bytes(&mut encoded_image).unwrap();

        assert_eq!(actual_bits, expected_bits);
        assert_eq!(actual_bits.len(), 40);
        assert!(encoded_image.next().is_none());
    }

    #[test]
    fn get_image_data_without_stuffed_zero_bytes_good_padding_with_no_marker() {
        let encoded_image: Vec<u8> = Vec::from([0x00, 0xFE, 0x00, 0xFF, 0x00]);
        let expected_bits: Vec<u8> = Vec::from([
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
            1, 1, 1,
        ]);

        let mut encoded_image = encoded_image.iter();

        let actual_bits = get_image_data_without_stuffed_zero_bytes(&mut encoded_image).unwrap();

        assert_eq!(actual_bits, expected_bits);
        assert_eq!(actual_bits.len(), 32);
        assert!(encoded_image.next().is_none());
    }

    #[test]
    fn contextcontext_a_good() {
        let img = Vec::from([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 26, 27, 28, 29, 210, 211,
            212,
        ]);
        let components: usize = 2;
        let width: usize = 4;
        let image_index: usize = 10;
        let a_h: u8 = 2;
        let p_: u8 = 8;

        let context = ContextContext {
            component: image_index % components,
            x_position: (image_index / components) % width,
            y_position: (image_index / components) / width,
            width,
            component_count: components,
            p_t: a_h,
            p_: p_,
            img: &img,
        };

        let a = context.r_a();

        assert_eq!(a, 9);
    }

    #[test]
    fn contextcontext_b_good() {
        let img = Vec::from([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 26, 27, 28, 29, 210, 211,
            212,
        ]);
        let components: usize = 2;
        let width: usize = 4;
        let image_index: usize = 10;
        let a_h: u8 = 2;
        let p_: u8 = 8;

        let context = ContextContext {
            component: image_index % components,
            x_position: (image_index / components) % width,
            y_position: (image_index / components) / width,
            width,
            component_count: components,
            p_t: a_h,
            p_: p_,
            img: &img,
        };

        let b = context.r_b();

        assert_eq!(b, 3);
    }

    #[test]
    fn contextcontext_c_good() {
        let img = Vec::from([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 26, 27, 28, 29, 210, 211,
            212,
        ]);
        let components: usize = 2;
        let width: usize = 4;
        let image_index: usize = 10;
        let a_h: u8 = 2;
        let p_: u8 = 8;

        let context = ContextContext {
            component: image_index % components,
            x_position: (image_index / components) % width,
            y_position: (image_index / components) / width,
            width,
            component_count: components,
            p_t: a_h,
            p_: p_,
            img: &img,
        };

        let c = context.r_c();

        assert_eq!(c, 1);
    }

    #[test]
    fn contextcontext_ix_good() {
        let img = Vec::from([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 26, 27, 28, 29, 210, 211,
            212,
        ]);
        let components: usize = 2;
        let width: usize = 4;
        let image_index: usize = 10;
        let a_h: u8 = 2;
        let p_: u8 = 8;

        let context = ContextContext {
            component: image_index % components,
            x_position: (image_index / components) % width,
            y_position: (image_index / components) / width,
            width,
            component_count: components,
            p_t: a_h,
            p_: p_,
            img: &img,
        };

        let ix = context.r_ix();

        assert_eq!(ix, 1 << (p_ - a_h - 1));
    }

    #[test]
    fn parse_scan_header_good() {
        let encoded_image = test_utils::get_file_as_byte_iter("F-18.ljpg");

        let mut e_i: Vec<u8> = Vec::with_capacity(encoded_image.len());
        for i in 0..encoded_image.len() {
            e_i.push(encoded_image[i]);
        }

        let mut encoded_image = e_i.iter();
        for _ in 0..0x72 {
            encoded_image.next();
        }

        let scan_header = parse_scan_header(&mut encoded_image).unwrap();

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
        assert_eq!(scan_header.a_l_p_t, 0x00);
        assert_eq!(encoded_image.len(), 107634);
        assert_eq!(encoded_image.next().unwrap(), &0xFC);
    }

    #[test]
    fn parse_huffman_info_good() {
        let encoded_image = test_utils::get_file_as_byte_iter("F-18.ljpg");
        let mut encoded_image = encoded_image.iter();

        for _ in 0..0x17 {
            encoded_image.next();
        }

        let (t_c, t_h, code_lengths) = parse_huffman_info(&mut encoded_image).unwrap();

        let expected = [
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                Some(0),
                Some(1),
                Some(2),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            [
                Some(3),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            [
                Some(4),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            [
                Some(5),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            [
                Some(6),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            [
                Some(7),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            [
                Some(8),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
        ];

        assert_eq!(code_lengths, expected);
        assert_eq!(t_c, 0);
        assert_eq!(t_h, 0);
    }

    #[test]
    fn make_ssss_tables_good() {
        let code_lengths = [
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                Some(0),
                Some(1),
                Some(2),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            [
                Some(3),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            [
                Some(4),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            [
                Some(5),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            [
                Some(6),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            [
                Some(7),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            [
                Some(8),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
        ];

        let expected = HashMap::from([
            (4, 0),
            (30, 4),
            (6, 2),
            (126, 6),
            (254, 7),
            (510, 8),
            (14, 3),
            (5, 1),
            (62, 5),
        ]);

        let (tables, min_code_length, max_code_length) = make_ssss_table(code_lengths);

        assert_eq!(tables, expected);
        assert_eq!(min_code_length, 2);
        assert_eq!(max_code_length, 8);
    }

    #[test]
    fn make_ssss_tables_good2() {
        let code_lengths = [
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                Some(0),
                Some(1),
                Some(2),
                Some(3),
                Some(4),
                Some(5),
                Some(6),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
        ];

        let expected = HashMap::from([(8, 0), (9, 1), (10, 2), (11, 3), (12, 4), (13, 5), (14, 6)]);

        let (tables, min_code_length, max_code_length) = make_ssss_table(code_lengths);

        assert_eq!(tables, expected);
        assert_eq!(min_code_length, 3);
        assert_eq!(max_code_length, 3);
    }

    #[test]
    fn parse_frame_header_good() {
        let encoded_image = test_utils::get_file_as_byte_iter("F-18.ljpg");

        let mut e_i: Vec<u8> = Vec::with_capacity(encoded_image.len());
        for i in 0..encoded_image.len() {
            e_i.push(encoded_image[i]);
        }

        let mut encoded_image = e_i.iter();
        for _ in 0..4 {
            encoded_image.next();
        }

        let frame_header = parse_frame_header(&mut encoded_image, 0xFFC3).unwrap();

        assert_eq!(frame_header.marker, 0xFFC3);
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
        assert_eq!(encoded_image.len(), 107739);
        assert_eq!(encoded_image.next().unwrap(), &0xFF);
    }

    #[test]
    fn is_jpg_true() {
        let encoded_image = Vec::from([0xFF, 0xD8, 0x08]);
        let mut encoded_image = encoded_image.iter();

        let expected = 0x08_u8;

        assert!(is_jpeg(&mut encoded_image).is_ok());
        assert_eq!(encoded_image.next().unwrap(), &expected);
    }

    // #[test]
    // #[should_panic(expected = "This doesn't seem to be a JPEG.")]
    // fn is_jpg_false() {
    //     let encoded_image = Vec::from([0xFF, 0xC4, 0x08]);
    //     let encoded_image = encoded_image.iter();

    //     is_jpeg(encoded_image);
    // }

    #[test]
    fn bytes_to_int_two_consumed_good() {
        let buffer = Vec::from([0xC2_u8, 0x1D, 0x8D, 0xE4, 0x0C, 0x9A]);
        let mut buffer = buffer.iter();
        let expected: u16 = 0xC21D_u16;
        assert_eq!(bytes_to_int_two_consumed(&mut buffer).unwrap(), expected);
        assert_eq!(buffer.next().unwrap(), &0x8D);
    }

    #[test]
    fn bytes_to_int_two_consumed_fail() {
        let buffer = Vec::from([0xC2_u8, 0x1D, 0x8D, 0xE4, 0x0C, 0x9A]);
        let mut buffer = buffer.iter();
        let expected = 7564_u16;
        assert_ne!(bytes_to_int_two_consumed(&mut buffer).unwrap(), expected);
        assert_eq!(buffer.next().unwrap(), &0x8D);
    }

    #[test]
    fn bytes_to_int_two_peeked_good() {
        let buffer = Vec::from([0xC2_u8, 0x1D, 0x8D, 0xE4, 0x0C, 0x9A]);
        let mut buffer = buffer.iter();
        let expected: u16 = 0xC21D_u16;
        assert_eq!(bytes_to_int_two_peeked(&mut buffer).unwrap(), expected);
        assert_eq!(buffer.next().unwrap(), &0x1D);
    }

    #[test]
    fn bytes_to_int_two_peeked_fail() {
        let buffer = Vec::from([0xC2_u8, 0x1D, 0x8D, 0xE4, 0x0C, 0x9A]);
        let mut buffer = buffer.iter();
        let expected = 7564_u16;
        assert_ne!(bytes_to_int_two_peeked(&mut buffer).unwrap(), expected);
        assert_eq!(buffer.next().unwrap(), &0x1D);
    }
}

// // let mut all_code_lengths: Vec<[[u8; 16]; 16]> = Vec::new();
// // all_code_lengths.push([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [0, 1, 2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [4, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [5, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]]);
// // all_code_lengths.push([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [0, 1, 2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [4, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [5, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [9, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]]);
// // all_code_lengths.push([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [0, 1, 2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [4, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [5, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]]);

// // let mut expected: Vec<[[u8; 16]; 16]> = Vec::new();
// // expected.push([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [0, 1, 2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [4, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [5, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]]);
// // expected.push([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [0, 1, 2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [4, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [5, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [9, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]]);
// // expected.push([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [0, 1, 2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [4, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [5, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
// //                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]]);
