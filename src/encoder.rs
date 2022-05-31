// https://www.youtube.com/watch?v=Kv1Hiv3ox8I
//https://www.ece.ucdavis.edu/cerl/reliablejpeg/compression/

use std::collections::HashMap;
use std::collections::HashSet;
use std::f64::consts::PI;

use color;

use crate::jpeg_errors::*;
use crate::jpeg_utils;
use crate::jpeg_utils::{ContextContext, Format, Marker::*, Precision};
use crate::Format::*;
use crate::Precision::*;

pub enum Colorspace {
    RGB,
    YCbCr,
}

pub enum ComponentCount {
    One = 1,
    Two = 2,
    Three = 3,
}

/// Encodes raw RGB images into JPG images
/// Input/raw images must be RGB and must have
/// both vertical and horizontal sampling
/// factors of 1.
pub(crate) fn encode(
    raw_image: Vec<u32>,
    width: u16,
    height: u16,
    target_format: Format,
    source_precision: Precision,
) -> Vec<u8> {
    let component_count = (raw_image.len() / (width as usize) / (height as usize)) as u8;
    let p_t = 5;
    let predictor = 2;
    let raw_image = set_color(raw_image, &target_format, &source_precision);
    let p_ = source_precision as u8;

    let pixel_differences = get_pixel_vs_prediction_difference(
        &raw_image,
        width as usize,
        component_count as usize,
        p_,
        p_t,
        predictor,
    );
    let ssss_vs_code = make_ssss_vs_code(&pixel_differences, component_count as usize);

    let mut frame_header = make_frame_header(width, height, &target_format, component_count, p_);
    let mut huffman_headers = make_huffman_headers(&ssss_vs_code);

    let mut image = encode_image(&pixel_differences, &ssss_vs_code);
    let mut encoded_image = Vec::new();
    encoded_image.append(&mut frame_header);
    encoded_image.append(&mut huffman_headers);
    encoded_image.append(&mut image);
    add_u16_to_vec(&mut encoded_image, EOI as u16);
    encoded_image
}

fn encode_image(pixel_differences: &Vec<i32>, ssss_vs_code: &Vec<HashMap<u8, u32>>) -> Vec<u8> {
    let mut image_buffer: Vec<bool> = Vec::with_capacity(pixel_differences.len() * 8);
    let component_count = ssss_vs_code.len();
    for (idx, diff) in pixel_differences.iter().enumerate() {
        let component = idx % component_count;
        let table = &ssss_vs_code[component];
        let ssss = difference_to_ssss(*diff);
        append_ssss_code(&mut image_buffer, *table.get(&ssss).unwrap());
        append_difference(&mut image_buffer, *diff, ssss);
    }
    convert_bits_to_bytes(image_buffer)
}

fn convert_bits_to_bytes(mut bits_buffer: Vec<bool>) -> Vec<u8> {
    while bits_buffer.len() % 8 != 0 {
        bits_buffer.push(true);
    }

    let mut bytes_buffer = Vec::with_capacity((bits_buffer.len() as f32 / 8. * 1.5) as usize);
    for bool_byte in bits_buffer.chunks(8) {
        let mut byte = 0u8;
        for (idx, bool_bit) in bool_byte.iter().enumerate() {
            if *bool_bit {
                byte &= 1 << idx;
            }
        }
        bytes_buffer.push(byte);
        if byte == 0xFF {
            bytes_buffer.push(0);
        }
    }

    bytes_buffer
}

fn append_difference(image_buffer: &mut Vec<bool>, diff: i32, ssss: u8) {
    if ssss == 0 || ssss == 16 {
        return;
    }
    let diff = if diff < 0 {
        (1 << ssss) + diff - 1
    } else {
        diff
    };

    for shift in (0..ssss).rev() {
        let bit = diff >> shift & 1;
        image_buffer.push(bit == 1);
    }
}

fn append_ssss_code(image_buffer: &mut Vec<bool>, code: u32) {
    for shift in (0..jpeg_utils::number_of_used_bits(&code) - 1).rev() {
        let bit = (code >> shift) & 1;
        image_buffer.push(bit == 1);
    }
}

fn make_ssss_vs_code(
    pixel_differences: &[i32],
    component_count: usize,
) -> Vec<HashMap<u8, u32>> {
    let mut ssss_vs_code = Vec::new();

    for component in 0..component_count {
        let ssss_frequencies = get_ssss_frequencies(
            pixel_differences
                .iter()
                .enumerate()
                .filter_map(|(idx, diff)| {
                    if idx % component_count == component {
                        Some(*diff)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>(),
        );
        ssss_vs_code.push(make_huffman_codes(ssss_frequencies));
    }

    ssss_vs_code
}

fn get_pixel_vs_prediction_difference(
    raw_image: &Vec<u32>,
    width: usize,
    component_count: usize,
    p_: u8,
    p_t: u8,
    predictor: u8,
) -> Vec<i32> {
    let mut pixel_differences = Vec::new();
    for (idx, &ix) in raw_image.iter().enumerate() {
        let p_x = jpeg_utils::make_prediciton(
            raw_image,
            idx,
            component_count,
            width,
            p_,
            p_t,
            predictor,
        ) as i32;
        pixel_differences.push((p_x - ix as i32) % 2i32.pow(16));
    }
    pixel_differences
}

fn make_huffman_headers(ssss_vs_code: &[HashMap<u8, u32>]) -> Vec<u8> {
    let mut total_huffman_header = Vec::new();

    for (component, table) in ssss_vs_code.iter().enumerate() {
        let t_c_t_h = {
            let t_c = 0u8;
            let t_h = component as u8;
            ((t_c & 0xF) << 4) | (t_h & 0xF)
        };
        let (l_i, v_ij) = {
            let code_vs_ssss = table
                .iter()
                .map(|(&k, &v)| (v, k))
                .collect::<HashMap<_, _>>();
            let mut length_and_sssses = HashMap::new();
            for (code, ssss) in code_vs_ssss {
                let sssses = length_and_sssses
                    .entry(jpeg_utils::number_of_used_bits(&code) - 1)
                    .or_insert(Vec::new());
                sssses.push(ssss);
            }
            let mut l_i = Vec::new();
            let mut v_ij: Vec<u8> = Vec::new();
            for length in 1..=16 {
                if let Some(sssses) = length_and_sssses.get(&length) {
                    l_i.push(sssses.len());
                    v_ij.extend(sssses.iter());
                } else {
                    l_i.push(0);
                }
            }
            (l_i, v_ij)
        };
        add_u16_to_vec(&mut total_huffman_header, DHT as u16);
        add_u16_to_vec(&mut total_huffman_header, 2 + 1 + 16 + v_ij.len() as u16);
        total_huffman_header.push(t_c_t_h);
        total_huffman_header.extend(l_i.iter().map(|v| *v as u8));
        total_huffman_header.extend(v_ij.iter().map(|v| *v as u8));
    }
    total_huffman_header
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone)]
struct HuffmanTableNode {
    frequency: u32,
    ssss: Option<u8>,
    last_child: Option<Box<HuffmanTableNode>>,
    next_to_last_child: Option<Box<HuffmanTableNode>>,
}

fn make_huffman_codes(ssss_frequencies: HashMap<u8, u32>) -> HashMap<u8, u32> {
    let mut nodes = ssss_frequencies
        .iter()
        .map(|(k, v)| HuffmanTableNode {
            frequency: *v,
            ssss: Some(*k),
            last_child: None,
            next_to_last_child: None,
        })
        .collect::<Vec<_>>();
    nodes.sort();
    nodes.reverse();
    while nodes.len() > 1 {
        let last = nodes.pop().unwrap();
        let next_to_last = nodes.pop().unwrap();
        nodes.push(HuffmanTableNode {
            frequency: last.frequency + next_to_last.frequency,
            ssss: None,
            last_child: Some(Box::from(last)),
            next_to_last_child: Some(Box::from(next_to_last)),
        })
    }
    extend_code(1, nodes[0].clone())
}

fn extend_code(code: u32, node: HuffmanTableNode) -> HashMap<u8, u32> {
    if node.ssss.is_some() {
        HashMap::from([(node.ssss.unwrap(), code)])
    } else {
        let last_child = Box::into_inner(node.last_child.unwrap());
        let last_child_code = code << 1 | 1; // Adding a one to the end of the code
        let mut last_childs_codes = extend_code(last_child_code, last_child);
        let next_to_last_child = Box::into_inner(node.next_to_last_child.unwrap());
        let next_to_last_child_code = code << 1; // Adding a zero to the end of the code
        let next_to_last_childs_codes = extend_code(next_to_last_child_code, next_to_last_child);
        last_childs_codes.extend(next_to_last_childs_codes);
        last_childs_codes
    }
}

fn difference_to_ssss(difference: i32) -> u8 {
    match difference {
        0 => 0,
        x if -1 <= x || x <= 1 => 1,
        x if -3 < x || x <= 3 => 2,
        x if -7 < x || x <= 7 => 3,
        x if -15 < x || x <= 15 => 4,
        x if -31 < x || x <= 31 => 5,
        x if -63 < x || x <= 63 => 6,
        x if -127 < x || x <= 127 => 7,
        x if -255 < x || x <= 255 => 8,
        x if -511 < x || x <= 511 => 9,
        x if -1023 < x || x <= 1023 => 10,
        x if -2047 < x || x <= 2047 => 11,
        x if -4095 < x || x <= 4095 => 12,
        x if -8191 < x || x <= 8191 => 13,
        x if -16383 < x || x <= 16383 => 14,
        x if -32767 < x || x <= 32767 => 15,
        32768 => 16,
        _ => panic!("That's not a huffman code!"),
    }
}

fn get_ssss_frequencies(pixel_differences: Vec<i32>) -> HashMap<u8, u32> {
    let mut frequencies = HashMap::new();
    for diff in pixel_differences {
        let ssss = difference_to_ssss(diff);
        let count = frequencies.entry(ssss).or_insert(0);
        *count += 1;
    }
    frequencies
}

/// Makes a Jpeg Frame Header
/// See 10918-1, B.2.2, P. 35
fn make_frame_header(
    width: u16,
    height: u16,
    target_format: &Format,
    component_count: u8,
    p_: u8,
) -> Vec<u8> {
    let mut frame_header = Vec::new();

    match target_format {
        BaselineSequential => add_u16_to_vec(&mut frame_header, SOF0 as u16),
        Lossless => add_u16_to_vec(&mut frame_header, SOF3 as u16),
    }
    let l_f = 2 + 1 + 2 + 2 + 1 + 3 * (component_count as u16);
    add_u16_to_vec(&mut frame_header, l_f);
    frame_header.push(p_);
    add_u16_to_vec(&mut frame_header, height);
    add_u16_to_vec(&mut frame_header, width);
    frame_header.push(component_count);
    for component in 0..component_count {
        frame_header.push(component);
        match target_format {
            BaselineSequential => {
                match component {
                    0 => frame_header.push(0x11),
                    1 | _ => frame_header.push(0x22),
                }
                frame_header.push(component);
            }
            Lossless => {
                frame_header.push(0x11);
                frame_header.push(0x0);
            }
        }
    }

    frame_header
}

fn set_color(
    raw_image: Vec<u32>,
    target_format: &Format,
    source_precision: &Precision,
) -> Vec<u32> {
    use Format::*;
    match target_format {
        Lossless => raw_image,
        BaselineSequential => {
            let raw_image = match source_precision.clone() {
                Eight => raw_image,
                // precision => raw_image
                //     .into_iter()
                //     .map(|x| x * 255 / 2u32.pow(&precision as u32) as u32)
                //     .collect::<Vec<_>>()
                Twelve => raw_image
                    .into_iter()
                    .map(|x| x * 255 / 2u32.pow(12) as u32)
                    .collect::<Vec<_>>(),
                Sixteen => raw_image
                    .into_iter()
                    .map(|x| x * 255 / u16::MAX as u32)
                    .collect::<Vec<_>>(),
            };
            let mut image = Vec::with_capacity(raw_image.len());
            raw_image.chunks(3).for_each(|p| {
                let c = color::rgb_to_ycbcr(p[0], p[1], p[2]);
                image.append(&mut Vec::from([c.0, c.1, c.2]));
            });
            image
        }
    }
}

fn add_u16_to_vec(vec: &mut Vec<u8>, number: u16) {
    vec.push((number >> 8) as u8);
    vec.push((number & 0x00FF) as u8);
}

// // input types "jfif"
// pub struct JPEGEncoder {
//     y: Vec<Vec<u8>>,
//     cb: Vec<Vec<u8>>,
//     cr: Vec<Vec<u8>>,
//     // image: Vec<u8>,
//     width: usize,
//     height: usize,
// }

// impl JPEGEncoder {
//     pub fn from_rgb(raw_image: Vec<u8>, width: usize, height: usize, format: String) {
//         let p_ = 8_u8;

//         let raw_image = rgb_to_ycbcr(raw_image);
//         let ((y, cb, cr), width, height) = reshape_and_split_channels(raw_image, width, height);
//         let y = break_into_8x8_blocks(y, width, height);
//         let cb = break_into_8x8_blocks(cb, width, height);
//         let cr = break_into_8x8_blocks(cr, width, height);
//         let cb = level_shift(cb, p_);
//         let cr = level_shift(cr, p_);
//     }

//     pub fn save(&self, file: String) {}
// }

// fn fdct(block: &Vec<i8>, tqi: u8) {
//     // Svu = DCT coefficient at horizontal frequency u, vertical frequency v
//     let lumiance_qt = Vec::from([
//         16.0, 11.0, 10.0, 16.0, 24.0, 40.0, 51.0, 61.0, 12.0, 12.0, 14.0, 19.0, 26.0, 58.0, 60.0,
//         55.0, 14.0, 13.0, 16.0, 24.0, 40.0, 57.0, 69.0, 56.0, 14.0, 17.0, 22.0, 29.0, 51.0, 87.0,
//         80.0, 62.0, 18.0, 22.0, 37.0, 56.0, 68.0, 109.0, 103.0, 77.0, 24.0, 35.0, 55.0, 64.0, 81.0,
//         104.0, 113.0, 92.0, 49.0, 64.0, 78.0, 87.0, 103.0, 121.0, 120.0, 101.0, 72.0, 92.0, 95.0,
//         98.0, 112.0, 100.0, 103.0, 99.0,
//     ]);

//     let crominance_qt = Vec::from([
//         17.0, 18.0, 24.0, 47.0, 99.0, 99.0, 99.0, 99.0, 18.0, 21.0, 26.0, 66.0, 99.0, 99.0, 99.0,
//         99.0, 24.0, 26.0, 56.0, 99.0, 99.0, 99.0, 99.0, 99.0, 47.0, 66.0, 99.0, 99.0, 99.0, 99.0,
//         99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
//         99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
//         99.0, 99.0, 99.0, 99.0,
//     ]);

//     // let mut dct_coeffs: Vec<f64> = Vec::with_capacity(64);
//     let mut dct_coeffs: [f64; 64] = [0.0; 64];

//     for vu in 0..64 {
//         let u = (vu % 8) as f64;
//         let v = (vu / 8) as f64;
//         let cu = if u == 0.0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };
//         let cv = if v == 0.0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };
//         let mut block_sum = 0.0;
//         for sample in block.iter() {
//             let mut outer_sum = 0.0;
//             for x in 0..8 {
//                 let mut inner_sum = 0.0;
//                 for y in 0..8 {
//                     // let a = 0;
//                     inner_sum += *sample as f64
//                         * ((2 * x + 1) as f64 * u * PI / 16.0).cos()
//                         * ((2 * y + 1) as f64 * v * PI / 16.0).cos()
//                 }
//                 outer_sum += inner_sum;
//             }
//             block_sum += 1.0 / 4.0 * cu * cv * outer_sum;
//         }
//         // dct_coeffs.push(block_sum);
//         dct_coeffs[vu] = block_sum;
//     }

//     let table = if tqi == 1 { lumiance_qt } else { crominance_qt };

//     let sqvu: Vec<f64> = dct_coeffs
//         .iter()
//         .zip(table.iter())
//         .map(|(s, q)| (s / q).round())
//         .collect();
// }

// fn level_shift(channel: Vec<Vec<u8>>, p_: u8) -> Vec<Vec<i8>> {
//     let shift = 2_i16.pow(p_ as u32 - 1);
//     let channel: Vec<Vec<i8>> = channel
//         .iter()
//         .map(|i| i.iter().map(|i| (*i as i16 - shift) as i8).collect())
//         .collect();

//     channel
// }

// fn break_into_8x8_blocks(raw_image: Vec<u8>, width: usize, height: usize) -> Vec<Vec<u8>> {
//     let mut image: Vec<Vec<u8>> = Vec::with_capacity((width / 8) * (height * 8));

//     for _ in 0..(width / 8) * (height / 8) {
//         image.push(Vec::with_capacity(8 * 8));
//     }

//     let blocks_wide = width / 8;
//     for (i, val) in raw_image.iter().enumerate() {
//         let block_in_row = (i % width) / 8;
//         let block_row_in_column = i / width / 8;

//         image[block_row_in_column * blocks_wide + block_in_row].push(*val);
//     }

//     image
// }

// fn reshape_and_split_channels(
//     raw_image: Vec<u8>,
//     width: usize,
//     height: usize,
// ) -> ((Vec<u8>, Vec<u8>, Vec<u8>), usize, usize) {
//     let w = if width % 16 == 0 {
//         width
//     } else {
//         width - width % 16
//     };
//     let h = if height % 16 == 0 {
//         height
//     } else {
//         height - height % 16
//     };

//     let mut y_: Vec<u8> = Vec::with_capacity(w * h);
//     let mut cb_: Vec<u16> = Vec::with_capacity(w * h);
//     let mut cr_: Vec<u16> = Vec::with_capacity(w * h);

//     for i in 0..(raw_image.len() / 3) {
//         let x = i % width;
//         let y = i / width;
//         if y == h {
//             break;
//         }
//         if x < w {
//             y_.push(raw_image[i * 3]);
//             cb_.push(raw_image[i * 3 + 1] as u16);
//             cr_.push(raw_image[i * 3 + 2] as u16);
//         }
//     }

//     let mut cb: Vec<u8> = Vec::with_capacity((w / 2) * (h / 2));
//     let mut cr: Vec<u8> = Vec::with_capacity((w / 2) * (h / 2));

//     for i in 0..(y_.len() / (2 * 2)) {
//         let j = (i / (w / 2)) * w + i * 2;

//         cb.push(((cb_[j] + cb_[j + 1] + cb_[j + w] + cb_[j + 1 + w]) / 4) as u8);
//         cr.push(((cr_[j] + cr_[j + 1] + cr_[j + w] + cr_[j + 1 + w]) / 4) as u8);
//     }

//     ((y_, cb, cr), w, h)
// }

#[cfg(test)]
mod tests {
    extern crate test;

    use std::env;
    use std::fs::File;
    use std::io::Read;
    use std::path::Path;
    use test::Bencher;

    use std::iter::Zip;

    use super::*;

    #[test]
    fn make_frame_header_lessless_good() {
        let width = 250;
        let height = 300;
        let target_format = &Format::Lossless;
        let component_count = 3;
        let precision = 8;
        let actual = make_frame_header(width, height, target_format, component_count, precision);

        let mut expected = Vec::new();
        expected.push(0xFF);
        expected.push(0xC3);
        expected.push(0);
        expected.push(17);
        expected.push(precision);
        expected.push((height >> 8) as u8);
        expected.push((height & 0xFF) as u8);
        expected.push((width >> 8) as u8);
        expected.push((width & 0xFF) as u8);
        expected.push(component_count);
        expected.push(0);
        expected.push(0x11);
        expected.push(0);
        expected.push(1);
        expected.push(0x11);
        expected.push(0);
        expected.push(2);
        expected.push(0x11);
        expected.push(0);

        assert_eq!(actual, expected);
    }

    #[test]
    fn add_u16_to_vec_good() {
        let mut actual = Vec::from([1, 2, 3]);
        add_u16_to_vec(&mut actual, 0xFFC0);
        let expected = Vec::from([1u8, 2, 3, 255, 192]);
        assert_eq!(actual, expected);
    }

    // #[test]
    // fn rgb_to_ycbcr_good() {
    //     let input_image: Vec<u8> = Vec::from([155, 90, 30, 18, 196, 18]);
    //     let expected_image: Vec<u8> = Vec::from([104, 92, 160, 121, 76, 62]);

    //     let actual_image = rgb_to_ycbcr(input_image);

    //     for (expected, actual) in expected_image.iter().zip(actual_image.iter()) {
    //         assert!(expected == actual);
    //     }
    // }

    // #[test]
    // fn reshape_and_split_channels_good() {
    //     let input_image = input_image_34x34_mtn();
    //     let expected_y = input_image_34x34_mtn_y();
    //     let expected_cb = input_image_34x34_mtn_cb();
    //     let expected_cr = input_image_34x34_mtn_cr();

    //     let ((y, cb, cr), width, height) = reshape_and_split_channels(input_image, 34, 34);

    //     assert_eq!(y, expected_y);
    //     assert_eq!(cb, expected_cb);
    //     assert_eq!(cr, expected_cr);
    //     assert_eq!(width, 32);
    //     assert_eq!(height, 32);
    // }

    // #[test]
    // fn break_into_8x8_good() {
    //     let cb = input_image_34x34_mtn_cb();
    //     let actual_cb = break_into_8x8_blocks(cb, 16, 16);
    //     let expected_cb = split_input_image_34x34_mtn_cb();

    //     assert_eq!(actual_cb, expected_cb);
    // }

    // #[test]
    // fn level_shift_good() {
    //     let input_cb = split_input_image_34x34_mtn_cb();
    //     let actual_cb = level_shift(input_cb, 8);

    //     let expected_cb: Vec<Vec<i8>> = split_and_shifted_input_image_34x34_mtn_cb();

    //     assert_eq!(actual_cb, expected_cb);
    // }

    // #[test]
    // fn fdct_good() {
    //     let block = &split_and_shifted_input_image_34x34_mtn_cb()[0];

    //     fdct(&block, 1);
    // }

    // // fn split_and_shifted_input_image_34x34_cb() -> Vec<Vec<i8>> {
    // //     Vec::from([Vec::from([-43, 89, -84, 15, 26, 74, -49, -36, -15, -20, -17, 70, 19, -11, -26, 34, -29, 18, -4, 23, -15, -29, 32, 16, 35, 40, 11, -62, -16, -25, -19, -62, -73, -5, -43, -22, -27, -9, 84, 22, -19, 62, 20, -19, -43, 8, 31, -27, -21, 40, 35, -32, -37, 43, -1, 47, 10, 27, -3, -34, -8, -9, -60, 0]),
    // //                Vec::from([5, 32, -49, 1, 21, 30, -4, 64, 57, -3, 48, 1, -16, 24, 107, -65, -45, 7, -18, 55, 12, -56, 43, -19, 21, -41, 23, 29, 10, -42, 48, -45, -25, 7, 3, -22, -21, 38, -14, -32, -76, 37, 25, 22, 43, -29, -14, -7, -5, -31, 7, 57, 9, -5, 50, 11, -30, -30, -55, 13, 23, 20, 2, -8]),
    // //                Vec::from([67, -63, 57, 18, 40, -19, -7, -34, 29, -23, -25, -23, 63, 9, -1, -5, -3, 7, 24, 16, -60, 10, 0, -49, 4, -4, 44, -77, -13, -9, -9, -75, 13, 15, 87, -12, 70, 32, 60, -14, 27, -36, -36, -11, 47, -3, -54, 21, -5, 43, 67, 46, 4, -79, 19, -20, 6, -11, -7, 4, 10, 21, 16, 43]),
    // //                Vec::from([-34, 10, 22, -57, -2, 1, -18, -35, -1, -22, -6, 8, -67, -6, 1, -49, -41, 43, -2, 47, -53, -28, -5, 18, 2, -83, -29, -3, 85, -4, 62, -1, -1, 23, -35, 11, 30, 46, 64, -32, 45, -45, 77, 48, -13, -21, -19, -40, 34, -1, -76, 25, -7, -20, 6, -15, -18, 13, -6, 50, -10, 11, -18, -10])])
    // // }

    // // fn split_input_image_34x34_cb() -> Vec<Vec<u8>> {
    // //     Vec::from([Vec::from([85, 217, 44, 143, 154, 202, 79, 92, 113, 108, 111, 198, 147, 117, 102, 162, 99, 146, 124, 151, 113, 99, 160, 144, 163, 168, 139, 66, 112, 103, 109, 66, 55, 123, 85, 106, 101, 119, 212, 150, 109, 190, 148, 109, 85, 136, 159, 101, 107, 168, 163, 96, 91, 171, 127, 175, 138, 155, 125, 94, 120, 119, 68, 128]),
    // //                Vec::from([133, 160, 79, 129, 149, 158, 124, 192, 185, 125, 176, 129, 112, 152, 235, 63, 83, 135, 110, 183, 140, 72, 171, 109, 149, 87, 151, 157, 138, 86, 176, 83, 103, 135, 131, 106, 107, 166, 114, 96, 52, 165, 153, 150, 171, 99, 114, 121, 123, 97, 135, 185, 137, 123, 178, 139, 98, 98, 73, 141, 151, 148, 130, 120]),
    // //                Vec::from([195, 65, 185, 146, 168, 109, 121, 94, 157, 105, 103, 105, 191, 137, 127, 123, 125, 135, 152, 144, 68, 138, 128, 79, 132, 124, 172, 51, 115, 119, 119, 53, 141, 143, 215, 116, 198, 160, 188, 114, 155, 92, 92, 117, 175, 125, 74, 149, 123, 171, 195, 174, 132, 49, 147, 108, 134, 117, 121, 132, 138, 149, 144, 171]),
    // //                Vec::from([94, 138, 150, 71, 126, 129, 110, 93, 127, 106, 122, 136, 61, 122, 129, 79, 87, 171, 126, 175, 75, 100, 123, 146, 130, 45, 99, 125, 213, 124, 190, 127, 127, 151, 93, 139, 158, 174, 192, 96, 173, 83, 205, 176, 115, 107, 109, 88, 162, 127, 52, 153, 121, 108, 134, 113, 110, 141, 122, 178, 118, 139, 110, 118])])
    // // }

    // // fn input_image_34x34_cr() -> Vec<u8> {
    // //     Vec::from([188, 87, 148, 127, 126, 171, 62, 61, 81, 169, 69, 83, 127, 115, 130, 133, 137, 138, 110, 179, 142, 128, 140, 170, 180, 138, 81, 63, 182, 119, 180, 190, 118, 180, 167, 120, 101, 194, 135, 159, 141, 150, 87, 124, 149, 139, 141, 117, 101, 209, 135, 79, 171, 81, 73, 98, 139, 159, 138, 143, 165, 148, 69, 160, 133, 72, 191, 122, 115, 47, 98, 102, 143, 100, 110, 118, 172, 57, 167, 177, 119, 122, 162, 80, 120, 98, 97, 99, 138, 146, 98, 108, 208, 114, 65, 170, 97, 126, 86, 78, 147, 65, 65, 119, 72, 100, 134, 98, 137, 123, 125, 179, 147, 144, 103, 144, 85, 172, 92, 189, 152, 60, 148, 107, 126, 109, 175, 155, 132, 120, 52, 73, 105, 143, 180, 122, 137, 135, 196, 86, 97, 104, 158, 81, 100, 194, 119, 78, 165, 66, 130, 207, 178, 133, 89, 63, 174, 132, 116, 146, 72, 97, 68, 130, 137, 127, 145, 118, 157, 170, 156, 215, 153, 105, 78, 91, 71, 135, 75, 107, 75, 127, 212, 187, 153, 165, 139, 105, 91, 66, 83, 98, 137, 144, 84, 87, 151, 97, 67, 134, 74, 127, 51, 79, 147, 174, 176, 148, 112, 109, 137, 177, 189, 86, 162, 125, 108, 131, 123, 89, 124, 109, 153, 110, 120, 60, 175, 169, 107, 75, 152, 87, 87, 125, 175, 99, 146, 100, 122, 94, 151, 118, 209, 138, 220, 172, 142, 181, 95, 129, 135, 133, 114, 125, 59, 131])
    // // }

    // // fn input_image_34x34_cb() -> Vec<u8> {
    // //     Vec::from([85, 217, 44, 143, 154, 202, 79, 92, 133, 160, 79, 129, 149, 158, 124, 192, 113, 108, 111, 198, 147, 117, 102, 162, 185, 125, 176, 129, 112, 152, 235, 63, 99, 146, 124, 151, 113, 99, 160, 144, 83, 135, 110, 183, 140, 72, 171, 109, 163, 168, 139, 66, 112, 103, 109, 66, 149, 87, 151, 157, 138, 86, 176, 83, 55, 123, 85, 106, 101, 119, 212, 150, 103, 135, 131, 106, 107, 166, 114, 96, 109, 190, 148, 109, 85, 136, 159, 101, 52, 165, 153, 150, 171, 99, 114, 121, 107, 168, 163, 96, 91, 171, 127, 175, 123, 97, 135, 185, 137, 123, 178, 139, 138, 155, 125, 94, 120, 119, 68, 128, 98, 98, 73, 141, 151, 148, 130, 120, 195, 65, 185, 146, 168, 109, 121, 94, 94, 138, 150, 71, 126, 129, 110, 93, 157, 105, 103, 105, 191, 137, 127, 123, 127, 106, 122, 136, 61, 122, 129, 79, 125, 135, 152, 144, 68, 138, 128, 79, 87, 171, 126, 175, 75, 100, 123, 146, 132, 124, 172, 51, 115, 119, 119, 53, 130, 45, 99, 125, 213, 124, 190, 127, 141, 143, 215, 116, 198, 160, 188, 114, 127, 151, 93, 139, 158, 174, 192, 96, 155, 92, 92, 117, 175, 125, 74, 149, 173, 83, 205, 176, 115, 107, 109, 88, 123, 171, 195, 174, 132, 49, 147, 108, 162, 127, 52, 153, 121, 108, 134, 113, 134, 117, 121, 132, 138, 149, 144, 171, 110, 141, 122, 178, 118, 139, 110, 118])
    // // }

    // // fn input_image_34x34_y() -> Vec<u8> {
    // //     Vec::from([145, 40, 167, 227, 102, 90, 19, 20, 186, 107, 58, 12, 36, 181, 101, 87, 219, 250, 74, 1, 169, 243, 81, 226, 77, 92, 141, 170, 205, 10, 106, 60, 203, 228, 93, 189, 83, 195, 177, 124, 140, 248, 17, 71, 103, 115, 31, 213, 160, 21, 45, 39, 210, 13, 224, 158, 127, 139, 222, 231, 62, 192, 91, 135, 65, 179, 162, 111, 52, 30, 23, 234, 230, 2, 214, 14, 80, 187, 239, 121, 150, 19, 33, 216, 155, 117, 134, 204, 66, 74, 65, 80, 109, 94, 243, 82, 143, 77, 78, 199, 64, 23, 31, 220, 104, 157, 121, 8, 169, 95, 44, 43, 146, 253, 222, 47, 255, 69, 249, 118, 57, 16, 248, 84, 90, 160, 186, 54, 175, 127, 5, 159, 85, 1, 141, 20, 71, 101, 164, 128, 49, 83, 193, 179, 131, 227, 62, 36, 28, 244, 9, 2, 61, 254, 55, 221, 234, 27, 183, 116, 180, 252, 191, 3, 151, 90, 249, 122, 118, 160, 137, 254, 112, 39, 221, 225, 141, 100, 88, 65, 229, 8, 38, 174, 68, 98, 162, 255, 36, 135, 82, 143, 150, 72, 138, 175, 115, 86, 124, 139, 111, 142, 239, 4, 136, 29, 104, 91, 25, 121, 41, 5, 129, 145, 206, 236, 179, 58, 128, 164, 83, 109, 220, 80, 24, 209, 107, 13, 131, 216, 2, 197, 27, 211, 101, 171, 215, 108, 28, 67, 32, 147, 216, 235, 205, 166, 253, 121, 143, 214, 170, 231, 202, 11, 237, 66, 140, 1, 158, 108, 41, 134, 54, 145, 7, 119, 242, 62, 65, 23, 223, 8, 136, 97, 206, 164, 247, 13, 252, 106, 250, 149, 180, 19, 239, 43, 246, 40, 92, 17, 204, 142, 25, 155, 238, 249, 193, 154, 21, 215, 162, 169, 182, 228, 167, 101, 232, 67, 135, 57, 188, 178, 189, 47, 225, 75, 160, 71, 168, 173, 102, 237, 64, 66, 143, 41, 238, 164, 216, 46, 57, 96, 234, 30, 51, 110, 14, 18, 116, 10, 106, 224, 39, 71, 109, 204, 99, 135, 61, 60, 68, 86, 141, 139, 250, 203, 188, 251, 136, 209, 133, 1, 228, 202, 87, 66, 11, 18, 122, 73, 221, 240, 34, 148, 195, 236, 30, 200, 199, 79, 110, 254, 71, 184, 93, 230, 134, 27, 193, 127, 224, 178, 44, 253, 191, 144, 15, 49, 20, 111, 170, 209, 179, 149, 172, 118, 233, 227, 245, 74, 154, 250, 63, 138, 32, 219, 114, 168, 183, 169, 29, 181, 165, 108, 92, 123, 255, 50, 94, 102, 22, 5, 220, 175, 89, 177, 201, 124, 28, 188, 10, 207, 75, 237, 213, 54, 252, 211, 60, 79, 132, 128, 200, 182, 173, 186, 196, 71, 49, 149, 51, 142, 120, 99, 82, 83, 91, 44, 204, 104, 217, 116, 158, 140, 78, 148, 121, 50, 161, 254, 85, 7, 226, 239, 63, 12, 195, 10, 98, 81, 199, 88, 187, 167, 94, 166, 225, 75, 210, 181, 165, 197, 228, 162, 6, 143, 206, 23, 251, 233, 170, 109, 219, 240, 113, 134, 56, 201, 77, 246, 8, 89, 159, 144, 122, 227, 76, 108, 252, 200, 248, 8, 179, 92, 73, 135, 231, 114, 222, 104, 125, 84, 243, 247, 199, 216, 102, 206, 221, 27, 147, 138, 254, 235, 198, 129, 90, 63, 68, 210, 191, 172, 96, 37, 71, 61, 130, 98, 185, 124, 100, 79, 161, 151, 67, 204, 25, 189, 43, 207, 246, 19, 110, 119, 137, 69, 34, 35, 91, 75, 20, 128, 52, 132, 188, 12, 77, 113, 211, 232, 54, 152, 208, 239, 168, 4, 104, 243, 188, 24, 250, 11, 135, 69, 191, 142, 70, 200, 238, 82, 236, 211, 254, 80, 234, 177, 163, 155, 6, 121, 113, 161, 33, 40, 149, 230, 133, 100, 10, 208, 176, 187, 117, 201, 68, 77, 144, 252, 221, 209, 173, 148, 132, 107, 112, 2, 153, 85, 44, 220, 165, 61, 78, 66, 218, 115, 159, 38, 97, 130, 14, 190, 240, 160, 52, 172, 226, 95, 178, 162, 182, 101, 81, 75, 106, 153, 132, 146, 166, 39, 157, 75, 243, 138, 126, 40, 206, 232, 50, 19, 80, 242, 251, 53, 209, 134, 170, 215, 17, 190, 113, 248, 212, 25, 152, 217, 99, 145, 63, 33, 204, 105, 148, 7, 82, 98, 56, 192, 177, 202, 199, 233, 89, 144, 142, 230, 237, 114, 187, 231, 197, 194, 168, 254, 154, 238, 155, 139, 123, 179, 188, 201, 222, 59, 151, 122, 120, 4, 193, 24, 48, 118, 93, 203, 143, 216, 226, 73, 35, 248, 88, 244, 206, 194, 221, 72, 78, 1, 21, 233, 139, 10, 46, 106, 208, 242, 120, 159, 24, 123, 128, 175, 200, 62, 99, 129, 70, 15, 32, 43, 187, 48, 77, 150, 180, 147, 66, 3, 210, 65, 2, 122, 58, 137, 103, 240, 27, 82, 141, 41, 191, 183, 202, 158, 142, 90, 243, 188, 4, 95, 220, 42, 45, 8, 84, 5, 93, 166, 39, 185, 118, 25, 75, 54, 9, 208, 71, 3, 189, 186, 63, 194, 107, 109, 33, 37, 22, 213, 111, 68, 222, 238, 112, 157, 201, 204, 139, 143, 104, 93, 203, 43, 171, 87, 30, 108, 102, 137, 236, 193, 141, 160, 19, 23, 142, 177, 72, 219, 205, 229, 21, 187, 115, 235, 224, 248, 220, 237, 44, 195, 179, 118, 47, 212, 155, 12, 207, 96, 65, 136, 36, 61, 18, 79, 10, 17, 56, 100, 129, 221, 150, 134, 174, 133, 218, 198, 163, 56, 31, 73, 212, 159, 47, 44, 155, 18, 129, 77, 213, 164, 8, 23, 196, 202, 36, 253, 223, 10, 88, 184, 149, 222, 42, 170, 144, 119, 9, 16, 189, 168, 116, 239, 50, 191, 46, 28, 158, 160, 37, 125, 43, 235, 240, 140, 86, 156, 38, 217, 27, 45, 179, 134, 228, 5, 124, 4, 54, 238, 121, 221, 89, 130, 131, 26, 252, 227, 67, 52, 246, 229, 87, 81, 53, 117, 140, 136, 102, 172, 10, 194, 122, 145, 160, 149, 253, 186, 19, 72, 30, 47, 95, 103, 143, 53])
    // // }

    // // fn input_image_34x34() -> Vec<u8> {
    // //     Vec::from([145, 88, 218, 40, 50, 208, 167, 237, 171, 227, 236, 6, 102, 27, 142, 90, 3, 251, 19, 118, 86, 20, 172, 242, 186, 200, 196, 107, 76, 32, 58, 191, 164, 12, 198, 202, 36, 221, 61, 181, 7, 82, 101, 134, 25, 87, 146, 104, 219, 79, 75, 250, 223, 151, 74, 188, 163, 1, 184, 225, 169, 11, 5, 243, 161, 33, 81, 123, 94, 226, 99, 72, 77, 209, 155, 92, 185, 136, 141, 112, 199, 170, 238, 96, 205, 175, 117, 10, 89, 149, 106, 240, 247, 60, 180, 66, 9, 197, 122, 174, 116, 168, 203, 46, 144, 228, 157, 183, 93, 143, 53, 189, 254, 120, 83, 18, 67, 195, 130, 133, 177, 70, 84, 124, 215, 98, 140, 97, 212, 248, 246, 64, 17, 253, 114, 71, 166, 206, 103, 22, 29, 115, 68, 78, 31, 4, 100, 213, 85, 16, 160, 49, 63, 21, 182, 37, 45, 244, 125, 39, 24, 165, 210, 108, 193, 13, 38, 48, 224, 241, 95, 158, 54, 73, 127, 153, 59, 139, 51, 159, 222, 128, 113, 231, 154, 55, 62, 207, 147, 192, 26, 110, 91, 137, 69, 135, 211, 152, 249, 132, 41, 8, 56, 220, 65, 15, 105, 179, 217, 216, 162, 47, 34, 111, 57, 178, 52, 232, 201, 30, 173, 43, 23, 176, 235, 234, 233, 190, 230, 119, 44, 2, 148, 229, 214, 156, 131, 14, 35, 109, 80, 42, 129, 187, 245, 28, 239, 255, 194, 121, 126, 138, 150, 252, 204, 19, 171, 177, 33, 152, 165, 216, 218, 56, 155, 156, 79, 117, 195, 15, 134, 168, 40, 204, 39, 139, 66, 70, 226, 74, 107, 88, 65, 191, 53, 80, 17, 99, 109, 224, 133, 94, 229, 202, 243, 81, 149, 82, 72, 247, 166, 138, 161, 214, 215, 132, 143, 201, 113, 77, 22, 115, 78, 96, 178, 199, 235, 162, 64, 29, 73, 23, 12, 123, 31, 212, 98, 220, 174, 194, 104, 151, 58, 157, 173, 237, 121, 37, 211, 8, 240, 63, 169, 10, 158, 95, 111, 245, 44, 208, 239, 43, 60, 112, 146, 250, 142, 253, 67, 198, 222, 110, 91, 47, 21, 241, 255, 124, 190, 69, 231, 41, 249, 181, 59, 118, 129, 14, 57, 89, 228, 16, 184, 187, 248, 206, 213, 84, 196, 114, 90, 251, 197, 160, 236, 189, 186, 13, 137, 54, 86, 230, 144, 207, 38, 51, 130, 18, 175, 120, 30, 127, 103, 167, 5, 3, 238, 159, 170, 135, 85, 163, 125, 1, 24, 102, 141, 188, 148, 20, 217, 147, 71, 7, 42, 101, 150, 75, 164, 154, 203, 128, 34, 153, 49, 192, 122, 83, 140, 92, 193, 87, 246, 179, 25, 108, 131, 100, 252, 227, 4, 68, 62, 242, 176, 36, 145, 225, 28, 50, 105, 244, 219, 48, 9, 209, 200, 2, 232, 45, 61, 76, 119, 254, 52, 136, 55, 97, 126, 221, 11, 172, 234, 180, 210, 27, 223, 6, 183, 93, 35, 116, 182, 185, 26, 32, 106, 205, 233, 46, 180, 125, 253, 252, 50, 23, 191, 245, 189, 3, 166, 159, 151, 78, 223, 90, 233, 218, 249, 44, 165, 122, 157, 21, 118, 54, 167, 160, 242, 123, 137, 22, 193, 254, 188, 228, 112, 147, 127, 39, 163, 199, 221, 237, 134, 225, 227, 149, 141, 20, 62, 100, 208, 183, 88, 26, 156, 65, 130, 43, 229, 119, 102, 8, 55, 96, 38, 251, 186, 174, 42, 66, 68, 231, 132, 98, 201, 210, 162, 71, 85, 255, 110, 176, 36, 250, 181, 135, 33, 170, 82, 17, 70, 143, 144, 178, 190, 244, 169, 232, 94, 76, 150, 79, 103, 72, 222, 187, 138, 235, 196, 175, 184, 203, 115, 37, 226, 86, 234, 97, 124, 48, 75, 139, 19, 30, 111, 11, 113, 142, 192, 202, 239, 7, 95, 4, 73, 114, 136, 16, 105, 29, 116, 61, 104, 46, 146, 91, 52, 155, 25, 246, 59, 121, 219, 204, 41, 168, 177, 5, 57, 158, 129, 207, 64, 145, 243, 194, 206, 56, 49, 236, 238, 241, 179, 99, 230, 58, 173, 224, 128, 10, 148, 164, 92, 195, 83, 248, 69, 109, 47, 87, 220, 18, 63, 80, 126, 185, 31, 12, 247, 117, 35, 84, 24, 140, 106, 209, 213, 9, 107, 182, 240, 13, 74, 200, 131, 89, 6, 216, 198, 214, 2, 45, 53, 197, 152, 161, 27, 205, 217, 211, 40, 154, 101, 120, 34, 171, 212, 81, 215, 133, 51, 108, 172, 77, 28, 15, 93, 67, 153, 1, 32, 14, 60, 147, 120, 233, 216, 70, 248, 235, 55, 56, 205, 139, 201, 166, 16, 96, 253, 251, 111, 121, 83, 171, 143, 199, 183, 214, 84, 26, 170, 88, 73, 231, 156, 177, 202, 222, 63, 11, 190, 58, 237, 59, 191, 66, 132, 203, 148, 131, 51, 133, 35, 184, 140, 72, 107, 1, 6, 163, 158, 208, 175, 108, 229, 85, 41, 102, 195, 134, 34, 95, 54, 210, 200, 145, 165, 91, 7, 126, 117, 119, 104, 141, 242, 46, 5, 62, 116, 45, 65, 224, 137, 23, 254, 174, 223, 128, 15, 8, 110, 196, 136, 82, 37, 97, 146, 221, 206, 255, 212, 164, 185, 33, 247, 151, 219, 13, 127, 9, 252, 194, 89, 106, 79, 197, 250, 27, 86, 149, 218, 240, 180, 209, 50, 19, 24, 52, 239, 31, 217, 43, 220, 159, 246, 28, 78, 40, 213, 207, 115, 124, 12, 243, 29, 114, 92, 69, 227, 17, 74, 38, 204, 44, 20, 142, 14, 10, 25, 48, 234, 155, 157, 241, 238, 30, 98, 249, 22, 100, 193, 53, 144, 154, 123, 60, 21, 153, 90, 215, 161, 49, 162, 244, 80, 169, 129, 4, 182, 152, 77, 228, 211, 122, 167, 181, 192, 101, 3, 125, 232, 39, 113, 67, 64, 42, 135, 18, 94, 57, 230, 118, 188, 150, 99, 178, 2, 87, 189, 105, 138, 47, 81, 226, 225, 198, 61, 75, 236, 68, 160, 36, 186, 71, 172, 109, 168, 112, 179, 173, 32, 245, 130, 187, 76, 103, 93, 176, 102, 131, 54, 237, 183, 53, 64, 156, 140, 66, 248, 148, 143, 113, 129, 41, 147, 80, 238, 132, 5, 164, 176, 13, 216, 150, 34, 46, 153, 212, 57, 127, 77, 96, 226, 45, 234, 172, 8, 30, 177, 145, 51, 100, 12, 110, 33, 144, 14, 26, 230, 18, 112, 25, 116, 187, 146, 10, 233, 75, 106, 98, 231, 224, 94, 92, 39, 159, 158, 71, 124, 88, 109, 173, 253, 204, 122, 229, 99, 44, 2, 135, 119, 221, 61, 32, 7, 60, 210, 189, 68, 72, 225, 86, 157, 117, 220, 15, 105, 180, 23, 121, 141, 52, 191, 139, 73, 178, 250, 218, 59, 203, 138, 142, 188, 137, 242, 251, 196, 199, 136, 128, 83, 209, 3, 222, 133, 11, 58, 1, 29, 179, 228, 4, 85, 202, 190, 187, 87, 141, 196, 66, 147, 39, 11, 215, 2, 18, 59, 238, 122, 25, 198, 73, 45, 100, 221, 203, 186, 240, 40, 180, 34, 173, 4, 148, 247, 67, 195, 166, 129, 236, 151, 58, 30, 146, 222, 200, 246, 128, 199, 106, 81, 79, 130, 152, 110, 162, 51, 254, 53, 14, 71, 60, 251, 184, 197, 88, 239, 95, 155, 82, 156, 161, 93, 99, 33, 230, 78, 56, 134, 234, 103, 27, 242, 249, 193, 121, 19, 127, 136, 41, 224, 42, 105, 178, 235, 107, 44, 223, 1, 253, 54, 248, 191, 228, 77, 144, 216, 13, 15, 112, 202, 49, 131, 3, 20, 133, 84, 111, 171, 140, 170, 194, 142, 209, 104, 83, 179, 70, 212, 149, 125, 68, 172, 153, 218, 118, 210, 8, 233, 231, 85, 227, 98, 101, 245, 167, 17, 74, 139, 217, 154, 164, 205, 250, 243, 48, 63, 252, 160, 138, 119, 214, 32, 96, 226, 219, 37, 244, 145, 38, 76, 47, 120, 26, 114, 206, 232, 168, 46, 69, 183, 143, 36, 169, 55, 117, 29, 204, 159, 181, 192, 126, 165, 24, 97, 108, 86, 6, 92, 16, 229, 123, 72, 113, 255, 31, 80, 50, 211, 90, 94, 174, 12, 102, 91, 43, 22, 158, 137, 5, 241, 115, 220, 135, 9, 175, 61, 57, 89, 64, 7, 177, 132, 116, 201, 157, 150, 124, 23, 163, 28, 189, 185, 188, 225, 21, 10, 182, 208, 207, 62, 109, 75, 52, 65, 237, 35, 176, 213, 255, 5, 54, 87, 123, 252, 238, 25, 211, 185, 223, 65, 177, 124, 208, 1, 190, 60, 184, 241, 79, 191, 221, 132, 192, 84, 128, 153, 247, 200, 188, 202, 182, 20, 86, 173, 42, 100, 186, 35, 248, 196, 107, 27, 71, 198, 55, 49, 64, 242, 149, 146, 244, 51, 92, 30, 142, 22, 108, 120, 69, 227, 99, 61, 220, 82, 249, 70, 83, 46, 154, 91, 133, 21, 44, 28, 137, 204, 171, 145, 104, 52, 175, 217, 105, 96, 116, 13, 114, 158, 97, 174, 140, 179, 31, 78, 189, 139, 148, 157, 40, 121, 106, 213, 50, 180, 237, 161, 250, 253, 254, 151, 66, 209, 127, 117, 36, 68, 236, 85, 141, 126, 7, 37, 3, 226, 125, 39, 239, 150, 207, 63, 129, 14, 12, 164, 112, 195, 155, 62, 10, 147, 169, 98, 163, 215, 81, 15, 45, 199, 243, 29, 88, 24, 176, 187, 152, 102, 167, 9, 131, 94, 160, 76, 166, 224, 235, 225, 4, 172, 75, 93, 212, 210, 2, 11, 181, 229, 74, 165, 53, 216, 197, 16, 58, 228, 231, 119, 162, 218, 101, 6, 95, 118, 143, 234, 183, 206, 232, 67, 23, 17, 193, 251, 33, 136, 233, 203, 115, 170, 48, 230, 109, 34, 72, 47, 80, 168, 18, 57, 41, 219, 130, 90, 240, 156, 222, 113, 111, 214, 134, 26, 38, 56, 138, 32, 201, 245, 59, 77, 43, 110, 246, 73, 19, 8, 194, 205, 89, 103, 135, 159, 178, 122, 144, 145, 241, 122, 66, 245, 227, 55, 146, 76, 103, 193, 108, 5, 159, 252, 11, 134, 200, 112, 153, 248, 167, 201, 8, 24, 136, 179, 3, 234, 92, 249, 158, 73, 21, 220, 135, 229, 51, 231, 237, 155, 114, 9, 116, 222, 93, 150, 104, 174, 80, 125, 56, 16, 84, 233, 183, 243, 140, 29, 247, 86, 42, 154, 47, 144, 26, 156, 181, 199, 244, 15, 216, 250, 202, 102, 53, 228, 206, 70, 1, 221, 238, 46, 27, 120, 72, 147, 253, 107, 138, 217, 59, 254, 165, 49, 235, 213, 32, 198, 31, 17, 129, 83, 194, 90, 197, 105, 63, 169, 226, 68, 164, 64, 210, 106, 74, 191, 36, 162, 172, 219, 99, 96, 143, 23, 37, 218, 180, 71, 175, 170, 61, 176, 224, 130, 7, 14, 98, 28, 60, 185, 149, 87, 124, 109, 30, 100, 88, 94, 79, 163, 95, 161, 121, 240, 151, 33, 195, 67, 10, 131, 204, 139, 123, 209, 6, 40, 184, 118, 65, 25, 255, 230, 189, 187, 18, 43, 160, 251, 207, 215, 127, 246, 85, 81, 19, 115, 97, 110, 148, 50, 119, 4, 186, 137, 89, 212, 69, 177, 166, 34, 214, 62, 35, 133, 101, 91, 111, 39, 75, 190, 236, 20, 41, 157, 128, 171, 223, 52, 38, 203, 132, 242, 205, 188, 117, 141, 12, 225, 78, 77, 2, 57, 113, 178, 58, 211, 13, 44, 232, 82, 173, 54, 48, 196, 152, 45, 192, 208, 142, 126, 239, 182, 22, 168, 143, 248, 4, 212, 35, 104, 31, 224, 243, 91, 217, 84, 179, 193, 175, 242, 237, 188, 158, 16, 24, 30, 139, 250, 39, 247, 11, 7, 152, 135, 122, 210, 69, 90, 88, 191, 129, 26, 142, 140, 50, 70, 253, 146, 200, 245, 138, 238, 34, 29, 82, 168, 73, 236, 65, 126, 211, 145, 120, 254, 54, 205, 80, 227, 244, 234, 58, 56, 177, 171, 251, 163, 46, 207, 155, 36, 108, 6, 199, 41, 121, 109, 203, 113, 239, 13, 161, 213, 22, 33, 94, 103, 40, 60, 206, 149, 141, 216, 230, 23, 166, 133, 53, 170, 100, 111, 12, 10, 105, 47, 208, 89, 96, 110, 67, 185, 150, 255, 164, 176, 99, 154, 187, 181, 1, 117, 222, 62, 201, 197, 8, 68, 137, 25, 77, 55, 147, 144, 21, 98, 252, 232, 151, 221, 27, 186, 209, 20, 32, 173, 223, 127, 148, 57, 167, 132, 228, 195, 107, 124, 231, 112, 128, 63, 2, 92, 19, 153, 15, 125, 85, 118, 204, 44, 45, 194, 220, 235, 184, 165, 74, 189, 61, 123, 233, 78, 225, 215, 66, 106, 169, 218, 119, 17, 115, 116, 198, 159, 76, 18, 38, 83, 102, 97, 229, 136, 130, 49, 37, 14, 219, 5, 190, 114, 183, 71, 249, 157, 3, 241, 9, 240, 64, 86, 160, 156, 48, 52, 79, 246, 172, 42, 72, 226, 202, 59, 95, 214, 43, 178, 192, 93, 162, 131, 180, 182, 51, 196, 101, 174, 134, 81, 28, 87, 75, 244, 129, 106, 125, 133, 153, 35, 22, 132, 68, 176, 146, 29, 216, 166, 94, 66, 39, 124, 234, 157, 178, 57, 75, 228, 247, 243, 96, 44, 138, 213, 159, 126, 183, 227, 40, 186, 249, 206, 51, 225, 232, 16, 175, 50, 32, 78, 19, 211, 224, 80, 115, 69, 242, 100, 70, 251, 28, 90, 53, 226, 88, 38, 117, 141, 149, 52, 5, 209, 200, 23, 134, 127, 121, 170, 2, 150, 215, 160, 104, 17, 196, 26, 190, 205, 54, 113, 55, 236, 248, 3, 83, 212, 43, 30, 25, 171, 60, 152, 191, 103, 217, 109, 240, 99, 173, 181, 145, 65, 223, 63, 18, 218, 33, 12, 119, 204, 111, 85, 105, 214, 180, 148, 64, 235, 7, 21, 72, 82, 58, 46, 98, 172, 250, 56, 182, 77, 192, 67, 42, 177, 162, 184, 202, 255, 49, 199, 221, 161, 233, 62, 1, 89, 229, 41, 144, 208, 76, 142, 61, 167, 230, 135, 9, 95, 137, 131, 11, 79, 185, 237, 45, 116, 114, 158, 27, 187, 165, 34, 231, 169, 252, 197, 140, 210, 194, 147, 13, 168, 36, 101, 254, 110, 8, 154, 91, 164, 238, 156, 47, 155, 71, 92, 139, 107, 74, 123, 136, 239, 179, 102, 207, 188, 20, 195, 201, 163, 219, 222, 86, 97, 59, 112, 253, 151, 14, 108, 122, 81, 245, 120, 37, 87, 4, 130, 174, 193, 241, 220, 24, 10, 84, 48, 246, 6, 118, 189, 128, 93, 15, 31, 203, 198, 73, 143, 143, 190, 216, 182, 28, 226, 204, 51, 73, 111, 168, 37, 30, 67, 76, 215, 114, 35, 134, 146, 248, 193, 38, 88, 108, 178, 244, 102, 238, 206, 253, 98, 194, 197, 57, 221, 189, 22, 72, 64, 167, 78, 153, 121, 1, 203, 156, 21, 247, 53, 233, 86, 165, 139, 109, 7, 10, 173, 40, 46, 125, 112, 106, 47, 116, 208, 223, 110, 242, 127, 50, 120, 177, 152, 159, 113, 130, 24, 192, 17, 123, 23, 80, 128, 252, 49, 175, 184, 232, 200, 71, 79, 62, 228, 63, 99, 246, 234, 129, 219, 140, 70, 104, 164, 15, 249, 211, 32, 94, 44, 43, 224, 169, 171, 155, 209, 126, 237, 172, 187, 154, 163, 48, 83, 201, 77, 136, 135, 150, 227, 26, 180, 179, 69, 147, 231, 115, 66, 132, 85, 3, 81, 74, 210, 239, 229, 65, 199, 101, 2, 96, 9, 122, 212, 161, 58, 218, 92, 137, 254, 131, 103, 198, 60, 240, 87, 251, 27, 16, 29, 82, 145, 107, 141, 59, 6, 41, 255, 222, 191, 55, 12, 183, 105, 97, 202, 31, 20, 158, 91, 18, 142, 186, 241, 90, 149, 207, 243, 61, 89, 188, 170, 236, 4, 235, 157, 95, 181, 174, 220, 52, 217, 42, 14, 162, 34, 196, 36, 195, 225, 33, 45, 11, 13, 8, 151, 214, 84, 124, 117, 5, 100, 160, 93, 56, 245, 166, 133, 68, 39, 230, 148, 185, 19, 213, 118, 250, 138, 25, 144, 205, 75, 176, 119, 54, 27, 119, 9, 120, 246, 208, 53, 131, 71, 83, 178, 3, 250, 169, 189, 247, 126, 186, 75, 64, 63, 223, 127, 194, 57, 124, 107, 241, 181, 109, 97, 51, 33, 86, 1, 37, 185, 103, 22, 16, 80, 213, 88, 211, 111, 69, 82, 68, 99, 114, 222, 85, 101, 238, 147, 209, 112, 154, 145, 157, 146, 242, 67, 81, 84, 215, 6, 4, 201, 244, 32, 204, 214, 190, 139, 98, 153, 143, 46, 8, 104, 60, 199, 93, 122, 38, 203, 73, 164, 43, 149, 183, 171, 184, 254, 87, 125, 159, 30, 192, 11, 108, 105, 95, 102, 117, 40, 137, 7, 232, 236, 110, 113, 193, 156, 41, 141, 245, 76, 160, 128, 166, 19, 28, 135, 23, 25, 140, 142, 252, 55, 177, 231, 206, 72, 180, 58, 219, 255, 196, 205, 168, 74, 229, 188, 132, 21, 172, 148, 187, 91, 94, 115, 39, 78, 235, 167, 225, 224, 13, 5, 248, 42, 49, 52, 59, 144, 227, 162, 54, 220, 216, 163, 237, 15, 48, 44, 176, 2, 195, 198, 26, 179, 158, 228, 118, 197, 121, 47, 253, 191, 212, 34, 210, 155, 233, 66, 12, 89, 217, 207, 24, 90, 96, 31, 165, 65, 170, 251, 136, 226, 152, 36, 173, 240, 61, 62, 14, 18, 138, 92, 79, 234, 20, 10, 130, 106, 17, 202, 116, 56, 50, 239, 100, 29, 249, 129, 161, 230, 221, 123, 70, 150, 45, 175, 134, 77, 243, 174, 200, 182, 133, 151, 35, 218, 61, 68, 198, 210, 248, 163, 181, 40, 56, 32, 49, 64, 75, 19, 133, 29, 139, 31, 126, 66, 73, 137, 206, 212, 175, 132, 159, 138, 82, 47, 231, 153, 44, 197, 200, 155, 204, 244, 18, 208, 33, 129, 30, 74, 77, 176, 72, 213, 84, 11, 164, 60, 35, 8, 99, 70, 23, 94, 135, 196, 180, 39, 202, 17, 58, 36, 214, 15, 253, 65, 224, 223, 51, 105, 10, 128, 174, 88, 118, 112, 184, 13, 100, 149, 166, 90, 222, 162, 6, 42, 216, 106, 170, 146, 63, 144, 59, 83, 119, 22, 103, 9, 92, 76, 16, 173, 98, 189, 236, 209, 168, 3, 78, 114, 185, 161, 177, 151, 192, 116, 157, 218, 239, 190, 193, 50, 24, 14, 191, 110, 142, 46, 25, 203, 28, 215, 172, 158, 188, 178, 160, 120, 255, 37, 80, 169, 125, 242, 241, 43, 95, 219, 235, 183, 225, 240, 233, 230, 140, 141, 115, 86, 123, 145, 156, 143, 111, 38, 102, 108, 217, 93, 71, 27, 171, 55, 45, 136, 201, 179, 122, 148, 134, 207, 150, 228, 247, 245, 5, 165, 12, 124, 62, 194, 4, 195, 57, 54, 205, 7, 238, 199, 251, 121, 21, 91, 221, 182, 1, 89, 48, 152, 130, 186, 113, 97, 147, 249, 232, 2, 254, 131, 85, 41, 26, 107, 154, 252, 226, 250, 227, 109, 69, 67, 167, 220, 52, 79, 243, 246, 187, 101, 229, 34, 20, 87, 104, 234, 81, 127, 237, 53, 211, 96, 117, 110, 151, 140, 38, 132, 136, 167, 93, 102, 242, 225, 172, 179, 244, 10, 69, 33, 194, 176, 168, 122, 238, 26, 145, 22, 237, 160, 111, 36, 149, 49, 208, 253, 233, 82, 186, 67, 196, 19, 64, 17, 72, 152, 189, 30, 62, 220, 47, 92, 24, 95, 211, 20, 103, 28, 127, 143, 121, 44, 53, 119, 216, 90, 27, 215, 5, 56, 83, 254, 212, 204, 58, 131, 202, 80, 97, 181, 241, 191, 164, 39, 230, 120, 86, 247, 228, 229, 187, 235, 104, 182, 192, 74, 173, 43, 76, 77, 118, 6, 201, 169, 101, 219, 133, 198, 213, 207, 255, 1, 155, 109, 177, 84, 190, 52, 32, 107, 57, 79, 175, 61, 116, 223, 105, 159, 171, 206, 15, 23, 197, 12, 170, 11, 165, 94, 66, 37, 184, 112, 245, 153, 236, 13, 139, 217, 125, 163, 209, 157, 96, 134, 124, 154, 100, 150, 199, 70, 252, 232, 46, 248, 135, 226, 42, 158, 106, 40, 9, 141, 147, 224, 249, 221, 195, 68, 14, 146, 222, 166, 240, 31, 200, 239, 98, 78, 126, 35, 81, 51, 59, 144, 218, 183, 123, 75, 108, 73, 41, 87, 193, 60, 250, 54, 4, 55, 185, 99, 8, 246, 148, 21, 85, 91, 234, 156, 128, 251, 162, 25, 16, 203, 45, 138, 210, 227, 89, 113, 2, 88, 243, 214, 117, 174, 205, 231, 3, 180, 7, 48, 142, 161, 50, 188, 130, 29, 34, 137, 114, 115, 71, 63, 129, 18, 65, 178, 202, 76, 168, 5, 84, 102, 77, 81, 234, 27, 105, 68, 12, 124, 155, 253, 50])
    // // }

    // fn split_and_shifted_input_image_34x34_mtn_cb() -> Vec<Vec<i8>> {
    //     Vec::from([
    //         Vec::from([
    //             57, 57, 59, 63, 66, 70, 67, 62, 65, 63, 61, 63, 66, 67, 71, 76, 62, 64, 64, 66, 67,
    //             65, 65, 68, 56, 60, 62, 65, 67, 67, 67, 63, 74, 77, 77, 76, 73, 70, 71, 68, 76, 78,
    //             75, 74, 70, 70, 74, 68, 72, 74, 73, 67, 68, 68, 68, 47, 66, 70, 69, 65, 63, 61, 33,
    //             -17,
    //         ]),
    //         Vec::from([
    //             61, 64, 57, 52, 52, 57, 57, 58, 79, 80, 71, 60, 58, 56, 73, 92, 72, 78, 76, 74, 79,
    //             84, 97, 102, 68, 69, 69, 67, 84, 80, 43, 32, 66, 63, 20, -16, 37, 7, -6, 6, 68, 7,
    //             -18, -11, -15, 5, 0, 30, -6, -16, -17, -16, -15, -5, 38, 0, -22, -15, -6, 0, -11,
    //             2, -5, 39,
    //         ]),
    //         Vec::from([
    //             63, 66, 63, 59, 56, 12, -11, -13, 64, 63, 60, 41, -8, 4, -6, -9, 64, 58, 25, -6,
    //             -11, -10, -5, -8, 50, 11, 0, 5, 0, -7, -12, -6, 3, 1, 7, 3, 1, -4, -2, 12, 24, 15,
    //             14, 11, 7, 4, 41, 47, 90, 94, 72, 50, 51, 42, 45, 32, 82, 107, 96, 108, 91, 92, 94,
    //             51,
    //         ]),
    //         Vec::from([
    //             -19, -7, -10, 15, 12, -3, 34, 9, -10, -1, -3, 5, 15, 2, -8, 4, -4, -11, 13, 24, 16,
    //             8, -2, -17, 9, 10, 21, 21, 21, 14, 6, 26, 29, 45, 27, 29, 21, 22, 37, 77, 78, 44,
    //             69, 29, 10, 6, 2, 50, 16, 11, 11, 43, 10, 1, -1, -10, 67, 68, 67, 52, 27, 31, 31,
    //             18,
    //         ]),
    //     ])
    // }

    // fn split_input_image_34x34_mtn_cb() -> Vec<Vec<u8>> {
    //     Vec::from([
    //         Vec::from([
    //             185, 185, 187, 191, 194, 198, 195, 190, 193, 191, 189, 191, 194, 195, 199, 204,
    //             190, 192, 192, 194, 195, 193, 193, 196, 184, 188, 190, 193, 195, 195, 195, 191,
    //             202, 205, 205, 204, 201, 198, 199, 196, 204, 206, 203, 202, 198, 198, 202, 196,
    //             200, 202, 201, 195, 196, 196, 196, 175, 194, 198, 197, 193, 191, 189, 161, 111,
    //         ]),
    //         Vec::from([
    //             189, 192, 185, 180, 180, 185, 185, 186, 207, 208, 199, 188, 186, 184, 201, 220,
    //             200, 206, 204, 202, 207, 212, 225, 230, 196, 197, 197, 195, 212, 208, 171, 160,
    //             194, 191, 148, 112, 165, 135, 122, 134, 196, 135, 110, 117, 113, 133, 128, 158,
    //             122, 112, 111, 112, 113, 123, 166, 128, 106, 113, 122, 128, 117, 130, 123, 167,
    //         ]),
    //         Vec::from([
    //             191, 194, 191, 187, 184, 140, 117, 115, 192, 191, 188, 169, 120, 132, 122, 119,
    //             192, 186, 153, 122, 117, 118, 123, 120, 178, 139, 128, 133, 128, 121, 116, 122,
    //             131, 129, 135, 131, 129, 124, 126, 140, 152, 143, 142, 139, 135, 132, 169, 175,
    //             218, 222, 200, 178, 179, 170, 173, 160, 210, 235, 224, 236, 219, 220, 222, 179,
    //         ]),
    //         Vec::from([
    //             109, 121, 118, 143, 140, 125, 162, 137, 118, 127, 125, 133, 143, 130, 120, 132,
    //             124, 117, 141, 152, 144, 136, 126, 111, 137, 138, 149, 149, 149, 142, 134, 154,
    //             157, 173, 155, 157, 149, 150, 165, 205, 206, 172, 197, 157, 138, 134, 130, 178,
    //             144, 139, 139, 171, 138, 129, 127, 118, 195, 196, 195, 180, 155, 159, 159, 146,
    //         ]),
    //     ])
    // }

    // fn input_image_34x34_mtn_cr() -> Vec<u8> {
    //     Vec::from([
    //         196, 196, 198, 202, 204, 208, 205, 201, 200, 203, 196, 192, 193, 199, 199, 198, 204,
    //         201, 199, 201, 204, 206, 210, 214, 217, 219, 210, 200, 198, 196, 208, 224, 201, 202,
    //         201, 203, 205, 203, 203, 206, 210, 217, 215, 214, 218, 223, 233, 236, 195, 197, 199,
    //         202, 204, 205, 206, 202, 206, 206, 207, 206, 222, 218, 181, 169, 213, 215, 214, 213,
    //         210, 208, 208, 205, 204, 200, 156, 121, 175, 145, 133, 146, 215, 216, 212, 212, 207,
    //         206, 211, 204, 205, 144, 118, 125, 122, 142, 138, 171, 211, 213, 211, 205, 205, 205,
    //         205, 183, 130, 120, 120, 120, 121, 131, 175, 137, 205, 208, 207, 203, 202, 200, 171,
    //         120, 115, 122, 130, 136, 126, 138, 130, 174, 204, 206, 202, 198, 196, 151, 126, 123,
    //         118, 130, 126, 150, 147, 132, 168, 142, 210, 208, 203, 183, 133, 145, 134, 130, 127,
    //         133, 130, 138, 148, 136, 127, 138, 208, 201, 168, 135, 130, 129, 133, 129, 131, 123,
    //         145, 155, 149, 142, 133, 118, 190, 150, 138, 142, 137, 131, 123, 128, 142, 142, 151,
    //         151, 153, 147, 139, 159, 138, 136, 141, 137, 134, 130, 130, 142, 160, 176, 157, 160,
    //         152, 153, 168, 208, 155, 146, 145, 141, 138, 135, 169, 173, 206, 174, 199, 159, 140,
    //         137, 133, 180, 217, 221, 200, 177, 179, 170, 169, 155, 141, 138, 141, 174, 141, 132,
    //         129, 118, 205, 232, 221, 233, 216, 217, 216, 171, 189, 193, 194, 182, 157, 159, 158,
    //         144,
    //     ])
    // }

    // fn input_image_34x34_mtn_cb() -> Vec<u8> {
    //     Vec::from([
    //         185, 185, 187, 191, 194, 198, 195, 190, 189, 192, 185, 180, 180, 185, 185, 186, 193,
    //         191, 189, 191, 194, 195, 199, 204, 207, 208, 199, 188, 186, 184, 201, 220, 190, 192,
    //         192, 194, 195, 193, 193, 196, 200, 206, 204, 202, 207, 212, 225, 230, 184, 188, 190,
    //         193, 195, 195, 195, 191, 196, 197, 197, 195, 212, 208, 171, 160, 202, 205, 205, 204,
    //         201, 198, 199, 196, 194, 191, 148, 112, 165, 135, 122, 134, 204, 206, 203, 202, 198,
    //         198, 202, 196, 196, 135, 110, 117, 113, 133, 128, 158, 200, 202, 201, 195, 196, 196,
    //         196, 175, 122, 112, 111, 112, 113, 123, 166, 128, 194, 198, 197, 193, 191, 189, 161,
    //         111, 106, 113, 122, 128, 117, 130, 123, 167, 191, 194, 191, 187, 184, 140, 117, 115,
    //         109, 121, 118, 143, 140, 125, 162, 137, 192, 191, 188, 169, 120, 132, 122, 119, 118,
    //         127, 125, 133, 143, 130, 120, 132, 192, 186, 153, 122, 117, 118, 123, 120, 124, 117,
    //         141, 152, 144, 136, 126, 111, 178, 139, 128, 133, 128, 121, 116, 122, 137, 138, 149,
    //         149, 149, 142, 134, 154, 131, 129, 135, 131, 129, 124, 126, 140, 157, 173, 155, 157,
    //         149, 150, 165, 205, 152, 143, 142, 139, 135, 132, 169, 175, 206, 172, 197, 157, 138,
    //         134, 130, 178, 218, 222, 200, 178, 179, 170, 173, 160, 144, 139, 139, 171, 138, 129,
    //         127, 118, 210, 235, 224, 236, 219, 220, 222, 179, 195, 196, 195, 180, 155, 159, 159,
    //         146,
    //     ])
    // }

    // fn input_image_34x34_mtn_y() -> Vec<u8> {
    //     Vec::from([
    //         156, 155, 156, 157, 158, 159, 162, 163, 164, 166, 168, 168, 165, 156, 154, 152, 150,
    //         154, 155, 153, 149, 152, 152, 153, 154, 156, 154, 152, 151, 144, 147, 151, 155, 154,
    //         154, 155, 157, 157, 159, 161, 162, 165, 169, 170, 168, 167, 165, 162, 162, 167, 171,
    //         169, 165, 158, 154, 150, 150, 155, 162, 172, 176, 173, 175, 176, 163, 161, 160, 158,
    //         157, 157, 159, 161, 164, 166, 168, 169, 169, 171, 171, 168, 179, 175, 171, 187, 175,
    //         156, 155, 154, 153, 155, 155, 158, 177, 194, 194, 206, 168, 166, 165, 163, 162, 160,
    //         162, 164, 165, 166, 167, 167, 166, 168, 172, 173, 177, 174, 174, 187, 181, 175, 171,
    //         170, 168, 176, 173, 161, 163, 181, 196, 198, 164, 163, 163, 163, 162, 161, 162, 165,
    //         166, 165, 164, 165, 164, 162, 167, 172, 172, 173, 178, 183, 181, 178, 184, 188, 184,
    //         186, 191, 193, 201, 205, 207, 212, 160, 160, 161, 161, 161, 159, 160, 164, 165, 165,
    //         164, 164, 163, 162, 161, 164, 165, 173, 178, 172, 173, 179, 170, 176, 184, 189, 189,
    //         193, 209, 207, 213, 207, 155, 155, 156, 154, 154, 157, 158, 160, 162, 164, 164, 165,
    //         165, 166, 162, 162, 165, 172, 173, 167, 173, 177, 180, 186, 181, 187, 200, 196, 186,
    //         185, 184, 163, 156, 157, 159, 158, 159, 159, 161, 162, 163, 166, 166, 167, 167, 167,
    //         165, 164, 169, 170, 169, 174, 175, 171, 159, 171, 193, 208, 201, 163, 131, 112, 101,
    //         119, 169, 171, 173, 173, 173, 168, 168, 168, 167, 166, 166, 168, 168, 169, 171, 165,
    //         171, 170, 169, 173, 157, 142, 93, 109, 175, 198, 154, 98, 80, 105, 103, 107, 176, 177,
    //         177, 174, 173, 176, 175, 174, 171, 169, 167, 168, 170, 177, 179, 168, 171, 170, 168,
    //         163, 126, 84, 78, 94, 108, 111, 111, 109, 111, 125, 148, 108, 175, 176, 178, 174, 172,
    //         172, 173, 170, 169, 171, 167, 167, 176, 170, 170, 183, 177, 176, 164, 113, 81, 101,
    //         103, 98, 99, 95, 134, 114, 109, 96, 171, 156, 173, 173, 175, 172, 170, 168, 171, 170,
    //         166, 168, 171, 172, 175, 185, 169, 176, 188, 162, 107, 81, 97, 90, 97, 104, 106, 96,
    //         107, 123, 148, 95, 117, 124, 171, 172, 173, 173, 171, 172, 168, 168, 169, 169, 173,
    //         171, 168, 180, 184, 166, 137, 107, 90, 94, 96, 91, 88, 106, 101, 101, 107, 108, 171,
    //         147, 92, 97, 169, 170, 172, 172, 171, 172, 165, 166, 171, 166, 167, 171, 174, 177, 158,
    //         122, 91, 84, 95, 105, 103, 98, 96, 102, 102, 97, 122, 106, 127, 163, 140, 116, 164,
    //         165, 168, 169, 169, 169, 165, 168, 170, 164, 167, 173, 175, 156, 111, 89, 88, 97, 99,
    //         96, 112, 103, 119, 105, 115, 95, 115, 135, 86, 115, 187, 154, 163, 164, 167, 168, 168,
    //         167, 167, 167, 168, 173, 176, 159, 136, 111, 93, 95, 89, 94, 105, 100, 113, 108, 121,
    //         122, 114, 104, 101, 129, 129, 112, 141, 128, 161, 163, 166, 168, 167, 166, 163, 163,
    //         167, 172, 156, 124, 101, 97, 97, 101, 95, 95, 105, 108, 116, 108, 104, 154, 130, 114,
    //         113, 108, 154, 167, 115, 114, 159, 161, 164, 165, 165, 159, 161, 164, 168, 152, 116,
    //         96, 107, 105, 109, 104, 97, 103, 109, 116, 107, 97, 92, 182, 169, 110, 133, 113, 109,
    //         170, 136, 127, 166, 163, 160, 165, 165, 164, 163, 155, 141, 96, 97, 122, 119, 112, 115,
    //         104, 100, 116, 121, 115, 108, 99, 109, 137, 139, 136, 122, 117, 108, 113, 125, 124,
    //         165, 158, 162, 170, 172, 162, 160, 129, 101, 83, 118, 139, 110, 110, 108, 116, 106,
    //         116, 124, 110, 124, 133, 124, 128, 134, 130, 127, 119, 111, 111, 120, 119, 166, 165,
    //         171, 171, 163, 140, 120, 94, 101, 96, 106, 121, 119, 118, 112, 115, 105, 108, 106, 95,
    //         118, 138, 140, 143, 147, 121, 121, 121, 121, 114, 103, 106, 165, 169, 164, 146, 127,
    //         107, 108, 102, 116, 109, 102, 108, 112, 109, 112, 105, 119, 128, 116, 118, 129, 143,
    //         145, 142, 146, 129, 137, 132, 116, 117, 102, 100, 169, 162, 142, 119, 106, 112, 126,
    //         119, 122, 115, 115, 113, 103, 128, 123, 107, 116, 123, 119, 134, 143, 143, 145, 143,
    //         146, 132, 131, 137, 129, 118, 123, 114, 163, 142, 119, 110, 114, 123, 128, 117, 125,
    //         120, 118, 118, 111, 96, 100, 124, 138, 138, 137, 127, 121, 152, 141, 128, 141, 142,
    //         132, 133, 129, 126, 179, 166, 132, 117, 110, 115, 127, 117, 128, 122, 126, 116, 119,
    //         126, 118, 111, 117, 143, 152, 145, 150, 145, 124, 144, 154, 139, 149, 138, 131, 135,
    //         145, 143, 173, 227, 109, 111, 121, 129, 135, 128, 129, 123, 136, 122, 115, 127, 130,
    //         126, 132, 136, 150, 151, 170, 197, 165, 155, 170, 135, 142, 138, 154, 150, 149, 194,
    //         194, 194, 146, 122, 125, 143, 135, 128, 136, 127, 135, 139, 128, 131, 134, 162, 159,
    //         159, 202, 209, 193, 195, 208, 172, 150, 153, 124, 119, 146, 130, 104, 155, 200, 181,
    //         169, 142, 133, 147, 153, 133, 141, 139, 130, 133, 135, 135, 144, 222, 201, 147, 190,
    //         189, 143, 127, 182, 195, 158, 137, 146, 132, 106, 125, 122, 110, 134, 167, 183, 190,
    //         202, 188, 168, 162, 154, 161, 165, 173, 169, 154, 164, 160, 145, 146, 145, 124, 113,
    //         149, 131, 180, 190, 130, 143, 139, 132, 130, 131, 120, 108, 106, 240, 243, 248, 235,
    //         233, 229, 200, 192, 192, 185, 181, 176, 175, 179, 163, 162, 157, 126, 132, 139, 111,
    //         114, 173, 173, 125, 127, 117, 120, 122, 116, 127, 109, 211, 238, 244, 249, 254, 248,
    //         240, 238, 223, 190, 195, 220, 208, 197, 169, 179, 198, 193, 178, 155, 162, 148, 144,
    //         153, 126, 127, 131, 137, 130, 122, 136, 130, 180, 200, 217, 224, 191, 196, 217, 243,
    //         246, 211, 210, 248, 246, 220, 167, 174, 177, 187, 205, 222, 225, 224, 213, 191, 188,
    //         159, 174, 170, 176, 185, 161, 135,
    //     ])
    // }

    // fn input_image_34x34_lines() -> Vec<u8> {
    //     Vec::from([
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255,
    //         255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 255, 255, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 255, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 255, 255, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
    //         255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0,
    //         0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0,
    //         255, 255, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0,
    //         0, 255, 255, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 255, 255, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0,
    //         0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 255, 255, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 255,
    //         255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0,
    //         0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 255, 255, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 255, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 255, 255, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 255, 255,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255,
    //         255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255,
    //         255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 255, 255, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 255, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 255, 255, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
    //         255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0,
    //         0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0,
    //         255, 255, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0,
    //         0, 255, 255, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 255, 255, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0,
    //         0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 255, 255, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 255,
    //         255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0,
    //         0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 255, 255, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 255, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 255, 255, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 255, 255,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255,
    //         255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255,
    //         255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 255, 255, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 255, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 255, 255, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
    //         255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0,
    //         0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0,
    //         255, 255, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0,
    //         0, 255, 255, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 255, 255, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0,
    //         0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 255, 255, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 255,
    //         255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0,
    //         0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 255, 255, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 255, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 255, 255, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 255, 255,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255,
    //         255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255,
    //         255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 255, 255, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 255, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 255, 255, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
    //         255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0,
    //         0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0,
    //         255, 255, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0,
    //         0, 255, 255, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 255, 255, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0,
    //         0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 255, 255, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 255,
    //         255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0,
    //         0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 255, 255, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 255, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 255, 255, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 255, 255,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255,
    //         255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255,
    //         255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 255, 255, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 255, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 255, 255, 255, 0, 0, 255, 0, 0,
    //         255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
    //         255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     ])
    // }

    // fn input_image_34x34_mtn() -> Vec<u8> {
    //     Vec::from([
    //         156, 186, 197, 155, 185, 196, 156, 186, 197, 157, 187, 198, 158, 188, 199, 159, 189,
    //         200, 162, 192, 203, 163, 193, 204, 164, 194, 205, 166, 196, 206, 168, 198, 208, 168,
    //         198, 208, 165, 195, 205, 156, 188, 199, 154, 186, 197, 152, 184, 195, 150, 182, 193,
    //         154, 184, 195, 155, 185, 196, 153, 183, 194, 149, 179, 190, 152, 180, 192, 152, 180,
    //         192, 153, 181, 193, 154, 182, 194, 156, 182, 197, 154, 180, 195, 152, 178, 193, 151,
    //         177, 192, 144, 170, 183, 147, 173, 186, 151, 174, 188, 149, 175, 188, 149, 175, 188,
    //         155, 185, 196, 154, 184, 195, 154, 184, 195, 155, 185, 196, 157, 187, 198, 157, 187,
    //         198, 159, 189, 200, 161, 191, 202, 162, 192, 203, 165, 195, 205, 169, 199, 209, 170,
    //         200, 210, 168, 198, 208, 167, 199, 210, 165, 197, 208, 162, 194, 205, 162, 194, 205,
    //         167, 197, 208, 171, 201, 212, 169, 199, 210, 165, 195, 206, 158, 186, 198, 154, 182,
    //         194, 150, 178, 190, 150, 176, 189, 155, 181, 194, 162, 188, 201, 172, 195, 209, 176,
    //         199, 213, 173, 197, 209, 175, 199, 209, 176, 198, 211, 171, 195, 207, 165, 189, 201,
    //         163, 191, 202, 161, 191, 201, 160, 190, 200, 158, 188, 198, 157, 187, 197, 157, 187,
    //         197, 159, 189, 199, 161, 191, 201, 164, 194, 204, 166, 194, 205, 168, 196, 207, 169,
    //         197, 208, 169, 199, 209, 171, 203, 214, 171, 205, 215, 168, 200, 211, 179, 211, 222,
    //         175, 205, 215, 171, 201, 211, 187, 215, 226, 175, 203, 214, 156, 183, 194, 155, 182,
    //         193, 154, 181, 192, 153, 177, 189, 155, 179, 191, 155, 179, 191, 158, 180, 193, 177,
    //         199, 210, 194, 217, 223, 194, 218, 222, 206, 226, 233, 187, 210, 216, 163, 186, 192,
    //         168, 196, 207, 166, 196, 206, 165, 195, 205, 163, 193, 203, 162, 192, 202, 160, 190,
    //         200, 162, 192, 202, 164, 194, 204, 165, 195, 205, 166, 194, 205, 167, 195, 206, 167,
    //         195, 206, 166, 196, 206, 168, 200, 211, 172, 206, 216, 173, 205, 216, 177, 209, 220,
    //         174, 204, 214, 174, 204, 214, 187, 215, 226, 181, 209, 220, 175, 202, 213, 171, 198,
    //         209, 170, 194, 206, 168, 192, 204, 176, 198, 209, 173, 195, 206, 161, 183, 194, 163,
    //         186, 194, 181, 202, 207, 196, 217, 220, 198, 219, 222, 189, 210, 213, 180, 201, 206,
    //         164, 192, 203, 163, 193, 203, 163, 193, 203, 163, 193, 203, 162, 192, 202, 161, 194,
    //         203, 162, 195, 204, 165, 195, 205, 166, 196, 206, 165, 195, 205, 164, 194, 204, 165,
    //         193, 204, 164, 194, 204, 162, 195, 204, 167, 200, 209, 172, 202, 212, 172, 202, 212,
    //         173, 203, 213, 178, 208, 218, 183, 211, 222, 181, 209, 220, 178, 205, 216, 184, 208,
    //         220, 188, 210, 221, 184, 206, 217, 186, 208, 219, 191, 213, 224, 193, 213, 224, 201,
    //         221, 230, 205, 225, 232, 207, 228, 233, 212, 233, 238, 222, 243, 248, 208, 228, 235,
    //         160, 188, 199, 160, 190, 200, 161, 191, 201, 161, 191, 201, 161, 191, 201, 159, 192,
    //         201, 160, 193, 202, 164, 194, 204, 165, 195, 205, 165, 195, 205, 164, 194, 204, 164,
    //         192, 203, 163, 193, 203, 162, 192, 202, 161, 191, 201, 164, 194, 204, 165, 195, 205,
    //         173, 203, 213, 178, 206, 217, 172, 200, 211, 173, 200, 211, 179, 203, 215, 170, 194,
    //         206, 176, 198, 209, 184, 206, 217, 189, 211, 222, 189, 211, 222, 193, 213, 224, 209,
    //         229, 238, 207, 227, 234, 213, 234, 239, 207, 227, 234, 202, 222, 229, 211, 231, 238,
    //         155, 183, 194, 155, 185, 195, 156, 186, 196, 154, 187, 196, 154, 187, 196, 157, 190,
    //         199, 158, 191, 200, 160, 193, 202, 162, 195, 204, 164, 194, 204, 164, 194, 204, 165,
    //         195, 205, 165, 195, 205, 166, 196, 206, 162, 190, 201, 162, 190, 201, 165, 193, 204,
    //         172, 201, 209, 173, 200, 209, 167, 194, 203, 173, 197, 207, 177, 201, 211, 180, 202,
    //         213, 186, 206, 217, 181, 201, 212, 187, 207, 216, 200, 220, 229, 196, 214, 224, 186,
    //         204, 214, 185, 203, 213, 184, 203, 210, 163, 181, 191, 137, 155, 165, 154, 174, 183,
    //         156, 184, 195, 157, 187, 197, 159, 189, 199, 158, 191, 200, 159, 192, 201, 159, 192,
    //         201, 161, 194, 203, 162, 195, 204, 163, 196, 205, 166, 196, 206, 166, 196, 206, 167,
    //         197, 207, 167, 197, 207, 167, 195, 206, 165, 193, 204, 164, 192, 203, 169, 196, 207,
    //         170, 197, 206, 169, 196, 205, 174, 198, 208, 175, 199, 209, 171, 193, 204, 159, 181,
    //         192, 171, 191, 202, 193, 213, 224, 208, 228, 237, 201, 219, 229, 163, 181, 191, 131,
    //         149, 159, 112, 130, 140, 101, 119, 129, 119, 137, 147, 119, 137, 147, 103, 123, 132,
    //         169, 197, 209, 171, 201, 211, 173, 203, 213, 173, 206, 215, 173, 206, 215, 168, 201,
    //         210, 168, 201, 210, 168, 201, 210, 167, 200, 209, 166, 199, 208, 166, 199, 208, 168,
    //         198, 208, 168, 198, 208, 169, 198, 206, 171, 198, 207, 165, 192, 201, 171, 195, 205,
    //         170, 194, 204, 169, 193, 203, 173, 196, 204, 157, 180, 188, 142, 162, 171, 93, 113,
    //         122, 109, 127, 137, 175, 193, 203, 198, 216, 226, 154, 171, 181, 98, 115, 125, 80, 97,
    //         107, 105, 123, 135, 103, 121, 133, 107, 125, 137, 106, 124, 136, 102, 120, 132, 176,
    //         204, 216, 177, 207, 217, 177, 207, 217, 174, 207, 216, 173, 206, 215, 176, 209, 218,
    //         175, 208, 217, 174, 207, 216, 171, 204, 213, 169, 202, 211, 167, 200, 209, 168, 198,
    //         208, 170, 199, 207, 177, 204, 213, 179, 203, 213, 168, 192, 202, 171, 195, 205, 170,
    //         192, 203, 168, 190, 201, 163, 186, 194, 126, 146, 155, 84, 104, 113, 78, 96, 106, 94,
    //         112, 122, 108, 126, 136, 111, 128, 138, 111, 128, 138, 109, 126, 136, 111, 128, 138,
    //         125, 141, 154, 148, 166, 178, 108, 126, 138, 102, 120, 132, 139, 157, 169, 175, 205,
    //         216, 176, 206, 217, 178, 208, 219, 174, 207, 216, 172, 205, 214, 172, 205, 214, 173,
    //         206, 215, 170, 200, 210, 169, 199, 209, 171, 201, 209, 167, 197, 205, 167, 196, 204,
    //         176, 205, 213, 170, 194, 204, 170, 193, 201, 183, 206, 214, 177, 200, 208, 176, 196,
    //         205, 164, 184, 193, 113, 133, 142, 81, 99, 109, 101, 120, 127, 103, 120, 128, 98, 115,
    //         123, 99, 116, 124, 95, 109, 118, 134, 148, 157, 114, 128, 137, 109, 126, 136, 96, 112,
    //         125, 171, 187, 200, 156, 172, 185, 123, 139, 152, 179, 195, 208, 173, 203, 214, 173,
    //         203, 214, 175, 205, 216, 172, 205, 214, 170, 203, 212, 168, 201, 210, 171, 204, 213,
    //         170, 200, 210, 166, 196, 206, 168, 198, 206, 171, 200, 208, 172, 201, 209, 175, 202,
    //         211, 185, 208, 216, 169, 189, 198, 176, 196, 205, 188, 208, 217, 162, 180, 190, 107,
    //         125, 135, 81, 99, 109, 97, 114, 124, 90, 107, 115, 97, 114, 122, 104, 121, 129, 106,
    //         120, 129, 96, 110, 119, 107, 121, 130, 123, 137, 146, 148, 162, 171, 95, 112, 122, 117,
    //         133, 146, 124, 140, 153, 128, 144, 157, 131, 147, 160, 171, 201, 212, 172, 202, 213,
    //         173, 203, 214, 173, 203, 213, 171, 201, 211, 172, 202, 212, 168, 198, 208, 168, 197,
    //         205, 169, 198, 206, 169, 198, 206, 173, 200, 209, 171, 198, 207, 168, 192, 202, 180,
    //         200, 209, 184, 203, 210, 166, 185, 192, 137, 156, 163, 107, 124, 132, 90, 107, 115, 94,
    //         111, 119, 96, 110, 119, 91, 105, 114, 88, 102, 111, 106, 121, 128, 101, 114, 122, 101,
    //         114, 122, 107, 120, 128, 108, 121, 129, 171, 185, 194, 147, 164, 174, 92, 109, 119, 97,
    //         114, 124, 144, 161, 171, 115, 132, 142, 169, 199, 210, 170, 200, 211, 172, 202, 213,
    //         172, 202, 212, 171, 201, 211, 172, 202, 212, 165, 193, 204, 166, 195, 203, 171, 198,
    //         207, 166, 193, 202, 167, 191, 201, 171, 195, 205, 174, 197, 205, 177, 195, 205, 158,
    //         175, 183, 122, 139, 147, 91, 108, 116, 84, 101, 109, 95, 112, 120, 105, 119, 128, 103,
    //         117, 126, 98, 112, 121, 96, 110, 119, 102, 115, 123, 102, 115, 123, 97, 110, 118, 122,
    //         135, 143, 106, 119, 127, 127, 140, 148, 163, 177, 186, 140, 157, 165, 116, 133, 143,
    //         126, 143, 153, 125, 142, 152, 164, 194, 205, 165, 195, 206, 168, 198, 209, 169, 199,
    //         209, 169, 199, 209, 169, 197, 208, 165, 193, 204, 168, 195, 204, 170, 194, 204, 164,
    //         188, 198, 167, 189, 200, 173, 195, 206, 175, 195, 204, 156, 173, 183, 111, 125, 134,
    //         89, 103, 112, 88, 102, 111, 97, 111, 120, 99, 113, 122, 96, 109, 117, 112, 125, 133,
    //         103, 116, 124, 119, 132, 140, 105, 115, 124, 115, 125, 134, 95, 105, 114, 115, 125,
    //         134, 135, 146, 152, 86, 99, 107, 115, 130, 137, 187, 202, 209, 154, 168, 177, 92, 106,
    //         115, 121, 138, 146, 163, 193, 204, 164, 194, 205, 167, 197, 208, 168, 198, 208, 168,
    //         198, 208, 167, 195, 206, 167, 194, 205, 167, 191, 201, 168, 190, 201, 173, 195, 206,
    //         176, 196, 207, 159, 179, 190, 136, 154, 164, 111, 125, 134, 93, 107, 116, 95, 109, 118,
    //         89, 103, 112, 94, 108, 117, 105, 119, 128, 100, 113, 121, 113, 126, 134, 108, 121, 129,
    //         121, 134, 142, 122, 132, 141, 114, 124, 133, 104, 114, 123, 101, 111, 120, 129, 140,
    //         146, 129, 140, 146, 112, 125, 133, 141, 156, 163, 128, 143, 150, 85, 100, 107, 97, 114,
    //         122, 161, 193, 206, 163, 193, 204, 166, 196, 207, 168, 196, 207, 167, 195, 206, 166,
    //         193, 204, 163, 190, 201, 163, 187, 197, 167, 189, 200, 172, 192, 203, 156, 174, 186,
    //         124, 142, 152, 101, 118, 128, 97, 111, 120, 97, 110, 118, 101, 114, 122, 95, 108, 116,
    //         95, 108, 116, 105, 118, 126, 108, 121, 129, 116, 129, 137, 108, 118, 127, 104, 114,
    //         123, 154, 165, 171, 130, 141, 147, 114, 123, 130, 113, 122, 129, 108, 117, 124, 154,
    //         165, 171, 167, 180, 186, 115, 130, 135, 114, 129, 134, 116, 131, 136, 100, 115, 122,
    //         159, 191, 204, 161, 190, 204, 164, 194, 205, 165, 193, 205, 165, 191, 204, 159, 186,
    //         197, 161, 185, 197, 164, 186, 199, 168, 188, 199, 152, 170, 182, 116, 132, 145, 96,
    //         113, 123, 107, 121, 132, 105, 118, 127, 109, 122, 130, 104, 114, 123, 97, 110, 118,
    //         103, 113, 122, 109, 122, 130, 116, 126, 135, 107, 118, 124, 97, 107, 116, 92, 103, 109,
    //         182, 193, 199, 169, 180, 186, 110, 119, 126, 133, 142, 149, 113, 122, 129, 109, 120,
    //         126, 170, 183, 189, 136, 149, 155, 127, 140, 146, 113, 126, 132, 93, 106, 114, 166,
    //         197, 215, 163, 192, 210, 160, 189, 205, 165, 192, 209, 165, 191, 206, 164, 186, 200,
    //         163, 182, 196, 155, 173, 187, 141, 157, 172, 96, 112, 125, 97, 111, 124, 122, 136, 149,
    //         119, 131, 143, 112, 122, 132, 115, 125, 135, 104, 112, 123, 100, 110, 119, 116, 125,
    //         134, 121, 132, 138, 115, 124, 131, 108, 117, 122, 99, 108, 115, 109, 118, 123, 137,
    //         146, 151, 139, 148, 153, 136, 145, 150, 122, 131, 136, 117, 126, 133, 108, 117, 124,
    //         113, 124, 130, 125, 136, 142, 124, 135, 141, 125, 136, 142, 111, 122, 128, 165, 194,
    //         212, 158, 187, 205, 162, 189, 206, 170, 196, 213, 172, 195, 211, 162, 181, 196, 160,
    //         178, 192, 129, 145, 160, 101, 115, 128, 83, 97, 110, 118, 130, 144, 139, 151, 165, 110,
    //         120, 132, 110, 118, 129, 108, 116, 127, 116, 124, 135, 106, 115, 124, 116, 125, 132,
    //         124, 133, 140, 110, 119, 124, 124, 133, 138, 133, 142, 147, 124, 133, 138, 128, 137,
    //         142, 134, 143, 148, 130, 139, 144, 127, 136, 141, 119, 128, 135, 111, 120, 127, 111,
    //         120, 127, 120, 129, 136, 119, 128, 135, 128, 137, 144, 126, 137, 143, 166, 193, 210,
    //         165, 192, 209, 171, 197, 212, 171, 194, 210, 163, 185, 199, 140, 159, 174, 120, 138,
    //         152, 94, 110, 125, 101, 115, 128, 96, 108, 122, 106, 116, 128, 121, 131, 143, 119, 127,
    //         138, 118, 126, 137, 112, 121, 130, 115, 124, 133, 105, 114, 123, 108, 117, 124, 106,
    //         115, 122, 95, 104, 109, 118, 127, 132, 138, 148, 150, 140, 150, 152, 143, 152, 157,
    //         147, 156, 161, 121, 130, 135, 121, 130, 135, 121, 130, 137, 121, 130, 137, 114, 123,
    //         130, 103, 112, 119, 106, 115, 122, 109, 118, 125, 128, 139, 143, 165, 191, 206, 169,
    //         192, 208, 164, 187, 201, 146, 168, 182, 127, 146, 160, 107, 125, 139, 108, 124, 137,
    //         102, 116, 129, 116, 128, 140, 109, 119, 131, 102, 110, 121, 108, 116, 127, 112, 121,
    //         130, 109, 118, 127, 112, 121, 130, 105, 114, 123, 119, 128, 135, 128, 137, 144, 116,
    //         125, 130, 118, 127, 132, 129, 138, 143, 143, 153, 155, 145, 155, 157, 142, 151, 156,
    //         146, 155, 160, 129, 138, 143, 137, 146, 151, 132, 141, 148, 116, 125, 132, 117, 126,
    //         133, 102, 111, 118, 100, 109, 116, 98, 107, 112, 110, 121, 125, 169, 191, 204, 162,
    //         181, 195, 142, 162, 173, 119, 137, 149, 106, 123, 133, 112, 126, 137, 126, 139, 148,
    //         119, 129, 139, 122, 130, 141, 115, 123, 134, 115, 122, 132, 113, 120, 130, 103, 110,
    //         118, 128, 137, 144, 123, 132, 139, 107, 116, 123, 116, 125, 130, 123, 132, 137, 119,
    //         128, 133, 134, 143, 148, 143, 153, 155, 143, 153, 155, 145, 155, 157, 143, 153, 155,
    //         146, 156, 158, 132, 141, 146, 131, 140, 145, 137, 146, 151, 129, 138, 143, 118, 127,
    //         132, 123, 132, 137, 114, 123, 128, 115, 124, 129, 113, 122, 127, 163, 181, 193, 142,
    //         159, 169, 119, 136, 146, 110, 124, 135, 114, 128, 137, 123, 136, 145, 128, 138, 147,
    //         117, 126, 135, 125, 132, 140, 120, 127, 135, 118, 122, 131, 118, 122, 131, 111, 116,
    //         122, 96, 103, 111, 100, 109, 114, 124, 133, 138, 138, 147, 152, 138, 147, 152, 137,
    //         147, 149, 127, 137, 139, 121, 131, 133, 152, 162, 164, 141, 151, 153, 128, 138, 140,
    //         141, 151, 153, 142, 151, 156, 132, 141, 146, 133, 142, 147, 129, 138, 143, 126, 135,
    //         140, 179, 188, 193, 166, 175, 180, 110, 120, 122, 118, 128, 130, 132, 149, 157, 117,
    //         131, 140, 110, 123, 131, 115, 125, 134, 127, 138, 144, 117, 126, 133, 128, 135, 143,
    //         122, 129, 137, 126, 131, 137, 116, 121, 127, 119, 122, 129, 126, 129, 136, 118, 121,
    //         126, 111, 118, 124, 117, 125, 128, 143, 151, 154, 152, 160, 163, 145, 153, 156, 150,
    //         158, 161, 145, 153, 156, 124, 132, 134, 144, 152, 155, 154, 162, 165, 139, 147, 150,
    //         149, 157, 160, 138, 146, 149, 131, 139, 142, 135, 143, 146, 145, 153, 156, 143, 151,
    //         154, 173, 181, 184, 227, 235, 238, 189, 197, 200, 120, 130, 132, 109, 122, 128, 111,
    //         122, 128, 121, 132, 138, 129, 138, 143, 135, 142, 148, 128, 135, 141, 129, 134, 138,
    //         123, 128, 132, 136, 139, 144, 122, 125, 130, 115, 119, 122, 127, 128, 133, 130, 134,
    //         137, 126, 131, 134, 132, 140, 142, 136, 144, 146, 150, 158, 160, 151, 159, 162, 170,
    //         178, 180, 197, 205, 207, 165, 173, 175, 155, 163, 166, 170, 178, 181, 135, 143, 146,
    //         142, 150, 153, 138, 146, 149, 154, 162, 165, 150, 158, 161, 149, 157, 160, 194, 202,
    //         205, 194, 202, 205, 194, 202, 205, 204, 212, 214, 149, 159, 160, 146, 155, 160, 122,
    //         132, 134, 125, 133, 136, 143, 151, 154, 135, 140, 144, 128, 133, 137, 136, 140, 143,
    //         127, 131, 134, 135, 136, 140, 139, 140, 144, 128, 129, 131, 131, 130, 135, 134, 135,
    //         137, 162, 168, 168, 159, 168, 167, 159, 168, 167, 202, 211, 210, 209, 217, 219, 193,
    //         201, 203, 195, 203, 205, 208, 216, 218, 172, 180, 182, 150, 158, 160, 153, 161, 163,
    //         124, 132, 134, 119, 127, 130, 146, 154, 157, 130, 138, 141, 104, 112, 115, 155, 163,
    //         165, 200, 208, 210, 181, 189, 191, 154, 162, 164, 154, 162, 164, 169, 177, 179, 142,
    //         147, 150, 133, 139, 139, 147, 152, 155, 153, 159, 159, 133, 137, 140, 141, 145, 146,
    //         139, 140, 142, 130, 131, 133, 133, 134, 136, 135, 136, 138, 135, 135, 137, 144, 146,
    //         145, 222, 228, 226, 201, 210, 207, 147, 156, 153, 190, 199, 198, 189, 198, 197, 143,
    //         152, 151, 127, 135, 137, 182, 190, 192, 195, 203, 205, 158, 166, 168, 137, 145, 147,
    //         146, 154, 156, 132, 140, 143, 106, 114, 117, 125, 133, 136, 122, 130, 132, 110, 118,
    //         120, 134, 142, 144, 167, 175, 177, 142, 151, 150, 214, 223, 222, 183, 189, 189, 190,
    //         194, 195, 202, 206, 205, 188, 192, 193, 168, 172, 171, 162, 163, 165, 154, 156, 155,
    //         161, 163, 162, 165, 167, 166, 173, 173, 173, 169, 169, 169, 154, 154, 154, 164, 166,
    //         165, 160, 165, 161, 145, 151, 147, 146, 152, 148, 145, 151, 149, 124, 130, 128, 113,
    //         119, 117, 149, 155, 155, 131, 137, 137, 180, 185, 188, 190, 195, 198, 130, 135, 138,
    //         143, 148, 151, 139, 144, 147, 132, 137, 140, 130, 135, 138, 131, 136, 139, 120, 125,
    //         128, 108, 114, 114, 106, 112, 112, 147, 153, 153, 189, 198, 197, 240, 245, 241, 243,
    //         247, 246, 248, 252, 251, 235, 239, 238, 233, 237, 236, 229, 231, 230, 200, 202, 201,
    //         192, 194, 193, 192, 194, 193, 185, 185, 185, 181, 181, 181, 176, 176, 176, 175, 177,
    //         174, 179, 184, 178, 163, 170, 163, 162, 168, 164, 157, 163, 159, 126, 132, 130, 132,
    //         138, 136, 139, 145, 145, 111, 117, 117, 114, 119, 122, 173, 178, 181, 173, 178, 181,
    //         125, 130, 133, 127, 132, 135, 117, 122, 125, 120, 125, 128, 122, 128, 128, 116, 122,
    //         122, 127, 133, 133, 109, 115, 115, 115, 121, 121, 125, 134, 131, 211, 216, 210, 238,
    //         240, 237, 244, 246, 243, 249, 251, 248, 254, 255, 253, 248, 250, 247, 240, 242, 239,
    //         238, 240, 237, 223, 225, 222, 190, 192, 189, 195, 197, 194, 220, 222, 219, 208, 210,
    //         207, 197, 202, 196, 169, 176, 168, 179, 186, 179, 198, 205, 198, 193, 199, 195, 178,
    //         184, 180, 155, 161, 159, 162, 168, 166, 148, 154, 154, 144, 150, 150, 153, 158, 161,
    //         126, 131, 134, 127, 133, 133, 131, 137, 137, 137, 143, 143, 130, 136, 136, 122, 128,
    //         128, 136, 142, 140, 130, 136, 134, 101, 107, 105, 130, 136, 134, 180, 185, 179, 200,
    //         202, 197, 217, 219, 214, 224, 226, 223, 191, 193, 190, 196, 198, 195, 217, 219, 216,
    //         243, 245, 242, 246, 248, 245, 211, 213, 210, 210, 212, 209, 248, 250, 247, 246, 251,
    //         245, 220, 225, 218, 167, 174, 166, 174, 181, 174, 177, 184, 177, 187, 193, 189, 205,
    //         211, 207, 222, 228, 226, 225, 231, 229, 224, 230, 230, 213, 219, 219, 191, 196, 199,
    //         188, 193, 196, 159, 165, 165, 174, 180, 180, 170, 176, 176, 176, 182, 182, 185, 191,
    //         189, 161, 167, 165, 135, 141, 139, 143, 149, 147, 141, 147, 145, 169, 171, 166, 170,
    //         172, 167, 196, 198, 193, 197, 199, 194, 155, 157, 152, 166, 168, 163, 174, 176, 171,
    //         193, 195, 190, 207, 209, 204, 201, 203, 198, 200, 202, 197, 215, 217, 214, 223, 228,
    //         222, 232, 237, 230, 213, 221, 210, 180, 187, 179, 198, 205, 197, 204, 211, 204, 232,
    //         238, 234, 230, 236, 234, 248, 254, 252, 242, 248, 248, 247, 253, 253, 246, 252, 252,
    //         229, 235, 235, 234, 240, 240, 237, 243, 243, 238, 244, 244, 238, 244, 244, 245, 249,
    //         248, 234, 238, 237, 216, 222, 218, 213, 219, 215, 183, 189, 185, 205, 207, 202, 210,
    //         212, 207, 207, 209, 204, 188, 190, 185, 184, 186, 181, 174, 176, 171, 165, 167, 162,
    //         160, 162, 157, 168, 170, 165, 199, 201, 196, 211, 213, 208, 185, 187, 184, 177, 179,
    //         174, 208, 213, 206, 240, 248, 237, 228, 235, 227, 226, 233, 225, 210, 217, 210, 197,
    //         204, 197, 215, 221, 217, 246, 252, 250, 234, 240, 238, 202, 208, 208, 205, 211, 211,
    //         194, 200, 200, 192, 198, 198, 209, 215, 215, 212, 218, 218, 222, 228, 228, 235, 239,
    //         238, 246, 251, 247, 245, 251, 247, 250, 255, 252, 236, 242, 238,
    //     ])
    // }
}
