const LUMINANCE_QUANTIZATION_TABLE: [u8; 64] = [
    16, 11, 10, 16, 124, 140, 151, 161, 12, 12, 14, 19, 126, 158, 160, 155, 14, 13, 16, 24, 140,
    157, 169, 156, 14, 17, 22, 29, 151, 187, 180, 162, 18, 22, 37, 56, 168, 109, 103, 177, 24, 35,
    55, 64, 181, 104, 113, 192, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103,
    199,
];

const CROMINANCE_QUANTIZATION_TABLE: [u8; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

pub enum Format {
    BaselineSequential,
    Lossless,
}

pub enum Precision {
    Eight = 8,
    Twelve = 12,
    Sixteen = 16,
}

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

pub(crate) struct ContextContext<'a> {
    pub(crate) component: usize,
    pub(crate) x_position: usize,
    pub(crate) y_position: usize,
    pub(crate) width: usize,
    pub(crate) component_count: usize,
    pub(crate) p_t: u8,
    pub(crate) p_: u8, // Sample precision
    pub(crate) img: &'a Vec<u32>,
}

impl ContextContext<'_> {
    pub(crate) fn r_a(&self) -> i32 {
        self.img[(self.x_position - 1) * self.component_count
            + self.y_position * self.width * self.component_count
            + self.component] as i32
    }
    pub(crate) fn r_b(&self) -> i32 {
        self.img[self.x_position * self.component_count
            + (self.y_position - 1) * self.width * self.component_count
            + self.component] as i32
    }
    pub(crate) fn r_c(&self) -> i32 {
        self.img[(self.x_position - 1) * self.component_count
            + (self.y_position - 1) * self.width * self.component_count
            + self.component] as i32
    }
    pub(crate) fn r_ix(&self) -> i32 {
        1 << (self.p_ - self.p_t - 1) as i32
    }
}

pub(crate) fn make_prediciton(
    raw_image: &Vec<u32>,
    idx: usize,
    component_count: usize,
    width: usize,
    p_: u8,
    p_t: u8,
    predictor: u8,
) -> u32 {
    let component = raw_image.len() % component_count;
    let context = ContextContext {
        component,
        x_position: (idx / component_count) % width,
        y_position: (idx / component_count) / width,
        width,
        component_count,
        p_t,
        p_,
        img: raw_image,
    };
    predict(context, predictor)
}

fn predict(context: ContextContext, mut predictor: u8) -> u32 {
    if context.x_position == 0 {
        if context.y_position == 0 {
            predictor = 8;
        } else {
            predictor = 2;
        }
    } else if context.y_position == 0 {
        predictor = 1;
    }

    match predictor {
        0 => 0,
        1 => context.r_a() as u32,
        2 => context.r_b() as u32,
        3 => context.r_c() as u32,
        4 => (context.r_a() + context.r_b() - context.r_c()) as u32,
        5 => (context.r_a() + ((context.r_b() - context.r_c()) >> 1)) as u32,
        6 => (context.r_b() + ((context.r_a() - context.r_c()) >> 1)) as u32,
        7 => ((context.r_a() + context.r_b()) / 2) as u32,
        _ => 2u32.pow((context.p_ - context.p_t - 1) as u32),
    }
}

pub(crate) fn number_of_used_bits(numb: &u32) -> usize {
    (32 - numb.leading_zeros()) as usize
}

#[cfg(test)]
mod tests {
    extern crate test;

    use super::*;

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
}
