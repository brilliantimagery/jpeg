use std::env;
use std::fs::File;
use std::io::Read;

pub(crate) fn get_file_as_byte_iter(file: &str) -> Vec<u8> {
    let mut path = env::current_dir().unwrap();
    path.push("tests");
    path.push("common");
    path.push(file);
    let path = path.as_path();
    let mut file = File::open(path).expect("The test file wasn't where it was expected.");
    let mut encoded_image = Vec::from([]);
    match file.read_to_end(&mut encoded_image) {
        Ok(_) => {}
        Err(_) => panic!("Ther was a test file read error"),
    }

    encoded_image
}
