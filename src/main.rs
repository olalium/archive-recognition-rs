use archive_recognition_rs::{Config, ImageEncodings};
use archive_recognition_rs::ImageProcessor;
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::process;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();

    let config = Config::new(&args).unwrap_or_else(|err| {
        println!("Problem parsing arguments: {}", err);
        process::exit(1);
    });

    let image_processor = ImageProcessor::new().unwrap_or_else(|err| {
        println!("Problem initializing impae processor: {}", err);
        process::exit(1);
    });

    run(&config, &image_processor).unwrap_or_else(|e| {
        println!("Application error: {}", e);
        process::exit(1);
    });
}

fn run(config: &Config, image_processor: &ImageProcessor) -> Result<(), Box<dyn Error>> {
    let dir_path = fs::canonicalize(&config.foldername)?;
    process_directory(&dir_path, &image_processor)?;
    Ok(())
}

fn process_directory(dir_path: &PathBuf, image_processor: &ImageProcessor) -> Result<(), Box<dyn Error>> {
    let absolute_path = fs::canonicalize(dir_path)?;
    println!("Inside dir: {:?}", absolute_path);
    for entry in fs::read_dir(dir_path)? {
        let entry = entry?.path();
        if entry.is_file() {
            process_file(&entry, &image_processor)?;
        } else {
            process_directory(&entry, &image_processor)?;
        }
    }
    Ok(())
}

fn process_file(file: &PathBuf, image_processor: &ImageProcessor) -> Result<(), Box<dyn Error>>  {
    let absolute_path = fs::canonicalize(file)?;
    let image_result = ImageEncodings::process_image(&absolute_path, image_processor);
    if image_result.is_err() {
        return Ok(println!("Something went wrong opening image: {}", image_result.err().unwrap().to_string()));
    }
    Ok(())
}