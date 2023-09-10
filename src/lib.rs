use std::path::PathBuf;
use dlib_face_recognition::{ImageMatrix, FaceDetectorCnn, FaceDetectorTrait, FaceLocations, FaceLandmarks, LandmarkPredictor, LandmarkPredictorTrait, FaceEncoderNetwork, FaceEncoderTrait, FaceEncodings};
use image::ImageError;

pub struct Config {
    pub foldername: String
}

impl Config {
    pub fn new(args: &[String]) -> Result<Config, &str> {
        if args.len() < 2 {
            return Err("Not enough arguments");
        }

        let foldername = args[1].clone();

        Ok(Config { foldername })
    }
}

pub struct ImageProcessor {
    face_detector: FaceDetectorCnn,
    landmark_predictor: LandmarkPredictor,
    face_encoder: FaceEncoderNetwork
}

impl ImageProcessor {
    pub fn new() -> Result<ImageProcessor, String> {
        Ok(ImageProcessor {
            face_detector: Self::load_face_detector()?,
            landmark_predictor: Self::load_landmark_predictor()?,
            face_encoder: Self::load_face_encoder()?
        })
    }

    fn load_face_detector() -> Result<FaceDetectorCnn, String> {
        Ok(FaceDetectorCnn::open("files/mmod_human_face_detector.dat")?)
    }

    fn load_landmark_predictor() -> Result<LandmarkPredictor, String> {
        Ok(LandmarkPredictor::open("files/shape_predictor_68_face_landmarks.dat")?)
    }

    fn load_face_encoder() -> Result<FaceEncoderNetwork, String> {
        Ok(FaceEncoderNetwork::open("files/dlib_face_recognition_resnet_model_v1.dat")?)
    }
}

pub struct ImageEncodings {
    pub matrix: ImageMatrix,
    pub locations: FaceLocations,
    pub landmarks: Vec<FaceLandmarks>,
    pub encodings: Vec<FaceEncodings>
}

impl ImageEncodings {
    pub fn process_image(img: &PathBuf, image_processor: &ImageProcessor) -> Result<ImageEncodings, ImageError> {
        let image_matrix = Self::get_image_matrix(img)?;
        let face_locations = Self::get_face_locations(&image_matrix, &image_processor.face_detector);
        let face_landmarks = Self::get_face_landmarks(&image_matrix, &face_locations, &image_processor.landmark_predictor);
        let face_encodings = Self::get_face_encodings(&image_matrix, &face_landmarks, &image_processor.face_encoder);
    
        println!("{} faces detected in file: {:?}", face_locations.len(), img);
        println!("{} face encodings in file: {:?}", face_encodings.len(), img);
        Ok(ImageEncodings{
            matrix: image_matrix,
            encodings: face_encodings,
            landmarks: face_landmarks,
            locations: face_locations
        })
    }
    
    fn get_image_matrix(img: &PathBuf) -> Result<ImageMatrix, ImageError> {
        let image = image::open(img)?.to_rgb8();
        Ok(ImageMatrix::from_image(&image))
    }
    
    fn get_face_locations(image_matrix: &ImageMatrix, face_detector: &FaceDetectorCnn) -> FaceLocations {
        face_detector.face_locations(image_matrix)
    }
    
    fn get_face_landmarks(image_matrix: &ImageMatrix, face_locations: &FaceLocations, landmark_predictor: &LandmarkPredictor) -> Vec<FaceLandmarks> {
        let mut face_landmarks: Vec<FaceLandmarks> = Vec::new();
        for location in face_locations.iter() {
            face_landmarks.push(landmark_predictor.face_landmarks(&image_matrix, location))
        }
        face_landmarks
    }
    
    fn get_face_encodings(image_matrix: &ImageMatrix, face_landmarks: &Vec<FaceLandmarks>, face_encoder: &FaceEncoderNetwork) -> Vec<FaceEncodings> {
        let mut face_encodings: Vec<FaceEncodings> = Vec::new();
        for landmarks in face_landmarks {
            face_encodings.push(face_encoder.get_face_encodings(&image_matrix, &[landmarks.clone()], 0))
        }
        face_encodings
    }
}