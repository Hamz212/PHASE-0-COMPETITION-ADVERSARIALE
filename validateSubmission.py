import importlib.util
import os
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SUBMISSION_FILE = "submission.zip"
SUBMISSION_PATH = SCRIPT_DIR / SUBMISSION_FILE
EXPECTED_FILES = {"model.py", "model.pth"}

MODEL_CLASS_NAME = "IDSModel"
MODEL_INPUT_SIZE = 76

MAX_PREDICT_TIME_SECONDS = 60


def validateSubmission():
    #Check if submission file exists
    if not os.path.exists(SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file '{SUBMISSION_FILE}' not found. \
            Please ensure you have created the submission zip file with the required files."
        )
    print("[+] Submission file found. Validating contents...")

    #Use a temporary directory (will be automatically cleaned up) to extract the submission zip file and validate its contents
    with tempfile.TemporaryDirectory() as tempDir:
        #Extract submission zip file to temporary directory
        with zipfile.ZipFile(SUBMISSION_PATH, "r") as zipRef:
            zipRef.extractall(tempDir)

        #Get all files from submission file and define as set
        extractedFiles = set(os.listdir(tempDir))

        #Subtracted two sets to find missing files in submission (if any files are missing, the set will be non-empty)
        missingFiles = EXPECTED_FILES - extractedFiles
        if missingFiles:
            raise ValueError(
                f"Submission is missing the following required files: {missingFiles}. \
                Please ensure you have included all required files in your submission zip file."
            )

        print(
            "[+] All required files found in submission. Validating model loading and prediction..."
        )

        #Get the paths for model.py and model.pth from the extracted files
        modelPyPath = os.path.join(tempDir, "model.py")
        modelPthPath = os.path.join(tempDir, "model.pth")

        #Dynamically import the model.py file from the extracted submission
        spec = importlib.util.spec_from_file_location("model", modelPyPath)

        #Import the model.py as a module
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, MODEL_CLASS_NAME):
            raise ValueError(
                f"Model class '{MODEL_CLASS_NAME}' not found in model.py. \
                Please ensure you have defined your model class with the name '{MODEL_CLASS_NAME}' in model.py."
            )

        print(
            f"[+] Model class '{MODEL_CLASS_NAME}' found in model.py. Validating model loading..."
        )

        #Instantiate the model
        model = module.IDSModel(inputSize=MODEL_INPUT_SIZE)

        #Load model weights from model.pth
        model.loadModel(modelPthPath)

        print(
            "[+] Model weights loaded successfully from model.pth. Validating model prediction..."
        )

        #Ensure prediction function exists
        if not hasattr(model, "predict"):
            raise ValueError(
                "Model does not have a 'predict' function. \
                Please ensure your model class has a 'predict' function that takes in samples and returns predictions."
            )

        #Ensure prediction function works by passing in dummy data and checking output shape
        #10 samples with input size features
        startTime = time.time()
        dummySamples = np.random.rand(10, MODEL_INPUT_SIZE)
        predictions = model.predict(dummySamples)

        #Ensure that the prediction function runs within the specified time limit
        endTime = time.time()
        if endTime - startTime > MAX_PREDICT_TIME_SECONDS:
            raise ValueError(
                f"Model 'predict' function took too long ({endTime - startTime:.2f} seconds). \
                Please ensure your 'predict' function is optimized for performance."
            )

        #Ensure predictions have correct shape (one dimensional array with length equal to number of sampels)
        if predictions.shape != (10,):
            raise ValueError(
                f"Model 'predict' function output shape is {predictions.shape}, but expected (10,). \
                Please ensure your 'predict' function returns a 1D array of predictions with shape (num_samples,)."
            )

        #Ensure predictions are binary (0 or 1) since this is a binary classification task
        uniqueValues = set(predictions.tolist())
        if not uniqueValues.issubset({0, 1}):
            raise ValueError(
                f"Model 'predict' function returned values {uniqueValues}, but expected binary values (0 and 1). \
                Please ensure your 'predict' function returns binary predictions (0 for normal, 1 for attack)."
            )

        print("[+] Model prediction function validated successfully.!")

        #Ref: https://textart4u.blogspot.com/2012/04/smiley-face-with-thumbs-up-text-art.html
        print(r"""
            ██╗   ██╗ █████╗ ██╗     ██╗██████╗
            ██║   ██║██╔══██╗██║     ██║██╔══██╗
            ██║   ██║███████║██║     ██║██║  ██║
            ╚██╗ ██╔╝██╔══██║██║     ██║██║  ██║
            ╚████╔╝ ██║  ██║███████╗██║██████╔╝
            ╚═══╝  ╚═╝  ╚═╝╚══════╝╚═╝╚═════╝

            🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉

                YOUR SUBMISSION IS VALID!

            ░░░░░░░░░░░░░░░░░░░░░░█████████
            ░░███████░░░░░░░░░░███▒▒▒▒▒▒▒▒███
            ░░█▒▒▒▒▒▒█░░░░░░░███▒▒▒▒▒▒▒▒▒▒▒▒▒███
            ░░░█▒▒▒▒▒▒█░░░░██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██
            ░░░░█▒▒▒▒▒█░░░██▒▒▒▒▒██▒▒▒▒▒▒██▒▒▒▒▒███
            ░░░░░█▒▒▒█░░░█▒▒▒▒▒▒████▒▒▒▒████▒▒▒▒▒▒██
            ░░░█████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██
            ░░░█▒▒▒▒▒▒▒▒▒▒▒▒█▒▒▒▒▒▒▒▒▒█▒▒▒▒▒▒▒▒▒▒▒██
            ░██▒▒▒▒▒▒▒▒▒▒▒▒▒█▒▒▒██▒▒▒▒▒▒▒▒▒▒██▒▒▒▒██
            ██▒▒▒███████████▒▒▒▒▒██▒▒▒▒▒▒▒▒██▒▒▒▒▒██
            █▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒█▒▒▒▒▒▒████████▒▒▒▒▒▒▒██
            ██▒▒▒▒▒▒▒▒▒▒▒▒▒▒█▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██
            ░█▒▒▒███████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██
            ░██▒▒▒▒▒▒▒▒▒▒████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒█
            ░░████████████░░░█████████████████
            Dr. Ranwa says:
            
            EXCELLENT WORK! 
        
            🚀 READY FOR THE LEADERBOARD 🚀

            🎉 GOOD LUCK!!! 🎉
""")


if __name__ == "__main__":
    validateSubmission()
