from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.freeze_support()  # Optional but recommended on Windows

    # Train the model on the COCO8 dataset for 100 epochs
    train_results = model.train(
                                    data="dataset_custom.yaml",  # Path to dataset configuration file
                                    epochs=25,  # Number of training epochs
                                    batch=64,
                                    imgsz=480,  # Image size for training
                                    workers=0,
                                    device="0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    )


# # Evaluate the model's performance on the validation set
# metrics = model.val()
#
# # Perform object detection on an image
# results = model("path/to/image.jpg")  # Predict on an image
# results[0].show()  # Display results
#
# # Export the model to ONNX format for deployment
# path = model.export(format="onnx")  # Returns the path to the exported model
