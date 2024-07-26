# Liked it? Please give a ⭐️ to build this 💪 stronger.
# 👋 Introduction
<p align="center">
    <a href="https://pypi.org/project/face-recognition/" target="blank"/>
        <img src="./images/demo.gif" alt="Facial Recognition Image" />
    </a>
</p>

`Project Goal`:Identify faces in video stream using Deep Dearning and OpenCV.


# 💻 [Face Recognition Library](https://pypi.org/project/face-recognition/).
Recognize and manipulate faces from Python or from the command line with
the world’s simplest face recognition library.

Built using dlib’s **state-of-the-art** face recognition
built with deep learning. The model has an accuracy of 99.38% on the Labeled Faces in the Wild benchmark.


# 🔥 Facial Recognition Summary steps.

> `Encoder.py' file.
- Grab labels and images.
- Extracts faces from the whole image.
- Create image embedding.
>`Recognizer.py`
- Extract faces from video frames.
- Create face embedding.
- Compare the face embeddings with the embeddings created earlier using the `encoder.py` file.
- Selects and outputs the label of the best match.


# 🏗️ How to reproduce the project
You can run this code locally with these few easy steps.

1. Clone the repository

```bash
https://github.com/Abayie/facial_recognition.git
```

2. Install dependencies

```bash
pip install opencv-python
pip install numpy
```
[Download and install Facial-Recognition library](https://pypi.org/project/face-recognition/)

**Nb:** You can simply use pip to install other python libraries whenever necessary.

3. Changing Images
- In the `known_images_folder` replace the images and rename the names with your own.

4. Running application.
- Launch command line.
- Navigate to the directory where `encoder.py` and `recognizer.py` files are.
- Type `python encoder.py` and press enter. Wait till you see `Learning Completed` on the console.
- Type `python recognizer.py` and press enter. This runs the application.


# 🛡️ License
This project is licensed under the MIT License - see the [`LICENSE`](LICENSE) file for details.

# 🙏 Support

We all need support and motivation. Please give this project a ⭐️ to encourage and show that you liked it. Don't forget to leave a star ⭐️ before you move away.

