# Stargan V2 Api

The refactored code for inference of [PyTorch implementation](https://github.com/clovaai/stargan-v2) of [StarGAN v2](https://arxiv.org/abs/1912.01865) wrapped into Flask-API. 

You can build a **Docker image** to run the application with the WSGI server (Gunicorn). For details check [this branch](https://github.com/koren-v/stargan-v2-api/tree/docker_gunicorn).

Otherwise, you need firstly load models weights:

`bash download.sh`

Then install packages:

`pip install -r requirements.txt`

Then run the following to launch the development server:: 

`python app.py`

And finally check: http://127.0.0.1:5000/

## Endpoints:

There is a single **/Interpolate** endpoint that expects POST method with JSON body with keys:

* src: *(file)* Source image ([Note](https://github.com/clovaai/stargan-v2#generating-interpolation-videos) that it should be cropped similar to examples from CelebA-HQ or AFHQ dataset)
* ref: *(file)* Reference image with the same requirement
* label: *(string)* The label of the target image. If you are interested in CelebA-HQ format, there ate two possible labels: "male", "female", if you are interested in AFHQ format, available next labels: "cat", "dog", "wild"

**The response is a file as src and ref**

Example of response:

![image](images/res.jpg)