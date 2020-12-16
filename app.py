import io
import base64

from flask import Flask, request, make_response, jsonify
from flask_restx import Resource, Api

from PIL import Image

from inference import Predictor, CELEBRITY_LABELS, ANIMAL_LABELS


application = Flask(__name__)
api = Api(application, version="0.0", title="StarGAN-v2-api")


@api.route("/Interpolate", methods=['POST'])
class Interpolator(Resource):
    @staticmethod
    def post():
        try:
            parameters = request.form
            if "label" in parameters:
                label = parameters["label"]
            else:
                message = "Input json doesn't contain 'label' parameter"
                return make_response(jsonify(error=message), 400)

            src = Image.open(request.files["src"]).convert("RGB")
            ref = Image.open(request.files["ref"]).convert("RGB")

            if label in ANIMAL_LABELS:
                entity = "animal"
            elif label in CELEBRITY_LABELS:
                entity = "celebrity"
            else:
                message = "'label' parameter must be in " \
                          f"{', '.join(str(value) for value in ANIMAL_LABELS)}" \
                          f" or in {', '.join(str(value) for value in CELEBRITY_LABELS)}"
                return make_response(jsonify(error=message), 400)

            predictor = Predictor(entity=entity)
            result_image = predictor.create_interpolation(label, src_image=src, ref_image=ref)

            raw_bytes = io.BytesIO()
            result_image.save(raw_bytes, "JPEG")
            raw_bytes.seek(0)

            image_base64 = base64.b64encode(raw_bytes.read())
            str_image = str(image_base64)

            return make_response(jsonify({"image": str_image}), 200)

        except Exception as e:
            message = f"Exception occurred: {e}"
            return make_response(jsonify(error=message), 400)


if __name__ == '__main__':
    application.run()
