import argparse
import logging
import os
import numpy as np
from flask import Flask
from flask import jsonify
from flask import request
from flask_api import status
from ModelAccessor import ModelAccessor
import InputProcessor as inputProcessor


class MeetupPredictor(Flask):
    modelAccessor = None

    def __init__(self, *args, **kwargs):
        super(MeetupPredictor, self).__init__(*args, **kwargs)
        self.add_url_rule('/api/rsvp-prediction',
                          view_func=self.rsvp_prediction, methods=["GET"])
        self.modelAccessor = ModelAccessor()

    def rsvp_prediction(self):
        if self.modelAccessor is None:
            return "Application is not yet ready", status.HTTP_200_OK

        params = {}
        if not inputProcessor.process_and_validate_input(request.args, params):
            return "Bad request", status.HTTP_400_BAD_REQUEST

        input = inputProcessor.transform_input(params, self.modelAccessor)

        # Predict the expected rsvp with the pre-trained XGBoost model.
        result = self.modelAccessor.prediction_model.predict(input)
        result_over = self.modelAccessor.over_prediction_model.predict(input)
        result_under = self.modelAccessor.under_prediction_model.predict(input)

        return jsonify({
            "yes_rsvps": {
                "lower": round(np.exp(result_under[0]) - 1),
                "expected": round(np.exp(result[0]) - 1),
                "upper": round(np.exp(result_over[0]) - 1)
            }
        })


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-address',
                        default=os.environ.get('ADDRESS', '0.0.0.0'),
                        type=str,
                        help='App address')
    parser.add_argument('-port',
                        default=os.environ.get('PORT', 8080),
                        type=int,
                        help='App port')
    parser.add_argument('-debug',
                        action='store_true',
                        help='Debug mode for Flask')
    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(
        format="%(asctime)s.%(msecs).3d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    )
    args = argument_parser()
    app = MeetupPredictor(__name__)
    app.run(host=args.address, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
