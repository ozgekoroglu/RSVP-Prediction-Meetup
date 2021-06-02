import dill as pickle
import joblib


class ModelAccessor:
    prediction_model = None
    over_prediction_model = None
    under_prediction_model = None
    lda_model = None
    lda_model_eventName = None

    def __init__(self):
        self.initialize_models()

    def initialize_models(self):
        # load the models from disk
        self.prediction_model = joblib.load('../models/finalized_model_rsvp_pred.sav')

        with open('../models/finalized_model_rsvp_pred_over.pkl', 'rb') as over_pred:
            self.over_prediction_model = pickle.load(over_pred)
        over_pred.close()

        with open('../models/finalized_model_rsvp_pred_under.pkl', 'rb') as under_pred:
            self.under_prediction_model = pickle.load(under_pred)
        under_pred.close()

        self.lda_model = joblib.load('../models/lda_model_event_desc.sav')
        self.lda_model_eventName = joblib.load('../models/lda_model_event_name.sav')
