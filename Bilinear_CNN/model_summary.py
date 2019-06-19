"""this will be used for printing the model summary """
from model_with_loss import build_model

def run():
    """method to run """
    model = build_model(no_class=44, no_last_layer_backbone=17, rate_learning=1.0, rate_decay_weight=1e-8, flg_debug=True)
    model.summary()


if __name__ == "__main__":
    run()
