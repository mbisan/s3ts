
from pytorch_lightning.utilities.model_summary import ModelSummary
from s3ts.models.wrapper import WrapperModel

model = WrapperModel(mode="TS",
             arch="RNN",
             target="cls", # "cls"
             n_classes=2,
             n_patterns=2,
             l_patterns=100,
             window_length=70,
             stride_series=False,
             window_time_stride=1,
             window_patt_stride=1,
             encoder_feats=40,
             decoder_feats=64,
             learning_rate=1E-4)
             
summary = ModelSummary(model, max_depth=2)
print(summary)