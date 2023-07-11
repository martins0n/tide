python pipe_raw.py  +experiment=etth2 experiment.horizon=96 +dataset=etth2 +model=etth2 model.max_epochs=50
python pipe_raw.py  +experiment=etth1 experiment.horizon=96 +dataset=etth1 +model=etth1 model.max_epochs=50
python pipe_raw.py +experiment=ettm1 experiment.horizon=96 +dataset=ettm1 +model=ettm1 model.max_epochs=50
python pipe_raw.py +experiment=ettm2 experiment.horizon=96 +dataset=ettm2 +model=ettm2 model.max_epochs=50
python pipe_raw.py +experiment=electricity_96 experiment.horizon=96 +dataset=electricity +model=electricity_96 model.max_epochs=50
python pipe_raw.py +experiment=traffic experiment.horizon=96 +dataset=traffic +model=traffic model.max_epochs=50
python pipe_raw.py +experiment=weather experiment.horizon=96 +dataset=weather +model=weather model.max_epochs=50
