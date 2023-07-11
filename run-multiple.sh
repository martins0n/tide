horizon=${horizon:-96}

# python pipe_raw.py  +experiment=etth2 experiment.horizon=$horizon +dataset=etth2 +model=etth2 model.max_epochs=50 accelerator=cuda
# python pipe_raw.py  +experiment=etth1 experiment.horizon=$horizon +dataset=etth1 +model=etth1 model.max_epochs=50 accelerator=cuda
# python pipe_raw.py +experiment=ettm1 experiment.horizon=$horizon +dataset=ettm1 +model=ettm1 model.max_epochs=50 accelerator=cuda
# python pipe_raw.py +experiment=ettm2 experiment.horizon=$horizon +dataset=ettm2 +model=ettm2 model.max_epochs=50 accelerator=cuda
python pipe_raw.py +experiment=electricity_96 experiment.horizon=$horizon +dataset=electricity +model=electricity_96 model.max_epochs=5 accelerator=cuda
python pipe_raw.py +experiment=traffic experiment.horizon=$horizon +dataset=traffic +model=traffic model.max_epochs=5 accelerator=cuda
python pipe_raw.py +experiment=weather experiment.horizon=$horizon +dataset=weather +model=weather model.max_epochs=5 accelerator=cuda
