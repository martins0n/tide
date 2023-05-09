# [Long-term Forecasting with TiDE: Time-series Dense Encoder](https://arxiv.org/pdf/2304.08424.pdf) - Unofficial Pytorch Implementation - WIP

## Commands

### Install
- `make install`

### TiDE

- `python pipe.py +experiment=pattern +dataset=pattern +model=pattern  model.max_epochs=500`
- `python pipe.py +experiment=electricity_96 +dataset=electricity +model=electricity_96 model.max_epochs=10 model.train_batch_size=512 model.test_batch_size=512 model.lr=0.0009`

### Naive
- `python pipe.py +experiment=electricity_96 +dataset=electricity +baseline=naive`
- `python pipe.py +experiment=pattern +dataset=pattern +baseline=naive`

### Data preparation
- `make datasets`

### ToDo
- [ ] Metrics reproducibility
- [ ] REVIN integration