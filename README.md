# [Long-term Forecasting with TiDE: Time-series Dense Encoder](https://arxiv.org/pdf/2304.08424.pdf) - Unofficial Pytorch Implementation - WIP

## Commands

### Install
- `make install`

### Data preparation
- `make datasets`

### Metrics and wandb link with commands
| name                                                             |   test mae |   paper mae |   test mse |  paper mse |
|:-----------------------------------------------------------------|-----------:|------:|-----------:|------:|
| [ETTh1 96](https://wandb.ai/martins0n/tide/runs/ui52fc9m)        |   0.450671 | 0.398 |   0.427044 | 0.375 |
| [ETTh1 192](https://wandb.ai/martins0n/tide/runs/ax9nkchm)       |   0.486024 | 0.422 |   0.47277  | 0.412 |
| [ETTh1 336](https://wandb.ai/martins0n/tide/runs/7mppixqo)       |   0.527694 | 0.433 |   0.527147 | 0.435 |
| [ETTh1 720](https://wandb.ai/martins0n/tide/runs/ajohb1gh)       |   0.60546  | 0.465 |   0.644379 | 0.454 |
| [ETTh2 96](https://wandb.ai/martins0n/tide/runs/8ja8qqag)        |   0.284329 | 0.336 |   0.169457 | 0.27  |
| [ETTh2 192](https://wandb.ai/martins0n/tide/runs/539j92dg)       |   0.318749 | 0.38  |   0.206157 | 0.332 |
| [ETTh2 336](https://wandb.ai/martins0n/tide/runs/ncj7tgln)       |   0.337509 | 0.407 |   0.226434 | 0.36  |
| [ETTh2 720](https://wandb.ai/martins0n/tide/runs/hw59hr9s)       |   0.400112 | 0.451 |   0.299051 | 0.419 |
| [ETTm1 96](https://wandb.ai/martins0n/tide/runs/ewcpdu4h)        |   0.369346 | 0.349 |   0.31828  | 0.306 |
| [ETTm1 192](https://wandb.ai/martins0n/tide/runs/sdb7te8i)       |   0.399559 | 0.366 |   0.365496 | 0.335 |
| [ETTm1 336](https://wandb.ai/martins0n/tide/runs/qlm1qsu8)       |   0.429598 | 0.384 |   0.408837 | 0.364 |
| [ETTm1 720](https://wandb.ai/martins0n/tide/runs/kvmfc3cp)       |   0.470707 | 0.413 |   0.45856  | 0.413 |
| [ETTm2 96](https://wandb.ai/martins0n/tide/runs/zs2qwtna)        |   0.225635 | 0.251 |   0.111662 | 0.161 |
| [ETTm2 192](https://wandb.ai/martins0n/tide/runs/6ciy0lpy)       |   0.248651 | 0.289 |   0.136481 | 0.215 |
| [ETTm2 336](https://wandb.ai/martins0n/tide/runs/3u83u724)       |   0.271058 | 0.326 |   0.161007 | 0.267 |
| [electricity 96](https://wandb.ai/martins0n/tide/runs/d94fut38)  |   0.236805 | 0.229 |   0.136566 | 0.132 |
| [electricity 192](https://wandb.ai/martins0n/tide/runs/4ja7dfx3) |   0.251745 | 0.243 |   0.151502 | 0.147 |
| [electricity 336](https://wandb.ai/martins0n/tide/runs/j1jnqlkd) |   0.283281 | 0.261 |   0.176307 | 0.161 |
| [electricity 720](https://wandb.ai/martins0n/tide/runs/9kbnl2yc) |   0.307335 | 0.294 |   0.205603 | 0.196 |
| [traffic 96](https://wandb.ai/martins0n/tide/runs/3f315828)      |   0.280911 | 0.253 |   0.438414 | 0.336 |
| [traffic 192](https://wandb.ai/martins0n/tide/runs/2f7quf86)     |   0.284845 | 0.257 |   0.434229 | 0.346 |
| [traffic 336](https://wandb.ai/martins0n/tide/runs/a2xtrh0r)     |   0.290309 | 0.26  |   0.498862 | 0.355 |
| [traffic 720](https://wandb.ai/martins0n/tide/runs/ermr1nhy)     |   0.309352 | 0.273 |   0.50618  | 0.386 |
| [weather 96](https://wandb.ai/martins0n/tide/runs/bst8g2bj)      |   0.225352 | 0.222 |   0.161776 | 0.166 |
| [weather 192](https://wandb.ai/martins0n/tide/runs/7yhjoq06)     |   0.256947 | 0.263 |   0.192686 | 0.209 |
| [weather 336](https://wandb.ai/martins0n/tide/runs/7dbsqbss)     |   0.289164 | 0.301 |   0.224491 | 0.254 |
| [weather 720](https://wandb.ai/martins0n/tide/runs/mem4t9cp)     |   0.330308 | 0.34  |   0.266972 | 0.313 |

### ToDo

- ➕➖ Metrics reproducibility
    - some issues with traffic and etth1
- [ ] REVIN integration