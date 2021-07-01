# torch_profiler_bug

### ENV
I'm running python 3.7.9 on linux

```bash
pip install torch-tb-profiler tensorboard pytorch_lightning

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```


I found something strange with pytorch_lightning and pytorch profilers
Running on pytorch 1.9 with cuda11 the training will hang just after the first epoch have started
I'have 2 gpus on my machine: rtx2060 & rtx3070 

running this will work, and the model starts training on my rtx3070
```python
profiler = PyTorchProfiler(
    on_trace_ready=torch.profiler.tensorboard_trace_handler("lightning_logs")
)
trainer = Trainer(
    profiler=profiler
)
```


running this will work, and the model starts training on my rtx2060
```python
profiler = PyTorchProfiler(
    on_trace_ready=torch.profiler.tensorboard_trace_handler("lightning_logs")
)
trainer = Trainer(
    gpus=[1],
    profiler=profiler
)
```
running this will not work, and the model won't start training on my rtx3070
```python
profiler = PyTorchProfiler(
    on_trace_ready=torch.profiler.tensorboard_trace_handler("lightning_logs")
)
trainer = Trainer(
    gpus=[0],
    profiler=profiler
)
```
The last snippet will work if I add `emit_nvtx=True`, but it won't emit the `torch-tb-profiler` i'm trying to generate


