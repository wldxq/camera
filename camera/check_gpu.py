import onnxruntime as ort

print("ONNX Runtime version:", ort.__version__)
print("Available providers:", ort.get_available_providers())

if "CUDAExecutionProvider" in ort.get_available_providers():
    print("GPU 可用：CUDAExecutionProvider 已启用")
else:
    print("GPU 不可用：当前没有检测到 CUDAExecutionProvider")