# ONNX Runtime Inference Demo

Use ORT to inference model at Android device by Java. Inference time is 17ms.

You can run this demo by following steps:

1. export your model to ONNX format following [pytorch official tutorial](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html).

    - [optional] optimize your ONNX model to ORT format by the following command:

``` bash
python -m onnxruntime.tools.convert_onnx_models_to_ort  /path/to/onnx/model --optimization_style Runtime
```

2. place your exported model file to `app/src/main/res/raw/`

3. reference your exported model file in `MainActivity.java`

4. build app and run it on your device. You can see the inference time in the `Run` panel.

    - [optional] you can use Android NNAPI to accelerate inference via GPU. You can open it in `MainActivity.java`.

<details>
    <summary>Show Output!</summary>
```
D/MainActivity: ONNXRuntime available provider: CPU <br/>
D/MainActivity: ONNXRuntime available provider: NNAPI <br/>
D/ORTAnalyzer: InputName: data, JavaType: FLOAT, ONNXType: ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, Shape: [1, 3, 224, 224] <br/>
D/ORTAnalyzer: OutputName: mobilenetv20_output_flatten0_reshape0, JavaType: FLOAT, ONNXType: ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, Shape: [1, 1000] <br/>
D/ORTAnalyzer: Warmup time cost: 14 ms <br/>
D/ORTAnalyzer: Warmup time cost: 12 ms <br/>
D/ORTAnalyzer: Warmup time cost: 12 ms <br/>
D/ORTAnalyzer: Warmup time cost: 11 ms <br/>
D/ORTAnalyzer: Warmup time cost: 11 ms <br/>
D/ORTAnalyzer: Time cost: 11 ms <br/>
D/ORTAnalyzer: Time cost: 11 ms <br/>
D/ORTAnalyzer: Time cost: 12 ms <br/>
D/ORTAnalyzer: Time cost: 11 ms <br/>
D/ORTAnalyzer: Time cost: 12 ms <br/>
D/ORTAnalyzer: Time cost: 13 ms <br/>
D/ORTAnalyzer: Time cost: 11 ms <br/>
D/ORTAnalyzer: Time cost: 11 ms <br/>
D/ORTAnalyzer: Time cost: 11 ms <br/>
D/ORTAnalyzer: Time cost: 11 ms <br/>
D/ORTAnalyzer: Time cost: 11 ms <br/>
D/ORTAnalyzer: Time cost: 11 ms <br/>
D/ORTAnalyzer: Time cost: 11 ms <br/>
D/ORTAnalyzer: Time cost: 11 ms <br/>
D/ORTAnalyzer: Time cost: 10 ms <br/>
D/ORTAnalyzer: Time cost: 10 ms <br/>
D/ORTAnalyzer: Time cost: 10 ms <br/>
D/ORTAnalyzer: Time cost: 9 ms <br/>
D/ORTAnalyzer: Time cost: 10 ms <br/>
D/ORTAnalyzer: Time cost: 10 ms <br/>
D/ORTAnalyzer: Average time cost: 11 ms <br/>
```
</details>
