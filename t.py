import onnx
import onnx_graphsurgeon as gs
import numpy as np

# 加载原始 ONNX 模型
graph = gs.import_onnx(onnx.load("/home/hy/wust_vision/model/opt-1208-001.onnx"))

# 假设原始模型的输入节点名为 'images'
input_tensor = graph.inputs[0]  # e.g., shape (1, H, W, 3) or (1, 3, H, W)

# 插入 Letterbox Plugin
letterbox_output = gs.Variable(
    name="letterbox_output", dtype=np.float32, shape=[1, 3, 416, 416]
)

letterbox_plugin = gs.Node(
    op="LetterboxPreprocess",  # 必须与 plugin type 对应
    name="LetterboxPreprocess_Node",
    inputs=[input_tensor],
    outputs=[letterbox_output],
    attrs={"out_w": 416, "out_h": 416},  # 你的 Plugin 创建器中读取的属性
)

# 替换 graph 输入为 plugin 输出
graph.nodes.append(letterbox_plugin)
graph.outputs[0].inputs[0] = letterbox_output

# 更新 graph 输入
graph.inputs = [input_tensor]  # 保留原始输入（也可更名为 raw_input）

# 保存修改后的模型
onnx.save(gs.export_onnx(graph), "model_with_letterbox.onnx")
