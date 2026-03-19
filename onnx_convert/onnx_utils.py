import json

import onnx
import datetime


def rename_onnx_node(model, names_to_replace):

    for old_name, new_name in names_to_replace:
        for node in model.graph.node:
            if node.name == old_name:
                node.name = new_name

        # also fix the input and output names
        for input in model.graph.input:
            if input.name == old_name:
                input.name = new_name

        for output in model.graph.output:
            if output.name == old_name:
                output.name = new_name

        # also fix node data in the downstream nodes
        for node in model.graph.node:
            for i, input in enumerate(node.input):
                if input == old_name:
                    node.input[i] = new_name

            for i, output in enumerate(node.output):
                if output == old_name:
                    node.output[i] = new_name

    return model


def _serialize_value(v):
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return json.dumps({str(i): item for i, item in enumerate(v)})
    return json.dumps(v)


def add_metadata(model, metadata):
    metadata_onnx = [
        onnx.StringStringEntryProto(key=k, value=_serialize_value(v)) for k, v in metadata.items()
    ]

    model.metadata_props.extend(metadata_onnx)

    return model
