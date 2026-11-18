import importlib.util
import inspect
import tempfile
import json
import math
import os
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader


def report(kind, **data):
    print("NNMIND_EVENT:" + json.dumps(dict(type=kind, **data), allow_nan=False), flush=True)


def source_module(path, original_path, name):
    # Put imported data scripts in their original package context so local imports work.
    directory = os.path.dirname(os.path.abspath(original_path)) if original_path else ""
    package = []
    root = directory
    while root and os.path.isfile(os.path.join(root, "__init__.py")):
        package.insert(0, os.path.basename(root))
        root = os.path.dirname(root)
    for location in [directory, root]:
        if location and location not in sys.path:
            sys.path.insert(0, location)
    full_name = ".".join(package + [name])
    spec = importlib.util.spec_from_file_location(full_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    if original_path:
        module.__file__ = original_path
        with open(path, encoding="utf-8") as source:
            exec(compile(source.read(), original_path, "exec"), module.__dict__)
    else:
        spec.loader.exec_module(module)
    return module


def dataset_loaders(splits, batch_size):
    if isinstance(splits, (tuple, list)) and len(splits) == 3:
        splits = dict(zip(("train", "validation", "test"), splits))
    if not isinstance(splits, dict):
        raise TypeError("build_datasets() must return train, validation, test datasets.")
    if "validation" not in splits and "val" in splits:
        splits = dict(splits, validation=splits["val"])
    loaders = {}
    for name in ("train", "validation", "test"):
        value = splits.get(name)
        if value is None:
            raise ValueError("Missing dataset split: " + name)
        if isinstance(value, DataLoader):
            loaders[name] = value
        else:
            loaders[name] = DataLoader(value, batch_size=batch_size, shuffle=name == "train", num_workers=0)
    return loaders


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(to_device(item, device) for item in value)
    return value


def unpack_batch(batch, device):
    if isinstance(batch, dict) and "inputs" in batch and "target" in batch:
        inputs, target = batch["inputs"], batch["target"]
    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
        inputs, target = batch
    else:
        raise ValueError("A batch must be (inputs, target), or {'inputs': ..., 'target': ...}.")
    return to_device(inputs, device), to_device(target, device)


def call_model(model, inputs, capture=False, overrides=None):
    context = {"_nnmind_context": True} if capture else {}
    if overrides:
        context["_nnmind_overrides"] = overrides
    if isinstance(inputs, dict):
        return model(**inputs, **context)
    if isinstance(inputs, (tuple, list)):
        return model(*inputs, **context)
    return model(inputs, **context)


def output_shapes(value):
    if isinstance(value, torch.Tensor):
        return str(list(value.shape))
    if isinstance(value, (list, tuple)):
        return ", ".join(output_shapes(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(str(name) + ": " + output_shapes(item) for name, item in value.items())
    return type(value).__name__


def model_parameters(model):
    layers = {}
    for node, name in getattr(model, "__nnmind_nodes__", {}).items():
        module = getattr(model, name)
        parameters = []
        for parameter_name, value in module.named_parameters():
            values = value.detach().real.flatten()[:8].cpu().tolist()
            parameters.append(dict(name=parameter_name, shape=list(value.shape), count=value.numel(),
                                   trainable=value.requires_grad,
                                   sample=[item if math.isfinite(item) else None for item in values]))
        layers[node] = parameters
    return layers


def scalar_loss(criterion, output, target):
    value = criterion(output, target)
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError("compute_loss(prediction, target) must return one scalar tensor.")
    result = value.item()
    if not math.isfinite(result):
        raise ValueError("LOSS became NaN or infinite.")
    return value, result


def run_epoch(model, loader, criterion, device, optimizer=None, contextual=False):
    model.train(optimizer is not None)
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            inputs, target = unpack_batch(batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = call_model(model, inputs, contextual)
            if contextual:
                prediction, tensors = output
                loss, result = scalar_loss(lambda p, t: criterion(p, t, tensors=tensors, inputs=inputs), prediction, target)
            else:
                loss, result = scalar_loss(criterion, output, target)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            weight = max(1, int(target.shape[0])) if isinstance(target, torch.Tensor) and target.ndim else 1
            total += result * weight
            count += weight
    if not count:
        raise ValueError("A dataset split produced no batches.")
    return total / count


def graph_input(value, device):
    if isinstance(value, dict) and set(value) <= {"tensor", "dtype"} and "tensor" in value:
        dtype = value.get("dtype", "float32")
        if dtype not in ("float32", "float16", "int64", "bool"):
            raise ValueError("Unsupported tensor dtype: " + str(dtype))
        return torch.tensor(value["tensor"], dtype=getattr(torch, dtype), device=device)
    if isinstance(value, dict):
        return {key: graph_input(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [graph_input(item, device) for item in value]
    return value


def run_result(value, depth=0):
    if depth > 6:
        return "…"
    if isinstance(value, torch.Tensor):
        sample = value.detach().real.flatten()[:16].cpu().tolist()
        return dict(shape=list(value.shape), dtype=str(value.dtype),
                    sample=[item if math.isfinite(item) else None for item in sample])
    if isinstance(value, (list, tuple)):
        return [run_result(item, depth + 1) for item in value[:16]]
    if isinstance(value, dict):
        return {str(key): run_result(item, depth + 1) for key, item in list(value.items())[:32]}
    return str(value)[:4000]


def generate_text(config, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = config["path"]
    options = dict(local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(path, **options)
    model = AutoModelForCausalLM.from_pretrained(path, use_safetensors=True,
             dtype=getattr(torch, config["precision"]), low_cpu_mem_usage=True, **options)
    if config.get("adapter"):
        from peft import PeftModel
        from pathlib import Path
        adapter = Path(config["adapter"])
        if not (adapter / "adapter_config.json").is_file() or not (adapter / "adapter_model.safetensors").is_file():
            raise ValueError("Select the adapter folder containing adapter_config.json and adapter_model.safetensors.")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False, local_files_only=True)
    model = model.to(device).eval()
    prompt = config["prompt"]
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                               tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False, add_special_tokens=not bool(tokenizer.chat_template)).to(device)
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=config["tokens"], do_sample=False,
                             pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    return tokenizer.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def load_input_script(path):
    origin = path or ""
    staged = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.py")
    if origin:
        os.chdir(os.path.dirname(origin))
    script = source_module(staged, origin, "nnmind_data")
    return script


def canvas_inputs(script, device, prefer_dataset=False):
    if not prefer_dataset and callable(getattr(script, "build_inputs", None)):
        return to_device(script.build_inputs(), device)
    splits = script.build_datasets()
    if isinstance(splits, dict):
        validation = splits.get("validation", splits.get("val"))
    elif isinstance(splits, (tuple, list)) and len(splits) == 3:
        validation = splits[1]
    else:
        validation = None
    if validation is None:
        raise ValueError("Add a validation split or define build_inputs() on the Input node.")
    loader = validation if isinstance(validation, DataLoader) else DataLoader(validation, batch_size=1)
    first = next(iter(loader), None)
    if first is None:
        raise ValueError("Validation dataset is empty.")
    return unpack_batch(first, device)[0]


def inspect_inputs(config):
    script = load_input_script(config.get("datasetPath"))
    prefer_dataset = config.get("preferDataset", False)
    inputs = canvas_inputs(script, torch.device("cpu"), prefer_dataset)
    ports = config["ports"]
    names = [port["name"] for port in ports]
    if isinstance(inputs, dict):
        missing = [name for name in names if name not in inputs]
        extra = [str(name) for name in inputs if name not in names]
        if missing or extra:
            raise ValueError(f"Script inputs do not match the canvas. Missing: {missing}; extra: {extra}.")
        values = [inputs[name] for name in names]
    elif isinstance(inputs, (tuple, list)):
        values = inputs
    else:
        values = [inputs]
    if len(values) != len(names):
        raise ValueError(f"The script returned {len(values)} inputs; the canvas expects {len(names)}.")

    def spec(value):
        return dict(dtype=str(value.dtype).removeprefix("torch."), shape=list(value.shape))

    result = []
    for port, value in zip(ports, values):
        if isinstance(value, torch.Tensor):
            item = dict(name=port["name"], kind="tensor", **spec(value))
        else:
            item = dict(name=port["name"], kind=type(value).__name__)
        if port.get("transform"):
            context = {"torch": torch, "nn": nn, "F": nn.functional, "__name__": "nnmind_input_transform"}
            exec(compile(port["transform"], "<input transform>", "exec"), context)
            with torch.inference_mode():
                transformed = context["NNMindImplementation"]().eval()(value)
            if isinstance(transformed, torch.Tensor):
                item["forwarded"] = spec(transformed)
            else:
                item["forwardedKind"] = type(transformed).__name__
        result.append(item)
    report("input-specs", source="direct" if not prefer_dataset and callable(getattr(script, "build_inputs", None)) else "validation", inputs=result)


def decode_canvas(model, inputs, descriptor, max_tokens):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(descriptor["tokenizer"], local_files_only=True, trust_remote_code=False)
    result, values = call_model(model, inputs, capture=True)
    logits = values[descriptor["output"]]
    names = descriptor["inputs"]
    envelope = values[names[0]]
    if isinstance(envelope, dict):
        ids, mask = envelope["input_ids"], envelope["attention_mask"]
    else:
        ids, mask = envelope, values[names[1]]
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or ids.ndim != 2 or ids.shape != mask.shape:
        raise ValueError("Text generation needs token IDs and mask [B,T] and CausalLM logits [B,T,V].")
    eos = tokenizer.eos_token_id
    finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
    generated = []
    for step in range(max_tokens):
        if step == 0:
            positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
            last = positions.masked_fill(mask == 0, -1).amax(dim=1)
            if (last < 0).any():
                raise ValueError("Every prompt must have at least one non-padding token.")
            scores = logits[torch.arange(logits.shape[0], device=logits.device), last]
        else:
            scores = logits[:, -1]
        next_id = scores.argmax(dim=-1)
        if eos is not None:
            next_id = torch.where(finished, eos, next_id)
        generated.append(next_id)
        token_mask = ~finished
        if eos is not None:
            finished |= next_id == eos
        if finished.all() or step + 1 == max_tokens:
            break
        ids = torch.cat((ids, next_id[:, None]), dim=1)
        mask = torch.cat((mask, token_mask.to(mask.dtype)[:, None]), dim=1)
        overrides = {names[0]: {**envelope, "input_ids": ids, "attention_mask": mask}} if isinstance(envelope, dict) else {names[0]: ids, names[1]: mask}
        _, values = call_model(model, inputs, capture=True, overrides=overrides)
        logits = values[descriptor["output"]]
    words = tokenizer.batch_decode(torch.stack(generated, dim=1).cpu().tolist(), skip_special_tokens=True)
    return "\n\n".join(words), result


def main():
    with open("settings.json", encoding="utf-8") as settings:
        config = json.load(settings)
    torch.manual_seed(42)
    if config["mode"] == "inspect-inputs":
        inspect_inputs(config)
        return
    device_name = config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in the selected PyTorch environment.")
    device = torch.device(device_name)
    if config["mode"] == "generate":
        report("result", text=generate_text(config["generation"], device))
        return
    model_module = source_module(os.path.abspath("model.py"), "", "nnmind_generated_model")
    model = model_module.GeneratedNet().to(device)
    if config["mode"] == "execute":
        if config.get("datasetPath") or not config.get("inputJson"):
            inputs = canvas_inputs(load_input_script(config.get("datasetPath")), device)
        else:
            inputs = graph_input(json.loads(config["inputJson"]), device)
        model.eval()
        with torch.inference_mode():
            causal = getattr(model, "__nnmind_causal__", [])
            if causal and config.get("canvasTokens"):
                decoded, output = decode_canvas(model, inputs, causal[0], config["canvasTokens"])
                report("result", text=decoded, details=json.dumps(run_result(output), ensure_ascii=False, indent=2))
            else:
                output = call_model(model, inputs)
                report("result", text=json.dumps(run_result(output), ensure_ascii=False, indent=2))
        report("parameters", layers=model_parameters(model))
        return
    data = load_input_script(config.get("datasetPath"))
    origin = config.get("datasetPath")
    loaders = dataset_loaders(data.build_datasets(), config["batchSize"])
    model.eval()
    with torch.no_grad():
        first = next(iter(loaders["validation"]), None)
        if first is None:
            raise ValueError("Validation dataset is empty.")
        inputs, _ = unpack_batch(first, device)
        output = call_model(model, inputs)
    report("ready", device=str(device), output=output_shapes(output))
    report("parameters", layers=model_parameters(model))
    if config["mode"] == "preview":
        return

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    losses = {"mse": nn.MSELoss, "crossEntropy": nn.CrossEntropyLoss, "l1": nn.L1Loss, "bce": nn.BCEWithLogitsLoss}
    contextual = config["loss"] == "custom"
    if contextual:
        function = source_module(os.path.abspath("loss.py"), "", "nnmind_loss").compute_loss
        signature = inspect.signature(function)
        accepts_keywords = any(item.kind == inspect.Parameter.VAR_KEYWORD for item in signature.parameters.values())
        def criterion(prediction, target, **context):
            keywords = {key: value for key, value in context.items() if accepts_keywords or key in signature.parameters}
            return function(prediction, target, **keywords)
    elif config["loss"] == "causal":
        def criterion(prediction, target):
            if prediction.ndim != 3 or target.ndim != 2 or prediction.shape[1] < 2:
                raise ValueError("Causal LM LOSS requires logits [B,T,V], labels [B,T], and at least two tokens.")
            return nn.functional.cross_entropy(prediction[:, :-1].transpose(1, 2).float(), target[:, 1:].long(), ignore_index=-100)
    else:
        criterion = losses[config["loss"]]().to(device)
    if origin:
        os.chdir(os.path.dirname(origin))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("This model has no trainable parameters.")
    optimizer_type = torch.optim.Adam if config["optimizer"] == "adam" else torch.optim.SGD
    optimizer = optimizer_type(parameters, lr=config["learningRate"])
    for epoch in range(1, config["epochs"] + 1):
        train = run_epoch(model, loaders["train"], criterion, device, optimizer, contextual)
        validation = run_epoch(model, loaders["validation"], criterion, device, contextual=contextual)
        report("epoch", epoch=epoch, train=train, validation=validation)
    report("test", loss=run_epoch(model, loaders["test"], criterion, device, contextual=contextual))
    report("parameters", layers=model_parameters(model))
    if config.get("savePath"):
        adapters = [(name, module) for name, module in model.named_children()
                    if callable(getattr(module, "save_adapter", None)) and any(p.requires_grad for p in module.parameters())]
        if adapters:
            path = tempfile.mkdtemp(prefix="nnmind-adapters-", dir=config["savePath"])
            for name, module in adapters:
                module.save_adapter(os.path.join(path, name))
            report("log", message="LoRA adapters saved: " + path)


if __name__ == "__main__":
    main()
