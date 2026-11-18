export const modalityDefinitions = {
  Tokenizer: `class NNMindTokenizer(nn.Module):
    def __init__(self, path, max_length):
        super().__init__()
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.max_length = max_length
        self.register_buffer("placement", torch.empty(0), persistent=False)

    def forward(self, text):
        if isinstance(text, (tuple, list)) and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=self.max_length, padding=isinstance(text, (tuple, list)))
        return {key: encoded[key].to(self.placement.device) for key in ("input_ids", "attention_mask")}`,
  VisionEncoder: `class NNMindVisionEncoder(nn.Module):
    def __init__(self, path):
        super().__init__()
        from transformers import AutoConfig, AutoImageProcessor, AutoModel, CLIPVisionModel
        options = dict(local_files_only=True, trust_remote_code=False)
        self.processor = AutoImageProcessor.from_pretrained(path, **options)
        config = AutoConfig.from_pretrained(path, **options)
        loader = CLIPVisionModel if config.model_type == "clip" else AutoModel
        self.encoder = loader.from_pretrained(path, use_safetensors=True, **options)

    def forward(self, image_path):
        from PIL import Image
        def load(path):
            with Image.open(path) as image:
                return image.convert("RGB")
        paths = image_path if isinstance(image_path, (list, tuple)) else [image_path]
        batch = self.processor(images=[load(path) for path in paths], return_tensors="pt")
        return self.encoder(**batch.to(next(self.encoder.parameters()).device)).last_hidden_state`,
  AudioEncoder: `class NNMindAudioEncoder(nn.Module):
    def __init__(self, path):
        super().__init__()
        from transformers import AutoFeatureExtractor, AutoModel
        options = dict(local_files_only=True, trust_remote_code=False)
        self.processor = AutoFeatureExtractor.from_pretrained(path, **options)
        self.encoder = AutoModel.from_pretrained(path, use_safetensors=True, **options)
        self.encoder = self.encoder.encoder if self.encoder.config.model_type == "whisper" else self.encoder

    def forward(self, audio_path):
        import wave
        import numpy as np
        with wave.open(str(audio_path), "rb") as audio:
            rate, channels, width = audio.getframerate(), audio.getnchannels(), audio.getsampwidth()
            if width != 2 or rate != self.processor.sampling_rate:
                raise ValueError(f"Use 16-bit PCM WAV at {self.processor.sampling_rate} Hz.")
            samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
        samples = samples.astype(np.float32).reshape(-1, channels).mean(axis=1) / 32768.0
        batch = self.processor(samples, sampling_rate=rate, return_tensors="pt")
        return self.encoder(**batch.to(next(self.encoder.parameters()).device)).last_hidden_state`,
} as const
