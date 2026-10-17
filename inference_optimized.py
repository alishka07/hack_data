"""Reuse only image preprocessing/backbone for prompts on the SAME frame.

The Grounding DINO fusion encoder still runs for each prompt. Nothing is shared
across frames. This preserves prompts, sampling frequency and FP32 inference.
"""
from contextlib import contextmanager

VERSION = 'same-frame-backbone-v2'


@contextmanager
def cached_backbone(backbone):
    had_instance_forward = 'forward' in backbone.__dict__
    instance_forward = backbone.__dict__.get('forward')
    original = backbone.forward
    cached = None

    def forward(*args, **kwargs):
        nonlocal cached
        if cached is None:
            features, positions = original(*args, **kwargs)
            cached = (tuple(features), tuple(positions))
        # GroundingDinoModel.forward appends extra levels to the positions list.
        return list(cached[0]), list(cached[1])

    backbone.forward = forward
    try:
        yield
    finally:
        if had_instance_forward:
            backbone.forward = instance_forward
        else:
            delattr(backbone, 'forward')
        cached = None


class PromptDetector:
    def __init__(self, loaded, fallback):
        self.processor, self.model, self.device = loaded
        self.loaded, self.fallback = loaded, fallback
        self.text_cache = {}
        config = getattr(self.model, 'config', None)
        self.supported = (getattr(config, 'model_type', None) == 'grounding-dino'
                          and not self.model.training
                          and hasattr(getattr(self.model, 'model', None), 'backbone'))

    def detect_groups(self, image, prompts, threshold):
        if not self.supported:
            return {key: self.fallback(image, self.loaded, prompt, threshold)
                    for key, prompt in prompts.items()}
        import torch
        with torch.inference_mode():
            image_inputs = self.processor(images=image, return_tensors='pt').to(self.device)
            grouped = {}
            with cached_backbone(self.model.model.backbone):
                for key, prompt in prompts.items():
                    if prompt not in self.text_cache:
                        self.text_cache[prompt] = self.processor(text=prompt, return_tensors='pt').to(self.device)
                    inputs = dict(image_inputs)
                    inputs.update(self.text_cache[prompt])
                    outputs = self.model(**inputs)
                    result = self.processor.post_process_grounded_object_detection(
                        outputs, inputs['input_ids'], threshold=threshold, text_threshold=.25,
                        target_sizes=[(image.height, image.width)])[0]
                    grouped[key] = [dict(label=label, box=box.cpu().tolist(), score=float(score))
                                    for label, box, score in zip(result['text_labels'], result['boxes'], result['scores'])]
                    # Do not keep the previous fusion output alive while the next
                    # prompt allocates its encoder/decoder activations.
                    del outputs, result, inputs
            del image_inputs
            # Release fragmented cached allocations between source frames. This
            # matters on smaller Windows GPUs sharing VRAM with the desktop.
            if str(self.device).startswith('cuda'):
                torch.cuda.empty_cache()
            return grouped
