import json

from calibre.utils.localization import _  # type: ignore

from .openai import ChatgptTranslate

load_translations()  # type: ignore


class DeepseekTranslate(ChatgptTranslate):
    name = 'DeepSeek'
    alias = 'DeepSeek (Chat)'
    endpoint = 'https://api.deepseek.com/chat/completions'
    temperature = 1.3

    concurrency_limit = 0
    request_interval = 0.0

    # deepseek-chat and deepseek-reasoner were retired on 2026-07-24.
    models: list[str] = ['deepseek-flash', 'deepseek-v4-pro']
    model: str | None = models[0]

    # Reasoning is opt-in and disabled by default.
    thinking: bool = False

    def __init__(self):
        super().__init__()
        self.model = self.config.get('model', self.model)
        self.thinking = bool(self.config.get('thinking', self.thinking))

    def get_models(self):
        return self.models

    def get_body(self, text):
        body = json.loads(super().get_body(text))
        body.update(thinking={
            'type': 'enabled' if self.thinking else 'disabled'})
        return json.dumps(body)

    def get_result(self, response):
        if self.stream:
            return self._parse_stream(response)
        try:
            data = json.loads(response)
            # Only the message content holds the translation. The
            # reasoning_content field, when present, is intentionally
            # ignored.
            content = data['choices'][0]['message']['content']
            if isinstance(content, str):
                return content
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            pass
        raise Exception(
            _('Can not parse returned response. Raw data: {}')
            .format(response[:500] + '...' if len(response) > 500
                    else response))

    def _extract_stream_content(self, data):
        """Return the translated text of a chunk, ignoring reasoning."""
        try:
            delta = data['choices'][0]['delta']
        except (KeyError, IndexError, TypeError):
            return None
        # The reasoning_content field of the delta is intentionally
        # ignored, so that only the translation is streamed out.
        content = delta.get('content') if isinstance(delta, dict) else None
        if content and isinstance(content, str):
            return content
        return None
