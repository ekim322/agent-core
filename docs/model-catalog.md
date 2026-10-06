# Model catalog

`ModelId` is a convenience shortlist of general-purpose API models across the
three built-in providers. It covers capability, everyday use and lower-cost
experiments. Applications can pass explicit model strings for other models;
this enum is not a runtime allowlist. Aliases can change underlying versions,
and access depends on provider and account availability.

| Provider | `ModelId` members | Selection |
| --- | --- | --- |
| OpenAI | `GPT_6_ASTRA`, `GPT_6_1_SOL`, `GPT_6_LUNA` | Capability, balanced and economical tiers |
| Anthropic | `CLAUDE_FABLE_5_1`, `CLAUDE_OPUS_5_5`, `CLAUDE_SONNET_5_5`, `CLAUDE_HAIKU_4_5` | Advanced reasoning, agentic work, speed/intelligence balance and fast calls |
| Google | `GEMINI_3_8_FLASH`, `GEMINI_3_5_FLASH_LITE`, `GEMINI_3_1_PRO_PREVIEW` | Current Flash, economical Flash-Lite and a Pro comparison option |

Gemini 3.1 Pro is a preview model. Haiku 4.5 is Anthropic's fast option.
Specialized image, audio, embedding and restricted-access models are outside
this shortlist. Omission does not imply that a model is deprecated.

## Official sources

Provider model catalogs and lifecycle notices are the source of truth for current
availability and status:

- [OpenAI model catalog](https://developers.openai.com/api/docs/models) and
  [API changelog](https://developers.openai.com/api/docs/changelog): current
  model IDs and announcements.
- [Claude model overview](https://platform.claude.com/docs/en/models/overview)
  and [Sonnet 5.5 announcement](https://www.anthropic.com/claude-sonnet-5-5):
  current public API IDs and model details.
- [Gemini model catalog](https://ai.google.dev/gemini-api/docs/models) and
  [lifecycle table](https://ai.google.dev/gemini-api/docs/deprecations): endpoint
  IDs, preview status and lifecycle information.
