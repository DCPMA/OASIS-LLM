# Prompt template (paper-verbatim, Kurdi et al. 2017 Appendix 1)

> Snapshot from `src/oasis_llm/prompts.py` at git `d08914f3`.
> Image-focused variant (the study used this; the internal-state variant is not used).
> One image per call, one dimension per call, base64-embedded image as the user message.
> Structured output schema (JSON): `rating` (1-7 int) + optional `reasoning` (<=30 word rationale).

## Valence — system prompt

```text
In this study you will be presented with a series of images. We are interested in the affective response that these images evoke. The dimension that we are asking you to rate is VALENCE.

Valence refers to the level of positivity or negativity intrinsic to an image. At one extreme of the valence scale, an image is very negative; at the other extreme, an image is very positive.

You will rate each image on a 7-point scale with the following labels:
1 = Very negative
2 = Moderately negative
3 = Somewhat negative
4 = Neutral
5 = Somewhat positive
6 = Moderately positive
7 = Very positive

Please respond with a single integer from 1 to 7.
```

## Valence — user message

```text
Please rate the VALENCE of this image on the 1-7 scale.
```

## Arousal — system prompt

```text
In this study you will be presented with a series of images. We are interested in the affective response that these images evoke. The dimension that we are asking you to rate is AROUSAL.

Arousal refers to the level of intensity or excitement intrinsic to an image. At one extreme of the arousal scale, an image is very low in arousal (calm, relaxing, sleepy); at the other extreme, an image is very high in arousal (stimulating, exciting, frenzied).

Note that arousal is orthogonal to valence: a calm scene can be either positive (a peaceful beach) or negative (a foggy graveyard), and an exciting scene can be either positive (a celebration) or negative (a violent confrontation). Rate arousal independently of valence.

You will rate each image on a 7-point scale with the following labels:
1 = Very low
2 = Moderately low
3 = Somewhat low
4 = Neither low nor high
5 = Somewhat high
6 = Moderately high
7 = Very high

Please respond with a single integer from 1 to 7.
```

## Arousal — user message

```text
Please rate the AROUSAL of this image on the 1-7 scale.
```

## Structured output schema

```json
{
  "type": "object",
  "properties": {
    "rating": {
      "type": "integer",
      "minimum": 1,
      "maximum": 7,
      "description": "Integer rating from 1 to 7"
    },
    "reasoning": {
      "type": "string",
      "description": "Brief one-sentence rationale (<=30 words)."
    }
  },
  "required": [
    "rating"
  ],
  "additionalProperties": false
}
```
