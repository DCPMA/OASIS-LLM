# Agent Start Here

This workspace is an OASIS study bundle: the published paper, an in-press manuscript, the participant- and trial-level data, the original R analysis script, the image assets, and a local mock of the experiment flow.

## Fastest Entry Points

- Experiment setup and procedure: [OASIS/Kurdi_BRM_2017.02-method.txt](OASIS/Kurdi_BRM_2017.02-method.txt)
- Published paper overview: [OASIS/Kurdi_BRM_2017.01-abstract-and-intro.txt](OASIS/Kurdi_BRM_2017.01-abstract-and-intro.txt)
- Published paper conclusion and resource notes: [OASIS/Kurdi_BRM_2017.04-conclusion.txt](OASIS/Kurdi_BRM_2017.04-conclusion.txt)
- Paper appendices and instruction text: [OASIS/Kurdi_BRM_2017.05-appendices.txt](OASIS/Kurdi_BRM_2017.05-appendices.txt)
- Earlier manuscript version: [OASIS/OASIS_Inpress.02-method.txt](OASIS/OASIS_Inpress.02-method.txt)
- Dataset codebook: [data/raw/OASIS_codebook.txt](data/raw/OASIS_codebook.txt)
- Participant-level data: [data/raw/OASIS_data.csv](data/raw/OASIS_data.csv)
- Trial-level data: [data/derived/OASIS_data_long.csv](data/derived/OASIS_data_long.csv)
- Analysis script: [scripts/OASIS.R](scripts/OASIS.R)
- Local image assets: [OASIS/images](OASIS/images)
- Local experiment mockup: [site/oasis-original-flow/index.html](site/oasis-original-flow/index.html)

## Reconstructed Main Study

- 900 open-access color images were standardized to 500 x 400 pixels.
- Images were split into 4 lists of 225 images each.
- Participants were recruited on U.S. Amazon Mechanical Turk, restricted to workers with at least 90% approval and at least 50 completed HITs.
- Final usable sample: 822 participants.
- Each participant rated only one dimension: valence or arousal.
- The main study used only image-centered instructions.
- Images were shown in individually randomized order.
- Ratings used a 7-point Likert scale.
- After the image ratings, participants completed a demographic questionnaire.

## Data Layout

- [data/raw/OASIS_data.csv](data/raw/OASIS_data.csv): 822 rows, one row per participant.
- Columns before the image ratings: `ID`, `List`, `Condition`, `Gender`, `Age`, `Ethnicity`, `Race`, `Ideol`, `Income`, `Education`, `valar`.
- Image rating columns in [data/raw/OASIS_data.csv](data/raw/OASIS_data.csv): `I1` through `I900`.
- [data/derived/OASIS_data_long.csv](data/derived/OASIS_data_long.csv): 739800 rows, trial-level long format.
- Core long-format columns: participant metadata, `valar`, image ID, `rating`, `theme`, and `category`.
- The long file contains 822 participants x 900 image rows.

## Practical Access Notes

- [data/derived/OASIS_data_long.csv](data/derived/OASIS_data_long.csv) is large enough that editor sync/read tools may fail. Prefer terminal or `Rscript` sampling over whole-file reads.
- In the main dataset, `Condition` is already image-centered and matches the rated dimension.
- Observed condition counts in [data/raw/OASIS_data.csv](data/raw/OASIS_data.csv): 413 valence, 409 arousal.
- Observed list counts in [data/raw/OASIS_data.csv](data/raw/OASIS_data.csv): List 1 = 203, List 2 = 204, List 3 = 212, List 4 = 203.
- The extracted text files in [OASIS](OASIS) were produced from the local PDFs with `pdftotext` and split by section headings for faster agent lookup.

## Suggested Starting Points By Task

- If you need the study method, start with [OASIS/Kurdi_BRM_2017.02-method.txt](OASIS/Kurdi_BRM_2017.02-method.txt).
- If you need variable meanings, start with [data/raw/OASIS_codebook.txt](data/raw/OASIS_codebook.txt).
- If you need analysis logic, start with [scripts/OASIS.R](scripts/OASIS.R).
- If you need actual stimuli for a UI or QA flow, start with [OASIS/images](OASIS/images).
- If you need to experience the procedure quickly, open [site/oasis-original-flow/index.html](site/oasis-original-flow/index.html).
