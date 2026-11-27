# Digikala Vectorized Datasets

Two datasets for semantic search across 1,283,496 product rows and 6,156,289 comment rows. Each release pairs 768-dimensional Yalda vectors with zero-based indices into the original CSV.

## Download and run

| Vectorized dataset | Build and search notebook | Array size |
| --- | --- | ---: |
| [Products](https://drive.google.com/file/d/1H9YIchZXzkdQXKIQCXMvryQ3TNPrshV2/view?usp=sharing) | [digikala_products_yalda.ipynb](digikala_products_yalda.ipynb) | 1.00 GB |
| [Comments](https://drive.google.com/file/d/1-3ciuF0SpfB1eDoL3SUd6aiWJPK-3LHo/view?usp=sharing) | [digikala_comments_yalda.ipynb](digikala_comments_yalda.ipynb) | 4.81 GB |

Open the corresponding notebook in Google Colab, keep `MODE = "use"`, set `RUN_SEARCH_EXAMPLE = True`, and run the cells in order. The notebook loads the release into the cloud runtime and shows the top five results for each of five sample queries, with cosine scores and source metadata. To use a copy from mounted Drive, set `VECTOR_FILE` to its path.

## Design decisions

- **Model and budget.** Initial preparation estimated about 460 million input tokens, which exceeded the project's embedding API budget. Running Yalda locally in Colab let us embed the selected fields with their original detail and avoid errors introduced by summarization. Yalda is trained for Persian semantic search, reports strong results on FaMTEB and has about 596 million parameters, making it practical to run on a free Colab GPU. [Yalda model card](https://huggingface.co/Yaldat/Yalda-Embedding)

- **Products.** The Persian title, brand and category hierarchy describe what the product is. The values in `title_fa`, `Brand`, `Category1`, `Category2` and `sub_category` are joined with `|`. On 10,000 products and 10 queries with Yalda, omitting field labels cut tokens by about **28%**. Category-based MRR@10 was 1.00 in both variants.

- **Comments.** The title, body, advantages and disadvantages are joined with `|` and labeled in Persian. The labels make the role of each field explicit, including whether a feature is a strength or a weakness.

- **Normalization and token limits.** Before tokenization, the pipeline applies Unicode NFC, standardizes Persian letters and digits, cleans whitespace and zero-width non-joiners, strips control characters, and flattens advantages and disadvantages lists. Empty fields are skipped. Stopword removal is disabled to preserve negation and tone. In the token audit, the **128-token** product limit and **512-token** comment limit fully covered more than **99.99%** and **99.93%** of valid sampled texts, respectively.

- **Index and metadata.** `source_row` is the zero-based position of the parsed data row. The original CSV provides identifiers for joins and grouping and fields such as price, ratings, buyer status and date for display and filtering. Keeping these fields in the CSV avoids duplication in the vector files. The row index identifies the exact record even when IDs or text repeat. [Pinned source CSVs](https://huggingface.co/datasets/RadeAI/Digikala_comments_products/tree/89c3133b169c8d3793db8834f56f32fee33d9db0)

- **Dimensions and storage.** Yalda ran in FP16. The first 768 of 1,024 output dimensions were retained and L2-normalized in FP32. Each row uses INT8 codes and an FP32 scale. In the comparison on vectors from the initial embedding API pilot, all five manually written queries kept the same top-five results and order. The INT8 format reduced the estimated array footprint, including row indices and validity flags, from **11.49 GB to 5.81 GB** at 768 dimensions, a **49.45%** reduction.

## Format and integration

Each NPZ contains `vectors` (INT8 codes), `source_row` (Int64 indices), `scales` (FP32 row scales) and `valid` (boolean flags). For cosine search, reconstruct each valid vector by multiplying its codes by its row scale and applying L2 normalization. Embed queries with the same pinned Yalda revision and its official `query` prompt, keep the first 768 dimensions and L2-normalize them. The example cells perform these steps in blocks. Keep the original CSV row order for metadata lookups.

Yalda query embedding and vector search run without an API key. The project's downstream language-model API requires requests from an **Iranian IP address**. If the Colab runtime cannot reach the gateway, call it from the team's environment with an Iranian IP address.

[Full report and figures](digikala_vectorization_report.pdf)
