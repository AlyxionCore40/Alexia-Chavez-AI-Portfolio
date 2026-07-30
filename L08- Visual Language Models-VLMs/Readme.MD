Lab 08: Vision Transformers and VLMs — Talking to Your Images

Course: ITAI 1378 — Module 08 Model: SmolVLM-500M-Instruct (Hugging Face) Platform: Google Colab

Problem Statement

Vision-Language Models (VLMs) let a single model both "see" an image and reason about it in natural language, replacing what used to require a separate specialist model for every vision task (classification, counting, reading text, etc.). This lab explores how a VLM actually works under the hood. Starting from how a Vision Transformer breaks an image into patches, through prompting one small open model to perform multiple jobs, into real workforce use cases (accessibility alt text, chart reading), and finally into the model's failure modes (hallucination) and deployment tradeoffs.

Approach
Patch visualization- Sliced a custom image into ViT-style patch grids at two patch sizes (48px → 64 patches, 96px → 16 patches) to see directly how "an image becomes a sentence of visual words," before touching any language model.
Loading and prompting the VLM — Loaded SmolVLM-500M-Instruct via Hugging Face transformers, and built a reusable ask(image, question) helper wrapping the model's chat template, generation, and response parsing.
Interviewing an image — Asked the same image three questions of increasing difficulty (factual, detail, reasoning/prediction) to compare answer quality across question types.
Job switching — Used prompt engineering alone (no retraining) to make the same model act as a classifier, a counter, and a reader/describer, demonstrating that "the prompt selects the job."
Workforce Task A — Accessibility alt text — Compared a weak prompt ("Describe this image") against a professional prompt encoding real alt-text rules (concise, specific, never starts with "an image of") on two different images.
Workforce Task B — Chart reading — Built a bar chart with known ground-truth values, then tested whether the VLM could correctly extract facts from it (highest value, a specific number, a trend direction).
Hallucination hunt — Designed two deliberate "trap" questions with false premises (a nonexistent second cat, a nonexistent printed year) to see whether the model would push back on a false assumption or fabricate an answer to satisfy it.
Agent preview — Reused the same ask() function with a decision-formatted prompt to preview how VLM perception feeds into agentic decision-making (Module 09).
Deployment audit — An open-ended reliability test of the VLM in an education scenario (describing a chart for students), evaluating what worked, what failed, and forming a go/no-go judgment.
Results

Interview an image (increasing difficulty):

Question type	Result
Easy factual	Correctly identified a black-and-white cat resting on a wooden surface
Detail	Correctly identified color/position; background description was vaguer ("green background in the background")
Reasoning/prediction	Plausible-sounding but unverifiable prediction about what the cat might do next

Chart reading (against known ground truth):

Question	Model answer	Ground truth	Correct?
Highest sales month	Jun	Jun	✅
March units sold	140	140	✅
April→May trend	Up	Down (180 → 165)	❌

The model got 2 of 3 chart questions right and answered the trend-direction question incorrectly, misreading a decrease as an increase — a useful, concrete example of the kind of subtle factual error a VLM can produce confidently on numeric/visual data.

Hallucination hunt (trap questions with false premises):

"What breed is the second cat in this picture?" (only one cat exists) — the model fabricated an answer on the cat image ("The second cat is a black and white cat") instead of correcting the false premise, though it correctly avoided the trap on the chart image ("breed... is not mentioned").
"What year is printed at the bottom of this chart?" (no year is printed) — the model correctly said no chart/year was present when shown the cat image, but on the chart image it answered "Jun" — a nonsensical, fabricated response to a premise that was still false (no year is printed anywhere on the chart).
Key Findings
Smaller ViT patches (48px) capture more distinguishable visual detail per "token," but scale up the number of patches the model has to process — since attention cost grows roughly with the square of patch count, this is a real detail-vs-compute tradeoff, not just a display setting.
A single small (500M parameter) open VLM can perform multiple distinct vision tasks — classification, counting, description — purely by changing the prompt, which is the core practical advantage VLMs have over maintaining separate specialist models for each task; a dedicated detector like YOLO still wins on speed and localization precision (bounding boxes) for object-finding tasks specifically.
Output-format constraints ("answer with just a number," "one word only") are what make a VLM usable as a software component rather than a chatbot — without them, wiring a VLM into a pipeline reliably breaks.
The model's factual reliability was uneven even on a simple, controlled chart: it got 2 of 3 fact-extraction questions exactly right but misread a clear downward trend as "up," showing that even confident-sounding chart answers need independent verification before being trusted in a real report.
Hallucination is a distinct and more dangerous failure mode than being simply wrong: when asked about a second cat that doesn't exist, the model fabricated a description rather than saying "there is only one cat in this image" — a fabricated detail is harder to catch than a wrong number, because it doesn't look like an error at all.
Technologies Used
Python 3
Hugging Face Transformers (AutoProcessor, AutoModelForImageTextToText) — loading and running SmolVLM-500M-Instruct
PyTorch — underlying model execution
Pillow (PIL) — image loading and patch slicing
Matplotlib — patch grid visualization and the ground-truth sales chart used for chart-reading tests
Google Colab — development/execution environment
How to Run
Open the notebook in Google Colab.
Run all cells top to bottom. The setup cell installs transformers and accelerate and auto-detects GPU vs. CPU; SmolVLM-500M-Instruct downloads automatically from Hugging Face on first use.
To test with your own image, upload it via the Colab file-folder icon and update the relevant image variable (img2, etc.) with its path.
All model responses print inline as each cell runs.

To run locally instead:

bash
pip install transformers accelerate torch pillow matplotlib
jupyter notebook L08_Chavez_Alexia_ITAI1378.ipynb

Requires Python 3.8+. A GPU is recommended but not required — SmolVLM-500M-Instruct runs on CPU as well.

Full reflection question answers (Section 10, 10 questions) and the deployment audit memo are included in the notebook itself.
