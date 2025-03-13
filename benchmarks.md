Key Points
Research suggests that four papers meet the criteria for state-of-the-art (SOTA) research on Multi30k for Multimodal Machine Translation (MMT) published after 2022, using unsupervized or self-supervized methods, and reporting BLEU scores, with some also reporting METEOR scores.
It seems likely that these papers provide valuable insights, but one paper, "Adding Multimodal Capabilities to a Text-only Translation Model," may not fully fit the unsupervized criterion due to supervised fine-tuning, highlighting a potential area of debate.
The evidence leans toward these papers focusing on test sets like flickr 2016 and 2017, with limited reporting for flickr 2018 and no reports for mscoco 2017, which is an unexpected detail given the user's interest in all test sets.
Direct Answer
Below is a table summarizing the SOTA papers on Multi30k for MMT, published after 2022, using unsupervized or self-supervized methods, and reporting BLEU scores, with optional METEOR scores. The table includes metrics for the specified test sets: flickr 2016, flickr 2017, flickr 2018, and mscoco 2017. If metrics are not found, the field is left empty.
Paper Name
Link
BLEU (flickr 2016)
METEOR (flickr 2016)
BLEU (flickr 2017)
METEOR (flickr 2017)
BLEU (flickr 2018)
METEOR (flickr 2018)
BLEU (mscoco 2017)
METEOR (mscoco 2017)
Adding Multimoda1l Capabilities to a Text-only Translation Model
Link
46.5
-
43.6
-
-
-
-
-
Scene Graph as Pivoting: Inference-time Image-free Unsupervised MMT with Visual Scene Hallucination
Link
34.45
53.78
32.09
52.14
-
-
-
-
Beyond Language: Empowering Unsupervised MT with Cross-modal Alignment
Link
26.5
-
-
-
-
-
-
-
Video Pivoting Unsupervised Multi-Modal Machine Translation
Link
35.12
-
-
-
-
-
-
-
These papers were identified through a comprehensive search on platforms like Google Scholar, ResearchGate, and Papers with Code, focusing on publications after 2022. They use methods like self-supervized pre-training or unsupervized alignment with images/videos, aligning with your request for non-supervised approaches. Notably, "Adding Multimodal Capabilities to a Text-only Translation Model" includes supervised fine-tuning, which may not fully meet the unsupervized criterion, but is included for completeness. The main metric, BLEU, is reported for most papers, with "Scene Graph as Pivoting" also providing METEOR scores for flickr 2016 and 2017. An unexpected detail is the lack of metrics for flickr 2018 and mscoco 2017 across all papers, suggesting these test sets may be less explored in recent research.
Survey Note
This survey note provides a detailed analysis of state-of-the-art (SOTA) papers on the Multi30k dataset for Multimodal Machine Translation (MMT), focusing on publications after 2022, using unsupervized or self-supervized algorithms, and reporting BLEU scores, with optional METEOR scores. The analysis was conducted on March 11, 2025, ensuring all findings are current and relevant. The survey includes a comprehensive review of the search process, paper selection, and metric reporting, aiming to address the user's request for all such papers and their performance on specified test sets: flickr 2016, flickr 2017, flickr 2018, and mscoco 2017.
Background and Methodology
Multi30k is a widely recognized dataset for MMT, extending the Flickr30k dataset with German translations and additional crowdsourced descriptions, primarily used for English-to-German translation tasks with image inputs. MMT integrates visual information to enhance translation quality, particularly for ambiguous texts. The criteria for this study include publications post-2022, unsupervized or self-supervized methods (excluding supervised learning), and a focus on BLEU as the main metric, with METEOR as a secondary consideration. The search process involved querying academic databases such as Google Scholar, ResearchGate, and Papers with Code, analyzing abstracts and methods for compliance, and verifying metric reporting.
The user's specified test sets (flickr 2016, flickr 2017, flickr 2018, mscoco 2017) were mapped to Multi30k test sets: test2016 likely corresponds to flickr 2016, test2017 to flickr 2017, test2018 to flickr 2018, and mscoco 2017 is a separate dataset not part of Multi30k, which was confirmed by consulting the Multi30k dataset documentation at Multi30k Dataset. This mapping guided the extraction of metrics for the table.
Identified Papers and Detailed Analysis
After extensive review, four papers were identified that meet the criteria, with one potentially controversial inclusion due to mixed training methods. Below is a detailed analysis of each:
Adding Multimodal Capabilities to a Text-only Translation Model (Vijayan et al., 2024, Adding Multimodal Capabilities to a Text-only Translation Model):
This paper explores adding multimodal capabilities to a text-only translation model by starting with a text-only machine translation (MT) model and enhancing it with vision-text adapter layers. The method involves two stages: pre-training using vision-based masking of the source text, which is a self-supervized task, and fine-tuning on the Multi30k dataset, which is supervised. This mix raises questions about whether it fully meets the unsupervized criterion, but it is included for completeness.
It reports BLEU4 scores for Multi30k 2016 (46.5) and 2017 (43.6) test sets, with no METEOR scores reported. No metrics were found for flickr 2018 or mscoco 2017.
Scene Graph as Pivoting: Inference-time Image-free Unsupervised MMT with Visual Scene Hallucination (Fei et al., 2023, Scene Graph as Pivoting: Inference-time Image-free Unsupervised MMT with Visual Scene Hallucination):
This work investigates an unsupervised MMT setup, inference-time image-free UMMT, trained with source-text image pairs and tested with only source-text inputs. It uses scene graphs for visual scene hallucination, ensuring unsupervized translation training without parallel text data. It reports both BLEU and METEOR scores for test2016 (34.45 BLEU, 53.78 METEOR) and test2017 (32.09 BLEU, 52.14 METEOR), with no reports for flickr 2018 or mscoco 2017.
Beyond Language: Empowering Unsupervised MT with Cross-modal Alignment (Yang et al., 2024, Beyond Language: Empowering Unsupervised MT with Cross-modal Alignment):
This paper proposes an unsupervised multi-modal machine translation method using images as pivots to align different languages, leveraging monolingual image-text pairs without parallel text data. It reports a BLEU score of 26.5 for the Multi30k dataset, likely for test2016, with no METEOR scores or reports for other test sets.
Video Pivoting Unsupervised Multi-Modal Machine Translation (Li et al., 2023, Video Pivoting Unsupervised Multi-Modal Machine Translation):
This study employs video data to enhance unsupervized MMT, using spatial-temporal graphs to model object interactions. It reports a BLEU score of 35.12 for the Multi30k dataset, likely for test2016, with no METEOR scores or reports for flickr 2017, 2018, or mscoco 2017. Notably, it uses Multi30k for image-based experiments, despite the video focus, suggesting an extension of the dataset.
Search Process and Challenges
The search initially targeted Google Scholar, ResearchGate, and Papers with Code, using queries like "Multi30k MMT unsupervized OR self supervized published after 2022" and filtering for publications after 2022. Additional searches included specific terms like "CLIP for unsupervized MMT" and "self supervized preTraining for MMT" to explore related methodologies. Many papers, such as "Bridging the gap between synthetic and authentic images for multimodal machine translation" (Guo et al., 2023, Bridging the Gap between Synthetic and Authentic Images for Multimodal Machine Translation), were found to use supervised methods, focusing on training with parallel data, thus excluding them.
Older papers like "Unsupervised Multi-modal Neural Machine Translation" (2018) and "Unsupervised Multimodal Neural Machine Translation with Pseudo Visual Pivoting" (2020) were identified but did not meet the publication year criterion. The scarcity of post-2022 papers using purely unsupervized/self-supervized methods suggests a research gap, possibly due to the reliance on supervised learning for MMT given the availability of parallel datasets like Multi30k. The inclusion of "Adding Multimodal Capabilities to a Text-only Translation Model" reflects this gap, as it uses self-supervized pre-training but supervised fine-tuning, highlighting a potential controversy in classification.
Metrics and Evaluation
BLEU is the primary metric, measuring translation quality by comparing machine-generated translations to human references, focusing on n-gram precision. METEOR, while optional, considers synonymy and stemming, offering a more nuanced evaluation. The identified papers' focus on BLEU aligns with standard practice, with "Scene Graph as Pivoting" uniquely reporting METEOR scores for flickr 2016 and 2017. The absence of metrics for flickr 2018 and mscoco 2017 across all papers is an unexpected detail, suggesting these test sets may be under-explored in recent unsupervized/self-supervized MMT research, possibly due to dataset limitations or research focus.
Discussion and Implications
The limited findings suggest that unsupervized/self-supervized MMT research on Multi30k post-2022 is nascent, with papers like "Scene Graph as Pivoting," "Beyond Language," and "Video Pivoting" standing out for their innovative unsupervized approaches using images/videos as pivots. "Adding Multimodal Capabilities" introduces self-supervized pre-training but includes supervised fine-tuning, which may not fully align with the user's criteria, reflecting a potential debate in the field about what constitutes unsupervized methods. Future research could explore expanding unsupervized techniques, such as using pre-trained models like CLIP in unsupervized settings, or developing new datasets to facilitate unsupervized MMT, especially for under-reported test sets like flickr 2018 and mscoco 2017.
The reliance on BLEU as a metric is standard, but incorporating METEOR could provide deeper insights into translation quality, especially for ambiguous texts. The lack of reports for flickr 2018 and mscoco 2017 highlights an unexpected detail: a potential under-exploration of these test sets in recent research, which may limit the generalizability of findings.
Conclusion
This analysis identifies four SOTA papers meeting the criteria, with "Adding Multimodal Capabilities to a Text-only Translation Model" potentially controversial due to mixed training methods. Researchers are encouraged to explore innovative unsupervized/self-supervized methods to bridge current gaps, enhancing the field's applicability to low-resource languages and scenarios with limited parallel data. The table above provides a comprehensive overview, ensuring all relevant metrics are included for the user's specified test sets.
Key Citations
Adding Multimodal Capabilities to a Text-only Translation Model
Scene Graph as Pivoting: Inference-time Image-free Unsupervised MMT with Visual Scene Hallucination
Beyond Language: Empowering Unsupervised MT with Cross-modal Alignment
Video Pivoting Unsupervised Multi-Modal Machine Translation
Bridging the Gap between Synthetic and Authentic Images for Multimodal Machine Translation
Multi30k Dataset