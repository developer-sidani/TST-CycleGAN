import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict

from cyclegan_tst.models.MultimodalGeneratorModel import MultimodalGeneratorModel
from cyclegan_tst.models.DiscriminatorModel import DiscriminatorModel
from cyclegan_tst.models.ClassifierModel import ClassifierModel

import logging


class MultimodalCycleGANModel(nn.Module):
    """
    CycleGAN model adapted for multimodal machine translation.
    Supports image inputs and text inputs with cycle consistency training.
    """

    def __init__(
            self,
            G_ab: Union[MultimodalGeneratorModel, None],
            G_ba: Union[MultimodalGeneratorModel, None],
            D_a: Union[DiscriminatorModel, None],
            D_b: Union[DiscriminatorModel, None],
            Cls: Union[ClassifierModel, None] = None,
            device=None,
    ):
        """
        Initialize the Multimodal CycleGAN model

        Args:
            G_ab: Generator for source to target translation
            G_ba: Generator for target to source translation
            D_a: Discriminator for source language
            D_b: Discriminator for target language
            Cls: Optional classifier for guided training
            device: Computation device
        """
        super(MultimodalCycleGANModel, self).__init__()

        if G_ab is None or G_ba is None or D_a is None or D_b is None:
            logging.warning("Some models are not provided, please invoke 'load_models' to initialize them")

        self.G_ab = G_ab  # Source -> Target
        self.G_ba = G_ba  # Target -> Source
        self.D_a = D_a  # Discriminator for source language
        self.D_b = D_b  # Discriminator for target language
        self.Cls = Cls  # Optional style/language classifier

        self.device = device
        logging.info(f"Device: {device}")

        # Move models to device
        if self.device:
            if self.G_ab:
                self.G_ab.mbart_model.to(self.device)
                self.G_ab.adapter.to(self.device)
            if self.G_ba:
                self.G_ba.mbart_model.to(self.device)
                self.G_ba.adapter.to(self.device)
            if self.D_a:
                self.D_a.model.to(self.device)
            if self.D_b:
                self.D_b.model.to(self.device)
            if self.Cls:
                self.Cls.model.to(self.device)

    def train(self):
        """Set all models to training mode"""
        self.G_ab.train()
        self.G_ba.train()
        self.D_a.train()
        self.D_b.train()
        if self.Cls:
            self.Cls.eval()  # Classifier always in eval mode

    def eval(self):
        """Set all models to evaluation mode"""
        self.G_ab.eval()
        self.G_ba.eval()
        self.D_a.eval()
        self.D_b.eval()
        if self.Cls:
            self.Cls.eval()

    def get_optimizer_parameters(self):
        """Get all trainable parameters for optimization"""
        optimization_parameters = []

        # G_ab parameters (exclude frozen CLIP)
        optimization_parameters += list(self.G_ab.mbart_model.parameters())
        optimization_parameters += list(self.G_ab.adapter.parameters())

        # G_ba parameters (exclude frozen CLIP)
        optimization_parameters += list(self.G_ba.mbart_model.parameters())
        optimization_parameters += list(self.G_ba.adapter.parameters())

        # Discriminator parameters
        optimization_parameters += list(self.D_a.model.parameters())
        optimization_parameters += list(self.D_b.model.parameters())

        return optimization_parameters

    def visual_consistency_loss(self, clip_model, image_emb, generated_text, device):
        """
        Compute consistency between image embedding and generated text embedding

        Args:
            clip_model: CLIP model for encoding
            image_emb: Image embeddings
            generated_text: Generated text
            device: Computation device

        Returns:
            Cosine distance loss
        """
        text_emb = self.G_ab.encode_text(generated_text, device)
        return 1 - F.cosine_similarity(image_emb, text_emb).mean()

    def training_cycle_stage1(
            self,
            images: List,
            target_captions: List[str],
            lambdas: List[float] = None,
            comet_experiment=None,
            loss_logging=None,
            training_step: int = None
    ):
        """
        Stage 1 training cycle - Image captioning pretraining

        Args:
            images: List of images or image paths
            target_captions: Target captions in target language
            lambdas: Loss weights [caption]
            comet_experiment: Optional Comet ML experiment for logging
            loss_logging: Dictionary for logging loss values
            training_step: Current training step
        """
        # Encode images
        image_embs = self.G_ab.encode_image(images, device=self.device)

        # Generate captions
        _, generated_captions, caption_loss = self.G_ab(
            image_embs,
            is_clip_embedding=True,
            reference_sentences=target_captions,
            device=self.device
        )

        # Update loss logging
        if comet_experiment is not None:
            with comet_experiment.train():
                comet_experiment.log_metric("Caption Loss", caption_loss, step=training_step)

        if loss_logging is not None:
            loss_logging['Caption Loss'].append(caption_loss.item())

        # Backward pass
        caption_loss.backward()

    def training_cycle_multimodal(
            self,
            src_sentences: List[str],
            tgt_sentences: List[str],
            images: List = None,
            lambdas: List[float] = None,  # [cycle, gen, disc_fake, disc_real, cls, visual]
            comet_experiment=None,
            loss_logging=None,
            training_step: int = None
    ):
        """
        Full multimodal CycleGAN training cycle

        Args:
            src_sentences: Source language sentences
            tgt_sentences: Target language sentences
            images: Optional images for visual consistency
            lambdas: Loss weights [cycle, gen, disc_fake, disc_real, cls, visual]
            comet_experiment: Optional Comet ML experiment for logging
            loss_logging: Dictionary for logging loss values
            training_step: Current training step
        """
        # Default lambdas if not provided
        if lambdas is None:
            lambdas = [10.0, 1.0, 1.0, 1.0, 0.0, 1.0]

        # Get image embeddings if images provided
        img_embs = None
        if images is not None:
            img_embs = self.G_ab.encode_image(images, device=self.device)

        # Get text embeddings
        src_embs = self.G_ab.encode_text(src_sentences, device=self.device)
        tgt_embs = self.G_ba.encode_text(tgt_sentences, device=self.device)

        # ---------- BEGIN : cycle A -> B ----------

        # Generate translation from A to B
        _, transferred_ab, direct_loss_ab = self.G_ab(
            src_embs,
            is_clip_embedding=True,
            reference_sentences=tgt_sentences,
            device=self.device
        )

        # D_b loss (adversarial)
        self.D_b.eval()  # This loss is only for Generator
        zeros = torch.zeros(len(transferred_ab), device=self.device)
        ones = torch.ones(len(transferred_ab), device=self.device)
        labels_fake_sentences = torch.column_stack((ones, zeros))
        _, loss_g_ab = self.D_b(transferred_ab, labels_fake_sentences, device=self.device)

        # Optional: Classifier-guided loss
        loss_g_ab_cls = 0
        if lambdas[4] != 0 and self.Cls is not None:
            # Target style label
            labels_style_b = torch.ones(len(transferred_ab), dtype=int, device=self.device)
            _, loss_g_ab_cls = self.Cls(transferred_ab, labels_style_b, device=self.device)

        # Optional: Visual consistency loss
        loss_visual_ab = 0
        if lambdas[5] != 0 and img_embs is not None:
            loss_visual_ab = self.visual_consistency_loss(
                self.G_ab.clip_model,
                img_embs,
                transferred_ab,
                self.device
            )

        # Generate cycle reconstruction B -> A
        transferred_ab_embs = self.G_ba.encode_text(transferred_ab, device=self.device)
        _, reconstructed_ba, cycle_loss_aba = self.G_ba(
            transferred_ab_embs,
            is_clip_embedding=True,
            reference_sentences=src_sentences,
            device=self.device
        )

        # Compute total generator loss for A->B->A cycle
        complete_loss_g_ab = (
                lambdas[0] * cycle_loss_aba +  # Cycle consistency
                lambdas[1] * loss_g_ab +  # Generator adversarial
                lambdas[4] * loss_g_ab_cls +  # Classifier guided
                lambdas[5] * loss_visual_ab  # Visual consistency
        )

        # Log losses
        if comet_experiment is not None:
            with comet_experiment.train():
                comet_experiment.log_metric("Cycle Loss A-B-A", lambdas[0] * cycle_loss_aba, step=training_step)
                comet_experiment.log_metric("Loss generator A-B", lambdas[1] * loss_g_ab, step=training_step)
                comet_experiment.log_metric("Direct Loss A-B", direct_loss_ab, step=training_step)
                if lambdas[4] != 0:
                    comet_experiment.log_metric("Classifier-guided A-B", lambdas[4] * loss_g_ab_cls, step=training_step)
                if lambdas[5] != 0 and img_embs is not None:
                    comet_experiment.log_metric("Visual Loss A-B", lambdas[5] * loss_visual_ab, step=training_step)

        if loss_logging is not None:
            loss_logging['Cycle Loss A-B-A'].append(lambdas[0] * cycle_loss_aba.item())
            loss_logging['Loss generator A-B'].append(lambdas[1] * loss_g_ab.item())
            loss_logging['Direct Loss A-B'].append(direct_loss_ab.item())
            if lambdas[4] != 0:
                loss_logging['Classifier-guided A-B'].append(lambdas[4] * loss_g_ab_cls.item())
            if lambdas[5] != 0 and img_embs is not None:
                loss_logging['Visual Loss A-B'].append(lambdas[5] * loss_visual_ab.item())

        # Backward pass for generator
        complete_loss_g_ab.backward()

        # Train discriminator B with fake samples
        self.D_b.train()
        zeros = torch.zeros(len(transferred_ab), device=self.device)
        ones = torch.ones(len(transferred_ab), device=self.device)
        # Fake samples should be classified as fake (class 1)
        labels_fake_sentences = torch.column_stack((zeros, ones))
        _, loss_d_b_fake = self.D_b(transferred_ab.detach(), labels_fake_sentences, device=self.device)

        # Train discriminator B with real samples
        # Real samples should be classified as real (class 0)
        labels_real_sentences = torch.column_stack((ones, zeros))
        _, loss_d_b_real = self.D_b(tgt_sentences, labels_real_sentences, device=self.device)

        # Total discriminator B loss
        loss_d_b = lambdas[2] * loss_d_b_fake + lambdas[3] * loss_d_b_real

        # Log discriminator loss
        if comet_experiment is not None:
            with comet_experiment.train():
                comet_experiment.log_metric("Loss D_B", loss_d_b, step=training_step)

        if loss_logging is not None:
            loss_logging['Loss D_B'].append(loss_d_b.item())

        # Backward pass for discriminator B
        loss_d_b.backward()

        # ---------- END : cycle A -> B ----------

        # ---------- BEGIN : cycle B -> A ----------

        # Generate translation from B to A
        _, transferred_ba, direct_loss_ba = self.G_ba(
            tgt_embs,
            is_clip_embedding=True,
            reference_sentences=src_sentences,
            device=self.device
        )

        # D_a loss (adversarial)
        self.D_a.eval()  # This loss is only for Generator
        zeros = torch.zeros(len(transferred_ba), device=self.device)
        ones = torch.ones(len(transferred_ba), device=self.device)
        labels_fake_sentences = torch.column_stack((ones, zeros))
        _, loss_g_ba = self.D_a(transferred_ba, labels_fake_sentences, device=self.device)

        # Optional: Classifier-guided loss
        loss_g_ba_cls = 0
        if lambdas[4] != 0 and self.Cls is not None:
            # Source style label
            labels_style_a = torch.zeros(len(transferred_ba), dtype=int, device=self.device)
            _, loss_g_ba_cls = self.Cls(transferred_ba, labels_style_a, device=self.device)

        # Optional: Visual consistency loss for B->A (if applicable)
        loss_visual_ba = 0

        # Generate cycle reconstruction A -> B
        transferred_ba_embs = self.G_ab.encode_text(transferred_ba, device=self.device)
        _, reconstructed_ab, cycle_loss_bab = self.G_ab(
            transferred_ba_embs,
            is_clip_embedding=True,
            reference_sentences=tgt_sentences,
            device=self.device
        )

        # Compute total generator loss for B->A->B cycle
        complete_loss_g_ba = (
                lambdas[0] * cycle_loss_bab +  # Cycle consistency
                lambdas[1] * loss_g_ba +  # Generator adversarial
                lambdas[4] * loss_g_ba_cls  # Classifier guided
        )

        # Log losses
        if comet_experiment is not None:
            with comet_experiment.train():
                comet_experiment.log_metric("Cycle Loss B-A-B", lambdas[0] * cycle_loss_bab, step=training_step)
                comet_experiment.log_metric("Loss generator B-A", lambdas[1] * loss_g_ba, step=training_step)
                comet_experiment.log_metric("Direct Loss B-A", direct_loss_ba, step=training_step)
                if lambdas[4] != 0:
                    comet_experiment.log_metric("Classifier-guided B-A", lambdas[4] * loss_g_ba_cls, step=training_step)

        if loss_logging is not None:
            loss_logging['Cycle Loss B-A-B'].append(lambdas[0] * cycle_loss_bab.item())
            loss_logging['Loss generator B-A'].append(lambdas[1] * loss_g_ba.item())
            loss_logging['Direct Loss B-A'].append(direct_loss_ba.item())
            if lambdas[4] != 0:
                loss_logging['Classifier-guided B-A'].append(lambdas[4] * loss_g_ba_cls.item())

        # Backward pass for generator
        complete_loss_g_ba.backward()

        # Train discriminator A with fake samples
        self.D_a.train()
        zeros = torch.zeros(len(transferred_ba), device=self.device)
        ones = torch.ones(len(transferred_ba), device=self.device)
        # Fake samples should be classified as fake (class 1)
        labels_fake_sentences = torch.column_stack((zeros, ones))
        _, loss_d_a_fake = self.D_a(transferred_ba.detach(), labels_fake_sentences, device=self.device)

        # Train discriminator A with real samples
        # Real samples should be classified as real (class 0)
        labels_real_sentences = torch.column_stack((ones, zeros))
        _, loss_d_a_real = self.D_a(src_sentences, labels_real_sentences, device=self.device)

        # Total discriminator A loss
        loss_d_a = lambdas[2] * loss_d_a_fake + lambdas[3] * loss_d_a_real

        # Log discriminator loss
        if comet_experiment is not None:
            with comet_experiment.train():
                comet_experiment.log_metric("Loss D_A", loss_d_a, step=training_step)

        if loss_logging is not None:
            loss_logging['Loss D_A'].append(loss_d_a.item())

        # Backward pass for discriminator A
        loss_d_a.backward()

        # ---------- END : cycle B -> A ----------

    def transfer(
            self,
            sentences: List[str],
            direction: str,
            images: List = None,
            device=None
    ):
        """
        Perform translation in either direction

        Args:
            sentences: Source sentences to translate
            direction: Either "AB" (source->target) or "BA" (target->source)
            images: Optional images for multimodal translation
            device: Computation device

        Returns:
            Translated sentences
        """
        if direction not in ["AB", "BA"]:
            raise ValueError("Direction must be either 'AB' or 'BA'")

        if images is not None:
            # Multimodal translation with images
            img_embs = self.G_ab.encode_image(images, device) if direction == "AB" else self.G_ba.encode_image(images,
                                                                                                               device)
            text_embs = self.G_ab.encode_text(sentences, device) if direction == "AB" else self.G_ba.encode_text(
                sentences, device)

            # Combine image and text embeddings (simple average for now, could be more sophisticated)
            combined_embs = (img_embs + text_embs) / 2

            if direction == "AB":
                _, translated = self.G_ab.forward(combined_embs, is_clip_embedding=True, device=device,
                                                  generate_only=True)
            else:
                _, translated = self.G_ba.forward(combined_embs, is_clip_embedding=True, device=device,
                                                  generate_only=True)
        else:
            # Text-only translation
            if direction == "AB":
                translated = self.G_ab.transfer(sentences, is_image=False, device=device)
            else:
                translated = self.G_ba.transfer(sentences, is_image=False, device=device)

        return translated

    def save_models(
            self,
            base_path: str
    ):
        """
        Save all model components

        Args:
            base_path: Directory to save models
        """
        import os
        os.makedirs(base_path, exist_ok=True)

        self.G_ab.save_model(base_path + "/G_ab/")
        self.G_ba.save_model(base_path + "/G_ba/")
        self.D_a.save_model(base_path + "/D_a/")
        self.D_b.save_model(base_path + "/D_b/")