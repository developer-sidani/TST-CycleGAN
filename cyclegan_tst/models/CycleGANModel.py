import torch
from torch import nn
from typing import List, Optional, Tuple, Union

from cyclegan_tst.models.GeneratorModel import GeneratorModel
from cyclegan_tst.models.DiscriminatorModel import DiscriminatorModel
from cyclegan_tst.models.ClassifierModel import ClassifierModel

import logging


class CycleGANModel(nn.Module):

    def __init__(
        self,
        G_ab : Union[GeneratorModel, None], 
        G_ba : Union[GeneratorModel, None], 
        D_ab : Union[DiscriminatorModel, None],
        D_ba : Union[DiscriminatorModel, None],
        Cls : Union[ClassifierModel, None],
        device = None,
        use_clip: bool = False
    ):
        """Initialization method for the CycleGANModel

        Args:
            G_ab (:obj:cyclegan_tst.models.GeneratorModel): Generator model for the mapping from A->B
            G_ba (:obj:cyclegan_tst.models.GeneratorModel): Generator model for the mapping from B->A
            D_b (:obj:cyclegan_tst.models.DiscriminatorModel): Discriminator model for B
            D_a (:obj:cyclegan_tst.models.DiscriminatorModel): Discriminator model for A
            Cls (:obj:cyclegan_tst.models.ClassifierModel): Style classifier
            device: Device to use for computation
            use_clip: Whether to use CLIP embeddings
        """
        super(CycleGANModel, self).__init__()
        
        if G_ab is None or G_ba is None or D_ab is None or D_ba is None:
            logging.warning("CycleGANModel: Some models are not provided, please invoke 'load_models' to initialize them from a previous checkpoint")

        self.G_ab = G_ab
        self.G_ba = G_ba
        self.D_ab = D_ab
        self.D_ba = D_ba
        self.Cls = Cls
        self.use_clip = use_clip

        self.device = device
        logging.info(f"Device: {device}")

        # all model to device
        self.G_ab.model.to(self.device)
        self.G_ba.model.to(self.device)
        self.D_ab.model.to(self.device)
        self.D_ba.model.to(self.device)
        if self.Cls is not None:
            self.Cls.model.to(self.device)
        
        # Move adapters to device if using CLIP
        if self.use_clip:
            if hasattr(self.G_ab, 'adapter'):
                self.G_ab.adapter.to(self.device)
            if hasattr(self.G_ba, 'adapter'):
                self.G_ba.adapter.to(self.device)

    def train(self):
        self.G_ab.train()
        self.G_ba.train()
        self.D_ab.train()
        self.D_ba.train()

    def eval(self):
        self.G_ab.eval()
        self.G_ba.eval()
        self.D_ab.eval()
        self.D_ba.eval()

    def get_optimizer_parameters(
        self
    ):
        optimization_parameters = []
        
        # Generator parameters
        optimization_parameters += list(self.G_ab.model.parameters())
        optimization_parameters += list(self.G_ba.model.parameters())
        
        # Add adapter parameters if using CLIP
        if self.use_clip:
            if hasattr(self.G_ab, 'adapter'):
                optimization_parameters += list(self.G_ab.adapter.parameters())
            if hasattr(self.G_ba, 'adapter'):
                optimization_parameters += list(self.G_ba.adapter.parameters())
        
        # Discriminator parameters
        optimization_parameters += list(self.D_ab.model.parameters())
        optimization_parameters += list(self.D_ba.model.parameters())
        
        return optimization_parameters

    def training_cycle(
        self, 
        sentences_a: List[str] = None,
        sentences_b: List[str] = None,
        images_a: List = None,
        images_b: List = None,
        target_sentences_ab: List[str] = None,
        target_sentences_ba: List[str] = None,
        lambdas: List[float] = None,
        comet_experiment = None,
        loss_logging = None,
        training_step: int = None
    ):
        """
        Training cycle for CycleGAN that can handle either text or image inputs
        
        Args:
            sentences_a: List of sentences from domain A
            sentences_b: List of sentences from domain B
            images_a: List of images from domain A
            images_b: List of images from domain B
            target_sentences_ab: Target sentences for A->B
            target_sentences_ba: Target sentences for B->A
            lambdas: Loss weights
            comet_experiment: Comet experiment for logging
            loss_logging: Dictionary for loss logging
            training_step: Current training step
        """
        print(f"\nDEBUG: --- TRAINING CYCLE START - Step {training_step} ---")
        
        if sentences_a is not None:
            print(f"DEBUG: Domain A text inputs: {len(sentences_a)}")
            print(f"DEBUG: First A sentence: {sentences_a[0][:50]}...")
        
        if sentences_b is not None:
            print(f"DEBUG: Domain B text inputs: {len(sentences_b)}")
            print(f"DEBUG: First B sentence: {sentences_b[0][:50]}...")
            
        if images_a is not None:
            print(f"DEBUG: Domain A image inputs: {len(images_a)}")
            
        if images_b is not None:
            print(f"DEBUG: Domain B image inputs: {len(images_b)}")
            
        print(f"DEBUG: Lambdas (weights): {lambdas}")
        
        # ---------- BEGIN : cycle A -> B ----------
        print(f"DEBUG: --- A->B Translation ---")

        # first half - select text or image input based on what's provided
        if self.use_clip and images_a is not None:
            print(f"DEBUG: Using images as input for G_ab")
            out_transferred_ab, transferred_ab = self.G_ab(images=images_a, device=self.device)
        else:
            print(f"DEBUG: Using text as input for G_ab")
            out_transferred_ab, transferred_ab = self.G_ab(sentences=sentences_a, device=self.device)
        
        print(f"DEBUG: G_ab output: {transferred_ab[0][:50]}...")
        
        # D_ab fake
        self.D_ab.eval()  # this loss is only for the Generator
        zeros = torch.zeros(len(transferred_ab))
        ones = torch.ones(len(transferred_ab))
        labels_fake_sentences = torch.column_stack((ones, zeros))  # one to the class index 0
        # print ("Discriminator labels:", labels_fake_sentences)
        _, loss_g_ab = self.D_ab(transferred_ab, labels_fake_sentences, device=self.device)
        print(f"DEBUG: Discriminator G_ab loss: {loss_g_ab.item()}")
        
        if lambdas[4] != 0:
            # labels_style_b_sentences = torch.column_stack((zeros, ones))
            labels_style_b_sentences = torch.ones(len(transferred_ab), dtype=int)
            _, loss_g_ab_cls = self.Cls(transferred_ab, labels_style_b_sentences, device=self.device)
            print(f"DEBUG: Classifier G_ab loss: {loss_g_ab_cls.item()}")
        
        # second half - use generated text for the cycle consistency 
        print(f"DEBUG: --- B->A Reconstruction ---")
        out_reconstructed_ba, reconstructed_ba, cycle_loss_aba = self.G_ba(sentences=transferred_ab, 
                                                                          target_sentences=sentences_a, 
                                                                          device=self.device)
                                                                          
        print(f"DEBUG: Reconstructed A: {reconstructed_ba[0][:50]}...")
        print(f"DEBUG: Cycle loss A->B->A: {cycle_loss_aba.item()}")

        complete_loss_g_ab = lambdas[0]*cycle_loss_aba + lambdas[1]*loss_g_ab
        print(f"DEBUG: Complete G_ab loss: {complete_loss_g_ab.item()}")
        
        if comet_experiment is not None:
            with comet_experiment.train():
                comet_experiment.log_metric(f"Cycle Loss A-B-A", lambdas[0]*cycle_loss_aba, step=training_step)
                comet_experiment.log_metric(f"Loss generator  A-B", lambdas[1]*loss_g_ab, step=training_step)
        loss_logging['Cycle Loss A-B-A'].append(lambdas[0]*cycle_loss_aba.item())
        loss_logging['Loss generator  A-B'].append(lambdas[1]*loss_g_ab.item())

        if lambdas[4] != 0:
            complete_loss_g_ab = complete_loss_g_ab + lambdas[4]*loss_g_ab_cls
            if comet_experiment is not None:
                with comet_experiment.train():
                    comet_experiment.log_metric(f"Classifier-guided A-B", lambdas[4]*loss_g_ab_cls, step=training_step)
            loss_logging['Classifier-guided A-B'].append(lambdas[4]*loss_g_ab_cls.item())
        
        complete_loss_g_ab.backward()
        
        # D_ab fake
        print(f"DEBUG: --- Discriminator B Training ---")
        self.D_ab.train()
        zeros = torch.zeros(len(transferred_ab))
        ones = torch.ones(len(transferred_ab))
        labels_fake_sentences = torch.column_stack((zeros, ones))  # one to the class index 1
        _, loss_d_ab_fake = self.D_ab(transferred_ab, labels_fake_sentences, device=self.device) 
        print(f"DEBUG: D_ab fake loss: {loss_d_ab_fake.item()}")
        
        # D_ab real
        zeros = torch.zeros(len(transferred_ab))
        ones = torch.ones(len(transferred_ab))
        labels_real_sentences = torch.column_stack((ones, zeros))  # one to the class index 0
        _, loss_d_ab_real = self.D_ab(sentences_b, labels_real_sentences, device=self.device)
        print(f"DEBUG: D_ab real loss: {loss_d_ab_real.item()}")
        
        complete_loss_d_ab = lambdas[2]*loss_d_ab_fake + lambdas[3]*loss_d_ab_real
        print(f"DEBUG: Complete D_ab loss: {complete_loss_d_ab.item()}")

        if comet_experiment is not None:
            with comet_experiment.train():
                comet_experiment.log_metric(f"Loss D(A->B)", complete_loss_d_ab, step=training_step)
        loss_logging['Loss D(A->B)'].append(complete_loss_d_ab.item())
        
        # backward A -> B
        complete_loss_d_ab.backward()
        
        # ----------  END : cycle A -> B  ----------
        
        # ---------- BEGIN : cycle B -> A ----------
        print(f"DEBUG: --- B->A Translation ---")
        # first half - select text or image input based on what's provided
        if self.use_clip and images_b is not None:
            print(f"DEBUG: Using images as input for G_ba")
            out_transferred_ba, transferred_ba = self.G_ba(images=images_b, device=self.device)
        else:
            print(f"DEBUG: Using text as input for G_ba")
            out_transferred_ba, transferred_ba = self.G_ba(sentences=sentences_b, device=self.device)

        print(f"DEBUG: G_ba output: {transferred_ba[0][:50]}...")
        
        # D_ba fake
        self.D_ba.eval()  # this loss is only for the Generator 
        zeros = torch.zeros(len(transferred_ba))
        ones = torch.ones(len(transferred_ba))
        labels_fake_sentences = torch.column_stack((ones, zeros))  # one to the class index 0
        _, loss_g_ba = self.D_ba(transferred_ba, labels_fake_sentences, device=self.device)
        print(f"DEBUG: Discriminator G_ba loss: {loss_g_ba.item()}")
        
        if lambdas[4] != 0:
            # labels_style_a_sentences = torch.column_stack((ones, zeros))
            labels_style_a_sentences = torch.zeros(len(transferred_ba), dtype=int)
            _, loss_g_ba_cls = self.Cls(transferred_ba, labels_style_a_sentences, device=self.device)
            print(f"DEBUG: Classifier G_ba loss: {loss_g_ba_cls.item()}")
        
        # second half 
        print(f"DEBUG: --- A->B Reconstruction ---")
        out_reconstructed_ab, reconstructed_ab, cycle_loss_bab = self.G_ab(sentences=transferred_ba, 
                                                                          target_sentences=sentences_b, 
                                                                          device=self.device)
                                                                          
        print(f"DEBUG: Reconstructed B: {reconstructed_ab[0][:50]}...")
        print(f"DEBUG: Cycle loss B->A->B: {cycle_loss_bab.item()}")

        complete_loss_g_ba = lambdas[0]*cycle_loss_bab + lambdas[1]*loss_g_ba
        print(f"DEBUG: Complete G_ba loss: {complete_loss_g_ba.item()}")
        
        if comet_experiment is not None:
            with comet_experiment.train():
                comet_experiment.log_metric(f"Cycle Loss B-A-B", lambdas[0]*cycle_loss_bab, step=training_step)
                comet_experiment.log_metric(f"Loss generator  B-A", lambdas[1]*loss_g_ba, step=training_step)
                
        loss_logging['Cycle Loss B-A-B'].append(lambdas[0]*cycle_loss_bab.item())
        loss_logging['Loss generator  B-A'].append(lambdas[1]*loss_g_ba.item())
                
        if lambdas[4] != 0:
            complete_loss_g_ba = complete_loss_g_ba + lambdas[4]*loss_g_ba_cls
            if comet_experiment is not None:
                with comet_experiment.train():
                    comet_experiment.log_metric(f"Classifier-guided B-A", lambdas[4]*loss_g_ba_cls, step=training_step)
            loss_logging['Classifier-guided B-A'].append(lambdas[4]*loss_g_ba_cls.item())
                
        # backward B -> A
        complete_loss_g_ba.backward()
        
        # D_ba fake
        print(f"DEBUG: --- Discriminator A Training ---")
        self.D_ba.train()
        zeros = torch.zeros(len(transferred_ba))
        ones = torch.ones(len(transferred_ba))
        labels_fake_sentences = torch.column_stack((zeros, ones))  # one to the class index 1
        _, loss_d_ba_fake = self.D_ba(transferred_ba, labels_fake_sentences, device=self.device)
        print(f"DEBUG: D_ba fake loss: {loss_d_ba_fake.item()}")
        
        # D_ba real
        zeros = torch.zeros(len(transferred_ba))
        ones = torch.ones(len(transferred_ba))
        labels_real_sentences = torch.column_stack((ones, zeros))  # one to the class index 0
        _, loss_d_ba_real = self.D_ba(sentences_a, labels_real_sentences, device=self.device)
        print(f"DEBUG: D_ba real loss: {loss_d_ba_real.item()}")
        
        complete_loss_d_ba = lambdas[2]*loss_d_ba_fake + lambdas[3]*loss_d_ba_real
        print(f"DEBUG: Complete D_ba loss: {complete_loss_d_ba.item()}")
        
        if comet_experiment is not None:
            with comet_experiment.train():
                comet_experiment.log_metric(f"Loss D(B->A)", complete_loss_d_ba, step=training_step)
        loss_logging['Loss D(B->A)'].append(complete_loss_d_ba.item())
                
        # backward B -> A
        complete_loss_d_ba.backward()
        # ---------- END : cycle B -> A ----------
        print(f"DEBUG: --- TRAINING CYCLE END ---\n")

    def save_models(
        self,
        base_path: Union[str]
    ):
        """Save all models in the CycleGAN"""
        self.G_ab.save_model(base_path + "/G_ab/")
        self.G_ba.save_model(base_path + "/G_ba/")
        self.D_ab.save_model(base_path + "/D_ab/")
        self.D_ba.save_model(base_path + "/D_ba/")

    def transfer(
        self,
        sentences: List[str] = None,
        images = None,
        direction: str = "AB"
    ):
        """
        Transfer content from one domain to another
        
        Args:
            sentences: Input sentences (if text-based)
            images: Input images (if using CLIP)
            direction: Direction of transfer ("AB" or "BA")
            
        Returns:
            List of generated sentences
        """
        if direction == "AB":
            if self.use_clip and images is not None:
                transferred_sentences = self.G_ab.transfer(images=images, device=self.device)
            else:
                transferred_sentences = self.G_ab.transfer(sentences=sentences, device=self.device)
        else:
            if self.use_clip and images is not None:
                transferred_sentences = self.G_ba.transfer(images=images, device=self.device)
            else:
                transferred_sentences = self.G_ba.transfer(sentences=sentences, device=self.device)
                
        return transferred_sentences
