import numpy as np
import pandas as pd
import pickle

import os
import sacrebleu


import torch
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Evaluator():

    def __init__(self, cycleGAN, args, experiment):
        """ Class for evaluation """
        super(Evaluator, self).__init__()

        self.cycleGAN = cycleGAN
        self.args = args
        self.experiment = experiment

        self.bleu = evaluate.load('sacrebleu')
        # self.rouge = evaluate.load('rouge')
        self.meteor = evaluate.load('meteor')
        # if args.bertscore: self.bertscore = evaluate.load('bertscore')
    

    def __compute_metric__(self, predictions, references, metric_name, direction=None):
        # predictions = list | references = list of lists
        scores = []
        if metric_name in ['bleu', 'meteor']:
            for pred, ref in zip(predictions, references):
                print("pred, ref:",pred, ref)
                if metric_name == 'bleu':
                    res = self.bleu.compute(predictions=[pred], references=[ref])
                    scores.append(res['score'])
                elif metric_name == 'meteor':
                    tmp_meteor = []
                    for r in ref:
                        res = self.meteor.compute(predictions=[pred], references=[r])['meteor']
                        tmp_meteor.append(100 * res)
                    scores.append(max(tmp_meteor))
                    # res = self.meteor.compute(predictions=pred, references=ref[0])['meteor']
                    # scores.append(100 * res)
                # elif metric_name == 'rouge':
                #     tmp_rouge1, tmp_rouge2, tmp_rougeL = [], [], []
                #     for r in ref:
                #         res = self.rouge.compute(predictions=[pred], references=[r], use_aggregator=False)
                #         tmp_rouge1.append(res['rouge1'][0].fmeasure)
                #         tmp_rouge2.append(res['rouge2'][0].fmeasure)
                #         tmp_rougeL.append(res['rougeL'][0].fmeasure)
                #     scores.append([max(tmp_rouge1), max(tmp_rouge2), max(tmp_rougeL)])
                # elif metric_name == 'bertscore':
                #     res = self.bertscore.compute(predictions=[pred], references=[ref], lang=self.args.lang)
                #     scores.extend(res['f1'])
        else:
            raise Exception(f"Metric {metric_name} is not supported.")
        return scores
    

    def __compute_classif_metrics__(self, pred_A, pred_B):
        device = self.cycleGAN.device
        truncation, padding = 'longest_first', 'max_length'
        if 'lambdas' not in vars(self.args) or self.args.lambdas[4] == 0 or self.args.pretrained_classifier_eval != self.args.pretrained_classifier_model:
            classifier = AutoModelForSequenceClassification.from_pretrained(self.args.pretrained_classifier_eval)
            classifier_tokenizer = AutoTokenizer.from_pretrained(f'{self.args.pretrained_classifier_eval}tokenizer/')
            classifier.to(device)
        else:
            classifier = self.cycleGAN.Cls.model
            classifier_tokenizer = self.cycleGAN.Cls.tokenizer
        classifier.eval()

        y_pred, y_true = [], np.concatenate((np.full(len(pred_A), 0), np.full(len(pred_B), 1)))

        for i in range(0, len(pred_A), self.args.batch_size):
            batch_a = pred_A[i:i+self.args.batch_size]
            inputs = classifier_tokenizer(batch_a, truncation=truncation, padding=padding, max_length=self.args.max_sequence_length, return_tensors="pt")
            inputs = inputs.to(device)
            with torch.no_grad():
                output = classifier(**inputs)
            y_pred.extend(np.argmax(output.logits.cpu().numpy(), axis=1))
        for i in range(0, len(pred_B), self.args.batch_size):
            batch_b = pred_B[i:i+self.args.batch_size]
            inputs = classifier_tokenizer(batch_b, truncation=truncation, padding=padding, max_length=self.args.max_sequence_length, return_tensors="pt")
            inputs = inputs.to(device)
            with torch.no_grad():
                output = classifier(**inputs)
            y_pred.extend(np.argmax(output.logits.cpu().numpy(), axis=1))
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        return acc, prec, rec, f1


    def run_eval_mono(self, epoch, current_training_step, phase, mono_dl_a_eval, mono_dl_b_eval):
        print(f'Start {phase}...')
        self.cycleGAN.eval() # set evaluation mode

        if self.args.comet_logging:
            if phase == 'validation': context = self.experiment.validate
            elif phase == 'test': context = self.experiment.test
        
        real_A, real_B = [], []
        pred_A, pred_B = [], []
        scores_AB_bleu_self, scores_BA_bleu_self = [], []
        scores_AB_meteor_self, scores_BA_meteor_self = [], []
        # scores_AB_r1_self, scores_BA_r1_self, scores_AB_r2_self, scores_BA_r2_self, scores_AB_rL_self, scores_BA_rL_self = [], [], [], [], [], []

        for batch in mono_dl_a_eval:
            mono_a = list(batch)
            with torch.no_grad():
                transferred = self.cycleGAN.transfer(sentences=mono_a, direction='AB')
            real_A.extend(mono_a)
            pred_B.extend(transferred)
            mono_a = [[s] for s in mono_a]
            scores_AB_bleu_self.extend(self.__compute_metric__(transferred, mono_a, 'bleu'))
            scores_AB_meteor_self.extend(self.__compute_metric__(transferred, mono_a, 'meteor'))
            # scores_rouge_self = np.array(self.__compute_metric__(transferred, mono_a, 'rouge'))
            # scores_AB_r1_self.extend(scores_rouge_self[:, 0].tolist())
            # scores_AB_r2_self.extend(scores_rouge_self[:, 1].tolist())
            # scores_AB_rL_self.extend(scores_rouge_self[:, 2].tolist())
        avg_AB_bleu_self = np.mean(scores_AB_bleu_self)
        avg_AB_meteteor_self = np.mean(scores_AB_meteor_self)
        # avg_AB_r1_self, avg_AB_r2_self, avg_AB_rL_self = np.mean(scores_AB_r1_self), np.mean(scores_AB_r2_self), np.mean(scores_AB_rL_self)

        for batch in mono_dl_b_eval:
            mono_b = list(batch)
            with torch.no_grad():
                transferred = self.cycleGAN.transfer(sentences=mono_b, direction='BA')
            real_B.extend(mono_b)
            pred_A.extend(transferred)
            mono_b = [[s] for s in mono_b]
            scores_BA_bleu_self.extend(self.__compute_metric__(transferred, mono_b, 'bleu'))
            scores_BA_meteor_self.extend(self.__compute_metric__(transferred, mono_b, 'meteor'))
            # scores_rouge_self = np.array(self.__compute_metric__(transferred, mono_b, 'rouge'))
            # scores_BA_r1_self.extend(scores_rouge_self[:, 0].tolist())
            # scores_BA_r2_self.extend(scores_rouge_self[:, 1].tolist())
            # scores_BA_rL_self.extend(scores_rouge_self[:, 2].tolist())
        avg_BA_bleu_self = np.mean(scores_BA_bleu_self)
        avg_BA_meteteor_self = np.mean(scores_BA_meteor_self)
        # avg_BA_r1_self, avg_BA_r2_self, avg_BA_rL_self = np.mean(scores_BA_r1_self), np.mean(scores_BA_r2_self), np.mean(scores_BA_rL_self)
        avg_2dir_bleu_self = (avg_AB_bleu_self + avg_BA_bleu_self) / 2
        avg_2dir_meteor_self = (avg_AB_meteteor_self + avg_BA_meteteor_self) / 2

        acc, _, _, _ = self.__compute_classif_metrics__(pred_A, pred_B)
        acc_scaled = acc * 100
        avg_acc_bleu_self = (avg_2dir_bleu_self + acc_scaled) / 2
        avg_acc_bleu_self_geom = (avg_2dir_bleu_self * acc_scaled)**0.5
        avg_acc_bleu_self_h = 2*avg_2dir_bleu_self*acc_scaled/(avg_2dir_bleu_self+acc_scaled+1e-6)

        avg_acc_meteor_self = (avg_2dir_meteor_self + acc_scaled) / 2
        avg_acc_meteor_self_geom = (avg_2dir_meteor_self * acc_scaled)**0.5
        avg_acc_meteor_self_h = 2*avg_2dir_meteor_self*acc_scaled/(avg_2dir_meteor_self+acc_scaled+1e-6)
    
        metrics = {'epoch':epoch, 'step':current_training_step,
                   'self-BLEU A->B':avg_AB_bleu_self, 'self-BLEU B->A':avg_BA_bleu_self,
                   'self-BLEU avg':avg_2dir_bleu_self,
                   'self-METEOR A->B':avg_AB_meteteor_self, 'self-METEOR B->A':avg_BA_meteteor_self,
                   'style accuracy':acc, 'acc-BLEU':avg_acc_bleu_self, 'acc-METEOR': avg_acc_meteor_self,'g-acc-BLEU':avg_acc_bleu_self_geom, 'h-acc-BLEU':avg_acc_bleu_self_h, 
                   'g-acc-METEOR':avg_acc_meteor_self_geom, 'h-acc-METEOR':avg_acc_meteor_self_h}
        
        if phase == 'validation':
            base_path = f"{self.args.save_base_folder}epoch_{epoch}/"
            if self.args.eval_strategy == 'epochs':
                suffix = f'epoch{epoch}'
                if epoch < self.args.additional_eval:
                    suffix += f'_step{current_training_step}'
            else: suffix = f'step{current_training_step}'
        else:
            if self.args.from_pretrained is not None:
                if self.args.save_base_folder is not None:
                    base_path = f"{self.args.save_base_folder}"
                else:
                    base_path = f"{self.args.from_pretrained}epoch_{epoch}/"
            else:
                base_path = f"{self.args.save_base_folder}test/epoch_{epoch}/"
            suffix = f'epoch{epoch}_test'
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        pickle.dump(metrics, open(f"{base_path}metrics_{suffix}.pickle", 'wb'))

        for m, v in metrics.items():
            if m not in ['epoch', 'step']:
                print(f'{m}: {v}')

        df_AB = pd.DataFrame()
        df_AB['A (source)'] = real_A
        df_AB['B (generated)'] = pred_B
        df_AB.to_csv(f"{base_path}AB_{suffix}.csv", sep=',', header=True)
        df_BA = pd.DataFrame()
        df_BA['B (source)'] = real_B
        df_BA['A (generated)'] = pred_A
        df_BA.to_csv(f"{base_path}BA_{suffix}.csv", sep=',', header=True)

        if self.args.comet_logging:
            with context():
                self.experiment.log_table(f'./AB_{suffix}.csv', tabular_data=df_AB, headers=True)
                self.experiment.log_table(f'./BA_{suffix}.csv', tabular_data=df_BA, headers=True)
                for m, v in metrics.items():
                    if m not in ['epoch', 'step']:
                        self.experiment.log_metric(m, v, step=current_training_step, epoch=epoch)
        del df_AB, df_BA
        print(f'End {phase}...')

    
    def run_eval_ref(self, epoch, current_training_step, phase, parallel_dl_evalAB, parallel_dl_evalBA):
        print(f'Start {phase}...')
        self.cycleGAN.eval() # set evaluation mode

        if self.args.comet_logging:
            if phase == 'validation': context = self.experiment.validate
            elif phase == 'test': context = self.experiment.test
        
        real_A, real_B = [], []
        pred_A, pred_B = [], []
        ref_A, ref_B = [], []
        scores_AB_bleu_self, scores_BA_bleu_self = [], []
        scores_AB_bleu_ref, scores_BA_bleu_ref = [], []
        scores_AB_meteor_self, scores_BA_meteor_self = [], []
        scores_AB_meteor_ref, scores_BA_meteor_ref = [], []
        # scores_AB_r1_self, scores_BA_r1_self, scores_AB_r2_self, scores_BA_r2_self, scores_AB_rL_self, scores_BA_rL_self = [], [], [], [], [], []
        # scores_AB_r1_ref, scores_BA_r1_ref, scores_AB_r2_ref, scores_BA_r2_ref, scores_AB_rL_ref, scores_BA_rL_ref = [], [], [], [], [], []
        # scores_AB_bscore, scores_BA_bscore = [], []

        for batch in parallel_dl_evalAB:
            parallel_a = list(batch[0])
            references_b = list(batch[1])
            if self.args.lowercase_ref:
                references_b = [[ref.lower() for ref in refs] for refs in references_b]
            with torch.no_grad():
                transferred = self.cycleGAN.transfer(sentences=parallel_a, direction='AB')
            real_A.extend(parallel_a)
            pred_B.extend(transferred)
            ref_B.extend(references_b)
            parallel_a = [[s] for s in parallel_a]
            scores_AB_bleu_self.extend(self.__compute_metric__(transferred, parallel_a, 'bleu'))
            scores_AB_bleu_ref.extend(self.__compute_metric__(transferred, references_b, 'bleu'))
            scores_AB_meteor_self.extend(self.__compute_metric__(transferred, parallel_a, 'meteor'))
            scores_AB_meteor_ref.extend(self.__compute_metric__(transferred, references_b, 'meteor'))
            # scores_rouge_self = np.array(self.__compute_metric__(transferred, parallel_a, 'rouge'))
            # scores_AB_r1_self.extend(scores_rouge_self[:, 0].tolist())
            # scores_AB_r2_self.extend(scores_rouge_self[:, 1].tolist())
            # scores_AB_rL_self.extend(scores_rouge_self[:, 2].tolist())
            # scores_rouge_ref = np.array(self.__compute_metric__(transferred, references_b, 'rouge'))
            # scores_AB_r1_ref.extend(scores_rouge_ref[:, 0].tolist())
            # scores_AB_r2_ref.extend(scores_rouge_ref[:, 1].tolist())
            # scores_AB_rL_ref.extend(scores_rouge_ref[:, 2].tolist())
            # if self.args.bertscore: scores_AB_bscore.extend(self.__compute_metric__(transferred, references_b, 'bertscore'))
            # else: scores_AB_bscore.extend([0])
        avg_AB_bleu_self, avg_AB_bleu_ref = np.mean(scores_AB_bleu_self), np.mean(scores_AB_bleu_ref)
        avg_AB_bleu_geom = (avg_AB_bleu_self*avg_AB_bleu_ref)**0.5
        avg_AB_meteteor_self, avg_AB_meteteor_ref = np.mean(scores_AB_meteor_self), np.mean(scores_AB_meteor_ref)
        avg_AB_meteteor_geom = (avg_AB_meteteor_self*avg_AB_meteteor_ref)**0.5
        # avg_AB_r1_self, avg_AB_r2_self, avg_AB_rL_self = np.mean(scores_AB_r1_self), np.mean(scores_AB_r2_self), np.mean(scores_AB_rL_self)
        # avg_AB_r1_ref, avg_AB_r2_ref, avg_AB_rL_ref = np.mean(scores_AB_r1_ref), np.mean(scores_AB_r2_ref), np.mean(scores_AB_rL_ref)
        # avg_AB_bscore = np.mean(scores_AB_bscore)

        for batch in parallel_dl_evalBA:
            parallel_b = list(batch[0])
            references_a = list(batch[1])
            if self.args.lowercase_ref:
                references_a = [[ref.lower() for ref in refs] for refs in references_a]
            with torch.no_grad():
                transferred = self.cycleGAN.transfer(sentences=parallel_b, direction='BA')
            real_B.extend(parallel_b)
            pred_A.extend(transferred)
            ref_A.extend(references_a)
            parallel_b = [[s] for s in parallel_b]
            scores_BA_bleu_self.extend(self.__compute_metric__(transferred, parallel_b, 'bleu'))
            scores_BA_bleu_ref.extend(self.__compute_metric__(transferred, references_a, 'bleu'))
            scores_BA_meteor_self.extend(self.__compute_metric__(transferred, parallel_b, 'meteor'))
            scores_BA_meteor_ref.extend(self.__compute_metric__(transferred, references_a, 'meteor'))
            # scores_rouge_self = np.array(self.__compute_metric__(transferred, parallel_b, 'rouge'))
            # scores_BA_r1_self.extend(scores_rouge_self[:, 0].tolist())
            # scores_BA_r2_self.extend(scores_rouge_self[:, 1].tolist())
            # scores_BA_rL_self.extend(scores_rouge_self[:, 2].tolist())
            # scores_rouge_ref = np.array(self.__compute_metric__(transferred, references_a, 'rouge'))
            # scores_BA_r1_ref.extend(scores_rouge_ref[:, 0].tolist())
            # scores_BA_r2_ref.extend(scores_rouge_ref[:, 1].tolist())
            # scores_BA_rL_ref.extend(scores_rouge_ref[:, 2].tolist())
            # if self.args.bertscore: scores_BA_bscore.extend(self.__compute_metric__(transferred, references_a, 'bertscore'))
            # else: scores_BA_bscore.extend([0])
        avg_BA_bleu_self, avg_BA_bleu_ref = np.mean(scores_BA_bleu_self), np.mean(scores_BA_bleu_ref)
        avg_BA_bleu_geom = (avg_BA_bleu_self*avg_BA_bleu_ref)**0.5
        avg_BA_meteteor_self, avg_BA_meteteor_ref = np.mean(scores_BA_meteor_self), np.mean(scores_BA_meteor_ref)
        avg_BA_meteteor_geom = (avg_BA_meteteor_self*avg_BA_meteteor_ref)**0.5
        # avg_BA_r1_self, avg_BA_r2_self, avg_BA_rL_self = np.mean(scores_BA_r1_self), np.mean(scores_BA_r2_self), np.mean(scores_BA_rL_self)
        # avg_BA_r1_ref, avg_BA_r2_ref, avg_BA_rL_ref = np.mean(scores_BA_r1_ref), np.mean(scores_BA_r2_ref), np.mean(scores_BA_rL_ref)
        # avg_BA_bscore = np.mean(scores_BA_bscore)
        avg_2dir_bleu_ref = (avg_AB_bleu_ref + avg_BA_bleu_ref) / 2
        avg_2dir_meteor_ref = (avg_AB_meteteor_ref + avg_BA_meteteor_ref) / 2

        metrics = {'epoch':epoch, 'step':current_training_step,
                   'self-BLEU A->B':avg_AB_bleu_self, 'self-BLEU B->A':avg_BA_bleu_self,
                   'ref-BLEU A->B':avg_AB_bleu_ref, 'ref-BLEU B->A':avg_BA_bleu_ref,
                   'ref-BLEU avg':avg_2dir_bleu_ref,
                   'g-BLEU A->B':avg_AB_bleu_geom, 'g-BLEU B->A':avg_BA_bleu_geom, 
                   'self-METEOR A->B':avg_AB_meteteor_self, 'self-METEOR B->A':avg_BA_meteteor_self,
                   'ref-METEOR A->B':avg_AB_meteteor_ref, 'ref-METEOR B->A':avg_BA_meteteor_ref,
                   'ref-METEOR avg':avg_2dir_meteor_ref,
                   }

        # if phase == 'test':
        #     acc, prec, rec, f1 = self.__compute_classif_metrics__(pred_A, pred_B)
        #     metrics['style accuracy'] = acc
        #     metrics['style precision'] = prec
        #     metrics['style recall'] = rec
        #     metrics['style F1 score'] = f1
        
        if phase == 'validation':
            base_path = f"{self.args.save_base_folder}epoch_{epoch}/"
            if self.args.eval_strategy == 'epochs':
                suffix = f'epoch{epoch}'
                if epoch < self.args.additional_eval:
                    suffix += f'_step{current_training_step}'
            else: suffix = f'step{current_training_step}'
        else:
            if self.args.from_pretrained is not None:
                if self.args.save_base_folder is not None:
                    base_path = f"{self.args.save_base_folder}"
                else:
                    base_path = f"{self.args.from_pretrained}epoch_{epoch}/"
            else:
                base_path = f"{self.args.save_base_folder}test/epoch_{epoch}/"
            suffix = f'epoch{epoch}_test'
            if self.args.from_pretrained and 'GYAFCfm' in self.args.from_pretrained:
                if 'family' in self.args.path_paral_test_ref: ds = 'family'
                elif 'music' in self.args.path_paral_test_ref: ds = 'music'
                suffix += f'_{ds}'
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        pickle.dump(metrics, open(f"{base_path}metrics_{suffix}.pickle", 'wb'))

        for m, v in metrics.items():
            if m not in ['epoch', 'step']:
                print(f'{m}: {v}')

        df_AB = pd.DataFrame()
        df_AB['A (source)'] = real_A
        df_AB['B (generated)'] = pred_B
        ref_B = np.array(ref_B)
        for i in range(self.args.n_references):
            df_AB[f'ref {i+1}'] = ref_B[:, i]
        df_AB.to_csv(f"{base_path}AB_{suffix}.csv", sep=',', header=True)
        df_BA = pd.DataFrame()
        df_BA['B (source)'] = real_B
        df_BA['A (generated)'] = pred_A
        ref_A = np.array(ref_A)
        for i in range(self.args.n_references):
            df_BA[f'ref {i+1}'] = ref_A[:, i]
        df_BA.to_csv(f"{base_path}BA_{suffix}.csv", sep=',', header=True)
        
        if self.args.comet_logging:    
            with context():
                self.experiment.log_table(f'./AB_{suffix}.csv', tabular_data=df_AB, headers=True)
                self.experiment.log_table(f'./BA_{suffix}.csv', tabular_data=df_BA, headers=True)
                for m, v in metrics.items():
                    if m not in ['epoch', 'step']:
                        self.experiment.log_metric(m, v, step=current_training_step, epoch=epoch)
        del df_AB, df_BA
        print(f'End {phase}...')
        

    def dummy_classif(self):
        # Get target language from args.style_b (as defined in train.py)
        target_lang = self.args.style_b
        
        # pred_A = English sentences (should be classified as language A/English)
        pred_A = [
            'Two young, White males are outside near many bushes.',
            'Several men in hard hats are operating a giant pulley system.',
            'A little girl climbing into a wooden playhouse.',
            'A man in a blue shirt is standing on a ladder cleaning a window.',
            'Two men are at the stove preparing food.',
            'A man in green holds a guitar while the other man observes his shirt.',
            'A man is smiling at a stuffed lion.',
            'A trendy girl talking on her cellphone while gliding slowly down the street.'
        ]
        
        # pred_B = Target language sentences (should be classified as language B)
        if target_lang == 'de':
            pred_B = [
                'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.',
                'Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.',
                'Ein kleines Mädchen klettert in ein Spielhaus aus Holz.',
                'Ein Mann in einem blauen Hemd steht auf einer Leiter und putzt ein Fenster.',
                'Zwei Männer stehen am Herd und bereiten Essen zu.',
                'Ein Mann in grün hält eine Gitarre, während der andere Mann sein Hemd ansieht.',
                'Ein Mann lächelt einen ausgestopften Löwen an.',
                'Ein schickes Mädchen spricht mit dem Handy während sie langsam die Straße entlangschwebt.'
            ]
        elif target_lang == 'fr':
            pred_B = [
                'Deux jeunes hommes blancs sont dehors près de buissons.',
                'Plusieurs hommes en casque font fonctionner un système de poulies géant.',
                'Une petite fille grimpe dans une maisonnette en bois.',
                'Un homme dans une chemise bleue se tient sur une échelle pour nettoyer une fenêtre.',
                'Deux hommes aux fourneaux préparent à manger.',
                'Un homme en vert tient une guitare tandis qu\'un autre homme observe sa chemise.',
                'Un homme sourit à un ours en peluche.',
                'Une fille branchée parle à son portable tout en glissant lentement dans la rue.'
            ]
        elif target_lang == 'cs':
            pred_B = [
                'Dva mladí bílí muži jsou venku poblíž mnoha keřů.',
                'Několik mužů v ochranných přilbách obsluhují velký kladkový systém.',
                'Malá dívka leze do dřevěného hračkového domu.',
                'Muž v modrém tričku stojí na žebříku a myje okno.',
                'Dva muži připravují u sporáku jídlo.',
                'Muž v zeleném drží kytaru zatímco další muž pozoruje jeho košili.',
                'Muž se směje na vycpaného lva.',
                'Modní dívka mluví do svého mobilního telefonu zatímco klouzá pomalu po ulici.'
            ]
        else:
            # Fallback to German if unknown language
            pred_B = [
                'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.',
                'Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.',
                'Ein kleines Mädchen klettert in ein Spielhaus aus Holz.',
                'Ein Mann in einem blauen Hemd steht auf einer Leiter und putzt ein Fenster.',
                'Zwei Männer stehen am Herd und bereiten Essen zu.',
                'Ein Mann in grün hält eine Gitarre, während der andere Mann sein Hemd ansieht.',
                'Ein Mann lächelt einen ausgestopften Löwen an.',
                'Ein schickes Mädchen spricht mit dem Handy während sie langsam die Straße entlangschwebt.'
            ]
        
        acc, _, _, _ = self.__compute_classif_metrics__(pred_A, pred_B)
        print(f'Dummy classification metrics computation end (target language: {target_lang})')


    def dummy_bscore(self):
        predictions = ['i just left this car wash and was very satisfied !',
                    "just like ordering anything if you 're seated .",
                    "one durable thing after another they do care to address .",
                    'food was warm , i had the ribs .',
                    'five stars is what i want to give .',
                    'if i could give more stars , i would .',
                    "they will tell you though .",
                    'she was happy being there .']
        references = [['i just left this car wash and was very satisfied !', 'i just left the car wash and i feel very satisfied', 'i did not leave this car wash and was very satisfied', 'i just left the car wash and i was very satisfied'],
                    ["i would recommend ordering something once you 're seated", "ordering anything if you 're seated", 'the ordering service is nice', 'the staffs here are very attentive'],
                    ['they addressed all the broken items', 'one correct thing after another they care to address', 'one broken thing after another they really care to address', 'they have good after sales service'],
                    ['food was hot ( and fresh ) , i had the ribs .', "food was n't cold ( well cooked ) , i had the ribs", 'the food is food , i had the ribs', 'food was warm , i had the ribs .'],
                    ['i want to give this a 5 out of 5 star rating .', '5 stars is what i would give', 'give five stars to him', 'give 5+ stars to him'],
                    ['i would give an extra star if it allowed me', 'if i could give more stars , i definitely would', 'if i could give more stars , i would', 'i would give you more stars if i could'],
                    ['they will tell you though .', 'they will tell you the details', 'they will tell you though .', 'they will tell you'],
                    ['she seemed happy to be there', 'she was cheerful being there .', 'she was happy because being here', 'she was happy being there']]
        scores = []
        # for pred, ref in zip(predictions, references):
        #     res = self.bertscore.compute(predictions=[pred], references=[ref], lang=self.args.lang)
        #     scores.append(res['f1'])
        print('Dummy BERTScore computation end')
