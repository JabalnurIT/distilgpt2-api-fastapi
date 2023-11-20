import json

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)

import os
import time
import datetime
import math
import numpy as np
import random
import pandas as pd
from datasets import Dataset
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import DefaultFlowCallback, AdamW, get_linear_schedule_with_warmup

from .gpt2_dataset import GPT2Dataset

import locale
locale.getpreferredencoding = lambda: "UTF-8"



with open("config.json") as json_file:
    config = json.load(json_file)


class Model:
    def __init__(self):

        self.tokenizer = GPT2Tokenizer.from_pretrained(config["DISTIL_GPT2_MODEL"],bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
        
        self.config = GPT2Config.from_json_file(config["PRE_TRAINED_CONFIG"])

        self.model = GPT2LMHeadModel.from_pretrained(config["PRE_TRAINED_MODEL"],config=self.config)
        self.model.resize_token_embeddings(len(self.tokenizer))

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.cuda()
        else:
            self.device = torch.device("cpu")
            self.model.to(self.device)

        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        
    def format_time(self, elapsed):
        return str(datetime.timedelta(seconds=int(round((elapsed)))))

    def generate(self, texts):
        texts = texts.split("###")
        len_texts = len(texts)
        
        self.model.eval()

        prompt = "<|startoftext|>"

        generated = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
        generated = generated.to(self.device)

        outputs = self.model.generate(
                                        generated,
                                        do_sample=True,
                                        top_k=50,
                                        max_length = 300,
                                        top_p=0.95,
                                        num_return_sequences=len_texts,
                                        )
        text_outputs = []
        for n in outputs:
            text_outputs.append(self.tokenizer.decode(n, skip_special_tokens=True))

        data_text = "###".join(text_outputs)
        return data_text

    def save_finetuned_model(self, output_dir):
        self.model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
        self.model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def retrain(self, texts):
        texts = texts.split("###")

        dataset = GPT2Dataset(texts, self.tokenizer, max_length=40)

        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        batch_size = 2
        train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        validation_dataloader = DataLoader(
                    val_dataset, # The validation samples.
                    sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                    batch_size = batch_size # Evaluate with this batch size.
                )

        # some parameters I cooked up that work reasonably well

        epochs = 10
        learning_rate = 5e-4
        warmup_steps = 1e2
        epsilon = 1e-8

        # this produces sample output every 100 steps
        sample_every = 100

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                lr = learning_rate,
                eps = epsilon
                )
        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        # This changes the learning rate as the training loop progresses
        scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps = warmup_steps,
                                                    num_training_steps = total_steps)

        # total_t0 = time.time()

        training_stats = []

        self.model = self.model.to(self.device)


        for epoch_i in range(0, epochs):

            # ========================================
            #               Training
            # ========================================

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # t0 = time.time()

            total_train_loss = 0

            self.model.train()

            for step, batch in enumerate(train_dataloader):
                print('1')

                b_input_ids = batch[0].to(self.device)
                b_labels = batch[0].to(self.device)
                b_masks = batch[1].to(self.device)

                self.model.zero_grad()

                outputs = self.model(  b_input_ids,
                                labels=b_labels,
                                attention_mask = b_masks,
                                token_type_ids=None
                                )

                loss = outputs[0]

                batch_loss = loss.item()
                total_train_loss += batch_loss

                # Get sample every x batches.
                print('2')
                if step % sample_every == 0 and not step == 0:

                    # elapsed = model.format_time(time.time() - t0)
                    # print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

                    self.model.eval()

                    sample_outputs = self.model.generate(
                                            bos_token_id=random.randint(1,30000),
                                            do_sample=True,
                                            top_k=50,
                                            max_length = 200,
                                            top_p=0.95,
                                            num_return_sequences=1
                                        )
                    # for i, sample_output in enumerate(sample_outputs):
                    #     print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
                    print('3')
                    self.model.train()
                print('4')
                # loss.backward()
                print('5')
                # self.optimizer.step()
                print('6')
                # scheduler.step()
                print('7')

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            # Measure how long this epoch took.
            # training_time = model.format_time(time.time() - t0)

            # print("")
            # print("  Average training loss: {0:.2f}".format(avg_train_loss))
            # print("  Training epoch took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================

            print("")
            print("Running Validation...")

            # t0 = time.time()

            self.model.eval()

            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:

                b_input_ids = batch[0].to(self.device)
                b_labels = batch[0].to(self.device)
                b_masks = batch[1].to(self.device)

                with torch.no_grad():

                    outputs  = self.model(b_input_ids,
        #                            token_type_ids=None,
                                    attention_mask = b_masks,
                                    labels=b_labels)

                    loss = outputs[0]

                batch_loss = loss.item()
                total_eval_loss += batch_loss

            avg_val_loss = total_eval_loss / len(validation_dataloader)

            # validation_time = model.format_time(time.time() - t0)

            # print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            # print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            # training_stats.append(
            #     {
            #         'epoch': epoch_i + 1,
            #         'Training Loss': avg_train_loss,
            #         'Valid. Loss': avg_val_loss,
            #         'Training Time': training_time,
            #         'Validation Time': validation_time
            #     }
            # )

        print("")
        print("Training complete!")
        # print("Total training took {:} (h:mm:ss)".format(model.format_time(time.time()-total_t0)))
        try:
            self.model.save_finetuned_model(output_dir=config["OUTPUT_DIR"])
            status = "Success"
        except Exception as e:
            status = "Error retraining model: " + str(e)
        return status
        
        




model = Model()


def get_model():
    return model

# fill_mask
# http POST http://127.0.0.1:8000/fill_mask text="Q23,George [MASK],1st [MASK] of the United States (1732–1799),[MASK],United States of America; Kingdom of Great Britain,Politician,1732,1799.0,natural causes,67.0"
# http POST http://127.0.0.1:8000/fill_mask text="Q42,[MASK] Adams,[MASK] writer and [MASK],[MASK],United Kingdom,Artist,1952,2001.0,natural causes,49.0"

# perplexity
# http POST http://127.0.0.1:8000/perplexity text="Q42,Douglas Adams,English writer and humorist,Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"

# retrain 
# http POST http://127.0.0.1:8000/retrain text="Q23,George Washington,1st president of the United States (1732–1799),Male,United States of America; Kingdom of Great Britain,Politician,1732,1799.0,natural causes,67.0"

# ['Q23,George Washington,1st president of the United States (1732–1799),Male,United States of America; Kingdom of Great Britain,Politician,1732,1799.0,natural causes,67.0','Q42,Douglas Adams,English writer and humorist,Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0','Q91,Abraham Lincoln,16th president of the United States (1809-1865),Male,United States of America,Politician,1809,1865.0,homicide,56.0','Q254,Wolfgang Amadeus Mozart,Austrian composer of the Classical period,Male,Archduchy of Austria; Archbishopric of Salzburg,Artist,1756,1791.0,,35.0']