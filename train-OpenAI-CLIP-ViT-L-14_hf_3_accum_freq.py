import os
import json
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
# Uncomment to use lightning-thunder
# import thunder
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import clip
from torch.optim.lr_scheduler import OneCycleLR
import random
from colorama import Fore, Style
from tqdm import tqdm

from sklearn.decomposition import PCA
import numpy as np

from transformers import CLIPProcessor, CLIPModel

training_losses = []
validation_losses = []
print("\n")

# Just scroll down to where it says:
# ======= CONFIGURE THIS! ======= 

# Save training plots with matplotlib to:
plots_folder = 'ft-plots'
os.makedirs(plots_folder, exist_ok=True)

# Save model .pt files to: 
ft_checkpoints_folder = 'ft-checkpoints'
os.makedirs(ft_checkpoints_folder, exist_ok=True)

# Save verbose text / training logs to:
text_logs_folder = 'ft-logs'
os.makedirs(text_logs_folder, exist_ok=True)



def adjust_unfreeze_rate(epoch, adjust_after=12, increase_rate=2):
    """
    Adjusts the rate of unfreezing after a certain number of epochs.
    :param epoch: Current epoch number.
    :param adjust_after: Epoch after which to increase unfreezing rate.
    :param increase_rate: How many layers to unfreeze per epoch after adjust_after.
    :return: Number of layers to unfreeze per epoch.
    """
    if epoch < adjust_after:
        return 1  # Initial slower unfreeze rate
    else:
        return increase_rate  # Increased rate after initial pass

def unfreeze_layers(model, epoch, total_layers=24, unfreeze_all=False):
    if unfreeze_all:
        for param in model.parameters():
            param.requires_grad = True
    else:
        unfreeze_every_n_epochs = adjust_unfreeze_rate(epoch)
        layers_to_unfreeze = (epoch // unfreeze_every_n_epochs) % total_layers
        layers_to_unfreeze = min(layers_to_unfreeze, total_layers)
        for i, (name, param) in enumerate(model.named_parameters()):
            if i >= total_layers - layers_to_unfreeze:
                param.requires_grad = True
            else:
                param.requires_grad = False
                

def monitor_gradient_norms(gradient_norms, threshold=1e-5):
    alert_messages = []
    for name, norms in gradient_norms.items():
        mean_norm = sum(norms) / len(norms)
        if mean_norm < threshold:  # Vanishing gradient
            alert_messages.append(Fore.RED + f"Vanishing gradient detected in {name} with mean norm {mean_norm:.2e}" + Style.RESET_ALL)
        elif mean_norm > 1000:  # Exploding gradient
            alert_messages.append(Fore.RED + f"Exploding gradient detected in {name} with mean norm {mean_norm:.2e}" + Style.RESET_ALL)
    if alert_messages:
        for message in alert_messages:
            print(message)
        # Optionally, you could also implement some automatic adjustment strategies here


def plot_gradient_norms(gradient_norms, epoch, use_log_scale=True):
    plt.figure(figsize=(20, 10))
    
    # Choose a colormap
    cmap = plt.get_cmap('Spectral')
    
    # Sort the layers by the maximum gradient norm value, descending
    sorted_layers = sorted(gradient_norms.items(), key=lambda item: max(item[1]), reverse=True)
    
    # Generate distinct colors from the colormap
    colors = cmap(range(len(sorted_layers)))
    
    for (layer_name, norms), color in zip(sorted_layers, colors):
        plt.plot(norms, label=layer_name, color=color)

    plt.xlabel('Batch')
    plt.ylabel('Gradient Norm')
    #plt.title(f'Gradient Norms for Epoch {epoch}{" - Log Scale" if use_log_scale else ""}')
    
    # Adjust legend: position at top right with smaller font size
    plt.legend(loc='upper right', fontsize='small')
    
    # If log scale is requested, change the y-axis to logarithmic
    if use_log_scale:
        plt.yscale('log')
        plt.title(f'Gradient Norms for Epoch {epoch}{" - Log Scale" if use_log_scale else ""}')
        plt.savefig(f"{plots_folder}/gradient_norms_epoch_{epoch}_log.png")
    else:
        plt.savefig(f"{plots_folder}/gradient_norms_epoch_{epoch}.png")
    
    plt.close()


def plot_training_info(epoch, training_losses, validation_losses, logits_images, logits_texts):
    epochs_x = range(1, epoch + 2)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    if len(training_losses) == len(epochs_x):
        plt.plot(epochs_x, training_losses, label='Training Loss')
    if len(validation_losses) == len(epochs_x):
        plt.plot(epochs_x, validation_losses, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    if len(logits_images) == len(epochs_x):
        plt.plot(epochs_x, logits_images, label='Average Logits')
    if len(logits_texts) == len(epochs_x):
        plt.plot(epochs_x, logits_texts, label='')
    plt.title('Average Logits Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Logits')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_folder}/combined_plot_epoch_{epoch + 1}.png")
    plt.close()
    

def calculate_metrics(logits, ground_truth):
    preds = torch.argmax(logits, dim=1)
    acc = accuracy_score(ground_truth.cpu(), preds.cpu())
    f1 = f1_score(ground_truth.cpu(), preds.cpu(), average='weighted')
    return acc, f1

'''
class ImageTextDataset(Dataset):
    def __init__(self, image_folder, annotations_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_paths = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')  # Convert to RGB
        if self.transform:
            image = self.transform(image)

        labels = self.annotations[self.image_paths[idx]]
        
        """
        Uses a random choice of multiple labels, if available.
        Example:
        todo: insert example here
        """
        if len(labels) >= 2:
            label = random.choice([labels[0], labels[1]])
        elif labels:
            label = labels[0]  # Fallback to the first label if less than 2 are available
        else:
            label = ''  # Fallback if no labels are available

        text = clip.tokenize([label])  # Tokenize the label

        return image, text.squeeze(0)  # Remove the extra dimension
'''
class ImageTextDataset(Dataset):
    def __init__(self, image_text_json_list, transform=None):
        self.image_text_json_list = sorted(image_text_json_list)
        self.transform = transform

    def __len__(self):
        return len(self.image_text_json_list)

    def __getitem__(self, idx):
        #get item info
        with open(self.image_text_json_list[idx], 'r') as json_file:
            metadata = json.load(json_file)

        image = metadata["image_file"]
        text = metadata["captions_dictionary"]["deepseek-vl-7b-chat_What is in this image?_llama3"]["caption_string"]

        #process image
        #image = Image.open(image_path).convert('RGB')  # Convert to RGB
        if self.transform:
            image = self.transform(image)

        #process label/text
        #text = clip.tokenize([label], truncate=True)  # Tokenize the label

        return image, text  # Remove the extra dimension




class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits_per_image, logits_per_text):
        labels = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
        loss_img = self.criterion(logits_per_image, labels)
        loss_txt = self.criterion(logits_per_text, labels)
        return (loss_img + loss_txt) / 2




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For GPU
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU setups

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable cuDNN benchmarking





from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()




# ======= CONFIGURE THIS! ======= 

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load model and preprocessing - CLIP model:
'''
clipmodel = 'ViT-L/14'
model, preprocess = clip.load(clipmodel, device=device)
'''
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# You will get loss = NaN with AdamW etc. if not full precision (case: you decide to remove automatic mixed precision (AMP)). Alas leave this be. =)
model = model.float()

# Advanced: Use thunder.jit - see github.com/Lightning-AI/lightning-thunder (not currently recommended for production use, but feel free to test it!)
# Needs further config - see above link for implementation details
# model = thunder.jit(model)

# Advanced: Unfreeze all of CLIP (default). Set to "False" to unfreeze slowly over X epochs. See the "def unfreeze" above for details.
unfreeze_all = True

# Recommended batch_size: As big as possible while still fitting your VRAM. 40 = fits for 24 GB with ViT-L/14 = SDXL Text Encoder.
# learning_rate: Typically 1e-5 to 1e-7. I find 1e-5 to always result in overfit with such a small batch size.
# Epochs: Defaulting to saving every 5 epochs (see very end of code). Consider saving even more often + stop training early and use previous checkpoint if you see overfit.
# If the val loss is not decreasing with loss or val even increases 
EPOCHS = 50
learning_rate = 5e-7
batch_size = 40

# Search this code for: label = random.choice([labels[0], labels[1]])
# -> If you have multiple labels for an image in your dataset (see "ft-X-example-my-dataset-labels.json" for details), you can set which labels to randomly choose from.
# It's like "noise, but in text" that can prevent CLIP from being over-confident in the training data (i.e. overfitting).
# Confusing labels will confuse CLIP, though. So, maybe don't use **all** the crazy labels you got from CLIP Interrogator aka CLIP+BLIP...

'''
# Define your own dataset and dataloader
dataset1 = ImageTextDataset("path/to/image/folder", "path/to/my-text-labels.json", transform=preprocess)
dataset2 = ImageTextDataset("path/to/image/folder/jitter-augmentation", "path/to/my-text-labels.json", transform=preprocess)
dataset3 = ImageTextDataset("path/to/image/folder/flip-augmentation", "path/to/my-text-labels.json", transform=preprocess)
'''
##argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--image_text_dirs", type=str, nargs='+', help="list of .list files")
parser.add_argument("--accum_freq", type=int, default=1, help="gradient accumulation frequency")
parser.add_argument("--deterministic", action='store_true', help="use deterministic training")
args = parser.parse_args()

#for deterministic training
if args.deterministic:
    set_seed(9999)
    print("\ntraining will be deterministic\n  --set_seed(9999)\n")

#########
#build image_text_list
image_text_dirs = args.image_text_dirs

#collect image_text_lists
image_text_list_files = []
for dir in image_text_dirs:
    for root, _, files in sorted(os.walk(dir)): #recursively search each dir
        for item in files:
            if item.endswith(".list"):
                item_path = os.path.join(root, item)
                print(f" --found: {item}")
                image_text_list_files.append(item_path)

#create list of jsons
image_text_json_list = []
for item in image_text_list_files:
    with open(item, "r") as f:
        list_temp = f.readlines() #each line to an item in list
        list_temp  = [line.strip() for line in list_temp ]  #remove newlines
        list_temp  = list(set(list_temp )) #remove duplicates
        list_temp  = [item for item in list_temp  if item] #remove empty items
        print(f" --found {len(list_temp)} items")
        #append to list
        image_text_json_list.extend(list_temp)

#split for train/val
random.shuffle(image_text_json_list)

val_len = (len(image_text_json_list)) // 10

val_image_text_json_list = image_text_json_list[:val_len]
val_image_text_json_list.sort()
train_image_text_json_list = image_text_json_list[val_len:]
train_image_text_json_list.sort()

print(f"len_val_image_text_json_list: {len(val_image_text_json_list)}")
print(f"train_image_text_json_list: {len(train_image_text_json_list)}")

#pass list to dataset
dataset1 = ImageTextDataset(train_image_text_json_list, transform=None)


# You can define many above, and then use only certain mixes for training:
concatenated_dataset = ConcatDataset([dataset1])
train_dataloader = DataLoader(concatenated_dataset, batch_size=batch_size, shuffle=True)

# Validation dataset and dataloader - use images from the training dataset that are NOT in the above training data! Recommended: 10-20% of full dataset.
val_dataset = ImageTextDataset(val_image_text_json_list, transform=None)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

total_steps = len(train_dataloader) * EPOCHS

'''
# Train with uneven learning rate across layers    
visual_parameters = [p for p in model.visual.parameters() if p.requires_grad]
transformer_parameters = [p for p in model.transformer.parameters() if p.requires_grad]

# Potentially useful if you get gigantic gradient norms at the delicate layers near the input
param_groups = [
    {'params': transformer_parameters[:len(transformer_parameters)//2], 'lr': 1e-6},  # First half of the transformer
    {'params': transformer_parameters[len(transformer_parameters)//2:], 'lr': 3e-6},   # Second half of the transformer
    {'params': visual_parameters[:len(visual_parameters)//2], 'lr': 1e-6},  # First half of the vision transformer
    {'params': visual_parameters[len(visual_parameters)//2:], 'lr': 3e-6},   # Second half of the vision transformer
]
'''
# Adjust for Hugging Face CLIP Model structure
visual_parameters = [p for p in model.vision_model.parameters() if p.requires_grad]
transformer_parameters = [p for p in model.text_model.parameters() if p.requires_grad]

# Now we can define param_groups with differential learning rates for different parts of the model
param_groups = [
    {'params': transformer_parameters[:len(transformer_parameters)//2], 'lr': 1e-6},  # First half of the text transformer
    {'params': transformer_parameters[len(transformer_parameters)//2:], 'lr': 3e-6},   # Second half of the text transformer
    {'params': visual_parameters[:len(visual_parameters)//2], 'lr': 1e-6},  # First half of the vision transformer
    {'params': visual_parameters[len(visual_parameters)//2:], 'lr': 3e-6},   # Second half of the vision transformer
]


# Default optimizer AdamW (not recommended). Set to "AdamW(param_groups, ...)" to use above differential learning rates 
# from torch.optim import AdamW
# optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.995), eps=1e-6, weight_decay=1e-2)

from adabelief_pytorch import AdaBelief
# Uncomment this to use the default, with rectify=True -> RAdam style (seems "good for CLIP"!):
# optimizer = AdaBelief(model.parameters(), lr=learning_rate, eps=1e-16, betas=(0.9, 0.999), weight_decay=1e-2, weight_decouple=False, rectify=True, print_change_log = False)

# If you are training on e.g. "sneaker designs" or something more "normal" (abundant in pre-training dataset), you may rather want to try the above default values first.
# I used this for "difficult" dataset of abstract images (unseen by CLIP pre-training) and "strange labels".
# weight_decay: Adds a penalty on the size of the weights to reduce overfitting by encouraging simpler models. Higher values typically increase regularization.
# However, as per my observation, it seems CLIP doesn't react to well to defaults like weight_decay=1e-2 that are often recommended.
# betas: Coefficients used for computing running averages of gradient and its square. Values closer to 1.0 result in slower updates.
# The first value controls the decay of the gradient moving average (momentum), and the second controls the decay of the squared gradient moving average (scaling). 
# Adjust the second value, leave the first at 0.9; default second value is 0.999, but I found 0.995 to improve results.
optimizer = AdaBelief(model.parameters(), lr=learning_rate, eps=1e-16, betas=(0.9, 0.995), weight_decay=1e-3, weight_decouple=False, rectify=True, print_change_log = False)

# Setup scheduler with a proportional warm-up phase. You may want to try anneal_strategy='cos' for cosine.
# pct_start=0.1 means that 10% of the training steps will be dedicated to ramping up the learning rate.
# anneal_strategy='linear': Gradually reduces the learning rate in a straight line from its maximum value to its minimum.
# anneal_strategy='cos': Reduces the learning rate following a cosine curve, providing a smoother transition at the beginning and end.
scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps, pct_start=0.1, anneal_strategy='linear')

# If you want warnings in friendly red letters about exploding or vanishing gradients, uncomment the line that says "monitor_gradient_norms(gradient_norms)" further below.

# ======= END OF CONFIG ======= 




model = model.float()
print(f"Precision: {model.dtype}")
print(f'Total batches: {len(train_dataloader)} @ Batch Size: {batch_size}')
print(f'accum_freq: {args.accum_freq}, total updates: {len(train_dataloader) / args.accum_freq}')
print("== START == \n")


def trainloop():
    contrastive_loss = ContrastiveLoss().to(device)
    logits_images = []
    logits_texts = []
    accum_freq = args.accum_freq  # Get accumulation frequency from args
    progress_bar_loss = 0.0


    #PCA analysis
    def pca_on_activations(model, dataloader):
        model.eval()
        activations = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = [Image.open(image).convert("RGB") for image in images]
                #inputs = processor(images=images, return_tensors="pt").to(device)
                #inputs = processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
                inputs = processor(images=images, return_tensors="pt", padding=True, truncation=True).to(device)
                features = model.vision_model(inputs.pixel_values).last_hidden_state  # Get vision model embeddings
                pooled_features = features.mean(dim=1)  # Pooling the features over the sequence length
                activations.append(pooled_features.cpu().numpy())  # Convert to NumPy and append
                '''
                images = images.to(device)
                features = model.encode_image(images)
                activations.append(features.cpu().numpy())
                '''
        activations = np.vstack(activations)
        pca = PCA(n_components=2)
        reduced_activations = pca.fit_transform(activations)
        return reduced_activations

    def plot_pca(reduced_activations, title, model_name):
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_activations[:, 0], reduced_activations[:, 1], alpha=0.5)
        plt.title(title)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.savefig(f"{plots_folder}/PCA_Plot_{model_name}.png")
        plt.close()

    name = f"base_CLIP"
    reduced_activations = pca_on_activations(model, val_dataloader)
    plot_pca(reduced_activations, f"{name} PCA Plot", name)


    for epoch in range(EPOCHS):
        gradient_norms = {}
        unfreeze_layers(model, epoch, total_layers=24, unfreeze_all=unfreeze_all)
        model.train()
        total_train_loss = 0.0
        #train_dataloader_prog = train_dataloader #why are these here?
        #train_dataloader_all = train_dataloader

        optimizer.zero_grad()  # Initialize optimizer zero grad outside the loop for accumulation

        train_accs, train_f1s, val_accs, val_f1s = [], [], [], []
        batch_logits_images = []
        batch_logits_texts = []

        #track accumulation steps
        accumulation_steps = 0
        

        accum_image_features = []
        accum_text_features = []
        accum_data = []
        accum_logit_scales = []


        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{EPOCHS}', leave=True)
        for batch_idx, (images, texts) in progress_bar:
            images = [Image.open(image).convert("RGB") for image in images]
            inputs = processor(
                images=images,
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)


            with torch.no_grad():
                with autocast():  # Apply autocast even during accumulation without gradients

                    outputs = model(**inputs)

                    image_embeds = outputs.image_embeds
                    text_embeds = outputs.text_embeds
                    logit_scale = model.logit_scale.exp()

                    accum_image_features.append(image_embeds)
                    accum_text_features.append(text_embeds)
                    accum_data.append(inputs)
                    accum_logit_scales.append(logit_scale)

            #if accum-freq reached, or is final batch: batch_idx == len(train_dataloader)
            # process accumulated batches
            if (batch_idx + 1) % args.accum_freq == 0 or batch_idx == len(train_dataloader):

                # Concatenate accumulated features
                all_image_features = torch.cat(accum_image_features, dim=0)
                all_text_features = torch.cat(accum_text_features, dim=0)
                logit_scale = accum_logit_scales[-1]  # Use the latest logit_scale

                #optimizer.zero_grad() #redundant, zero_grad after scheduler.step()

                # Recompute the forward pass for the accumulated batches with gradient tracking
                for j, inputs in enumerate(accum_data):
                    with autocast():  # Autocast for mixed precision during forward pass with gradients
                        outputs = model(**inputs)
                        image_embeds = outputs.image_embeds
                        text_embeds = outputs.text_embeds
                        logit_scale = model.logit_scale.exp()

                        # Replace the cached features with the recomputed ones
                        all_image_features = torch.cat(
                            accum_image_features[:j] + [image_embeds] + accum_image_features[j+1:], dim=0
                        )
                        all_text_features = torch.cat(
                            accum_text_features[:j] + [text_embeds] + accum_text_features[j+1:], dim=0
                        )

                        # Compute logits over the accumulated features
                        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
                        logits_per_text = logits_per_image.t()

                        '''
                        #just a reminder of how class ContrastiveLoss works
                        class ContrastiveLoss(nn.Module):
                            def __init__(self):
                                super(ContrastiveLoss, self).__init__()
                                self.criterion = nn.CrossEntropyLoss()

                            def forward(self, logits_per_image, logits_per_text):
                                labels = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
                                loss_img = self.criterion(logits_per_image, labels)
                                loss_txt = self.criterion(logits_per_text, labels)
                                return (loss_img + loss_txt) / 2
                        '''

                        # Compute the loss
                        total_loss = contrastive_loss(logits_per_image, logits_per_text)

                        #append logits
                        batch_logits_images.append(outputs.logits_per_image.mean().item())
                        batch_logits_texts.append(outputs.logits_per_text.mean().item())

                    # Backpropagate the scaled loss
                    scaler.scale(total_loss).backward()

                # Step the optimizer with scaled gradients
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                # Reset accumulators
                accum_image_features, accum_text_features = [], []
                accum_data, accum_logit_scales = [], []


                #calculate metrics for this batch
                current_batch_size = inputs['pixel_values'].size(0)  # or inputs['input_ids'] for the text batch size
                ground_truth = torch.arange(current_batch_size, device=device)
                acc, f1 = calculate_metrics(outputs.logits_per_image, ground_truth)
                train_accs.append(acc)
                train_f1s.append(f1)


                # Store gradient norms for plot
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        grad_norm = parameter.grad.norm().item()
                        gradient_norms.setdefault(name, []).append(grad_norm)
                
                # OPTIONAL DEBUG
                # vanishing in positional_embedding_res and exploding in visual.conv1.weight seems to frequently happen with AdamW
                # use this line to debug (and be spammed with red messages about exploding and vanishing gradients):
                monitor_gradient_norms(gradient_norms)
            
                total_train_loss += total_loss.item()

                progress_bar_loss = total_train_loss / (batch_idx + 1)

            if batch_idx >= args.accum_freq:
                progress_bar.set_postfix({
                    'loss': f'{progress_bar_loss:.4f}', 
                    'Logits Image': f'{batch_logits_images[-1]:.3f}', 
                    'Text': f'{batch_logits_texts[-1]:.3f}'
                })
            else:
                progress_bar.set_postfix({
                    'loss': 'N/A', 
                    'Logits Image': 'N/A', 
                    'Text': 'N/A'
                })

            if (batch_idx + 1) % args.accum_freq == 0:
                epoch_train_acc = sum(train_accs) / len(train_accs)
                epoch_train_f1 = sum(train_f1s) / len(train_f1s)
                with open(f"{text_logs_folder}/log_details_train.txt", "a", encoding='utf-8') as f:
                    f.write(f"Epoch {epoch + 1}/{EPOCHS}, Batch: {batch_idx + 1}/{len(train_dataloader)}, Loss: {total_loss.item():.4f}, Training Acc: {epoch_train_acc:.4f}, Training F1: {epoch_train_f1:.4f}\n")


        avg_train_loss = total_train_loss / len(train_dataloader)
        training_losses.append(avg_train_loss)
        
        epoch_avg_logits_image = sum(batch_logits_images) / len(batch_logits_images)
        epoch_avg_logits_text = sum(batch_logits_texts) / len(batch_logits_texts)
        logits_images.append(epoch_avg_logits_image)
        logits_texts.append(epoch_avg_logits_text)
        
        plot_gradient_norms(gradient_norms, epoch)      

        # Validation
        model.eval()
        total_val_loss = 0.0       
        print("Running Validation...")
        with torch.no_grad():
            for images, texts in val_dataloader:
                images = [Image.open(image).convert("RGB") for image in images]
                inputs = processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=True).to(device)

                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text

                current_batch_size = inputs['pixel_values'].size(0)
                ground_truth = torch.arange(current_batch_size, device=device)

                '''
                images, texts = images.to(device), texts.to(device)
                logits_per_image, logits_per_text = model(images, texts)
                '''
                
                val_loss = contrastive_loss(logits_per_image, logits_per_text)
                total_val_loss += val_loss.item()
                val_acc, val_f1 = calculate_metrics(logits_per_image, ground_truth)
                val_accs.append(val_acc)
                val_f1s.append(val_f1)

        avg_val_loss = total_val_loss / len(val_dataloader)
        validation_losses.append(avg_val_loss)
        if epoch >= 1:
            plot_training_info(epoch, training_losses, validation_losses, logits_images, logits_texts)
        
        epoch_val_acc = sum(val_accs) / len(val_accs)
        epoch_val_f1 = sum(val_f1s) / len(val_f1s)   
        
        if epoch >= 1:
            # Plot losses
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, epoch + 2), training_losses, label='Training Loss')
            plt.plot(range(1, epoch + 2), validation_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss Over Epochs')
            plt.legend()
            plt.savefig(f"{plots_folder}/loss_plot_epoch_{epoch + 1}.png")
            plt.close()        
        
        
        print(Fore.YELLOW + "======================== STATS =============================")
        print(Fore.YELLOW + f"Epoch {epoch + 1}/{EPOCHS} - Validation Acc: {epoch_val_acc:.4f}, Validation F1: {epoch_val_f1:.4f}")
        print(Fore.YELLOW + f"Epoch {epoch + 1}/{EPOCHS} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print(Fore.YELLOW + "============================================================" + Style.RESET_ALL)
        
        with open(f"{text_logs_folder}/log_training.txt", "a", encoding='utf-8') as f:
            f.write("======================== STATS =============================\n")
            f.write(f"Epoch {epoch + 1}/{EPOCHS} - Validation Acc: {epoch_val_acc:.4f}, Validation F1: {epoch_val_f1:.4f}\n")
            f.write(f"Epoch {epoch + 1}/{EPOCHS} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}\n")
            f.write("============================================================\n")

        # Save model every 5 epochs + save final model
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            '''
            model_path = f"{ft_checkpoints_folder}/clip_ft_{epoch+1}.pt"
            torch.save(model, model_path)      
            '''
            model_path = f"{ft_checkpoints_folder}/clip_ft_{epoch+1}"
            model.save_pretrained(model_path)  # Save the model in Hugging Face format
            processor.save_pretrained(model_path)  # Save the processor to ensure compatibility with the saved model
            print(Fore.GREEN + f"Model saved: {model_path}" + Style.RESET_ALL)
            

        #PCA Analysis
        #if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
        if (epoch + 1) % 1 == 0 or epoch == EPOCHS - 1:
            name = f"epoch_{epoch + 1}"
            reduced_activations = pca_on_activations(model, val_dataloader)
            plot_pca(reduced_activations, f"{name} PCA Plot", name)


trainloop()
