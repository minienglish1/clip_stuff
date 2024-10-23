##heavily modified from https://github.com/zer0int/CLIP-fine-tune/
##all credits to original author

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

from transformers import CLIPProcessor, CLIPModel
from torch.cuda.amp import autocast, GradScaler
from adabelief_pytorch import AdaBelief

from accelerate import Accelerator



##########
#main
##########

def main():

    ##########
    #classes & functions
    ##########

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

    print("\nbegin main\n")

    ###########
    #configuration
    ###########

    ##argparse
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_text_dirs", type=str, nargs='+', help="list of .list files")
    parser.add_argument("--epochs", type=int, default=50, help="num of training epochs")
    parser.add_argument("--lr", type=float, default=5e-7, help="learning rate")  # Corrected learning rate
    parser.add_argument("--batch_size", type=int, default=40, help="batch_size")
    args = parser.parse_args()

    #recommended batch_size: >=40
    #learning_rate: 1e-5 to 1e-7. 1e-5 will overfit with bsz 40
    #Epochs: Defaulting to saving every 5 epochs
    # If the val loss is not decreasing with loss or val even increases
    image_text_dirs = args.image_text_dirs
    epochs = args.epochs
    learning_rate = args.lr  # Corrected access to lr argument
    batch_size = args.batch_size


    ##setup accelerator object
    accelerator = Accelerator(mixed_precision="fp16")


    ##variables & folders

    #core components
    #scaler = GradScaler()
    device = accelerator.device
    unfreeze_all = True # Advanced: Unfreeze all of CLIP (default). Set to "False" to unfreeze slowly over X epochs. See the "def unfreeze" above for details.

    #losses
    training_losses = []
    validation_losses = []

    ##folders
    #matplotlib plots folder:
    plots_folder = 'ft-plots'
    os.makedirs(plots_folder, exist_ok=True)

    #model checkpoints folder: 
    ft_checkpoints_folder = 'ft-checkpoints'
    os.makedirs(ft_checkpoints_folder, exist_ok=True)

    #logs folder:
    text_logs_folder = 'ft-logs'
    os.makedirs(text_logs_folder, exist_ok=True)


    ##load CLIP model:
    print("\nloading clip model")
    '''
    #original CLIP
    clipmodel = 'ViT-L/14'
    model, preprocess = clip.load(clipmodel, device=device)
    '''
    #huggingface model
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    #set to full precision to avoid NaN during training
    model = model.float()



    ###########
    #dataset
    ###########
    print("\ncreating dataset")
    '''
    # Define your own dataset and dataloader
    dataset1 = ImageTextDataset("path/to/image/folder", "path/to/my-text-labels.json", transform=preprocess)
    dataset2 = ImageTextDataset("path/to/image/folder/jitter-augmentation", "path/to/my-text-labels.json", transform=preprocess)
    dataset3 = ImageTextDataset("path/to/image/folder/flip-augmentation", "path/to/my-text-labels.json", transform=preprocess)
    '''

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

    total_steps = len(train_dataloader) * epochs

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



    print("\nconfiguring optimizer")
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


    #default
    #optimizer = AdaBelief(model.parameters(), lr=learning_rate, eps=1e-16, betas=(0.9, 0.999), weight_decay=1e-2, weight_decouple=False, rectify=True, print_change_log = False)

    #for "difficult" dataset
    optimizer = AdaBelief(model.parameters(), lr=learning_rate, eps=1e-16, betas=(0.9, 0.995), weight_decay=1e-3, weight_decouple=False, rectify=True, print_change_log = False)

    # Setup scheduler with a proportional warm-up phase. You may want to try anneal_strategy='cos' for cosine.
    # pct_start=0.1 means that 10% of the training steps will be dedicated to ramping up the learning rate.
    # anneal_strategy='linear': Gradually reduces the learning rate in a straight line from its maximum value to its minimum.
    # anneal_strategy='cos': Reduces the learning rate following a cosine curve, providing a smoother transition at the beginning and end.
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps, pct_start=0.1, anneal_strategy='linear')


    ##accelerate prepare
    print("\naccelerate prepare")
    #change logit_scale from scalar to 1D tensor for FSDP compatibility
    model.logit_scale = torch.nn.Parameter(torch.ones(1) * model.logit_scale.item())
    #when using FSDP, prepare model first
    model = accelerator.prepare(model)
    #then prepare everything else
    optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        optimizer, train_dataloader, val_dataloader, scheduler
    )


    print(f"Precision: {model.dtype}")
    print(f'Total batches: {len(train_dataloader)} @ Batch Size: {batch_size}')
    print("== START == \n")


    def trainloop():

        class ContrastiveLoss(nn.Module):
            def __init__(self):
                super(ContrastiveLoss, self).__init__()
                self.criterion = nn.CrossEntropyLoss()

            def forward(self, logits_per_image, logits_per_text):
                labels = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
                loss_img = self.criterion(logits_per_image, labels)
                loss_txt = self.criterion(logits_per_text, labels)
                return (loss_img + loss_txt) / 2

        contrastive_loss = ContrastiveLoss()
        logits_images = []
        logits_texts = []

        """
        def pca_on_activations(model, dataloader, processor, device):
            model.eval()
            activations = []

            with torch.no_grad():
                for images, _ in dataloader:
                    # Convert images to RGB and process them
                    images = [Image.open(image).convert("RGB") for image in images]
                    inputs = processor(images=images, return_tensors="pt", padding=True, truncation=True).to(device)


                    # Ensure dtype consistency between inputs and model weights
                    features = model.vision_model(inputs['pixel_values']).last_hidden_state  # Get vision model embeddings
                    pooled_features = features.mean(dim=1)  # Pool the features over the sequence length

                    # Append pooled features to activations
                    activations.append(pooled_features.cpu().numpy())  # Convert to NumPy and append

            # Perform PCA on the pooled activations
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
        
        print("\npca analysis")
        name = f"base_CLIP"
        reduced_activations = pca_on_activations(model, val_dataloader, processor, device)
        plot_pca(reduced_activations, f"{name} PCA Plot", name)
        """


        for epoch in range(epochs):
            gradient_norms = {}
            unfreeze_layers(model, epoch, total_layers=24, unfreeze_all=unfreeze_all)
            model.train()
            total_train_loss = 0.0
            train_dataloader_prog = train_dataloader
            train_dataloader_all = train_dataloader
            progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{epochs}', leave=True)
            for batch_idx, (images, texts) in progress_bar:
                '''
                images, texts = images.to(device), texts.to(device)
                '''
                #print("convert to rgb")
                images = [Image.open(image).convert("RGB") for image in images]
                #inputs = processor(images=images, text=texts, return_tensors="pt", padding=True).to(device)
                #print("move to device")
                inputs = processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=True).to(device)

                train_accs, train_f1s, val_accs, val_f1s = [], [], [], []
                batch_logits_images = []
                batch_logits_texts = []


                optimizer.zero_grad()
                #with autocast():
                with accelerator.autocast():
                    #print("forward pass")
                    '''
                    logits_per_image, logits_per_text = model(images, texts)
                    '''
                    outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    logits_per_text = outputs.logits_per_text
                    
                    current_batch_size = inputs['pixel_values'].size(0)  # or inputs['input_ids'] for the text batch size
                    ground_truth = torch.arange(current_batch_size, device=device)
                    total_loss = contrastive_loss(logits_per_image, logits_per_text)
                    acc, f1 = calculate_metrics(logits_per_image, ground_truth)
                    train_accs.append(acc)
                    train_f1s.append(f1)

                    #print("backward pass")
                    accelerator.backward(total_loss)
                    optimizer.step()
                    scheduler.step()


                    #scaler.scale(total_loss).backward()
                    #scaler.step(optimizer)
                    #scheduler.step()
                    #scaler.update()


                batch_logits_images.append(logits_per_image.mean().item())
                batch_logits_texts.append(logits_per_text.mean().item())
                        
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

                progress_bar.set_postfix({'loss': f'{total_train_loss / (batch_idx + 1):.4f}  --  Logits Image: {batch_logits_images[-1]:.3f}, Text: {batch_logits_texts[-1]:.3f}'})

                epoch_train_acc = sum(train_accs) / len(train_accs)
                epoch_train_f1 = sum(train_f1s) / len(train_f1s)
                with open(f"{text_logs_folder}/log_details_train.txt", "a", encoding='utf-8') as f:
                    f.write(f"Epoch {epoch + 1}/{epochs}, Batch: {batch_idx + 1}/{len(train_dataloader)}, Loss: {total_loss.item():.4f}, Training Acc: {epoch_train_acc:.4f}, Training F1: {epoch_train_f1:.4f}\n")

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

                    with accelerator.autocast():
                        outputs = model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        logits_per_text = outputs.logits_per_text

                    current_batch_size = inputs['pixel_values'].size(0)
                    #ground_truth = torch.arange(current_batch_size, device=device)
                    ground_truth = torch.arange(current_batch_size, device=logits_per_image.device)

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
            print(Fore.YELLOW + f"Epoch {epoch + 1}/{epochs} - Validation Acc: {epoch_val_acc:.4f}, Validation F1: {epoch_val_f1:.4f}")
            print(Fore.YELLOW + f"Epoch {epoch + 1}/{epochs} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
            print(Fore.YELLOW + "============================================================" + Style.RESET_ALL)
            
            with open(f"{text_logs_folder}/log_training.txt", "a", encoding='utf-8') as f:
                f.write("======================== STATS =============================\n")
                f.write(f"Epoch {epoch + 1}/{epochs} - Validation Acc: {epoch_val_acc:.4f}, Validation F1: {epoch_val_f1:.4f}\n")
                f.write(f"Epoch {epoch + 1}/{epochs} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}\n")
                f.write("============================================================\n")

            # Save model every 5 epochs + save final model
            if (epoch + 1) % 1 == 0 or epoch == epochs - 1:
                '''
                model_path = f"{ft_checkpoints_folder}/clip_ft_{epoch+1}.pt"
                torch.save(model, model_path)      
                '''
                print("\nsaving model")
                model_path = f"{ft_checkpoints_folder}/clip_ft_{epoch+1}"
                model.save_pretrained(model_path)  # Save the model in Hugging Face format
                processor.save_pretrained(model_path)  # Save the processor to ensure compatibility with the saved model
                print(Fore.GREEN + f"Model saved: {model_path}" + Style.RESET_ALL)
                
            """
            #PCA Analysis
            #if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            if (epoch + 1) % 1 == 0 or epoch == epochs - 1:
                print("\nrunning pca analysis")
                name = f"epoch_{epoch + 1}"
                reduced_activations = pca_on_activations(model, val_dataloader, processor, device)
                plot_pca(reduced_activations, f"{name} PCA Plot", name)
            """


    trainloop()


if __name__ == "__main__":
    main()