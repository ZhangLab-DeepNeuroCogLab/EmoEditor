import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import argparse
from data import EmoPairDataset
from ip2p import InstructPix2Pix
from PIL import Image
import datetime
from emo_model import EmoDirectionEncoder


if __name__ == '__main__':
   
   parser = argparse.ArgumentParser()
    
   parser.add_argument('--dataset_name', type=str, default="EmoPair")
   parser.add_argument('--random_seed', type=int, default=None,
                        help="try a different seed or set it to `None` -- the model will generate one randomly.")
   parser.add_argument('--batch_size', type=int, default=32)
   parser.add_argument('--max_epoch', type=int, default=1000)
      
   parser.add_argument('--save_model_dir', type=str, default="./save_models")
   parser.add_argument('--train_save_dir', type=str, default="./save_train_res")
   parser.add_argument('--image_size', type=int, default=224)
    
   args = parser.parse_args()
   
   now = datetime.datetime.now()
   nowtime = str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute)
   args.save_model_dir = os.path.join(args.save_model_dir, nowtime)
   args.train_save_dir = os.path.join(args.train_save_dir, nowtime)
   
   if not os.path.exists(args.save_model_dir):
      os.makedirs(args.save_model_dir)
   if not os.path.exists(args.train_save_dir):
      os.makedirs(args.train_save_dir)
    
   device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
   
   if args.random_seed != None:
      torch.manual_seed(args.random_seed)
      torch.cuda.manual_seed(args.random_seed)
   
   emo_enc = EmoDirectionEncoder().to(device)
   
   ip2p_model = InstructPix2Pix(device).to(device).half()
   ip2p_model.requires_grad_(False)
   ip2p_model.unet.requires_grad_(True)

   loss_mse = nn.MSELoss().to(device)
   optimizer = optim.Adam(list(emo_enc.parameters()) + list(ip2p_model.unet.parameters()), lr=0.00001, betas=(0.9, 0.999), eps=1e-3)
    
   train_tf = transforms.Compose([
      transforms.Resize((args.image_size, args.image_size)),
      transforms.ToTensor(),
   ])
   
   train_dataset = EmoPairDataset(dataset_name=args.dataset_name, transform=train_tf)
   train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
   
   iter_step = 0

   print("Start training")
   for epoch in range(0, args.max_epoch):
      
      for idx, (prompt, emo_trans, img_source, img_target, emo_direction) in enumerate(train_loader):
         
         emo_enc.train()
         ip2p_model.unet.train()
         
         img_source = img_source.to(device)
         img_target = img_target.to(device)    
         
         prompt = list(prompt)
         text_embedding = ip2p_model.pipe._encode_prompt(prompt, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)
                  
         emo_direction = emo_direction.to(device).float()
         encoder_hidden_states = emo_enc(emo_direction)            
         
         latents = ip2p_model.auto_encoder.encode(img_target.half()).latent_dist.sample()
         latents = latents * ip2p_model.auto_encoder.config.scaling_factor     
         noise = torch.randn_like(latents)
         timesteps = torch.randint(0, ip2p_model.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
         timesteps = timesteps.long()
         noisy_latents = ip2p_model.scheduler.add_noise(latents, noise, timesteps)
         original_image_embeds = ip2p_model.auto_encoder.encode(img_source.half()).latent_dist.mode()         
         concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)
         model_pred = ip2p_model.unet(concatenated_noisy_latents, timesteps, encoder_hidden_states.half()).sample
                  
         # l_cos
         cosine_similarity = F.cosine_similarity(text_embedding.float().view(args.batch_size, -1), encoder_hidden_states.float().view(args.batch_size, -1), dim=1)
         l_emb = torch.mean(1 - cosine_similarity)
         # l_noise
         l_noise = loss_mse(model_pred.float(), noise.float())
         
         # total_loss
         loss = l_noise + l_emb * 0.5

         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
         
         print("epoch: %4d| iter: %4d| loss: %.5f| l_noise: %.5f" % (epoch, iter_step, loss, l_noise))
         
         if iter_step % 10000 == 0:
            with torch.no_grad():
               edited_img = ip2p_model(encoder_hidden_states.half(), img_source.half(), img_source.half())
            ims = torch.cat([img_source, edited_img, img_target], dim=3)[0]
            ims = ims.mul(255).clamp(0, 255).byte()
            ims = ims.permute(1, 2, 0).data.cpu().numpy()
            ims = Image.fromarray(ims)
            fullpath = '%s/epoch%03d_iter%d_%s_%s.png' % (args.train_save_dir, epoch, iter_step, emo_trans[0], prompt[0])
            ims.save(fullpath)
            print("Train image saved.")

         if iter_step % 10000 == 0:
            torch.save(emo_enc.state_dict(), os.path.join(args.save_model_dir, 'epoch%03d_iter%d.pt' % (epoch, iter_step)))
            torch.save(ip2p_model.state_dict(), os.path.join(args.save_model_dir, 'epoch%03d_iter%d_ip2p.pt' % (epoch, iter_step)))
            
         iter_step = iter_step + 1
            
   torch.save(emo_enc.state_dict(), os.path.join(args.save_model_dir, 'epoch%03d_iter%d.pt' % (epoch, iter_step)))     
   torch.save(ip2p_model.state_dict(), os.path.join(args.save_model_dir, 'epoch%03d_iter%d_ip2p.pt' % (epoch, iter_step)))

