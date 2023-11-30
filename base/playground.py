from loss_functions import MSELoss_Block
from layers import *
from data import *
from calcs import *
import torch
from transformers_blocks.transformers_tokenizers import *
from transformers_blocks.transformers_models import *
from diffusers_blocks.diffusers_schedulers import *
from diffusers_blocks.diffusers_models import *
from bridges import *

my_playground = Playground("my_playground")
my_playground.mode = "train"


def test01():
    t = torch.tensor([[1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10]], dtype=torch.float, device="cuda")
    data_block = Tensor_Block(my_playground)
    data_block.set_parameters({"tensor": t})
    data_block.set_device("cuda")
    data_block.set_custom_name("data block")

    linear_block = Linear_Block(my_playground)
    linear_block.set_parameters({'in_features': 10, 'out_features': 2, 'bias': True, 'device': 'cuda', 'dtype': torch.float32})
    linear_block.set_custom_name("linear block")

    print_block = Print_Block(my_playground)
    print_block.set_custom_name("print block")

    bridge1 = Accumulate_Forward_Bridge(data_block, linear_block)
    bridge2 = Accumulate_Forward_Bridge(linear_block, print_block)

    bridge1.forward()
    bridge2.forward()


def test02():
    label = torch.tensor([[5, 5], [1, 10]], dtype=torch.float, device="cuda")
    label_block = Tensor_Block(my_playground)
    label_block.set_parameters({"tensor": label})
    label_block.set_device("cuda")
    label_block.set_custom_name("label block")

    t = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.float, device="cuda")
    data_block = Tensor_Block(my_playground)
    data_block.set_parameters({"tensor": t})
    data_block.set_device("cuda")
    data_block.set_custom_name("data block")

    linear_block1 = Linear_Block(my_playground)
    linear_block1.set_parameters({'in_features': 10, 'out_features': 8, 'bias': True, 'device': 'cuda', 'dtype': torch.float32})
    linear_block1.set_custom_name("linear block")

    linear_block2 = Linear_Block(my_playground)
    linear_block2.set_parameters({'in_features': 8, 'out_features': 2, 'bias': True, 'device': 'cuda', 'dtype': torch.float32})
    linear_block2.set_custom_name("linear block2")

    network_block = Integrated_Network_Block(my_playground)
    network_block.add_layer(0, linear_block1)
    network_block.add_layer(1, linear_block2)
    network_block.set_custom_name("network block")

    loss_block = MSELoss_Block(my_playground)
    loss_block.set_custom_name("loss block")

    print_block = Print_Block(my_playground)
    print_block.set_custom_name("print block")

    back_block = Backward_Block(my_playground)
    back_block.set_custom_name("back block")

    optimizer_block = Optimizer_Block("SGD", my_playground)
    optimizer_block.set_parameters({'target network': 'network block', 'lr': 0.001, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'maximize': False, 'foreach': None, 'differentiable': False})
    optimizer_block.set_custom_name("optimizer block")

    bridge1 = Accumulate_Forward_Bridge(data_block, network_block)
    bridge2 = Accumulate_Forward_Bridge(network_block, loss_block)
    bridge3 = Accumulate_Forward_Bridge(label_block, loss_block)
    bridge4 = Accumulate_Forward_Bridge(loss_block, print_block)
    bridge5 = Accumulate_Forward_Bridge(print_block, back_block)
    bridge6 = Accumulate_Forward_Bridge(back_block, optimizer_block)

    bridge1.forward()
    bridge2.forward()
    bridge3.forward()
    bridge4.forward()
    bridge5.forward()
    bridge6.forward()


def test03():
    dataset = Image_Dataset_Block_for_single_folder_with_RE(my_playground)
    dataset.set_parameters({"root dir": "D:\\datasets\\train", "re expressions": ["cat.*", "dog.*"]})
    dataset.add_transformation(0, "RandomCrop")
    dataset.image_transformations[0].set_parameters({'size': (4, 4), 'padding': None, 'pad_if_needed': False, 'fill': 0, 'padding_mode': 'constant'})
    dataset.set_custom_name("dataset")

    dataloader = Dataloader_Block(my_playground)
    dataloader.set_parameters({'batch_size': 4,
                                'shuffle': True,
                                'sampler': None,
                                'batch_sampler': None,
                                'num_workers': 0,
                                'collate_fn': None,
                                'pin_memory': False,
                                'drop_last': False,
                                'timeout': 0,
                                'worker_init_fn': None})
    dataloader.set_custom_name("dataloader")

    print_block = Print_Block(my_playground)
    print_block.set_custom_name("print block")

    bridge1 = Accumulate_Forward_Bridge(dataset, dataloader)
    bridge2 = Accumulate_Forward_Bridge(dataloader, print_block)

    bridge1.forward()
    bridge2.forward()


def test04():
    my_playground.print_flow = False
    my_playground.add_global_variable("height")
    my_playground.set_global_variable("height", 512)
    my_playground.add_global_variable("width")
    my_playground.set_global_variable("width", 512)
    my_playground.add_global_variable("num_inference_steps")
    my_playground.set_global_variable("num_inference_steps", 50)
    my_playground.add_global_variable("guidance_scale")
    my_playground.set_global_variable("guidance_scale", 7.5)
    my_playground.add_global_variable("loop1")
    my_playground.set_global_variable("loop1", 0)

    string_list_block = List_Block(my_playground)
    string_list_block.set_parameters({"list": "['hello world']"})
    string_list_block.set_custom_name("string list block")

    tokenizer_temp_block = CLIPTokenizer_Block(my_playground)
    tokenizer_temp_block.set_parameters({"pretrained": "openai/clip-vit-large-patch14", "advanced_pretrained": {}})
    tokenizer_temp_block.set_custom_name("tokenizer temp block")

    attribute_block = Attribute_Block_Block(my_playground)
    attribute_block.set_parameters({"attribute name": "model_max_length"})
    attribute_block.set_custom_name("attribute block")

    tokenizer_block = CLIPTokenizer_Block(my_playground)
    tokenizer_block.set_parameters({"pretrained": "openai/clip-vit-large-patch14", "advanced_pretrained": {}})
    tokenizer_block.set_forward_parameters({'text': None, 'text_pair': None, 'text_target': None, 'text_pair_target': None, 'add_special_tokens': True, 'padding': 'max_length', 'truncation': True, 'max_length': None, 'stride': 0, 'is_split_into_words': False, 'pad_to_multiple_of': None, 'return_tensors': 'pt', 'return_token_type_ids': None, 'return_attention_mask': None, 'return_overflowing_tokens': False, 'return_special_tokens_mask': False, 'return_offsets_mapping': False, 'return_length': False, 'verbose': True})
    tokenizer_block.set_custom_name("tokenizer block")

    idx_block = Idx_Block(my_playground)
    idx_block.set_forward_parameters({"target": None, "idx": "input_ids"})
    idx_block.set_custom_name("idx block")

    idx_block2 = Idx_Block(my_playground)
    idx_block2.set_forward_parameters({"target": None, "idx": "input_ids"})
    idx_block2.set_custom_name("idx block2")

    encoder_block = CLIPTextModel_Block(my_playground)
    encoder_block.set_parameters({"pretrained": 'openai/clip-vit-large-patch14', "advanced_pretrained": {}})
    encoder_block.set_custom_name("encoder block")

    attribute_data_block = Attribute_Data_Block(my_playground)
    attribute_data_block.set_parameters({"attribute name": "last_hidden_state"})
    attribute_data_block.set_custom_name("attribute data block")

    attribute_data_block2 = Attribute_Data_Block(my_playground)
    attribute_data_block2.set_parameters({"attribute name": "last_hidden_state"})
    attribute_data_block2.set_custom_name("attribute data block2")

    merge_block_1 = Merge_Block(my_playground)
    merge_block_1.set_custom_name("merge block 1")

    list_block = List_Block(my_playground)
    list_block.set_parameters({"list": "['']"})
    list_block.set_custom_name("list block")

    len_block = Len_Block(my_playground)
    len_block.set_custom_name("len block")

    multiply_block = Multiply_Block(my_playground)
    multiply_block.set_custom_name("multiply block")

    print_block = Print_Block(my_playground)
    print_block.set_custom_name("print block")

    text_embedding = Tensor_Cat_Block(my_playground)
    text_embedding.set_custom_name("text embedding")

    tensor_size_block = Tensor_Size_Block(my_playground)
    tensor_size_block.set_custom_name("tensor size block")

    scheduler = LMSDiscreteScheduler_Block(my_playground)
    scheduler.set_parameters({'num_train_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012, 'beta_schedule': 'scaled_linear', 'trained_betas': None, 'prediction_type': 'epsilon'})
    scheduler.set_custom_name("scheduler")

    int_block1 = Int_Block(my_playground)
    int_block1.set_parameters({"int": my_playground.get_global_variable("num_inference_steps")})
    int_block1.set_custom_name("int block1")

    sigmas_ = Attribute_Block_Block(my_playground)
    sigmas_.set_parameters({"attribute name": "sigmas"})
    sigmas_.set_custom_name("sigmas_")

    sigmas = Attribute_Block_Block(my_playground)
    sigmas.set_parameters({"attribute name": "sigmas"})
    sigmas.set_custom_name("sigmas")

    loop1_block = Get_Global_Variable_Block(my_playground)
    loop1_block.set_parameters({"global variable": "loop1"})
    loop1_block.set_custom_name("loop1")

    loop1_increment_block = Increase_Global_Variable_Block(my_playground)
    loop1_increment_block.set_parameters({"global variable": "loop1", "increment": 1})
    loop1_increment_block.set_custom_name("loop1 increment block")

    sigma_ = Idx_Block(my_playground)
    sigma_.set_custom_name("sigma_")

    sigma = Idx_Block(my_playground)
    sigma.set_custom_name("sigma")

    text_embeddings_shape = Attribute_Data_Block(my_playground)
    text_embeddings_shape.set_parameters({"attribute name": "shape"})
    text_embeddings_shape.set_custom_name("text_embeddings.shape")

    text_embeddings_shape_0 = Idx_Block(my_playground)
    text_embeddings_shape_0.set_parameters({"target": any, "idx": 0})
    text_embeddings_shape_0.set_custom_name("text_embeddings.shape[0]")

    text_embeddings_shape_0_divide_2 = Floor_Divide_Block(my_playground)
    text_embeddings_shape_0_divide_2.set_forward_parameters({"dividend": None, "divisor": 2})
    text_embeddings_shape_0_divide_2.set_custom_name("text_embeddings_shape_0_divide_2")

    unet_ = UNet2DConditionModel_Block(my_playground)
    unet_.set_parameters({"pretrained": 'CompVis/stable-diffusion-v1-4', "advanced_pretrained": {"subfolder": "unet"}})
    unet_.set_custom_name("unet_")

    unet_inchannels = Attribute_Block_Block(my_playground)
    unet_inchannels.set_parameters({"attribute name": "in_channels"})
    unet_inchannels.set_custom_name("unet.in_channels")

    height = Get_Global_Variable_Block(my_playground)
    height.set_parameters({"global variable": "height"})
    height.set_custom_name("height")

    height_div_8 = Floor_Divide_Block(my_playground)
    height_div_8.set_forward_parameters({"dividend": None, "divisor": 8})
    height_div_8.set_custom_name("height div 8")

    width = Get_Global_Variable_Block(my_playground)
    width.set_parameters({"global variable": "width"})
    width.set_custom_name("width")

    width_div_8 = Floor_Divide_Block(my_playground)
    width_div_8.set_forward_parameters({"dividend": None, "divisor": 8})
    width_div_8.set_custom_name("width div 8")

    shape = Pack_List_Block(my_playground)
    shape.set_custom_name("shape")

    random_tensor = Random_Tensor_Block(my_playground)
    random_tensor.set_custom_name("random tensor")

    latent = Multiply_Block(my_playground)
    latent.set_custom_name("latent")

    latent_model_input = Code_Block(my_playground)
    latent_model_input.set_code("""
return torch.cat([args[0]] * 2) / ((args[1]**2 + 1) ** 0.5)
    """)
    latent_model_input.set_parameters("latent model input")

    loop1_block_ = Get_Global_Variable_Block(my_playground)
    loop1_block_.set_parameters({"global variable": "loop1"})
    loop1_block_.set_custom_name("loop1_")

    unet = UNet2DConditionModel_Block(my_playground)
    unet.set_parameters({"pretrained": 'CompVis/stable-diffusion-v1-4', "advanced_pretrained": {"subfolder": "unet"}})
    unet.set_custom_name("unet")

    noise_pred = Idx_Block(my_playground)
    noise_pred.set_parameters({"target": None, "idx": "sample"})
    noise_pred.set_custom_name("noise pred")

    scheduler_timesteps = Attribute_Block_Block(my_playground)
    scheduler_timesteps.set_parameters({"attribute name": "timesteps"})
    scheduler_timesteps.set_custom_name("scheduler timesteps")

    t = Idx_Block(my_playground)
    t.set_custom_name("t")

    t_to_gpu = ToGPU(my_playground)
    t_to_gpu.set_custom_name("t_to_gpu")

    text_embedding_to_gpu = ToGPU(my_playground)
    text_embedding_to_gpu.set_custom_name("text_embedding_to_gpu")

    noise_pred_chunk = Chunk_Block(my_playground)
    noise_pred_chunk.set_forward_parameters({"input": None, "chunks": 2, "dim": 0})
    noise_pred_chunk.set_custom_name("noise_pred_chunk")

    noise_pred_uncond = Idx_Block(my_playground)
    noise_pred_uncond.set_forward_parameters({"target": any, "idx": 0})
    noise_pred_uncond.set_custom_name("noise_pred_uncond")

    noise_pred_text = Idx_Block(my_playground)
    noise_pred_text.set_forward_parameters({"target": any, "idx": 1})
    noise_pred_text.set_custom_name("noise_pred_text")

    guidance_scale = Get_Global_Variable_Block(my_playground)
    guidance_scale.set_parameters({"global variable": "guidance_scale"})
    guidance_scale.set_custom_name("guidance_scale")

    actual_noise_pred = Code_Block(my_playground)
    actual_noise_pred.set_code("""
return args[0] + args[2] * (args[1] - args[0])
    """)

    scheduler_step = Attribute_Block_Block(my_playground)
    scheduler_step.set_parameters({"attribute name": "step"})
    scheduler_step.set_custom_name("scheduler_step")

    call_scheduler_step = Call_Method_Block(my_playground)
    call_scheduler_step.set_custom_name("call_scheduler_step")

    prev_sample = Idx_Block(my_playground)
    prev_sample.set_forward_parameters({"target": any, "idx": "prev_sample"})

    vae = AutoencoderKL_Block(my_playground)
    vae.set_parameters({"pretrained": 'CompVis/stable-diffusion-v1-4', "advanced_pretrained": {"subfolder": "vae"}})
    vae.set_custom_name("vae")

    float_block = Float_Block(my_playground)
    float_block.set_parameters({"float": "5.489980785067252"})
    float_block.set_custom_name("float block")

    latent_for_decode = Multiply_Block(my_playground)
    latent_for_decode.set_custom_name("latent_for_decode")

    vae_decode = Attribute_Block_Block(my_playground)
    vae_decode.set_parameters({"attribute name": "decode"})
    vae_decode.set_custom_name("vae_decode")

    call_vae_decode = Call_Method_Block(my_playground)
    call_vae_decode.set_custom_name("call_vae_decode")

    pil_images = Code_Block(my_playground)
    pil_images.set_code("""
    imgs = (args[0] / 2 + 0.5).clamp(0, 1)
    imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    imgs = (imgs * 255).round().astype('uint8')
    pil_images = [Image.fromarray(image) for image in imgs]
    return pil_images
        """)
    pil_images.set_custom_name("actual_noise_pred")

    save = Save_PIL_Block(my_playground)
    save.set_forward_parameters({"path": "a.jpg"})
    save.set_custom_name("save")






    bridge1 = Sleep_Bridge(tokenizer_temp_block, attribute_block)
    bridge2 = Accumulate_Forward_Bridge(attribute_block, tokenizer_block, "max_length")
    bridge3 = Accumulate_Forward_Bridge(string_list_block, tokenizer_block)
    bridge4 = Accumulate_Forward_Bridge(tokenizer_block, idx_block)
    bridge5 = Accumulate_Forward_Bridge(idx_block, encoder_block)
    bridge6 = Accumulate_Forward_Bridge(encoder_block, attribute_data_block)
    bridge7 = Accumulate_Forward_Bridge(attribute_data_block, merge_block_1)
    bridge8 = Accumulate_Forward_Bridge(list_block, multiply_block)
    bridge9 = Accumulate_Forward_Bridge(string_list_block, len_block)
    bridge10 = Accumulate_Forward_Bridge(len_block, multiply_block)
    bridge11 = Clear_Forward_Bridge(multiply_block, tokenizer_block)
    bridge12 = Accumulate_Forward_Bridge(attribute_block, tokenizer_block, "max_length")
    bridge13 = Accumulate_Forward_Bridge(tokenizer_block, idx_block2)
    bridge14 = Clear_Forward_Bridge(idx_block2, encoder_block)
    bridge15 = Accumulate_Forward_Bridge(encoder_block, attribute_data_block2)
    bridge16 = Accumulate_Forward_Bridge(attribute_data_block2, merge_block_1)
    bridge17 = Accumulate_Forward_Bridge(merge_block_1, text_embedding)
    bridge18 = Accumulate_Forward_Bridge(text_embedding, tensor_size_block)
    bridge19 = Clear_Forward_Bridge(tensor_size_block, print_block)
    bridge20 = Call_Method_Bridge(scheduler, int_block1)
    bridge20.set_func_name("set_timesteps")
    bridge21 = Sleep_Bridge(scheduler, sigmas_)
    bridge22 = Clear_Forward_Bridge(sigmas_, sigma_)
    bridge23 = Accumulate_Forward_Bridge(loop1_block, sigma_)
    bridge24 = Accumulate_Forward_Bridge(text_embedding, text_embeddings_shape)
    bridge25 = Accumulate_Forward_Bridge(text_embeddings_shape, text_embeddings_shape_0)
    bridge26 = Accumulate_Forward_Bridge(text_embeddings_shape_0, text_embeddings_shape_0_divide_2)
    bridge27 = Sleep_Bridge(unet_, unet_inchannels)
    bridge28 = Accumulate_Forward_Bridge(height, height_div_8)
    bridge29 = Accumulate_Forward_Bridge(width, width_div_8)
    bridge30 = Accumulate_Forward_Bridge(text_embeddings_shape_0_divide_2, shape)
    bridge31 = Accumulate_Forward_Bridge(unet_inchannels, shape)
    bridge32 = Accumulate_Forward_Bridge(height_div_8, shape)
    bridge33 = Accumulate_Forward_Bridge(width_div_8, shape)
    bridge34 = Accumulate_Forward_Bridge(shape, random_tensor)
    bridge35 = Clear_Forward_Bridge(random_tensor, latent)
    bridge36 = Accumulate_Forward_Bridge(sigma_, latent)

    bridge37 = Sleep_Bridge(scheduler, sigmas)
    bridge38 = Clear_Forward_Bridge(sigmas, sigma)
    bridge39 = Accumulate_Forward_Bridge(loop1_block, sigma)
    bridge40 = Clear_Forward_Bridge(latent, latent_model_input)
    bridge41 = Accumulate_Forward_Bridge(sigma, latent_model_input)
    bridge42 = Accumulate_Forward_Bridge(scheduler, scheduler_timesteps)
    bridge43 = Accumulate_Forward_Bridge(scheduler_timesteps, t)
    bridge44 = Accumulate_Forward_Bridge(loop1_block, t)
    bridge45 = Clear_Forward_Bridge(t, t_to_gpu)
    bridge46 = Clear_Forward_Bridge(text_embedding, text_embedding_to_gpu)
    bridge47 = Clear_Forward_Bridge(latent_model_input, unet)
    bridge48 = Accumulate_Forward_Bridge(t_to_gpu, unet)
    bridge49 = Accumulate_Forward_Bridge(text_embedding_to_gpu, unet, "encoder_hidden_states")
    bridge50 = Clear_Forward_Bridge(unet, noise_pred)
    bridge51 = Clear_Forward_Bridge(noise_pred, noise_pred_chunk)
    bridge52 = Clear_Forward_Bridge(noise_pred_chunk, noise_pred_uncond)
    bridge53 = Clear_Forward_Bridge(noise_pred_chunk, noise_pred_text)
    bridge54 = Clear_Forward_Bridge(noise_pred_uncond, actual_noise_pred)
    bridge55 = Accumulate_Forward_Bridge(noise_pred_text, actual_noise_pred)
    bridge56 = Accumulate_Forward_Bridge(guidance_scale, actual_noise_pred)
    bridge57 = Clear_Forward_Bridge(scheduler, scheduler_step)
    bridge58 = Clear_Forward_Bridge(scheduler_step, call_scheduler_step)
    bridge59 = Accumulate_Forward_Bridge(actual_noise_pred, call_scheduler_step)
    bridge60 = Accumulate_Forward_Bridge(loop1_block, call_scheduler_step)
    bridge61 = Accumulate_Forward_Bridge(latent, call_scheduler_step)
    bridge62 = Clear_Forward_Bridge(call_scheduler_step, prev_sample)
    bridge63 = Clear_Forward_Bridge(prev_sample, latent)
    bridge63_5 = Clear_Forward_Bridge(prev_sample, loop1_increment_block)

    bridge64 = Clear_Forward_Bridge(latent, latent_for_decode)
    bridge65 = Sleep_Bridge(vae, vae_decode)
    bridge66 = Clear_Forward_Bridge(vae_decode, call_vae_decode)
    bridge67 = Accumulate_Forward_Bridge(latent_for_decode, call_vae_decode)
    bridge68 = Clear_Forward_Bridge(call_vae_decode, pil_images)
    bridge69 = Clear_Forward_Bridge(pil_images, save)


    bridge1.forward()
    bridge2.forward()
    bridge3.forward()
    bridge4.forward()
    bridge5.forward()
    bridge6.forward()
    bridge7.forward()
    bridge8.forward()
    bridge9.forward()
    bridge10.forward()
    bridge11.forward()
    bridge12.forward()
    bridge13.forward()
    bridge14.forward()
    bridge15.forward()
    bridge16.forward()
    bridge17.forward()
    bridge18.forward()
    bridge19.forward()
    bridge20.forward()
    bridge21.forward()
    bridge22.forward()
    bridge23.forward()
    bridge24.forward()
    bridge25.forward()
    bridge26.forward()
    bridge27.forward()
    bridge28.forward()
    bridge29.forward()
    bridge30.forward()
    bridge31.forward()
    bridge32.forward()
    bridge33.forward()
    bridge34.forward()
    bridge35.forward()
    bridge36.forward()

    bridge37.forward()
    bridge38.forward()
    bridge39.forward()
    bridge40.forward()
    bridge41.forward()
    bridge42.forward()
    bridge43.forward()
    bridge44.forward()
    bridge45.forward()
    bridge46.forward()
    bridge47.forward()
    bridge48.forward()
    bridge49.forward()
    bridge50.forward()
    bridge51.forward()
    bridge52.forward()
    bridge53.forward()
    bridge54.forward()
    bridge55.forward()
    bridge56.forward()
    bridge57.forward()
    bridge58.forward()
    bridge59.forward()
    bridge60.forward()
    bridge61.forward()
    bridge62.forward()
    bridge63.forward()
    bridge63_5.forward()
    print(latent.in_data[0].size())

    bridge64.forward()
    bridge65.forward()
    bridge66.forward()
    bridge67.forward()
    bridge68.forward()
    bridge69.forward()


def test05():
    my_playground.print_flow = False
    my_playground.add_global_variable("height")
    my_playground.set_global_variable("height", 512)
    my_playground.add_global_variable("width")
    my_playground.set_global_variable("width", 512)
    my_playground.add_global_variable("num_inference_steps")
    my_playground.set_global_variable("num_inference_steps", 50)
    my_playground.add_global_variable("guidance_scale")
    my_playground.set_global_variable("guidance_scale", 7.5)
    my_playground.add_global_variable("loop1")
    my_playground.set_global_variable("loop1", 0)

    scheduler = LMSDiscreteScheduler_Block(my_playground)
    scheduler.set_parameters({'num_train_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012, 'beta_schedule': 'scaled_linear', 'trained_betas': None, 'prediction_type': 'epsilon'})
    scheduler.set_custom_name("scheduler")

    int_block1 = Int_Block(my_playground)
    int_block1.set_parameters({"int": my_playground.get_global_variable("num_inference_steps")})
    int_block1.set_custom_name("int block1")

    loop1_block = Get_Global_Variable_Block(my_playground)
    loop1_block.set_parameters({"global variable": "loop1"})
    loop1_block.set_custom_name("loop1")

    random = Random_Tensor_Block(my_playground)
    random.set_forward_parameters({"shape": [1, 4, 64, 64]})

    latent = Merge_Block(my_playground)

    noise_pred_chunk = Chunk_Block(my_playground)
    noise_pred_chunk.set_forward_parameters({"input": None, "chunks": 2, "dim": 0})
    noise_pred_chunk.set_custom_name("noise_pred_chunk")

    noise_pred_uncond = Idx_Block(my_playground)
    noise_pred_uncond.set_forward_parameters({"target": any, "idx": 0})
    noise_pred_uncond.set_custom_name("noise_pred_uncond")

    noise_pred_text = Idx_Block(my_playground)
    noise_pred_text.set_forward_parameters({"target": any, "idx": 1})
    noise_pred_text.set_custom_name("noise_pred_text")

    guidance_scale = Get_Global_Variable_Block(my_playground)
    guidance_scale.set_parameters({"global variable": "guidance_scale"})
    guidance_scale.set_custom_name("guidance_scale")

    actual_noise_pred = Code_Block(my_playground)
    actual_noise_pred.set_code("""
return args[0] + args[2] * (args[1] - args[0])
    """)
    actual_noise_pred.set_custom_name("actual_noise_pred")

    scheduler_step = Attribute_Block_Block(my_playground)
    scheduler_step.set_parameters({"attribute name": "step"})
    scheduler_step.set_custom_name("scheduler_step")

    call_scheduler_step = Call_Method_Block(my_playground)
    call_scheduler_step.set_custom_name("call_scheduler_step")

    prev_sample = Idx_Block(my_playground)
    prev_sample.set_forward_parameters({"target": any, "idx": "prev_sample"})
    prev_sample.set_custom_name("prev sample")

    test = Random_Tensor_Block(my_playground)
    test.set_forward_parameters({"shape": [2, 4, 64, 64]})
    test.set_custom_name("test")






    vae = AutoencoderKL_Block(my_playground)
    vae.set_parameters({"pretrained": 'CompVis/stable-diffusion-v1-4', "advanced_pretrained": {"subfolder": "vae"}})
    vae.set_custom_name("vae")

    float_block = Float_Block(my_playground)
    float_block.set_parameters({"float": "5.489980785067252"})
    float_block.set_custom_name("float block")

    latent_for_decode = Multiply_Block(my_playground)
    latent_for_decode.set_custom_name("latent_for_decode")

    vae_decode = Attribute_Block_Block(my_playground)
    vae_decode.set_parameters({"attribute name": "decode"})
    vae_decode.set_custom_name("vae_decode")

    call_vae_decode = Call_Method_Block(my_playground)
    call_vae_decode.set_custom_name("call_vae_decode")

    pil_images = Code_Block(my_playground)
    pil_images.set_code("""
imgs = (args[0] / 2 + 0.5).clamp(0, 1)
imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
imgs = (imgs * 255).round().astype('uint8')
pil_images = [Image.fromarray(image) for image in imgs]
return pil_images
    """)
    pil_images.set_custom_name("actual_noise_pred")

    save = Save_PIL_Block(my_playground)
    save.set_forward_parameters({"path": "a.jpg"})
    save.set_custom_name("save")










    bridge0 = Clear_Forward_Bridge(random, latent)
    bridge20 = Call_Method_Bridge(scheduler, int_block1)
    bridge20.set_func_name("set_timesteps")
    bridgetest = Clear_Forward_Bridge(test, noise_pred_chunk)
    bridge52 = Clear_Forward_Bridge(noise_pred_chunk, noise_pred_uncond)
    bridge53 = Clear_Forward_Bridge(noise_pred_chunk, noise_pred_text)
    bridge54 = Clear_Forward_Bridge(noise_pred_uncond, actual_noise_pred)
    bridge55 = Accumulate_Forward_Bridge(noise_pred_text, actual_noise_pred)
    bridge56 = Accumulate_Forward_Bridge(guidance_scale, actual_noise_pred)
    bridge57 = Clear_Forward_Bridge(scheduler, scheduler_step)
    bridge58 = Clear_Forward_Bridge(scheduler_step, call_scheduler_step)
    bridge59 = Accumulate_Forward_Bridge(actual_noise_pred, call_scheduler_step)
    bridge60 = Accumulate_Forward_Bridge(loop1_block, call_scheduler_step)
    bridge61 = Accumulate_Forward_Bridge(latent, call_scheduler_step)
    bridge62 = Clear_Forward_Bridge(call_scheduler_step, prev_sample)
    bridge63 = Clear_Forward_Bridge(prev_sample, latent)

    bridge64 = Clear_Forward_Bridge(latent, latent_for_decode)
    bridge65 = Sleep_Bridge(vae, vae_decode)
    bridge66 = Clear_Forward_Bridge(vae_decode, call_vae_decode)
    bridge67 = Accumulate_Forward_Bridge(latent_for_decode, call_vae_decode)
    bridge68 = Clear_Forward_Bridge(call_vae_decode, pil_images)
    bridge69 = Clear_Forward_Bridge(pil_images, save)







    bridge0.forward()
    bridge20.forward()
    bridgetest.forward()
    bridge52.forward()
    bridge53.forward()
    bridge54.forward()
    bridge55.forward()
    bridge56.forward()
    bridge57.forward()
    bridge58.forward()
    bridge59.forward()
    bridge60.forward()
    bridge61.forward()
    print(call_scheduler_step.in_data[0])
    print(call_scheduler_step.in_data[1].size())
    print(call_scheduler_step.in_data[2])
    print(call_scheduler_step.in_data[3].size())
    bridge62.forward()
    bridge63.forward()
    print(latent.in_data[0].size())


    bridge64.forward()
    bridge65.forward()
    bridge66.forward()
    bridge67.forward()
    bridge68.forward()
    bridge69.forward()

test05()
