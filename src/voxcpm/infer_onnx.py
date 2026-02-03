
import time

from pathlib import Path

import soundfile as sf
import numpy as np
import onnxruntime
from pydub import AudioSegment
from .utils.text_normalize import TextNormalizer
from transformers import LlamaTokenizerFast
                                         # The generated audio path.

def audio_normalizer(_audio, target_value=8192.0):
    _audio = _audio.astype(np.float32)
    rms = np.sqrt(np.mean((_audio * _audio), dtype=np.float32), dtype=np.float32)
    _audio *= (target_value / (rms + 1e-7))
    np.clip(_audio, -32768.0, 32767.0, out=_audio)
    return _audio.astype(np.int16)


def mask_multichar_chinese_tokens(tokenizer):
    multichar_tokens = {
        token for token in tokenizer.vocab.keys()
        if len(token) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in token)
    }

    class CharTokenizerWrapper:
        def __init__(self, base_tokenizer) -> None:
            self.tokenizer = base_tokenizer
            self.multichar_tokens = multichar_tokens

        def tokenize(self, text: str, **kwargs):
            if not isinstance(text, str):
                raise TypeError(f"Expected string input, got {type(text)}")

            tokens = self.tokenizer.tokenize(text, **kwargs)
            processed = []

            for token in tokens:
                clean_token = token.replace("▁", "")

                if clean_token in self.multichar_tokens:
                    chars = list(clean_token)
                    processed.extend(chars)
                else:
                    processed.append(token)

            return processed

        def __call__(self, text: str, **kwargs):
            try:
                tokens = self.tokenize(text, **kwargs)
                result = self.tokenizer.convert_tokens_to_ids(tokens)
                return result
            except Exception as e:
                raise ValueError(f"Tokenization failed: {str(e)}") from e

    return CharTokenizerWrapper(tokenizer)

class VoxcpmOnnx():
    def __init__ (self, model_dir:str):
        self.model_dir = model_dir
        onnx_model_A = str( Path(model_dir) / 'text_embed.onnx')  # Assign a path where the exported VoxCPM model stored.
        onnx_model_B = str( Path(model_dir) / 'vae_encoder.onnx')
        onnx_model_C = str( Path(model_dir) / 'feat_encoder.onnx')
        onnx_model_D = str( Path(model_dir) / 'feat_cond.onnx')
        onnx_model_E = str( Path(model_dir) / 'concat.onnx')
        onnx_model_F = str( Path(model_dir) / 'main.onnx')
        onnx_model_G = str( Path(model_dir) / 'feat_decoder.onnx')
        onnx_model_H = str( Path(model_dir) / 'vae_decoder.onnx')

        # === Decoding limits & tokens ===
        self.STOP_TOKEN = [2] #[1]  # The stop_id in VoxCPM is "1"
        MAX_SEQ_LEN = 1024  # The max decode length; keep the same as exported model.
        self.MIN_SEQ_LEN = 2  # The min decode length
        self.DECODE_LIMIT_FACTOR = 1  # Decode length limit factor, integer >= 1

        # === Audio configuration ===
        self.IN_SAMPLE_RATE = 44100  # Input prompt audio sample rate; keep the same as exported model.
        self.OUT_SAMPLE_RATE = 44100  # Output audio sample rate; keep the same as exported model.

        # === Guidance, diffusion & randomness ===
        FIXED_TIMESTEPS = 10  # Fixed self.timesteps; keep the same as exported model.
        CFG_VALUE = 1 #2.5  # Lower values result in more natural speech for long text, while higher values stay closer to the original sound features.
        RANDOM_SEED = 1  # Global random seed

        # === ONNX / runtime configuration ===
        MAX_THREADS = 0  # Parallel CPU threads, 0 for auto
        self.DEVICE_ID = 0  # Device id, default 0

        ORT_Accelerate_Providers = []  # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
        # else keep empty.

        if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
            provider_options = [
                {
                    'self.device_type': 'CPU',  # [CPU, NPU, GPU, GPU.0, GPU.1]]
                    'precision': 'ACCURACY',  # [FP32, FP16, ACCURACY]
                    'num_of_threads': MAX_THREADS if MAX_THREADS != 0 else 8,  # The default value is 8. Edit freely.
                    'num_streams': 1,
                    'enable_opencl_throttling': False,
                    'enable_qdq_optimizer': False,  # Enable it carefully
                    'disable_dynamic_shapes': False
                }
            ]
            self.device_type = 'cpu'
        elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
            provider_options = [
                {
                    'self.DEVICE_ID': self.DEVICE_ID,
                    'gpu_mem_limit': 24 * 1024 * 1024 * 1024,  # 24 GB
                    'arena_extend_strategy': 'kNextPowerOfTwo',  # ["kNextPowerOfTwo", "kSameAsRequested"]
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',  # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
                    'sdpa_kernel': '2',  # ["0", "1", "2"]
                    'use_tf32': '1',
                    'fuse_conv_bias': '0',  # Set to '0' to avoid potential errors when enabled.
                    'cudnn_conv_use_max_workspace': '1',
                    'cudnn_conv1d_pad_to_nc1d': '0',
                    'tunable_op_enable': '0',
                    'tunable_op_tuning_enable': '0',
                    'tunable_op_max_tuning_duration_ms': 10,
                    'do_copy_in_default_stream': '0',
                    'enable_cuda_graph': '0',  # Set to '0' to avoid potential errors when enabled.
                    'prefer_nhwc': '0',
                    'enable_skip_layer_norm_strict_mode': '0',
                    'use_ep_level_unified_stream': '0',
                }
            ]
            self.device_type = 'cuda'
        elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
            provider_options = [
                {
                    'self.DEVICE_ID': self.DEVICE_ID,
                    'performance_preference': 'high_performance',  # [high_performance, default, minimum_power]
                    'device_filter': 'npu'  # [any, npu, gpu]
                }
            ]
            self.device_type = 'dml'
        else:
            # Please config by yourself for others providers.
            self.device_type = 'cpu'
            provider_options = None

        # settings
        onnxruntime.set_seed(RANDOM_SEED)
        session_opts = onnxruntime.SessionOptions()
        run_options = onnxruntime.RunOptions()
        session_opts.log_severity_level = 4                 # Fatal level, it an adjustable value.
        session_opts.log_verbosity_level = 4                # Fatal level, it an adjustable value.
        run_options.log_severity_level = 4                  # Fatal level, it an adjustable value.
        run_options.log_verbosity_level = 4                 # Fatal level, it an adjustable value.
        session_opts.inter_op_num_threads = MAX_THREADS     # Run different nodes with num_threads. Set 0 for auto.
        session_opts.intra_op_num_threads = MAX_THREADS     # Under the node, execute the operators with num_threads. Set 0 for auto.
        session_opts.enable_cpu_mem_arena = True            # True for execute speed; False for less memory usage.
        session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_opts.add_session_config_entry('session.set_denormal_as_zero', '1')
        session_opts.add_session_config_entry('session.intra_op.allow_spinning', '1')
        session_opts.add_session_config_entry('session.inter_op.allow_spinning', '1')
        session_opts.add_session_config_entry('session.enable_quant_qdq_cleanup', '1')
        session_opts.add_session_config_entry('session.qdq_matmulnbits_accuracy_level', '4')
        session_opts.add_session_config_entry('session.use_device_allocator_for_initializers', '1')
        session_opts.add_session_config_entry('session.graph_optimizations_loop_level', '2')
        session_opts.add_session_config_entry('optimization.enable_gelu_approximation', '1')
        session_opts.add_session_config_entry('optimization.minimal_build_optimizations', '')
        session_opts.add_session_config_entry('optimization.enable_cast_chain_elimination', '1')
        run_options.add_run_config_entry('disable_synchronize_execution_providers', '1')
        
        
        self.ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
        self.in_name_A = self.ort_session_A.get_inputs()
        self.out_name_BA = self.ort_session_A.get_outputs()
        self.in_name_A = self.in_name_A[0].name
        self.out_name_BA = [self.out_name_BA[0].name]
        
        self.ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
        self.in_name_B = self.ort_session_B.get_inputs()
        self.out_name_BB = self.ort_session_B.get_outputs()
        self.in_name_B = self.in_name_B[0].name
        self.out_name_BB = [self.out_name_BB[0].name]
        
        self.ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
        self.in_name_C = self.ort_session_C.get_inputs()
        self.out_name_BC = self.ort_session_C.get_outputs()
        self.in_name_C = self.in_name_C[0].name
        self.out_name_BC = [self.out_name_BC[0].name]
        
        self.ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
        model_dtype_D = self.ort_session_D._inputs_meta[0].type
        if 'float16' in model_dtype_D:
            model_dtype_D = np.float16
        else:
            model_dtype_D = np.float32
        self.in_name_D = self.ort_session_D.get_inputs()
        self.out_name_BD = self.ort_session_D.get_outputs()
        self.in_name_D = self.in_name_D[0].name
        self.out_name_BD = [self.out_name_BD[0].name]
        
        self.ort_session_E = onnxruntime.InferenceSession(onnx_model_E, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
        self.in_name_E = self.ort_session_E.get_inputs()
        self.out_name_BE = self.ort_session_E.get_outputs()
        self.in_name_E = [self.in_name_E[i].name for i in range(len(self.in_name_E))]
        self.out_name_BE = [self.out_name_BE[i].name for i in range(len(self.out_name_BE))]
        
        self.ort_session_F = onnxruntime.InferenceSession(onnx_model_F, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
        print(f"\nUsable Providers: {self.ort_session_F.get_providers()}\n")
        model_dtype_F = self.ort_session_F._inputs_meta[0].type
        if 'float16' in model_dtype_F:
            model_dtype_F = np.float16
        else:
            model_dtype_F = np.float32
        self.in_name_F = self.ort_session_F.get_inputs()
        self.out_name_BF = self.ort_session_F.get_outputs()
        amount_of_outputs_F = len(self.out_name_BF)
        self.in_name_F = [self.in_name_F[i].name for i in range(len(self.in_name_F))]
        self.out_name_BF = [self.out_name_BF[i].name for i in range(amount_of_outputs_F)]
        
        self.ort_session_G = onnxruntime.InferenceSession(onnx_model_G, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
        model_dtype_G = self.ort_session_G._inputs_meta[2].type
        if 'float16' in model_dtype_G:
            model_dtype_G = np.float16
        else:
            model_dtype_G = np.float32
        self.in_name_G = self.ort_session_G.get_inputs()
        self.out_name_BG = self.ort_session_G.get_outputs()
        self.in_name_G = [self.in_name_G[i].name for i in range(len(self.in_name_G))]
        self.out_name_BG = [self.out_name_BG[i].name for i in range(len(self.out_name_BG))]
        
        self.ort_session_H = onnxruntime.InferenceSession(onnx_model_H, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
        model_dtype_H = self.ort_session_H._inputs_meta[0].type
        if 'float16' in model_dtype_H:
            model_dtype_H = np.float16
        else:
            model_dtype_H = np.float32
        
        shape_value_in_H = self.ort_session_H._inputs_meta[0].shape[1]
        if isinstance(shape_value_in_H, str):
            self.DYNAMIC_SHAPE_VAE_DECODE = True
        else:
            self.DYNAMIC_SHAPE_VAE_DECODE = False
        
        self.in_name_H = self.ort_session_H.get_inputs()
        self.out_name_BH = self.ort_session_H.get_outputs()
        self.in_name_H = self.in_name_H[0].name
        self.out_name_BH = [self.out_name_BH[i].name for i in range(len(self.out_name_BH))]
        self.half_decode_len = 7056  # Fixed for VoxCPM1.5
        
        # ==============================================================================
        # 1. Configuration & Constants Calculation
        # ==============================================================================
        self.generate_limit = MAX_SEQ_LEN - 1
        self.num_keys_values = amount_of_outputs_F - 4
        self.num_layers = self.num_keys_values // 2
        
        # Pre-calculate indices for clarity
        self.num_keys_values_plus_1 = self.num_keys_values + 1
        self.num_keys_values_plus_2 = self.num_keys_values + 2
        self.num_keys_values_plus_3 = self.num_keys_values + 3
        self.num_keys_values_plus_4 = self.num_keys_values + 4
        self.num_keys_values_plus_5 = self.num_keys_values + 5
        
        # ==============================================================================
        # 2. Initialize Static ORT Values (Scalars, Masks, & Buffers)
        # ==============================================================================
        # Simple Scalars and Lengths
        self.init_ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), self.device_type, self.DEVICE_ID)
        self.init_history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), self.device_type, self.DEVICE_ID)
        self.init_concat_text_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), self.device_type, self.DEVICE_ID)
        
        # Special Tokens
        init_audio_start_ids = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([[101]], dtype=np.int32), self.device_type, self.DEVICE_ID)
        
        # Attention Masks
        self.init_attention_mask_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), self.device_type, self.DEVICE_ID)
        self.init_attention_mask_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), self.device_type, self.DEVICE_ID)
        
        # Large Zero Buffers (Cache & Embeddings)
        shape_keys = (self.ort_session_F._inputs_meta[0].shape[0], 1, self.ort_session_F._inputs_meta[0].shape[2], 0)
        shape_vals = (self.ort_session_F._inputs_meta[self.num_layers].shape[0], 1, 0, self.ort_session_F._inputs_meta[self.num_layers].shape[3])
        shape_embed = (1, 0, self.ort_session_F._inputs_meta[self.num_keys_values_plus_1].shape[2])
        shape_latent = (self.ort_session_H._inputs_meta[0].shape[0], 0, self.ort_session_H._inputs_meta[0].shape[2])
        
        self.init_past_keys_F = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape_keys, dtype=model_dtype_F), self.device_type, self.DEVICE_ID)
        self.init_past_values_F = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape_vals, dtype=model_dtype_F), self.device_type, self.DEVICE_ID)
        self.init_feat_embed = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape_embed, dtype=model_dtype_F), self.device_type, self.DEVICE_ID)
        self.init_latent_pred = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape_latent, dtype=model_dtype_H), self.device_type, self.DEVICE_ID)
        
        # Config Values (CFG)
        cfg_value = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([CFG_VALUE], dtype=model_dtype_G), self.device_type, self.DEVICE_ID)
        cfg_value_minus = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1.0 - CFG_VALUE], dtype=model_dtype_G), self.device_type, self.DEVICE_ID)
        
        # Pre-calculate Time Steps
        self.timesteps = FIXED_TIMESTEPS - 1
        self.init_cfm_steps = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int32), self.device_type, self.DEVICE_ID)
        
        # Audio Post-processing
        self.blank_segment = np.zeros((1, 1, int(self.OUT_SAMPLE_RATE * 0.1)), dtype=np.int16)
        
        # ==============================================================================
        # 3. Session Setup & IO Bindings
        # ==============================================================================
        # Initialize Input Feeds
        self.input_feed_A = {}
        self.input_feed_B = {}
        self.input_feed_C = {}
        self.input_feed_D = {}
        self.input_feed_E = {}
        self.input_feed_F = {}
        self.input_feed_G = {}
        self.input_feed_H = {}
        self.input_feed_I = {}
        
        # Session A: Audio Start Embedding
        self.input_feed_A[self.in_name_A] = init_audio_start_ids
        self.audio_start_embed = self.ort_session_A.run_with_ort_values(self.out_name_BA, self.input_feed_A)[0]
        
        # Session D: IO Binding & Initialization
        self.input_feed_D[self.in_name_D] = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros([1, self.ort_session_D._inputs_meta[0].shape[1], self.ort_session_D._inputs_meta[0].shape[2]], dtype=model_dtype_D), self.device_type, self.DEVICE_ID)
        self.init_feat_cond_0 = self.ort_session_D.run_with_ort_values(self.out_name_BD, self.input_feed_D)[0]
        
        # Session G: IO Binding & Fixed Inputs
        self.input_feed_G[self.in_name_G[4]] = cfg_value
        self.input_feed_G[self.in_name_G[5]] = cfg_value_minus
        
    def infer(self,
              target_text: list[str] | str,
              prompt_text:str|None=None,
              prompt_audio_path:str|None = None,
              streaming:bool=False,
              text_normalize:bool=True,
              audio_normalize:bool=False,
              )-> tuple[np.ndarray, int]:

        if type(target_text) == str:
            target_text = [target_text]

        # === Feature flags ===
        #streaming = False Enable streaming synthesis. Unlike the official implementation, this version processes a single latent at a time for faster performance, albeit with potential discontinuities during piece-by-piece decoding.
        #text_normalize = True Use text normalizer
        #audio_normalize = False Use an audio normalizer to stabilize loudness, though this may result in a loss of original audio characteristics.


        # ==============================================================================
        # 4. Preprocessing & Prompt Handling
        # ==============================================================================
        tokenizer = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(self.model_dir))
        text_normalizer = TextNormalizer()
        
        # Handle Audio/Text Prompt
        if prompt_audio_path:
            if prompt_text:
                use_prompt_audio = True
                # Process Audio
                audio = np.array(AudioSegment.from_file(prompt_audio_path).set_channels(1).set_frame_rate(self.IN_SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
                if audio_normalize:
                    audio = audio_normalizer(audio)
                audio = onnxruntime.OrtValue.ortvalue_from_numpy(audio.reshape(1, 1, -1), self.device_type, self.DEVICE_ID)
            else:
                use_prompt_audio = False
                print("Warning: No prompt text provided, so the prompt audio will be ignored.\n")
        else:
            use_prompt_audio = False
            print("Info: No prompt audio provided, using ransom seed to generate voice.\n")
        
        count_time = time.time()
        if use_prompt_audio:
            # Run Audio Encoder (Session B)
            self.input_feed_B[self.in_name_B] = audio
            audio_feat = self.ort_session_B.run_with_ort_values(self.out_name_BB, self.input_feed_B)[0]
            
            # Run Feature Condition (Session D)
            self.input_feed_D[self.in_name_D] = audio_feat
            init_feat_cond = self.ort_session_D.run_with_ort_values(self.out_name_BD, self.input_feed_D)[0]
            
            # Process Text
            if text_normalize:
                prompt_text = text_normalizer.normalize(prompt_text)
                prompt_text = prompt_text.replace("_", "")
                prompt_text = prompt_text.replace("!", "")
                prompt_text = prompt_text.replace("?", "")

            prompt_ids = np.array([tokenizer(prompt_text)], dtype=np.int32)
            prompt_text_len = prompt_ids.shape[-1]
            
            # Run Text Encoder (Session A)
            self.input_feed_A[self.in_name_A] = onnxruntime.OrtValue.ortvalue_from_numpy(prompt_ids, self.device_type, self.DEVICE_ID)
            prompt_embed = self.ort_session_A.run_with_ort_values(self.out_name_BA, self.input_feed_A)[0]
        else:
            # Use Defaults
            init_feat_cond = self.init_feat_cond_0
            prompt_text_len = 0.0
        
        # ==============================================================================
        # 5. Main Generation Loop
        # ==============================================================================
        save_audio_out = []
        
        for sentence in target_text:
            print(f"Convert to Speech: {sentence}")
            #japanese always use g2p, need normalizer
            #if text_normalize:
            sentence = text_normalizer.normalize(sentence)

            sentence = sentence.replace("_", "")
            sentence = sentence.replace("!", "")
            sentence = sentence.replace("?", "")

            # 5.1 Encode Target Text
            target_ids = np.array([tokenizer(sentence)], dtype=np.int32)
            self.input_feed_A[self.in_name_A] = onnxruntime.OrtValue.ortvalue_from_numpy(target_ids, self.device_type, self.DEVICE_ID)
            target_embed = self.ort_session_A.run_with_ort_values(self.out_name_BA, self.input_feed_A)[0]
            
            # 5.2 Combine Embeddings (Session E)
            if use_prompt_audio:
                self.input_feed_E[self.in_name_E[0]] = prompt_embed
                self.input_feed_E[self.in_name_E[1]] = target_embed
                target_embed, _ = self.ort_session_E.run_with_ort_values(self.out_name_BE, self.input_feed_E)
        
            self.input_feed_E[self.in_name_E[0]] = target_embed
            self.input_feed_E[self.in_name_E[1]] = self.audio_start_embed
            concat_embed, concat_text_len = self.ort_session_E.run_with_ort_values(self.out_name_BE, self.input_feed_E)
        
            # 5.3 Calculate Max Length & Initial Features
            if use_prompt_audio:
                self.input_feed_C[self.in_name_C] = audio_feat
                feat_embed = self.ort_session_C.run_with_ort_values(self.out_name_BC, self.input_feed_C)[0]
                
                self.input_feed_E[self.in_name_E[0]] = concat_embed
                self.input_feed_E[self.in_name_E[1]] = feat_embed
                concat_embed, ids_len = self.ort_session_E.run_with_ort_values(self.out_name_BE, self.input_feed_E)
            else:
                feat_embed = self.init_feat_embed
                ids_len = concat_text_len
            ##################################################################################
            generate_limit = self.generate_limit   # モデルの最大長

            phone_text = sentence.replace("'", "")
            phone_text = phone_text.replace("/", "")
            len_text = len(phone_text)

            phone_prompt_text = prompt_text.replace("'", "")
            phone_prompt_text = phone_prompt_text.replace("/", "")
            len_prompt_text= len(phone_prompt_text)

            # 現在のテキスト長から上限を算出
            if len_prompt_text < len_text:
                dynamic_max = (len_text - len_prompt_text) * self.DECODE_LIMIT_FACTOR + 10
            else:
                dynamic_max = (len_prompt_text - len_text) * self.DECODE_LIMIT_FACTOR + 10

            if dynamic_max < 10:
                dynamic_max = len_text * self.DECODE_LIMIT_FACTOR
            else:
                dynamic_max = len_text * self.DECODE_LIMIT_FACTOR + 5

            max_len = int(min(dynamic_max, generate_limit))

            #max_len = min((concat_text_len.numpy() - prompt_text_len) * self.DECODE_LIMIT_FACTOR + 10, self.generate_limit - ids_len.numpy())
            ####################################################3
            # 5.4 Prepare Decoder Inputs (Session F)
            self.input_feed_F[self.in_name_F[self.num_keys_values]] = self.init_history_len
            self.input_feed_F[self.in_name_F[self.num_keys_values_plus_1]] = feat_embed
            self.input_feed_F[self.in_name_F[self.num_keys_values_plus_2]] = concat_text_len
            self.input_feed_F[self.in_name_F[self.num_keys_values_plus_3]] = concat_embed
            self.input_feed_F[self.in_name_F[self.num_keys_values_plus_4]] = ids_len
            self.input_feed_F[self.in_name_F[self.num_keys_values_plus_5]] = self.init_attention_mask_1
        
            # Reset KV Cache
            for i in range(self.num_layers):
                self.input_feed_F[self.in_name_F[i]] = self.init_past_keys_F
            for i in range(self.num_layers, self.num_keys_values):
                self.input_feed_F[self.in_name_F[i]] = self.init_past_values_F
        
            # Copy initial condition to avoid overwrite
            feat_cond = init_feat_cond
        
            # Prepare Latent Storage
            if not streaming:
                self.save_latent = self.init_latent_pred if self.DYNAMIC_SHAPE_VAE_DECODE else []
        
            # --------------------------------------------------------------------------
            # 5.5 Auto-regressive Decoding Loop
            # --------------------------------------------------------------------------
            num_decode = 0
            start_decode = time.time()
            
            while num_decode < max_len:
                # --- Run Transformer (Session F) ---
                all_outputs_F = self.ort_session_F.run_with_ort_values(self.out_name_BF, self.input_feed_F)
        
                # --- Run Flow Matching / Diffusion (Session G) ---
                self.input_feed_G[self.in_name_G[0]] = self.init_cfm_steps
                self.input_feed_G[self.in_name_G[1]] = all_outputs_F[self.num_keys_values_plus_1]
                self.input_feed_G[self.in_name_G[2]] = all_outputs_F[self.num_keys_values_plus_2]
                self.input_feed_G[self.in_name_G[3]] = feat_cond
        
                for i in range(self.timesteps):
                    all_outputs_G = self.ort_session_G.run_with_ort_values(self.out_name_BG, self.input_feed_G)
                    self.input_feed_G[self.in_name_G[0]] = all_outputs_G[0]
                    self.input_feed_G[self.in_name_G[1]] = all_outputs_G[1]
        
                latent_pred = all_outputs_G[1]
        
                # --- Handle Output (Stream or Save) ---
                if streaming:
                    if num_decode < 1:
                        pre_latent_pred = latent_pred
                    else:
                        self.input_feed_E[self.in_name_E[0]] = pre_latent_pred
                        self.input_feed_E[self.in_name_E[1]] = latent_pred
                        self.save_latent, _ = self.ort_session_E.run_with_ort_values(self.out_name_BE, self.input_feed_E)
                        self.input_feed_H[self.in_name_H] = self.save_latent
                        audio_out, _ = self.ort_session_H.run_with_ort_values(self.out_name_BH, self.input_feed_H)
                        pre_latent_pred = latent_pred
                        audio_out = audio_out.numpy()
                        if num_decode > 1:
                            audio_out = audio_out[..., self.half_decode_len:]
                        save_audio_out.append(audio_out)
                else:
                    if self.DYNAMIC_SHAPE_VAE_DECODE:
                        self.input_feed_E[self.in_name_E[0]] = self.save_latent
                        self.input_feed_E[self.in_name_E[1]] = latent_pred
                        self.save_latent, _ = self.ort_session_E.run_with_ort_values(self.out_name_BE, self.input_feed_E)
                    else:
                        self.save_latent.append(latent_pred)
        
                # --- Check Stop Token ---
                if num_decode >= self.MIN_SEQ_LEN and all_outputs_F[self.num_keys_values_plus_3].numpy() in self.STOP_TOKEN:
                    break
        
                # --- Update Inputs for Next Iteration ---
                self.input_feed_C[self.in_name_C] = latent_pred
                feat_embed = self.ort_session_C.run_with_ort_values(self.out_name_BC, self.input_feed_C)[0]
        
                self.input_feed_D[self.in_name_D] = latent_pred
                feat_cond = self.ort_session_D.run_with_ort_values(self.out_name_BD, self.input_feed_D)[0]
        
                self.input_feed_F.update(zip(self.in_name_F[:self.num_keys_values_plus_1], all_outputs_F))
                self.input_feed_F[self.in_name_F[self.num_keys_values_plus_1]] = feat_embed
                self.input_feed_F[self.in_name_F[self.num_keys_values_plus_3]] = feat_embed
        
                if num_decode < 1:
                    # First Step Initialization
                    self.input_feed_F[self.in_name_F[self.num_keys_values_plus_2]] = self.init_concat_text_len
                    self.input_feed_F[self.in_name_F[self.num_keys_values_plus_4]] = self.init_ids_len_1
                    self.input_feed_F[self.in_name_F[self.num_keys_values_plus_5]] = self.init_attention_mask_0
                
                num_decode += 1
                print(f"    Decode: {num_decode}")
        
            print(f"\nDecode Speed: {((num_decode + 1) / (time.time() - start_decode)):.3f} token/s\n")
        
            # 5.6 Finalize Sentence Audio (Non-streaming)
            if not streaming:
                if self.DYNAMIC_SHAPE_VAE_DECODE:
                    self.input_feed_H[self.in_name_H] = self.save_latent
                    audio_out, _ = self.ort_session_H.run_with_ort_values(self.out_name_BH, self.input_feed_H)
                    save_audio_out.append(audio_out.numpy())
                else:
                    self.input_feed_E[self.in_name_E[0]] = self.save_latent[0]
                    self.input_feed_E[self.in_name_E[1]] = self.save_latent[1]
                    concat_latent, _ = self.ort_session_E.run_with_ort_values(self.out_name_BE, self.input_feed_E)
                    self.input_feed_H[self.in_name_H] = concat_latent
                    audio_out, _ = self.ort_session_H.run_with_ort_values(self.out_name_BH, self.input_feed_H)
                    save_audio_out.append(audio_out.numpy())
                    for i in range(2, len(self.save_latent)):
                        self.input_feed_E[self.in_name_E[0]] = self.save_latent[i - 1]
                        self.input_feed_E[self.in_name_E[1]] = self.save_latent[i]
                        concat_latent, _ = self.ort_session_E.run_with_ort_values(self.out_name_BE, self.input_feed_E)
                        self.input_feed_H[self.in_name_H] = concat_latent
                        audio_out, _ = self.ort_session_H.run_with_ort_values(self.out_name_BH, self.input_feed_H)
                        audio_out = audio_out.numpy()[..., self.half_decode_len:]
                        save_audio_out.append(audio_out)
        
            save_audio_out.append(self.blank_segment)
        
        # ==============================================================================
        # 6. Post-Processing & Stats
        # ==============================================================================
        cost_time = time.time() - count_time
        audio_out = np.concatenate(save_audio_out, axis=-1).reshape(-1)
        if audio_normalize:
            audio_out = audio_normalizer(audio_out)
            
        
        
        total_audio_duration = (audio_out.shape[-1] - self.blank_segment.shape[-1] * len(target_text)) / self.OUT_SAMPLE_RATE
        rtf = cost_time / total_audio_duration
        
        print(f"\nGenerate Complete.")
        #print(f"Saving to: {generated_audio_path}")
        print(f"Time Cost: {cost_time:.3f} Seconds")
        print(f"RTF: {rtf:.3f}")
        
        return  (audio_out, self.OUT_SAMPLE_RATE)

    
def main():
    prompt_audio_path = "./scripts/to_onnx/example/basic_ref_zh.wav"  # optional: path to a prompt speech for voice cloning else None.
    prompt_text = "对，这就是我，万人敬仰的太乙真人。"  # The reference text for the prompt speech.
    target_text = ["やあ。こんにちは、元気？"]  # The test query after the export process.
    generated_audio_path = r"./generated.wav"  
    tts = VoxcpmOnnx(model_dir="./checkpoints/ja-jp-v1-rev-a-onnx")
    audio_out, sr = tts.infer(
        target_text = target_text,
        prompt_text=prompt_text,
        prompt_audio_path=prompt_audio_path,
        text_normalize=False
        )
    
    sf.write(generated_audio_path, audio_out, sr, format='WAVEX')

if __name__ == "__main__":
    main()