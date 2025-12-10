import torch
import sounddevice as sd
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.modeling_whisper import WhisperAttention
import queue
import os
import soundfile as sf

try:
    import multihead_attention_algorithm
    print("Cpp fajl je uspesno uvezen!")
except ImportError:
    print("Greska! Nije moguce uvesti cpp fajl.")
    multihead_attention_algorithm = None

# --- Podesavanja --
MODEL_NAME = "openai/whisper-base"
RECORD_SECONDS = 4
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024

# Flagovi za test
USE_CPP_ATTENTION = True     # True koristi C++ funkciju
USE_MICROPHONE = False        # True - hvata sa mikrofona False - uzima .wav fajl umesto mikrofona
TEST_WAV = "whisper.wav"         

q = queue.Queue()

def audio_callback(indata, frames, time, status):
    q.put(indata.copy())

#Moj custom attention block
class AttentionWhisperBlock(WhisperAttention):
    def __init__(self, embed_dim, num_heads, dropout, is_decoder=False):
        super().__init__(embed_dim, num_heads, dropout, is_decoder)

    def forward(self, hidden_states, key_value_states=None, past_key_value=None,
                attention_mask=None, layer_head_mask=None, output_attentions=False):

        if multihead_attention_algorithm is None:
            raise RuntimeError("multihead_attention_algorithm nije ucitan!")

        #Q, K, V iz PyTorcha
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        
        scale_factor = 1.0 / np.sqrt(self.head_dim)
        print("Skaliranje...")
        
        #pretvori u numpy list
        q_list = Q.detach().cpu().numpy() * scale_factor
        k_list = K.detach().cpu().numpy()
        v_list = V.detach().cpu().numpy()
        print("Skaliranje zavrseno...")
        if not os.path.exists("matrice"):
            os.makedirs("matrice")

        print("Saving Q, K and V...")
        
        #Čuvamo Q, K, V (Reshape u 2D jer txt fajl ocekuje matricu)
        #q_list je oblika (Batch, SeqLen, EmbedDim), uzimamo [0]
        np.savetxt("matrice/multihead_ulaz_Q.txt", q_list[0])
        np.savetxt("matrice/multihead_ulaz_K.txt", k_list[0])
        np.savetxt("matrice/multihead_ulaz_V.txt", v_list[0])

        #Čuvamo W_out i b_out
        #Moramo transponovati W jer PyTorch cuva kao (Out, In), a mi mnozimo (In, Out)
        np.savetxt("matrice/multihead_W_out.txt", self.out_proj.weight.detach().cpu().numpy().T)
        np.savetxt("matrice/multihead_b_out.txt", self.out_proj.bias.detach().cpu().numpy())

        outputs = []
        for b in range(q_list.shape[0]):
            attn_out_b = multihead_attention_algorithm.attention_core(
                q_list[b].tolist(),
                k_list[b].tolist(),
                v_list[b].tolist(),
                int(self.num_heads)
            )
            
            outputs.append(np.array(attn_out_b, dtype=np.float32))

        attn_output_np = np.stack(outputs, axis=0)
        attn_output = torch.from_numpy(attn_output_np).to(hidden_states.device, dtype=hidden_states.dtype)

        #finalni linearni sloj
        attn_output = self.out_proj(attn_output)

        #grananje
        if self.is_decoder:
            if key_value_states is not None:
                return attn_output, None
            else:
                return attn_output, None, None
        else:
            return attn_output, None, None

#offline test(koristi .wav fajl)
def offline_test(processor, model, device):
    print(f"Ucitavanje {TEST_WAV} fajla...")
    audio, sr = sf.read(TEST_WAV)
    if sr != 16000:
        import resampy
        audio = resampy.resample(audio, sr, 16000)
        sr = 16000
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    predicted_ids = model.generate(inputs.input_features.to(device))
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    print("Transkripcija:", text[0] if text else "<nista>")
    print("---------------------------")


#glavni deo programa
def main():
    print(f"Učitavanje modela '{MODEL_NAME}'...")
    device = "cpu"
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

    if USE_CPP_ATTENTION:
        print("Koristimo custom C++ funkciju")
        original_layer = model.model.encoder.layers[0].self_attn
        novi_sloj = AttentionWhisperBlock(
            embed_dim=original_layer.embed_dim,
            num_heads=original_layer.num_heads,
            dropout=original_layer.dropout
        )
        novi_sloj.load_state_dict(original_layer.state_dict())
        model.model.encoder.layers[0].self_attn = novi_sloj
        print("Zamena uspesna.\n")
    else:
        print("Koristimo originalni HuggingFace attention.\n")

    if not USE_MICROPHONE:
        offline_test(processor, model, device)
        return

    #ako koristimo mikrofon
    while True:
        try:
            num_blocks_to_record = int(RECORD_SECONDS * SAMPLE_RATE / BLOCK_SIZE)
            audio_data = []

            print("-------------------------------------------")
            input(f"Pritisnite Enter da započnete snimanje od {RECORD_SECONDS} sekundi...")

            print("Snimam...")
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                                blocksize=BLOCK_SIZE, callback=audio_callback):
                for _ in range(num_blocks_to_record):
                    audio_data.append(q.get())
            print("Snimanje završeno, obrađujem snimak...")

            if audio_data:
                full_audio = np.concatenate(audio_data, axis=0).flatten()
                input_features = processor(full_audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(device)
                predicted_ids = model.generate(input_features)
                forced_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

                print("\n****FINALNA TRANSKRIPCIJA****")
                print(transcription[0].strip() if transcription and transcription[0].strip() else "Nema prepoznatog govora")
                print("---------------------------\n")

        except KeyboardInterrupt:
            print("\nProgram prekinut.")
            break
        except Exception as e:
            print(f"Došlo je do greške: {e}")
            break

if __name__ == "__main__":
    main()
