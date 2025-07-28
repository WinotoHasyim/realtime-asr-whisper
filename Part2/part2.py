import torch
import numpy as np
import sounddevice as sd
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from queue import Queue, Empty
import sys
import threading
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

# --- Configuration ---
# Model
MODEL_ID = "Winoto/whisper-base-4bit-quantized" # Your quantized model

# Audio Stream
SAMPLE_RATE = 16000  # Whisper requires 16000Hz
CHANNELS = 1
FRAME_MS = 30  # Duration of a single audio frame in milliseconds
FRAME_SAMPLES = int(SAMPLE_RATE * (FRAME_MS / 1000.0))

# Voice Activity Detection (VAD)
# A simple energy-based VAD.
SILENCE_THRESHOLD_RMS = 0.05 # RMS threshold to consider a chunk as silence. Adjust if needed.
SILENCE_CHUNKS_TO_RESET = 10 # Number of consecutive silent chunks to finalize a sentence. (10 = 1 second of silence)

# --- Global State ---
audio_queue = Queue()
is_running = True

def audio_callback(indata, frames, time, status):
    """This function is called by sounddevice for each new audio chunk."""
    if status:
        print(status, file=sys.stderr)
    # Simply put the new audio data into the queue.
    audio_queue.put(indata.copy())

def transcribe_chunk(model, processor, audio_chunk, prompt_text=""):
    """
    Transcribes a single audio chunk using the Whisper model.
    The prompt from the previous sentence is used for context.
    """
    if audio_chunk is None or audio_chunk.size == 0:
        return ""

    # 1. Preprocess audio to the format Whisper expects
    inputs = processor(audio_chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(model.device, dtype=torch.float16)

    # 2. Prepare prompt IDs if there is context
    if prompt_text.strip():
        # Use the processor's method to correctly prepare prompt IDs
        prompt_ids = processor.get_prompt_ids(prompt_text, return_tensors="pt").to(model.device)
    else:
        prompt_ids = None

    # 3. Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            prompt_ids=prompt_ids,
            no_repeat_ngram_size=3, # To avoid repeating phrases
            task="transcribe"
        )

    # 4. Decode and return the new text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return transcription.strip()


def processing_thread_main(model, processor):
    """
    The main worker thread. It pulls audio from the queue, manages buffers,
    and calls the transcription function.
    """
    global is_running
    
    # Buffers and state for the transcription logic
    audio_buffer = np.array([], dtype=np.float32)
    prompt_context = ""
    silent_chunks_count = 0
    
    print("\n✅ Processing thread started.")

    while is_running:
        try:
            # --- 1. Accumulate audio from the queue ---
            # Get all available audio chunks from the queue to process in a batch.
            chunk_list = []
            while not audio_queue.empty():
                chunk_list.append(audio_queue.get_nowait())
            
            if not chunk_list:
                sd.sleep(100) # Sleep if no audio
                # If we are sleeping, it's a silent chunk.
                silent_chunks_count += 1
                new_audio = np.array([], dtype=np.float32)
            else:
                 new_audio = np.concatenate(chunk_list).flatten()

            # --- 2. Voice Activity Detection (VAD) ---
            is_speech = False
            if new_audio.size > 0:
                rms = np.sqrt(np.mean(new_audio**2))
                if rms > SILENCE_THRESHOLD_RMS:
                    is_speech = True

            # --- 3. Update State Based on VAD ---
            if is_speech:
                silent_chunks_count = 0
                audio_buffer = np.concatenate([audio_buffer, new_audio])
            # Note: silent_chunks_count is already incremented if there was no audio

            # --- 4. Transcription Logic ---
            # Condition to finalize: if a pause is long enough AND there's something to finalize.
            if silent_chunks_count >= SILENCE_CHUNKS_TO_RESET and audio_buffer.size > 0:
                final_transcription = transcribe_chunk(model, processor, audio_buffer, prompt_context)
                
                if final_transcription.strip():
                    sys.stdout.write('\r' + "✅ " + final_transcription + '\n')
                    sys.stdout.flush()
                    # Process the final transcription to create a clean and length-limited prompt.
                    prompt_context = process_prompt(final_transcription)
                
                # Reset buffer AFTER finalizing.
                audio_buffer = np.array([], dtype=np.float32)
                current_transcription = ""

            # Condition to do live transcription: if we just received speech.
            elif is_speech:
                current_transcription = transcribe_chunk(model, processor, audio_buffer, prompt_context)
                sys.stdout.write('\r' + "🎤 " + current_transcription)
                sys.stdout.flush()

        except Empty:
            continue
        except Exception as e:
            print(f"Error in processing thread: {e}")
            is_running = False

def process_prompt(text, max_words=50):
    """
    Processes text for the prompt by removing consecutive duplicates
    and limiting the total word count to the most recent words.
    """
    # 1. Remove consecutive repeated words
    words = text.split()
    if not words:
        return ""
    
    result_words = [words[0]]
    for i in range(1, len(words)):
        if words[i] != words[i-1]:
            result_words.append(words[i])
            
    # 2. Limit the total word count (taking the most recent words)
    if len(result_words) > max_words:
        result_words = result_words[-max_words:]
            
    return " ".join(result_words)

def main():
    """Main function to set up and run the application."""
    global is_running

    print("Starting real-time transcription application...")

    # 1. --- Load Quantized Model and Processor ---
    print("🔧 Loading Whisper model with 4-bit quantization...")
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # --- List and select microphone device ---
    print("\n🎤 Available audio input devices:")
    devices = sd.query_devices()
    input_devices = []

    # Get default input and output devices
    default_input, default_output = sd.default.device

    print("Note: '>' indicates default input device, '<' indicates default output device\n")

    for i, device in enumerate(devices):
        # Only show input devices (max_input_channels > 0)
        if device['max_input_channels'] > 0:
            input_devices.append(i)
            prefix = ">"  if i == default_input else " "
            suffix = " <" if i == default_output else ""
            print(f"  {prefix} [{i}] {device['name']}{suffix}")
        
    # Ask user to select a device
    while True:
        try:
            device_id = input("\nEnter the number of the microphone to use: ")
            device_id = int(device_id)
            if device_id in input_devices:
                print(f"✅ Selected device: {devices[device_id]['name']}")
                break
            else:
                print("❌ Invalid device number. Please try again.")
        except ValueError:
            print("❌ Please enter a valid number.")

    # --- Start the processing thread ---
    proc_thread = threading.Thread(target=processing_thread_main, args=(model, processor))
    proc_thread.start()

    # --- Start Microphone Stream ---
    print("🎤 Starting microphone stream... You can speak now.\n")
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='float32',
            blocksize=FRAME_SAMPLES,
            callback=audio_callback,
            device=device_id
        ):
            # The main thread just needs to wait for the user to stop the program
            while is_running:
                sd.sleep(1000)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Error with microphone stream: {e}")
    finally:
        # --- Cleanup ---
        is_running = False
        proc_thread.join() # Wait for the processing thread to finish
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()