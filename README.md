## ⚠️ Important Notes

### 🧠 Requirements

This project uses:

* **PyTorch** (with CUDA)
* **BitsAndBytes** (requires CUDA)
* **sounddevice** (depends on PortAudio)

If you **already have PyTorch installed** (with the correct CUDA version), you can skip installing it again.

Otherwise, please install PyTorch manually using the official guide:
👉 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

or you can uncomment the first 3 lines in the `requirements.txt`

---

### 🎛️ PortAudio Requirement

This app uses `sounddevice`, which depends on the **PortAudio** library.
Make sure PortAudio is installed on your system **before running the app**.

---

### 🐍 Setup Instructions (Windows Example)

Use a **virtual environment** to install dependencies cleanly:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Then run the real-time transcription app:

```bash
cd Part2
py part2.py
```

For part1, it is recommended to try to run it in colab.

<details>
<summary><b>Part 1: Model Optimization</b></summary>

**Quantized model on Hugging Face:**  
[https://huggingface.co/Winoto/whisper-base-4bit-quantized](https://huggingface.co/Winoto/whisper-base-4bit-quantized)

### 1. Why I Chose My Model and Optimization Technique

* **Model (`openai/whisper-base`)**: i chose the `whisper-base` model because it seemed like a really good and safe choice. its from the whisper family and i've already used it once in my practicum so i knew it would be good at understanding speech. i picked the 'base' size so it would be strong but not too huge for my computer.

* **Optimization (Quantization)**: for optimization i picked Quantization because the idea of making a model smaller and faster is interesting to me. i wanted to see if i could make it more efficient for the real-time part of the test (part 2) and quantization is supposed to reduce memory and make it faster, so it was the perfect thing to try.

---

### 2. How Quantization Works

the main idea of quantization is that, it just makes the numbers inside the model less detailed to save a bunch of space.

it's like when we have a really big photo file, we can save it with fewer colors and the picture will look almost the same but the file size will be way smaller. THe quantization method does that to the model's weights.

it takes the 32-bit numbers and squishes them down to a simpler format (e.g. 4-bit). so because every number is smaller, the whole model gets way smaller too, which means it uses less memory and can run faster.

---

### 3. Potential Trade-offs of Quantization

the main trade-off is that it can make the model a little less accurate.

when we make the numbers less detailed, we lose a little bit of information. so the model might make a few more mistakes, maybe it might get confused by noisy audio or tricky words.

but the new ways of doing quantization are really smart about it. from my inference test it looks like the quantized one was basically the same as the original. we get a much more efficient model and it's pretty much just as good

---

### Summary of Steps and Challenges

#### Steps I Took
1.  first i loaded the normal `whisper-base` model.
2.  then i created a 4-bit quantization config using bitsandbytes.
3.  i loaded the same model again but applied my quantization config to it to make the smaller version.
4.  after that i ran a transcription on an audio file with both models so i could compare their outputs, speed, and memory usage.
5.  finally i used `.save_pretrained()` to save the smaller, quantized model to a local folder.

#### Challenges I Faced
my main challenge was just understanding how quantization worked at first. i had to read a bit about what all the settings in `BitsAndBytesConfig` meant, like `bnb_4bit_quant_type="nf4"`. also making sure i had the right versions of all the libraries installed so they would work with my GPU was a little tricky. at first i wasn't sure how to save the final model but i figured out that `save_pretrained` was the right way to save it locally.

</details>

<details>
<summary><b>Part 2: Real-Time Streaming Transcription</b></summary>

### 1. What are the main challenges in building a low-latency streaming transcription system?

* **AI models are slow**: the biggest challenge is that the AI models are usually really big and slow. we say something and we have to wait for the model to "think". for real-time we need it to be super fast, which is why i used the smaller quantized model from part 1 to speed things up.

* **Knowing when to transcribe**: another challenge is figuring out *when* to show the text. if we transcribe every tiny sound immediately, the text will be messy and keep changing as more audio gives it context. but if we wait too long to get more audio, it's not "real-time" anymore. it's like a trade-off between being fast and being accurate.

---

### 2. How did you handle the continuous flow of audio data?

i used a 'producer-consumer' idea with a queue. my strategy was to have two threads working together so that the app never misses anything i say.

* **Buffering Strategy**: One thread's only job is to listen to the microphone (`audio_callback`). it captures tiny 30ms chunks of audio and immediately puts them into a `Queue`. this is the 'producer' and it's super fast so it never loses any audio. a second thread, the 'consumer' (`processing_thread_main`), pulls the audio from the queue and collects it into a bigger `audio_buffer` for processing.

* **Segmentation Strategy**: to figure out when a sentence ends, i used a simple VAD (Voice Activity Detection). it just checks the volume of the incoming audio (the RMS value). if it's quiet for a long enough time (i set it to about 1 second), the program decides the sentence is finished. it then finalizes the transcription for that audio buffer and clears the buffer to start fresh for the next sentence. i set the `SILENCE_THRESHOLD_RMS` manually using my own mic to know whats the range of RMS when i'm speaking or when it's silent.

---

### 3. If you were to improve your system for a production environment, what are two improvements you would make?

1.  **Use a Much Better VAD**: first, i would use a much smarter VAD system. the one i built just checks the volume, which is okay but can get confused by fan noise or soft speech. i would use a proper pre-trained VAD model (like Silero VAD). it would be way more accurate at knowing exactly when speech starts and stops, which would prevent weird cutoff words at the beginning or end of sentences. in fact, i've actually tried this approach in my previous tries but it just won't work so i just stick with a simple VAD.

2.  **Make the Output More Stable**: second, i'd stop the live text from flickering and changing so much. for a professional tool, we want the text to be stable once it appears. i read about a technique in the Whisper-Streaming paper called ["Local Agreement"](https://arxiv.org/pdf/2307.14743). the basic idea is that the system waits until it's more confident about a word or phrase before showing it to the user. this would make the output feel much more polished and reliable, instead of changing constantly as more context arrives.

---

### System Architecture and Libraries Used

#### Architecture

My system uses a **multi-threaded producer-consumer architecture**. the idea is simple:

* **Producer (The Ear)**: the 'producer' is my `audio_callback` function. the sounddevice library runs this for me automatically in the background every 30 milliseconds which is super fast. its only job is to grab that tiny piece of sound and drop it straight into the shared `Queue`. it doesn't do any thinking, it just catches everything!

* **Consumer (The Brain)**: the 'consumer' is where all the magic happens, in the `processing_thread_main` function that runs in a constant loop. it's the 'brain' of the operation and does a few things in order:
    1.  **Grabs Audio**: first it empties out the queue, grabbing all the audio that the producer left for it.
    2.  **Checks for Speech (VAD)**: then it checks if the audio it just got has any sound in it. if its loud enough it knows i'm talking. if the queue was empty it just counts that as a moment of silence.
    3.  **Grows the Buffer**: if i am talking, it adds the new sound to a bigger audio buffer that keeps growing. this way the model gets more and more context which is good.
    4.  **Transcribes**: this is the last step. if i'm talking it sends the whole big buffer to the whisper model to get the live text with the 🎤 emoji. when it sees i've been quiet for a little bit, it knows the sentence is over. it does one last transcription to get the final text, prints it with the ✅, and then clears everything out to get ready for the next sentence.

this whole setup is good because the slow AI stuff doesnt mess with the fast audio recording part. so the app feels fast and doesnt miss anything i say.

#### A Note on Context (The Prompt)

the `prompt_context` variable in the code is like the model's **short-term memory**.

after a sentence is finished and printed with a ✅, the program doesn't just forget it. it saves a cleaned-up version of that sentence (with duplicate words removed and limited in length). when i start talking again, the program sends both the *new* audio and that *previous sentence* to the Whisper model.

this is super useful because it gives the model a hint about what we're talking about. so if i say "My favorite fruit is fried chicken" and then the next sentence is "I like to eat it", the model gets the audio for "I like to eat it" and the context "My favorite fruit is apple". this helps it understand that "it" probably refers to an apple. it just makes the model a little smarter and the transcription more accurate. more context is always good

#### Libraries

* **`torch`**: This is the main deep learning library that the Whisper model runs on.
* **`transformers`**: library from Hugging Face makes it super easy to download, load, and run pre-trained models like Whisper.
* **`sounddevice`**: library i used to get the audio from my microphone. It relies on the PortAudio library so make sure it is installed.
* **`numpy`**: I used this for all the audio data math, like combining audio chunks and calculating the volume (RMS) for my simple voice activity detection.
* **`threading` and `queue`**: libraries i used to build the producer-consumer architecture so that audio capture and transcription could happen at the same time without interfering with each other.


</details>