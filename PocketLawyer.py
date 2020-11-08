#!/usr/bin/env python

"""Google Cloud Speech API sample application using the streaming API.

NOTE: This module requires the dependencies `pyaudio` and `termcolor`.
To install using pip:

    pip install pyaudio
    pip install termcolor

Example usage:
    python transcribe_streaming_infinite.py
"""

# [START speech_transcribe_infinite_streaming]

import re
import sys
import time
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/root/cloud_speech.json"
import pyaudio
from six.moves import queue
from google.oauth2 import service_account
import io
import datetime as dt
from picamera import PiCamera
from signal import pause
from google.cloud import speech
#import gtts
import aiy.voice.audio
import aiy.board
import aiy.voice
from monkeylearn import MonkeyLearn
import json
ml = MonkeyLearn('ecc75d02c2dc75fcbf8ed4767b4709c50735d415')
#from google.cloud import language_v1
destination = '/home/pi/video'
camera = PiCamera()
def record_video():
    filename = os.path.join(destination, dt.datetime.now().strftime('%Y-%m-%d_%H.%M.%S.h264'))
    camera.resolution = (640,480)
    camera.start_recording('a.h264')
   # tts= gtts.gTTS(text='hello officer', lang='en')
   # tts.save('hello.mp3')
    audioplayer = aiy.voice.audio.play_wav_async('helloofficer.mp3')
    audiorecorder = aiy.voice.audio.record_file_async(aiy.voice.audio.AudioFormat(sample_rate_hz=44100, num_channels=2, bytes_per_sample=2),filename="recording1.wav",filetype="wav",device='default')
    #audioplayer.stdout.close()
    camera.wait_recording(1)
    
def finish_video():
    camera.stop_recording()
    camera.close()
# Audio recording parameters
STREAMING_LIMIT = 240000  # 4 minutes
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms
credentials = service_account.Credentials.from_service_account_file("/root/cloud_speech.json")
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[0;33m'


def get_current_time():
    """Return Current Time in MS."""

    return int(round(time.time() * 1000))


class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk_size):
        self._rate = rate
        self.chunk_size = chunk_size
        self._num_channels = 1
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

    def __enter__(self):

        self.closed = False
        return self

    def __exit__(self, type, value, traceback):

        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, *args, **kwargs):
        """Continuously collect data from the audio stream, into the buffer."""

        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        """Stream Audio from microphone to API and to local buffer"""

        while not self.closed:
            data = []

            if self.new_stream and self.last_audio_input:

                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)

                if chunk_time != 0:

                    if self.bridging_offset < 0:
                        self.bridging_offset = 0

                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time

                    chunks_from_ms = round((self.final_request_end_time -
                                            self.bridging_offset) / chunk_time)

                    self.bridging_offset = (round((
                        len(self.last_audio_input) - chunks_from_ms)
                                                  * chunk_time))

                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        data.append(self.last_audio_input[i])

                self.new_stream = False

            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            self.audio_input.append(chunk)

            if chunk is None:
                return
            data.append(chunk)
            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)

                    if chunk is None:
                        return
                    data.append(chunk)
                    self.audio_input.append(chunk)

                except queue.Empty:
                    break

            yield b''.join(data)


def listen_print_loop(responses, stream):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
   # client = language.LanguageServiceClient()

    for response in responses:

        if get_current_time() - stream.start_time > STREAMING_LIMIT:
            stream.start_time = get_current_time()
            break

        if not response.results:
            continue

        result = response.results[0]

        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        result_seconds = 0
        result_micros = 0

        if result.result_end_time.seconds:
            result_seconds = result.result_end_time.seconds

        if result.result_end_time.microseconds:
            result_micros = result.result_end_time.microseconds

        stream.result_end_time = int((result_seconds * 1000) + (result_micros / 1000))

        corrected_time = (stream.result_end_time - stream.bridging_offset
                          + (STREAMING_LIMIT * stream.restart_counter))
        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.

        if result.is_final:

            sys.stdout.write(GREEN)
            sys.stdout.write('\033[K')
            sys.stdout.write(str(corrected_time) + ': ' + transcript + '\n')

            stream.is_final_end_time = stream.result_end_time
            stream.last_transcript_was_final = True

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r'\b(exit|quit|goodbye officer)\b', transcript, re.I):
                sys.stdout.write(YELLOW)
                sys.stdout.write('Exiting...\n')
                stream.closed = True
                break

        else:
            sys.stdout.write(RED)
            sys.stdout.write('\033[K')
            sys.stdout.write(str(corrected_time) + ': ' + transcript + '\r')

            stream.last_transcript_was_final = False
     #   document = language.Document(content=responses, type_=language_v1.Document.Type.PLAIN_TEXT)
      #  sentiment = client.analyze_sentiment(request={'document':document}).document_sentiment
       # print("Sentiment: {}, {}".format(sentiment.score, sentiment.magnitude))
        #print(response)        
        #print(transcript)
        #data2 = json.loads(data)
        #print(str(data2))
        data = [transcript]
        model_id = 'cl_pi3C7JiL'
        response2 = ml.classifiers.classify(model_id,data)
        print(response2.body)
def main():
    """start bidirectional streaming from microphone input to speech API"""
    record_video()
    print('recording')
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code='en-US',
        max_alternatives=1)
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)
    print(mic_manager.chunk_size)
    sys.stdout.write(YELLOW)
    sys.stdout.write('\nListening, say "Quit" or "Exit" to stop.\n\n')
    sys.stdout.write('End (ms)       Transcript Results/Status\n')
    sys.stdout.write('=====================================================\n')

    with mic_manager as stream:

        while not stream.closed:
            sys.stdout.write(YELLOW)
            sys.stdout.write('\n' + str(
                STREAMING_LIMIT * stream.restart_counter) + ': NEW REQUEST\n')

            stream.audio_input = []
            audio_generator = stream.generator()

            requests = (speech.StreamingRecognizeRequest(
                audio_content=content)for content in audio_generator)

            responses = client.streaming_recognize(streaming_config,
                                                   requests)

            # Now, put the transcription responses to use.
            listen_print_loop(responses, stream)

            if stream.result_end_time > 0:
                stream.final_request_end_time = stream.is_final_end_time
            stream.result_end_time = 0
            stream.last_audio_input = []
            stream.last_audio_input = stream.audio_input
            stream.audio_input = []
            stream.restart_counter = stream.restart_counter + 1

            if not stream.last_transcript_was_final:
                sys.stdout.write('\n')
            stream.new_stream = True

    finish_video()

if __name__ == '__main__':

    main()
# [END speech_transcribe_infinite_streaming]
