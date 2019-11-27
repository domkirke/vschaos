import numpy as np

def readaudio(infile, samplerate, dtype=float):
    """Read audio file"""
    audio = None
    if infile.endswith('.npy'):
        audio = np.load(infile).astype(dtype)
    if audio is None and infile.endswith('.wav'):
        import wave
        try:
            sf = wave.open(infile, 'r')
            sr = sf.getframerate()
            nframes = sf.getnframes()
            bits = sf.getsampwidth()*8
            chns = sf.getnchannels()
            if sr == samplerate:
                samples = sf.readframes(nframes)
                audio = np.fromstring(samples, dtype=eval("np.int%i"%bits)).reshape(nframes, chns)
                audio = np.mean(audio, axis=1)
                audio /= 2**(bits-1)
            sf.close()
        except:
            pass
    if audio is None:
        import subprocess
        for cmd in ('ffmpeg','avconv'):
            call = [cmd, "-v", "quiet", "-y", "-i", infile, "-f", "f32le", "-ac", "1", "-ar", str(samplerate), "pipe:1"]
            try:
                samples = subprocess.check_output(call)
            except OSError:
                continue
            audio = np.frombuffer(samples, dtype=np.float32).astype(dtype)
            break

    return audio
