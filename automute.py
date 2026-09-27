"""
- PyAudioWPatch adds WASAPI loopback support on Windows, to listen
  to what's played out of your default audio
- pycaw mutes/unmutes

Run:
    Set options in config
    python automute.py
"""

import time
import threading
import numpy as np
import pyaudiowpatch as pyaudio
from pycaw.pycaw import AudioUtilities

# CONFIG
THRESHOLD_DB = -13.0            # dBfs level that triggers a mute.
                                # Silence would be like -100, loud explosions like -10
                                # Start around -15 and adjust

ENABLE_FREQUENCY_TRIGGER = True # False to disable the pitch-based trigger. Doesn't work great
FREQ_CUTOFF_HZ = 5000
FREQ_TRIGGER_DB = -90.0

MUTE_DURATION_SECONDS = 2
CHECK_INTERVAL_SECONDS = 0.01
COOLDOWN_AFTER_UNMUTE = 0.0

DEBUG = True                    # True to continuously print data for setting levels
DEBUG_PRINT_INTERVAL = 0.5

def get_volume_interface():
    device = AudioUtilities.GetSpeakers()
    return device.EndpointVolume


def set_mute(volume_iface, mute: bool):
    volume_iface.SetMute(1 if mute else 0, None)


def is_muted(volume_iface) -> bool:
    return bool(volume_iface.GetMute())


def get_default_loopback_device(p: pyaudio.PyAudio):
    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

    if not default_speakers.get("isLoopbackDevice", False):
        for loopback in p.get_loopback_device_info_generator():
            if default_speakers["name"] in loopback["name"]:
                return loopback
        raise RuntimeError(
            "Could not find a loopback device matching the default output device. "
            "Try listing devices with p.get_loopback_device_info_generator()."
        )
    return default_speakers


def dbfs_from_chunk(data: bytes) -> float:
    samples = np.frombuffer(data, dtype=np.int16)
    if samples.size == 0:
        return -100.0
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
    if rms <= 0:
        return -100.0
    # 32768 is the max amplitude for 16-bit audio
    db = 20 * np.log10(rms / 32768.0)
    return db


def high_freq_dbfs(data: bytes, sample_rate: int, cutoff_hz: float) -> float:
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)
    n = samples.size
    if n == 0:
        return -100.0

    windowed = samples * np.hanning(n)
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    mask = freqs >= cutoff_hz
    if not np.any(mask):
        return -100.0

    magnitudes = np.abs(spectrum[mask])
    high_rms = np.sqrt(np.mean(magnitudes ** 2)) / (n / 2)
    if high_rms <= 0:
        return -100.0

    return 20 * np.log10(high_rms / 32768.0)


def dominant_frequency(data: bytes, sample_rate: int) -> float:
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)
    n = samples.size
    if n == 0:
        return 0.0

    windowed = samples * np.hanning(n)
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    magnitudes = np.abs(spectrum)
    # Ignore bin 0 (DC / zero Hz), which is meaningless as a "pitch"
    if magnitudes.size <= 1:
        return 0.0
    peak_index = np.argmax(magnitudes[1:]) + 1
    return float(freqs[peak_index])


def unmute_after_delay(volume_iface, delay: float, state: dict):
    time.sleep(delay)
    set_mute(volume_iface, False)
    state["muted_until"] = 0.0
    state["cooldown_until"] = time.time() + COOLDOWN_AFTER_UNMUTE
    print(f"[{time.strftime('%H:%M:%S')}] Unmuted automatically.")


def main():
    p = pyaudio.PyAudio()
    volume_iface = get_volume_interface()

    device = get_default_loopback_device(p)
    print(f"Monitoring loopback device: {device['name']}")
    print(f"Loudness threshold: {THRESHOLD_DB} dBFS")
    if ENABLE_FREQUENCY_TRIGGER:
        print(f"Frequency threshold: >= {FREQ_CUTOFF_HZ} Hz content at {FREQ_TRIGGER_DB} dBFS")
    print(f"Mute duration: {MUTE_DURATION_SECONDS}s")

    sample_rate = int(device["defaultSampleRate"])
    stream = p.open(
        format=pyaudio.paInt16,
        channels=device["maxInputChannels"],
        rate=sample_rate,
        input=True,
        input_device_index=device["index"],
        frames_per_buffer=1024,
    )

    state = {"muted_until": 0.0, "cooldown_until": 0.0}
    last_debug_print = 0.0

    try:
        while True:
            data = stream.read(1024, exception_on_overflow=False)
            level_db = dbfs_from_chunk(data)

            loud_trigger = level_db >= THRESHOLD_DB
            freq_trigger = False
            freq_db = None
            if ENABLE_FREQUENCY_TRIGGER:
                freq_db = high_freq_dbfs(data, sample_rate, FREQ_CUTOFF_HZ)
                freq_trigger = freq_db >= FREQ_TRIGGER_DB

            now = time.time()

            if DEBUG and (now - last_debug_print) >= DEBUG_PRINT_INTERVAL:
                dom_freq = dominant_frequency(data, sample_rate)
                freq_part = f" | high-freq(>={FREQ_CUTOFF_HZ}Hz): {freq_db:6.1f} dBFS" if ENABLE_FREQUENCY_TRIGGER else ""
                print(
                    f"[DEBUG] volume: {level_db:6.1f} dBFS{freq_part} "
                    f"| dominant pitch: {dom_freq:6.0f} Hz "
                    f"| muted: {is_muted(volume_iface)}"
                )
                last_debug_print = now

            if (
                (loud_trigger or freq_trigger)
                and not is_muted(volume_iface)
                and now >= state["cooldown_until"]
            ):
                set_mute(volume_iface, True)
                state["muted_until"] = now + MUTE_DURATION_SECONDS

                if loud_trigger and freq_trigger:
                    reason = f"loud ({level_db:.1f} dBFS) + high-pitched ({freq_db:.1f} dBFS)"
                elif loud_trigger:
                    reason = f"loud ({level_db:.1f} dBFS)"
                else:
                    reason = f"high-pitched ({freq_db:.1f} dBFS @ >={FREQ_CUTOFF_HZ}Hz)"

                print(
                    f"[{time.strftime('%H:%M:%S')}] Trigger: {reason} "
                    f"-> muted for {MUTE_DURATION_SECONDS}s"
                )
                threading.Thread(
                    target=unmute_after_delay,
                    args=(volume_iface, MUTE_DURATION_SECONDS, state),
                    daemon=True,
                ).start()

            time.sleep(CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
    