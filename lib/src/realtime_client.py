"""
Generic WebSocket client    s
Provider-agnostic design, use whatever
"""

import sys
import json
import base64
import threading
import time
from typing import Optional
from queue import Queue, Empty
from collections import deque

try:
    import numpy as np
except (ImportError, ModuleNotFoundError) as e:
    print("ERROR: python-numpy is not available in this Python environment.", file=sys.stderr)
    print(f"ImportError: {e}", file=sys.stderr)
    sys.exit(1)

try:
    import websocket
except (ImportError, ModuleNotFoundError) as e:
    print("ERROR: websocket-client is not available in this Python environment.", file=sys.stderr)
    print(f"ImportError: {e}", file=sys.stderr)
    print("\nThis is a required dependency. Please install it:", file=sys.stderr)
    print("  pip install websocket-client>=1.6.0", file=sys.stderr)
    sys.exit(1)


class RealtimeClient:
    """Generic WebSocket client for realtime transcription APIs"""
    
    def __init__(self, mode: str = 'transcribe'):
        """
        Realtime client for transcription or conversation.
        
        Args:
            mode: 'transcribe' for speech-to-text, 'converse' for voice-to-AI
        """
        self.ws = None
        self.url = None
        self.api_key = None
        self.model = None
        self.instructions = None
        self.mode = mode
        self.language = None  # Language code for transcription (None = auto-detect)
        
        # Connection state
        self.connected = False
        self.connecting = False
        self.receiver_thread = None
        self.receiver_running = False
        
        # Event handling
        self.event_queue = Queue()
        self.response_event = threading.Event()
        self.current_response_text = ""
        self.response_complete = False
        
        # Audio streaming
        self.audio_chunks = deque()
        self.audio_buffer_seconds = 0.0
        self.max_buffer_seconds = 5.0
        self.sample_rate = 24000  # OpenAI Realtime API requires 24kHz
        
        # Reconnection
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delays = [1, 2, 4, 8, 16]  # Exponential backoff
        
        # Threading
        self.lock = threading.Lock()

        # Track if buffer was committed (by VAD or manual)
        # Prevents double-commit error when VAD auto-commits on speech end
        self._buffer_committed = False

        # Track if speech is in progress (started but not yet transcribed)
        # This helps us know if we need to wait for a pending transcription
        self._speech_in_progress = False

        # Track number of pending transcriptions (committed but not yet transcribed)
        # VAD can commit multiple segments, each needs a transcription.completed
        self._pending_transcriptions = 0

        # Track when audio was last sent (for detecting uncommitted audio)
        self._last_audio_sent_time = 0

        # VAD turn detection configuration (defaults)
        self.vad_threshold = 0.5
        self.vad_prefix_padding_ms = 300
        self.vad_silence_duration_ms = 500
        
    def connect(self, url: str, api_key: str, model: str, instructions: Optional[str] = None) -> bool:
        """
        Establish WebSocket connection with authentication.
        
        Args:
            url: WebSocket URL (e.g., 'wss://api.openai.com/v1/realtime?model=gpt-realtime-mini-2025-12-15')
            api_key: API key for authentication
            model: Model identifier
            instructions: Optional session instructions/prompt
        
        Returns:
            True if connection successful, False otherwise
        """
        self.url = url
        self.api_key = api_key
        self.model = model
        self.instructions = instructions
        
        return self._connect_internal()
    
    def _connect_internal(self) -> bool:
        """Internal connection logic with reconnection support"""
        if self.connecting:
            return False
        
        self.connecting = True
        
        try:
            # Prepare headers with authentication
            headers = {
                'Authorization': f'Bearer {self.api_key}'
            }
            
            print(f'[REALTIME] Connecting to {self.url}...', flush=True)
            
            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(
                self.url,
                header=[f'{k}: {v}' for k, v in headers.items()],
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start WebSocket in a separate thread
            ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            ws_thread.start()
            
            # Wait for connection (with timeout)
            timeout = 10.0
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.connected:
                print(f'[REALTIME] Connected successfully', flush=True)
                self.reconnect_attempts = 0
                
                # Always send session.update with audio format configuration
                self._send_session_update()
                
                return True
            else:
                print(f'[REALTIME] Connection timeout', flush=True)
                return False
                
        except Exception as e:
            print(f'[REALTIME] Connection error: {e}', flush=True)
            return False
        finally:
            self.connecting = False
    
    def _on_open(self, _ws):
        """WebSocket connection opened"""
        with self.lock:
            self.connected = True
            self.connecting = False
        
        # Start receiver thread
        if not self.receiver_running:
            self.receiver_running = True
            self.receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
            self.receiver_thread.start()
    
    def _on_message(self, _ws, message):
        """Handle incoming WebSocket message"""
        try:
            event = json.loads(message)
            self.event_queue.put(event)
        except json.JSONDecodeError as e:
            print(f'[REALTIME] Failed to parse event: {e}', flush=True)
    
    def _on_error(self, _ws, error):
        """Handle WebSocket error"""
        print(f'[REALTIME] WebSocket error: {error}', flush=True)
    
    def _on_close(self, _ws, close_status_code, _close_msg):
        """Handle WebSocket close"""
        with self.lock:
            self.connected = False
        
        print(f'[REALTIME] WebSocket closed (code: {close_status_code})', flush=True)
        
        # Attempt reconnection if not intentionally closed
        if self.receiver_running and close_status_code != 1000:  # 1000 = normal closure
            self._attempt_reconnect()
    
    def _receiver_loop(self):
        """Background thread to process incoming events"""
        while self.receiver_running:
            try:
                # Get event with timeout
                event = self.event_queue.get(timeout=0.1)
                self._handle_event(event)
            except Empty:
                continue
            except Exception as e:
                print(f'[REALTIME] Error in receiver loop: {e}', flush=True)
    
    def _handle_event(self, event: dict):
        """Handle a single event from the server"""
        event_type = event.get('type', '')

        # Log session events
        if event_type in ('session.created', 'session.updated'):
            print(f'[REALTIME] Session event: {event_type}', flush=True)
        
        # Response events (conversational API)
        elif event_type == 'response.created':
            print(f'[REALTIME] Response created', flush=True)
            with self.lock:
                self.current_response_text = ""
                self.response_complete = False
            self.response_event.clear()
        
        elif event_type == 'response.output_text.delta':
            # Accumulate text deltas
            delta = event.get('delta', '')
            if delta:
                with self.lock:
                    self.current_response_text += delta
        
        elif event_type == 'response.output_text.done':
            # Final text available
            text = event.get('text', '')
            with self.lock:
                if text:
                    self.current_response_text = text
                text_len = len(self.current_response_text)
            print(f'[REALTIME] Response text done ({text_len} chars)', flush=True)

        elif event_type == 'response.done':
            # Response complete
            with self.lock:
                self.response_complete = True
            self.response_event.set()
            print(f'[REALTIME] Response done', flush=True)
        
        # Transcription events
        # VAD can send multiple transcription events when it detects silences
        # We accumulate them to get the complete transcription
        elif event_type == 'conversation.item.input_audio_transcription.completed':
            transcript = event.get('transcript', '').strip()
            with self.lock:
                if transcript:
                    if self.current_response_text:
                        self.current_response_text += ' ' + transcript
                    else:
                        self.current_response_text = transcript
                # Note: Do NOT reset _speech_in_progress here - a new speech segment
                # may have started while this transcription was pending. Let speech_stopped
                # handle that, or reset at the end of commit_and_get_text.
                # Decrement pending transcriptions counter
                if self._pending_transcriptions > 0:
                    self._pending_transcriptions -= 1
                # Only mark complete if no more pending transcriptions
                self.response_complete = (self._pending_transcriptions == 0)
            # Signal if no more pending (or always signal to update waiters)
            self.response_event.set()
            print(f'[REALTIME] Transcription segment completed ({len(transcript)} chars) | pending={self._pending_transcriptions}', flush=True)

        elif event_type == 'conversation.item.input_audio_transcription.failed':
            # Transcription failed (e.g., audio unintelligible)
            error = event.get('error', {})
            error_message = error.get('message', 'Unknown transcription error')
            with self.lock:
                # Note: Do NOT reset _speech_in_progress here - same reason as above
                # Decrement pending transcriptions counter
                if self._pending_transcriptions > 0:
                    self._pending_transcriptions -= 1
                self.response_complete = (self._pending_transcriptions == 0)
            self.response_event.set()
            print(f'[REALTIME] Transcription segment failed: {error_message} | pending={self._pending_transcriptions}', flush=True)

        elif event_type == 'input_audio_buffer.committed':
            with self.lock:
                self._buffer_committed = True
                self._pending_transcriptions += 1
            print(f'[REALTIME] Audio buffer committed (pending transcription segments={self._pending_transcriptions})', flush=True)

        elif event_type == 'input_audio_buffer.speech_started':
            with self.lock:
                # New speech detected - mark that we have audio pending transcription
                self._speech_in_progress = True
                # Reset commit tracking - new speech means we haven't committed THIS audio yet
                self._buffer_committed = False
            print(f'[REALTIME] Speech detected', flush=True)

        elif event_type == 'input_audio_buffer.speech_stopped':
            with self.lock:
                self._speech_in_progress = False
            print(f'[REALTIME] Speech ended', flush=True)

        elif event_type == 'error':
            error = event.get('error', {})
            error_message = error.get('message', 'Unknown error')
            print(f'[REALTIME] Server error: {error_message}', flush=True)
            self.response_complete = True
            self.response_event.set()  # Unblock waiting thread
    
    def _send_session_update(self):
        """Send session.update event based on mode"""
        if not self.connected or not self.ws:
            return
        
        if self.mode == 'transcribe':
            # Transcription-only session
            # Build transcription config - omit language for auto-detect
            transcription_config = {'model': self.model}
            if self.language:
                transcription_config['language'] = self.language

            session_data = {
                'type': 'transcription',
                'audio': {
                    'input': {
                        'format': {
                            'type': 'audio/pcm',
                            'rate': 24000
                        },
                        'transcription': transcription_config,
                        'turn_detection': {
                            'type': 'server_vad',
                            'threshold': self.vad_threshold,
                            'prefix_padding_ms': self.vad_prefix_padding_ms,
                            'silence_duration_ms': self.vad_silence_duration_ms
                        }
                    }
                }
            }
        else:
            # Conversational session (voice-to-AI) - no VAD, manual commit
            session_data = {
                'type': 'realtime',
                'output_modalities': ['text'],  # Text output only (no audio response)
                'audio': {
                    'input': {
                        'format': {
                            'type': 'audio/pcm',
                            'rate': 24000
                        },
                        'turn_detection': None  # Manual commit on stop
                    }
                },
                'instructions': self.instructions or 'You are a helpful assistant. Respond to the user based on what they say.'
            }
        
        event = {
            'type': 'session.update',
            'session': session_data
        }
        
        try:
            self.ws.send(json.dumps(event))
            print(f'[REALTIME] Sent session.update', flush=True)
        except Exception as e:
            print(f'[REALTIME] Failed to send session.update: {e}', flush=True)
    
    def update_language(self, language: Optional[str]):
        """Update the language for transcription and resend session.update
        
        Args:
            language: Language code (e.g., 'en', 'it', 'fr') or None for auto-detect
        """
        self.language = language
        if self.connected:
            self._send_session_update()
    
    def _attempt_reconnect(self):
        """Attempt to reconnect with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            print(f'[REALTIME] Max reconnection attempts reached', flush=True)
            return False
        
        delay = self.reconnect_delays[min(self.reconnect_attempts, len(self.reconnect_delays) - 1)]
        self.reconnect_attempts += 1
        
        print(f'[REALTIME] Reconnecting (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}) in {delay}s...', flush=True)
        time.sleep(delay)
        
        if self._connect_internal():
            # Re-send session.update after reconnect (always needed for audio format)
            self._send_session_update()
            return True
        
        return False
    
    def _float32_to_pcm16(self, audio_data: np.ndarray) -> bytes:
        """Convert float32 numpy array to PCM16 bytes"""
        # Clip to [-1, 1] range
        audio_clipped = np.clip(audio_data, -1.0, 1.0)
        
        # Convert to int16
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        
        # Convert to bytes (little-endian)
        return audio_int16.tobytes()
    
    def clear_audio_buffer(self):
        """Clear the server-side audio buffer before starting a new recording."""
        if not self.connected or not self.ws:
            return
        try:
            event = {'type': 'input_audio_buffer.clear'}
            self.ws.send(json.dumps(event))
            self.audio_buffer_seconds = 0.0
            with self.lock:
                self._buffer_committed = False  # Reset commit tracking for new recording
                self._speech_in_progress = False  # Reset speech tracking for new recording
                self._pending_transcriptions = 0  # Reset pending transcriptions counter
                # Clear old transcription state to prevent returning stale results
                self.current_response_text = ""
                self.response_complete = False
            self.response_event.clear()
        except Exception as e:
            print(f'[REALTIME] Failed to clear buffer: {e}', flush=True)
    
    def append_audio(self, audio_chunk: np.ndarray):
        """
        Append audio chunk to WebSocket stream.

        Args:
            audio_chunk: NumPy array of audio samples (float32, mono, 16kHz)
        """
        if not self.connected or not self.ws:
            return

        try:
            # Convert to PCM16
            pcm_bytes = self._float32_to_pcm16(audio_chunk)

            # Encode to base64
            base64_audio = base64.b64encode(pcm_bytes).decode('utf-8')

            # Send input_audio_buffer.append event
            event = {
                'type': 'input_audio_buffer.append',
                'audio': base64_audio
            }

            self.ws.send(json.dumps(event))

            # Track when audio was last sent
            self._last_audio_sent_time = time.time()
            
            # Track buffer size for backpressure
            chunk_duration = len(audio_chunk) / self.sample_rate
            self.audio_buffer_seconds += chunk_duration
            
            # Check backpressure
            # Reset buffer counter periodically to prevent overflow (audio is streamed directly)
            if self.audio_buffer_seconds > self.max_buffer_seconds:
                self.audio_buffer_seconds = 0.0
            
        except Exception as e:
            print(f'[REALTIME] Failed to append audio: {e}', flush=True)
    
    def commit_and_get_text(self, timeout: float = 30.0) -> str:
        """
        Commit audio buffer and wait for transcription result.

        With server VAD enabled, transcription happens automatically when speech ends.
        This method commits any remaining audio and waits for the transcript.

        Args:
            timeout: Maximum time to wait for transcription (seconds)

        Returns:
            Final transcript text, or empty string on timeout/error
        """
        if not self.connected or not self.ws:
            print('[REALTIME] Not connected, cannot commit', flush=True)
            return ""

        try:
            with self.lock:
                # Capture current state
                has_text = bool(self.current_response_text)
                speech_pending = self._speech_in_progress
                buffer_committed = self._buffer_committed
                pending_transcriptions = self._pending_transcriptions

                # Prepare for waiting if needed
                self.response_complete = False
                self.response_event.clear()

            print(f'[REALTIME] commit_and_get_text called | has_text={has_text}, speech_pending={speech_pending}, buffer_committed={buffer_committed}, pending_transcriptions={pending_transcriptions}', flush=True)

            # Scenario 1: We have uncommitted audio that VAD didn't detect as speech
            # This MUST be checked before concluding "no speech detected"
            # In converse mode (no VAD), this is the normal path
            time_since_last_audio = time.time() - self._last_audio_sent_time
            if time_since_last_audio < 1.0 and not buffer_committed and not has_text and not speech_pending and pending_transcriptions == 0:
                # Audio was sent recently but VAD never triggered - force a commit to capture any speech
                print(f'[REALTIME] Recent audio ({time_since_last_audio:.2f}s ago) but VAD never triggered (yet), forcing commit...', flush=True)
                commit_event = {'type': 'input_audio_buffer.commit'}
                self.ws.send(json.dumps(commit_event))

                # For converse mode, request a response from the model
                if self.mode == 'converse':
                    response_event = {
                        'type': 'response.create',
                        'response': {
                            'output_modalities': ['text']
                        }
                    }
                    self.ws.send(json.dumps(response_event))

                # Wait for the transcription (transcribe mode) or response (converse mode)
                wait_start = time.time()
                while True:
                    if self.response_event.wait(timeout=timeout):
                        with self.lock:
                            # In converse mode, wait for response.done (response_complete)
                            # In transcribe mode, wait for all pending transcriptions
                            if self.mode == 'converse':
                                if self.response_complete:
                                    break
                            else:
                                if self._pending_transcriptions == 0:
                                    break
                            self.response_event.clear()
                    else:
                        print(f'[REALTIME] TIMEOUT waiting for {"response" if self.mode == "converse" else "transcription"} ({timeout}s)', flush=True)
                        break

                wait_duration = time.time() - wait_start
                with self.lock:
                    result = self.current_response_text.strip()
                    self.current_response_text = ""
                    self.response_complete = False
                    self._buffer_committed = False
                    self._pending_transcriptions = 0
                    self.audio_buffer_seconds = 0.0
                print(f'[REALTIME] Got {"response" if self.mode == "converse" else "transcription"} after forced commit ({wait_duration:.2f}s)', flush=True)
                return result

            # Scenario 2: No speech was ever detected AND no recent audio - nothing to transcribe
            if not has_text and not speech_pending and pending_transcriptions == 0:
                print(f'[REALTIME] No speech detected, nothing to transcribe', flush=True)
                return ""

            # Scenario 3: Speech is still in progress (user stopped before silence_duration)
            # We need to commit and wait for the transcription
            if speech_pending:
                if not buffer_committed:
                    print(f'[REALTIME] Speech pending, commiting and waiting for transcription...', flush=True)
                    commit_event = {'type': 'input_audio_buffer.commit'}
                    self.ws.send(json.dumps(commit_event))
                else:
                    print(f'[REALTIME] Speech pending, already committed, waiting for transcription...', flush=True)

                # For converse mode, request a response from the model
                if self.mode == 'converse':
                    response_event = {
                        'type': 'response.create',
                        'response': {
                            'output_modalities': ['text']
                        }
                    }
                    self.ws.send(json.dumps(response_event))

                # Wait for the transcription (transcribe mode) or response (converse mode)
                wait_start = time.time()
                while True:
                    if self.response_event.wait(timeout=timeout):
                        with self.lock:
                            # In converse mode, wait for response.done (response_complete)
                            # In transcribe mode, wait for all pending transcriptions
                            if self.mode == 'converse':
                                if self.response_complete:
                                    break
                            else:
                                if self._pending_transcriptions == 0:
                                    break
                            self.response_event.clear()
                    else:
                        print(f'[REALTIME] TIMEOUT waiting for {"response" if self.mode == "converse" else "transcription"} ({timeout}s)', flush=True)
                        break

                wait_duration = time.time() - wait_start
                with self.lock:
                    result = self.current_response_text.strip()
                    # Reset state for next recording
                    self.current_response_text = ""
                    self.response_complete = False
                    self._buffer_committed = False
                    self._speech_in_progress = False
                    self._pending_transcriptions = 0
                    self.audio_buffer_seconds = 0.0
                print(f'[REALTIME] Got {"response" if self.mode == "converse" else "transcription"} after {wait_duration:.2f}s', flush=True)
                return result

            # Scenario 4: There are pending transcriptions (VAD committed but transcription not received yet)
            # We need to wait for them
            if pending_transcriptions > 0:
                print(f'[REALTIME] {pending_transcriptions} pending transcription(s), waiting...', flush=True)

                wait_start = time.time()
                while True:
                    if self.response_event.wait(timeout=timeout):
                        with self.lock:
                            if self._pending_transcriptions == 0:
                                break
                            # More pending, keep waiting
                            self.response_event.clear()
                    else:
                        print(f'[REALTIME] TIMEOUT waiting for pending transcription ({timeout}s)', flush=True)
                        break

                wait_duration = time.time() - wait_start
                with self.lock:
                    result = self.current_response_text.strip()
                    # Reset state for next recording
                    self.current_response_text = ""
                    self.response_complete = False
                    self._buffer_committed = False
                    self._pending_transcriptions = 0
                    self.audio_buffer_seconds = 0.0
                print(f'[REALTIME] Got {"response" if self.mode == "converse" else "transcription"} after {wait_duration:.2f}s', flush=True)
                return result

            # Scenario 5: We have accumulated text and no pending transcriptions
            # All VAD segments have been transcribed, return the accumulated text
            with self.lock:
                result = self.current_response_text.strip()
                # Reset state for next recording
                self.current_response_text = ""
                self.response_complete = False
                self._buffer_committed = False
                self.audio_buffer_seconds = 0.0
            return result

        except Exception as e:
            print(f'[REALTIME] Error in commit_and_get_text: {e}', flush=True)
            return ""

    def close(self):
        """Close WebSocket connection and cleanup"""
        self.receiver_running = False
        
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        
        if self.receiver_thread and self.receiver_thread.is_alive():
            self.receiver_thread.join(timeout=1.0)
        
        with self.lock:
            self.connected = False
        
        print('[REALTIME] Connection closed', flush=True)
    
    def set_max_buffer_seconds(self, seconds: float):
        """Set maximum buffer size in seconds for backpressure handling"""
        self.max_buffer_seconds = max(1.0, seconds)  # Minimum 1 second

    def set_vad_config(self, threshold: float = 0.5, prefix_padding_ms: int = 300, silence_duration_ms: int = 500):
        """Configure VAD turn detection parameters.

        Args:
            threshold: Voice detection sensitivity (0.0-1.0, lower = more sensitive)
            prefix_padding_ms: Audio to include before detected speech starts
            silence_duration_ms: Silence duration to trigger a transcription segment
        """
        self.vad_threshold = threshold
        self.vad_prefix_padding_ms = prefix_padding_ms
        self.vad_silence_duration_ms = silence_duration_ms
