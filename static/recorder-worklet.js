/**
 * AudioWorkletProcessor for recording PCM samples.
 * Replaces the deprecated ScriptProcessorNode.
 */
class RecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._active = true;
    this.port.onmessage = (e) => {
      if (e.data === 'stop') this._active = false;
    };
  }

  process(inputs) {
    if (!this._active) return false;
    const input = inputs[0];
    if (input && input[0]) {
      // Copy so the buffer isn't recycled before postMessage delivers it
      this.port.postMessage(new Float32Array(input[0]));
    }
    return true;
  }
}

registerProcessor('recorder-processor', RecorderProcessor);
