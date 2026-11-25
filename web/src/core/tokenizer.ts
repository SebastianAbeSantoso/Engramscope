// The model's tokenizer in the browser: @huggingface/tokenizers 0.2.0 on the pinned tokenizer.json
// (Gate G2: 0 mismatches against Python `tokenizers` on 1,000 strings).

import { Tokenizer } from '@huggingface/tokenizers'

export class ModelTokenizer {
  private readonly tok: Tokenizer

  constructor(tokenizerJson: object, tokenizerConfig: object) {
    this.tok = new Tokenizer(tokenizerJson, tokenizerConfig)
  }

  /** Token ids, as `Tokenizer.encode(text)` in the Python reference (special tokens added). */
  encode(text: string): number[] {
    return Array.from(this.tok.encode(text, { add_special_tokens: true }).ids)
  }
}

/** The text the reference encodes for a phrase: a leading space, as in the benchmark. */
export function phraseText(phrase: string): string {
  return ' ' + phrase.trim()
}
