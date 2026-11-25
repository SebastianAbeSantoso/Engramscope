// Decision 6: the app names the model in plain text, except with the ?generic URL flag (for recording
// the video), which swaps the name for a generic label.

export const GENERIC = new URLSearchParams(location.search).has('generic')

export const MODEL_NAME = GENERIC ? 'an open-weight frontier model' : 'DeepSeek-V4.1-Flash'
export const MODEL_NAME_CAP = GENERIC ? 'An open-weight frontier model' : 'DeepSeek-V4.1-Flash'
export const MODEL_OWNER = GENERIC ? "the model's developers" : 'DeepSeek'
