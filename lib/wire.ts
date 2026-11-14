// The three wires: OpenAI Chat Completions, Anthropic Messages, Gemini
// generateContent. This file translates requests and answers between them.
//
// What the trial's client checks on every answer (and what every translation
// below is built to preserve):
//   - the provider's answer text arrives WHOLE: it starts with "ANSWER ECHO"
//     and carries the echo of what the provider parsed (system prompt, user
//     text, tool names, max_tokens, stop sequences, image, JSON mode). A field
//     the translation rewrites or drops loses the request, so requests pass
//     through with every echo field intact: system text, user text (images
//     included), tool names, stop sequences, max_tokens and JSON mode.
//   - usage counts are the provider's own: prompt and completion tokens exactly
//     as the provider reported them. An OpenAI stream only carries usage when
//     the request asked for it, so upstream OpenAI calls always set
//     stream_options.include_usage, and the usage chunk is forwarded.
//   - the stop reason maps end_turn to stop, max_tokens to length, tool_use to
//     tool_calls and refusal to content_filter. The simulated providers always
//     answer with the plain stop family, so client answers say "stop" on the
//     OpenAI wire and "end_turn" on the Anthropic one.
//   - streaming event order: OpenAI chunks end with a finish chunk and [DONE];
//     Anthropic events run message_start, content_block_start, deltas,
//     content_block_stop, message_delta, message_stop. An Anthropic stream
//     needs its input tokens in message_start, before any content, so a
//     stream from an OpenAI or Gemini provider (whose usage arrives last) is
//     buffered and re-emitted; same-wire streams pass through untouched.
//
// Everything here is a pure function of its arguments: no clock, no
// randomness, fixed ids and timestamps, so a replay emits the same bytes.

import { isRecord, type Json } from './brief.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** Fixed epoch second for every listed model's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;
/** Fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({
      id: m.id,
      object: 'model',
      created: FIXED_OPENAI_CREATED,
      owned_by: m.provider,
      free: true,
    })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.display_name,
      created_at: FIXED_ANTHROPIC_CREATED_AT,
      free: true,
    })),
    has_more: false,
  };
}

/** An error in Anthropic's shape. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** One server-sent event line block. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** The text of a string or of a block array's text blocks. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

/** Split a data: URL into its media type and base64 payload, or null. */
function dataUrlParts(url: string): { media: string; data: string } | null {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { media: m[1] !== '' ? m[1] : 'image/png', data: m[2] };
}

// ---------------------------------------------------------------------------
// Images: the same picture in three spellings.
// ---------------------------------------------------------------------------

/** An OpenAI image_url part as an Anthropic image block. */
function openAIImageToAnthropic(part: Json): Json | null {
  const url = isRecord(part.image_url) ? stringOf(part.image_url.url) : '';
  if (url === '') return null;
  const data = dataUrlParts(url);
  if (data !== null) {
    return { type: 'image', source: { type: 'base64', media_type: data.media, data: data.data } };
  }
  return { type: 'image', source: { type: 'url', url } };
}

/** An Anthropic image block as an OpenAI image_url part. */
function anthropicImageToOpenAI(block: Json): Json | null {
  if (!isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** An OpenAI image_url part as a Gemini content part. */
function openAIImageToGemini(part: Json): Json | null {
  const url = isRecord(part.image_url) ? stringOf(part.image_url.url) : '';
  if (url === '') return null;
  const data = dataUrlParts(url);
  if (data !== null) return { inlineData: { mimeType: data.media, data: data.data } };
  return { fileData: { fileUri: url } };
}

/** An Anthropic image block as a Gemini content part. */
function anthropicImageToGemini(block: Json): Json | null {
  if (!isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
    return { inlineData: { mimeType: media, data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { fileData: { fileUri: source.url } };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Tools: the same function in three spellings. Only the names are echoed,
// but the definitions pass through whole so nothing is lost.
// ---------------------------------------------------------------------------

/** OpenAI tools as Anthropic tools. */
function openAIToolsToAnthropic(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: stringOf(raw.function.description),
        input_schema: isRecord(raw.function.parameters)
          ? raw.function.parameters
          : { type: 'object', properties: {} },
      },
    ];
  });
}

/** Anthropic tools as OpenAI tools. */
function anthropicToolsToOpenAI(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        type: 'function',
        function: {
          name: raw.name,
          description: stringOf(raw.description),
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      },
    ];
  });
}

/** Either wire's tools as Gemini function declarations. */
function toolsToGeminiDeclarations(clientApi: 'openai' | 'anthropic', tools: unknown): Json[] {
  if (clientApi === 'openai') {
    if (!Array.isArray(tools)) return [];
    return tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: stringOf(raw.function.description),
          parameters: isRecord(raw.function.parameters)
            ? raw.function.parameters
            : { type: 'object', properties: {} },
        },
      ];
    });
  }
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        name: raw.name,
        description: stringOf(raw.description),
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    ];
  });
}

// ---------------------------------------------------------------------------
// Request translation: client wire -> provider wire.
// ---------------------------------------------------------------------------

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function openAIStopToAnthropic(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function anthropicStopToOpenAI(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
export function geminiStopToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
export function geminiStopToAnthropic(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  return 'end_turn';
}

function maxTokensOf(body: Json): number | null {
  if (typeof body.max_tokens === 'number') return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number') return body.max_completion_tokens;
  return null;
}

function stopsOfOpenAI(body: Json): string[] {
  if (typeof body.stop === 'string') return [body.stop];
  if (Array.isArray(body.stop)) return body.stop.filter((s): s is string => typeof s === 'string');
  return [];
}

function stopsOfAnthropic(body: Json): string[] {
  if (Array.isArray(body.stop_sequences)) {
    return body.stop_sequences.filter((s): s is string => typeof s === 'string');
  }
  return [];
}

/** An Anthropic Messages request as an OpenAI Chat Completions request. */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        const resultContent =
          typeof block.content === 'string'
            ? block.content
            : Array.isArray(block.content)
              ? textOf(block.content)
              : JSON.stringify(block.content ?? '');
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: resultContent,
        });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') {
        parts.push({ type: 'text', text: block.text });
      } else if (block.type === 'image') {
        const part = anthropicImageToOpenAI(block);
        if (part !== null) parts.push(part);
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  const maxTokens = maxTokensOf(body);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopsOfAnthropic(body);
  if (stops.length > 0) out.stop = stops;
  const tools = anthropicToolsToOpenAI(body.tools);
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    // The OpenAI wire only reports usage when asked: always ask, so the
    // usage chunk arrives and can be forwarded (or translated) in full.
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** An OpenAI Chat Completions request as an Anthropic Messages request. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  let system = '';
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
          },
        ],
      });
      continue;
    }
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: Json = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) input = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        } else if (isRecord(call.function.arguments)) {
          input = call.function.arguments;
        }
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input,
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    const blocks: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') {
        blocks.push({ type: 'text', text: block.text });
      } else if (block.type === 'image_url') {
        const image = openAIImageToAnthropic(block);
        if (image !== null) blocks.push(image);
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  const maxTokens = maxTokensOf(body);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopsOfOpenAI(body);
  if (stops.length > 0) out.stop_sequences = stops;
  const tools = openAIToolsToAnthropic(body.tools);
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function)) {
    out.tool_choice = { type: 'tool', name: stringOf(choice.function.name) };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** OpenAI messages as Gemini contents; the system message becomes the instruction. */
function openAIMessagesToGemini(messages: unknown): { system: string; contents: Json[] } {
  let system = '';
  const contents: Json[] = [];
  if (!Array.isArray(messages)) return { system, contents };
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (raw.role === 'tool') {
      const text = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      if (text !== '') contents.push({ role: 'user', parts: [{ text }] });
      continue;
    }
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const content = raw.content;
    const parts: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') {
          parts.push({ text: block.text });
        } else if (block.type === 'image_url') {
          const part = openAIImageToGemini(block);
          if (part !== null) parts.push(part);
        }
      }
    }
    if (parts.length > 0) contents.push({ role, parts });
  }
  return { system, contents };
}

/** Anthropic messages as Gemini contents; the system becomes the instruction. */
function anthropicMessagesToGemini(
  systemBody: unknown,
  messages: unknown,
): { system: string; contents: Json[] } {
  const system = textOf(systemBody);
  const contents: Json[] = [];
  if (!Array.isArray(messages)) return { system, contents };
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const content = raw.content;
    const parts: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') {
          parts.push({ text: block.text });
        } else if (block.type === 'image') {
          const part = anthropicImageToGemini(block);
          if (part !== null) parts.push(part);
        } else if (block.type === 'tool_result') {
          const text =
            typeof block.content === 'string'
              ? block.content
              : Array.isArray(block.content)
                ? textOf(block.content)
                : '';
          if (text !== '') parts.push({ text });
        }
      }
    }
    if (parts.length > 0) contents.push({ role, parts });
  }
  return { system, contents };
}

/** The shared Gemini generation config: output cap, stops, JSON mode. */
function geminiConfig(opts: {
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
}): Json {
  const config: Json = {};
  if (opts.maxTokens !== null) config.maxOutputTokens = opts.maxTokens;
  if (opts.stops.length > 0) config.stopSequences = opts.stops;
  if (opts.jsonMode) config.responseMimeType = 'application/json';
  return config;
}

/** An OpenAI Chat Completions request as a Gemini generateContent body. */
export function openAIToGemini(body: Json): Json {
  const { system, contents } = openAIMessagesToGemini(body.messages);
  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = toolsToGeminiDeclarations('openai', body.tools);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const jsonMode =
    isRecord(body.response_format) &&
    (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const config = geminiConfig({
    maxTokens: maxTokensOf(body),
    stops: stopsOfOpenAI(body),
    jsonMode,
  });
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent body. */
export function anthropicToGemini(body: Json): Json {
  const { system, contents } = anthropicMessagesToGemini(body.system, body.messages);
  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = toolsToGeminiDeclarations('anthropic', body.tools);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  // Anthropic's wire carries no JSON-mode flag, so its requests never ask
  // for one here; the model's own json capability is what the router checks.
  const config = geminiConfig({
    maxTokens: maxTokensOf(body),
    stops: stopsOfAnthropic(body),
    jsonMode: false,
  });
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Response translation: provider wire -> client wire (JSON bodies).
// ---------------------------------------------------------------------------

export interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: string;
}

/** The text of a Gemini candidates array, joined in order. */
function geminiTextOf(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : null;
  const content = first !== null && isRecord(first.content) ? first.content : null;
  const parts = content !== null && Array.isArray(content.parts) ? content.parts : [];
  return parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
}

/** A Gemini answer body as its text, usage and raw stop reason. */
export function geminiAnswerOf(body: Json): ProviderAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : null;
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: geminiTextOf(body),
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
    stop: first !== null && typeof first.finishReason === 'string' ? first.finishReason : 'STOP',
  };
}

/** An OpenAI answer body as its text, usage and stop reason. */
export function openAIAnswerOf(body: Json): ProviderAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: typeof message.content === 'string' ? message.content : textOf(message.content),
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
  };
}

/** An Anthropic answer body as its text, usage and stop reason. */
export function anthropicAnswerOf(body: Json): ProviderAnswer {
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: textOf(body.content),
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
  };
}

/** A provider's JSON answer as an OpenAI Chat Completions object. */
export function answerToOpenAI(
  providerWire: 'openai' | 'anthropic' | 'gemini',
  body: Json,
  model: string,
): Json {
  const answer =
    providerWire === 'openai'
      ? openAIAnswerOf(body)
      : providerWire === 'anthropic'
        ? anthropicAnswerOf(body)
        : geminiAnswerOf(body);
  const finish =
    providerWire === 'anthropic'
      ? anthropicStopToOpenAI(answer.stop)
      : providerWire === 'gemini'
        ? geminiStopToOpenAI(answer.stop)
        : answer.stop;
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: finish,
      },
    ],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** A provider's JSON answer as an Anthropic Messages object. */
export function answerToAnthropic(
  providerWire: 'openai' | 'anthropic' | 'gemini',
  body: Json,
  model: string,
): Json {
  const answer =
    providerWire === 'openai'
      ? openAIAnswerOf(body)
      : providerWire === 'anthropic'
        ? anthropicAnswerOf(body)
        : geminiAnswerOf(body);
  const stop =
    providerWire === 'openai'
      ? openAIStopToAnthropic(answer.stop)
      : providerWire === 'gemini'
        ? geminiStopToAnthropic(answer.stop)
        : answer.stop;
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: stop,
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

// ---------------------------------------------------------------------------
// Streaming: parse an upstream SSE byte stream into frames, and emit a
// client's stream in the right order.
// ---------------------------------------------------------------------------

export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split SSE text into frames, tolerating CRLF. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** Read an upstream SSE body fully and return its frames. */
export async function readSseFrames(stream: ReadableStream<Uint8Array>): Promise<SseFrame[]> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let text = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      text += decoder.decode(value, { stream: true });
    }
    text += decoder.decode();
  } finally {
    reader.releaseLock();
  }
  return parseSseFrames(text);
}

/** One OpenAI chat chunk. */
export function openAIChunk(
  model: string,
  delta: Json,
  finish: string | null,
): Json {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  };
}

/** The OpenAI usage chunk that closes a stream which asked for usage. */
export function openAIUsageChunk(model: string, input: number, output: number): Json {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: {
      prompt_tokens: input,
      completion_tokens: output,
      total_tokens: input + output,
    },
  };
}

/**
 * An Anthropic provider's event stream as OpenAI chunks: text deltas become
 * content chunks, message_delta becomes the finish chunk plus the usage
 * chunk, and the stream ends with [DONE]. Runs on the fly, in arrival order.
 */
export function anthropicFramesToOpenAI(frames: SseFrame[], model: string): string[] {
  const out: string[] = [];
  let started = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'end_turn';
  const start = (): void => {
    if (started) return;
    started = true;
    out.push(sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null)));
  };
  for (const frame of frames) {
    if (frame.event === null) continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta') {
      start();
      if (isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
        out.push(sseEvent(null, openAIChunk(model, { content: data.delta.text }, null)));
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') {
        stop = data.delta.stop_reason;
      }
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  start();
  out.push(
    sseEvent(null, openAIChunk(model, {}, anthropicStopToOpenAI(stop))),
    sseEvent(null, openAIUsageChunk(model, inputTokens, outputTokens)),
    'data: [DONE]\n\n',
  );
  return out;
}

/**
 * A Gemini provider's event stream as OpenAI chunks: each frame's text
 * becomes a content chunk, and the frame carrying finishReason closes the
 * stream with the finish chunk, the usage chunk and [DONE].
 */
export function geminiFramesToOpenAI(frames: SseFrame[], model: string): string[] {
  const out: string[] = [];
  let started = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let reason: unknown = null;
  const start = (): void => {
    if (started) return;
    started = true;
    out.push(sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null)));
  };
  for (const frame of frames) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : null;
    const text =
      first !== null && isRecord(first.content) && Array.isArray(first.content.parts)
        ? first.content.parts
            .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
            .join('')
        : '';
    if (text !== '') {
      start();
      out.push(sseEvent(null, openAIChunk(model, { content: text }, null)));
    }
    if (first !== null && typeof first.finishReason === 'string') reason = first.finishReason;
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  start();
  out.push(
    sseEvent(null, openAIChunk(model, {}, geminiStopToOpenAI(reason))),
    sseEvent(null, openAIUsageChunk(model, inputTokens, outputTokens)),
    'data: [DONE]\n\n',
  );
  return out;
}

/**
 * A provider's streamed answer, fully read, as the Anthropic event sequence:
 * message_start (with the input tokens), one text block, message_delta (with
 * the stop reason and output tokens), message_stop. The input tokens are
 * only known once the upstream stream ends, so this buffers first.
 */
export function bufferedToAnthropic(
  providerWire: 'openai' | 'gemini',
  frames: SseFrame[],
  model: string,
): string[] {
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let finish: string | null = null;
  if (providerWire === 'openai') {
    for (const frame of frames) {
      if (frame.data.trim() === '[DONE]') continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(frame.data);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      if (isRecord(choice.delta) && typeof choice.delta.content === 'string') {
        text += choice.delta.content;
      }
      if (typeof choice.finish_reason === 'string') finish = choice.finish_reason;
    }
  } else {
    for (const frame of frames) {
      let data: unknown;
      try {
        data = JSON.parse(frame.data);
      } catch {
        continue;
      }
      if (!isRecord(data)) continue;
      const candidates = Array.isArray(data.candidates) ? data.candidates : [];
      const first = isRecord(candidates[0]) ? candidates[0] : null;
      if (first !== null && isRecord(first.content) && Array.isArray(first.content.parts)) {
        text += first.content.parts
          .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
          .join('');
      }
      if (first !== null && typeof first.finishReason === 'string') finish = first.finishReason;
      if (isRecord(data.usageMetadata)) {
        inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
        outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
      }
    }
  }
  const stop =
    providerWire === 'openai' ? openAIStopToAnthropic(finish) : geminiStopToAnthropic(finish);
  return [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    }),
    ...(text === ''
      ? []
      : [
          sseEvent('content_block_delta', {
            type: 'content_block_delta',
            index: 0,
            delta: { type: 'text_delta', text },
          }),
        ]),
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stop, stop_sequence: null },
      usage: { output_tokens: outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  ];
}
