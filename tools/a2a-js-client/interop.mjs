// A2A interop check: the official JavaScript SDK client (@a2a-js/sdk, pinned in package-lock.json) talks to
// FinSight's A2A endpoint (Python a2a-sdk server). Every step prints what the JS client saw, and a wire log
// records each HTTP request (JSON-RPC method, A2A-Version header, HTTP status).
//
//   cd tools/a2a-js-client && npm ci
//   node interop.mjs --url http://127.0.0.1:8861 [--api-key KEY] [--json summary.json]
//
// Exit code 0 only when every required check passes. tests/test_a2a_js_interop.py runs this script against
// an in-process server and is skipped when node or node_modules are missing.

import { randomUUID } from 'node:crypto';
import { writeFileSync } from 'node:fs';
import { argv, exit } from 'node:process';

import { Role, TaskState, taskStateToJSON } from '@a2a-js/sdk';
import {
  ClientFactory,
  ClientFactoryOptions,
  DefaultAgentCardResolver,
  JsonRpcTransportFactory,
} from '@a2a-js/sdk/client';

const args = parseArgs(argv.slice(2));
const baseUrl = (args.url ?? 'http://127.0.0.1:8861').replace(/\/$/, '');
const QUESTION = args.question ?? '贵州茅台的市盈率是多少';
const STREAM_QUESTION = args['stream-question'] ?? '贵州茅台最近走势怎么样';
const VAGUE_QUESTION = args['vague-question'] ?? '它的市盈率呢';
const CLARIFICATION_REPLY = args.reply ?? '贵州茅台';
const TERMINAL = new Set([
  TaskState.TASK_STATE_COMPLETED,
  TaskState.TASK_STATE_FAILED,
  TaskState.TASK_STATE_CANCELED,
  TaskState.TASK_STATE_REJECTED,
]);

// ------------------------------------------------------------------------------------------ wire logging

const wire = [];
async function loggingFetch(input, init = {}) {
  const url = input instanceof URL ? input.href : typeof input === 'string' ? input : input.url;
  const headers = new Headers(init.headers ?? {});
  if (args['api-key']) headers.set('X-API-Key', args['api-key']);
  let method = init.method ?? 'GET';
  if (typeof init.body === 'string') {
    try {
      method = `${method} ${JSON.parse(init.body).method}`;
    } catch {
      /* not JSON-RPC */
    }
  }
  const response = await fetch(input, { ...init, headers });
  wire.push({
    request: `${method} ${new URL(url).pathname}`,
    a2a_version: headers.get('A2A-Version'),
    status: response.status,
    content_type: response.headers.get('content-type'),
  });
  return response;
}

// ------------------------------------------------------------------------------------------ helpers

const out = (line = '') => console.log(line);
const stateName = (state) => taskStateToJSON(state);

function textMessage(text, { taskId = '', contextId = '' } = {}) {
  return {
    messageId: randomUUID(),
    contextId,
    taskId,
    role: Role.ROLE_USER,
    parts: [{ content: { $case: 'text', value: text }, mediaType: 'text/plain', filename: '', metadata: undefined }],
    metadata: undefined,
    extensions: [],
    referenceTaskIds: [],
  };
}

function sendRequest(message, configuration = undefined) {
  return { tenant: '', message, configuration, metadata: undefined };
}

function partsText(parts = []) {
  return parts
    .filter((part) => part.content?.$case === 'text')
    .map((part) => part.content.value)
    .join('\n');
}

function partsData(parts = []) {
  const data = parts.find((part) => part.content?.$case === 'data');
  return data ? data.content.value : undefined;
}

function describeTask(task) {
  const artifacts = Object.fromEntries((task.artifacts ?? []).map((artifact) => [artifact.name, artifact]));
  const evidence = artifacts.evidence ? partsData(artifacts.evidence.parts) ?? {} : {};
  return {
    task_id: task.id,
    context_id: task.contextId,
    state: stateName(task.status?.state),
    answer: artifacts.answer ? partsText(artifacts.answer.parts) : '',
    evidence_used: evidence.evidence_used ?? [],
    trace_id: evidence.trace_id ?? null,
    status_message: task.status?.message ? partsText(task.status.message.parts) : '',
    history_length: (task.history ?? []).length,
    artifact_names: (task.artifacts ?? []).map((artifact) => artifact.name),
  };
}

function eventLine(event) {
  const payload = event.payload;
  switch (payload?.$case) {
    case 'task':
      return ['task', `task ${payload.value.id.slice(0, 8)} ${stateName(payload.value.status?.state)}`];
    case 'statusUpdate': {
      const status = payload.value.status;
      const text = status?.message ? partsText(status.message.parts) : '';
      return ['statusUpdate', `${stateName(status?.state)} ${text}`.trim()];
    }
    case 'artifactUpdate': {
      const artifact = payload.value.artifact;
      const preview = partsText(artifact.parts).replace(/\n/g, ' ').slice(0, 70) || '(structured data part)';
      return ['artifactUpdate', `artifact ${artifact.name}: ${preview}`];
    }
    case 'message':
      return ['message', partsText(payload.value.parts).slice(0, 70)];
    default:
      return ['unknown', JSON.stringify(event).slice(0, 70)];
  }
}

function errorInfo(error) {
  return { name: error?.constructor?.name ?? 'Error', code: error?.code ?? null, message: String(error?.message ?? error) };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

const checks = [];
function check(name, ok, detail = '') {
  checks.push({ name, ok: Boolean(ok), detail });
  out(`   [${ok ? 'PASS' : 'FAIL'}] ${name}${detail ? ` (${detail})` : ''}`);
}

// ------------------------------------------------------------------------------------------ steps

async function main() {
  const summary = { base_url: baseUrl };
  const factory = new ClientFactory(
    ClientFactoryOptions.createFrom(ClientFactoryOptions.default, {
      cardResolver: new DefaultAgentCardResolver({ fetchImpl: loggingFetch }),
      transports: [new JsonRpcTransportFactory({ fetchImpl: loggingFetch })],
    }),
  );

  // 1. Discovery
  out(`== 1. Agent card via ClientFactory.createFromUrl(${baseUrl})`);
  const client = await factory.createFromUrl(baseUrl);
  const card = await client.getAgentCard();
  const iface = card.supportedInterfaces[0];
  out(`   ${card.name} ${card.version}; interface ${iface.url} (${iface.protocolBinding} ${iface.protocolVersion})`);
  out(`   streaming=${card.capabilities?.streaming} pushNotifications=${card.capabilities?.pushNotifications}`);
  out(`   skills: ${card.skills.map((skill) => skill.id).join(', ')}; client protocol version ${client.protocolVersion}`);
  summary.card = { name: card.name, skills: card.skills.map((skill) => skill.id), interface: iface };
  check('agent card parsed by the JS SDK', card.name === 'FinSight' && iface.protocolBinding === 'JSONRPC');

  // 2. SendMessage (blocking)
  out(`\n== 2. sendMessage: ${QUESTION}`);
  const sent = await client.sendMessage(sendRequest(textMessage(QUESTION)));
  const answered = describeTask(sent);
  summary.send = answered;
  out(`   state=${answered.state} artifacts=${answered.artifact_names.join(',')} trace=${answered.trace_id}`);
  out(`   evidence_used: ${answered.evidence_used.join(', ')}`);
  out(`   > ${answered.answer.split('\n')[0]}`);
  check('SendMessage returns a completed task', answered.state === 'TASK_STATE_COMPLETED');
  check('answer artifact cites evidence', answered.evidence_used.length > 0 && answered.answer.includes('['));

  // 3. GetTask
  out(`\n== 3. getTask(${answered.task_id.slice(0, 8)}…, historyLength=10)`);
  const fetched = describeTask(await client.getTask({ tenant: '', id: answered.task_id, historyLength: 10 }));
  summary.get = fetched;
  out(`   state=${fetched.state} artifacts=${fetched.artifact_names.join(',')} history=${fetched.history_length}`);
  check('GetTask returns the same completed task', fetched.task_id === answered.task_id && fetched.state === answered.state);
  const trimmed = await client.getTask({ tenant: '', id: answered.task_id, historyLength: 0 });
  check('GetTask honours historyLength=0', (trimmed.history ?? []).length === 0, `history=${(trimmed.history ?? []).length}`);
  try {
    await client.getTask({ tenant: '', id: randomUUID() });
    check('GetTask on an unknown id raises TaskNotFoundError', false);
  } catch (error) {
    summary.get_unknown = errorInfo(error);
    check('GetTask on an unknown id raises TaskNotFoundError', summary.get_unknown.name.includes('TaskNotFound'), summary.get_unknown.name);
  }

  // 4. input-required, then the reply on the same task
  out(`\n== 4. sendMessage (needs clarification): ${VAGUE_QUESTION}`);
  const pending = describeTask(await client.sendMessage(sendRequest(textMessage(VAGUE_QUESTION))));
  summary.clarification = pending;
  out(`   state=${pending.state} question: ${pending.status_message}`);
  check('vague question pauses in input-required', pending.state === 'TASK_STATE_INPUT_REQUIRED');
  if (pending.state === 'TASK_STATE_INPUT_REQUIRED') {
    const resumed = describeTask(
      await client.sendMessage(
        sendRequest(textMessage(CLARIFICATION_REPLY, { taskId: pending.task_id, contextId: pending.context_id })),
      ),
    );
    summary.resumed = resumed;
    out(`   reply "${CLARIFICATION_REPLY}" on the same task -> state=${resumed.state}`);
    out(`   > ${resumed.answer.split('\n')[0]}`);
    check('reply resumes the same task to completed', resumed.task_id === pending.task_id && resumed.state === 'TASK_STATE_COMPLETED');
  }

  // 5. SendStreamingMessage
  out(`\n== 5. sendMessageStream: ${STREAM_QUESTION}`);
  const kinds = [];
  let streamTask = '';
  let finalState = '';
  for await (const event of client.sendMessageStream(sendRequest(textMessage(STREAM_QUESTION)))) {
    const [kind, line] = eventLine(event);
    kinds.push(kind);
    out(`   [${kind}] ${line}`);
    if (kind === 'task') streamTask = event.payload.value.id;
    if (kind === 'statusUpdate') finalState = stateName(event.payload.value.status?.state);
  }
  summary.stream = { kinds, final_state: finalState, task_id: streamTask };
  check('stream starts with the task and ends completed', kinds[0] === 'task' && finalState === 'TASK_STATE_COMPLETED');
  check('stream carries working updates and both artifacts', kinds.filter((k) => k === 'statusUpdate').length >= 3 && kinds.filter((k) => k === 'artifactUpdate').length === 2);

  // 6. SubscribeToTask (resubscribe) on a running task
  out(`\n== 6. resubscribeTask on a running task (sendMessage with returnImmediately=true first)`);
  const started = await client.sendMessage(
    sendRequest(textMessage(STREAM_QUESTION), {
      acceptedOutputModes: [],
      taskPushNotificationConfig: undefined,
      returnImmediately: true,
    }),
  );
  const startedState = stateName(started.status?.state);
  out(`   sendMessage(returnImmediately) -> task ${started.id.slice(0, 8)} ${startedState}`);
  const resubKinds = [];
  let resubFinal = '';
  try {
    for await (const event of client.resubscribeTask({ tenant: '', id: started.id })) {
      const [kind, line] = eventLine(event);
      resubKinds.push(kind);
      out(`   [${kind}] ${line}`);
      if (kind === 'statusUpdate') resubFinal = stateName(event.payload.value.status?.state);
      if (kind === 'task' && TERMINAL.has(event.payload.value.status?.state)) resubFinal = stateName(event.payload.value.status.state);
    }
    summary.resubscribe = { started_state: startedState, kinds: resubKinds, final_state: resubFinal };
    check('returnImmediately returns before the run finishes', !TERMINAL.has(started.status?.state), startedState);
    check('resubscribe streams the running task to completion', resubKinds[0] === 'task' && resubFinal === 'TASK_STATE_COMPLETED', resubFinal);
  } catch (error) {
    summary.resubscribe = { started_state: startedState, error: errorInfo(error) };
    check('resubscribe streams the running task to completion', false, `${errorInfo(error).name}: ${errorInfo(error).message}`);
  }
  try {
    const events = [];
    for await (const event of client.resubscribeTask({ tenant: '', id: answered.task_id })) events.push(event);
    summary.resubscribe_completed = { events: events.length };
    check('resubscribe on a completed task raises UnsupportedOperationError (A2A 1.0 §9.4.6)', false, `${events.length} events`);
  } catch (error) {
    summary.resubscribe_completed = errorInfo(error);
    check(
      'resubscribe on a completed task raises UnsupportedOperationError (A2A 1.0 §9.4.6)',
      summary.resubscribe_completed.name.includes('UnsupportedOperation'),
      `${summary.resubscribe_completed.name}: ${summary.resubscribe_completed.message}`,
    );
  }

  // 7. CancelTask
  out(`\n== 7. cancelTask on a running task`);
  const toCancel = await client.sendMessage(
    sendRequest(textMessage(STREAM_QUESTION), {
      acceptedOutputModes: [],
      taskPushNotificationConfig: undefined,
      returnImmediately: true,
    }),
  );
  out(`   sendMessage(returnImmediately) -> task ${toCancel.id.slice(0, 8)} ${stateName(toCancel.status?.state)}`);
  try {
    const canceled = await client.cancelTask({ tenant: '', id: toCancel.id, metadata: undefined });
    out(`   cancelTask -> ${stateName(canceled.status?.state)}`);
    await sleep(Number(args['settle-ms'] ?? 3000)); // let the (now cancelled) run finish in the background
    const after = await client.getTask({ tenant: '', id: toCancel.id });
    out(`   getTask after ${args['settle-ms'] ?? 3000} ms -> ${stateName(after.status?.state)}; artifacts=${(after.artifacts ?? []).length}`);
    summary.cancel = { state: stateName(canceled.status?.state), after: stateName(after.status?.state), artifacts: (after.artifacts ?? []).length };
    check('cancelTask returns a canceled task', canceled.status?.state === TaskState.TASK_STATE_CANCELED);
    check('a canceled task stays canceled (no late answer)', after.status?.state === TaskState.TASK_STATE_CANCELED && (after.artifacts ?? []).length === 0);
  } catch (error) {
    summary.cancel = { error: errorInfo(error) };
    check('cancelTask returns a canceled task', false, `${errorInfo(error).name}: ${errorInfo(error).message}`);
  }
  try {
    await client.cancelTask({ tenant: '', id: answered.task_id, metadata: undefined });
    check('cancelTask on a completed task raises TaskNotCancelableError', false);
  } catch (error) {
    summary.cancel_completed = errorInfo(error);
    check('cancelTask on a completed task raises TaskNotCancelableError', summary.cancel_completed.name.includes('TaskNotCancelable'), summary.cancel_completed.name);
  }

  // Wire log
  out('\n== Wire log (JS SDK fetch calls)');
  for (const entry of wire) {
    out(`   ${entry.request} A2A-Version=${entry.a2a_version ?? '-'} -> ${entry.status} ${entry.content_type ?? ''}`);
  }
  summary.wire = wire;
  summary.checks = checks;
  const failed = checks.filter((item) => !item.ok);
  out(`\n== ${checks.length - failed.length}/${checks.length} checks passed`);
  if (args.json) writeFileSync(args.json, JSON.stringify(summary, null, 2));
  return failed.length === 0 ? 0 : 1;
}

function parseArgs(list) {
  const parsed = {};
  for (let index = 0; index < list.length; index += 1) {
    const item = list[index];
    if (item.startsWith('--')) {
      const key = item.slice(2);
      const next = list[index + 1];
      if (next === undefined || next.startsWith('--')) parsed[key] = true;
      else {
        parsed[key] = next;
        index += 1;
      }
    }
  }
  return parsed;
}

main()
  .then((code) => exit(code))
  .catch((error) => {
    console.error(error);
    exit(2);
  });
