"use strict";

const state = {
  records: [],
  index: 0,
  qtypes: [],
  qsubtypesByQtype: {},
  listRequest: 0,
  detailRequest: 0,
};

const elements = {
  search: document.querySelector("#search-input"),
  qtype: document.querySelector("#qtype-filter"),
  qsubtype: document.querySelector("#qsubtype-filter"),
  previous: document.querySelector("#previous-button"),
  next: document.querySelector("#next-button"),
  position: document.querySelector("#position-input"),
  railPosition: document.querySelector("#rail-position"),
  railTotal: document.querySelector("#rail-total"),
  resultSummary: document.querySelector("#result-summary"),
  status: document.querySelector("#status-message"),
  empty: document.querySelector("#empty-state"),
  detail: document.querySelector("#record-detail"),
  id: document.querySelector("#record-id"),
  qtypeValue: document.querySelector("#record-qtype"),
  qsubtypeValue: document.querySelector("#record-qsubtype"),
  csv: document.querySelector("#record-csv"),
  question: document.querySelector("#record-question"),
  answer: document.querySelector("#record-answer"),
  table: document.querySelector("#table-container"),
  tableSize: document.querySelector("#table-size"),
};

function setStatus(message, isError = false) {
  elements.status.textContent = message;
  elements.status.classList.toggle("is-error", isError);
}

function replaceOptions(select, values, allLabel, selectedValue = "") {
  const fragment = document.createDocumentFragment();
  const allOption = document.createElement("option");
  allOption.value = "";
  allOption.textContent = allLabel;
  fragment.appendChild(allOption);

  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    fragment.appendChild(option);
  }

  select.replaceChildren(fragment);
  select.value = values.includes(selectedValue) ? selectedValue : "";
}

function availableSubtypes() {
  if (elements.qtype.value) {
    return state.qsubtypesByQtype[elements.qtype.value] || [];
  }
  return [...new Set(Object.values(state.qsubtypesByQtype).flat())].sort();
}

function syncFilterOptions(qtypes, qsubtypesByQtype) {
  const selectedQtype = elements.qtype.value;
  const selectedQsubtype = elements.qsubtype.value;
  state.qtypes = qtypes;
  state.qsubtypesByQtype = qsubtypesByQtype;
  replaceOptions(elements.qtype, qtypes, "All QTypes", selectedQtype);
  replaceOptions(
    elements.qsubtype,
    availableSubtypes(),
    "All QSubtypes",
    selectedQsubtype,
  );
}

async function fetchJson(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with HTTP ${response.status}`);
  }
  return payload;
}

async function loadRecords(resetIndex = true) {
  const requestNumber = ++state.listRequest;
  setStatus("Loading…");

  const params = new URLSearchParams();
  const query = elements.search.value.trim();
  if (query) params.set("query", query);
  if (elements.qtype.value) params.set("qtype", elements.qtype.value);
  if (elements.qsubtype.value) params.set("qsubtype", elements.qsubtype.value);

  try {
    const payload = await fetchJson(`/api/records?${params.toString()}`);
    if (requestNumber !== state.listRequest) return;
    state.records = payload.records;
    if (resetIndex) state.index = 0;
    state.index = Math.min(state.index, Math.max(0, state.records.length - 1));
    syncFilterOptions(payload.qtypes, payload.qsubtypes_by_qtype);
    setStatus("");
    renderSelection();
  } catch (error) {
    if (requestNumber !== state.listRequest) return;
    state.records = [];
    renderSelection();
    setStatus(error.message, true);
  }
}

function renderNavigation() {
  const count = state.records.length;
  const current = count ? state.index + 1 : 0;
  elements.railPosition.textContent = String(current).padStart(3, "0");
  elements.railTotal.textContent = String(count).padStart(3, "0");
  elements.position.value = count ? String(current) : "";
  elements.position.max = String(Math.max(1, count));
  elements.position.disabled = count === 0;
  elements.previous.disabled = current <= 1;
  elements.next.disabled = current === 0 || current >= count;
  elements.resultSummary.textContent = count === 1 ? "1 matching record" : `${count} matching records`;
}

function clearTable(message = "Select a record to load its table.", isError = false) {
  const paragraph = document.createElement("p");
  paragraph.className = isError ? "table-error" : "table-placeholder";
  paragraph.textContent = message;
  elements.table.replaceChildren(paragraph);
  elements.tableSize.textContent = "";
}

function renderSelection() {
  renderNavigation();
  if (!state.records.length) {
    elements.empty.hidden = false;
    elements.detail.hidden = true;
    clearTable("No table to display.");
    return;
  }
  elements.empty.hidden = true;
  elements.detail.hidden = false;
  loadRecordDetail(state.records[state.index].record_index);
}

function setRecordMetadata(record) {
  elements.id.textContent = record.id;
  elements.qtypeValue.textContent = record.qtype;
  elements.qsubtypeValue.textContent = record.qsubtype;
  elements.csv.textContent = record.csv_file;
  elements.question.textContent = record.question;
  elements.answer.textContent = record.gold_answer;
}

function renderTable(table) {
  const rows = table.rows || [];
  const sourceHeaders = table.headers || [];
  const width = Math.max(sourceHeaders.length, ...rows.map((row) => row.length), 0);
  if (width === 0) {
    clearTable("This CSV file is empty.");
    return;
  }

  const headers = Array.from(
    { length: width },
    (_, index) => sourceHeaders[index] || `Column ${index + 1}`,
  );
  const tableElement = document.createElement("table");
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  for (const header of headers) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = header;
    headerRow.appendChild(cell);
  }
  thead.appendChild(headerRow);
  tableElement.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tableRow = document.createElement("tr");
    for (let column = 0; column < width; column += 1) {
      const cell = document.createElement("td");
      cell.textContent = row[column] ?? "";
      tableRow.appendChild(cell);
    }
    tbody.appendChild(tableRow);
  }
  tableElement.appendChild(tbody);
  elements.table.replaceChildren(tableElement);
  elements.tableSize.textContent = `${rows.length} rows × ${width} columns`;
}

async function loadRecordDetail(recordIndex) {
  const requestNumber = ++state.detailRequest;
  setRecordMetadata(state.records[state.index]);
  clearTable("Loading CSV table…");
  try {
    const record = await fetchJson(`/api/records/by-index/${recordIndex}`);
    if (requestNumber !== state.detailRequest) return;
    setRecordMetadata(record);
    if (record.table_error) {
      clearTable(record.table_error, true);
      return;
    }
    renderTable(record.table);
  } catch (error) {
    if (requestNumber !== state.detailRequest) return;
    clearTable(error.message, true);
  }
}

function moveTo(index) {
  if (!state.records.length) return;
  state.index = Math.max(0, Math.min(index, state.records.length - 1));
  renderSelection();
}

function debounce(callback, delay) {
  let timer;
  return (...args) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => callback(...args), delay);
  };
}

elements.search.addEventListener("input", debounce(() => loadRecords(true), 250));
elements.qtype.addEventListener("change", () => {
  replaceOptions(elements.qsubtype, availableSubtypes(), "All QSubtypes");
  loadRecords(true);
});
elements.qsubtype.addEventListener("change", () => loadRecords(true));
elements.previous.addEventListener("click", () => moveTo(state.index - 1));
elements.next.addEventListener("click", () => moveTo(state.index + 1));
elements.position.addEventListener("change", () => {
  const requested = Number.parseInt(elements.position.value, 10);
  moveTo(Number.isFinite(requested) ? requested - 1 : state.index);
});

loadRecords(true);
