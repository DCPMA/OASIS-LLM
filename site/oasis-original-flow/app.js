const labelSets = {
  valence: [
    "Very negative",
    "Moderately negative",
    "Somewhat negative",
    "Neutral",
    "Somewhat positive",
    "Moderately positive",
    "Very positive",
  ],
  arousal: [
    "Very low",
    "Moderately low",
    "Somewhat low",
    "Neither low nor high",
    "Somewhat high",
    "Moderately high",
    "Very high",
  ],
};

const listBank = [
  {
    name: "List 1",
    stimuli: [
      { file: "Beach 1.jpg", cue: "open scene" },
      { file: "Dog 1.jpg", cue: "animal" },
      { file: "Birthday 1.jpg", cue: "people" },
      { file: "Gun 1.jpg", cue: "object" },
      { file: "Thunderstorm 1.jpg", cue: "weather" },
      { file: "Flowers 1.jpg", cue: "nature" },
    ],
  },
  {
    name: "List 2",
    stimuli: [
      { file: "Baby 1.jpg", cue: "people" },
      { file: "Shark 1.jpg", cue: "animal" },
      { file: "Fire 1.jpg", cue: "disaster" },
      { file: "Wedding 1.jpg", cue: "people" },
      { file: "Spider 1.jpg", cue: "animal" },
      { file: "Doctor 1.jpg", cue: "people" },
    ],
  },
  {
    name: "List 3",
    stimuli: [
      { file: "Cat 1.jpg", cue: "animal" },
      { file: "Car crash 1.jpg", cue: "accident" },
      { file: "Sunset 1.jpg", cue: "scene" },
      { file: "Surgery 1.jpg", cue: "medical" },
      { file: "Camping 1.jpg", cue: "scene" },
      { file: "Funeral 1.jpg", cue: "people" },
    ],
  },
  {
    name: "List 4",
    stimuli: [
      { file: "Lake 1.jpg", cue: "scene" },
      { file: "Angry face 1.jpg", cue: "people" },
      { file: "Rock climbing 1.jpg", cue: "activity" },
      { file: "Cemetery 1.jpg", cue: "scene" },
      { file: "Dessert 1.jpg", cue: "object" },
      { file: "Police 1.jpg", cue: "people" },
    ],
  },
];

const state = {
  step: "start",
  dimension: null,
  listName: null,
  trialSet: [],
  trialIndex: 0,
  instructions: [],
  instructionIndex: 0,
  ratings: [],
  demographics: {},
  startedAt: null,
};

const root = document.querySelector("#screen-root");

document.addEventListener("click", (event) => {
  const action = event.target.closest("[data-action]");

  if (!action) {
    return;
  }

  const { action: type } = action.dataset;

  if (type === "begin") {
    beginSession(action.dataset.mode);
  }

  if (type === "next-instruction") {
    advanceInstruction();
  }

  if (type === "restart") {
    resetState();
    render();
  }

  if (type === "download") {
    downloadResponses();
  }
});

document.addEventListener("submit", (event) => {
  if (event.target.matches("#rating-form")) {
    event.preventDefault();
    submitRating(event.target);
  }

  if (event.target.matches("#demographics-form")) {
    event.preventDefault();
    submitDemographics(event.target);
  }
});

document.addEventListener("keydown", (event) => {
  if (state.step !== "trial") {
    return;
  }

  const value = Number.parseInt(event.key, 10);

  if (!Number.isInteger(value) || value < 1 || value > 7) {
    return;
  }

  const input = document.querySelector(`#rating-${value}`);

  if (input) {
    input.checked = true;
  }
});

function beginSession(mode) {
  const selectedDimension =
    mode === "auto"
      ? Math.random() < 0.5
        ? "valence"
        : "arousal"
      : mode;
  const selectedList = listBank[Math.floor(Math.random() * listBank.length)];

  state.step = "instructions";
  state.dimension = selectedDimension;
  state.listName = selectedList.name;
  state.trialSet = shuffle([...selectedList.stimuli]);
  state.trialIndex = 0;
  state.ratings = [];
  state.demographics = {};
  state.startedAt = Date.now();
  state.instructions = buildInstructions(selectedDimension);
  state.instructionIndex = 0;

  render();
}

function buildInstructions(dimension) {
  const screens = [
    {
      eyebrow: "Screen 1",
      title: "General study overview",
      body:
        "You will see a short sequence of images presented one at a time. Use your first impression, answer on the scale shown, and keep moving forward. This reconstruction keeps the original one-way structure but shortens the length from 225 images to 6.",
      aside:
        "The published study warned that some pictures could be disturbing. This local mock keeps that warning and includes a mixed-intensity sample, but avoids the more explicit original stimuli.",
    },
    {
      eyebrow: "Screen 2",
      title:
        dimension === "valence"
          ? "Valence instructions"
          : "Arousal instructions",
      body:
        dimension === "valence"
          ? "Rate how positive or negative the image itself seems. Focus on the picture, not on whether your answer is correct. A low score means very negative, a high score means very positive."
          : "Rate how calming or activating the image itself seems. Focus on the picture, not on whether it is good or bad. A low score means very low activation, a high score means very high activation.",
      aside:
        "The original main study used image-centered instructions only. Each participant rated just one dimension for an entire list.",
    },
  ];

  if (dimension === "arousal") {
    screens.push({
      eyebrow: "Screen 3",
      title: "Arousal is separate from valence",
      body:
        "Arousal is about intensity, not positivity. A picture can feel negative and highly activating, positive and highly activating, or neutral and highly activating. Rate the level of activation, not whether the content is pleasant.",
      aside:
        "This extra clarification mirrors the published procedure, where the arousal condition received an additional explanation to reduce confusion with valence.",
    });
  }

  return screens;
}

function advanceInstruction() {
  if (state.instructionIndex < state.instructions.length - 1) {
    state.instructionIndex += 1;
    render();
    return;
  }

  state.step = "trial";
  render();
}

function submitRating(form) {
  const formData = new FormData(form);
  const rating = Number.parseInt(formData.get("rating"), 10);

  if (!rating) {
    return;
  }

  const stimulus = state.trialSet[state.trialIndex];

  state.ratings.push({
    dimension: state.dimension,
    list: state.listName,
    file: stimulus.file,
    cue: stimulus.cue,
    trialNumber: state.trialIndex + 1,
    rating,
    ratedAt: new Date().toISOString(),
  });

  if (state.trialIndex < state.trialSet.length - 1) {
    state.trialIndex += 1;
    render();
    return;
  }

  state.step = "demographics";
  render();
}

function submitDemographics(form) {
  const formData = new FormData(form);
  state.demographics = Object.fromEntries(formData.entries());
  state.step = "debrief";
  render();
}

function downloadResponses() {
  const payload = buildPayload();
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `oasis-mock-${payload.session.dimension}-${Date.now()}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

function buildPayload() {
  return {
    session: {
      mock: true,
      dimension: state.dimension,
      list: state.listName,
      trialCount: state.trialSet.length,
      durationSeconds: Math.round((Date.now() - state.startedAt) / 1000),
      exportedAt: new Date().toISOString(),
    },
    ratings: state.ratings,
    demographics: state.demographics,
  };
}

function resetState() {
  state.step = "start";
  state.dimension = null;
  state.listName = null;
  state.trialSet = [];
  state.trialIndex = 0;
  state.instructions = [];
  state.instructionIndex = 0;
  state.ratings = [];
  state.demographics = {};
  state.startedAt = null;
}

function render() {
  if (state.step === "start") {
    renderStart();
    return;
  }

  if (state.step === "instructions") {
    renderInstruction();
    return;
  }

  if (state.step === "trial") {
    renderTrial();
    return;
  }

  if (state.step === "demographics") {
    renderDemographics();
    return;
  }

  renderDebrief();
}

function renderStart() {
  root.innerHTML = `
    <section class="panel panel-grid">
      <div class="hero-grid">
        <div class="panel-card">
          <div class="pill-row">
            <span class="pill">822 participants in the published sample</span>
            <span class="pill">1 dimension per participant</span>
            <span class="pill">4 lists in the published design</span>
          </div>
          <h2>Experience the original structure without the original length.</h2>
          <p>
            The real study showed 225 images from one list and asked each participant to rate either
            valence or arousal for the whole session. This mock preserves that structure, uses a random
            list assignment, and keeps navigation forward-only.
          </p>
          <p class="warning">
            Content note: the real study warned participants that some images could be sexually explicit,
            violent, or traumatic. This mock uses a safer subset of local OASIS images, but some pictures
            may still feel intense.
          </p>
        </div>

        <div class="overview-grid">
          <div class="metric">
            <div class="stat-label">Mock length</div>
            <div class="stat-value">6 images</div>
          </div>
          <div class="metric">
            <div class="stat-label">Original study length</div>
            <div class="stat-value">225 images</div>
          </div>
          <div class="metric">
            <div class="stat-label">Original average time</div>
            <div class="stat-value">25.13 min</div>
          </div>
        </div>
      </div>

      <div class="actions">
        <button class="button-primary" data-action="begin" data-mode="auto">Auto-assign me like the study</button>
        <button class="button-secondary" data-action="begin" data-mode="valence">Experience the valence condition</button>
        <button class="button-warm" data-action="begin" data-mode="arousal">Experience the arousal condition</button>
      </div>
    </section>
  `;
}

function renderInstruction() {
  const screen = state.instructions[state.instructionIndex];
  const nextLabel =
    state.instructionIndex === state.instructions.length - 1
      ? "Begin the rating task"
      : "Continue";

  root.innerHTML = `
    <section class="panel instruction-grid">
      <article class="instruction-card">
        <p class="eyebrow">${screen.eyebrow}</p>
        <h2>${screen.title}</h2>
        <p>${screen.body}</p>
        <div class="actions">
          <button class="button-primary" data-action="next-instruction">${nextLabel}</button>
        </div>
      </article>

      <aside class="instruction-card">
        <div class="pill-row">
          <span class="pill">Assigned condition: ${capitalize(state.dimension)}</span>
          <span class="pill">Assigned list: ${state.listName}</span>
        </div>
        <p>${screen.aside}</p>
        <p class="subtle">
          In the published study, participants could not go back to previous screens after advancing.
        </p>
      </aside>
    </section>
  `;
}

function renderTrial() {
  const stimulus = state.trialSet[state.trialIndex];
  const labels = labelSets[state.dimension];
  const progress = ((state.trialIndex + 1) / state.trialSet.length) * 100;
  const imagePath = encodeURI(`../../OASIS/images/${stimulus.file}`);

  root.innerHTML = `
    <section class="panel trial-panel">
      <div class="trial-topline">
        <div>
          <p class="eyebrow">Rating task</p>
          <div class="trial-counter">Trial ${state.trialIndex + 1} of ${state.trialSet.length}</div>
        </div>
        <div class="pill-row">
          <span class="pill">${capitalize(state.dimension)}</span>
          <span class="pill">${state.listName}</span>
        </div>
      </div>

      <div class="progress" aria-hidden="true">
        <div class="progress-fill" style="width: ${progress.toFixed(1)}%"></div>
      </div>

      <div class="trial-grid">
        <article>
          <img class="study-image" src="${imagePath}" alt="OASIS stimulus ${stimulus.file}" />
        </article>

        <article class="trial-copy">
          <h2>${capitalize(state.dimension)}</h2>
          <p>
            Rate the image itself on the scale below. Use your first impression and keep moving.
          </p>
          <p class="subtle">
            Local cue for orientation: ${stimulus.cue}. This cue is not shown in the original study.
          </p>

          <form id="rating-form">
            <div class="scale-grid">
              ${labels
      .map(
        (label, index) => `
                    <div class="rating-option">
                      <input id="rating-${index + 1}" type="radio" name="rating" value="${index + 1}" required />
                      <label for="rating-${index + 1}">
                        <span class="scale-number">${index + 1}</span>
                        <span class="scale-label">${label}</span>
                      </label>
                    </div>
                  `,
      )
      .join("")}
            </div>

            <div class="dual-actions">
              <p class="hint">Keyboard shortcut: press 1-7 to select a rating.</p>
              <button class="button-primary" type="submit">Record rating and continue</button>
            </div>
          </form>
        </article>
      </div>
    </section>
  `;
}

function renderDemographics() {
  root.innerHTML = `
    <section class="panel panel-grid">
      <article class="demographics">
        <p class="eyebrow">Post-task questionnaire</p>
        <h2>Demographics</h2>
        <p>
          The published study collected demographics after the rating task. These fields are optional here,
          but they mirror the original structure closely enough for you to feel the sequencing.
        </p>

        <form id="demographics-form" class="demographics-grid">
          ${selectField("Gender", "gender", ["", "Female", "Male", "Nonbinary", "Prefer not to say"])}
          ${numberField("Age", "age")}
          ${selectField("Ethnicity", "ethnicity", ["", "Hispanic or Latino", "Not Hispanic or Latino", "Unknown"])}
          ${selectField("Race", "race", ["", "American Indian/Alaska Native", "East Asian", "South Asian", "Native Hawaiian or other Pacific Islander", "Black or African American", "White", "More than one race", "Other or unknown"])}
          ${selectField("Ideology", "ideology", ["", "Strongly conservative", "Moderately conservative", "Slightly conservative", "Neutral", "Slightly liberal", "Moderately liberal", "Strongly liberal", "Prefer not to answer"])}
          ${selectField("Income", "income", ["", "below $25,000", "$25,000 to $44,999", "$50,000 to $69,999", "$70,000 to $99,999", "$100,000 or above", "Prefer not to answer"])}
          ${selectField("Education", "education", ["", "Grade school/some high school", "High school diploma", "Some college, no degree", "College degree", "Graduate degree"])}
          ${textField("Current ZIP code", "zipCurrent")}
          ${textField("ZIP code lived in longest", "zipLongest")}
          <div class="field field-wide">
            <label for="notes">Notes</label>
            <textarea id="notes" name="notes" rows="4" placeholder="Optional reflections on how the flow felt."></textarea>
          </div>

          <div class="field field-wide">
            <button class="button-primary" type="submit">Finish mock session</button>
          </div>
        </form>
      </article>
    </section>
  `;
}

function renderDebrief() {
  const payload = buildPayload();
  const preview = JSON.stringify(payload, null, 2);

  root.innerHTML = `
    <section class="panel panel-grid">
      <article class="summary-card">
        <p class="eyebrow">Debrief</p>
        <h2>Session complete</h2>
        <p>
          You just walked through the structure of the published OASIS task: single-dimension assignment,
          forward-only image ratings, then demographics.
        </p>

        <div class="summary-grid">
          <div class="metric">
            <div class="stat-label">Condition</div>
            <div class="stat-value">${capitalize(state.dimension)}</div>
          </div>
          <div class="metric">
            <div class="stat-label">List</div>
            <div class="stat-value">${state.listName}</div>
          </div>
          <div class="metric">
            <div class="stat-label">Recorded ratings</div>
            <div class="stat-value">${state.ratings.length}</div>
          </div>
        </div>
      </article>

      <article class="download-card">
        <h3>Export or restart</h3>
        <p>
          Nothing is uploaded anywhere. If you want to keep the session, download the local JSON payload.
        </p>
        <div class="actions">
          <button class="button-primary" data-action="download">Download local responses</button>
          <button class="button-secondary" data-action="restart">Run the flow again</button>
        </div>
      </article>

      <article class="download-card">
        <h3>Payload preview</h3>
        <pre>${escapeHtml(preview)}</pre>
      </article>
    </section>
  `;
}

function selectField(label, name, options) {
  return `
    <div class="field">
      <label for="${name}">${label}</label>
      <select id="${name}" name="${name}">
        ${options
      .map((option) => `<option value="${escapeAttribute(option)}">${escapeHtml(option || "Select one")}</option>`)
      .join("")}
      </select>
    </div>
  `;
}

function textField(label, name) {
  return `
    <div class="field">
      <label for="${name}">${label}</label>
      <input id="${name}" name="${name}" type="text" />
    </div>
  `;
}

function numberField(label, name) {
  return `
    <div class="field">
      <label for="${name}">${label}</label>
      <input id="${name}" name="${name}" type="number" min="18" max="120" />
    </div>
  `;
}

function shuffle(array) {
  for (let index = array.length - 1; index > 0; index -= 1) {
    const randomIndex = Math.floor(Math.random() * (index + 1));
    [array[index], array[randomIndex]] = [array[randomIndex], array[index]];
  }

  return array;
}

function capitalize(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function escapeAttribute(value) {
  return String(value).replaceAll('"', "&quot;");
}

render();