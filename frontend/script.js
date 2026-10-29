 
const questionBox = document.getElementById("question");
const askBtn = document.getElementById("askBtn");
const loading = document.getElementById("loading");
const errorBox = document.getElementById("error");
const answerCard = document.getElementById("answerCard");
const answerDiv = document.getElementById("answer");
const scoreP = document.getElementById("score");
const sourcesCard = document.getElementById("sourcesCard");
const sourcesDiv = document.getElementById("sources");

 
const examples = [
  "What are the eligibility criteria for ST students to apply for the scholarship?",
  "What documents are required to apply for the scholarship?",
  "What is the maximum annual family income allowed for students applying for this scholarship?",
  "What is the application process for the scholarship, and where should students apply?"
];

 
const examplesDiv = document.getElementById("examples");
examples.forEach(function (text) {
  const btn = document.createElement("button");
  btn.className = "chip";
  btn.textContent = text;
  btn.onclick = function () {
    questionBox.value = text;
    askQuestion();
  };
  examplesDiv.appendChild(btn);
});

 
function clearResult() {
  errorBox.classList.add("hidden");
  answerCard.classList.add("hidden");
  sourcesCard.classList.add("hidden");
  sourcesDiv.innerHTML = "";
}

// main function
async function askQuestion() {
  const question = questionBox.value.trim();
  if (question === "") {
    return;
  }

  clearResult();
  loading.classList.remove("hidden");
  askBtn.disabled = true;

  try {
    // send the question to the backend
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question })
    });
    const data = await response.json();

    if (!response.ok) {
      showError(data.error || "Something went wrong.");
    } else {
      showResult(data);
    }
  } catch (err) {
    showError("Could not reach the server. Is server.py running?");
  }

  loading.classList.add("hidden");
  askBtn.disabled = false;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function showResult(data) {
  // answer (textContent is used so the text is shown safely)
  answerDiv.textContent = data.answer;
  answerDiv.className = data.found ? "answer found" : "answer notfound";
  scoreP.textContent = "Best match score: " + data.top_score +
                       "  |  Chunks used: " + data.sources.length;
  answerCard.classList.remove("hidden");

  // sources
  if (data.sources.length > 0) {
    data.sources.forEach(function (s, i) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = (i + 1) + ". " + s.source + " | page " + s.page + " | score " + s.score;
      const text = document.createElement("p");
      text.textContent = s.text;
      details.appendChild(summary);
      details.appendChild(text);
      sourcesDiv.appendChild(details);
    });
    sourcesCard.classList.remove("hidden");
  }
}

// button click and Ctrl+Enter
askBtn.onclick = askQuestion;
questionBox.addEventListener("keydown", function (e) {
  if (e.key === "Enter" && e.ctrlKey) {
    askQuestion();
  }
});
