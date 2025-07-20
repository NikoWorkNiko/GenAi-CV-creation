// Referenz auf das Element
const loading = document.getElementById("loading-screen");

fetch('/static/json/ai_quotes.json')
  .then(response => response.json())
  .then(quotes => {
    const randomQuote = quotes[Math.floor(Math.random() * quotes.length)];
    const quoteBox = document.getElementById('loading-quote');
    if (quoteBox) quoteBox.innerText = randomQuote;
  });

// Timeout-Handle
let showLoaderTimeout = setTimeout(() => {
    loading.style.display = "flex";
}, 2000); // show after time X

// Wenn DOM vollständig geladen ist: Ladebildschirm entfernen (falls sichtbar)
document.addEventListener("DOMContentLoaded", function () {
    clearTimeout(showLoaderTimeout); // zeige ihn gar nicht erst
    loading.classList.add("hidden");
    setTimeout(() => loading.style.display = "none", 300);
});

// Ladebildschirm beim Verlassen sofort zeigen (z. B. Linkklick → Redirect)
window.addEventListener("beforeunload", function () {
    loading.style.display = "flex";
    loading.classList.remove("hidden");
});
