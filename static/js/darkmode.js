document.addEventListener("DOMContentLoaded", function () {
    const toggleBtn = document.getElementById('darkModeToggle');
    const icon = document.getElementById('darkModeIcon');
    const body = document.body;

    // Funktion zum Icon-Wechsel
    function updateIcon() {
        if (body.classList.contains('dark-mode')) {
            icon.textContent = '☀️'; // Lightmode Icon
        } else {
            icon.textContent = '🌙'; // Darkmode Icon
        }
    }

    // Zustand laden
    if (localStorage.getItem('darkmode') === 'enabled') {
        body.classList.add('dark-mode');
    }
    updateIcon();

    // Klick-Event
    toggleBtn.addEventListener('click', () => {
        body.classList.toggle('dark-mode');
        if (body.classList.contains('dark-mode')) {
            localStorage.setItem('darkmode', 'enabled');
        } else {
            localStorage.setItem('darkmode', 'disabled');
        }
        updateIcon();
    });
});
