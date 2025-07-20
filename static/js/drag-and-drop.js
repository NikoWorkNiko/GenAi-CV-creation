document.addEventListener("DOMContentLoaded", function () {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const fileName = document.getElementById("file-name");

// Open file dialog on click
    dropzone.addEventListener("click", () => fileInput.click());

// Show file name on selection
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            fileName.textContent = fileInput.files[0].name;
        }
    });

// Drag & Drop
    dropzone.addEventListener("dragover", e => {
        e.preventDefault();
        dropzone.classList.add("bg-light");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("bg-light");
    });

    dropzone.addEventListener("drop", e => {
        e.preventDefault();
        dropzone.classList.remove("bg-light");
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            fileName.textContent = e.dataTransfer.files[0].name;
        }
    });
});