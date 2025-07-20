document.addEventListener('DOMContentLoaded', () => {
  const url = window.pdfUrl;

  let canvas = document.getElementById('pdf-canvas');
  let ctx = canvas.getContext('2d');

  const scale = 1.0;

  pdfjsLib.getDocument(url).promise.then(pdfDoc => {
    pdfDoc.getPage(1).then(page => {
      const viewport = page.getViewport({scale});
      canvas.height = viewport.height;
      canvas.width = viewport.width;

      const renderContext = {
        canvasContext: ctx,
        viewport: viewport
      };
      page.render(renderContext);
    });
  });
});

